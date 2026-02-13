# CUDA allocator contention benchmark.
#
# Tests whether PyTorch's CUDA caching allocator causes contention when
# two threads allocate/free GPU memory simultaneously.
#
# Three allocation modes:
#   static:   Pre-allocated tensors, reused every step (like bench2)
#   dynamic:  Intermediate tensors allocated fresh each step, freed at end
#   heavy:    Aggressive alloc/free — allocate+free extra buffers per layer,
#             varying sizes, simulating vLLM's KV cache and output processing
#
# The CUDA caching allocator has a per-device mutex. Even though each thread
# uses a different device, PyTorch's allocator may have global bookkeeping.
#
# Usage:
#   python cuda_alloc_bench.py
#   python cuda_alloc_bench.py --alloc-mode heavy
#   python cuda_alloc_bench.py --hidden-size 1024 --num-layers 16

import argparse
import threading
import time
from multiprocessing import Process, Queue


def transformer_step(
    device_index,
    num_iters,
    hidden_size,
    num_layers,
    batch_size,
    alloc_mode,
    result_holder,
):
    """Simulate transformer decode steps with varying allocation patterns."""
    import torch

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dtype = torch.float16

    # Model weights — always pre-allocated (these are static in real models).
    embed_weight = torch.randn(32000, hidden_size, device=device, dtype=dtype)
    layers = []
    for _ in range(num_layers):
        layers.append(
            {
                "wq": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                "wk": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                "wv": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                "wo": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                "w_gate": torch.randn(
                    hidden_size, hidden_size * 4, device=device, dtype=dtype
                ),
                "w_up": torch.randn(
                    hidden_size, hidden_size * 4, device=device, dtype=dtype
                ),
                "w_down": torch.randn(
                    hidden_size * 4, hidden_size, device=device, dtype=dtype
                ),
                "norm1": torch.ones(hidden_size, device=device, dtype=dtype),
                "norm2": torch.ones(hidden_size, device=device, dtype=dtype),
            }
        )
    lm_head = torch.randn(hidden_size, 32000, device=device, dtype=dtype)
    token_ids = torch.randint(0, 32000, (batch_size,), device=device)

    def rmsnorm(x, weight):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        return (x * weight).to(dtype)

    # For "heavy" mode: pre-define sizes for extra allocations per layer.
    # Simulates KV cache block alloc, output buffers, scatter indices, etc.
    extra_sizes = [
        (batch_size, hidden_size),  # KV cache block
        (batch_size, hidden_size // 8),  # attention scores
        (batch_size * 4,),  # block indices
        (batch_size, 32000),  # logit buffer (large!)
        (batch_size, hidden_size * 2),  # concat buffer
    ]

    # Warmup.
    for _ in range(3):
        h = embed_weight[token_ids]
        for layer in layers:
            residual = h
            h = rmsnorm(h, layer["norm1"])
            q = h @ layer["wq"]
            k = h @ layer["wk"]
            v = h @ layer["wv"]
            h = residual + v @ layer["wo"]
            residual = h
            h = rmsnorm(h, layer["norm2"])
            gate = h @ layer["w_gate"]
            up = h @ layer["w_up"]
            h = residual + (torch.nn.functional.silu(gate) * up) @ layer["w_down"]
        logits = h @ lm_head
        token_ids = logits.argmax(dim=-1)
    torch.cuda.synchronize(device)

    # Timed loop.
    t0 = time.perf_counter()
    for step in range(num_iters):
        h = embed_weight[token_ids]

        for li, layer in enumerate(layers):
            if alloc_mode == "static":
                # Standard forward — PyTorch reuses cached allocations.
                residual = h
                h = rmsnorm(h, layer["norm1"])
                q = h @ layer["wq"]
                k = h @ layer["wk"]
                v = h @ layer["wv"]
                h = residual + v @ layer["wo"]
                residual = h
                h = rmsnorm(h, layer["norm2"])
                gate = h @ layer["w_gate"]
                up = h @ layer["w_up"]
                h = residual + (torch.nn.functional.silu(gate) * up) @ layer["w_down"]

            elif alloc_mode == "dynamic":
                # Force fresh allocations by creating new tensors explicitly.
                residual = h.clone()
                h = rmsnorm(h, layer["norm1"])
                q = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
                torch.mm(h, layer["wq"], out=q)
                k = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
                torch.mm(h, layer["wk"], out=k)
                v = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
                torch.mm(h, layer["wv"], out=v)
                attn_out = torch.empty(
                    batch_size, hidden_size, device=device, dtype=dtype
                )
                torch.mm(v, layer["wo"], out=attn_out)
                h = residual + attn_out
                del q, k, v, attn_out, residual

                residual = h.clone()
                h = rmsnorm(h, layer["norm2"])
                gate = torch.empty(
                    batch_size, hidden_size * 4, device=device, dtype=dtype
                )
                torch.mm(h, layer["w_gate"], out=gate)
                up = torch.empty(
                    batch_size, hidden_size * 4, device=device, dtype=dtype
                )
                torch.mm(h, layer["w_up"], out=up)
                mlp_out = (torch.nn.functional.silu(gate) * up) @ layer["w_down"]
                h = residual + mlp_out
                del gate, up, mlp_out, residual

            elif alloc_mode == "heavy":
                # Aggressive alloc/free — extra buffers per layer.
                residual = h.clone()
                h = rmsnorm(h, layer["norm1"])

                # Attention projections with explicit allocation.
                q = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
                torch.mm(h, layer["wq"], out=q)
                k = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
                torch.mm(h, layer["wk"], out=k)
                v = torch.empty(batch_size, hidden_size, device=device, dtype=dtype)
                torch.mm(h, layer["wv"], out=v)
                attn_out = torch.empty(
                    batch_size, hidden_size, device=device, dtype=dtype
                )
                torch.mm(v, layer["wo"], out=attn_out)
                h = residual + attn_out
                del q, k, v, attn_out, residual

                # Extra allocations simulating KV cache management.
                # vLLM allocates/updates KV cache blocks, scatter indices,
                # output buffers each step.
                extras = []
                for size in extra_sizes:
                    buf = torch.empty(*size, device=device, dtype=dtype)
                    extras.append(buf)
                # "Use" them minimally to prevent optimization.
                extras[0].fill_(0)
                extras[-1].zero_()
                del extras  # Free all at once.

                residual = h.clone()
                h = rmsnorm(h, layer["norm2"])
                gate = torch.empty(
                    batch_size, hidden_size * 4, device=device, dtype=dtype
                )
                torch.mm(h, layer["w_gate"], out=gate)
                up = torch.empty(
                    batch_size, hidden_size * 4, device=device, dtype=dtype
                )
                torch.mm(h, layer["w_up"], out=up)
                mlp_out = (torch.nn.functional.silu(gate) * up) @ layer["w_down"]
                h = residual + mlp_out
                del gate, up, mlp_out, residual

                # More extra allocs: simulate output processing buffers.
                out_buf = torch.empty(batch_size, 32000, device=device, dtype=dtype)
                indices = torch.randint(0, 32000, (batch_size,), device=device)
                gathered = out_buf[torch.arange(batch_size, device=device), indices]
                del out_buf, indices, gathered

        logits = h @ lm_head
        token_ids = logits.argmax(dim=-1)

        if step % 10 == 0:
            _ = token_ids.cpu()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    result_holder["device"] = device_index
    result_holder["elapsed"] = elapsed
    result_holder["ops_per_sec"] = num_iters / elapsed
    result_holder["ms_per_step"] = elapsed / num_iters * 1000


def subprocess_entry(
    device_index,
    num_iters,
    hidden_size,
    num_layers,
    batch_size,
    alloc_mode,
    result_queue,
):
    result = {}
    transformer_step(
        device_index, num_iters, hidden_size, num_layers, batch_size, alloc_mode, result
    )
    result_queue.put(result)


def print_results(label, results):
    total_ops = sum(r["ops_per_sec"] for r in results)
    max_elapsed = max(r["elapsed"] for r in results)
    print(f"\n  {label}:")
    for r in results:
        print(
            f"    cuda:{r['device']}: {r['ops_per_sec']:.1f} steps/s "
            f"({r['ms_per_step']:.2f} ms/step)"
        )
    print(f"    Total: {total_ops:.1f} steps/s, wall: {max_elapsed:.2f}s")
    return total_ops


def run_all(num_iters, hidden_size, num_layers, batch_size, alloc_mode):
    print(f"\n{'=' * 60}")
    print(f"Allocation mode: {alloc_mode}")
    print(f"{'=' * 60}")

    # Single GPU.
    r = {}
    transformer_step(0, num_iters, hidden_size, num_layers, batch_size, alloc_mode, r)
    single = print_results("Single GPU", [r])

    # Threaded.
    results = [{} for _ in range(2)]
    threads = []
    for i in range(2):
        t = threading.Thread(
            target=transformer_step,
            args=(
                i,
                num_iters,
                hidden_size,
                num_layers,
                batch_size,
                alloc_mode,
                results[i],
            ),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    threaded = print_results("Threaded (2 GPU)", results)

    # Subprocess.
    rq = Queue()
    procs = []
    for i in range(2):
        p = Process(
            target=subprocess_entry,
            args=(i, num_iters, hidden_size, num_layers, batch_size, alloc_mode, rq),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    sub_results = []
    while not rq.empty():
        sub_results.append(rq.get_nowait())
    sub_results.sort(key=lambda r: r["device"])
    subproc = print_results("Subprocess (2 GPU)", sub_results)

    ratio = threaded / subproc
    print(f"\n  Threaded vs subprocess: {ratio:.2f}x", end="")
    if ratio < 0.95:
        print(f"  --> {(1 - ratio) * 100:.1f}% loss")
    else:
        print("  --> OK")

    return single, threaded, subproc


def main():
    parser = argparse.ArgumentParser(description="CUDA allocator contention benchmark")
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--alloc-mode", choices=["static", "dynamic", "heavy"], help="Run only one mode"
    )
    args = parser.parse_args()

    print(
        f"Config: hidden={args.hidden_size}, layers={args.num_layers}, "
        f"batch={args.batch_size}, iters={args.num_iters}"
    )

    modes = [args.alloc_mode] if args.alloc_mode else ["static", "dynamic", "heavy"]
    results = {}
    for mode in modes:
        s, t, p = run_all(
            args.num_iters, args.hidden_size, args.num_layers, args.batch_size, mode
        )
        results[mode] = {"single": s, "threaded": t, "subprocess": p}

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("Summary: Threaded/Subprocess ratio by allocation mode")
        print(f"{'=' * 60}")
        for mode, r in results.items():
            ratio = r["threaded"] / r["subprocess"]
            bar = "#" * int(ratio * 40)
            print(f"  {mode:8s}: {ratio:.2f}x  {bar}")
        print("\n  If ratio drops with more allocation, the CUDA caching")
        print("  allocator's internal locks are causing contention.")


if __name__ == "__main__":
    main()
