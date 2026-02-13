# CUDA contention benchmark v2: realistic LLM-like workload.
#
# Simulates a transformer decode step more faithfully than v1:
# - Multiple linear layers (like attention Q/K/V projections + MLP)
# - Small element-wise ops (RMSNorm, activations, residual adds)
# - Index/gather operations (like token embedding lookup, KV cache scatter)
# - Tensor creation/destruction churn
# - Periodic CPU readback (like reading sampled token IDs)
#
# This should stress the same code paths as a real LLM step — many CUDA
# kernel launches, mixed large/small ops, Python object churn.
#
# Usage:
#   python cuda_contention_bench2.py
#   python cuda_contention_bench2.py --hidden-size 1024 --num-layers 16
#   python cuda_contention_bench2.py --batch-size 64

import argparse
import threading
import time
from multiprocessing import Process, Queue


def transformer_step(
    device_index, num_iters, hidden_size, num_layers, batch_size, result_holder
):
    """Simulate transformer decode steps on one GPU.

    Each iteration roughly models one decode step:
    - Token embedding lookup
    - N transformer layers (attention projections + MLP + norms)
    - Final logits projection
    - Argmax sampling
    - Periodic CPU readback
    """
    import torch

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dtype = torch.float16

    # "Model weights" — pre-allocated, like a real model.
    embed_weight = torch.randn(32000, hidden_size, device=device, dtype=dtype)
    layers = []
    for _ in range(num_layers):
        layers.append(
            {
                # Attention: Q, K, V, O projections.
                "wq": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                "wk": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                "wv": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                "wo": torch.randn(hidden_size, hidden_size, device=device, dtype=dtype),
                # MLP: gate, up, down projections (like LLaMA).
                "w_gate": torch.randn(
                    hidden_size, hidden_size * 4, device=device, dtype=dtype
                ),
                "w_up": torch.randn(
                    hidden_size, hidden_size * 4, device=device, dtype=dtype
                ),
                "w_down": torch.randn(
                    hidden_size * 4, hidden_size, device=device, dtype=dtype
                ),
                # RMSNorm weights.
                "norm1": torch.ones(hidden_size, device=device, dtype=dtype),
                "norm2": torch.ones(hidden_size, device=device, dtype=dtype),
            }
        )
    lm_head = torch.randn(hidden_size, 32000, device=device, dtype=dtype)

    # Input token IDs.
    token_ids = torch.randint(0, 32000, (batch_size,), device=device)

    def rmsnorm(x, weight):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        return (x * weight).to(dtype)

    # Warmup.
    for _ in range(3):
        h = embed_weight[token_ids]  # [batch, hidden]
        for layer in layers:
            # Attention (simplified — no actual attention, just projections).
            residual = h
            h = rmsnorm(h, layer["norm1"])
            _q = h @ layer["wq"]
            _k = h @ layer["wk"]
            v = h @ layer["wv"]
            # Simulate attention output (just use v as placeholder).
            attn_out = v @ layer["wo"]
            h = residual + attn_out
            # MLP.
            residual = h
            h = rmsnorm(h, layer["norm2"])
            gate = h @ layer["w_gate"]
            up = h @ layer["w_up"]
            h = (torch.nn.functional.silu(gate) * up) @ layer["w_down"]
            h = residual + h
        logits = h @ lm_head
        token_ids = logits.argmax(dim=-1)
    torch.cuda.synchronize(device)

    # Timed loop.
    t0 = time.perf_counter()
    for i in range(num_iters):
        # Embedding lookup.
        h = embed_weight[token_ids]

        for layer in layers:
            # Pre-attention norm.
            residual = h
            h = rmsnorm(h, layer["norm1"])

            # Attention projections.
            _q = h @ layer["wq"]
            _k = h @ layer["wk"]
            v = h @ layer["wv"]
            attn_out = v @ layer["wo"]
            h = residual + attn_out

            # Post-attention norm + MLP.
            residual = h
            h = rmsnorm(h, layer["norm2"])
            gate = h @ layer["w_gate"]
            up = h @ layer["w_up"]
            h = (torch.nn.functional.silu(gate) * up) @ layer["w_down"]
            h = residual + h

        # LM head + sampling.
        logits = h @ lm_head
        token_ids = logits.argmax(dim=-1)

        # Periodic CPU readback (like reading sampled tokens).
        if i % 10 == 0:
            _ = token_ids.cpu()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    result_holder["device"] = device_index
    result_holder["elapsed"] = elapsed
    result_holder["ops_per_sec"] = num_iters / elapsed
    result_holder["ms_per_step"] = elapsed / num_iters * 1000


def subprocess_worker(
    device_index, num_iters, hidden_size, num_layers, batch_size, result_queue
):
    result = {}
    transformer_step(
        device_index, num_iters, hidden_size, num_layers, batch_size, result
    )
    result_queue.put(result)


def run_single(num_iters, hidden_size, num_layers, batch_size):
    result = {}
    transformer_step(0, num_iters, hidden_size, num_layers, batch_size, result)
    return [result]


def run_threaded(num_iters, hidden_size, num_layers, batch_size, num_gpus=2):
    results = [{} for _ in range(num_gpus)]
    threads = []
    for i in range(num_gpus):
        t = threading.Thread(
            target=transformer_step,
            args=(i, num_iters, hidden_size, num_layers, batch_size, results[i]),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def run_subprocess(num_iters, hidden_size, num_layers, batch_size, num_gpus=2):
    result_queue = Queue()
    procs = []
    for i in range(num_gpus):
        p = Process(
            target=subprocess_worker,
            args=(i, num_iters, hidden_size, num_layers, batch_size, result_queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    results.sort(key=lambda r: r["device"])
    return results


def print_results(label, results):
    total_ops = sum(r["ops_per_sec"] for r in results)
    max_elapsed = max(r["elapsed"] for r in results)
    print(f"\n  {label}:")
    for r in results:
        print(
            f"    cuda:{r['device']}: {r['ops_per_sec']:.1f} steps/s "
            f"({r['ms_per_step']:.2f} ms/step, {r['elapsed']:.2f}s)"
        )
    print(f"    Total: {total_ops:.1f} steps/s, wall: {max_elapsed:.2f}s")
    return total_ops


def main():
    parser = argparse.ArgumentParser(
        description="CUDA contention benchmark v2: LLM-like workload"
    )
    parser.add_argument(
        "--num-iters", type=int, default=200, help="Decode steps per GPU (default: 200)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Model hidden size (default: 512)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=8, help="Transformer layers (default: 8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size / concurrent requests (default: 32)",
    )
    args = parser.parse_args()

    print(
        f"Config: hidden={args.hidden_size}, layers={args.num_layers}, "
        f"batch={args.batch_size}, iters={args.num_iters}"
    )
    print(
        f"Ops per step: ~{args.num_layers * 9 + 3} kernel launches "
        f"(7 matmuls + 2 norms per layer, embed + lm_head + argmax)"
    )

    print(f"\n{'=' * 60}")

    single_ops = print_results(
        "Single GPU (baseline)",
        run_single(args.num_iters, args.hidden_size, args.num_layers, args.batch_size),
    )

    threaded_ops = print_results(
        "Threaded (2 GPU, same process)",
        run_threaded(
            args.num_iters, args.hidden_size, args.num_layers, args.batch_size
        ),
    )

    subprocess_ops = print_results(
        "Subprocess (2 GPU, separate processes)",
        run_subprocess(
            args.num_iters, args.hidden_size, args.num_layers, args.batch_size
        ),
    )

    print("\n  --- Summary ---")
    print(
        f"    Threaded vs single:     {threaded_ops / single_ops:.2f}x (ideal: 2.00x)"
    )
    print(
        f"    Subprocess vs single:   {subprocess_ops / single_ops:.2f}x (ideal: 2.00x)"
    )
    ratio = threaded_ops / subprocess_ops
    print(f"    Threaded vs subprocess: {ratio:.2f}x")
    if ratio < 0.95:
        loss = (1 - ratio) * 100
        print(f"    --> {loss:.1f}% throughput loss from in-process contention")
    else:
        print("    --> No significant contention detected")


if __name__ == "__main__":
    main()
