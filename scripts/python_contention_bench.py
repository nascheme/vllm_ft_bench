# Python-level contention benchmark for free-threaded Python.
#
# Tests whether heavy Python object manipulation (dicts, lists, dataclasses)
# between CUDA steps causes contention in free-threaded mode.
#
# Simulates the Python work vLLM does per step:
# - Scheduler: iterate dicts, check request states, build output lists
# - Output processing: create RequestOutput objects, string manipulation
# - KV cache management: dict lookups, list slicing
#
# Three levels of Python work mixed with CUDA ops:
#   none:   Pure CUDA (baseline, same as bench2)
#   light:  Small dict/list ops per step
#   heavy:  Realistic vLLM-scale Python work per step

import argparse
import threading
import time
from multiprocessing import Process, Queue


class FakeRequest:
    """Mimics vLLM Request object — lots of attributes, created/destroyed."""

    __slots__ = (
        "request_id",
        "prompt_tokens",
        "output_tokens",
        "num_computed",
        "status",
        "kv_block_ids",
        "sampling_params",
    )

    def __init__(self, request_id, prompt_len):
        self.request_id = request_id
        self.prompt_tokens = list(range(prompt_len))
        self.output_tokens = []
        self.num_computed = 0
        self.status = "running"
        self.kv_block_ids = list(range(prompt_len // 16 + 1))
        self.sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 128,
        }


class FakeRequestOutput:
    """Mimics vLLM RequestOutput."""

    __slots__ = ("request_id", "prompt_token_ids", "outputs", "finished")

    def __init__(self, request_id, prompt_ids, output_ids, finished):
        self.request_id = request_id
        self.prompt_token_ids = prompt_ids
        self.outputs = [{"token_ids": output_ids, "text": f"token_{len(output_ids)}"}]
        self.finished = finished


def simulate_scheduler_work(requests, step_num):
    """Simulate scheduler.schedule() — iterate requests, build batch."""
    scheduled = {}
    num_tokens = 0
    for req_id, req in requests.items():
        if req.status != "running":
            continue
        tokens_to_schedule = 1  # decode step
        scheduled[req_id] = tokens_to_schedule
        num_tokens += tokens_to_schedule
        if num_tokens >= 256:
            break
    return scheduled


def simulate_update_from_output(requests, scheduled, step_num):
    """Simulate scheduler.update_from_output() — update request state."""
    finished_ids = []
    outputs = []
    for req_id, num_tokens in scheduled.items():
        req = requests[req_id]
        req.output_tokens.append(step_num)
        req.num_computed += num_tokens

        finished = len(req.output_tokens) >= req.sampling_params["max_tokens"]
        if finished:
            req.status = "finished"
            finished_ids.append(req_id)

        # Build output (like output_processor.process_outputs).
        outputs.append(
            FakeRequestOutput(
                req_id,
                req.prompt_tokens[:],  # copy, like vLLM does
                req.output_tokens[:],
                finished,
            )
        )

    for req_id in finished_ids:
        del requests[req_id]

    return outputs


def worker(
    device_index,
    num_iters,
    hidden_size,
    num_layers,
    batch_size,
    python_work,
    result_holder,
):
    """Run transformer steps + Python work on one GPU."""
    import torch

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dtype = torch.float16

    # Model weights.
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

    # Fake request pool (for Python work modes).
    requests = {}
    if python_work != "none":
        for i in range(batch_size * 4):
            requests[f"req_{i}"] = FakeRequest(f"req_{i}", prompt_len=256)

    # Warmup.
    for _ in range(3):
        h = embed_weight[token_ids]
        for layer in layers:
            residual = h
            h = rmsnorm(h, layer["norm1"])
            _q = h @ layer["wq"]
            _k = h @ layer["wk"]
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
        # --- Python work BEFORE GPU step (like scheduler.schedule) ---
        if python_work == "light":
            # Light: small dict iteration + list append.
            batch = {k: 1 for k in list(requests.keys())[:batch_size]}
        elif python_work == "heavy":
            batch = simulate_scheduler_work(requests, step)
        else:
            batch = {}

        # --- GPU step (transformer forward) ---
        h = embed_weight[token_ids]
        for layer in layers:
            residual = h
            h = rmsnorm(h, layer["norm1"])
            _q = h @ layer["wq"]
            _k = h @ layer["wk"]
            v = h @ layer["wv"]
            h = residual + v @ layer["wo"]
            residual = h
            h = rmsnorm(h, layer["norm2"])
            gate = h @ layer["w_gate"]
            up = h @ layer["w_up"]
            h = residual + (torch.nn.functional.silu(gate) * up) @ layer["w_down"]
        logits = h @ lm_head
        token_ids = logits.argmax(dim=-1)

        # --- Python work AFTER GPU step (like update_from_output) ---
        if python_work == "light":
            # Light: create some output objects.
            for req_id in list(batch.keys())[:10]:
                _ = FakeRequestOutput(req_id, [1, 2, 3], [step], False)
        elif python_work == "heavy":
            if batch:
                outputs = simulate_update_from_output(requests, batch, step)
                # Replenish finished requests (keep pool stable).
                for out in outputs:
                    if out.finished:
                        new_id = f"req_{step}_{out.request_id}"
                        requests[new_id] = FakeRequest(new_id, prompt_len=256)

        # Periodic CPU readback.
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
    python_work,
    result_queue,
):
    result = {}
    worker(
        device_index,
        num_iters,
        hidden_size,
        num_layers,
        batch_size,
        python_work,
        result,
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


def run_all(num_iters, hidden_size, num_layers, batch_size, python_work):
    print(f"\n{'=' * 60}")
    print(f"Python work: {python_work}")
    print(f"{'=' * 60}")

    # Single.
    r = {}
    worker(0, num_iters, hidden_size, num_layers, batch_size, python_work, r)
    single = print_results("Single GPU", [r])

    # Threaded.
    results = [{} for _ in range(2)]
    threads = []
    for i in range(2):
        t = threading.Thread(
            target=worker,
            args=(
                i,
                num_iters,
                hidden_size,
                num_layers,
                batch_size,
                python_work,
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
            args=(i, num_iters, hidden_size, num_layers, batch_size, python_work, rq),
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
    parser = argparse.ArgumentParser(
        description="Python contention benchmark for free-threaded Python"
    )
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(
        f"Config: hidden={args.hidden_size}, layers={args.num_layers}, "
        f"batch={args.batch_size}, iters={args.num_iters}"
    )

    results = {}
    for python_work in ["none", "light", "heavy"]:
        s, t, p = run_all(
            args.num_iters,
            args.hidden_size,
            args.num_layers,
            args.batch_size,
            python_work,
        )
        results[python_work] = {"single": s, "threaded": t, "subprocess": p}

    print(f"\n{'=' * 60}")
    print("Summary: Threaded/Subprocess ratio by Python work level")
    print(f"{'=' * 60}")
    for level, r in results.items():
        ratio = r["threaded"] / r["subprocess"]
        bar = "#" * int(ratio * 40)
        print(f"  {level:8s}: {ratio:.2f}x  {bar}")
    print("\n  If ratio decreases with more Python work, the contention")
    print("  is in Python runtime (refcounting, allocator, GC).")
    print("  If ratio is stable, look at vLLM-specific global state.")


if __name__ == "__main__":
    main()
