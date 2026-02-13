# CUDA driver lock contention benchmark.
#
# Measures whether running CUDA operations on two GPUs from threads in the
# same process is slower than from separate processes. No vLLM — pure PyTorch.
#
# Runs a tight loop of matmuls (simulating model forward passes) and measures
# throughput. Three modes:
#
#   --mode single     Single GPU baseline (cuda:0 only)
#   --mode threaded   Two GPUs, one thread each, same process
#   --mode subprocess Two GPUs, one process each
#
# If threaded is slower than subprocess, it confirms CUDA driver lock
# contention as the bottleneck.
#
# Usage:
#   python cuda_contention_bench.py --mode single
#   python cuda_contention_bench.py --mode threaded
#   python cuda_contention_bench.py --mode subprocess
#   python cuda_contention_bench.py --all          # run all three

import argparse
import time
import threading
from multiprocessing import Process, Queue


def gpu_workload(device_index, num_iters, size, result_holder):
    """Run a tight loop of matmuls on one GPU.

    Simulates the core GPU work of an LLM decode step: repeated small-to-medium
    matrix multiplications with synchronization after each batch.
    """
    import torch

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    # Pre-allocate tensors (simulates model weights + activations).
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup.
    for _ in range(10):
        _c = torch.mm(a, b)
    torch.cuda.synchronize(device)

    # Timed loop.
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    ops_per_sec = num_iters / elapsed
    result_holder["device"] = device_index
    result_holder["elapsed"] = elapsed
    result_holder["ops_per_sec"] = ops_per_sec


def gpu_workload_mixed(device_index, num_iters, size, result_holder):
    """Mixed workload: matmuls + small ops + allocations.

    More realistic than pure matmuls — includes the kind of small CUDA API
    calls (tensor creation, copies, small kernels) that are most affected
    by driver lock contention.
    """
    import torch

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup.
    for _ in range(10):
        c = torch.mm(a, b)
        _ = c.sum().item()
    torch.cuda.synchronize(device)

    # Timed loop: matmul + small ops per iteration.
    t0 = time.perf_counter()
    for i in range(num_iters):
        # Large compute (like model forward).
        c = torch.mm(a, b)
        # Small ops (like token handling, index ops).
        idx = torch.tensor([i % size], device=device)
        row = c[idx]
        # Occasional sync (like reading output tokens back to CPU).
        if i % 50 == 0:
            _ = row.cpu()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    result_holder["device"] = device_index
    result_holder["elapsed"] = elapsed
    result_holder["ops_per_sec"] = num_iters / elapsed


def subprocess_worker(device_index, num_iters, size, workload, result_queue):
    """Entry point for subprocess mode."""
    result = {}
    if workload == "matmul":
        gpu_workload(device_index, num_iters, size, result)
    else:
        gpu_workload_mixed(device_index, num_iters, size, result)
    result_queue.put(result)


def run_single(num_iters, size, workload_fn):
    """Single GPU baseline."""
    result = {}
    workload_fn(0, num_iters, size, result)
    return [result]


def run_threaded(num_iters, size, workload_fn, num_gpus=2):
    """Two GPUs, threaded."""
    results = [{} for _ in range(num_gpus)]
    threads = []
    for i in range(num_gpus):
        t = threading.Thread(
            target=workload_fn,
            args=(i, num_iters, size, results[i]),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


def run_subprocess(num_iters, size, workload_name, num_gpus=2):
    """Two GPUs, separate processes."""
    result_queue = Queue()
    procs = []
    for i in range(num_gpus):
        p = Process(
            target=subprocess_worker,
            args=(i, num_iters, size, workload_name, result_queue),
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
            f"    cuda:{r['device']}: {r['ops_per_sec']:.0f} ops/s "
            f"({r['elapsed']:.3f}s)"
        )
    print(f"    Total: {total_ops:.0f} ops/s, wall time: {max_elapsed:.3f}s")
    return total_ops, max_elapsed


def main():
    parser = argparse.ArgumentParser(
        description="CUDA driver lock contention benchmark (pure PyTorch)"
    )
    parser.add_argument(
        "--mode", choices=["single", "threaded", "subprocess"], help="Run mode"
    )
    parser.add_argument("--all", action="store_true", help="Run all modes")
    parser.add_argument(
        "--num-iters", type=int, default=2000, help="Iterations per GPU (default: 2000)"
    )
    parser.add_argument(
        "--size", type=int, default=1024, help="Matrix size NxN (default: 1024)"
    )
    parser.add_argument(
        "--workload",
        choices=["matmul", "mixed"],
        default="both",
        help="Workload type (default: both)",
    )
    args = parser.parse_args()

    if not args.mode and not args.all:
        args.all = True

    workloads = []
    if args.workload == "matmul":
        workloads = [("matmul", gpu_workload)]
    elif args.workload == "mixed":
        workloads = [("mixed", gpu_workload_mixed)]
    else:
        workloads = [("matmul", gpu_workload), ("mixed", gpu_workload_mixed)]

    for workload_name, workload_fn in workloads:
        print(f"\n{'=' * 60}")
        print(f"Workload: {workload_name}  (size={args.size}, iters={args.num_iters})")
        print(f"{'=' * 60}")

        modes = []
        if args.all:
            modes = ["single", "threaded", "subprocess"]
        else:
            modes = [args.mode]

        all_results = {}

        for mode in modes:
            if mode == "single":
                results = run_single(args.num_iters, args.size, workload_fn)
                ops, elapsed = print_results("Single GPU (baseline)", results)
                all_results["single"] = ops

            elif mode == "threaded":
                results = run_threaded(args.num_iters, args.size, workload_fn)
                ops, elapsed = print_results("Threaded (2 GPU, same process)", results)
                all_results["threaded"] = ops

            elif mode == "subprocess":
                results = run_subprocess(args.num_iters, args.size, workload_name)
                ops, elapsed = print_results(
                    "Subprocess (2 GPU, separate processes)", results
                )
                all_results["subprocess"] = ops

        # Summary comparison.
        if len(all_results) > 1:
            print(f"\n  --- Summary ({workload_name}) ---")
            baseline = all_results.get("single", 0)
            if baseline and "threaded" in all_results:
                ratio = all_results["threaded"] / baseline
                print(f"    Threaded vs single:    {ratio:.2f}x (ideal: 2.00x)")
            if baseline and "subprocess" in all_results:
                ratio = all_results["subprocess"] / baseline
                print(f"    Subprocess vs single:  {ratio:.2f}x (ideal: 2.00x)")
            if "threaded" in all_results and "subprocess" in all_results:
                ratio = all_results["threaded"] / all_results["subprocess"]
                print(
                    f"    Threaded vs subprocess: {ratio:.2f}x (1.00 = no contention)"
                )
                if ratio < 0.95:
                    loss = (1 - ratio) * 100
                    print(
                        f"    --> {loss:.1f}% throughput loss from "
                        f"in-process CUDA contention"
                    )


if __name__ == "__main__":
    main()
