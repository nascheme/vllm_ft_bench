"""CUDA pipeline benchmark — simulates the vLLM async-copy-stream pattern.

Measures:
  Phase A: Kernel launch overhead (threaded vs subprocess vs single)
  Phase B: Pipeline throughput (sync vs async copy, 1 vs 2 copy streams)
  Phase C: Same-GPU multi-thread contention (shared vs dedicated streams)
  Phase D: Saturation sweep & sub-batch splitting
    D1: Compute intensity vs multi-stream gain (layer count sweep)
    D2: Fixed-work sub-batch splitting (split total batch across streams)
    D3: Per-step latency distribution (p50/p95/p99 per sub-batch and wall)

Usage:
  python scripts/cuda_pipeline_bench.py
  python scripts/cuda_pipeline_bench.py --hidden-size 1024 --num-layers 32
  python scripts/cuda_pipeline_bench.py --phase B
  python scripts/cuda_pipeline_bench.py --phase D --total-batch-size 128
  python scripts/cuda_pipeline_bench.py --phase D1 --num-layers 16
"""

from __future__ import annotations

import argparse
import threading
import time
from multiprocessing import Process, Queue

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Core workload
# ---------------------------------------------------------------------------


def make_weights(
    num_layers: int,
    hidden_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Create random weight matrices simulating transformer layers + lm_head."""
    weights = [
        torch.randn(
            hidden_size, hidden_size, device=device, dtype=torch.float16
        )
        for _ in range(num_layers)
    ]
    lm_head = torch.randn(
        hidden_size, vocab_size, device=device, dtype=torch.float16
    )
    return weights, lm_head


def forward_and_sample(
    weights: list[torch.Tensor],
    lm_head: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Chain of matmuls + lm_head projection + argmax.  Returns [batch_size]."""
    for w in weights:
        x = x @ w
    logits = x @ lm_head
    return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Step pipeline
# ---------------------------------------------------------------------------


def run_steps(
    device_index: int,
    num_steps: int,
    warmup_steps: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    vocab_size: int,
    copy_mode: str,  # 'none' | 'sync' | 'async'
    num_copy_streams: int,  # 1 or 2 (for async)
    compute_stream_mode: str,  # 'default' | 'dedicated'
    barrier: threading.Barrier | None = None,
) -> dict:
    """Run forward+sample steps with configurable copy pipeline."""
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    weights, lm_head = make_weights(
        num_layers, hidden_size, vocab_size, device
    )
    x = torch.randn(
        batch_size, hidden_size, device=device, dtype=torch.float16
    )

    # Streams
    compute_stream = (
        torch.cuda.Stream(device=device)
        if compute_stream_mode == "dedicated"
        else torch.cuda.default_stream(device)
    )
    copy_streams = [
        torch.cuda.Stream(device=device) for _ in range(num_copy_streams)
    ]
    copy_events = [torch.cuda.Event() for _ in range(num_copy_streams)]

    cpu_launch_times: list[float] = []
    copy_wait_times: list[float] = []
    prev_cpu_tensor = None
    prev_event_idx = None

    total_steps = warmup_steps + num_steps

    # Barrier-synchronised start
    if barrier is not None:
        torch.cuda.synchronize(device)
        barrier.wait()

    t_start = time.perf_counter()

    for step in range(total_steps):
        recording = step >= warmup_steps

        # --- Sync previous async copy (pipeline: lag by 1) ---
        if copy_mode == "async" and prev_event_idx is not None:
            t0 = time.perf_counter()
            copy_events[prev_event_idx].synchronize()
            dt = time.perf_counter() - t0
            if recording:
                copy_wait_times.append(dt * 1000)
            # Consume the result
            _ = prev_cpu_tensor.tolist()

        # --- Forward pass ---
        t0 = time.perf_counter()
        with torch.cuda.stream(compute_stream):
            token_ids = forward_and_sample(weights, lm_head, x)
        dt_launch = time.perf_counter() - t0
        if recording:
            cpu_launch_times.append(dt_launch * 1000)

        # --- Copy ---
        if copy_mode == "none":
            pass
        elif copy_mode == "sync":
            t0 = time.perf_counter()
            torch.cuda.synchronize(device)
            cpu_tensor = token_ids.to("cpu")
            dt = time.perf_counter() - t0
            if recording:
                copy_wait_times.append(dt * 1000)
            _ = cpu_tensor.tolist()
        elif copy_mode == "async":
            stream_idx = step % num_copy_streams
            cs = copy_streams[stream_idx]
            cs.wait_stream(compute_stream)
            with torch.cuda.stream(cs):
                cpu_tensor = token_ids.to("cpu", non_blocking=True)
            copy_events[stream_idx].record(cs)
            prev_cpu_tensor = cpu_tensor
            prev_event_idx = stream_idx

    # Drain final async copy
    if copy_mode == "async" and prev_event_idx is not None:
        t0 = time.perf_counter()
        copy_events[prev_event_idx].synchronize()
        dt = time.perf_counter() - t0
        if num_steps > 0:
            copy_wait_times.append(dt * 1000)
        _ = prev_cpu_tensor.tolist()
    elif copy_mode == "none":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - t_start

    result = {
        "device": device_index,
        "elapsed": elapsed,
        "steps_per_sec": num_steps / elapsed if elapsed > 0 else 0,
        "ms_per_step": elapsed / num_steps * 1000 if num_steps > 0 else 0,
    }

    if cpu_launch_times:
        arr = np.array(cpu_launch_times)
        result["cpu_launch_p50"] = float(np.median(arr))
        result["cpu_launch_p95"] = float(np.percentile(arr, 95))

    if copy_wait_times:
        arr = np.array(copy_wait_times)
        result["copy_wait_p50"] = float(np.median(arr))
        result["copy_wait_p95"] = float(np.percentile(arr, 95))

    return result


def run_steps_shared_weights(
    device_index: int,
    num_steps: int,
    warmup_steps: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    vocab_size: int,
    weights: list[torch.Tensor],
    lm_head: torch.Tensor,
    barrier: threading.Barrier | None = None,
) -> dict:
    """Run forward+sample steps using pre-created shared weights.

    Records per-step wall-clock times for latency analysis.
    Uses dedicated compute stream, async copy with 1 stream.
    """
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    x = torch.randn(
        batch_size, hidden_size, device=device, dtype=torch.float16
    )

    compute_stream = torch.cuda.Stream(device=device)
    copy_stream = torch.cuda.Stream(device=device)
    copy_event = torch.cuda.Event()

    step_times: list[float] = []
    step_end_times: list[float] = []
    prev_cpu_tensor = None
    prev_has_copy = False

    total_steps = warmup_steps + num_steps

    if barrier is not None:
        torch.cuda.synchronize(device)
        barrier.wait()

    t_start = time.perf_counter()

    for step in range(total_steps):
        recording = step >= warmup_steps
        t_step_start = time.perf_counter()

        # Sync previous async copy
        if prev_has_copy:
            copy_event.synchronize()
            _ = prev_cpu_tensor.tolist()

        # Forward pass
        with torch.cuda.stream(compute_stream):
            token_ids = forward_and_sample(weights, lm_head, x)

        # Async copy
        copy_stream.wait_stream(compute_stream)
        with torch.cuda.stream(copy_stream):
            cpu_tensor = token_ids.to("cpu", non_blocking=True)
        copy_event.record(copy_stream)
        prev_cpu_tensor = cpu_tensor
        prev_has_copy = True

        t_step_end = time.perf_counter()
        if recording:
            step_times.append((t_step_end - t_step_start) * 1000)
            step_end_times.append(t_step_end)

    # Drain final copy
    if prev_has_copy:
        copy_event.synchronize()
        _ = prev_cpu_tensor.tolist()

    elapsed = time.perf_counter() - t_start

    result = {
        "device": device_index,
        "elapsed": elapsed,
        "steps_per_sec": num_steps / elapsed if elapsed > 0 else 0,
        "ms_per_step": elapsed / num_steps * 1000 if num_steps > 0 else 0,
        "step_times": step_times,
        "step_end_times": step_end_times,
    }
    return result


# ---------------------------------------------------------------------------
# Execution wrappers
# ---------------------------------------------------------------------------


def run_single(device_index: int, **kw) -> list[dict]:
    r = run_steps(device_index, **kw)
    return [r]


def run_threaded(device_indices: list[int], **kw) -> list[dict]:
    barrier = threading.Barrier(len(device_indices))
    results: list[dict | None] = [None] * len(device_indices)

    def worker(idx, dev):
        results[idx] = run_steps(dev, barrier=barrier, **kw)

    threads = [
        threading.Thread(target=worker, args=(i, d))
        for i, d in enumerate(device_indices)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in results if r is not None]


def run_threaded_shared_weights(
    device_index: int,
    num_threads: int,
    weights: list[torch.Tensor],
    lm_head: torch.Tensor,
    **kw,
) -> list[dict]:
    """Spawn N threads on the same GPU, all sharing the same weight tensors.

    Each thread gets its own dedicated stream and processes its sub-batch.
    """
    barrier = threading.Barrier(num_threads)
    results: list[dict | None] = [None] * num_threads

    def worker(idx):
        results[idx] = run_steps_shared_weights(
            device_index,
            weights=weights,
            lm_head=lm_head,
            barrier=barrier,
            **kw,
        )

    threads = [
        threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in results if r is not None]


def _subprocess_worker(device_index: int, queue: Queue, kw: dict):
    r = run_steps(device_index, **kw)
    queue.put(r)


def run_subprocess(device_indices: list[int], **kw) -> list[dict]:
    queue: Queue = Queue()
    procs = [
        Process(target=_subprocess_worker, args=(d, queue, kw))
        for d in device_indices
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    results = []
    while not queue.empty():
        results.append(queue.get_nowait())
    results.sort(key=lambda r: r["device"])
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_results(label: str, results: list[dict]) -> float:
    total_sps = sum(r["steps_per_sec"] for r in results)
    max_elapsed = max(r["elapsed"] for r in results)
    print(f"\n  {label}:")
    for r in results:
        line = f"    cuda:{r['device']}: {r['steps_per_sec']:.1f} steps/s ({r['ms_per_step']:.2f} ms/step)"
        if "cpu_launch_p50" in r:
            line += f"  launch: p50={r['cpu_launch_p50']:.2f} p95={r['cpu_launch_p95']:.2f} ms"
        if "copy_wait_p50" in r:
            line += f"  copy_wait: p50={r['copy_wait_p50']:.2f} p95={r['copy_wait_p95']:.2f} ms"
        print(line)
    print(f"    Total: {total_sps:.1f} steps/s, wall: {max_elapsed:.2f}s")
    return total_sps


def print_ratio(label: str, value: float, ideal: float | None = None):
    s = f"    {label}: {value:.2f}x"
    if ideal is not None:
        s += f" (ideal: {ideal:.2f}x)"
    print(s)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def common_kw(args) -> dict:
    return dict(
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
    )


def phase_a(args):
    """Phase A: Kernel Launch Overhead (copy_mode='none')."""
    print("\n" + "=" * 60)
    print("Phase A: Kernel Launch Overhead")
    print("=" * 60)

    kw = {
        **common_kw(args),
        "copy_mode": "none",
        "num_copy_streams": 1,
        "compute_stream_mode": "default",
    }

    single = print_results("1 GPU single-thread", run_single(0, **kw))

    num_gpus = torch.cuda.device_count()
    devs = list(range(min(num_gpus, 2)))

    if len(devs) >= 2:
        threaded = print_results("2 GPU threaded", run_threaded(devs, **kw))
        subproc = print_results(
            "2 GPU subprocess", run_subprocess(devs, **kw)
        )

        print("\n  --- Phase A Summary ---")
        print_ratio("Threaded vs single", threaded / single, ideal=2.0)
        print_ratio("Subprocess vs single", subproc / single, ideal=2.0)
        ratio = threaded / subproc
        print_ratio("Threaded vs subprocess", ratio)
        if ratio < 0.95:
            print(
                f"    --> {(1 - ratio) * 100:.1f}% throughput loss from driver lock contention"
            )
        else:
            print("    --> No significant driver lock contention")
    else:
        print("\n  (Skipping multi-GPU tests — only 1 GPU available)")


def phase_b(args):
    """Phase B: Pipeline Throughput (sync vs async copy)."""
    print("\n" + "=" * 60)
    print("Phase B: Pipeline Throughput")
    print("=" * 60)

    base_kw = common_kw(args)
    num_gpus = torch.cuda.device_count()
    devs = list(range(min(num_gpus, 2)))

    # Sync copy — single
    sync_kw = {
        **base_kw,
        "copy_mode": "sync",
        "num_copy_streams": 1,
        "compute_stream_mode": "default",
    }
    sync_single = print_results("Sync copy, 1 GPU", run_single(0, **sync_kw))

    # Async 1 stream — single
    async1_kw = {
        **base_kw,
        "copy_mode": "async",
        "num_copy_streams": 1,
        "compute_stream_mode": "default",
    }
    async1_single = print_results(
        "Async 1-stream, 1 GPU", run_single(0, **async1_kw)
    )

    sync_threaded = None
    async1_threaded = None
    async1_subproc = None
    async2_threaded = None

    if len(devs) >= 2:
        sync_threaded = print_results(
            "Sync copy, 2 GPU threaded", run_threaded(devs, **sync_kw)
        )
        async1_threaded = print_results(
            "Async 1-stream, 2 GPU threaded", run_threaded(devs, **async1_kw)
        )
        async1_subproc = print_results(
            "Async 1-stream, 2 GPU subprocess",
            run_subprocess(devs, **async1_kw),
        )

        async2_kw = {
            **base_kw,
            "copy_mode": "async",
            "num_copy_streams": 2,
            "compute_stream_mode": "default",
        }
        async2_threaded = print_results(
            "Async 2-stream, 2 GPU threaded", run_threaded(devs, **async2_kw)
        )

    print("\n  --- Phase B Summary ---")
    print_ratio("Async vs sync (1 GPU)", async1_single / sync_single)
    if (
        sync_threaded
        and async1_threaded
        and async1_subproc
        and async2_threaded
    ):
        print_ratio(
            "Async vs sync (2 GPU threaded)", async1_threaded / sync_threaded
        )
        print_ratio(
            "Threaded vs subprocess (async)", async1_threaded / async1_subproc
        )
        print_ratio(
            "2-stream vs 1-stream (threaded)",
            async2_threaded / async1_threaded,
        )


def phase_c(args):
    """Phase C: Same-GPU Multi-Thread Contention."""
    print("\n" + "=" * 60)
    print("Phase C: Same-GPU Multi-Thread Contention")
    print("=" * 60)

    base_kw = common_kw(args)
    copy_kw = {"copy_mode": "async", "num_copy_streams": 1}
    thread_counts = list(range(1, args.max_threads_per_gpu + 1))

    # Baseline: 1 thread on cuda:0, dedicated stream
    baseline_kw = {**base_kw, **copy_kw, "compute_stream_mode": "dedicated"}
    baseline = print_results(
        "Baseline: 1 thread, cuda:0, dedicated stream",
        run_single(0, **baseline_kw),
    )

    # Sweep thread counts with dedicated streams
    dedicated_results: list[tuple[int, float]] = [(1, baseline)]
    for n in thread_counts:
        if n == 1:
            continue
        kw = {**base_kw, **copy_kw, "compute_stream_mode": "dedicated"}
        sps = print_results(
            f"{n} threads cuda:0, dedicated streams",
            run_threaded([0] * n, **kw),
        )
        dedicated_results.append((n, sps))

    # Sweep thread counts with shared default stream
    shared_results: list[tuple[int, float]] = []
    for n in thread_counts:
        if n == 1:
            kw = {**base_kw, **copy_kw, "compute_stream_mode": "default"}
            sps = print_results(
                "1 thread cuda:0, shared default stream",
                run_single(0, **kw),
            )
        else:
            kw = {**base_kw, **copy_kw, "compute_stream_mode": "default"}
            sps = print_results(
                f"{n} threads cuda:0, shared default stream",
                run_threaded([0] * n, **kw),
            )
        shared_results.append((n, sps))

    print("\n  --- Phase C Summary: Throughput Curve ---")
    print(
        f"    {'threads':>7}  {'dedicated':>12}  {'vs base':>8}  {'shared':>12}  {'vs base':>8}  {'ded/shared':>10}"
    )
    for (n_d, sps_d), (n_s, sps_s) in zip(dedicated_results, shared_results):
        assert n_d == n_s
        print(
            f"    {n_d:>7}  {sps_d:>10.1f}/s  {sps_d / baseline:>7.2f}x"
            f"  {sps_s:>10.1f}/s  {sps_s / baseline:>7.2f}x"
            f"  {sps_d / sps_s:>9.2f}x"
        )


def phase_d1(args):
    """D1: Compute Intensity vs Multi-Stream Gain."""
    print("\n" + "=" * 60)
    print("Phase D1: Compute Intensity vs Multi-Stream Gain")
    print("=" * 60)

    num_threads = 4
    layer_counts = [4, 8, 16, 32, 64, 128]

    base_kw = {
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "copy_mode": "async",
        "num_copy_streams": 1,
        "compute_stream_mode": "dedicated",
    }

    rows: list[tuple[int, float, float]] = []

    for nl in layer_counts:
        kw = {**base_kw, "num_layers": nl}
        print(f"\n  --- num_layers={nl} ---")
        try:
            r1 = run_single(0, **kw)
            sps_1t = print_results(f"  1 thread, {nl} layers", r1)
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {nl} layers (1 thread), skipping")
            torch.cuda.empty_cache()
            break

        try:
            rn = run_threaded([0] * num_threads, **kw)
            sps_nt = print_results(
                f"  {num_threads} threads, {nl} layers", rn
            )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {nl} layers ({num_threads} threads), skipping")
            torch.cuda.empty_cache()
            break

        rows.append((nl, sps_1t, sps_nt))

    if rows:
        print("\n  --- D1 Summary ---")
        print(
            f"    {'layers':>6}  {'1-thread sps':>12}  "
            f"{'4-thread sps':>12}  {'speedup':>8}  "
            f"{'ms/step(1T)':>11}  {'ms/step(4T)':>11}"
        )
        for nl, sps1, spsn in rows:
            speedup = spsn / sps1 if sps1 > 0 else 0
            ms1 = 1000.0 / sps1 if sps1 > 0 else 0
            msn = 1000.0 / spsn if spsn > 0 else 0
            print(
                f"    {nl:>6}  {sps1:>10.1f}/s  {spsn:>10.1f}/s"
                f"  {speedup:>7.3f}x  {ms1:>10.2f}  {msn:>10.2f}"
            )


def phase_d2(args) -> list[tuple[int, int, float, float, list[dict]]]:
    """D2: Fixed-Work Sub-Batch Splitting.

    Returns list of (splits, sub_batch, wall_time, eff_tput, results) for D3.
    """
    print("\n" + "=" * 60)
    print("Phase D2: Fixed-Work Sub-Batch Splitting")
    print("=" * 60)

    total_batch = args.total_batch_size
    split_counts = [1, 2, 4, 8]
    device_index = 0
    device = torch.device(f"cuda:{device_index}")

    step_kw = {
        "num_steps": args.num_steps,
        "warmup_steps": args.warmup_steps,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
    }

    # Create shared weights once
    weights, lm_head = make_weights(
        args.num_layers, args.hidden_size, args.vocab_size, device
    )

    rows: list[tuple[int, int, float, float, list[dict]]] = []
    baseline_tput = None

    for n_splits in split_counts:
        if total_batch % n_splits != 0:
            continue
        sub_batch = total_batch // n_splits
        print(f"\n  --- {n_splits} split(s), sub_batch={sub_batch} ---")

        try:
            if n_splits == 1:
                results = [
                    run_steps_shared_weights(
                        device_index,
                        batch_size=sub_batch,
                        num_layers=args.num_layers,
                        weights=weights,
                        lm_head=lm_head,
                        **step_kw,
                    )
                ]
            else:
                results = run_threaded_shared_weights(
                    device_index,
                    num_threads=n_splits,
                    weights=weights,
                    lm_head=lm_head,
                    batch_size=sub_batch,
                    num_layers=args.num_layers,
                    **step_kw,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at {n_splits} splits, skipping")
            torch.cuda.empty_cache()
            continue

        wall_time = max(r["elapsed"] for r in results)
        eff_tput = total_batch * args.num_steps / wall_time
        avg_step_ms = wall_time / args.num_steps * 1000

        if baseline_tput is None:
            baseline_tput = eff_tput

        vs = eff_tput / baseline_tput if baseline_tput > 0 else 0

        print(
            f"    wall={wall_time:.2f}s  eff_tput={eff_tput:.1f} tok/s"
            f"  vs_baseline={vs:.2f}x  avg_step={avg_step_ms:.2f}ms"
        )
        rows.append((n_splits, sub_batch, wall_time, eff_tput, results))

    if rows:
        baseline_tput = rows[0][3]
        print("\n  --- D2 Summary ---")
        print(
            f"    {'splits':>6}  {'sub_batch':>9}  {'wall_time(s)':>12}  "
            f"{'eff_tput(tok/s)':>15}  {'vs_baseline':>11}  {'avg_step_ms':>11}"
        )
        for n_splits, sub_batch, wall_time, eff_tput, _ in rows:
            vs = eff_tput / baseline_tput if baseline_tput > 0 else 0
            avg_step_ms = wall_time / args.num_steps * 1000
            print(
                f"    {n_splits:>6}  {sub_batch:>9}  {wall_time:>12.2f}  "
                f"{eff_tput:>15.1f}  {vs:>10.2f}x  {avg_step_ms:>11.2f}"
            )

    return rows


def phase_d3(args, d2_rows: list[tuple[int, int, float, float, list[dict]]]):
    """D3: Per-Step Latency Distribution."""
    print("\n" + "=" * 60)
    print("Phase D3: Per-Step Latency Distribution")
    print("=" * 60)

    if not d2_rows:
        print("  No D2 data available, skipping D3")
        return

    baseline_p50 = None
    rows: list[tuple] = []

    for n_splits, sub_batch, _, _, results in d2_rows:
        # Per-sub-batch step latency
        all_step_times = []
        for r in results:
            all_step_times.extend(r.get("step_times", []))

        if not all_step_times:
            continue

        arr = np.array(all_step_times)
        p50 = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))

        # Wall step latency: max across threads per step
        # Align by step_end_times — for each step, take the max end time
        # minus the previous max end time (or start)
        if n_splits == 1:
            wall_steps = np.array(results[0].get("step_times", []))
        else:
            # Use step_end_times to compute per-wall-step durations
            end_arrays = [np.array(r["step_end_times"]) for r in results]
            # All threads should have same number of steps
            min_len = min(len(a) for a in end_arrays)
            if min_len > 0:
                stacked = np.stack([a[:min_len] for a in end_arrays])
                wall_ends = np.max(stacked, axis=0)  # max end time per step
                wall_steps = np.diff(wall_ends) * 1000  # ms
            else:
                wall_steps = np.array([])

        if len(wall_steps) > 0:
            wall_p50 = float(np.median(wall_steps))
            wall_p95 = float(np.percentile(wall_steps, 95))
        else:
            wall_p50 = p50
            wall_p95 = p95

        if baseline_p50 is None:
            baseline_p50 = wall_p50

        vs_p50 = wall_p50 / baseline_p50 if baseline_p50 > 0 else 0

        rows.append(
            (n_splits, sub_batch, p50, p95, p99, wall_p50, wall_p95, vs_p50)
        )

    if rows:
        print("\n  --- D3 Summary ---")
        print(
            f"    {'splits':>6}  {'sub_batch':>9}  {'step_p50(ms)':>12}  "
            f"{'step_p95(ms)':>12}  {'step_p99(ms)':>12}  "
            f"{'wall_step_p50(ms)':>17}  {'vs_baseline_p50':>15}"
        )
        for n_splits, sub_batch, p50, p95, p99, wp50, wp95, vs in rows:
            print(
                f"    {n_splits:>6}  {sub_batch:>9}  {p50:>12.2f}  "
                f"{p95:>12.2f}  {p99:>12.2f}  "
                f"{wp50:>17.2f}  {vs:>14.2f}x"
            )


def phase_d(args):
    """Phase D: Saturation Sweep & Sub-Batch Splitting."""
    phase = args.phase

    if phase in ("D", "D1"):
        phase_d1(args)
    if phase in ("D", "D2", "D3"):
        d2_rows = phase_d2(args)
    if phase in ("D", "D3"):
        phase_d3(args, d2_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CUDA pipeline benchmark — simulates vLLM async-copy-stream pattern"
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--max-threads-per-gpu",
        type=int,
        default=8,
        help="Max threads per GPU for Phase C sweep (default: 8)",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=128,
        help="Total batch size for Phase D sub-batch splitting (default: 128)",
    )
    parser.add_argument(
        "--phase",
        choices=["A", "B", "C", "D", "D1", "D2", "D3", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"CUDA pipeline bench — {num_gpus} GPU(s)")
    print(
        f"  hidden={args.hidden_size}, layers={args.num_layers}, "
        f"batch={args.batch_size}, vocab={args.vocab_size}"
    )
    print(f"  steps={args.num_steps}, warmup={args.warmup_steps}")

    if args.phase in ("A", "all"):
        phase_a(args)
    if args.phase in ("B", "all"):
        phase_b(args)
    if args.phase in ("C", "all"):
        phase_c(args)
    if args.phase in ("D", "D1", "D2", "D3"):
        phase_d(args)

    print()


if __name__ == "__main__":
    main()
