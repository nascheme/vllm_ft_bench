"""CUDA Event blocking=True/False microbenchmark.

Investigates whether torch.cuda.Event(blocking=True/False) affects CPU spinning
behavior during event.synchronize().

Hypothesis: blocking=True should cause the CPU to block (sleep/yield) rather
than busyloop. If cpu_frac ≈ 1.0 for both modes, the CPU is spinning regardless.

Key metric: cpu_frac = cpu_time / wall_time
  ~1.0  → CPU is spinning (busy-waiting)
  ~0.0  → CPU is blocking (sleeping/yielding)

Usage:
  uv run scripts/cuda_event_blocking_bench.py
  uv run scripts/cuda_event_blocking_bench.py --wait-ms 1 5 20 --num-iters 200
  uv run scripts/cuda_event_blocking_bench.py --profile
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# GPU sleep calibration
# ---------------------------------------------------------------------------


def calibrate_cycles_per_ms(
    device: torch.device, target_ms: float = 10.0
) -> float:
    """Estimate how many torch.cuda._sleep cycles correspond to one millisecond.

    Runs a binary-search / timing loop to find cycles_per_ms empirically.
    """
    # Warm up
    torch.cuda._sleep(1000)
    torch.cuda.synchronize(device)

    # Start with a rough estimate and refine
    cycles = int(1_000_000)
    for _ in range(5):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        torch.cuda._sleep(cycles)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 0:
            cycles = int(cycles * target_ms / elapsed_ms)

    # Final measurement
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    torch.cuda._sleep(cycles)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    cycles_per_ms = cycles / elapsed_ms if elapsed_ms > 0 else cycles
    return cycles_per_ms


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


def measure_sync(
    blocking: bool,
    wait_ms: float,
    num_iters: int,
    warmup: int,
    cycles_per_ms: float,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[list[float], list[float]]:
    """Run num_iters iterations and return (wall_times_ms, cpu_times_ms).

    Each iteration:
      1. Launch GPU sleep kernel on stream
      2. Record event on stream
      3. Measure wall + CPU time of event.synchronize()
    """
    cycles = int(wait_ms * cycles_per_ms)
    event = torch.cuda.Event(blocking=blocking)

    wall_times: list[float] = []
    cpu_times: list[float] = []

    total = warmup + num_iters
    for i in range(total):
        # Launch sleep on the stream
        with torch.cuda.stream(stream):
            torch.cuda._sleep(cycles)
        event.record(stream)

        # Measure synchronize
        t_wall0 = time.perf_counter()
        t_cpu0 = time.process_time()
        event.synchronize()
        t_wall1 = time.perf_counter()
        t_cpu1 = time.process_time()

        if i >= warmup:
            wall_times.append((t_wall1 - t_wall0) * 1000.0)
            cpu_times.append((t_cpu1 - t_cpu0) * 1000.0)

    return wall_times, cpu_times


# ---------------------------------------------------------------------------
# Profiler helper
# ---------------------------------------------------------------------------


def run_profiled(
    blocking: bool,
    wait_ms: float,
    cycles_per_ms: float,
    device: torch.device,
    stream: torch.cuda.Stream,
    profile_iters: int,
    output_path: str,
) -> None:
    """Run a small profiled trace and export Chrome JSON."""
    cycles = int(wait_ms * cycles_per_ms)
    event = torch.cuda.Event(blocking=blocking)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_stack=False,
        record_shapes=False,
    ) as prof:
        for _ in range(profile_iters):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(cycles)
            event.record(stream)
            event.synchronize()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prof.export_chrome_trace(output_path)
    print(f"  Profiler trace written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark torch.cuda.Event(blocking=True/False) CPU spin behavior"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index"
    )
    parser.add_argument(
        "--wait-ms",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 20.0],
        metavar="MS",
        help="GPU sleep durations to test in ms (default: 1.0 5.0 20.0)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Iterations per (blocking, wait_ms) condition (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Export a Chrome trace via torch.profiler to output/event_blocking_profile.json",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=20,
        help="Iterations to capture in profiler trace (default: 20)",
    )
    parser.add_argument(
        "--profile-wait-ms",
        type=float,
        default=5.0,
        help="GPU sleep duration for profiler trace (default: 5.0)",
    )
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    stream = torch.cuda.Stream(device=device)

    print(f"cuda_event_blocking_bench — cuda:{args.device}")
    print(
        f"  wait_ms={args.wait_ms}  num_iters={args.num_iters}  warmup={args.warmup}"
    )
    print()

    # Calibrate GPU sleep
    print("Calibrating GPU sleep cycles... ", end="", flush=True)
    cycles_per_ms = calibrate_cycles_per_ms(device)
    print(f"{cycles_per_ms:.0f} cycles/ms")
    print()

    # Header
    col_w = dict(blocking=9, wait_ms=9, wall_p50=12, cpu_p50=11, cpu_frac=10)
    header = (
        f"  {'blocking':>{col_w['blocking']}}"
        f"  {'wait_ms':>{col_w['wait_ms']}}"
        f"  {'wall_p50_ms':>{col_w['wall_p50']}}"
        f"  {'cpu_p50_ms':>{col_w['cpu_p50']}}"
        f"  {'cpu_frac':>{col_w['cpu_frac']}}"
        f"  {'wall_p95_ms':>{col_w['wall_p50']}}"
        f"  {'cpu_p95_ms':>{col_w['cpu_p50']}}"
        f"  interpretation"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for blocking in (False, True):
        for wait_ms in args.wait_ms:
            wall_times, cpu_times = measure_sync(
                blocking=blocking,
                wait_ms=wait_ms,
                num_iters=args.num_iters,
                warmup=args.warmup,
                cycles_per_ms=cycles_per_ms,
                device=device,
                stream=stream,
            )

            wall_arr = np.array(wall_times)
            cpu_arr = np.array(cpu_times)

            wall_p50 = float(np.median(wall_arr))
            wall_p95 = float(np.percentile(wall_arr, 95))
            cpu_p50 = float(np.median(cpu_arr))
            cpu_p95 = float(np.percentile(cpu_arr, 95))
            cpu_frac = cpu_p50 / wall_p50 if wall_p50 > 0 else 0.0

            if cpu_frac >= 0.8:
                interp = "SPIN (CPU busy)"
            elif cpu_frac <= 0.1:
                interp = "BLOCK (CPU idle)"
            else:
                interp = f"PARTIAL ({cpu_frac:.0%} CPU)"

            print(
                f"  {str(blocking):>{col_w['blocking']}}"
                f"  {wait_ms:>{col_w['wait_ms']}.1f}"
                f"  {wall_p50:>{col_w['wall_p50']}.3f}"
                f"  {cpu_p50:>{col_w['cpu_p50']}.3f}"
                f"  {cpu_frac:>{col_w['cpu_frac']}.3f}"
                f"  {wall_p95:>{col_w['wall_p50']}.3f}"
                f"  {cpu_p95:>{col_w['cpu_p50']}.3f}"
                f"  {interp}"
            )

    print()
    print("Key: cpu_frac = cpu_p50 / wall_p50")
    print(
        "  ~1.0 → CPU is spinning (busy-waiting) during event.synchronize()"
    )
    print(
        "  ~0.0 → CPU is blocking (sleeping/yielding) during event.synchronize()"
    )

    if args.profile:
        print()
        print(
            f"Running profiler trace (blocking=False, wait_ms={args.profile_wait_ms})..."
        )
        run_profiled(
            blocking=False,
            wait_ms=args.profile_wait_ms,
            cycles_per_ms=cycles_per_ms,
            device=device,
            stream=stream,
            profile_iters=args.profile_iters,
            output_path="output/event_blocking_profile_False.json",
        )
        print(
            f"Running profiler trace (blocking=True,  wait_ms={args.profile_wait_ms})..."
        )
        run_profiled(
            blocking=True,
            wait_ms=args.profile_wait_ms,
            cycles_per_ms=cycles_per_ms,
            device=device,
            stream=stream,
            profile_iters=args.profile_iters,
            output_path="output/event_blocking_profile_True.json",
        )
        print()
        print(
            "Load traces in Chrome at chrome://tracing or https://ui.perfetto.dev"
        )


if __name__ == "__main__":
    main()
