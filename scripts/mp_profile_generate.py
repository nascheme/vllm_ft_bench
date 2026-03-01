# Multi-process vLLM benchmark with CUDA event profiling.
#
# Same approach as threaded_profile_generate.py but uses subprocesses.
# Compares step times and batch dynamics to isolate threading effects.
#
# Uses create_engine() + manual step loop (NOT LLM.generate()) so the
# engine mode is identical to the threaded version.

import os
from multiprocessing import Process, Queue

from vllm_ft.util import make_arg_parser


def engine_generate_profiled(engine, device_index):
    """Step loop with CUDA event timing per step."""
    import time
    import torch

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    outputs = {}
    wall_times = []
    gpu_events = []
    batch_sizes = []

    while engine.has_unfinished_requests():
        core = engine.engine_core.engine_core
        num_running, num_waiting = core.scheduler.get_request_counts()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        wall_t0 = time.perf_counter()
        start_event.record(torch.cuda.current_stream(device))

        step_outputs = engine.step()

        end_event.record(torch.cuda.current_stream(device))
        wall_t1 = time.perf_counter()

        wall_times.append((wall_t1 - wall_t0) * 1000)
        gpu_events.append((start_event, end_event))
        batch_sizes.append(num_running + num_waiting)

        for ro in step_outputs:
            if ro.finished:
                outputs[ro.request_id] = ro

    torch.cuda.synchronize(device)
    gpu_times = [s.elapsed_time(e) for s, e in gpu_events]

    return list(outputs.values()), wall_times, gpu_times, batch_sizes


def run_subprocess_profiled(device_index, model, num_requests, num_gpus, result_queue):
    """Subprocess: create engine, preload requests, step with profiling."""
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

    import argparse
    from vllm import EngineArgs
    from vllm.tokenizers import get_tokenizer
    from vllm.usage.usage_lib import UsageContext
    from vllm_ft.util import (
        apply_forward_context_monkey_patch,
        build_request_items,
        create_engine,
        render_request,
    )

    apply_forward_context_monkey_patch()

    tokenizer = get_tokenizer(model)
    worker_args = argparse.Namespace(
        model=model,
        num_requests=num_requests,
        input_len=1024,
        output_len=128,
        prompt_source="random",
        dataset=None,
    )
    all_request_items = build_request_items(worker_args, tokenizer)

    # Contiguous partition (same as threaded_profile_generate).
    total = len(all_request_items)
    chunk = total // num_gpus
    start_idx = device_index * chunk
    end_idx = start_idx + chunk if device_index < num_gpus - 1 else total
    my_items = all_request_items[start_idx:end_idx]

    engine_args = EngineArgs(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    # device_index=0 because CUDA_VISIBLE_DEVICES remaps to cuda:0.
    engine = create_engine(engine_args, 0, UsageContext.LLM_CLASS)

    renderer = engine.renderer
    for i, (req, sp) in enumerate(my_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"gpu{device_index}_{i}", proc_input, sp)

    t0 = time.time()
    outputs, wall_times, gpu_times, batch_sizes = engine_generate_profiled(engine, 0)
    elapsed = time.time() - t0

    prompt_tokens = sum(
        len(ro.prompt_token_ids) for ro in outputs if ro.prompt_token_ids
    )
    output_tokens = sum(
        sum(len(o.token_ids) for o in ro.outputs if o) for ro in outputs
    )

    result_queue.put(
        {
            "device_index": device_index,
            "completed": len(outputs),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "elapsed": elapsed,
            "wall_times": wall_times,
            "gpu_times": gpu_times,
            "batch_sizes": batch_sizes,
        }
    )


def pcts(data):
    s = sorted(data)
    n = len(s)
    if n == 0:
        return (0, 0, 0, 0, 0)
    return (s[0], s[n // 2], s[int(n * 0.9)], s[min(int(n * 0.99), n - 1)], s[-1])


def steady_state(times):
    n = len(times)
    ss_start = max(1, n // 20)
    ss_end = max(ss_start + 1, n - n // 10)
    return times[ss_start:ss_end]


def print_profile(label, wall_times, gpu_times, batch_sizes):
    n = len(wall_times)
    if n == 0:
        print(f"  {label}: (no steps)")
        return

    total_wall = sum(wall_times)
    total_gpu = sum(gpu_times)

    ss_wall = steady_state(wall_times)
    ss_gpu = steady_state(gpu_times)
    overheads = [w - g for w, g in zip(wall_times, gpu_times)]
    ss_overhead = steady_state(overheads)
    ss_batch = steady_state(batch_sizes)

    print(f"\n  {label} ({n} steps, {len(ss_wall)} steady-state):")
    print(
        f"    Total: wall={total_wall:.0f}ms  gpu={total_gpu:.0f}ms  "
        f"overhead={total_wall - total_gpu:.0f}ms "
        f"({(total_wall - total_gpu) / total_wall * 100:.1f}%)"
    )

    if len(ss_wall) > 2:
        mn, p50, p90, p99, mx = pcts(ss_wall)
        print(
            f"    SS wall (ms):     "
            f"min={mn:.1f}  p50={p50:.1f}  p90={p90:.1f}  "
            f"p99={p99:.1f}  max={mx:.1f}"
        )
        mn, p50, p90, p99, mx = pcts(ss_gpu)
        print(
            f"    SS GPU (ms):      "
            f"min={mn:.1f}  p50={p50:.1f}  p90={p90:.1f}  "
            f"p99={p99:.1f}  max={mx:.1f}"
        )
        mn, p50, p90, p99, mx = pcts(ss_overhead)
        print(
            f"    SS overhead (ms): "
            f"min={mn:.1f}  p50={p50:.1f}  p90={p90:.1f}  "
            f"p99={p99:.1f}  max={mx:.1f}"
        )
        mn, p50, p90, p99, mx = pcts(ss_batch)
        print(f"    SS batch:         min={mn}  p50={p50}  p90={p90}  max={mx}")


def main():
    parser = make_arg_parser(
        "Multi-process vLLM benchmark with CUDA event profiling.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Subprocess dual GPU (cuda:0 + cuda:1) — same engine mode as threaded")
    print("=" * 70)

    rq = Queue()
    procs = []
    for i in range(args.num_gpus):
        p = Process(
            target=run_subprocess_profiled,
            args=(i, args.model, args.num_requests, args.num_gpus, rq),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=300)

    results = []
    while not rq.empty():
        results.append(rq.get())
    results.sort(key=lambda r: r["device_index"])

    if not results:
        print("No results!")
        return

    gen_elapsed = max(r["elapsed"] for r in results)
    total_completed = sum(r["completed"] for r in results)
    print(
        f"\n  {total_completed} reqs, {gen_elapsed:.1f}s, "
        f"{total_completed / gen_elapsed:.1f} req/s"
    )

    for r in results:
        print_profile(
            f"Subprocess cuda:{r['device_index']}",
            r["wall_times"],
            r["gpu_times"],
            r["batch_sizes"],
        )

    # Compare with threaded_profile_generate output.
    print(f"\n{'=' * 70}")
    print("Compare these results with threaded_profile_generate.py output.")
    print("If batch sizes and step counts match, the gap is NOT from")
    print("batch dynamics. If they differ, scheduling behaves differently")
    print("in-process vs subprocess.")
    print("=" * 70)


if __name__ == "__main__":
    main()
