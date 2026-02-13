# Threaded multi-GPU vLLM benchmark with CUDA event profiling.
#
# Measures GPU-only time vs wall time per engine.step() to determine
# whether the threading slowdown is in GPU kernel execution or in
# Python/CPU work surrounding the GPU calls.
#
# Runs two phases in separate subprocesses (clean GPU memory each time):
#   Phase 1: Single GPU baseline (cuda:0 only)
#   Phase 2: Threaded dual GPU (cuda:0 + cuda:1)

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

    # Resolve GPU times.
    torch.cuda.synchronize(device)
    gpu_times = [s.elapsed_time(e) for s, e in gpu_events]

    return list(outputs.values()), wall_times, gpu_times, batch_sizes


def run_single_gpu(model, num_requests, result_queue):
    """Phase 1: single GPU baseline in a clean subprocess."""
    import os
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import argparse
    from vllm import EngineArgs
    from vllm.tokenizers import get_tokenizer
    from vllm.usage.usage_lib import UsageContext
    from vllm_ft.util import (
        apply_forward_context_monkey_patch,
        build_request_items,
        create_engine,
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
    request_items = build_request_items(worker_args, tokenizer)
    # Take first half (same partition as cuda:0 in dual mode).
    half = len(request_items) // 2
    my_items = request_items[:half]

    engine_args = EngineArgs(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    engine = create_engine(engine_args, 0, UsageContext.LLM_CLASS)

    input_processor = engine.input_processor
    for i, (req, sp) in enumerate(my_items):
        ecr = input_processor.process_inputs(
            f"single_{i}",
            req.prompt,
            sp,
            arrival_time=time.time(),
            supported_tasks=engine.get_supported_tasks(),
        )
        engine.add_request(ecr.request_id, ecr, sp, prompt_text=req.prompt)

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
            "completed": len(outputs),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "elapsed": elapsed,
            "wall_times": wall_times,
            "gpu_times": gpu_times,
            "batch_sizes": batch_sizes,
        }
    )


def run_threaded_dual(model, num_requests, num_gpus, result_queue):
    """Phase 2: threaded dual GPU in a clean subprocess."""
    import os
    import time

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import argparse
    import threading
    import torch
    from vllm import EngineArgs
    from vllm.tokenizers import get_tokenizer
    from vllm.usage.usage_lib import UsageContext
    from vllm_ft.util import (
        apply_forward_context_monkey_patch,
        build_request_items,
        create_engine,
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
    request_items = build_request_items(worker_args, tokenizer)
    total = len(request_items)
    chunk = total // num_gpus

    engine_args = EngineArgs(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    engines = []
    for i in range(num_gpus):
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))

    def engine_worker(engine, device_index, my_items, result, barrier):
        torch.cuda.set_device(device_index)
        input_processor = engine.input_processor
        for i, (req, sp) in enumerate(my_items):
            ecr = input_processor.process_inputs(
                f"gpu{device_index}_{i}",
                req.prompt,
                sp,
                arrival_time=time.time(),
                supported_tasks=engine.get_supported_tasks(),
            )
            engine.add_request(ecr.request_id, ecr, sp, prompt_text=req.prompt)

        barrier.wait()
        t0 = time.time()
        outputs, wall_times, gpu_times, batch_sizes = engine_generate_profiled(
            engine, device_index
        )
        t1 = time.time()

        prompt_tokens = sum(
            len(ro.prompt_token_ids) for ro in outputs if ro.prompt_token_ids
        )
        output_tokens = sum(
            sum(len(o.token_ids) for o in ro.outputs if o) for ro in outputs
        )
        result["completed"] = len(outputs)
        result["prompt_tokens"] = prompt_tokens
        result["output_tokens"] = output_tokens
        result["elapsed"] = t1 - t0
        result["wall_times"] = wall_times
        result["gpu_times"] = gpu_times
        result["batch_sizes"] = batch_sizes

    barrier = threading.Barrier(num_gpus)
    results = [{} for _ in range(num_gpus)]
    threads = []
    for i in range(num_gpus):
        start_idx = i * chunk
        end_idx = start_idx + chunk if i < num_gpus - 1 else total
        t = threading.Thread(
            target=engine_worker,
            args=(engines[i], i, request_items[start_idx:end_idx], results[i], barrier),
        )
        threads.append(t)

    gen_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gen_elapsed = time.time() - gen_start

    result_queue.put(
        {
            "gen_elapsed": gen_elapsed,
            "engines": results,
        }
    )


def pcts(data):
    """Return (min, p50, p90, p99, max) of a list."""
    s = sorted(data)
    n = len(s)
    if n == 0:
        return (0, 0, 0, 0, 0)
    return (s[0], s[n // 2], s[int(n * 0.9)], s[min(int(n * 0.99), n - 1)], s[-1])


def steady_state(times):
    """Return middle portion: skip first 5% and last 10%."""
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
    overheads = [w - g for w, g in zip(wall_times, gpu_times)]

    ss_wall = steady_state(wall_times)
    ss_gpu = steady_state(gpu_times)
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
        "Threaded multi-GPU vLLM benchmark with CUDA event profiling.",
    )
    args = parser.parse_args()

    # --- Phase 1: Single GPU ---
    print("=" * 70)
    print("Phase 1: Single GPU baseline (cuda:0, in subprocess)")
    print("=" * 70)

    q1 = Queue()
    p1 = Process(target=run_single_gpu, args=(args.model, args.num_requests, q1))
    p1.start()
    p1.join(timeout=300)

    if q1.empty():
        print("Single GPU phase failed!")
        return
    single = q1.get()

    print(
        f"  {single['completed']} reqs, {single['elapsed']:.1f}s, "
        f"{single['completed'] / single['elapsed']:.1f} req/s"
    )
    print_profile(
        "Single GPU cuda:0",
        single["wall_times"],
        single["gpu_times"],
        single["batch_sizes"],
    )

    # --- Phase 2: Threaded dual GPU ---
    print(f"\n{'=' * 70}")
    print("Phase 2: Threaded dual GPU (in subprocess)")
    print("=" * 70)

    q2 = Queue()
    p2 = Process(
        target=run_threaded_dual,
        args=(args.model, args.num_requests, args.num_gpus, q2),
    )
    p2.start()
    p2.join(timeout=600)

    if q2.empty():
        print("Threaded dual phase failed!")
        return
    dual = q2.get()

    total_completed = sum(e["completed"] for e in dual["engines"])
    print(
        f"  {total_completed} reqs, {dual['gen_elapsed']:.1f}s, "
        f"{total_completed / dual['gen_elapsed']:.1f} req/s"
    )

    for i, e in enumerate(dual["engines"]):
        print_profile(
            f"Threaded cuda:{i}", e["wall_times"], e["gpu_times"], e["batch_sizes"]
        )

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("Comparison: Single vs Threaded (cuda:0, steady-state p50)")
    print("=" * 70)

    ss_sw = steady_state(single["wall_times"])
    ss_sg = steady_state(single["gpu_times"])
    # Use cuda:0 from dual for comparison.
    e0 = dual["engines"][0]
    ss_dw = steady_state(e0["wall_times"])
    ss_dg = steady_state(e0["gpu_times"])

    if ss_sw and ss_dw:
        sw_p50 = sorted(ss_sw)[len(ss_sw) // 2]
        sg_p50 = sorted(ss_sg)[len(ss_sg) // 2]
        dw_p50 = sorted(ss_dw)[len(ss_dw) // 2]
        dg_p50 = sorted(ss_dg)[len(ss_dg) // 2]

        print(
            f"  p50 wall time:  single={sw_p50:.1f}ms  "
            f"threaded={dw_p50:.1f}ms  "
            f"ratio={dw_p50 / sw_p50:.2f}x"
        )
        print(
            f"  p50 GPU time:   single={sg_p50:.1f}ms  "
            f"threaded={dg_p50:.1f}ms  "
            f"ratio={dg_p50 / sg_p50:.2f}x"
        )
        so = sw_p50 - sg_p50
        do = dw_p50 - dg_p50
        print(
            f"  p50 overhead:   single={so:.1f}ms  "
            f"threaded={do:.1f}ms  "
            f"ratio={do / max(0.01, so):.2f}x"
        )

        print()
        if dg_p50 / sg_p50 > 1.05:
            print(
                f"  GPU kernels are {dg_p50 / sg_p50:.0%} slower → "
                f"CUDA-level contention (driver lock, streams, etc.)"
            )
        elif dw_p50 / sw_p50 > 1.05:
            print(
                f"  GPU time is similar but wall time is "
                f"{dw_p50 / sw_p50:.0%} higher →\n"
                f"  CPU/Python overhead in vLLM between kernel launches"
            )
        else:
            print("  No significant per-step slowdown detected")


if __name__ == "__main__":
    main()
