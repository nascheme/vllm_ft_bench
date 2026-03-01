# Threaded multi-GPU vLLM with a dedicated step-loop thread per GPU.
#
# Architecture per GPU:
#
#   Thread A (engine_worker): add requests → run output loop → collect results
#   Thread B (step thread):   tight step_fn() loop → push outputs to queue
#
# Thread B owns the EngineCore's scheduler and GPU exclusively.
# Thread A processes outputs (CPU) concurrently with Thread B's next step (GPU).
# Aborts cross from Thread A to Thread B via EngineCore.aborts_queue, which is
# already a thread-safe queue.Queue drained at the top of each step().
#
# All requests are added by Thread A before Thread B starts, so add_request()
# and step() never run concurrently — no scheduler locking needed.
#
# Key measurement: Thread B records the inter-step gap (time from
# output_queue.put() returning to the start of the next step_fn() call).
# Compare this with the "other (gap)" column in threaded_step_breakdown.py:
# if the gap shrinks significantly, process_outputs() was a meaningful
# contributor to the inter-step dead time in the single-thread design.

import queue
import time

from vllm_ft.util import make_arg_parser


# ---------------------------------------------------------------------------
# Thread B: dedicated step loop
# ---------------------------------------------------------------------------


def step_thread_fn(raw_core, device_index, output_queue, step_stats):
    """Dedicated step-loop thread (Thread B).

    Owns the EngineCore exclusively: calls step_fn() + post_step() in a tight
    loop, pushing each EngineCoreOutputs onto output_queue.  Signals Thread A
    to stop by pushing None as a sentinel after the last step.

    Measures the inter-step gap: the elapsed time from when output_queue.put()
    returns (Thread B has handed off outputs to Thread A) to when the next
    step_fn() call begins.  In the single-thread design this gap included
    process_outputs() time (~0.2ms/step).  Here it should collapse to the cost
    of checking scheduler.has_requests() plus queue bookkeeping.

    raw_core is the EngineCore object (engine.engine_core.engine_core).
    """
    import torch
    from vllm.v1.engine import EngineCoreOutputs

    torch.cuda.set_device(device_index)

    step_count = 0
    empty_steps = 0
    gap_count = 0
    total_gap_ms = 0.0
    min_gap_ms = float("inf")
    max_gap_ms = 0.0

    t_after_put = None  # timestamp after the previous output_queue.put()

    while raw_core.scheduler.has_requests():
        # Measure gap: time from previous put() to now (top of step loop).
        t_loop_top = time.perf_counter()
        if t_after_put is not None:
            gap_ms = (t_loop_top - t_after_put) * 1000
            total_gap_ms += gap_ms
            gap_count += 1
            if gap_ms < min_gap_ms:
                min_gap_ms = gap_ms
            if gap_ms > max_gap_ms:
                max_gap_ms = gap_ms

        # Run one engine step (schedule → GPU execute → update_from_output).
        outputs_dict, model_executed = raw_core.step_fn()
        raw_core.post_step(model_executed)
        step_count += 1

        # Extract this engine's outputs (index 0; no data parallelism).
        core_out = (outputs_dict and outputs_dict.get(0)) or EngineCoreOutputs()

        if not core_out.outputs:
            empty_steps += 1

        # Hand off to Thread A and record the moment we're done.
        output_queue.put(core_out)
        t_after_put = time.perf_counter()

    # Sentinel: tells Thread A there are no more outputs.
    output_queue.put(None)

    step_stats["steps"] = step_count
    step_stats["empty_steps"] = empty_steps
    step_stats["gap_count"] = gap_count
    step_stats["total_gap_ms"] = total_gap_ms
    step_stats["min_gap_ms"] = min_gap_ms if gap_count > 0 else 0.0
    step_stats["max_gap_ms"] = max_gap_ms


# ---------------------------------------------------------------------------
# Thread A: output processor (also adds requests and drives the barrier)
# ---------------------------------------------------------------------------


def engine_worker(engine, device_index, request_items, result, barrier):
    """Per-GPU worker (Thread A).

    1. Adds all requests to the engine (tokenization already done).
    2. Waits at the barrier so all GPUs start their step threads together.
    3. Spawns Thread B (the step thread).
    4. Runs the output processing loop until Thread B signals done.
    5. Collects throughput statistics into result.
    """
    import threading

    import torch

    torch.cuda.set_device(device_index)

    input_processor = engine.input_processor
    output_processor = engine.output_processor

    # --- Phase 1: add all requests ---
    for i, (req, sp) in enumerate(request_items):
        ecr = input_processor.process_inputs(
            f"gpu{device_index}_{i}",
            req.prompt,
            sp,
            arrival_time=time.time(),
            supported_tasks=engine.get_supported_tasks(),
        )
        engine.add_request(ecr.request_id, ecr, sp, prompt_text=req.prompt)

    # Grab the raw EngineCore (InprocClient wraps it).
    raw_core = engine.engine_core.engine_core

    # --- Phase 2: synchronise start across all GPUs ---
    barrier.wait()

    # --- Phase 3: launch Thread B ---
    output_queue = queue.Queue()
    step_stats = {}
    step_thread = threading.Thread(
        target=step_thread_fn,
        args=(raw_core, device_index, output_queue, step_stats),
        name=f"StepLoop::cuda{device_index}",
        daemon=True,
    )

    t0 = time.time()
    step_thread.start()

    # --- Phase 4: output processing loop ---
    finished = {}
    total_process_ms = 0.0

    while True:
        item = output_queue.get()

        # None is the sentinel from Thread B: no more outputs.
        if item is None:
            break

        if not item.outputs:
            continue

        t_proc_start = time.perf_counter()
        processed = output_processor.process_outputs(
            item.outputs,
            engine_core_timestamp=item.timestamp,
            iteration_stats=None,
        )
        output_processor.update_scheduler_stats(item.scheduler_stats)
        total_process_ms += (time.perf_counter() - t_proc_start) * 1000

        # Post stop-string aborts back to Thread B via the thread-safe
        # aborts_queue that EngineCore already drains at each step start.
        if processed.reqs_to_abort:
            raw_core.aborts_queue.put(processed.reqs_to_abort)

        for ro in processed.request_outputs:
            if ro.finished:
                finished[ro.request_id] = ro

    t1 = time.time()
    step_thread.join()

    # --- Phase 5: collect results ---
    prompt_tokens = 0
    output_tokens = 0
    for ro in finished.values():
        if ro.prompt_token_ids:
            prompt_tokens += len(ro.prompt_token_ids)
        output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)

    result["completed"] = len(finished)
    result["prompt_tokens"] = prompt_tokens
    result["output_tokens"] = output_tokens
    result["elapsed"] = t1 - t0
    result["process_ms"] = total_process_ms
    result["step_stats"] = step_stats


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_threaded(args):
    """All vllm imports happen here — called directly or from a subprocess."""
    import os
    import threading

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import EngineArgs
    from vllm.tokenizers import get_tokenizer
    from vllm.usage.usage_lib import UsageContext

    from vllm_ft.util import (
        apply_forward_context_monkey_patch,
        build_request_items,
        create_engine,
        print_throughput_results,
    )

    apply_forward_context_monkey_patch()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    engines = []
    for i in range(args.num_gpus):
        use_graphs = getattr(args, "cuda_graphs", False)
        print(f"Creating engine on cuda:{i} (cuda_graphs={use_graphs}) ...")
        e = create_engine(
            engine_args, i, UsageContext.LLM_CLASS, cuda_graphs=use_graphs
        )
        core = e.engine_core.engine_core
        print(
            f"  async_scheduling={core.async_scheduling}, "
            f"batch_queue={'ON' if core.batch_queue is not None else 'OFF'}, "
            f"batch_queue_size={core.batch_queue_size}"
        )
        engines.append(e)
    print(f"All {args.num_gpus} engines created.")

    total = len(request_items)
    chunk = total // args.num_gpus

    barrier = threading.Barrier(args.num_gpus)
    results = [{} for _ in range(args.num_gpus)]
    threads = []

    for i in range(args.num_gpus):
        start_idx = i * chunk
        end_idx = start_idx + chunk if i < args.num_gpus - 1 else total
        my_items = request_items[start_idx:end_idx]

        t = threading.Thread(
            target=engine_worker,
            args=(engines[i], i, my_items, results[i], barrier),
            name=f"LLM::engine{i}",
        )
        threads.append(t)

    gen_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gen_elapsed = time.time() - gen_start

    engine_stats = [
        {
            "completed": r["completed"],
            "prompt_tokens": r["prompt_tokens"],
            "output_tokens": r["output_tokens"],
        }
        for r in results
    ]
    print_throughput_results(gen_elapsed, engine_stats)

    print(f"\nInference time: {gen_elapsed:.1f}s")
    for i, r in enumerate(results):
        ss = r.get("step_stats", {})
        steps = ss.get("steps", 0)
        empty = ss.get("empty_steps", 0)
        gap_n = ss.get("gap_count", 0)
        total_gap = ss.get("total_gap_ms", 0.0)
        avg_gap = total_gap / gap_n if gap_n > 0 else 0.0
        min_gap = ss.get("min_gap_ms", 0.0)
        max_gap = ss.get("max_gap_ms", 0.0)
        nonempty = steps - empty

        print(
            f"  cuda:{i}: {r['completed']} reqs, {r['elapsed']:.1f}s, "
            f"{steps} steps ({empty} empty), "
            f"process_outputs: {r['process_ms']:.0f}ms total "
            f"({r['process_ms'] / max(1, nonempty):.2f}ms/step)"
        )
        if gap_n > 0:
            print(
                f"    inter-step gap (Thread B idle between put→next step_fn): "
                f"avg={avg_gap:.3f}ms  min={min_gap:.3f}ms  max={max_gap:.3f}ms  "
                f"total={total_gap:.0f}ms"
            )


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def _clean_subprocess_entry(args_ns_dict):
    """Entry point for clean subprocess — imports vllm fresh."""
    import argparse

    args = argparse.Namespace(**args_ns_dict)
    run_threaded(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = make_arg_parser(
        "Threaded multi-GPU vLLM: dedicated step-loop thread per GPU.",
    )
    parser.add_argument(
        "--in-subprocess",
        action="store_true",
        help="Run in a clean subprocess (no inherited vllm state)",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Enable FULL CUDA graph capture (reduces driver lock contention)",
    )
    args = parser.parse_args()

    if args.in_subprocess:
        from multiprocessing import Process

        ns = {k: v for k, v in vars(args).items() if k != "in_subprocess"}
        p = Process(target=_clean_subprocess_entry, args=(ns,))
        p.start()
        p.join()
    else:
        run_threaded(args)
