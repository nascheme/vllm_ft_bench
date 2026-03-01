# Dual-engine multi-GPU threaded vLLM throughput benchmark.
#
# Architecture: two independent LLMEngine instances (one per GPU) fed from
# a shared tokenized-request queue, with a single tokenizer thread.
#
#   Tokenizer Thread (CPU)         Engine Thread 0 (cuda:0)   Engine Thread 1 (cuda:1)
#     input_processor.process()    add_request (from queue)   add_request (from queue)
#     tokenized_queue.put(ecr)     engine0.step()             engine1.step()
#                                  (continuous streaming)      (continuous streaming)
#
# Each engine thread continuously pulls requests from the shared queue and
# steps the engine.  The engine's internal scheduler handles batching.
# The shared queue naturally load-balances between engine threads because
# each step() releases the GIL for GPU work, giving the other thread
# time to pull from the queue.
#
# For TP=1/PP=1, parallel-state init is idempotent so both engines
# coexist in one process with zero vLLM source modifications.
#
# Uses vllm.benchmarks.datasets.RandomDataset for benchmark-comparable
# prompts, and reports tokens/sec metrics matching `vllm bench throughput`.

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import queue
import sys
import threading
import time

import torch

from vllm import EngineArgs
from vllm.tokenizers import get_tokenizer
from vllm.usage.usage_lib import UsageContext

from vllm_ft.util import (
    apply_forward_context_monkey_patch,
    build_request_items,
    create_engine,
    make_arg_parser,
    print_prompt_length_histogram,
    print_throughput_results,
    render_request,
)

apply_forward_context_monkey_patch()

# ---------------------------------------------------------------------------


def get_engine_stats(engine):
    """Read real-time stats from a V1 LLMEngine (InprocClient path).

    Access path:
        engine.engine_core          -> InprocClient
        engine.engine_core.engine_core -> EngineCore
        .scheduler                  -> Scheduler
        .kv_cache_manager.usage     -> float 0.0-1.0
        .get_request_counts()       -> (running, waiting)
    """
    try:
        scheduler = engine.engine_core.engine_core.scheduler
        num_running, num_waiting = scheduler.get_request_counts()
        return {
            "usage": scheduler.kv_cache_manager.usage,
            "running": num_running,
            "waiting": num_waiting,
        }
    except Exception as e:
        return {"usage": -0.1, "running": -1, "waiting": -1, "error": str(e)}


def status_reporter_worker(engines, engine_stats, num_requests, stop_event, start_time):
    """Background thread to print a live status line."""
    prev_output_tokens = 0
    prev_time = start_time
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        total_done = sum(s[0] for s in engine_stats)
        total_output_toks = sum(s[2] for s in engine_stats)

        # Compute output tok/s over the last reporting interval.
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            out_tps = (total_output_toks - prev_output_tokens) / dt
        else:
            out_tps = 0.0
        prev_output_tokens = total_output_toks
        prev_time = now

        reports = []
        for i, engine in enumerate(engines):
            s = get_engine_stats(engine)
            reports.append(
                f"GPU{i}: {s['usage'] * 100:4.1f}% KV |"
                f" {s['running']:3d} Run | {s['waiting']:3d} Wait"
            )

        status_line = " | ".join(reports)
        sys.stdout.write(
            f"\r[{elapsed:5.1f}s] {total_done:4d}/{num_requests} done"
            f" | {out_tps:6.0f} out tok/s | {status_line}   "
        )
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write("\n")


DISPATCH_QUEUE_HIGH_WATER = 16  # Back off when an engine queue exceeds this.
DISPATCH_BACKPRESSURE_SLEEP = 0.1  # Seconds to sleep when back-pressure fires.


def smart_dispatcher_worker(renderer, request_items, request_queues, engines):
    """Routes requests: round-robin when idle, lowest-KV-usage when busy."""
    rr_idx = 0
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)

        while True:
            stats = [get_engine_stats(e) for e in engines]
            usages = [s["usage"] for s in stats]

            if all(u < 0.10 for u in usages):
                # Both engines nearly empty — round-robin for even distribution.
                target_idx = rr_idx
                rr_idx = (rr_idx + 1) % len(engines)
            else:
                # Send to whichever engine has more KV headroom.
                target_idx = int(usages[1] < usages[0])

            if usages[target_idx] < 0.2:
                max_queue = 16
            else:
                max_queue = 2

            # Back-pressure: don't flood the queues faster than engines consume.
            if (
                request_queues[target_idx].qsize() > max_queue
                or usages[target_idx] > 0.9
            ):
                time.sleep(DISPATCH_BACKPRESSURE_SLEEP)
                print(
                    "waiting",
                    target_idx,
                    request_queues[target_idx].qsize(),
                    usages[target_idx],
                )
            else:
                print(
                    "put",
                    target_idx,
                    request_queues[target_idx].qsize(),
                    usages[target_idx],
                )
                request_queues[target_idx].put((str(i), proc_input, sp))
                break

    # Signal each engine worker to shut down.
    for q in request_queues:
        q.put(None)


def engine_worker(engine, device_index, my_queue, stats):
    """Pull requests from a dedicated queue and step the engine."""
    torch.cuda.set_device(device_index)

    done = False
    while not done:
        # When the engine is idle, block until a request arrives.
        # When the engine has in-flight work, just grab whatever is
        # available and fall through to step() immediately — we must
        # keep calling step() so requests complete and KV cache frees up.
        if engine.has_unfinished_requests():
            get = my_queue.get_nowait
        else:
            get = my_queue.get

        while True:
            try:
                item = get()
            except queue.Empty:
                break
            if item is None:
                done = True
                break
            req_id, proc_input, sp = item
            engine.add_request(req_id, proc_input, sp)
            # After the first item, always non-blocking for the rest.
            get = my_queue.get_nowait

        # Step the engine and collect finished outputs.
        if engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    stats[0] += 1
                    if output.prompt_token_ids:
                        stats[1] += len(output.prompt_token_ids)
                    stats[2] += sum(len(o.token_ids) for o in output.outputs if o)

    # Drain remaining in-flight requests after sentinel.
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                stats[0] += 1
                if output.prompt_token_ids:
                    stats[1] += len(output.prompt_token_ids)
                stats[2] += sum(len(o.token_ids) for o in output.outputs if o)


def main():
    parser = make_arg_parser(
        "Dual-engine multi-GPU threaded vLLM throughput benchmark.",
    )
    args = parser.parse_args()

    # 1. Generate dataset (CPU, before engine creation).
    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        async_scheduling=False,
    )

    # 2. Create engines sequentially (parallel state init is idempotent
    #    for TP=1, PP=1 — second engine reuses the already-initialized state).
    engines = []
    for i in range(args.num_gpus):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))
    print(f"All {args.num_gpus} engines created.")

    # Two separate queues for steered delivery
    request_queues = [queue.Queue() for _ in range(args.num_gpus)]
    stop_reporter = threading.Event()

    # Smart Dispatcher Thread
    dispatch_thread = threading.Thread(
        target=smart_dispatcher_worker,
        args=(
            engines[0].renderer,
            request_items,
            request_queues,
            engines,
        ),
        name="LLM::dispatcher",
    )
    dispatch_thread.start()

    # Engine Threads
    stats = [[0, 0, 0] for _ in range(args.num_gpus)]
    engine_threads = []
    for i, engine in enumerate(engines):
        t = threading.Thread(
            target=engine_worker,
            args=(engine, i, request_queues[i], stats[i]),
            name=f"LLM::engine{i}",
        )
        t.start()
        engine_threads.append(t)

    # 5. Wait for completion with live status line.
    start_time = time.time()

    reporter = threading.Thread(
        target=status_reporter_worker,
        args=(engines, stats, len(request_items), stop_reporter, start_time),
    )
    reporter.start()

    for t in engine_threads:
        t.join()

    stop_reporter.set()

    # 6. Report results (matches `vllm bench throughput` format).
    end_time = time.time()
    elapsed = end_time - start_time

    engine_stats = [
        {"completed": s[0], "prompt_tokens": s[1], "output_tokens": s[2]} for s in stats
    ]
    print_throughput_results(elapsed, engine_stats)

    requests_only = [req for req, _ in request_items]
    print_prompt_length_histogram(requests_only)

    dispatch_thread.join()
    reporter.join()


if __name__ == "__main__":
    main()
