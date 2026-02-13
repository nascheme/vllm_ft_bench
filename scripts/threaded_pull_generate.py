# Dual-engine multi-GPU threaded vLLM throughput benchmark — pull design.
#
# Architecture: two independent LLMEngine instances (one per GPU), each with
# its own engine thread that *pulls* from a shared tokenized-request queue.
# A single tokenizer thread pre-tokenizes all requests into the shared queue.
#
#   Tokenizer Thread (CPU)         Engine Thread 0 (cuda:0)   Engine Thread 1 (cuda:1)
#     input_processor.process()    pull from shared queue     pull from shared queue
#     shared_queue.put(ecr)        add_request (throttled)    add_request (throttled)
#                                  engine0.step()             engine1.step()
#
# Each engine thread self-throttles based on two signals:
#   1. in_flight count — requests added but not yet finished
#   2. KV cache usage  — checked periodically from the scheduler
#
# When either signal is high the engine thread stops pulling and just
# calls step() to drain work.  When both are low it pulls more.
# The shared queue naturally load-balances: whichever engine finishes
# work first pulls the next item.
#
# For TP=1/PP=1, parallel-state init is idempotent so both engines
# coexist in one process with zero vLLM source modifications.

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
)

apply_forward_context_monkey_patch()

# ---------------------------------------------------------------------------
# Throttling knobs
# ---------------------------------------------------------------------------

MAX_IN_FLIGHT = 256  # Stop pulling when this many requests are in the engine.
KV_USAGE_HIGH = 0.9  # Stop pulling when KV cache usage exceeds this.
KV_CHECK_INTERVAL = 0.25  # Seconds between KV usage polls (not every step).
MAX_PULL_PER_STEP = 16  # Max requests to pull from queue per step cycle.


def get_engine_stats(engine):
    """Read real-time stats from a V1 LLMEngine (InprocClient path)."""
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


def tokenizer_worker(
    input_processor, supported_tasks, request_items, shared_queue, done_event
):
    """Pre-tokenize all requests into the shared queue."""
    for i, (req, sp) in enumerate(request_items):
        ecr = input_processor.process_inputs(
            str(i),
            req.prompt,
            sp,
            arrival_time=time.time(),
            supported_tasks=supported_tasks,
        )
        shared_queue.put((ecr, req.prompt, sp))
    done_event.set()


def engine_worker(engine, device_index, shared_queue, tok_done, stats):
    """Pull-based engine thread: self-throttles based on load.

    The engine thread owns the decision of when to pull new work.  It
    tracks in_flight (added minus finished) and periodically samples
    KV cache usage.  When either is high it stops pulling and just
    steps the engine to drain completions.
    """
    torch.cuda.set_device(device_index)

    in_flight = 0
    last_kv_check = 0.0
    cached_kv_usage = 0.0

    while True:
        # --- Decide whether we can accept more work ---
        now = time.time()
        if now - last_kv_check >= KV_CHECK_INTERVAL:
            s = get_engine_stats(engine)
            cached_kv_usage = s["usage"]
            last_kv_check = now

        can_pull = (in_flight < MAX_IN_FLIGHT) and (cached_kv_usage < KV_USAGE_HIGH)

        # --- Pull phase ---
        pulled = 0
        if can_pull:
            for _ in range(MAX_PULL_PER_STEP):
                # If engine is idle, block on the first pull to avoid busy-wait.
                if not engine.has_unfinished_requests() and pulled == 0:
                    try:
                        item = shared_queue.get(timeout=0.5)
                    except queue.Empty:
                        if tok_done.is_set() and shared_queue.empty():
                            break
                        continue
                else:
                    try:
                        item = shared_queue.get_nowait()
                    except queue.Empty:
                        break

                if item is None:
                    # Shouldn't happen — we use tok_done instead of sentinels,
                    # but handle gracefully.
                    break

                ecr, prompt_text, sp = item
                engine.add_request(ecr.request_id, ecr, sp, prompt_text=prompt_text)
                in_flight += 1
                pulled += 1

        # --- Step phase ---
        if engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    in_flight -= 1
                    stats[0] += 1
                    if output.prompt_token_ids:
                        stats[1] += len(output.prompt_token_ids)
                    stats[2] += sum(len(o.token_ids) for o in output.outputs if o)
        elif tok_done.is_set() and shared_queue.empty():
            break


def main():
    parser = make_arg_parser(
        "Dual-engine multi-GPU threaded vLLM throughput benchmark (pull design).",
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

    # 2. Create engines sequentially.
    engines = []
    for i in range(args.num_gpus):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))
    print(f"All {args.num_gpus} engines created.")

    # Shared queue — both engine threads pull from it.
    shared_queue = queue.Queue()
    tok_done = threading.Event()
    stop_reporter = threading.Event()

    # Tokenizer thread
    tok_thread = threading.Thread(
        target=tokenizer_worker,
        args=(
            engines[0].input_processor,
            engines[0].get_supported_tasks(),
            request_items,
            shared_queue,
            tok_done,
        ),
        name="LLM::tok",
    )
    tok_thread.start()

    # Engine threads
    stats = [[0, 0, 0] for _ in range(args.num_gpus)]
    engine_threads = []
    for i, engine in enumerate(engines):
        t = threading.Thread(
            target=engine_worker,
            args=(engine, i, shared_queue, tok_done, stats[i]),
            name=f"LLM::engine{i}",
        )
        t.start()
        engine_threads.append(t)

    # Status reporter
    start_time = time.time()
    reporter = threading.Thread(
        target=status_reporter_worker,
        args=(engines, stats, len(request_items), stop_reporter, start_time),
    )
    reporter.start()

    for t in engine_threads:
        t.join()

    stop_reporter.set()

    # Report results.
    elapsed = time.time() - start_time

    engine_stats = [
        {"completed": s[0], "prompt_tokens": s[1], "output_tokens": s[2]} for s in stats
    ]
    print_throughput_results(elapsed, engine_stats)

    requests_only = [req for req, _ in request_items]
    print_prompt_length_histogram(requests_only)

    tok_thread.join()
    reporter.join()


if __name__ == "__main__":
    main()
