# Static-partition dual-engine multi-GPU threaded vLLM throughput benchmark.
#
# Pre-partitions requests evenly across engines (round-robin) instead of
# using a shared queue.  Includes instrumentation to diagnose throughput
# gaps vs dp_generate.py.
#
# Two modes (--preload):
#   default:   tokenizer thread feeds per-engine queues (trickle)
#   --preload: tokenize and add ALL requests before starting engine steps
#              (matches LLM.generate() behavior in dp_generate.py)

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import queue
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
    print_throughput_results,
)

apply_forward_context_monkey_patch()

# ---------------------------------------------------------------------------

MAX_PULL_PER_STEP = 16


class StepTimings:
    """Per-engine step timing records."""

    def __init__(self, device_index):
        self.device_index = device_index
        self.step_records = []  # (wall_ms, num_running, num_finished_this_step)
        self.step_count = 0

    def record_step(self, wall_ms, num_running, num_finished):
        self.step_records.append((wall_ms, num_running, num_finished))
        self.step_count += 1

    def summary(self):
        if not self.step_records:
            return "  (no steps)"
        wall_times = [r[0] for r in self.step_records]
        batch_sizes = [r[1] for r in self.step_records]
        n = len(wall_times)
        wall_times.sort()
        batch_sizes_sorted = sorted(batch_sizes)
        return (
            f"  steps: {n}\n"
            f"  step time (ms):  min={wall_times[0]:.2f}  "
            f"p50={wall_times[n // 2]:.2f}  "
            f"p90={wall_times[int(n * 0.9)]:.2f}  "
            f"p99={wall_times[int(n * 0.99)]:.2f}  "
            f"max={wall_times[-1]:.2f}\n"
            f"  batch size:      min={batch_sizes_sorted[0]}  "
            f"p50={batch_sizes_sorted[n // 2]}  "
            f"p90={batch_sizes_sorted[int(n * 0.9)]}  "
            f"max={batch_sizes_sorted[-1]}\n"
            f"  total step time: {sum(r[0] for r in self.step_records):.0f}ms"
        )

    def ramp_up_summary(self):
        """Show the first 20 steps to diagnose batch ramp-up."""
        if not self.step_records:
            return "  (no steps)"
        lines = ["  step | wall_ms | batch_size | finished"]
        for i, (wall_ms, batch_size, finished) in enumerate(self.step_records[:20]):
            lines.append(
                f"  {i:4d} | {wall_ms:7.2f} | {batch_size:10d} | {finished:8d}"
            )
        return "\n".join(lines)


def tokenizer_worker(
    input_processor,
    supported_tasks,
    request_items,
    per_engine_queues,
    done_event,
):
    num_engines = len(per_engine_queues)
    for i, (req, sp) in enumerate(request_items):
        ecr = input_processor.process_inputs(
            str(i),
            req.prompt,
            sp,
            arrival_time=time.time(),
            supported_tasks=supported_tasks,
        )
        target = i % num_engines
        per_engine_queues[target].put((ecr, req.prompt, sp))
    done_event.set()


def engine_worker(
    engine,
    device_index,
    my_queue,
    tok_done,
    stats,
    timings,
    start_barrier,
):
    torch.cuda.set_device(device_index)
    core = engine.engine_core.engine_core

    # Wait for all threads to be ready (important for --preload mode).
    start_barrier.wait()

    while True:
        for _ in range(MAX_PULL_PER_STEP):
            try:
                ecr, prompt_text, sp = my_queue.get_nowait()
                engine.add_request(
                    ecr.request_id,
                    ecr,
                    sp,
                    prompt_text=prompt_text,
                )
            except queue.Empty:
                break

        if engine.has_unfinished_requests():
            num_running, num_waiting = core.scheduler.get_request_counts()

            t0 = time.perf_counter()
            request_outputs = engine.step()
            wall_ms = (time.perf_counter() - t0) * 1000

            finished_count = 0
            for output in request_outputs:
                if output.finished:
                    finished_count += 1
                    stats[0] += 1
                    if output.prompt_token_ids:
                        stats[1] += len(output.prompt_token_ids)
                    stats[2] += sum(len(o.token_ids) for o in output.outputs if o)

            timings.record_step(wall_ms, num_running + num_waiting, finished_count)
        elif tok_done.is_set() and my_queue.empty():
            break
        else:
            try:
                ecr, prompt_text, sp = my_queue.get(timeout=0.5)
                engine.add_request(
                    ecr.request_id,
                    ecr,
                    sp,
                    prompt_text=prompt_text,
                )
            except queue.Empty:
                if tok_done.is_set():
                    break


def main():
    parser = make_arg_parser(
        "Static-partition dual-engine threaded vLLM throughput benchmark.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Tokenize and add ALL requests before starting engine steps "
        "(matches LLM.generate() batch behavior).",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Enable CUDA graph capture (CUDAGraphMode.FULL) in each engine.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)
    num_requests = len(request_items)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    engines = []
    for i in range(args.num_gpus):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(
            create_engine(
                engine_args, i, UsageContext.LLM_CLASS, cuda_graphs=args.cuda_graphs
            )
        )
    print(f"All {args.num_gpus} engines created.")

    per_engine_queues = [
        queue.Queue(maxsize=num_requests) for _ in range(args.num_gpus)
    ]
    tok_done = threading.Event()

    if args.preload:
        # Tokenize all requests and add them directly to engines before
        # starting the step loop.  This matches LLM.generate() behavior.
        print("Preloading: tokenizing and adding all requests to engines ...")
        input_processor = engines[0].input_processor
        supported_tasks = engines[0].get_supported_tasks()
        for i, (req, sp) in enumerate(request_items):
            ecr = input_processor.process_inputs(
                str(i),
                req.prompt,
                sp,
                arrival_time=time.time(),
                supported_tasks=supported_tasks,
            )
            target = i % args.num_gpus
            engines[target].add_request(
                ecr.request_id,
                ecr,
                sp,
                prompt_text=req.prompt,
            )
        tok_done.set()
        print(
            f"Preloaded {num_requests} requests "
            f"({num_requests // args.num_gpus} per engine)."
        )
    else:
        tok_thread = threading.Thread(
            target=tokenizer_worker,
            args=(
                engines[0].input_processor,
                engines[0].get_supported_tasks(),
                request_items,
                per_engine_queues,
                tok_done,
            ),
            name="LLM::tok",
        )
        tok_thread.start()

    all_timings = [StepTimings(i) for i in range(args.num_gpus)]
    stats = [[0, 0, 0] for _ in range(args.num_gpus)]
    start_barrier = threading.Barrier(args.num_gpus)

    engine_threads = []
    for i, engine in enumerate(engines):
        t = threading.Thread(
            target=engine_worker,
            args=(
                engine,
                i,
                per_engine_queues[i],
                tok_done,
                stats[i],
                all_timings[i],
                start_barrier,
            ),
            name=f"LLM::engine{i}",
        )
        t.start()
        engine_threads.append(t)

    start_time = time.time()
    while any(t.is_alive() for t in engine_threads):
        time.sleep(2)
        elapsed = time.time() - start_time
        total = sum(s[0] for s in stats)
        print(f"  [{elapsed:.1f}s] {total}/{num_requests} completed ...")

    if not args.preload:
        tok_thread.join()
    for t in engine_threads:
        t.join()

    elapsed = time.time() - start_time

    # --- Results ---

    engine_stats = [
        {"completed": s[0], "prompt_tokens": s[1], "output_tokens": s[2]} for s in stats
    ]
    print_throughput_results(elapsed, engine_stats)

    # Per-engine step profile.
    print(f"\n{'=' * 70}")
    print(f"Per-engine step profile ({'preload' if args.preload else 'trickle'} mode)")
    print(f"{'=' * 70}")
    for t in all_timings:
        print(f"\ncuda:{t.device_index}:")
        print(t.summary())

    # Ramp-up: first 20 steps.
    print(f"\n{'=' * 70}")
    print("Batch ramp-up (first 20 steps per engine)")
    print(f"{'=' * 70}")
    for t in all_timings:
        print(f"\ncuda:{t.device_index}:")
        print(t.ramp_up_summary())

    # Tail: last 20 steps.
    print(f"\n{'=' * 70}")
    print("Tail drain (last 20 steps per engine)")
    print(f"{'=' * 70}")
    for t in all_timings:
        print(f"\ncuda:{t.device_index}:")
        recs = t.step_records[-20:]
        lines = ["  step | wall_ms | batch_size | finished"]
        offset = max(0, len(t.step_records) - 20)
        for i, (wall_ms, batch_size, finished) in enumerate(recs):
            lines.append(
                f"  {offset + i:4d} | {wall_ms:7.2f} | {batch_size:10d} | {finished:8d}"
            )
        print("\n".join(lines))


if __name__ == "__main__":
    main()
