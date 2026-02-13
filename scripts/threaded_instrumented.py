# Instrumented dual-engine multi-GPU threaded vLLM throughput benchmark.
#
# Same architecture as threaded_generate.py, but captures per-step timing
# to diagnose throughput gaps vs multi-process (dp_generate.py).
#
# Key diagnostics:
#   1. Per-phase breakdown (schedule, execute, update, etc.)
#   2. Per-step wall time with overlap detection (are both engines
#      actually running in parallel, or serializing on CUDA driver locks?)
#   3. Tail drain analysis (how long the last N requests take)
#   4. Step time distribution (percentiles)

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import queue
import statistics
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

MAX_PULL_PER_STEP = 8


class StepTimings:
    """Accumulates per-phase timing and per-step records for one engine."""

    def __init__(self, device_index):
        self.device_index = device_index

        # Aggregate phase timings.
        self.schedule_ns = 0
        self.execute_ns = 0
        self.sample_ns = 0
        self.update_ns = 0
        self.pull_ns = 0
        self.output_proc_ns = 0
        self.step_count = 0
        self.execute_count = 0

        # Per-step records: (wall_start, wall_end, execute_ms, num_running)
        # for overlap and tail analysis.
        self.step_records = []

        # Completion timeline: (wall_time, cumulative_completed)
        self.completions = []
        self._completed = 0

    def record_step(self, wall_start, wall_end, execute_ns, num_running):
        self.step_records.append(
            (
                wall_start,
                wall_end,
                execute_ns / 1e6,  # convert to ms
                num_running,
            )
        )

    def record_completions(self, wall_time, count):
        self._completed += count
        self.completions.append((wall_time, self._completed))

    def phase_summary(self, elapsed_s):
        n = max(self.step_count, 1)
        e = max(self.execute_count, 1)

        def ms(ns):
            return ns / 1e6

        def avg_ms(ns, count):
            return ns / 1e6 / count

        total_tracked = (
            self.schedule_ns
            + self.execute_ns
            + self.sample_ns
            + self.update_ns
            + self.pull_ns
            + self.output_proc_ns
        )
        return (
            f"  steps: {self.step_count}  (model executed: {self.execute_count})\n"
            f"  schedule:       {ms(self.schedule_ns):8.1f}ms total, "
            f"{avg_ms(self.schedule_ns, n):6.3f}ms/step\n"
            f"  execute_model:  {ms(self.execute_ns):8.1f}ms total, "
            f"{avg_ms(self.execute_ns, e):6.3f}ms/step\n"
            f"  sample_tokens:  {ms(self.sample_ns):8.1f}ms total, "
            f"{avg_ms(self.sample_ns, e):6.3f}ms/step\n"
            f"  update_output:  {ms(self.update_ns):8.1f}ms total, "
            f"{avg_ms(self.update_ns, n):6.3f}ms/step\n"
            f"  pull_from_queue:{ms(self.pull_ns):8.1f}ms total, "
            f"{avg_ms(self.pull_ns, n):6.3f}ms/step\n"
            f"  output_proc:    {ms(self.output_proc_ns):8.1f}ms total, "
            f"{avg_ms(self.output_proc_ns, n):6.3f}ms/step\n"
            f"  total tracked:  {ms(total_tracked):8.1f}ms "
            f"({ms(total_tracked) / (elapsed_s * 1000) * 100:.1f}% of wall time)\n"
            f"  schedule+update (CPU overhead): "
            f"{ms(self.schedule_ns + self.update_ns):8.1f}ms "
            f"({(self.schedule_ns + self.update_ns) / max(total_tracked, 1) * 100:.1f}%)"
        )

    def step_time_distribution(self):
        """Return percentile summary of per-step execute times."""
        exec_times = [r[2] for r in self.step_records if r[2] > 0]
        if not exec_times:
            return "  (no execute steps recorded)"
        exec_times.sort()
        n = len(exec_times)
        p50 = exec_times[n // 2]
        p90 = exec_times[int(n * 0.9)]
        p99 = exec_times[int(n * 0.99)]
        return (
            f"  execute_model per step (ms): "
            f"min={exec_times[0]:.2f}  p50={p50:.2f}  "
            f"p90={p90:.2f}  p99={p99:.2f}  max={exec_times[-1]:.2f}"
        )


def instrument_engine_core(engine, timings):
    """Monkey-patch EngineCore methods to record per-phase timings."""
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    executor = core.model_executor

    orig_schedule = scheduler.schedule
    orig_execute = executor.execute_model
    orig_sample = getattr(executor, "sample_tokens", None)
    orig_update = scheduler.update_from_output

    def timed_schedule():
        t0 = time.perf_counter_ns()
        result = orig_schedule()
        timings.schedule_ns += time.perf_counter_ns() - t0
        return result

    def timed_execute(scheduler_output, **kwargs):
        t0 = time.perf_counter_ns()
        result = orig_execute(scheduler_output, **kwargs)
        timings.execute_ns += time.perf_counter_ns() - t0
        return result

    def timed_update(scheduler_output, model_output):
        t0 = time.perf_counter_ns()
        result = orig_update(scheduler_output, model_output)
        timings.update_ns += time.perf_counter_ns() - t0
        return result

    scheduler.schedule = timed_schedule
    executor.execute_model = timed_execute
    scheduler.update_from_output = timed_update

    if orig_sample is not None:

        def timed_sample(*args, **kwargs):
            t0 = time.perf_counter_ns()
            result = orig_sample(*args, **kwargs)
            timings.sample_ns += time.perf_counter_ns() - t0
            return result

        executor.sample_tokens = timed_sample


def tokenizer_worker(
    input_processor, supported_tasks, request_items, tokenized_queue, done_event
):
    for i, (req, sp) in enumerate(request_items):
        ecr = input_processor.process_inputs(
            str(i),
            req.prompt,
            sp,
            arrival_time=time.time(),
            supported_tasks=supported_tasks,
        )
        tokenized_queue.put((ecr, req.prompt, sp))
    done_event.set()


def engine_worker(
    engine,
    device_index,
    tokenized_queue,
    tok_done,
    stats,
    timings,
    global_start_time,
):
    torch.cuda.set_device(device_index)

    # Access scheduler for request counts.
    core = engine.engine_core.engine_core
    scheduler = core.scheduler

    while True:
        # Pull requests from the shared queue.
        t0 = time.perf_counter_ns()
        for _ in range(MAX_PULL_PER_STEP):
            try:
                ecr, prompt_text, sp = tokenized_queue.get_nowait()
                engine.add_request(
                    ecr.request_id,
                    ecr,
                    sp,
                    prompt_text=prompt_text,
                )
            except queue.Empty:
                break
        timings.pull_ns += time.perf_counter_ns() - t0

        if engine.has_unfinished_requests():
            timings.step_count += 1

            # Get request count before step.
            num_running, num_waiting = scheduler.get_request_counts()

            wall_start = time.perf_counter()
            exec_t0 = time.perf_counter_ns()
            request_outputs = engine.step()
            exec_elapsed = time.perf_counter_ns() - exec_t0
            wall_end = time.perf_counter()

            if request_outputs:
                timings.execute_count += 1

            timings.record_step(
                wall_start,
                wall_end,
                exec_elapsed,
                num_running + num_waiting,
            )

            t0 = time.perf_counter_ns()
            finished_count = 0
            for output in request_outputs:
                if output.finished:
                    finished_count += 1
                    stats[0] += 1
                    if output.prompt_token_ids:
                        stats[1] += len(output.prompt_token_ids)
                    stats[2] += sum(len(o.token_ids) for o in output.outputs if o)
            timings.output_proc_ns += time.perf_counter_ns() - t0

            if finished_count > 0:
                timings.record_completions(
                    time.time() - global_start_time,
                    finished_count,
                )
        elif tok_done.is_set() and tokenized_queue.empty():
            break
        else:
            try:
                ecr, prompt_text, sp = tokenized_queue.get(timeout=0.5)
                engine.add_request(
                    ecr.request_id,
                    ecr,
                    sp,
                    prompt_text=prompt_text,
                )
            except queue.Empty:
                if tok_done.is_set():
                    break


def compute_overlap(all_timings):
    """Compute what fraction of time both engines had GPU work in-flight.

    Merges the step intervals from all engines and checks overlap.
    Returns (both_active_s, any_active_s, overlap_ratio).
    """
    if len(all_timings) < 2:
        return 0, 0, 0

    # Collect all interval edges as events: (time, +1 for start, -1 for end).
    events = []
    for t in all_timings:
        for wall_start, wall_end, _, _ in t.step_records:
            events.append((wall_start, 1))
            events.append((wall_end, -1))

    if not events:
        return 0, 0, 0

    events.sort()
    active = 0
    prev_time = events[0][0]
    both_active = 0.0
    any_active = 0.0

    for evt_time, delta in events:
        dt = evt_time - prev_time
        if active >= 2:
            both_active += dt
        if active >= 1:
            any_active += dt
        active += delta
        prev_time = evt_time

    ratio = both_active / any_active if any_active > 0 else 0
    return both_active, any_active, ratio


def print_tail_analysis(all_timings, elapsed_s, num_requests):
    """Show how the last 10% of requests drain."""
    # Merge completions from all engines, sorted by time.
    all_completions = []
    for t in all_timings:
        for wall_time, cum in t.completions:
            all_completions.append((wall_time, t.device_index, cum))
    all_completions.sort()

    if not all_completions:
        return

    # Compute global cumulative completions.
    per_engine_cum = {}
    global_timeline = []
    for wall_time, dev, cum in all_completions:
        per_engine_cum[dev] = cum
        total = sum(per_engine_cum.values())
        global_timeline.append((wall_time, total))

    # Find when we hit 90% and report the last 10% drain.
    threshold_90 = int(num_requests * 0.9)
    time_at_90 = None
    for wall_time, total in global_timeline:
        if total >= threshold_90 and time_at_90 is None:
            time_at_90 = wall_time

    if time_at_90 is None:
        return

    tail_time = elapsed_s - time_at_90
    tail_pct = tail_time / elapsed_s * 100

    print("\nTail drain analysis:")
    print(f"  90% complete at:  {time_at_90:.1f}s")
    print(f"  100% complete at: {elapsed_s:.1f}s")
    print(f"  Last 10% took:    {tail_time:.1f}s ({tail_pct:.1f}% of total time)")

    # Show per-engine activity in the tail.
    print(f"\n  Steps in tail (after {time_at_90:.1f}s):")
    for t in all_timings:
        tail_steps = [
            r
            for r in t.step_records
            if r[0]
            >= time_at_90 - (all_timings[0].step_records[0][0] if t.step_records else 0)
        ]
        # Use absolute perf_counter comparison — just count steps in last tail_time.
        if t.step_records:
            cutoff = t.step_records[-1][1] - tail_time
            tail_steps = [r for r in t.step_records if r[0] >= cutoff]
            tail_exec_ms = [r[2] for r in tail_steps if r[2] > 0]
            tail_running = [r[3] for r in tail_steps]
            if tail_exec_ms:
                print(
                    f"    cuda:{t.device_index}: {len(tail_steps)} steps, "
                    f"avg exec={statistics.mean(tail_exec_ms):.2f}ms, "
                    f"avg batch={statistics.mean(tail_running):.1f} reqs"
                )
            else:
                print(f"    cuda:{t.device_index}: {len(tail_steps)} steps (no exec)")


def main():
    parser = make_arg_parser(
        "Instrumented dual-engine multi-GPU threaded vLLM throughput benchmark.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)
    num_requests = len(request_items)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        async_scheduling=False,
    )
    engines = []
    for i in range(args.num_gpus):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))
    print(f"All {args.num_gpus} engines created.")

    # Instrument each engine's EngineCore.
    all_timings = [StepTimings(i) for i in range(args.num_gpus)]
    for engine, timings in zip(engines, all_timings):
        instrument_engine_core(engine, timings)

    # Start tokenizer thread.
    tokenized_queue = queue.Queue(maxsize=num_requests)
    tok_done = threading.Event()

    tok_thread = threading.Thread(
        target=tokenizer_worker,
        args=(
            engines[0].input_processor,
            engines[0].get_supported_tasks(),
            request_items,
            tokenized_queue,
            tok_done,
        ),
        name="LLM::tok",
    )
    tok_thread.start()

    # Start engine threads.
    stats = [[0, 0, 0] for _ in range(args.num_gpus)]
    engine_threads = []
    start_time = time.time()
    for i, engine in enumerate(engines):
        t = threading.Thread(
            target=engine_worker,
            args=(
                engine,
                i,
                tokenized_queue,
                tok_done,
                stats[i],
                all_timings[i],
                start_time,
            ),
            name=f"LLM::engine{i}",
        )
        t.start()
        engine_threads.append(t)

    while any(t.is_alive() for t in engine_threads):
        time.sleep(2)
        elapsed = time.time() - start_time
        total = sum(s[0] for s in stats)
        print(f"  [{elapsed:.1f}s] {total}/{num_requests} completed ...")

    tok_thread.join()
    for t in engine_threads:
        t.join()

    elapsed = time.time() - start_time

    # --- Results ---

    engine_stats = [
        {"completed": s[0], "prompt_tokens": s[1], "output_tokens": s[2]} for s in stats
    ]
    print_throughput_results(elapsed, engine_stats)

    # Per-engine phase breakdown.
    print(f"\n{'=' * 70}")
    print("Per-engine step timing breakdown")
    print(f"{'=' * 70}")
    for timings in all_timings:
        print(f"\ncuda:{timings.device_index}:")
        print(timings.phase_summary(elapsed))
        print(timings.step_time_distribution())

    # Aggregate CPU vs GPU.
    print(f"\n{'=' * 70}")
    print("Aggregate (all engines)")
    print(f"{'=' * 70}")
    total_schedule = sum(t.schedule_ns for t in all_timings)
    total_execute = sum(t.execute_ns for t in all_timings)
    total_sample = sum(t.sample_ns for t in all_timings)
    total_update = sum(t.update_ns for t in all_timings)
    total_cpu = total_schedule + total_update
    total_gpu = total_execute + total_sample
    total_all = total_cpu + total_gpu
    print(
        f"  CPU (schedule + update): {total_cpu / 1e6:8.1f}ms "
        f"({total_cpu / max(total_all, 1) * 100:.1f}%)"
    )
    print(
        f"  GPU (execute + sample):  {total_gpu / 1e6:8.1f}ms "
        f"({total_gpu / max(total_all, 1) * 100:.1f}%)"
    )

    # Overlap analysis.
    both_active, any_active, overlap_ratio = compute_overlap(all_timings)
    print(f"\n{'=' * 70}")
    print("GPU overlap analysis (are both engines truly parallel?)")
    print(f"{'=' * 70}")
    print(f"  Time both engines executing: {both_active:.2f}s")
    print(f"  Time any engine executing:   {any_active:.2f}s")
    print(f"  Overlap ratio:               {overlap_ratio:.1%}")
    if overlap_ratio < 0.3:
        print(
            "  --> LOW overlap: engines are mostly serializing "
            "(CUDA driver contention likely)"
        )
    elif overlap_ratio < 0.7:
        print("  --> MODERATE overlap: partial parallelism")
    else:
        print("  --> HIGH overlap: engines running in parallel as expected")

    # Tail drain.
    print(f"\n{'=' * 70}")
    print("Tail drain analysis")
    print(f"{'=' * 70}")
    print_tail_analysis(all_timings, elapsed, num_requests)


if __name__ == "__main__":
    main()
