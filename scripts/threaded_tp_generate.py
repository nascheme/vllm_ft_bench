# Single-process threaded tensor-parallel vLLM throughput benchmark.
#
# Eliminates SHM IPC spin-waiting overhead by running TP workers as threads
# within one process.  NCCL handles GPU communication; control plane uses
# direct Python object passing (zero-copy).
#
#   python scripts/threaded_tp_generate.py \
#       --model HuggingFaceTB/SmolLM2-360M --num-gpus 2 \
#       --num-requests 10 --prompt-source hardcoded

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import queue
import threading
import time
from collections import defaultdict

import torch
import torch._dynamo  # noqa: E402

torch._dynamo.config.disable = True

from vllm import EngineArgs  # noqa: E402
from vllm.tokenizers import get_tokenizer  # noqa: E402

from vllm_ft.util import (  # noqa: E402
    apply_forward_context_monkey_patch,
    build_request_items,
    generate_cudagraph_capture_sizes,
    get_speculative_config,
    make_arg_parser,
    print_throughput_results,
    render_request,
)
from vllm_ft.threaded_tp import (  # noqa: E402
    apply_parallel_state_tls_patch,
    apply_vllm_config_tls_patch,
)

apply_forward_context_monkey_patch()
apply_parallel_state_tls_patch()
apply_vllm_config_tls_patch()

# ---------------------------------------------------------------------------

MAX_PULL_PER_STEP = 16


# ---------------------------------------------------------------------------
# Phase timing — lightweight per-step CPU breakdown
# ---------------------------------------------------------------------------


class PhaseTimer:
    """Accumulates per-phase wall times across steps."""

    def __init__(self):
        self.totals = defaultdict(float)   # phase -> total ms
        self.counts = defaultdict(int)     # phase -> count
        self._t0 = None
        self._phase = None

    def begin(self, phase: str):
        now = time.perf_counter()
        if self._phase is not None:
            elapsed = (now - self._t0) * 1000
            self.totals[self._phase] += elapsed
            self.counts[self._phase] += 1
        self._phase = phase
        self._t0 = now

    def end(self):
        if self._phase is not None:
            elapsed = (time.perf_counter() - self._t0) * 1000
            self.totals[self._phase] += elapsed
            self.counts[self._phase] += 1
            self._phase = None

    def summary(self):
        lines = ["  phase                    total_ms    calls   avg_ms"]
        for phase in sorted(self.totals, key=lambda k: -self.totals[k]):
            total = self.totals[phase]
            count = self.counts[phase]
            avg = total / count if count else 0
            lines.append(f"  {phase:<24s} {total:>8.1f}  {count:>7d}  {avg:>7.3f}")
        return "\n".join(lines)


def _patch_step_with_phase_timer(core, phase_timer):
    """Monkey-patch EngineCore.step / step_with_batch_queue to record phase times."""

    def timed_step():
        phase_timer.begin("schedule")
        if not core.scheduler.has_requests():
            phase_timer.end()
            return {}, False
        scheduler_output = core.scheduler.schedule()

        phase_timer.begin("execute_model")
        future = core.model_executor.execute_model(scheduler_output, non_block=True)

        phase_timer.begin("grammar_bitmask")
        grammar_output = core.scheduler.get_grammar_bitmask(scheduler_output)

        phase_timer.begin("future.result")
        model_output = future.result()

        if model_output is None:
            phase_timer.begin("sample_tokens")
            model_output = core.model_executor.sample_tokens(grammar_output)

        phase_timer.begin("process_aborts")
        core._process_aborts_queue()

        phase_timer.begin("update_from_output")
        engine_core_outputs = core.scheduler.update_from_output(
            scheduler_output, model_output
        )
        phase_timer.end()

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def timed_step_bq():
        batch_queue = core.batch_queue
        assert batch_queue is not None
        assert len(batch_queue) < core.batch_queue_size

        is_ec_consumer = getattr(core, "is_ec_consumer", True)
        is_pooling_model = getattr(core, "is_pooling_model", False)

        model_executed = False
        deferred_scheduler_output = None

        if core.scheduler.has_requests():
            phase_timer.begin("schedule")
            scheduler_output = core.scheduler.schedule()

            phase_timer.begin("execute_model")
            exec_future = core.model_executor.execute_model(
                scheduler_output, non_block=True
            )

            if is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if is_pooling_model or not model_executed:
                future = exec_future
            else:
                pending = getattr(
                    scheduler_output, "pending_structured_output_tokens", None
                )
                if not pending:
                    phase_timer.begin("grammar_bitmask")
                    grammar_output = core.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    phase_timer.begin("sample_tokens")
                    future = core.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < core.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    phase_timer.end()
                    return None, True

        elif not batch_queue:
            phase_timer.end()
            return None, False

        phase_timer.begin("future.result")
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        model_output = future.result()
        if model_output is None:
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

        phase_timer.begin("process_aborts")
        core._process_aborts_queue()

        phase_timer.begin("update_from_output")
        engine_core_outputs = core.scheduler.update_from_output(
            scheduler_output, model_output
        )

        if deferred_scheduler_output:
            phase_timer.begin("deferred_sample")
            grammar_output = core.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = core.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft(
                (future, deferred_scheduler_output, exec_future)
            )

        phase_timer.end()
        return engine_core_outputs, model_executed

    if core.batch_queue is not None:
        core.step_fn = timed_step_bq
    else:
        core.step_fn = timed_step


class StepTimings:
    """Step timing records."""

    def __init__(self):
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
        if not self.step_records:
            return "  (no steps)"
        lines = ["  step | wall_ms | batch_size | finished"]
        for i, (wall_ms, batch_size, finished) in enumerate(self.step_records[:20]):
            lines.append(
                f"  {i:4d} | {wall_ms:7.2f} | {batch_size:10d} | {finished:8d}"
            )
        return "\n".join(lines)


def main():
    parser = make_arg_parser(
        "Single-process threaded TP vLLM throughput benchmark.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Tokenize and add ALL requests before starting engine steps.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        default=None,
        metavar="TEXT",
        help="Custom prompt(s) to run. Repeatable. Overrides --prompt-source/dataset.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        default=False,
        help="Print prompt and generated text for each completed request.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="DIR",
        help="Enable torch.profiler, write trace to DIR (e.g. output/trace).",
    )
    parser.add_argument(
        "--profile-wait",
        type=int,
        default=50,
        help="Steps to skip before profiling (warmup). Default: 50.",
    )
    parser.add_argument(
        "--profile-active",
        type=int,
        default=20,
        help="Steps to profile. Default: 20.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)

    if args.prompt:
        from vllm import SamplingParams
        from vllm.benchmarks.datasets import SampleRequest

        sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.output_len)
        request_items = []
        for prompt_text in args.prompt:
            token_ids = tokenizer.encode(prompt_text)
            req = SampleRequest(
                prompt=prompt_text,
                prompt_len=len(token_ids),
                expected_output_len=args.output_len,
            )
            request_items.append((req, sp))
        print(f"Custom prompts ready: {len(request_items)} requests")
    else:
        request_items = build_request_items(args, tokenizer)
    num_requests = len(request_items)

    print(
        f"Creating threaded TP engine with tp_size={args.num_gpus}, "
        f"model={args.model} ..."
    )

    import logging

    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor.abstract import Executor

    engine_args = EngineArgs(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        enforce_eager=True,  # avoid torch.compile (unsupported on free-threaded)
        gpu_memory_utilization=0.8,
        speculative_config=get_speculative_config(args),
    )

    # Suppress noisy warnings during config creation.
    _seed_logger = logging.getLogger("vllm.engine.arg_utils")
    _seed_prev = _seed_logger.level
    _seed_logger.setLevel(logging.ERROR)
    from vllm.usage.usage_lib import UsageContext

    vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    _seed_logger.setLevel(_seed_prev)

    # Re-enable CUDA graphs (FULL mode, no torch.compile needed).
    # enforce_eager zeroed out capture sizes; we regenerate them.
    if args.cuda_graphs:
        from vllm.config import CUDAGraphMode

        vllm_config.model_config.enforce_eager = False
        cc = vllm_config.compilation_config
        cc.cudagraph_mode = CUDAGraphMode.FULL
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        sizes = generate_cudagraph_capture_sizes(
            max_seqs, vllm_config.speculative_config
        )
        cc.cudagraph_capture_sizes = sizes
        cc.max_cudagraph_capture_size = sizes[-1]
        print(f"CUDA graphs enabled (FULL mode, {len(sizes)} capture sizes)")

    # Override executor to our threaded TP executor.
    vllm_config.parallel_config.distributed_executor_backend = (
        "vllm_ft.threaded_tp.ThreadedTPExecutor"
    )

    executor_class = Executor.get_class(vllm_config)
    engine = LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
        usage_context=UsageContext.LLM_CLASS,
    )
    print("Engine created.")

    # --- Add requests ---
    if args.preload:
        print("Preloading: tokenizing and adding all requests ...")
        renderer = engine.renderer
        for i, (req, sp) in enumerate(request_items):
            proc_input = render_request(renderer, req.prompt)
            engine.add_request(str(i), proc_input, sp)
        print(f"Preloaded {num_requests} requests.")
    else:
        renderer = engine.renderer
        req_queue = queue.Queue(maxsize=num_requests)
        tok_done = threading.Event()

        def tokenizer_worker():
            for i, (req, sp) in enumerate(request_items):
                proc_input = render_request(renderer, req.prompt)
                req_queue.put((str(i), proc_input, sp))
            tok_done.set()

        tok_thread = threading.Thread(
            target=tokenizer_worker, name="LLM::tok", daemon=True
        )
        tok_thread.start()

    # --- Phase timer ---
    phase_timer = PhaseTimer()
    core = engine.engine_core.engine_core
    _patch_step_with_phase_timer(core, phase_timer)

    # --- Torch profiler (optional) ---
    profiler = None
    if args.profile:
        os.makedirs(args.profile, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=args.profile_active,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile),
            record_shapes=True,
            with_stack=True,
        )
        print(
            f"Profiler: skip {args.profile_wait} warmup steps, "
            f"capture {args.profile_active} steps → {args.profile}"
        )

    # --- Step loop ---
    timings = StepTimings()
    stats = {"completed": 0, "prompt_tokens": 0, "output_tokens": 0}
    finished_outputs = [] if args.show_output else None
    profiler_started = False

    start_time = time.time()

    while True:
        # Pull requests from queue (trickle mode).
        if not args.preload:
            for _ in range(MAX_PULL_PER_STEP):
                try:
                    req_id, proc_input, sp = req_queue.get_nowait()
                    engine.add_request(req_id, proc_input, sp)
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
                    stats["completed"] += 1
                    if output.prompt_token_ids:
                        stats["prompt_tokens"] += len(output.prompt_token_ids)
                    stats["output_tokens"] += sum(
                        len(o.token_ids) for o in output.outputs if o
                    )
                    if finished_outputs is not None:
                        finished_outputs.append(output)

            timings.record_step(wall_ms, num_running + num_waiting, finished_count)

            # Start profiler after warmup.
            if (
                profiler is not None
                and not profiler_started
                and timings.step_count >= args.profile_wait
            ):
                profiler_started = True
                profiler.start()
                print(f"  Profiler started at step {timings.step_count}")

            if profiler_started and profiler is not None:
                profiler.step()
                steps_captured = timings.step_count - args.profile_wait
                if steps_captured >= args.profile_active:
                    profiler.stop()
                    profiler = None
                    print(
                        f"  Profiler stopped after {steps_captured} steps, "
                        f"trace written to {args.profile}/"
                    )
        else:
            if args.preload:
                break
            if tok_done.is_set() and req_queue.empty():
                break
            try:
                req_id, proc_input, sp = req_queue.get(timeout=0.5)
                engine.add_request(req_id, proc_input, sp)
            except queue.Empty:
                if tok_done.is_set():
                    break

        # Progress report.
        elapsed = time.time() - start_time
        if timings.step_count % 100 == 0 and timings.step_count > 0:
            print(
                f"  [{elapsed:.1f}s] {stats['completed']}/{num_requests} "
                f"completed, step {timings.step_count} ..."
            )

    if not args.preload:
        tok_thread.join()

    elapsed = time.time() - start_time

    # --- Results ---
    engine_stats = [stats]
    print_throughput_results(elapsed, engine_stats)
    print(f"Inference time: {elapsed:.1f}s")

    print(f"\n{'=' * 70}")
    print("Step profile")
    print(f"{'=' * 70}")
    print(timings.summary())

    print(f"\n{'=' * 70}")
    print("Phase breakdown (CPU time)")
    print(f"{'=' * 70}")
    print(phase_timer.summary())

    print(f"\n{'=' * 70}")
    print("Batch ramp-up (first 20 steps)")
    print(f"{'=' * 70}")
    print(timings.ramp_up_summary())

    print(f"\n{'=' * 70}")
    print("Tail drain (last 20 steps)")
    print(f"{'=' * 70}")
    recs = timings.step_records[-20:]
    lines = ["  step | wall_ms | batch_size | finished"]
    offset = max(0, len(timings.step_records) - 20)
    for i, (wall_ms, batch_size, finished) in enumerate(recs):
        lines.append(
            f"  {offset + i:4d} | {wall_ms:7.2f} | {batch_size:10d} | {finished:8d}"
        )
    print("\n".join(lines))

    if finished_outputs:
        print(f"\n{'=' * 70}")
        print(f"Generated outputs ({len(finished_outputs)} requests)")
        print(f"{'=' * 70}")
        for output in sorted(finished_outputs, key=lambda o: int(o.request_id)):
            prompt = output.prompt or "(unknown)"
            text = output.outputs[0].text if output.outputs else "(no output)"
            print(f"\n--- Request {output.request_id} ---")
            print(f"Prompt:  {prompt}")
            print(f"Output:  {text}")


if __name__ == "__main__":
    main()
