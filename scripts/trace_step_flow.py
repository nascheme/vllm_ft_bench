# Execution-order tracer for the vLLM step() call chain.
#
# Prints one line per hook-point entry/exit so you can read the exact
# sequence of calls, with millisecond wall-clock offsets from step start.
#
# Runs two phases:
#   Phase 1 – cuda:0 alone, STEPS_TO_TRACE steps
#   Phase 2 – cuda:0 + cuda:1 threaded, STEPS_TO_TRACE steps each
#
# Each print line:
#   [thread]  step=N  +XXX.XXXms  ENTER/EXIT  component_name
#
# Set NUM_REQS and STEPS_TO_TRACE at the top to taste.

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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
    render_request,
)

apply_forward_context_monkey_patch()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_REQS = 6  # total requests split across two engines
STEPS_TO_TRACE = 4  # how many steps (per engine) to emit trace lines for

# ---------------------------------------------------------------------------
# Thread-local state
# ---------------------------------------------------------------------------

_tls = threading.local()  # .step_num, .t_step_start, .tracing


def _t():
    """ms since current step started."""
    t0 = getattr(_tls, "t_step_start", None)
    if t0 is None:
        return 0.0
    return (time.perf_counter() - t0) * 1000.0


def _trace(event: str, component: str):
    if not getattr(_tls, "tracing", False):
        return
    tname = threading.current_thread().name
    step = getattr(_tls, "step_num", "?")
    print(f"  [{tname:12s}] step={step}  +{_t():8.3f}ms  {event:5s}  {component}")


# ---------------------------------------------------------------------------
# Hook installer
# ---------------------------------------------------------------------------


def _wrap(obj, attr, name, *, enter_msg=None, exit_msg=None):
    """Replace obj.attr with a version that calls _trace on entry and exit."""
    orig = getattr(obj, attr)
    enter = enter_msg or name
    exit_ = exit_msg or name

    def wrapper(*args, **kwargs):
        _trace("ENTER", enter)
        result = orig(*args, **kwargs)
        _trace("EXIT ", exit_)
        return result

    setattr(obj, attr, wrapper)
    return orig


def install_trace_hooks(engine):
    inproc = engine.engine_core  # InprocClient
    core = inproc.engine_core  # EngineCore
    scheduler = core.scheduler
    executor = core.model_executor
    output_proc = engine.output_processor
    model_runner = executor.driver_worker.worker.model_runner

    # ------------------------------------------------------------------
    # Startup diagnostics
    # ------------------------------------------------------------------
    step_fn_name = getattr(core.step_fn, "__name__", str(core.step_fn))
    bq_size = core.batch_queue_size
    bq_exists = core.batch_queue is not None
    async_sched = getattr(core, "async_scheduling", "?")
    use_async = getattr(model_runner, "use_async_scheduling", "?")
    async_thread = getattr(executor, "async_output_thread", None)
    print(
        f"\n  [diag] step_fn            = {step_fn_name}"
        f"\n  [diag] batch_queue_size   = {bq_size}  (batch_queue exists: {bq_exists})"
        f"\n  [diag] core.async_scheduling       = {async_sched}"
        f"\n  [diag] model_runner.use_async_scheduling = {use_async}"
        f"\n  [diag] executor.async_output_thread = {async_thread!r}"
    )

    # ------------------------------------------------------------------
    # Wrap step_fn to mark early-return vs pop path
    # ------------------------------------------------------------------
    orig_step_fn = core.step_fn

    def traced_step_fn():
        outputs, model_executed = orig_step_fn()
        if getattr(_tls, "tracing", False):
            if outputs is None:
                _trace(
                    "     ", "step_fn returned None  ← EARLY RETURN (future not ready)"
                )
            else:
                total_tok_outputs = sum(
                    len(v.outputs) for v in outputs.values() if v is not None
                )
                _trace(
                    "     ",
                    f"step_fn returned outputs  ← POP PATH ({total_tok_outputs} req outputs)",
                )
        return outputs, model_executed

    core.step_fn = traced_step_fn

    # ------------------------------------------------------------------
    # Wrap async_output_thread.submit to intercept every GPU-future and
    # instrument its .result() call (that is where the GPU→CPU sync waits).
    # ------------------------------------------------------------------
    if async_thread is not None:
        orig_submit = async_thread.submit

        def traced_submit(fn, *args, **kwargs):
            submit_time = time.perf_counter()
            tracing_at_submit = getattr(_tls, "tracing", False)

            # Wrap the submitted function to time its internals.
            # fn is a bound method (AsyncGPUModelRunnerOutput.get_output).
            # NOTE: copy_event here is NOT the same event that get_output()
            # uses internally (async_copy_ready_event).  This means
            # copy_sync will read 0.0ms and the real GPU sync wait will
            # appear in tolist+rest.  See trace_tolist_patch.py for the
            # corrected measurement.
            obj = getattr(fn, "__self__", None)
            copy_event = getattr(obj, "copy_event", None)

            def timed_fn():
                t0 = time.perf_counter()
                sched = (t0 - submit_time) * 1000

                if copy_event is not None:
                    copy_event.synchronize()
                t1 = time.perf_counter()
                sync_ms = (t1 - t0) * 1000

                result = fn()  # includes real GPU sync + tolist (see NOTE above)
                t2 = time.perf_counter()
                rest_ms = (t2 - t1) * 1000

                if tracing_at_submit:
                    print(
                        f"  [async_thread] sched={sched:.1f}ms  "
                        f"copy_sync={sync_ms:.1f}ms  "
                        f"tolist+rest={rest_ms:.1f}ms"
                    )
                return result

            future = orig_submit(timed_fn)

            if tracing_at_submit:
                _trace(
                    "     ",
                    "async_thread.submit() → future created (GPU copy launched)",
                )

            orig_result = future.result

            def traced_result(timeout=None):
                _trace(
                    "ENTER", "future.result() ← blocks here waiting for GPU→CPU copy"
                )
                val = orig_result(timeout)
                _trace("EXIT ", "future.result() ← GPU→CPU copy done, output ready")
                return val

            future.result = traced_result
            return future

        async_thread.submit = traced_submit
    else:
        print(
            "  [diag] WARNING: async_output_thread is None — futures are pre-set, no async GPU copy"
        )

    # ------------------------------------------------------------------
    # Named component hooks (same as before)
    # ------------------------------------------------------------------
    _wrap(scheduler, "schedule", "scheduler.schedule")
    _wrap(executor, "execute_model", "executor.execute_model")
    _wrap(scheduler, "get_grammar_bitmask", "scheduler.get_grammar_bitmask")
    _wrap(executor, "sample_tokens", "executor.sample_tokens")
    _wrap(model_runner, "_sample", "model_runner._sample")
    _wrap(model_runner, "_bookkeeping_sync", "model_runner._bookkeeping_sync")
    _wrap(core, "_process_aborts_queue", "engine_core._process_aborts_queue")
    _wrap(scheduler, "update_from_output", "scheduler.update_from_output")
    _wrap(output_proc, "process_outputs", "output_processor.process_outputs")

    # Capture batch_size from schedule's return value (double-wrap is fine).
    orig_schedule = scheduler.schedule

    def schedule_with_bs(*args, **kwargs):
        result = orig_schedule(*args, **kwargs)
        if result is not None and getattr(_tls, "tracing", False):
            bs = len(result.num_scheduled_tokens)
            tname = threading.current_thread().name
            step = getattr(_tls, "step_num", "?")
            print(
                f"  [{tname:12s}] step={step}  +{_t():8.3f}ms           batch_size={bs}"
            )
        return result

    scheduler.schedule = schedule_with_bs


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


def run_traced(engine, device_index, request_items, prefix, max_steps):
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"{prefix}_{i}", proc_input, sp)

    step_num = 0
    finished = {}
    while engine.has_unfinished_requests():
        step_num += 1
        _tls.step_num = step_num
        _tls.t_step_start = time.perf_counter()
        _tls.tracing = step_num <= max_steps

        if _tls.tracing:
            tname = threading.current_thread().name
            print(f"\n  [{tname:12s}] ── step {step_num} START ──")

        _trace("ENTER", "engine.step")
        outputs = engine.step()
        _trace("EXIT ", "engine.step")

        if _tls.tracing:
            elapsed = _t()
            tname = threading.current_thread().name
            print(f"  [{tname:12s}] step={step_num}  total={elapsed:.3f}ms")

        for ro in outputs:
            if ro.finished:
                finished[ro.request_id] = ro

    return finished


def threaded_worker(
    engine, device_index, request_items, prefix, result_dict, barrier, max_steps
):
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"{prefix}_{i}", proc_input, sp)

    barrier.wait()

    step_num = 0
    finished = {}
    while engine.has_unfinished_requests():
        step_num += 1
        _tls.step_num = step_num
        _tls.t_step_start = time.perf_counter()
        _tls.tracing = step_num <= max_steps

        if _tls.tracing:
            tname = threading.current_thread().name
            print(f"\n  [{tname:12s}] ── step {step_num} START ──")

        _trace("ENTER", "engine.step")
        outputs = engine.step()
        _trace("EXIT ", "engine.step")

        if _tls.tracing:
            elapsed = _t()
            tname = threading.current_thread().name
            print(f"  [{tname:12s}] step={step_num}  total={elapsed:.3f}ms")

        for ro in outputs:
            if ro.finished:
                finished[ro.request_id] = ro

    result_dict["finished"] = len(finished)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser("Execution-order tracer.")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    args.num_prompts = NUM_REQS
    request_items = build_request_items(args, tokenizer)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    engines = []
    for i in range(2):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))

    for e in engines:
        install_trace_hooks(e)

    half = len(request_items) // 2
    items_0 = request_items[:half]
    items_1 = request_items[half:]

    SEP = "=" * 68

    # ------------------------------------------------------------------
    # Phase 1: cuda:0 alone
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Phase 1: cuda:0 alone")
    print(SEP)
    threading.current_thread().name = "single-gpu "
    run_traced(engines[0], 0, items_0, "s0", STEPS_TO_TRACE)

    # ------------------------------------------------------------------
    # Phase 2: threaded
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Phase 2: cuda:0 + cuda:1 threaded")
    print(SEP)

    barrier = threading.Barrier(2)
    results = [{}, {}]
    threads = [
        threading.Thread(
            target=threaded_worker,
            args=(engines[0], 0, items_0, "t0", results[0], barrier, STEPS_TO_TRACE),
            name="cuda:0      ",
        ),
        threading.Thread(
            target=threaded_worker,
            args=(engines[1], 1, items_1, "t1", results[1], barrier, STEPS_TO_TRACE),
            name="cuda:1      ",
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()
