# Diagnostic script for the async output pipeline in vLLM V1.
#
# Instruments AsyncGPUModelRunnerOutput.get_output() to measure:
#   - async_copy_ready_event.synchronize() time (GPU→CPU copy stream wait)
#   - .tolist() + rest time
#   - sampled_token_ids_cpu tensor shape
#
# Key finding: the 600-780ms per step is entirely in the copy-stream sync,
# NOT in .tolist().  The tensor shape is (batch, 1) for non-spec-decode,
# making .tolist() trivially fast (~0ms).  See TIMING.md Phase 4.
#
# Two modes:
#   Default (diagnostic):  Logs shapes, sync time, and tolist time per step.
#   --patch-tolist:        Per-row slicing (only useful for spec-decode with
#                          wide tensors; no-op for the common (batch, 1) case).
#
# Based on trace_step_flow.py.  Runs two phases:
#   Phase 1 – cuda:0 alone, N steps
#   Phase 2 – cuda:0 + cuda:1 threaded, N steps each
#
# Target class (discovered via runtime introspection):
#   AsyncGPUModelRunnerOutput  (vllm/v1/worker/gpu_model_runner.py)
#     .sampled_token_ids_cpu    torch.Tensor [batch, max_gen_len]
#     .async_copy_ready_event   torch.Event
#     .vocab_size               int
#     ._model_runner_output     ModelRunnerOutput
#     ._invalid_req_indices     list[int]
#     ._logprobs_tensors_cpu    LogprobsTensors | None

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

NUM_REQS = 6
DEFAULT_STEPS = 4

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
    print(
        f"  [{tname:12s}] step={step}  +{_t():8.3f}ms  {event:5s}  {component}"
    )


# ---------------------------------------------------------------------------
# Hook installer (reused from trace_step_flow.py)
# ---------------------------------------------------------------------------


def _wrap(obj, attr, name):
    """Replace obj.attr with a version that calls _trace on entry and exit."""
    orig = getattr(obj, attr)

    def wrapper(*args, **kwargs):
        _trace("ENTER", name)
        result = orig(*args, **kwargs)
        _trace("EXIT ", name)
        return result

    setattr(obj, attr, wrapper)
    return orig


def install_trace_hooks(engine):
    """Install component-level tracing hooks (same as trace_step_flow.py)."""
    inproc = engine.engine_core
    core = inproc.engine_core
    scheduler = core.scheduler
    executor = core.model_executor
    output_proc = engine.output_processor
    model_runner = executor.driver_worker.worker.model_runner

    # Wrap step_fn
    orig_step_fn = core.step_fn

    def traced_step_fn():
        outputs, model_executed = orig_step_fn()
        if getattr(_tls, "tracing", False):
            if outputs is None:
                _trace("     ", "step_fn → None (EARLY RETURN)")
            else:
                n = sum(
                    len(v.outputs) for v in outputs.values() if v is not None
                )
                _trace("     ", f"step_fn → outputs ({n} req outputs)")
        return outputs, model_executed

    core.step_fn = traced_step_fn

    # Named component hooks
    _wrap(scheduler, "schedule", "scheduler.schedule")
    _wrap(executor, "execute_model", "executor.execute_model")
    _wrap(scheduler, "get_grammar_bitmask", "scheduler.get_grammar_bitmask")
    _wrap(executor, "sample_tokens", "executor.sample_tokens")
    _wrap(model_runner, "_sample", "model_runner._sample")
    _wrap(model_runner, "_bookkeeping_sync", "model_runner._bookkeeping_sync")
    _wrap(core, "_process_aborts_queue", "engine_core._process_aborts_queue")
    _wrap(scheduler, "update_from_output", "scheduler.update_from_output")
    _wrap(output_proc, "process_outputs", "output_processor.process_outputs")

    # Batch-size annotation
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
# Diagnostic mode: instrument get_output() to log shapes and per-line timings
# ---------------------------------------------------------------------------


def _find_async_output(fn):
    """Find the async output object from a submitted callable.

    The executor submits either:
      - result.get_output   (bound method → fn.__self__ is the object)
      - get_output_list     (closure capturing result → walk fn.__closure__)
    """
    # Try bound method first.
    obj = getattr(fn, "__self__", None)
    if obj is not None and hasattr(obj, "sampled_token_ids_cpu"):
        return obj

    # Walk closure cells for the closure path.
    closure = getattr(fn, "__closure__", None)
    if closure is not None:
        for cell in closure:
            try:
                val = cell.cell_contents
                if hasattr(val, "sampled_token_ids_cpu") and hasattr(
                    val, "async_copy_ready_event"
                ):
                    return val
            except ValueError:
                pass  # empty cell

    return None


def install_tolist_diagnostics(engine):
    """Wrap async_thread.submit to instrument get_output() with shape and
    timing diagnostics.

    Finds the AsyncGPUModelRunnerOutput via fn.__self__ (bound method) or
    fn.__closure__ (closure wrapper), then logs sampled_token_ids_cpu shape,
    copy sync time, and tolist+rest time.
    """
    executor = engine.engine_core.engine_core.model_executor
    async_thread = getattr(executor, "async_output_thread", None)
    if async_thread is None:
        print("  [diag] WARNING: async_output_thread is None")
        return

    orig_submit = async_thread.submit

    def diagnostic_submit(fn, *args, **kwargs):
        submit_time = time.perf_counter()
        tracing_at_submit = getattr(_tls, "tracing", False)

        def timed_fn():
            t0 = time.perf_counter()
            sched_ms = (t0 - submit_time) * 1000

            # Find the AsyncGPUModelRunnerOutput object.
            obj = _find_async_output(fn)

            # Get tensor shape before get_output() consumes it.
            shape = "?"
            if obj is not None:
                tensor = getattr(obj, "sampled_token_ids_cpu", None)
                if tensor is not None and hasattr(tensor, "shape"):
                    shape = tuple(tensor.shape)

            # Time async_copy_ready_event.synchronize() separately.
            event = getattr(obj, "async_copy_ready_event", None)
            if event is not None:
                event.synchronize()
            t1 = time.perf_counter()
            sync_ms = (t1 - t0) * 1000

            # Call the actual get_output (which re-syncs, then .tolist()).
            result = fn()
            t2 = time.perf_counter()
            tolist_rest_ms = (t2 - t1) * 1000

            if tracing_at_submit:
                print(
                    f"  [get_output] sched={sched_ms:.1f}ms  "
                    f"sync={sync_ms:.1f}ms  "
                    f"tolist+rest={tolist_rest_ms:.1f}ms  "
                    f"sampled_token_ids_cpu={shape}"
                )
            return result

        future = orig_submit(timed_fn)

        if tracing_at_submit:
            _trace("     ", "async_thread.submit() → future created")

        orig_result = future.result

        def traced_result(timeout=None):
            _trace("ENTER", "future.result()")
            val = orig_result(timeout)
            _trace("EXIT ", "future.result()")
            return val

        future.result = traced_result
        return future

    async_thread.submit = diagnostic_submit


# ---------------------------------------------------------------------------
# Patch mode: replace get_output() with per-row sliced version
# ---------------------------------------------------------------------------


def install_tolist_patch(engine):
    """Monkey-patch AsyncGPUModelRunnerOutput.get_output to avoid full
    .tolist() on the 2D sampled_token_ids_cpu tensor.

    The original does:
        sampled_token_ids_cpu.tolist()  →  list[list[int]]  [batch, max_gen_len]
    then clears invalid rows.

    Our patch:
      - For max_gen_len == 1 (non-spec-decode, the common case):
        converts each row individually via row.tolist(), clearing invalid
        rows. This avoids a single giant .tolist() call.
      - For max_gen_len > 1 (spec decode): falls through to original.

    The class is AsyncGPUModelRunnerOutput (vllm/v1/worker/gpu_model_runner.py).
    """
    from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

    orig_get_output = AsyncGPUModelRunnerOutput.get_output

    def patched_get_output(self):
        t0 = time.perf_counter()

        max_gen_len = self.sampled_token_ids_cpu.shape[-1]

        # Only optimize the non-spec-decode path (max_gen_len == 1).
        # For spec decode, fall through to the original which uses
        # RejectionSampler.parse_output.
        if max_gen_len != 1:
            result = orig_get_output(self)
            t1 = time.perf_counter()
            if getattr(_tls, "tracing", False):
                print(
                    f"  [patched_get_output] spec_decode path  "
                    f"shape={tuple(self.sampled_token_ids_cpu.shape)}  "
                    f"total={((t1 - t0) * 1000):.1f}ms"
                )
            return result

        # --- Optimized non-spec-decode path ---
        self.async_copy_ready_event.synchronize()
        t1 = time.perf_counter()

        # Release device tensors early.
        del self._logprobs_tensors
        del self._sampled_token_ids

        # Per-row .tolist() instead of full 2D .tolist().
        tensor = self.sampled_token_ids_cpu
        invalid = set(self._invalid_req_indices)
        batch_size = tensor.shape[0]
        valid_sampled_token_ids = [
            [] if i in invalid else tensor[i].tolist()
            for i in range(batch_size)
        ]
        t2 = time.perf_counter()

        # Logprobs (same as original).
        logprobs_lists = None
        if self._logprobs_tensors_cpu is not None:
            logprobs_lists = self._logprobs_tensors_cpu.tolists()
        t3 = time.perf_counter()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists

        if getattr(_tls, "tracing", False):
            sync_ms = (t1 - t0) * 1000
            tolist_ms = (t2 - t1) * 1000
            logprobs_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000
            print(
                f"  [patched_get_output] shape={tuple(tensor.shape)}  "
                f"batch={batch_size}  invalid={len(invalid)}  "
                f"sync={sync_ms:.1f}ms  tolist={tolist_ms:.1f}ms  "
                f"logprobs={logprobs_ms:.1f}ms  total={total_ms:.1f}ms"
            )

        return output

    # Patch at class level so all instances use it.
    AsyncGPUModelRunnerOutput.get_output = patched_get_output

    # Also instrument the async_thread.submit to trace future.result() timing.
    executor = engine.engine_core.engine_core.model_executor
    async_thread = getattr(executor, "async_output_thread", None)
    if async_thread is not None:
        orig_submit = async_thread.submit

        def patched_submit(fn, *args, **kwargs):
            tracing_at_submit = getattr(_tls, "tracing", False)
            future = orig_submit(fn, *args, **kwargs)

            if tracing_at_submit:
                _trace("     ", "async_thread.submit() → future created")

            orig_result_fn = future.result

            def traced_result(timeout=None):
                _trace("ENTER", "future.result()")
                val = orig_result_fn(timeout)
                _trace("EXIT ", "future.result()")
                return val

            future.result = traced_result
            return future

        async_thread.submit = patched_submit


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
    engine,
    device_index,
    request_items,
    prefix,
    result_dict,
    barrier,
    max_steps,
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
    parser = make_arg_parser(
        "Diagnostic/patch for .tolist() overhead in async output."
    )
    parser.add_argument(
        "--patch-tolist",
        action="store_true",
        default=False,
        help="Enable the per-row slicing monkey-patch instead of diagnostic mode.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of steps to trace per engine (default: {DEFAULT_STEPS}).",
    )
    args = parser.parse_args()

    mode = "PATCH" if args.patch_tolist else "DIAGNOSTIC"
    print(f"Mode: {mode}")
    print(f"Steps to trace: {args.steps}")

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

    # Install hooks on all engines.
    for e in engines:
        install_trace_hooks(e)
        if args.patch_tolist:
            install_tolist_patch(e)
        else:
            install_tolist_diagnostics(e)

    half = len(request_items) // 2
    items_0 = request_items[:half]
    items_1 = request_items[half:]

    SEP = "=" * 68

    # ------------------------------------------------------------------
    # Phase 1: cuda:0 alone
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"Phase 1: cuda:0 alone  ({mode} mode)")
    print(SEP)
    threading.current_thread().name = "single-gpu "
    run_traced(engines[0], 0, items_0, "s0", args.steps)

    # ------------------------------------------------------------------
    # Phase 2: threaded
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print(f"Phase 2: cuda:0 + cuda:1 threaded  ({mode} mode)")
    print(SEP)

    barrier = threading.Barrier(2)
    results = [{}, {}]
    threads = [
        threading.Thread(
            target=threaded_worker,
            args=(
                engines[0],
                0,
                items_0,
                "t0",
                results[0],
                barrier,
                args.steps,
            ),
            name="cuda:0      ",
        ),
        threading.Thread(
            target=threaded_worker,
            args=(
                engines[1],
                1,
                items_1,
                "t1",
                results[1],
                barrier,
                args.steps,
            ),
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
