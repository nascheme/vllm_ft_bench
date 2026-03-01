# Step-level breakdown profiler: identifies WHERE threading overhead accumulates.
#
# Uses monkey-patching to instrument key methods inside the normal engine.step()
# call, so we don't need to replicate internal logic (avoiding async output
# handling issues).
#
# Instruments:
#   - scheduler.schedule()
#   - model_executor.execute_model() submit
#   - scheduler.update_from_output()
#   - output_processor.process_outputs()
#   - Overall engine.step() wall time
#
# Runs two phases in the SAME process:
#   1. cuda:0 alone (baseline)
#   2. cuda:0 + cuda:1 threaded
# Then compares per-component times.

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


# Thread-local storage for per-step timing accumulation.
_timing_tls = threading.local()


def _install_timing_hooks(engine):
    """Monkey-patch engine internals to record per-step timings.

    Timings accumulate in thread-local storage so multiple engines
    can be instrumented simultaneously without interference.
    """
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    executor = core.model_executor
    output_proc = engine.output_processor

    # --- scheduler.schedule() ---
    orig_schedule = scheduler.schedule

    def timed_schedule(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_schedule(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        step = getattr(_timing_tls, "current_step", None)
        if step is not None:
            step["schedule"] = dt
        return result

    scheduler.schedule = timed_schedule

    # --- model_executor.execute_model() ---
    orig_execute = executor.execute_model

    def timed_execute(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_execute(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        step = getattr(_timing_tls, "current_step", None)
        if step is not None:
            step["execute_model"] = dt
        return result

    executor.execute_model = timed_execute

    # --- scheduler.update_from_output() ---
    orig_update = scheduler.update_from_output

    def timed_update(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_update(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        step = getattr(_timing_tls, "current_step", None)
        if step is not None:
            step["update_from_output"] = dt
        return result

    scheduler.update_from_output = timed_update

    # --- output_processor.process_outputs() ---
    orig_process = output_proc.process_outputs

    def timed_process(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_process(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000
        step = getattr(_timing_tls, "current_step", None)
        if step is not None:
            step["process_outputs"] = dt
        return result

    output_proc.process_outputs = timed_process


def timed_engine_step(engine):
    """Call engine.step() with timing hooks active. Returns (outputs, timings)."""
    step_timings = {}
    _timing_tls.current_step = step_timings

    t0 = time.perf_counter()
    outputs = engine.step()
    step_timings["total_step"] = (time.perf_counter() - t0) * 1000

    _timing_tls.current_step = None
    return outputs, step_timings


def run_instrumented(engine, device_index, request_items, prefix):
    """Run engine with per-step instrumentation."""
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"{prefix}_{i}", proc_input, sp)

    all_timings = []
    finished = {}

    t0 = time.time()
    while engine.has_unfinished_requests():
        step_outputs, timings = timed_engine_step(engine)
        all_timings.append(timings)
        for ro in step_outputs:
            if ro.finished:
                finished[ro.request_id] = ro
    elapsed = time.time() - t0

    output_tokens = sum(
        sum(len(o.token_ids) for o in ro.outputs if o) for ro in finished.values()
    )
    return {
        "completed": len(finished),
        "output_tokens": output_tokens,
        "elapsed": elapsed,
        "steps": len(all_timings),
        "timings": all_timings,
    }


def threaded_instrumented_worker(
    engine, device_index, request_items, prefix, result, barrier
):
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"{prefix}_{i}", proc_input, sp)

    barrier.wait()

    all_timings = []
    finished = {}

    t0 = time.time()
    while engine.has_unfinished_requests():
        step_outputs, timings = timed_engine_step(engine)
        all_timings.append(timings)
        for ro in step_outputs:
            if ro.finished:
                finished[ro.request_id] = ro
    elapsed = time.time() - t0

    output_tokens = sum(
        sum(len(o.token_ids) for o in ro.outputs if o) for ro in finished.values()
    )
    result["completed"] = len(finished)
    result["output_tokens"] = output_tokens
    result["elapsed"] = elapsed
    result["steps"] = len(all_timings)
    result["timings"] = all_timings


def summarize_timings(label, timings):
    """Print per-component timing summary."""
    if not timings:
        print(f"  {label}: (no steps)")
        return

    components = [
        "schedule",
        "execute_model",
        "update_from_output",
        "process_outputs",
        "total_step",
    ]

    # Skip first 5% and last 10% for steady state.
    n = len(timings)
    ss_start = max(1, n // 20)
    ss_end = max(ss_start + 1, n - n // 10)
    ss_timings = timings[ss_start:ss_end]

    print(f"\n  {label} ({n} total steps, {len(ss_timings)} steady-state):")
    print(
        f"    {'Component':<22s}  {'Total(ms)':>10s}  {'p50(ms)':>8s}  "
        f"{'p90(ms)':>8s}  {'p99(ms)':>8s}  {'max(ms)':>8s}"
    )
    print(f"    {'-' * 22}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for comp in components:
        vals = [t.get(comp, 0) for t in ss_timings]
        if not vals or all(v == 0 for v in vals):
            continue
        total = sum(vals)
        s = sorted(vals)
        nn = len(s)
        p50 = s[nn // 2]
        p90 = s[int(nn * 0.9)]
        p99 = s[min(int(nn * 0.99), nn - 1)]
        mx = s[-1]
        print(
            f"    {comp:<22s}  {total:10.1f}  {p50:8.2f}  "
            f"{p90:8.2f}  {p99:8.2f}  {mx:8.2f}"
        )

    # Also show "other" time (total_step minus known components).
    other_vals = []
    for t in ss_timings:
        known = sum(t.get(c, 0) for c in components if c != "total_step")
        other_vals.append(t.get("total_step", 0) - known)
    if other_vals:
        s = sorted(other_vals)
        nn = len(s)
        total = sum(other_vals)
        p50 = s[nn // 2]
        p90 = s[int(nn * 0.9)]
        p99 = s[min(int(nn * 0.99), nn - 1)]
        mx = s[-1]
        print(
            f"    {'other (gap)':22s}  {total:10.1f}  {p50:8.2f}  "
            f"{p90:8.2f}  {p99:8.2f}  {mx:8.2f}"
        )


def main():
    parser = make_arg_parser("Step-level breakdown profiler.")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    # Create both engines upfront.
    engines = []
    for i in range(2):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))

    # Install timing hooks on both engines.
    for e in engines:
        _install_timing_hooks(e)

    total = len(request_items)
    half = total // 2
    items_0 = request_items[:half]
    items_1 = request_items[half:]

    # --- Test 1: cuda:0 alone ---
    print(f"\n{'=' * 70}")
    print("Test 1: cuda:0 alone (baseline)")
    print(f"{'=' * 70}")
    r0 = run_instrumented(engines[0], 0, items_0, "single0")
    print(
        f"  {r0['completed']} reqs, {r0['elapsed']:.1f}s, "
        f"{r0['completed'] / r0['elapsed']:.1f} req/s, "
        f"{r0['steps']} steps"
    )
    summarize_timings("Single cuda:0", r0["timings"])

    # --- Test 2: Both threaded ---
    print(f"\n{'=' * 70}")
    print("Test 2: cuda:0 + cuda:1 threaded")
    print(f"{'=' * 70}")

    barrier = threading.Barrier(2)
    results = [{}, {}]

    threads = []
    for i, (engine, items) in enumerate([(engines[0], items_0), (engines[1], items_1)]):
        t = threading.Thread(
            target=threaded_instrumented_worker,
            args=(engine, i, items, f"thr{i}", results[i], barrier),
            name=f"engine{i}",
        )
        threads.append(t)

    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_elapsed = time.time() - t_start

    total_reqs = sum(r["completed"] for r in results)
    print(
        f"  Total: {total_reqs} reqs, {t_elapsed:.1f}s, "
        f"{total_reqs / t_elapsed:.1f} req/s"
    )

    for i, r in enumerate(results):
        print(
            f"  cuda:{i}: {r['completed']} reqs, {r['elapsed']:.1f}s, "
            f"{r['steps']} steps"
        )
        summarize_timings(f"Threaded cuda:{i}", r["timings"])

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("Comparison: component-level overhead (single vs threaded cuda:0)")
    print(f"{'=' * 70}")

    components = [
        "schedule",
        "execute_model",
        "update_from_output",
        "process_outputs",
        "total_step",
    ]

    s_timings = r0["timings"]
    t_timings = results[0]["timings"]

    # Steady-state slices.
    sn = len(s_timings)
    s_ss = s_timings[max(1, sn // 20) : max(2, sn - sn // 10)]
    tn = len(t_timings)
    t_ss = t_timings[max(1, tn // 20) : max(2, tn - tn // 10)]

    print(
        f"\n  {'Component':<22s}  {'Single p50':>10s}  {'Thread p50':>10s}  "
        f"{'Ratio':>7s}  {'Δ(ms)':>7s}"
    )
    print(f"  {'-' * 22}  {'-' * 10}  {'-' * 10}  {'-' * 7}  {'-' * 7}")

    for comp in components:
        s_vals = sorted([t.get(comp, 0) for t in s_ss])
        t_vals = sorted([t.get(comp, 0) for t in t_ss])
        if not s_vals or not t_vals:
            continue
        sp50 = s_vals[len(s_vals) // 2]
        tp50 = t_vals[len(t_vals) // 2]
        ratio = tp50 / sp50 if sp50 > 0.001 else float("inf")
        delta = tp50 - sp50
        print(f"  {comp:<22s}  {sp50:10.3f}  {tp50:10.3f}  {ratio:7.2f}  {delta:+7.3f}")

    # "other" comparison
    s_other = sorted(
        [
            t.get("total_step", 0)
            - sum(t.get(c, 0) for c in components if c != "total_step")
            for t in s_ss
        ]
    )
    t_other = sorted(
        [
            t.get("total_step", 0)
            - sum(t.get(c, 0) for c in components if c != "total_step")
            for t in t_ss
        ]
    )
    sp50 = s_other[len(s_other) // 2]
    tp50 = t_other[len(t_other) // 2]
    print(
        f"  {'other (gap)':22s}  {sp50:10.3f}  {tp50:10.3f}  "
        f"{tp50 / sp50 if sp50 > 0.001 else float('inf'):7.2f}  {tp50 - sp50:+7.3f}"
    )


if __name__ == "__main__":
    main()
