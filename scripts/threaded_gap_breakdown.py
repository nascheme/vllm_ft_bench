# Decomposes the "other (gap)" from threaded_step_breakdown.py into labelled
# sub-components.
#
# Background
# ----------
# threaded_step_breakdown.py instruments four calls inside engine.step() and
# reports everything else as "other (gap)".  For cuda:0 at batch≈20 that gap
# is ~15ms per step, nearly 2× the single-GPU value of ~7.7ms.
#
# With the default async_scheduling=True, max_concurrent_batches=2 and the
# active step function is EngineCore.step_with_batch_queue().  Each call
# does PUSH then conditionally POP:
#
#   PUSH (every call):
#     scheduler.schedule()
#     executor.execute_model(non_block=True)   ← gpu_model_runner.execute_model()
#         INSIDE: _update_states + _prepare_inputs + _model_forward + compute_logits
#         Returns AsyncGPUModelRunnerOutput → submitted to async thread pool
#     executor.sample_tokens()                 ← gpu_model_runner.sample_tokens()
#         INSIDE:
#           _sample()          — launches sampling kernel (async)
#           _update_states_after_model_execute() — CPU bookkeeping of request states
#           _bookkeeping_sync()
#               async GPU→CPU copy then async_copy_ready_event.synchronize()
#               THIS IS THE REAL CUDA SYNC for the forward pass
#               Python loop    — O(batch_size) token_ids_cpu updates
#               ModelRunnerOutput build
#     submit result.get_output to async thread → append future to batch_queue
#
#   POP (when batch_queue is full, i.e. every other call):
#     future.result()          ← blocks until async thread completes get_output()
#     _process_aborts_queue()
#     scheduler.update_from_output()
#
# The instrumentation below wraps the same scheduler/executor methods
# regardless of which step function calls them, so the hooks capture the
# right data for both step() and step_with_batch_queue().
#
# Hypothesis: sample_tokens() (specifically _bookkeeping_sync()) owns most of
# the ~15ms gap, and the time scales with batch size because:
#   1. The CUDA sync waits for the full forward pass + sampling kernels.
#   2. The Python loop iterates over every in-flight request.
#
# What this script adds vs. threaded_step_breakdown.py
# -----------------------------------------------------
#   get_grammar_bitmask   — scheduler call between execute_model and future.result()
#   sample_tokens         — executor call (full sampling + bookkeeping)
#     _sample             — GPU sampling kernel sub-component
#     _bookkeeping_sync   — CUDA sync + Python loop sub-component
#   batch_size            — len(scheduler_output.num_scheduled_tokens) per step
#
# Inter-component gaps (residuals after all named components):
#   gap_A  schedule → execute_model        (Python call overhead)
#   gap_B  execute_model → get_grammar_bitmask
#   gap_C  get_grammar_bitmask → sample_tokens  (future.result() cost)
#   gap_D  sample_tokens → update_from_output   (_process_aborts_queue)
#   gap_E  update_from_output → process_outputs (outer engine.step wrapper)
#   gap_F  process_outputs → step end           (cleanup)

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

_timing_tls = threading.local()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAMED = [
    "schedule",
    "execute_model",
    "get_grammar_bitmask",
    "sample_tokens",
    "_sample",
    "_bookkeeping_sync",
    "update_from_output",
    "process_outputs",
    "total_step",
]

# Components that are sub-items of sample_tokens (don't double-count in gap).
_SAMPLE_SUBS = {"_sample", "_bookkeeping_sync"}

# Top-level components that together should cover total_step (used for gap calc).
_TOP_LEVEL = [
    "schedule",
    "execute_model",
    "get_grammar_bitmask",
    "sample_tokens",
    "update_from_output",
    "process_outputs",
]

# Finer gap labels — named gaps within the step sequence.
_GAP_LABELS = ["gap_A", "gap_B", "gap_C", "gap_D", "gap_E", "gap_F"]


def _current_step():
    return getattr(_timing_tls, "current_step", None)


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def _install_timing_hooks(engine):
    """Monkey-patch engine internals to record per-step timings.

    Each patched call stores its wall-clock duration in the thread-local
    current_step dict under the component's name.  Timestamps at the
    boundaries (immediately before/after each top-level component) are also
    stored so inter-component gaps can be computed precisely.
    """
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    executor = core.model_executor
    output_proc = engine.output_processor

    # Reach through WorkerWrapperBase.__getattr__ → GPUWorker.model_runner.
    model_runner = executor.driver_worker.worker.model_runner

    def _wrap(obj, attr, key, *, ts_before=None, ts_after=None):
        """Replace obj.attr with a timed wrapper.

        ts_before / ts_after: optional string keys under which to store the
        wall-clock timestamp immediately before / after the call.
        """
        orig = getattr(obj, attr)

        def wrapper(*args, **kwargs):
            step = _current_step()
            if step is not None and ts_before is not None:
                step[ts_before] = time.perf_counter()
            t0 = time.perf_counter()
            result = orig(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000
            if step is not None:
                step[key] = dt
                if ts_after is not None:
                    step[ts_after] = time.perf_counter()
            return result

        setattr(obj, attr, wrapper)

    # Top-level components — record boundary timestamps for gap computation.
    _wrap(
        scheduler,
        "schedule",
        "schedule",
        ts_before="ts_before_schedule",
        ts_after="ts_after_schedule",
    )
    _wrap(
        executor,
        "execute_model",
        "execute_model",
        ts_before="ts_before_execute_model",
        ts_after="ts_after_execute_model",
    )
    _wrap(
        scheduler,
        "get_grammar_bitmask",
        "get_grammar_bitmask",
        ts_before="ts_before_bitmask",
        ts_after="ts_after_bitmask",
    )
    _wrap(
        executor,
        "sample_tokens",
        "sample_tokens",
        ts_before="ts_before_sample_tokens",
        ts_after="ts_after_sample_tokens",
    )
    _wrap(
        scheduler,
        "update_from_output",
        "update_from_output",
        ts_before="ts_before_update",
        ts_after="ts_after_update",
    )
    _wrap(
        output_proc,
        "process_outputs",
        "process_outputs",
        ts_before="ts_before_process",
        ts_after="ts_after_process",
    )

    # Sub-components of sample_tokens (no boundary timestamps needed).
    _wrap(model_runner, "_sample", "_sample")
    _wrap(model_runner, "_bookkeeping_sync", "_bookkeeping_sync")

    # Capture batch_size from schedule's return value.
    orig_schedule = scheduler.schedule

    def schedule_with_batch(*args, **kwargs):
        result = orig_schedule(*args, **kwargs)
        step = _current_step()
        if step is not None and result is not None:
            step["batch_size"] = len(result.num_scheduled_tokens)
        return result

    # Replace the already-wrapped schedule with batch-capturing one.
    scheduler.schedule = schedule_with_batch


def timed_engine_step(engine):
    """Run engine.step() with timing active.  Returns (outputs, timings)."""
    step_timings = {}
    _timing_tls.current_step = step_timings

    ts0 = time.perf_counter()
    outputs = engine.step()
    step_timings["total_step"] = (time.perf_counter() - ts0) * 1000
    step_timings["ts_step_start"] = ts0
    step_timings["ts_step_end"] = time.perf_counter()

    # Compute named inter-component gaps using stored timestamps.
    # Each gap = time between the end of one named component and the start
    # of the next.  All in milliseconds.
    ts = step_timings
    pairs = [
        ("gap_A", "ts_step_start", "ts_before_schedule"),
        ("gap_A", "ts_after_schedule", "ts_before_execute_model"),
        ("gap_B", "ts_after_execute_model", "ts_before_bitmask"),
        ("gap_C", "ts_after_bitmask", "ts_before_sample_tokens"),
        ("gap_D", "ts_after_sample_tokens", "ts_before_update"),
        ("gap_E", "ts_after_update", "ts_before_process"),
        ("gap_F", "ts_after_process", "ts_step_end"),
    ]
    # gap_A spans two sub-gaps (before schedule + between schedule and execute)
    # merge them by summation.
    gap_accum = {}
    for label, t_start_key, t_end_key in pairs:
        t_start = ts.get(t_start_key)
        t_end = ts.get(t_end_key)
        if t_start is not None and t_end is not None:
            dt = (t_end - t_start) * 1000
            gap_accum[label] = gap_accum.get(label, 0.0) + dt

    for label, val in gap_accum.items():
        step_timings[label] = val

    # Also store the sample_tokens internal gap (between _sample and _bookkeeping).
    # This captures _update_states_after_model_execute() time.
    # (No timestamps recorded there — infer from durations.)
    st = ts.get("sample_tokens", 0)
    s = ts.get("_sample", 0)
    bs = ts.get("_bookkeeping_sync", 0)
    if st > 0:
        ts["sample_other"] = max(0.0, st - s - bs)

    _timing_tls.current_step = None
    return outputs, step_timings


# ---------------------------------------------------------------------------
# Runners (same structure as threaded_step_breakdown.py)
# ---------------------------------------------------------------------------


def run_instrumented(engine, device_index, request_items, prefix):
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _steady_state(timings):
    """Trim first 5% and last 10% of steps for steady-state analysis."""
    n = len(timings)
    start = max(1, n // 20)
    end = max(start + 1, n - n // 10)
    return timings[start:end]


def _percentiles(vals, ps=(50, 90, 99)):
    s = sorted(vals)
    n = len(s)
    result = []
    for p in ps:
        idx = min(int(n * p / 100), n - 1)
        result.append(s[idx])
    return result


def summarize_timings(label, timings):
    if not timings:
        print(f"  {label}: (no steps)")
        return

    ss = _steady_state(timings)
    n = len(timings)
    nss = len(ss)

    print(f"\n  {label}  ({n} total steps, {nss} steady-state):")

    hdr = f"    {'Component':<24s}  {'Total ms':>10s}  {'p50':>7s}  {'p90':>7s}  {'p99':>7s}  {'max':>7s}  {'batch p50':>9s}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    # Ordered display: named components then gaps.
    display_order = (
        _TOP_LEVEL + ["_sample", "_bookkeeping_sync", "sample_other"] + _GAP_LABELS
    )

    for comp in display_order:
        vals = [t.get(comp, 0.0) for t in ss if comp in t]
        if not vals or all(v == 0.0 for v in vals):
            continue

        total = sum(vals)
        p50, p90, p99 = _percentiles(vals)
        mx = max(vals)

        # For components that scale with batch, show p50 of batch_size for
        # steps where that component was non-zero.
        batch_vals = [
            t.get("batch_size", 0)
            for t in ss
            if comp in t and t.get("batch_size", 0) > 0
        ]
        batch_p50 = _percentiles(batch_vals)[0] if batch_vals else 0

        indent = "      " if comp in _SAMPLE_SUBS or comp == "sample_other" else "    "
        tag = "↳ " if comp in _SAMPLE_SUBS or comp == "sample_other" else "  "
        print(
            f"{indent}{tag}{comp:<22s}  {total:10.1f}  {p50:7.2f}  {p90:7.2f}  {p99:7.2f}  {mx:7.2f}  {batch_p50:9d}"
        )

    # Total step for reference.
    total_vals = [t.get("total_step", 0.0) for t in ss]
    total_sum = sum(total_vals)
    p50, p90, p99 = _percentiles(total_vals)
    mx = max(total_vals)
    print(
        f"    {'  total_step':<24s}  {total_sum:10.1f}  {p50:7.2f}  {p90:7.2f}  {p99:7.2f}  {mx:7.2f}"
    )

    # Summary: what fraction of total_step is accounted for by top-level components.
    top_sum_vals = [sum(t.get(c, 0.0) for c in _TOP_LEVEL) for t in ss]
    residual_vals = [t.get("total_step", 0.0) - x for t, x in zip(ss, top_sum_vals)]
    acc_frac = sum(top_sum_vals) / max(total_sum, 0.001) * 100
    res_p50 = _percentiles(residual_vals)[0]
    print(
        f"\n    Top-level components account for {acc_frac:.1f}% of total step time."
        f"  Residual gap p50 = {res_p50:.2f}ms"
        f"  (Python overhead between component calls)"
    )


def print_comparison(
    single_timings, threaded_timings, label_s="Single cuda:0", label_t="Threaded cuda:0"
):
    """Side-by-side p50 comparison, single vs. threaded."""
    ss_s = _steady_state(single_timings)
    ss_t = _steady_state(threaded_timings)

    print(
        f"\n  {'Component':<24s}  {label_s:>13s}  {label_t:>13s}  {'Ratio':>6s}  {'Δ ms':>7s}"
    )
    print("  " + "-" * 70)

    display_order = (
        _TOP_LEVEL
        + ["_sample", "_bookkeeping_sync", "sample_other"]
        + _GAP_LABELS
        + ["total_step"]
    )

    for comp in display_order:
        s_vals = [t.get(comp, 0.0) for t in ss_s if comp in t]
        t_vals = [t.get(comp, 0.0) for t in ss_t if comp in t]
        if not s_vals and not t_vals:
            continue

        sp50 = _percentiles(s_vals)[0] if s_vals else 0.0
        tp50 = _percentiles(t_vals)[0] if t_vals else 0.0
        ratio = tp50 / sp50 if sp50 > 0.001 else float("inf")
        delta = tp50 - sp50

        indent = "    " if comp in _SAMPLE_SUBS or comp == "sample_other" else "  "
        tag = "↳ " if comp in _SAMPLE_SUBS or comp == "sample_other" else "  "
        print(
            f"{indent}{tag}{comp:<22s}  {sp50:13.3f}  {tp50:13.3f}  {ratio:6.2f}  {delta:+7.3f}"
        )


def print_batch_correlation(label, timings):
    """Show how key components vary across batch-size quantiles."""
    ss = _steady_state(timings)
    # Group steps by batch_size quantile (quartiles).
    steps_with_bs = [
        (t.get("batch_size", 0), t) for t in ss if t.get("batch_size", 0) > 0
    ]
    if not steps_with_bs:
        return

    steps_with_bs.sort(key=lambda x: x[0])
    n = len(steps_with_bs)
    q25 = steps_with_bs[: n // 4]
    q75 = steps_with_bs[3 * n // 4 :]

    def med_batch(group):
        return _percentiles([x[0] for x in group])[0]

    def med_comp(group, comp):
        vals = [x[1].get(comp, 0.0) for x in group if comp in x[1]]
        return _percentiles(vals)[0] if vals else 0.0

    print(f"\n  {label} — batch-size correlation (bottom quartile vs. top quartile):")
    print(
        f"    {'Component':<24s}  {'Q1 batch':>9s}  {'Q1 p50 ms':>10s}  {'Q3 batch':>9s}  {'Q3 p50 ms':>10s}  {'ratio':>6s}"
    )
    print("    " + "-" * 75)

    for comp in [
        "sample_tokens",
        "_sample",
        "_bookkeeping_sync",
        "sample_other",
        "execute_model",
    ]:
        if not any(comp in t for t in ss):
            continue
        b1 = med_batch(q25)
        v1 = med_comp(q25, comp)
        b3 = med_batch(q75)
        v3 = med_comp(q75, comp)
        ratio = v3 / v1 if v1 > 0.001 else float("inf")
        print(
            f"    {comp:<24s}  {b1:9.0f}  {v1:10.3f}  {b3:9.0f}  {v3:10.3f}  {ratio:6.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser("Fine-grained step gap breakdown.")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
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
        _install_timing_hooks(e)

    total = len(request_items)
    half = total // 2
    items_0 = request_items[:half]
    items_1 = request_items[half:]

    SEP = "=" * 72

    # ------------------------------------------------------------------
    # Test 1: cuda:0 alone
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Test 1: cuda:0 alone (single-GPU baseline)")
    print(SEP)
    r0 = run_instrumented(engines[0], 0, items_0, "single0")
    print(
        f"  {r0['completed']} reqs, {r0['elapsed']:.1f}s, "
        f"{r0['completed'] / r0['elapsed']:.1f} req/s, {r0['steps']} steps"
    )
    summarize_timings("Single cuda:0", r0["timings"])
    print_batch_correlation("Single cuda:0", r0["timings"])

    # ------------------------------------------------------------------
    # Test 2: Both threaded
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Test 2: cuda:0 + cuda:1 threaded")
    print(SEP)

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
            f"  cuda:{i}: {r['completed']} reqs, {r['elapsed']:.1f}s, {r['steps']} steps"
        )
        summarize_timings(f"Threaded cuda:{i}", r["timings"])
        print_batch_correlation(f"Threaded cuda:{i}", r["timings"])

    # ------------------------------------------------------------------
    # Comparison: single vs. threaded cuda:0
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Comparison: single vs. threaded cuda:0  (p50, steady-state)")
    print(SEP)
    print_comparison(r0["timings"], results[0]["timings"])

    # ------------------------------------------------------------------
    # Summary: gap attribution
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("Gap attribution summary (threaded cuda:0, steady-state p50)")
    print(SEP)

    ss_t = _steady_state(results[0]["timings"])
    total_p50 = _percentiles([t.get("total_step", 0) for t in ss_t])[0]

    # Old "other (gap)" as computed by threaded_step_breakdown.py.
    old_gap_components = [
        "schedule",
        "execute_model",
        "update_from_output",
        "process_outputs",
    ]
    old_named_p50 = sum(
        _percentiles([t.get(c, 0) for t in ss_t if c in t])[0]
        for c in old_gap_components
    )
    old_gap_p50 = total_p50 - old_named_p50

    # New breakdown of that old gap.
    new_named = ["get_grammar_bitmask", "sample_tokens"] + _GAP_LABELS
    attrib = {}
    for c in new_named:
        vals = [t.get(c, 0) for t in ss_t if c in t]
        attrib[c] = _percentiles(vals)[0] if vals else 0.0

    print(f"\n  Old 'other (gap)' p50:  {old_gap_p50:.2f}ms")
    print(f"  {'Component':<24s}  {'p50 ms':>8s}  {'% of old gap':>12s}")
    print("  " + "-" * 48)
    accounted = 0.0
    for c, v in attrib.items():
        pct = v / old_gap_p50 * 100 if old_gap_p50 > 0 else 0
        accounted += v
        print(f"  {c:<24s}  {v:8.3f}  {pct:11.1f}%")
    remaining = old_gap_p50 - accounted
    rem_pct = remaining / old_gap_p50 * 100 if old_gap_p50 > 0 else 0
    print(f"  {'(remaining unaccounted)':<24s}  {remaining:8.3f}  {rem_pct:11.1f}%")


if __name__ == "__main__":
    main()
