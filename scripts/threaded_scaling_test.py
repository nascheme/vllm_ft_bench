# Threading scaling test: 1 GPU vs 2 GPU in the same process.
#
# Runs three configurations back-to-back in the same process:
#   1. Single engine on cuda:0 (baseline)
#   2. Single engine on cuda:1 (baseline)
#   3. Two engines threaded (cuda:0 + cuda:1)
#
# If single engines each process 500 reqs in ~42s and dual does it in ~56s,
# the 33% overhead is from CUDA driver lock contention.
# If single engines are also slow (~56s), something else is wrong.

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


def engine_generate(engine, device_index):
    """Tight step loop."""
    torch.cuda.set_device(device_index)
    outputs = {}
    steps = 0
    while engine.has_unfinished_requests():
        for ro in engine.step():
            if ro.finished:
                outputs[ro.request_id] = ro
        steps += 1
    return list(outputs.values()), steps


def load_and_run(engine, device_index, request_items):
    """Load requests into engine and run."""
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"gpu{device_index}_{i}", proc_input, sp)

    t0 = time.time()
    outputs, steps = engine_generate(engine, device_index)
    elapsed = time.time() - t0

    output_tokens = sum(
        sum(len(o.token_ids) for o in ro.outputs if o) for ro in outputs
    )
    return {
        "completed": len(outputs),
        "output_tokens": output_tokens,
        "elapsed": elapsed,
        "steps": steps,
    }


def threaded_worker(engine, device_index, request_items, result, barrier):
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"gpu{device_index}_{i}", proc_input, sp)

    barrier.wait()

    t0 = time.time()
    outputs, steps = engine_generate(engine, device_index)
    elapsed = time.time() - t0

    output_tokens = sum(
        sum(len(o.token_ids) for o in ro.outputs if o) for ro in outputs
    )
    result["completed"] = len(outputs)
    result["output_tokens"] = output_tokens
    result["elapsed"] = elapsed
    result["steps"] = steps


def main():
    parser = make_arg_parser("Threading scaling test: 1 GPU vs 2 GPU.")
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

    # Contiguous partition.
    total = len(request_items)
    half = total // 2
    items_0 = request_items[:half]
    items_1 = request_items[half:]

    # --- Test 1: cuda:0 alone ---
    print(f"\n{'=' * 60}")
    print("Test 1: cuda:0 alone (500 reqs)")
    print(f"{'=' * 60}")
    r0 = load_and_run(engines[0], 0, items_0)
    print(
        f"  {r0['completed']} reqs, {r0['elapsed']:.1f}s, "
        f"{r0['completed'] / r0['elapsed']:.1f} req/s, "
        f"{r0['steps']} steps, {r0['output_tokens']} output toks"
    )

    # --- Test 2: cuda:1 alone ---
    print(f"\n{'=' * 60}")
    print("Test 2: cuda:1 alone (500 reqs)")
    print(f"{'=' * 60}")
    r1 = load_and_run(engines[1], 1, items_1)
    print(
        f"  {r1['completed']} reqs, {r1['elapsed']:.1f}s, "
        f"{r1['completed'] / r1['elapsed']:.1f} req/s, "
        f"{r1['steps']} steps, {r1['output_tokens']} output toks"
    )

    # --- Test 3: Both threaded ---
    print(f"\n{'=' * 60}")
    print("Test 3: cuda:0 + cuda:1 threaded (1000 reqs)")
    print(f"{'=' * 60}")

    # Need fresh request IDs — re-add with different prefix.
    barrier = threading.Barrier(2)
    results = [{}, {}]

    # Use fresh request items with new IDs.
    threads = []
    for i, (engine, items) in enumerate([(engines[0], items_0), (engines[1], items_1)]):
        # Re-prefix items for unique request IDs.
        t = threading.Thread(
            target=threaded_worker,
            args=(engine, i, items, results[i], barrier),
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
            f"    cuda:{i}: {r['completed']} reqs, {r['elapsed']:.1f}s, "
            f"{r['completed'] / r['elapsed']:.1f} req/s, "
            f"{r['steps']} steps, {r['output_tokens']} output toks"
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  cuda:0 alone:    {r0['elapsed']:.1f}s")
    print(f"  cuda:1 alone:    {r1['elapsed']:.1f}s")
    print(
        f"  cuda:0 threaded: {results[0]['elapsed']:.1f}s  "
        f"(ratio: {results[0]['elapsed'] / r0['elapsed']:.2f}x)"
    )
    print(
        f"  cuda:1 threaded: {results[1]['elapsed']:.1f}s  "
        f"(ratio: {results[1]['elapsed'] / r1['elapsed']:.2f}x)"
    )
    print("\n  If ratio > 1.05: threading adds overhead (CUDA contention)")
    print("  If ratio ≈ 1.0:  no overhead, gap is elsewhere")


if __name__ == "__main__":
    main()
