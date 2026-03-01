# Test CUDA graph capture with enforce_eager=True (torch.compile disabled).
#
# Verifies that FULL CUDA graphs work on free-threaded Python 3.14t
# without torch.compile, then benchmarks single vs threaded to see
# if reduced driver API calls mitigate the contention.

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
    print_throughput_results,
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


def load_and_run(engine, device_index, request_items, prefix):
    """Load requests into engine and run."""
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"{prefix}_{i}", proc_input, sp)

    t0 = time.time()
    outputs, steps = engine_generate(engine, device_index)
    elapsed = time.time() - t0

    prompt_tokens = sum(
        len(ro.prompt_token_ids) for ro in outputs if ro.prompt_token_ids
    )
    output_tokens = sum(
        sum(len(o.token_ids) for o in ro.outputs if o) for ro in outputs
    )
    return {
        "completed": len(outputs),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "elapsed": elapsed,
        "steps": steps,
    }


def threaded_worker(engine, device_index, request_items, prefix, result, barrier):
    torch.cuda.set_device(device_index)
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"{prefix}_{i}", proc_input, sp)

    barrier.wait()

    t0 = time.time()
    outputs, steps = engine_generate(engine, device_index)
    elapsed = time.time() - t0

    prompt_tokens = sum(
        len(ro.prompt_token_ids) for ro in outputs if ro.prompt_token_ids
    )
    output_tokens = sum(
        sum(len(o.token_ids) for o in ro.outputs if o) for ro in outputs
    )
    result["completed"] = len(outputs)
    result["prompt_tokens"] = prompt_tokens
    result["output_tokens"] = output_tokens
    result["elapsed"] = elapsed
    result["steps"] = steps


def main():
    parser = make_arg_parser("CUDA graph test: enforce_eager + FULL graphs.")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    # Create engines with CUDA graphs enabled.
    engines = []
    for i in range(args.num_gpus):
        print(f"Creating engine on cuda:{i} (CUDA graphs=FULL) ...")
        engines.append(
            create_engine(
                engine_args,
                i,
                UsageContext.LLM_CLASS,
                cuda_graphs=True,
            )
        )
    print(f"All {args.num_gpus} engines created.\n")

    total = len(request_items)
    half = total // 2
    items_0 = request_items[:half]
    items_1 = request_items[half:]

    # --- Test 1: cuda:0 alone ---
    print(f"{'=' * 60}")
    print("Test 1: cuda:0 alone (CUDA graphs)")
    print(f"{'=' * 60}")
    r0 = load_and_run(engines[0], 0, items_0, "single0")
    print(
        f"  {r0['completed']} reqs, {r0['elapsed']:.1f}s, "
        f"{r0['completed'] / r0['elapsed']:.1f} req/s, "
        f"{r0['steps']} steps"
    )

    # --- Test 2: Both threaded ---
    print(f"\n{'=' * 60}")
    print("Test 2: cuda:0 + cuda:1 threaded (CUDA graphs)")
    print(f"{'=' * 60}")

    barrier = threading.Barrier(2)
    results = [{}, {}]
    threads = []
    for i, (engine, items) in enumerate([(engines[0], items_0), (engines[1], items_1)]):
        t = threading.Thread(
            target=threaded_worker,
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
    engine_stats = [
        {
            "completed": r["completed"],
            "prompt_tokens": r["prompt_tokens"],
            "output_tokens": r["output_tokens"],
        }
        for r in results
    ]
    print_throughput_results(t_elapsed, engine_stats)

    for i, r in enumerate(results):
        print(
            f"  cuda:{i}: {r['completed']} reqs, {r['elapsed']:.1f}s, "
            f"{r['completed'] / r['elapsed']:.1f} req/s, "
            f"{r['steps']} steps"
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(
        f"  cuda:0 alone:    {r0['elapsed']:.1f}s  "
        f"({r0['completed'] / r0['elapsed']:.1f} req/s)"
    )
    print(f"  Threaded total:  {t_elapsed:.1f}s  ({total_reqs / t_elapsed:.1f} req/s)")
    print(
        f"  cuda:0 threaded: {results[0]['elapsed']:.1f}s  "
        f"(ratio: {results[0]['elapsed'] / r0['elapsed']:.2f}x)"
    )
    print(f"  cuda:1 threaded: {results[1]['elapsed']:.1f}s")
    print("\n  Compare with non-graph threaded (~17 req/s, 1.2x ratio)")


if __name__ == "__main__":
    main()
