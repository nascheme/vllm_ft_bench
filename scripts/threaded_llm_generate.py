# Threaded multi-GPU vLLM benchmark — mimics LLM.generate() per engine.
#
# Each engine thread preloads all its requests then runs a tight step loop
# (no queue polling, no per-step timing). This matches what LLM.generate()
# does internally, making it a fair comparison with dp_generate.py.
#
# The ONLY difference from dp_generate.py is threads vs processes.

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
    """Tight step loop — mimics LLM._run_engine(use_tqdm=False)."""
    torch.cuda.set_device(device_index)
    outputs = {}
    while engine.has_unfinished_requests():
        for ro in engine.step():
            if ro.finished:
                outputs[ro.request_id] = ro
    return list(outputs.values())


def engine_worker(engine, device_index, request_items, result, barrier):
    """Preload requests, wait for all engines, then generate."""
    torch.cuda.set_device(device_index)
    renderer = engine.renderer

    # Add all requests (like LLM.generate() does).
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(f"gpu{device_index}_{i}", proc_input, sp)

    # Sync so both engines start stepping at the same time.
    barrier.wait()

    t0 = time.time()
    outputs = engine_generate(engine, device_index)
    t1 = time.time()

    prompt_tokens = 0
    output_tokens = 0
    for ro in outputs:
        if ro.prompt_token_ids:
            prompt_tokens += len(ro.prompt_token_ids)
        output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)

    result["completed"] = len(outputs)
    result["prompt_tokens"] = prompt_tokens
    result["output_tokens"] = output_tokens
    result["elapsed"] = t1 - t0


def main():
    parser = make_arg_parser(
        "Threaded multi-GPU vLLM benchmark (LLM.generate()-style step loop).",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    engines = []
    for i in range(args.num_gpus):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))
    print(f"All {args.num_gpus} engines created.")

    # Contiguous partition (same as dp_generate.py).
    total = len(request_items)
    chunk = total // args.num_gpus

    barrier = threading.Barrier(args.num_gpus)
    results = [{} for _ in range(args.num_gpus)]
    threads = []

    for i in range(args.num_gpus):
        start_idx = i * chunk
        end_idx = start_idx + chunk if i < args.num_gpus - 1 else total
        my_items = request_items[start_idx:end_idx]

        t = threading.Thread(
            target=engine_worker,
            args=(engines[i], i, my_items, results[i], barrier),
            name=f"LLM::engine{i}",
        )
        threads.append(t)

    gen_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gen_elapsed = time.time() - gen_start

    engine_stats = [
        {
            "completed": r["completed"],
            "prompt_tokens": r["prompt_tokens"],
            "output_tokens": r["output_tokens"],
        }
        for r in results
    ]
    print_throughput_results(gen_elapsed, engine_stats)

    print(f"Inference time: {gen_elapsed:.1f}s")
    for i, r in enumerate(results):
        print(
            f"  cuda:{i}: {r['completed']} reqs, "
            f"{r['prompt_tokens']} prompt toks, {r['output_tokens']} output toks, "
            f"{r['elapsed']:.1f}s"
        )


if __name__ == "__main__":
    main()
