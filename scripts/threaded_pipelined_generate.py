# Threaded multi-GPU vLLM with pipelined output processing.
#
# With --in-subprocess: runs all threaded work inside a clean subprocess
# that imports vllm fresh (no inherited state from parent).
#
# Without: runs directly in the main process.

import time

from vllm_ft.util import make_arg_parser


def engine_generate_pipelined(engine, device_index):
    """Step loop with output processing timing."""
    import torch

    torch.cuda.set_device(device_index)

    engine_core = engine.engine_core
    output_processor = engine.output_processor
    finished = {}

    step_count = 0
    empty_steps = 0
    total_process_ms = 0.0

    while engine.has_unfinished_requests():
        outputs = engine_core.get_output()
        step_count += 1

        if not outputs.outputs:
            empty_steps += 1
            continue

        t0 = time.perf_counter()
        processed = output_processor.process_outputs(
            outputs.outputs,
            engine_core_timestamp=outputs.timestamp,
            iteration_stats=None,
        )
        output_processor.update_scheduler_stats(outputs.scheduler_stats)
        total_process_ms += (time.perf_counter() - t0) * 1000

        engine_core.abort_requests(processed.reqs_to_abort)

        for ro in processed.request_outputs:
            if ro.finished:
                finished[ro.request_id] = ro

    return (list(finished.values()), step_count, empty_steps, total_process_ms)


def engine_worker(engine, device_index, request_items, result, barrier):
    import torch

    torch.cuda.set_device(device_index)
    input_processor = engine.input_processor

    for i, (req, sp) in enumerate(request_items):
        ecr = input_processor.process_inputs(
            f"gpu{device_index}_{i}",
            req.prompt,
            sp,
            arrival_time=time.time(),
            supported_tasks=engine.get_supported_tasks(),
        )
        engine.add_request(ecr.request_id, ecr, sp, prompt_text=req.prompt)

    barrier.wait()

    t0 = time.time()
    outputs, steps, empty, process_ms = engine_generate_pipelined(engine, device_index)
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
    result["steps"] = steps
    result["empty_steps"] = empty
    result["process_ms"] = process_ms


def run_threaded(args):
    """All vllm imports happen here — called either directly or in subprocess."""
    import os
    import threading

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import EngineArgs
    from vllm.tokenizers import get_tokenizer
    from vllm.usage.usage_lib import UsageContext

    from vllm_ft.util import (
        apply_forward_context_monkey_patch,
        build_request_items,
        create_engine,
        print_throughput_results,
    )

    apply_forward_context_monkey_patch()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    engines = []
    for i in range(args.num_gpus):
        use_graphs = getattr(args, "cuda_graphs", False)
        print(f"Creating engine on cuda:{i} (cuda_graphs={use_graphs}) ...")
        e = create_engine(
            engine_args, i, UsageContext.LLM_CLASS, cuda_graphs=use_graphs
        )
        core = e.engine_core.engine_core
        print(
            f"  async_scheduling={core.async_scheduling}, "
            f"batch_queue={'ON' if core.batch_queue is not None else 'OFF'}, "
            f"batch_queue_size={core.batch_queue_size}"
        )
        engines.append(e)
    print(f"All {args.num_gpus} engines created.")

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

    print(f"\nInference time: {gen_elapsed:.1f}s")
    for i, r in enumerate(results):
        print(
            f"  cuda:{i}: {r['completed']} reqs, {r['elapsed']:.1f}s, "
            f"{r['steps']} steps ({r['empty_steps']} empty), "
            f"process_outputs: {r['process_ms']:.0f}ms total "
            f"({r['process_ms'] / max(1, r['steps'] - r['empty_steps']):.2f}ms/step)"
        )


def _clean_subprocess_entry(args_ns_dict):
    """Entry point for clean subprocess — imports vllm fresh."""
    import argparse

    args = argparse.Namespace(**args_ns_dict)
    run_threaded(args)


if __name__ == "__main__":
    parser = make_arg_parser(
        "Threaded multi-GPU vLLM with pipelined output processing.",
    )
    parser.add_argument(
        "--in-subprocess",
        action="store_true",
        help="Run in a clean subprocess (no inherited vllm state)",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Enable FULL CUDA graph capture (reduces driver lock contention)",
    )
    args = parser.parse_args()

    if args.in_subprocess:
        # Fork from a CLEAN parent (no vllm/torch imported at module level).
        from multiprocessing import Process

        # Convert Namespace to dict for pickling.
        ns = {k: v for k, v in vars(args).items() if k != "in_subprocess"}
        p = Process(target=_clean_subprocess_entry, args=(ns,))
        p.start()
        p.join()
    else:
        run_threaded(args)
