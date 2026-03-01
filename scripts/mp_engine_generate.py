# Static-partition multi-GPU vLLM throughput benchmark (subprocesses + create_engine).
#
# Like mp_static_generate.py but uses create_engine() + a manual preload step
# loop instead of LLM.generate().  This enables --cuda-graphs (CUDAGraphMode.FULL)
# for a fair apples-to-apples comparison with threaded_static_generate.py.
#
# Architecture:
#   Main process:  build dataset, partition, spawn workers, collect results
#   Worker 0:      create_engine(cuda:0, ...), preload requests, step loop, report
#   Worker 1:      create_engine(cuda:1, ...), preload requests, step loop, report
#
# Because each worker sets CUDA_VISIBLE_DEVICES to a single card, the engine
# always sees it as cuda:0 — device_index=0 is passed to create_engine().

from multiprocessing import Process, Queue

from vllm_ft.util import (
    make_arg_parser,
    print_throughput_results,
)


def engine_worker(
    device_index,
    num_gpus,
    model,
    num_requests,
    cuda_graphs,
    result_queue,
    dataset,
    prompt_source,
    input_len,
    output_len,
):
    """Subprocess: create engine, preload requests, run step loop, report."""
    import argparse
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    # Keep EngineCore in-process (no nested subprocess spawning).
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import EngineArgs
    from vllm.tokenizers import get_tokenizer
    from vllm.usage.usage_lib import UsageContext

    from vllm_ft.util import build_request_items, create_engine, render_request

    # Build dataset independently in each worker (same as mp_static_generate.py).
    tokenizer = get_tokenizer(model)
    worker_args = argparse.Namespace(
        model=model,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        prompt_source=prompt_source,
        dataset=dataset,
    )
    all_request_items = build_request_items(worker_args, tokenizer)

    # Contiguous partition (same as mp_static_generate.py).
    total = len(all_request_items)
    chunk = total // num_gpus
    start_idx = device_index * chunk
    end_idx = start_idx + chunk if device_index < num_gpus - 1 else total
    my_request_items = all_request_items[start_idx:end_idx]

    # enforce_eager=True so that create_engine() controls graph capture via
    # cuda_graphs=True/False rather than vllm's default compile path.
    engine_args_obj = EngineArgs(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    # CUDA_VISIBLE_DEVICES remaps the physical card to cuda:0 in this process.
    engine = create_engine(
        engine_args_obj, 0, UsageContext.LLM_CLASS, cuda_graphs=cuda_graphs
    )

    # Preload: tokenize and add all requests before stepping (matches
    # LLM.generate() / threaded_static_generate.py --preload behaviour).
    renderer = engine.renderer
    for i, (req, sp) in enumerate(my_request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(str(i), proc_input, sp)

    # Step loop.
    completed = 0
    prompt_tokens = 0
    output_tokens = 0

    gen_start = time.time()
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                completed += 1
                if output.prompt_token_ids:
                    prompt_tokens += len(output.prompt_token_ids)
                output_tokens += sum(len(o.token_ids) for o in output.outputs if o)
    gen_end = time.time()

    result_queue.put(
        {
            "gpu_index": device_index,
            "completed": completed,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "gen_start": gen_start,
            "gen_end": gen_end,
        }
    )


def main():
    parser = make_arg_parser(
        "Subprocess static-partition multi-GPU vLLM benchmark (create_engine + step loop).",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Enable CUDA graph capture (CUDAGraphMode.FULL) in each engine.",
    )
    args = parser.parse_args()

    result_queue = Queue()

    print(
        f"Spawning {args.num_gpus} engine subprocesses "
        f"(cuda_graphs={args.cuda_graphs}) ..."
    )
    procs = []
    for gpu in range(args.num_gpus):
        p = Process(
            target=engine_worker,
            args=(
                gpu,
                args.num_gpus,
                args.model,
                args.num_requests,
                args.cuda_graphs,
                result_queue,
                args.dataset,
                args.prompt_source,
                args.input_len,
                args.output_len,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=300)
        if p.exitcode is None:
            print(f"Killing process {p.pid} (timeout)")
            p.kill()
        elif p.exitcode:
            print(f"Worker {p.pid} exited with code {p.exitcode}")

    # Collect results.
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    results.sort(key=lambda r: r["gpu_index"])

    if not results:
        print("No results collected — workers may have failed.")
        return

    gen_start = min(r["gen_start"] for r in results)
    gen_end = max(r["gen_end"] for r in results)
    gen_elapsed = gen_end - gen_start

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
    for r in results:
        rank_elapsed = r["gen_end"] - r["gen_start"]
        print(
            f"  cuda:{r['gpu_index']}: {r['completed']} reqs, "
            f"{r['prompt_tokens']} prompt toks, {r['output_tokens']} output toks, "
            f"{rank_elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
