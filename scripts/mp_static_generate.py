# Static-partition dual-engine multi-GPU vLLM throughput benchmark (subprocesses).
#
# Identical to threaded_static_generate.py --preload but uses subprocesses
# instead of threads for the engine workers. This isolates whether the
# throughput gap vs dp_generate.py is from in-process interference
# (CUDA driver contention, CPU cache thrashing, etc.).
#
# Architecture:
#   Main process:  build dataset, partition, spawn workers, collect results
#   Worker 0:      create_engine(cuda:0), add requests, step loop, report
#   Worker 1:      create_engine(cuda:1), add requests, step loop, report

from multiprocessing import Process, Queue

from vllm_ft.util import (
    make_arg_parser,
    print_throughput_results,
)


def engine_worker(device_index, num_gpus, model, num_requests, result_queue):
    """Subprocess: build data and run LLM.generate() — matches dp_generate.py."""
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)

    from vllm import LLM
    from vllm.tokenizers import get_tokenizer

    from vllm_ft.util import build_request_items
    import argparse

    # Build dataset independently (same as dp_generate.py workers).
    tokenizer = get_tokenizer(model)
    worker_args = argparse.Namespace(
        model=model,
        num_requests=num_requests,
        input_len=1024,
        output_len=128,
        prompt_source="random",
        dataset=None,
    )
    all_request_items = build_request_items(worker_args, tokenizer)

    # Contiguous partition (same as dp_generate.py).
    total = len(all_request_items)
    chunk = total // num_gpus
    start_idx = device_index * chunk
    end_idx = start_idx + chunk if device_index < num_gpus - 1 else total
    my_request_items = all_request_items[start_idx:end_idx]

    llm = LLM(
        model=model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )

    prompts = [req.prompt for req, _ in my_request_items]
    sampling_params = [sp for _, sp in my_request_items]

    gen_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_end = time.time()

    prompt_tokens = 0
    output_tokens = 0
    for ro in outputs:
        if ro.prompt_token_ids:
            prompt_tokens += len(ro.prompt_token_ids)
        output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)

    result_queue.put(
        {
            "gpu_index": device_index,
            "completed": len(outputs),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "gen_start": gen_start,
            "gen_end": gen_end,
        }
    )


def main():
    parser = make_arg_parser(
        "Subprocess static-partition multi-GPU vLLM throughput benchmark.",
    )
    args = parser.parse_args()

    result_queue = Queue()

    print(f"Spawning {args.num_gpus} engine subprocesses ...")
    procs = []
    for gpu in range(args.num_gpus):
        p = Process(
            target=engine_worker,
            args=(gpu, args.num_gpus, args.model, args.num_requests, result_queue),
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
