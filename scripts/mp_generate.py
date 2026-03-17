# Multi-process multi-GPU vLLM throughput benchmark.
#
# Each GPU runs in a separate process with its own LLM instance, isolated
# via CUDA_VISIBLE_DEVICES.  No vLLM data-parallel machinery is used
# (vLLM DP requires MoE models); this simply runs independent single-GPU
# engines in parallel — the standard multi-process approach.
#
# Comparable to threaded_generate.py for benchmarking threads vs processes.
#
#   Parent process                Worker 0 (cuda:0)          Worker 1 (cuda:1)
#     generate dataset            LLM (single GPU)           LLM (single GPU)
#     spawn workers               llm.generate(my_half)      llm.generate(my_half)
#     collect results             → result_queue              → result_queue
#     report throughput

import time
from multiprocessing import Process, Queue

from vllm_ft.util import make_arg_parser, print_throughput_results


def worker(
    gpu_index,
    num_workers,
    model,
    num_requests,
    input_len,
    output_len,
    prompt_source,
    dataset,
    result_queue,
    cuda_graphs=False,
):
    """Worker process: create one LLM on a single GPU, generate, report."""
    import argparse
    import os

    # Isolate this process to a single GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    from vllm.tokenizers import get_tokenizer

    from vllm import LLM
    from vllm_ft.util import build_request_items

    # Build the same dataset in every worker (same seed -> same data),
    # then take this worker's slice.
    worker_args = argparse.Namespace(
        model=model,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        prompt_source=prompt_source,
        dataset=dataset,
    )
    tokenizer = get_tokenizer(model)
    all_request_items = build_request_items(worker_args, tokenizer)

    # Static partition: each worker takes its slice.
    total = len(all_request_items)
    chunk = total // num_workers
    start_idx = gpu_index * chunk
    end_idx = start_idx + chunk if gpu_index < num_workers - 1 else total
    my_request_items = all_request_items[start_idx:end_idx]
    prompts = [req.prompt for req, _ in my_request_items]
    sampling_params = [sp for _, sp in my_request_items]

    if cuda_graphs:
        # Patch EngineArgs.create_engine_config to enable CUDAGraphMode.FULL
        # before LLM.__init__ triggers engine + GPU worker setup.  This mirrors
        # what create_engine(cuda_graphs=True) does for the LLMEngine path.
        from vllm.config import CUDAGraphMode

        from vllm import EngineArgs

        _orig_create_engine_config = EngineArgs.create_engine_config

        def _patched_create_engine_config(self, usage_context=None):
            cfg = _orig_create_engine_config(self, usage_context)
            cfg.model_config.enforce_eager = False
            cc = cfg.compilation_config
            cc.cudagraph_mode = CUDAGraphMode.FULL
            max_seqs = cfg.scheduler_config.max_num_seqs
            max_size = min(max_seqs * 2, 512)
            sizes = [i for i in [1, 2, 4] if i <= max_size]
            if max_size >= 8:
                sizes += list(range(8, min(max_size + 1, 256), 8))
            if max_size >= 256:
                sizes += list(range(256, max_size + 1, 16))
            cc.cudagraph_capture_sizes = sizes
            cc.max_cudagraph_capture_size = sizes[-1]
            return cfg

        EngineArgs.create_engine_config = _patched_create_engine_config

    llm = LLM(
        model=model,
        enforce_eager=not cuda_graphs,
        gpu_memory_utilization=0.8,
    )

    # Time only the generate() call for fair comparison with threaded script.
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
            "gpu_index": gpu_index,
            "num_completed": len(outputs),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "gen_start": gen_start,
            "gen_end": gen_end,
        }
    )


def main():
    parser = make_arg_parser(
        "Multi-process multi-GPU vLLM throughput benchmark.",
    )
    args = parser.parse_args()

    result_queue = Queue()

    print(
        f"Starting {args.num_gpus} workers, {args.num_requests} requests "
        f"(input={args.input_len}, output={args.output_len}, "
        f"cuda_graphs={args.cuda_graphs}) ..."
    )

    wall_start = time.time()

    procs = []
    for gpu in range(args.num_gpus):
        p = Process(
            target=worker,
            args=(
                gpu,
                args.num_gpus,
                args.model,
                args.num_requests,
                args.input_len,
                args.output_len,
                args.prompt_source,
                args.dataset,
                result_queue,
                args.cuda_graphs,
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

    wall_elapsed = time.time() - wall_start

    # Collect results from all workers.
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    results.sort(key=lambda r: r["gpu_index"])

    if not results:
        print("No results collected — workers may have failed.")
        return

    # Inference time: from earliest generate() start to latest generate() end.
    gen_start = min(r["gen_start"] for r in results)
    gen_end = max(r["gen_end"] for r in results)
    gen_elapsed = gen_end - gen_start

    engine_stats = [
        {
            "completed": r["num_completed"],
            "prompt_tokens": r["prompt_tokens"],
            "output_tokens": r["output_tokens"],
        }
        for r in results
    ]
    print_throughput_results(gen_elapsed, engine_stats)

    # Extra dp-specific reporting.
    print(
        f"Inference time: {gen_elapsed:.1f}s  "
        f"(wall clock incl. setup: {wall_elapsed:.1f}s)"
    )
    for r in results:
        rank_elapsed = r["gen_end"] - r["gen_start"]
        print(
            f"  cuda:{r['gpu_index']}: {r['num_completed']} reqs, "
            f"{r['prompt_tokens']} prompt toks, {r['output_tokens']} output toks, "
            f"{rank_elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
