# Dual-engine multi-GPU threaded vLLM throughput benchmark.
#
# Architecture: two independent LLMEngine instances (one per GPU) fed from
# a shared tokenized-request queue, with a single tokenizer thread.
#
#   Tokenizer Thread (CPU)         Engine Thread 0 (cuda:0)   Engine Thread 1 (cuda:1)
#     input_processor.process()    add_request (from queue)   add_request (from queue)
#     tokenized_queue.put(ecr)     engine0.step()             engine1.step()
#                                  (continuous streaming)      (continuous streaming)
#
# Each engine thread continuously pulls requests from the shared queue and
# steps the engine.  The engine's internal scheduler handles batching.
# The shared queue naturally load-balances between engine threads because
# each step() releases the GIL for GPU work, giving the other thread
# time to pull from the queue.
#
# For TP=1/PP=1, parallel-state init is idempotent so both engines
# coexist in one process with zero vLLM source modifications.
#
# Uses vllm.benchmarks.datasets.RandomDataset for benchmark-comparable
# prompts, and reports tokens/sec metrics matching `vllm bench throughput`.

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import queue
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
)

apply_forward_context_monkey_patch()

# ---------------------------------------------------------------------------

# Max requests to pull from the shared queue per engine step.  Keeps the
# pull small enough that both engine threads get a fair share, while still
# giving the scheduler enough requests to form efficient batches.
MAX_PULL_PER_STEP = 8


def tokenizer_worker(
    input_processor, supported_tasks, request_items, tokenized_queue, done_event
):
    """Background thread: tokenize SampleRequests into EngineCoreRequests.

    request_items is a list of (SampleRequest, SamplingParams) tuples,
    allowing per-request sampling params (e.g. variable max_tokens for
    ShareGPT).
    """
    for i, (req, sp) in enumerate(request_items):
        ecr = input_processor.process_inputs(
            str(i),
            req.prompt,
            sp,
            arrival_time=time.time(),
            supported_tasks=supported_tasks,
        )
        tokenized_queue.put((ecr, req.prompt, sp))
    done_event.set()


def engine_worker(engine, device_index, tokenized_queue, tok_done, stats):
    """Engine thread: continuously pull requests and step the engine.

    Instead of collecting a full batch and processing it to completion,
    this streams requests into the engine a few at a time.  The engine's
    internal scheduler forms optimal GPU batches.  Between step() calls
    the GIL is released for CUDA work, letting the other engine thread
    pull from the shared queue — naturally balancing load.
    """
    torch.cuda.set_device(device_index)

    while True:
        # Pull a limited number of requests from the shared queue.
        for _ in range(MAX_PULL_PER_STEP):
            try:
                ecr, prompt_text, sp = tokenized_queue.get_nowait()
                engine.add_request(
                    ecr.request_id,
                    ecr,
                    sp,
                    prompt_text=prompt_text,
                )
            except queue.Empty:
                break

        if engine.has_unfinished_requests():
            request_outputs = engine.step()
            for output in request_outputs:
                if output.finished:
                    stats[0] += 1
                    if output.prompt_token_ids:
                        stats[1] += len(output.prompt_token_ids)
                    stats[2] += sum(len(o.token_ids) for o in output.outputs if o)
        elif tok_done.is_set() and tokenized_queue.empty():
            break
        else:
            # No work yet — block briefly for the tokenizer to catch up.
            try:
                ecr, prompt_text, sp = tokenized_queue.get(timeout=0.5)
                engine.add_request(
                    ecr.request_id,
                    ecr,
                    sp,
                    prompt_text=prompt_text,
                )
            except queue.Empty:
                if tok_done.is_set():
                    break


def main():
    parser = make_arg_parser(
        "Dual-engine multi-GPU threaded vLLM throughput benchmark.",
    )
    args = parser.parse_args()

    # 1. Generate dataset (CPU, before engine creation).
    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)
    num_requests = len(request_items)

    # 2. Create engines sequentially (parallel state init is idempotent
    #    for TP=1, PP=1 — second engine reuses the already-initialized state).
    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        async_scheduling=False,
    )
    engines = []
    for i in range(args.num_gpus):
        print(f"Creating engine on cuda:{i} ...")
        engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))
    print(f"All {args.num_gpus} engines created.")

    # 3. Start tokenizer thread.
    tokenized_queue = queue.Queue(maxsize=num_requests)
    tok_done = threading.Event()

    tok_thread = threading.Thread(
        target=tokenizer_worker,
        args=(
            engines[0].input_processor,
            engines[0].get_supported_tasks(),
            request_items,
            tokenized_queue,
            tok_done,
        ),
        name="LLM::tok",
    )
    tok_thread.start()

    # 4. Start engine threads — one per GPU.
    # stats per engine: [num_completed, prompt_tokens, output_tokens]
    stats = [[0, 0, 0] for _ in range(args.num_gpus)]
    engine_threads = []
    for i, engine in enumerate(engines):
        t = threading.Thread(
            target=engine_worker,
            args=(
                engine,
                i,
                tokenized_queue,
                tok_done,
                stats[i],
            ),
            name=f"LLM::engine{i}",
        )
        t.start()
        engine_threads.append(t)

    # 5. Wait for completion with progress updates.
    start_time = time.time()
    while any(t.is_alive() for t in engine_threads):
        time.sleep(2)
        elapsed = time.time() - start_time
        total = sum(s[0] for s in stats)
        print(f"  [{elapsed:.1f}s] {total}/{num_requests} completed ...")

    tok_thread.join()
    for t in engine_threads:
        t.join()

    # 6. Report results (matches `vllm bench throughput` format).
    elapsed = time.time() - start_time

    engine_stats = [
        {"completed": s[0], "prompt_tokens": s[1], "output_tokens": s[2]} for s in stats
    ]
    print_throughput_results(elapsed, engine_stats)


if __name__ == "__main__":
    main()
