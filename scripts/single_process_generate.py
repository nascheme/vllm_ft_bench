# SPDX-License-Identifier: Apache-2.0
#
# Single-process vLLM text generation.
#
# This script does the same thing as simple_generate.py but uses LLMEngine
# directly with multiprocessing disabled.  Everything runs in a single
# process and a single thread:
#
#   InputProcessor (tokenize)
#       -> EngineCore (schedule + execute on GPU)
#           -> OutputProcessor (detokenize)
#
# Key classes (all from vllm, no modifications):
#   LLMEngine       - orchestrates the pipeline
#   InprocClient    - in-process EngineCore wrapper (no ZMQ, no subprocess)
#   UniProcExecutor - runs the GPU worker in-process
#   EngineCore      - scheduler + executor + KV cache management
#   GPUModelRunner  - calls model.forward() on the GPU
#
# This is a stepping stone toward a multi-threaded architecture where
# the scheduling, GPU execution, and output processing can run on
# separate threads communicating via queue.Queue objects.

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time
from vllm import EngineArgs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine

from vllm_ft.util import build_request_items, make_arg_parser, render_request


def main():
    parser = make_arg_parser(
        "Single-process vLLM text generation.",
        default_prompt_source="hardcoded",
        default_num_requests=3,
    )
    args = parser.parse_args()

    # 1. Create engine configuration.
    engine_args = EngineArgs(model=args.model, enforce_eager=True, seed=42)

    # 2. Create LLMEngine with multiprocessing explicitly disabled.
    #    - Uses InprocClient (direct calls) instead of SyncMPClient (ZMQ IPC)
    #    - Uses UniProcExecutor (in-process worker) instead of MultiprocExecutor
    engine = LLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.LLM_CLASS,
        enable_multiprocessing=False,
    )

    request_items = build_request_items(args, engine.tokenizer.tokenizer)
    # Use the sampling params from the first item.
    sampling_params = request_items[0][1]

    start_time = time.time()
    duration = 30  # seconds
    num_prompts = 0

    while True:
        # 3. Add requests via Renderer API.
        prompts = [req.prompt for req, _ in request_items]
        renderer = engine.renderer
        for i, prompt in enumerate(prompts * 100):
            proc_input = render_request(renderer, prompt)
            engine.add_request(str(i), proc_input, sampling_params)

        # 4. Step loop -- all in-process, single-threaded.
        #    Each step() call does:
        #      Scheduler.schedule()              -> pick requests, allocate KV blocks
        #      UniProcExecutor.execute_model()   -> GPUModelRunner.execute_model()
        #                                           -> model.forward() on GPU
        #                                           -> Sampler.sample()
        #      Scheduler.update_from_output()    -> append tokens, check stop
        #      OutputProcessor.process_outputs() -> detokenize to text
        while engine.has_unfinished_requests():
            request_outputs = engine.step()
            for output in request_outputs:
                if output.finished:
                    print(
                        f"Prompt: {output.prompt!r}, "
                        f"Generated text: {output.outputs[0].text!r}"
                    )
                num_prompts += 1

        elapsed = time.time() - start_time
        rate = num_prompts / elapsed
        print(f"Did {rate:.1f} prompts/sec")
        if elapsed > duration:
            break  # run for long enough


if __name__ == "__main__":
    main()
