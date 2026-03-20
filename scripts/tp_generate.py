# Tensor-parallel multi-GPU vLLM throughput benchmark.
#
# Uses vLLM's built-in tensor parallelism to shard a single model across
# multiple GPUs.  Unlike mp_generate.py (which runs independent engines),
# this runs one engine with the model split across GPUs.
#
#   python scripts/tp_generate.py --model meta-llama/Llama-3.2-1B-Instruct --num-gpus 2

import time

from vllm import LLM
from vllm.tokenizers import get_tokenizer

from vllm_ft.util import (
    build_request_items,
    make_arg_parser,
    print_throughput_results,
)


def main():
    parser = make_arg_parser(
        "Tensor-parallel multi-GPU vLLM throughput benchmark.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)
    prompts = [req.prompt for req, _ in request_items]
    sampling_params = [sp for _, sp in request_items]

    if args.cuda_graphs:
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

    print(
        f"Creating LLM with tensor_parallel_size={args.num_gpus}, "
        f"model={args.model}, cuda_graphs={args.cuda_graphs} ..."
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        enforce_eager=not args.cuda_graphs,
        gpu_memory_utilization=0.8,
    )

    print(f"Generating {len(prompts)} requests ...")
    gen_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_elapsed = time.time() - gen_start

    prompt_tokens = 0
    output_tokens = 0
    for ro in outputs:
        if ro.prompt_token_ids:
            prompt_tokens += len(ro.prompt_token_ids)
        output_tokens += sum(len(o.token_ids) for o in ro.outputs if o)

    engine_stats = [
        {
            "completed": len(outputs),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        }
    ]
    print_throughput_results(gen_elapsed, engine_stats)
    print(f"Inference time: {gen_elapsed:.1f}s")


if __name__ == "__main__":
    main()
