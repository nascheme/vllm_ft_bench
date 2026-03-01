# Minimal LLMEngine step loop benchmark.
#
# Compares our create_engine() + manual step loop against LLM.generate()
# to isolate the performance difference. No threading, no queues.
#
# Usage:
#   python bench_step_loop.py --mode engine   # our LLMEngine path
#   python bench_step_loop.py --mode llm      # LLM.generate() path

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import time

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


def run_engine_mode(args, request_items):
    """Our create_engine() + manual step loop."""
    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    engine = create_engine(engine_args, 0, UsageContext.LLM_CLASS)

    # Add all requests up front (same as LLM.generate).
    renderer = engine.renderer
    for i, (req, sp) in enumerate(request_items):
        proc_input = render_request(renderer, req.prompt)
        engine.add_request(str(i), proc_input, sp)

    # Step loop — matches LLM._run_engine() exactly.
    completed = 0
    t0 = time.time()
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for o in outputs:
            if o.finished:
                completed += 1
    elapsed = time.time() - t0
    print(
        f"engine mode: {completed} reqs in {elapsed:.1f}s = "
        f"{completed / elapsed:.1f} req/s"
    )


def run_llm_mode(args, request_items):
    """LLM.generate() — the reference."""
    from vllm import LLM

    llm = LLM(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    )
    prompts = [req.prompt for req, _ in request_items]
    # Use per-request sampling params for ShareGPT compatibility.
    params = [sp for _, sp in request_items]

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    print(
        f"llm mode: {len(outputs)} reqs in {elapsed:.1f}s = "
        f"{len(outputs) / elapsed:.1f} req/s"
    )


def main():
    parser = make_arg_parser(
        "Minimal step loop benchmark: engine vs LLM.generate()",
        default_num_requests=500,
    )
    parser.add_argument(
        "--mode",
        choices=["engine", "llm"],
        required=True,
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    if args.mode == "engine":
        run_engine_mode(args, request_items)
    else:
        run_llm_mode(args, request_items)


if __name__ == "__main__":
    main()
