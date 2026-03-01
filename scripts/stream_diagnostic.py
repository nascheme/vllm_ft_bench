"""Diagnostic: verify each engine thread gets its own CUDA stream.

Creates two LLMEngines (sequentially on main thread) then runs them on
separate threads, logging:
  - torch.cuda.current_stream() identity from each thread
  - vllm.utils.torch_utils.current_stream() identity from each thread
  - Stream pointers during engine.step()

Usage:
  python scripts/stream_diagnostic.py
  python scripts/stream_diagnostic.py --model <model>
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import threading

import torch
from vllm import EngineArgs
from vllm.tokenizers import get_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils.torch_utils import current_stream as vllm_current_stream

from vllm_ft.util import (
    apply_forward_context_monkey_patch,
    build_request_items,
    create_engine,
    make_arg_parser,
)

apply_forward_context_monkey_patch()

# ---------------------------------------------------------------------------

print_lock = threading.Lock()


def log(device_index, msg):
    with print_lock:
        print(f"  [cuda:{device_index} tid={threading.current_thread().name}] {msg}")


def stream_info(stream):
    """Return a compact string identifying a CUDA stream."""
    return f"Stream(id={stream.stream_id}, ptr={stream.cuda_stream:#x})"


def step_worker(device_index, engine, request_items, results):
    """Run on a thread: log stream identity and run a few steps."""
    torch.cuda.set_device(device_index)

    # Check streams — vLLM's current_stream() lazily creates a dedicated
    # stream per thread (via threading.local)
    pytorch_stream = torch.cuda.current_stream()
    log(device_index, f"torch.cuda.current_stream() = {stream_info(pytorch_stream)}")

    vllm_stream = vllm_current_stream()
    log(device_index, f"vllm current_stream()        = {stream_info(vllm_stream)}")

    pytorch_after = torch.cuda.current_stream()
    log(device_index, f"torch current_stream (after)  = {stream_info(pytorch_after)}")

    # Add a few requests and run steps
    input_processor = engine.input_processor
    supported_tasks = engine.get_supported_tasks()
    for i, (req, sp) in enumerate(request_items[:4]):
        ecr = input_processor.process_inputs(
            str(i), req.prompt, sp,
            arrival_time=0.0, supported_tasks=supported_tasks,
        )
        engine.add_request(ecr.request_id, ecr, sp, prompt_text=req.prompt)

    step_streams = []
    step_count = 0
    while engine.has_unfinished_requests() and step_count < 100:
        s = torch.cuda.current_stream()
        step_streams.append(s.cuda_stream)
        engine.step()
        step_count += 1

    log(device_index, f"Ran {step_count} steps")

    unique_ptrs = set(step_streams)
    log(device_index, f"Unique stream pointers during steps: {len(unique_ptrs)}")
    for ptr in unique_ptrs:
        log(device_index, f"  {ptr:#x}")

    results[device_index] = {
        "vllm_stream_ptr": vllm_stream.cuda_stream,
        "pytorch_stream_ptr": pytorch_after.cuda_stream,
        "step_stream_ptrs": unique_ptrs,
    }


def main():
    parser = make_arg_parser("CUDA stream diagnostic")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"Stream diagnostic — {num_gpus} GPU(s)")

    engine_args = EngineArgs(
        model=args.model,
        enforce_eager=True,
    )

    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)

    if num_gpus < 2:
        print("  Need 2 GPUs, exiting")
        return

    # Create engines sequentially on main thread (parallel state init
    # requires this — same as all other threaded scripts)
    engines = []
    for dev in range(2):
        print(f"Creating engine on cuda:{dev} ...")
        engines.append(create_engine(engine_args, dev, UsageContext.LLM_CLASS))

    # Log main thread streams for comparison
    print("\n  --- Main Thread Stream ---")
    main_pytorch = torch.cuda.current_stream()
    print(f"  torch.cuda.current_stream() = {stream_info(main_pytorch)}")
    main_vllm = vllm_current_stream()
    print(f"  vllm current_stream()        = {stream_info(main_vllm)}")

    # Run engines on separate threads
    print(f"\n{'=' * 60}")
    print("Stepping two engines on separate threads")
    print(f"{'=' * 60}")

    results = {}
    threads = []
    for dev in range(2):
        t = threading.Thread(
            target=step_worker,
            args=(dev, engines[dev], request_items, results),
            name=f"engine-{dev}",
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n  --- Stream Comparison ---")
    r0, r1 = results[0], results[1]
    same_vllm = r0["vllm_stream_ptr"] == r1["vllm_stream_ptr"]
    same_pytorch = r0["pytorch_stream_ptr"] == r1["pytorch_stream_ptr"]
    print(
        f"  vLLM streams same?    {same_vllm}  "
        f"(0: {r0['vllm_stream_ptr']:#x}, 1: {r1['vllm_stream_ptr']:#x})"
    )
    print(
        f"  PyTorch streams same? {same_pytorch}  "
        f"(0: {r0['pytorch_stream_ptr']:#x}, 1: {r1['pytorch_stream_ptr']:#x})"
    )
    overlap = r0["step_stream_ptrs"] & r1["step_stream_ptrs"]
    print(
        f"  Step stream overlap?  {bool(overlap)}  "
        f"(shared ptrs: {[f'{p:#x}' for p in overlap] if overlap else 'none'})"
    )

    print()


if __name__ == "__main__":
    main()
