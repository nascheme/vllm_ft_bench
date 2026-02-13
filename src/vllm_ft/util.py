"""Shared utilities for vllm-ft benchmark/generation scripts.

All vllm imports are lazy (inside functions) so that dp_generate.py workers
can set CUDA_VISIBLE_DEVICES before vllm is imported.
"""

import argparse
import os
import sys
import threading
from contextlib import contextmanager

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_NUM_GPUS = 2
DEFAULT_INPUT_LEN = 1024
DEFAULT_OUTPUT_LEN = 128
DEFAULT_DATASET = "ShareGPT_V3_unfiltered_cleaned_split.json"

HARDCODED_PROMPTS = [
    "Hello, my name is",
    "The leader of Canada is",
    "The capital of Russia is",
]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def make_arg_parser(
    description, default_prompt_source="random", default_num_requests=1000
):
    """Return an ArgumentParser with common flags.

    Scripts can add their own flags before calling parser.parse_args().
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to ShareGPT JSON file. Overrides PROMPT_DATASET env var and "
            "the built-in default. Use empty string to force random prompts."
        ),
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        choices=["hardcoded", "random"],
        default=default_prompt_source,
        help=f"Prompt source when no dataset is active (default: {default_prompt_source}).",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=default_num_requests,
        help=f"Number of requests to generate (default: {default_num_requests}).",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=DEFAULT_INPUT_LEN,
        help=f"Input length for random prompts (default: {DEFAULT_INPUT_LEN}).",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=DEFAULT_OUTPUT_LEN,
        help=f"Output length (default: {DEFAULT_OUTPUT_LEN}).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help=f"Number of GPUs (default: {DEFAULT_NUM_GPUS}).",
    )
    return parser


# ---------------------------------------------------------------------------
# Dataset / request building
# ---------------------------------------------------------------------------


def _resolve_dataset(args):
    """Return the dataset path to use, or None for random/hardcoded prompts.

    Resolution order (highest priority first):
      1. ``--dataset`` CLI flag (``args.dataset``).  An explicit empty string
         means "no dataset, use random prompts".
      2. ``PROMPT_DATASET`` environment variable.  An empty string also means
         "no dataset".
      3. Built-in default: ``ShareGPT_V3_unfiltered_cleaned_split.json`` in
         the current working directory.
    """
    if args.dataset is not None:
        # Explicit CLI value — honour it verbatim (empty string → no dataset).
        return args.dataset if args.dataset else None

    env_val = os.environ.get("PROMPT_DATASET")
    if env_val is not None:
        # Env var set — honour it (empty string → no dataset).
        return env_val if env_val else None

    # Fall back to the built-in default path.
    return DEFAULT_DATASET


def build_request_items(args, tokenizer):
    """Build a list of (SampleRequest, SamplingParams) from CLI args.

    Dataset resolution (see ``_resolve_dataset`` for full priority rules):
      1. ``--dataset PATH`` CLI flag overrides everything.
      2. ``PROMPT_DATASET`` env var overrides the default.
      3. Default: ``ShareGPT_V3_unfiltered_cleaned_split.json`` in cwd.
      Set either the env var or ``--dataset`` to an empty string to force
      random/hardcoded prompts.

    Three prompt modes:
      1. ShareGPT (dataset resolved above): per-request max_tokens.
      2. Random (--prompt-source random): fixed output len, ignore_eos.
      3. Hardcoded (--prompt-source hardcoded): 3 short prompts.
    """
    from vllm import SamplingParams
    from vllm.benchmarks.datasets import RandomDataset, ShareGPTDataset, SampleRequest

    dataset_path = _resolve_dataset(args)

    if dataset_path:
        print(
            f"Loading ShareGPT dataset from {dataset_path} "
            f"({args.num_requests} requests) ..."
        )
        dataset = ShareGPTDataset(dataset_path=dataset_path, random_seed=42)
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_requests,
        )
        request_items = [
            (
                req,
                SamplingParams(
                    n=1,
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=req.expected_output_len,
                ),
            )
            for req in requests
        ]
        print(f"ShareGPT dataset ready: {len(requests)} requests")
        return request_items

    if args.prompt_source == "random":
        print(
            f"Generating {args.num_requests} random requests "
            f"(input={args.input_len}, output={args.output_len}) ..."
        )
        dataset = RandomDataset(dataset_path=None, random_seed=42)
        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_requests,
            input_len=args.input_len,
            output_len=args.output_len,
        )
        sp = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.output_len,
        )
        request_items = [(req, sp) for req in requests]
        print(f"Random dataset ready: {len(requests)} requests")
        return request_items

    # Hardcoded prompts.
    sp = SamplingParams(temperature=0.8, top_p=0.95)
    request_items = []
    for prompt_text in HARDCODED_PROMPTS:
        token_ids = tokenizer.encode(prompt_text)
        req = SampleRequest(
            prompt=prompt_text,
            prompt_len=len(token_ids),
            expected_output_len=args.output_len,
        )
        request_items.append((req, sp))
    print(f"Hardcoded prompts ready: {len(request_items)} requests")
    return request_items


# ---------------------------------------------------------------------------
# Forward-context monkey-patch (thread-local storage)
# ---------------------------------------------------------------------------


def apply_forward_context_monkey_patch():
    """Replace vllm.forward_context globals with thread-local equivalents.

    Must be called after vllm is imported but before engine threads start.
    """
    import vllm.forward_context as _fc

    _forward_ctx_tls = threading.local()

    orig_get = _fc.get_forward_context
    orig_is_available = _fc.is_forward_context_available
    orig_override = _fc.override_forward_context

    def _tl_get_forward_context():
        ctx = getattr(_forward_ctx_tls, "ctx", None)
        assert ctx is not None, (
            "Forward context is not set. "
            "Please use `set_forward_context` to set the forward context."
        )
        return ctx

    def _tl_is_forward_context_available():
        return getattr(_forward_ctx_tls, "ctx", None) is not None

    @contextmanager
    def _tl_override_forward_context(forward_context):
        prev = getattr(_forward_ctx_tls, "ctx", None)
        _forward_ctx_tls.ctx = forward_context
        try:
            yield
        finally:
            _forward_ctx_tls.ctx = prev

    patches = [
        ("get_forward_context", orig_get, _tl_get_forward_context),
        (
            "is_forward_context_available",
            orig_is_available,
            _tl_is_forward_context_available,
        ),
        (
            "override_forward_context",
            orig_override,
            _tl_override_forward_context,
        ),
    ]

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for attr, orig, new in patches:
            try:
                if getattr(mod, attr, None) is orig:
                    setattr(mod, attr, new)
            except TypeError, AttributeError:
                pass


# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------


def create_engine(
    engine_args, device_index, usage_context, multiprocess_mode=False, cuda_graphs=False
):
    """Create an LLMEngine pinned to a specific GPU.

    Replicates LLMEngine.from_engine_args() but patches device_config.device
    after create_engine_config(), because DeviceConfig.__post_init__ normalizes
    torch.device("cuda:N") to torch.device("cuda"), stripping the index.

    When multiprocess_mode=False (default), runs EngineCore in-process.
    When multiprocess_mode=True, spawns an EngineCore subprocess per engine
    (via SyncMPClient + ZMQ), giving full async scheduling and pipelining.

    When cuda_graphs=True, re-enables CUDA graph capture after enforce_eager
    disables it. Uses FULL mode (captures entire forward pass as a graph),
    which works without torch.compile. This reduces CUDA driver API calls
    from hundreds per step to one graph replay, mitigating driver lock
    contention in multi-threaded multi-GPU setups.

    When async_scheduling is enabled (in-process mode), also fixes the
    executor's async output thread to set the correct CUDA device.
    """
    import torch
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor.abstract import Executor

    _ROPE_DICT.clear()

    vllm_config = engine_args.create_engine_config(usage_context)
    vllm_config.device_config.device = torch.device(f"cuda:{device_index}")

    # Re-enable CUDA graphs after enforce_eager disabled them.
    # FULL mode captures the entire forward pass as a single CUDA graph,
    # which works without torch.compile (CompilationMode stays NONE).
    # We must also clear enforce_eager on the model_config because the
    # GPU worker checks it again before capture (gpu_worker.py line 502).
    if cuda_graphs:
        from vllm.config import CUDAGraphMode

        vllm_config.model_config.enforce_eager = False
        cc = vllm_config.compilation_config
        cc.cudagraph_mode = CUDAGraphMode.FULL
        # Generate capture sizes (enforce_eager zeroed these out).
        max_seqs = vllm_config.scheduler_config.max_num_seqs
        max_size = min(max_seqs * 2, 512)
        sizes = [i for i in [1, 2, 4] if i <= max_size]
        if max_size >= 8:
            sizes += list(range(8, min(max_size + 1, 256), 8))
        if max_size >= 256:
            sizes += list(range(256, max_size + 1, 16))
        cc.cudagraph_capture_sizes = sizes
        cc.max_cudagraph_capture_size = sizes[-1]
    executor_class = Executor.get_class(vllm_config)
    engine = LLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_stats=False,
        usage_context=usage_context,
        multiprocess_mode=multiprocess_mode,
    )

    # Fix async output thread CUDA device for multi-GPU threading
    # (only relevant for in-process mode).
    if not multiprocess_mode:
        _fix_async_output_thread(engine, device_index)

    return engine


def _fix_async_output_thread(engine, device_index):
    """Replace the executor's async output ThreadPoolExecutor with one
    that sets torch.cuda.set_device() on its worker thread.

    Without this fix, the async output thread calls
    async_copy_ready_event.synchronize() without the correct CUDA device
    context, causing 'CUDA driver error: invalid argument' on multi-GPU.
    """
    from concurrent.futures import ThreadPoolExecutor

    import torch

    executor = engine.engine_core.engine_core.model_executor
    if getattr(executor, "async_output_thread", None) is None:
        return

    # Shut down the old thread pool and replace with a device-aware one.
    executor.async_output_thread.shutdown(wait=False)
    executor.async_output_thread = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=f"WorkerAsyncOutput-cuda{device_index}",
        initializer=lambda: torch.cuda.set_device(device_index),
    )


# ---------------------------------------------------------------------------
# Throughput reporting
# ---------------------------------------------------------------------------


def print_throughput_results(elapsed, engine_stats):
    """Print throughput results in the standard vllm bench format.

    Args:
        elapsed: Wall-clock seconds.
        engine_stats: list of dicts with keys 'completed', 'prompt_tokens',
                      'output_tokens'.
    """
    total_completed = sum(s["completed"] for s in engine_stats)
    total_prompt_tokens = sum(s["prompt_tokens"] for s in engine_stats)
    total_output_tokens = sum(s["output_tokens"] for s in engine_stats)
    total_tokens = total_prompt_tokens + total_output_tokens

    print(
        f"\nThroughput: {total_completed / elapsed:.2f} requests/s, "
        f"{total_tokens / elapsed:.2f} total tokens/s, "
        f"{total_output_tokens / elapsed:.2f} output tokens/s"
    )
    print(f"Total num prompt tokens:  {total_prompt_tokens}")
    print(f"Total num output tokens:  {total_output_tokens}")
    for i, s in enumerate(engine_stats):
        print(
            f"  cuda:{i}: {s['completed']} reqs, "
            f"{s['prompt_tokens']} prompt toks, {s['output_tokens']} output toks"
        )


# ---------------------------------------------------------------------------
# Prompt-length histogram
# ---------------------------------------------------------------------------


def print_prompt_length_histogram(requests):
    """Print a text histogram of prompt token lengths."""
    bins = [
        (1, 8),
        (9, 32),
        (33, 128),
        (129, 256),
        (257, 512),
        (513, 1024),
        (1025, 2048),
        (2049, 4096),
        (4097, float("inf")),
    ]
    counts = [0] * len(bins)
    for req in requests:
        for j, (lo, hi) in enumerate(bins):
            if lo <= req.prompt_len <= hi:
                counts[j] += 1
                break

    total = len(requests)
    max_count = max(counts) if counts else 1
    bar_width = 40

    print("\nPrompt length distribution (tokens):")
    for (lo, hi), count in zip(bins, counts):
        if count == 0:
            continue
        bar = "#" * max(1, int(count / max_count * bar_width))
        hi_str = f"{hi:>5d}" if hi != float("inf") else "  inf"
        print(f"  {lo:>5d} - {hi_str}: {count:5d} ({count / total * 100:5.1f}%) {bar}")
