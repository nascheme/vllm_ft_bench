# Script Inventory

All scripts live in `scripts/`. They are standalone entrypoints run directly
with Python 3.14t (free-threaded build). The shared utility package is
`src/vllm_ft/util.py`.

Hardware assumption: 2x RTX 2060 (6 GB each). All scripts use TP=1/PP=1.

---

## Baseline / Reference Scripts

**`simple_generate.py`**
Simplest baseline. Uses the high-level `vllm.LLM.generate()` API on a single
GPU. Starting point before any threading experiments.

**`single_process_generate.py`**
Like `simple_generate.py` but uses `LLMEngine` directly with multiprocessing
disabled (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), running everything in one thread.
Stepping stone toward the threaded designs; makes the step loop explicit.

**`mp_generate.py`**
Multi-process baseline. Spawns one process per GPU via `CUDA_VISIBLE_DEVICES`
isolation, each running `LLM.generate()`. The standard DP approach that
threaded scripts aim to match.

**`mp_static_generate.py`**
Same as `mp_generate.py` but with static round-robin request partitioning
(pre-split before any process starts). Removes load-balancing as a variable
when comparing threaded vs. process isolation.

**`mp_engine_generate.py`**
Like `mp_static_generate.py` but uses `create_engine()` + a manual preload
step loop instead of `LLM.generate()`, and sets
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. Supports `--cuda-graphs`. This makes it
the apples-to-apples multi-process baseline for the threaded scripts, which
use the same methodology. Comparing threaded results against this script
(rather than `mp_static_generate.py`) isolates threading overhead from the
`LLM.generate()` / `VLLM_ENABLE_V1_MULTIPROCESSING` difference.

---

## Core Threaded Inference Scripts

**`threaded_generate.py`**
First threaded dual-engine design. Two `LLMEngine` instances (one per GPU) on
separate threads pulling from a **shared** tokenized-request queue. GIL release
during GPU work naturally load-balances between threads.

**`threaded_static_generate.py`** ← *primary benchmark script*
Dual-engine threaded with **static** round-robin request partitioning. Supports
`--preload` to add all requests before stepping (matches `LLM.generate()`
behavior). Supports `--cuda-graphs` to enable CUDA graph capture. Use this for
apples-to-apples comparison against `mp_static_generate.py`.

**`threaded_dispatch_generate.py`**
Improved dual-engine threading. Adds a smart dispatcher routing requests to
**per-engine queues** based on KV cache usage (round-robin when idle,
lowest-usage-first when busy).

**`threaded_pull_generate.py`**
Pull-based dual-engine design. Each engine thread self-throttles by monitoring
its own in-flight count and KV cache usage before pulling from a shared queue.

**`dispatcher_generate.py`**
Most structured threaded design. Wraps engines in a `Dispatcher` class with a
`submit() → Future` API and explicit least-load routing via per-engine queues.

**`threaded_llm_generate.py`**
Threaded dual-engine benchmark that preloads all requests before stepping,
mirroring `LLM.generate()` internals. Useful for isolating the thread-vs-process
difference from the streaming-vs-batch difference.

**`threaded_mp_generate.py`**
Hybrid: thread-based coordination but each engine's GPU work runs in a
subprocess (`multiprocess_mode=True`, ZMQ). Gets process-level GPU isolation
with thread-level coordination convenience.

---

## Diagnostic / Profiling Scripts

**`bench_step_loop.py`**
Compares `create_engine()` + manual `step()` loop against `LLM.generate()`.
Single GPU, no threading. Isolates the overhead of the custom step loop itself.

**`threaded_instrumented.py`**
`threaded_generate.py` with heavy per-step diagnostics: per-phase timing, overlap
detection between the two engine threads, tail-drain analysis, step-time
percentiles.

**`threaded_step_breakdown.py`**
Monkey-patches vLLM internals to time individual sub-components of
`engine.step()` (scheduler, model execution, output processing). Runs
single-GPU then dual-GPU in the same process to pinpoint where threading
overhead accumulates.

**`threaded_profile_generate.py`**
Runs single-GPU then dual-GPU threaded phases in subprocesses (for clean GPU
memory), measuring GPU time via CUDA events vs wall time per step.

**`mp_profile_generate.py`**
Same CUDA event profiling as `threaded_profile_generate.py` but using
subprocesses, for direct step-time distribution comparison.

**`threaded_scaling_test.py`**
Runs three configurations back-to-back in the same process (cuda:0 alone,
cuda:1 alone, both threaded) to directly measure the overhead from running two
engines together vs each solo.

**`threaded_pipelined_generate.py`**
Explores a pipelined output-processing architecture; times output
post-processing separately from the GPU step.

---

## Contention / Microbenchmark Scripts

**`cuda_contention_bench.py`**
Pure PyTorch (no vLLM) matmul loop on two GPUs. Three modes: single GPU, two
GPUs threaded, two GPUs in subprocesses. Baseline measurement of CUDA driver
lock contention between threads.

**`cuda_contention_bench2.py`**
More realistic CUDA contention benchmark. Simulates a transformer decode step
(attention projections, MLP, norms, token sampling) to stress the same
kernel-launch patterns as a real vLLM step.

**`cuda_alloc_bench.py`**
Tests whether PyTorch's CUDA caching allocator causes cross-thread contention.
Three allocation modes (static, dynamic, heavy) simulate varying tensor
alloc/free churn.

**`python_contention_bench.py`**
Benchmarks Python-level overhead in free-threaded mode. Mixes realistic
vLLM-like Python work (scheduler dicts, request objects, output processing)
with CUDA ops to isolate Python object manipulation as a contention source.

**`cudagraph_test.py`**
Tests CUDA graph capture with `enforce_eager=True` on free-threaded Python
3.14t, then benchmarks single vs. threaded to confirm driver-call reduction.

---

## vLLM Thread-Safety Workarounds (in `util.py`)

All multi-engine scripts depend on these fixes applied in `util.py`. They patch
around vLLM assumptions that one process = one engine:

| Issue | Fix |
|---|---|
| `DeviceConfig.__post_init__` strips `cuda:N` → `cuda` | Patch `device_config.device` after `create_engine_config()` |
| `_forward_context` is a module-level global (not thread-safe) | Monkey-patch to `threading.local()` via `apply_forward_context_monkey_patch()` |
| Async output thread has no CUDA device set | Wrap executor's `ThreadPoolExecutor` to call `torch.cuda.set_device()` on worker thread |
| `_ROPE_DICT` cache shares tensors across GPUs | Call `_ROPE_DICT.clear()` before each engine init |
