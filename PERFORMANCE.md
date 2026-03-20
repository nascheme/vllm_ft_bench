# Performance Analysis

Benchmarks comparing single-process threaded vLLM engines against the
multi-process baseline.

**Hardware:** 2x RTX 2060 (6 GB each), 64 CPU cores, Linux VM
**Model:** SmolLM2-360M-Instruct
**Dataset:** ShareGPT (~1000 requests, variable prompt/output lengths)
**Python:** CPython 3.14t (free-threaded build)

---

## Headline Numbers

Two multi-process baselines are needed for a fair comparison.
`mp_static_generate.py` uses `LLM.generate()` with `VLLM_ENABLE_V1_MULTIPROCESSING=1`
(EngineCore in a dedicated child subprocess) and represents the practical
performance ceiling. `mp_engine_generate.py` uses the same `create_engine()` +
manual step loop + `VLLM_ENABLE_V1_MULTIPROCESSING=0` as the threaded scripts,
but in separate processes — the apples-to-apples baseline for threading overhead.

| Configuration | Script | Throughput | vs. mp_static | vs. mp_engine |
|---|---|---|---|---|
| Single GPU | simple_generate.py | ~11.7 req/s | — | — |
| **mp_static** (LLM.generate, V1_MP=1) | mp_static_generate.py | 23.54 req/s | 1.00× | — |
| mp_engine, no CUDA graphs (step loop, V1_MP=0) | mp_engine_generate.py | 21.29 req/s | 0.91× | 1.00× |
| mp_engine + CUDA graphs | mp_engine_generate.py --cuda-graphs | 22.72 req/s | 0.97× | 1.07× |
| **Threaded, no CUDA graphs** (step loop, V1_MP=0) | threaded_static_generate.py --preload | 20.82 req/s | 0.88× | **0.98×** |
| **Threaded + CUDA graphs** | threaded_pipelined_generate.py --cuda-graphs | 22.79 req/s | 0.97× | **1.00×** |

Threaded + CUDA graphs achieves **parity with the equivalent multi-process
configuration** (`mp_engine_generate.py`). The remaining ~3% gap to `mp_static`
is not from threading — it exists identically between `mp_static` and
`mp_engine` (both in separate processes) and is attributable to `LLM.generate()`
internals and `VLLM_ENABLE_V1_MULTIPROCESSING=1`. See Open Questions.

### Step-time breakdown (p50, steady-state decode)

Results from `threaded_step_breakdown.py`. Three columns show the asymmetry
between the two engine threads: cuda:0 draws larger batches (p50 batch≈20),
cuda:1 draws smaller ones (p50 batch≈7).

```
Component           Single GPU   Threaded     Threaded     Ratio (single
                    (alone)      cuda:0       cuda:1       vs. t-cuda:0)
schedule              0.18ms       0.30ms       0.10ms       1.70×
execute_model        13.93ms      17.02ms      15.92ms       1.22×
update_from_output    0.12ms       0.23ms       0.08ms       1.83×
process_outputs       0.14ms      0.23ms        0.07ms       1.59×
other (gap)           7.68ms      14.99ms       0.39ms       1.95×  ←
total step           22.74ms      31.57ms      17.81ms       1.39×
```

The `execute_model` overhead is modest (1.22×). The dominant new cost is
`other (gap)` — time not covered by the four named components.
`threaded_gap_breakdown.py` (see also TIMING.md Phase 1) identified that
this gap is primarily `sample_tokens()`: GPU synchronization via
`_bookkeeping_sync()` (which calls `async_copy_ready_event.synchronize()` to
wait for the GPU→CPU copy stream) plus minor contributions from
`get_grammar_bitmask`, batch-queue management, and `_process_aborts_queue`.
The gap nearly doubles for cuda:0 (+7.32ms) but is essentially unchanged for
cuda:1 (0.39ms). This asymmetry tracks batch size: larger batches produce
more GPU work before the copy stream can complete, extending the sync wait.
The net throughput impact is small (~2%) because cuda:1 drives wall time and
cuda:1 is not penalized.

---

## What Has Been Ruled Out

### CPU scheduling overhead
`scheduler.schedule()` + `scheduler.update_from_output()` account for only
**~1.3%** of total step time. Even doubling their cost would not explain the
observed gap.

### Async scheduling / batch pipelining
`async_scheduling=True` is the **default** (it sets `max_concurrent_batches=2`,
activating `step_with_batch_queue` instead of `step`). Toggling it off
(`async_scheduling=False`) does not change throughput — GPU execution time
dominates regardless.

### Batch formation / request preloading
Preloading all requests before stepping (matching `LLM.generate()` behavior)
gives the same throughput as streaming requests via a tokenizer thread. The
scheduler always has enough requests to form efficient batches.

### Load balancing / tail drain
The shared-queue design caused severe tail drain (last 10% of requests taking
46% of wall time) due to workload imbalance. Static partitioning eliminates
this, but the per-engine slowdown remains.

### Engine configuration overhead
`create_engine()` + manual `engine.step()` loop produces identical single-GPU
throughput to `LLM.generate()` (~11 req/s). The step loop itself adds no cost.

### Multiprocess mode within the engine (single-GPU)
`VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process EngineCore) vs. default
multiprocess mode gives the same single-GPU throughput. However, a ~10%
dual-GPU gap persists between `mp_static_generate.py` (23.54 req/s, V1_MP=1)
and `mp_engine_generate.py` (21.29 req/s, V1_MP=0) even though both run in
separate processes with no threading involved. The cause is almost certainly
`VLLM_ENABLE_V1_MULTIPROCESSING=1`: when the EngineCore runs in a dedicated
child subprocess, its asyncio event loop gets its own CPU and is never
preempted by the step-loop caller. With `=0`, the event loop shares the
process with the step loop, slowing the internal round-trips visible as
`other (gap)` in the step-time table. See Open Questions.

### GIL vs. free-threaded Python (for engine threads)
Testing with both standard (GIL) and free-threaded (`--disable-gil`) builds
produced essentially identical throughput. PyTorch already releases the GIL
during CUDA kernel launches, so the GPU-bound workload sees no benefit from
GIL removal in the engine threads themselves.

### Threading itself
When compared apples-to-apples against `mp_engine_generate.py` (same step-loop
architecture, same `VLLM_ENABLE_V1_MULTIPROCESSING=0`, but separate OS
processes), threading adds ≤2% overhead without CUDA graphs and is
indistinguishable from zero with CUDA graphs. Earlier analysis overstated the
threading penalty because it compared the threaded step-loop against
`mp_static_generate.py`, which uses a different methodology on two axes
simultaneously (`LLM.generate()` and `VLLM_ENABLE_V1_MULTIPROCESSING=1`).

### CUDA stream sharing between engine threads
Each engine thread already gets its own dedicated CUDA stream via vLLM's
`current_stream()` (`vllm/utils/torch_utils.py`), which uses
`threading.local()` to lazily create a new `torch.cuda.Stream()` per thread.
Verified with `stream_diagnostic.py`: two engine threads showed different
stream pointers, consistent across 100 steps, with no overlap. No
monkey-patching needed. CUDA graph capture also uses the per-thread stream.

### Sub-batch splitting across streams
Phase D of `cuda_pipeline_bench.py` tested splitting a fixed batch across
multiple CUDA streams sharing the same weights. Throughput degrades monotonically
(1.00x → 0.31x at 8 splits) because the GPU's matmul hardware already
parallelizes across the batch dimension internally. Multi-stream gains only
exist when each stream brings genuinely new work (separate engines), not when
redistributing a fixed workload.

---

## CUDA Graphs

CUDA graphs provide a consistent ~7% throughput improvement in both threaded
and multi-process configurations:

```
mp_engine, no CUDA graphs:  21.29 req/s
mp_engine + CUDA graphs:    22.72 req/s  (+6.7%)

Threaded, no CUDA graphs:   20.82 req/s
Threaded + CUDA graphs:     22.79 req/s  (+9.5%)
```

The benefit is symmetric — CUDA graphs are not a threading-specific fix.
They reduce step-time variance by collapsing per-step kernel launches into a
single `cudaGraphLaunch`, which cuts the high-end tail (p90/p99 step times)
and lets the two engines finish more closely in time. Since wall time is
determined by the slower engine, tightening the tail improves throughput even
when mean step time barely changes.

Enabling CUDA graphs in `create_engine(cuda_graphs=True)` uses `FULL` mode
(captures the entire forward pass as a single graph), which works without
`torch.compile`. The `enforce_eager=True` flag still suppresses triton/compile;
only CUDA graph capture mode is changed.

Note: CUDA graphs are benchmarked via `threaded_pipelined_generate.py
--cuda-graphs` (which pipelines output processing separately from the GPU
step) rather than `threaded_static_generate.py`, which does not have a
`--cuda-graphs` flag.

---

## Open Questions

Threading overhead is now established as negligible (≤2% vs. equivalent
multi-process). The open questions focus on what accounts for the ~10% gap
between `mp_static` and the `create_engine` step-loop approach, and on
longer-term architecture.

### 1. Why is `LLM.generate()` + `VLLM_ENABLE_V1_MULTIPROCESSING=1` ~10% faster?

`mp_static_generate.py` (23.54 req/s) vs. `mp_engine_generate.py` (21.29 req/s)
both run in separate processes with no threading, but differ in methodology:
`LLM.generate()` vs. `create_engine()` + step loop, and `V1_MP=1` vs. `=0`.

Most likely cause: with `V1_MP=1`, the EngineCore asyncio event loop runs in
a dedicated child subprocess and is never preempted by the calling process.
With `V1_MP=0`, the event loop shares the process with the step loop, producing
the `other (gap)` overhead visible in the step-time breakdown — 7.68ms per step
even in single-GPU mode. With `V1_MP=1` this gap would likely shrink or
disappear, recovering most of the 10% difference.

**How to investigate:** Profile with `V1_MP=1` inside a threaded context once
the thread-safety issues are resolved; measure whether `other (gap)` drops.
Alternatively, run `threaded_step_breakdown.py` in a subprocess to isolate
whether the gap is an in-process effect.

### 2. Biased reference counting in free-threaded Python

Free-threaded Python 3.14t uses *biased reference counting*: objects are
assumed to be owned by one thread. When a thread other than the "owner" touches
an object's refcount, it requires atomic operations and can cause cache-line
contention. With threading overhead at ~2% overall this is likely a second-order
effect, but it may account for part of that residual.

**How to investigate:** Profile with `perf record` looking for cache-miss
hotspots in refcount operations.

### 3. CPU cache thrashing

Two engine threads on different CPU cores competing for L3 cache. Each engine's
working set (scheduler state, KV cache metadata, Python interpreter state)
evicts the other's data, causing higher cache miss rates.

**How to investigate:** Pin engine threads to specific CPU cores/NUMA nodes
with `os.sched_setaffinity()` and compare.

### 4. PyTorch allocator and internal state

PyTorch's CUDA caching allocator is per-device but process-wide. Internal
PyTorch state (autograd bookkeeping, cuDNN handle management) may also have
subtle serialization points.

**How to investigate:** `torch.cuda.memory_stats()` for allocator contention;
try `PYTORCH_NO_CUDA_MEMORY_CACHING=1` to isolate allocator effects.

---

## TP=2 Multi-Process Baseline (Llama-3.2-1B-Instruct)

Tensor-parallel (TP=2) benchmarks comparing GIL-enabled Python, free-threaded
Python, and free-threaded Python with threads instead of processes. All runs
use `--cuda-graphs` with 500 requests.

| Configuration | Script | Throughput | Inference Time |
|---|---|---|---|
| GIL-enabled Python (multi-process) | tp_generate.py | 13.40 req/s, 5226 tok/s | 37.3s |
| Free-threaded Python (multi-process) | tp_generate.py | 13.32 req/s, 5193 tok/s | 37.5s |
| Free-threaded Python (multi-thread) | threaded_tp_generate.py | 13.31 req/s, 5187 tok/s | 37.6s |

Free-threading adds **little to no overhead** when vLLM's standard
multi-process TP is used (0.6% difference, well within noise). When the
multi-process layer is replaced with threads (eliminating IPC), the
free-threaded setup matches multi-process performance — the threaded TP
result (13.31 req/s) is within 0.7% of the GIL-enabled baseline (13.40 req/s).

This confirms that for TP workloads, the free-threaded Python runtime is not a
bottleneck: GPU execution dominates, and the threading/IPC choice is immaterial
to throughput.

---

## Profiling Setup

The `samply` profiler has been used to get CPU-level insight into where time
is spent during threaded execution. It provides flame graphs that can reveal
whether bottlenecks are in Python object manipulation, CUDA driver calls,
mutex waits, or GPU kernel execution.

To get useful profiles, run with these env vars set:

    VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHON_GIL=0 PYTHONPERFSUPPORT=1

Ideally pytorch and CUDA should be built with debugging symbols enabled (`-G`
option to nvcc).

Relevant benchmark scripts for investigation:
- `scripts/threaded_step_breakdown.py` — per-component step timing
- `scripts/threaded_scaling_test.py` — single vs. dual engine in-process
- `scripts/cuda_contention_bench.py` — pure CUDA driver lock baseline
- `scripts/python_contention_bench.py` — Python-level object churn baseline
