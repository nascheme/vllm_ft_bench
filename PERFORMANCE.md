# Performance Analysis

Benchmarks comparing single-process threaded vLLM engines against the
multi-process baseline.

**Hardware:** 2x RTX 2060 (6 GB each), 64 CPU cores, Linux VM
**Model:** SmolLM2-360M-Instruct
**Dataset:** ShareGPT (~1000 requests, variable prompt/output lengths)
**Python:** CPython 3.14t (free-threaded build)

---

## Headline Numbers

| Configuration | Throughput | Per-GPU | vs. Multi-process |
|---|---|---|---|
| Single GPU (baseline) | ~11 req/s | 11 req/s | — |
| Multi-process (2 GPU, separate processes) | ~24 req/s | ~12 req/s | 1.0× (baseline) |
| Threaded (2 GPU, same process) | ~16 req/s | ~8 req/s | 0.67× |
| **Threaded + CUDA graphs** | **~23 req/s** | **~11.5 req/s** | **~0.95×** |

The threaded + CUDA graphs configuration achieves approximately **95% of
multi-process throughput** in a single process.

### Step-time breakdown (p50, steady-state decode)

```
Component             Single    Threaded   Ratio
schedule               0.19ms    0.29ms    1.5×   (trivial)
execute_model         14.07ms   32.56ms    2.3×   ← bulk of overhead
update_from_output     0.12ms    0.19ms    1.6×   (trivial)
process_outputs        0.14ms    0.22ms    1.6×   (trivial)
total step            22.71ms   34.65ms    1.5×
```

The `execute_model` slowdown dominates. The CPU-only phases (scheduling,
output processing) show only modest scaling costs.

---

## What Has Been Ruled Out

### CPU scheduling overhead
`scheduler.schedule()` + `scheduler.update_from_output()` account for only
**~1.3%** of total step time. Even doubling their cost would not explain the
observed gap.

### Async scheduling / batch pipelining
Enabling `async_scheduling=True` (pipelines CPU scheduling with GPU execution)
does not change throughput. GPU execution time dominates regardless.

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

### Multiprocess mode within the engine
`VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process EngineCore) vs. default
multiprocess mode gives the same single-GPU throughput.

### GIL vs. free-threaded Python (for engine threads)
Testing with both standard (GIL) and free-threaded (`--disable-gil`) builds
produced essentially identical throughput. PyTorch already releases the GIL
during CUDA kernel launches, so the GPU-bound workload sees no benefit from
GIL removal in the engine threads themselves.

---

## The CUDA Graph Mitigation

The primary `execute_model` overhead comes from **CUDA driver API call
contention**. Each vLLM step launches hundreds of CUDA kernels (attention,
linear layers, norms, sampling). These launches go through the CUDA driver,
which serializes across threads within a process via a per-process mutex.

**CUDA graphs** collapse many kernel launches into a single `cudaGraphLaunch`
call. This dramatically reduces driver API traffic and the associated mutex
contention.

Enabling CUDA graphs in `create_engine(cuda_graphs=True)` uses `FULL` mode
(captures the entire forward pass as a single graph), which works without
`torch.compile`. The `enforce_eager=True` flag still suppresses triton/compile;
only CUDA graph capture mode is changed.

```
Without CUDA graphs:  ~16 req/s  (0.67× multi-process)
With CUDA graphs:     ~23 req/s  (0.95× multi-process)
```

---

## Open Questions

The CUDA graph result shows the driver-lock hypothesis captures most of the
overhead, but the story isn't complete. Approximately 5% gap remains, and there
are reasons to believe additional contention sources are active.

### 1. `queue.Queue` mutex contention

`queue.Queue` uses a `threading.Lock` (or `threading.Condition`) internally.
In free-threaded Python, this is a real OS-level mutex, not protected by the
GIL. Multiple engine threads and dispatcher threads contending on the same
queue may cause measurable overhead — especially at high step rates where queue
operations are frequent relative to GPU step time.

**How to investigate:** Replace shared `queue.Queue` with lock-free alternatives
(e.g., per-engine pre-partitioned queues with no sharing at step time) and
measure whether throughput changes.

### 2. Biased reference counting in free-threaded Python

Free-threaded Python 3.14t uses *biased reference counting*: objects are
assumed to be owned by one thread. When a thread other than the "owner" touches
an object's refcount, it requires atomic operations and can cause cache-line
contention. vLLM's step loop creates and destroys many Python objects per step
(request dicts, output lists, sampling results, scheduler state). Two engine
threads doing this simultaneously — especially if they share objects like the
tokenizer or config — may incur atomic refcount overhead.

This is distinct from the CUDA driver lock: it affects all Python code, not
just CUDA API calls, and would not be visible in pure-PyTorch contention
benchmarks (which have minimal Python object churn per iteration).

### 3. Other vLLM global state

Beyond the four known workarounds (see `SCRIPTS.md`), other module-level or
class-level state in vLLM may serialize between threads:
- Logging and metrics infrastructure
- Output processor state (detokenizer, incremental decode buffers)
- Attention backend workspace buffers (lazily allocated global singletons in
  some backends)

**How to investigate:** Instrument vLLM internals at finer granularity;
profile lock wait time with `py-spy` or `perf record`.

### 4. CPU cache thrashing

Two engine threads on different CPU cores competing for L3 cache. Each engine's
working set (scheduler state, KV cache metadata, Python interpreter state)
evicts the other's data, causing higher cache miss rates.

**How to investigate:** Pin engine threads to specific CPU cores/NUMA nodes
with `os.sched_setaffinity()` and compare.

### 5. PyTorch allocator and internal state

PyTorch's CUDA caching allocator is per-device but process-wide. Internal
PyTorch state (autograd bookkeeping, cuDNN handle management) may also have
subtle serialization points.

**How to investigate:** `torch.cuda.memory_stats()` for allocator contention;
try `PYTORCH_NO_CUDA_MEMORY_CACHING=1` to isolate allocator effects.

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
