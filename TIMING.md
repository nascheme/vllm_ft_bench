# Gap Measurement Investigation

Documents the investigation into the "other (gap)" residual seen in
`threaded_step_breakdown.py`, from initial hypothesis through definitive
root-cause identification.

---

## Starting Point

`threaded_step_breakdown.py` instruments four calls inside `engine.step()` and
reports everything else as "other (gap)".  The benchmark results show:

```
Component           Single GPU   Threaded     Threaded
                    (alone)      cuda:0       cuda:1
schedule              0.18ms       0.30ms       0.10ms
execute_model        13.93ms      17.02ms      15.92ms
update_from_output    0.12ms       0.23ms       0.08ms
process_outputs       0.14ms       0.23ms       0.07ms
other (gap)           7.68ms      14.99ms       0.39ms  ←
total step           22.74ms      31.57ms      17.81ms
```

Three things stood out:

1. The gap is **~15ms for cuda:0 but only ~0.4ms for cuda:1**, despite both
   running in the same process simultaneously.
2. The gap exists in single-GPU mode too (7.68ms), so it is not caused by
   two-thread contention — it is structural to the in-process step path.
3. `threaded_eventloop_generate.py` confirmed that pipelining
   `process_outputs` off the step loop does nothing: Thread B's inter-step idle
   time is 0.005ms.  The gap is entirely inside `step_fn()`.

---

## Phase 1: Source Analysis — Identifying `sample_tokens`

Reading the vLLM V1 source revealed the full call sequence inside
`EngineCore.step()`.  The critical finding: **`sample_tokens()` was not in the
four components being timed**, so it fell entirely into "other (gap)".

This led to `threaded_gap_breakdown.py` which confirmed that `sample_tokens`
(specifically `_bookkeeping_sync`) accounted for the gap.

---

## Phase 2: Async Scheduling Discovery

Further investigation revealed that with `async_scheduling=True` (the default),
vLLM uses a **completely different step function**: `step_with_batch_queue`
instead of `step`.  This function implements a pipelining scheme:

### Architecture of `step_with_batch_queue`

The function maintains a `batch_queue` (deque, size 2) and alternates between
two paths each call:

**PUSH path (odd steps):** Schedule → execute_model → sample_tokens → submit
future to async thread → early return `None` (no outputs yet).

**POP path (even steps):** Schedule → execute_model → sample_tokens → submit
future → then pop the *previous* step's future from the queue →
`future.result()` (blocks) → `_process_aborts_queue` →
`scheduler.update_from_output` → return outputs.

The async thread (`WorkerAsyncOutput`, a single-worker `ThreadPoolExecutor`)
runs `AsyncOutput.get_output()` which:
1. `copy_event.synchronize()` — waits for the GPU→CPU DMA on a separate CUDA
   copy stream
2. `.tolist()` — converts numpy arrays to Python lists
3. List trimming, logprobs formatting

### Key source locations

- `step_with_batch_queue`: `vllm/v1/engine/core.py`
- `AsyncOutput.__init__` / `get_output()`: `vllm/v1/worker/gpu/async_utils.py`
- Thread pool creation: `vllm/v1/executor/uniproc_executor.py` (1-worker pool)
- Submit site: `uniproc_executor.collective_rpc()` — submits
  `result.get_output` when `isinstance(result, AsyncModelRunnerOutput)`

---

## Phase 3: `trace_step_flow.py` — Execution Order Tracing

Built `scripts/trace_step_flow.py` to trace every component with millisecond
wall-clock offsets.  Runs two phases: single-GPU, then dual-GPU threaded.

### Tracing methodology

Monkey-patches at the engine level (no vllm source modifications):
- Wraps `step_fn`, `scheduler.schedule`, `executor.execute_model`,
  `executor.sample_tokens`, `model_runner._sample`,
  `model_runner._bookkeeping_sync`, `_process_aborts_queue`,
  `scheduler.update_from_output`, `output_processor.process_outputs`
- Wraps `async_thread.submit()` to intercept futures and instrument
  `future.result()`
- Wraps the submitted function (`get_output`) to split timing between
  `copy_event.synchronize()` and `.tolist()` + rest

### Results: Component-level timing

Confirmed the pipelining structure:

| Step | Path | `execute_model` | `future.result()` | Total |
|------|------|------------------|--------------------|-------|
| 1 | EARLY RETURN | 38ms | n/a | 40ms |
| 2 | POP (30 outputs) | 15ms | **692ms** | 711ms |
| 3 | POP (57 outputs) | 720ms (prefill) | 0.01ms | 726ms |
| 4 | POP (89 outputs) | 16ms | **601ms** | 621ms |

Step 3's `execute_model` taking 720ms was a large chunked-prefill batch (99
tokens).  The `future.result()` was instant because the GPU→CPU copy had
plenty of time to complete during that long forward pass.

Dual-GPU threading showed identical `future.result()` wait times (~660-700ms)
— no overlap benefit from the second GPU.

`_process_aborts_queue` was consistently <0.02ms (instant).

### Results: Future Internals Breakdown

Added timing inside the submitted function to split `copy_event.synchronize()`
from `.tolist()`:

```
  [async_thread] sched=0.8ms   copy_sync=0.0ms  tolist+rest=711.1ms
  [async_thread] sched=695.2ms copy_sync=0.0ms  tolist+rest=701.5ms
  [async_thread] sched=0.0ms   copy_sync=0.0ms  tolist+rest=620.6ms
  [async_thread] sched=601.4ms copy_sync=0.0ms  tolist+rest=780.2ms
```

These results initially appeared to show `.tolist()` as the bottleneck, but
this was a measurement error — see Phase 4 below.

---

## ~~Root Cause: `.tolist()` — CPU-Bound Python Object Allocation~~
## RETRACTED — `.tolist()` is NOT the bottleneck

The Phase 3 analysis above contained a critical measurement error.
`trace_step_flow.py` synchronized a **different event object** (`copy_event`
from the old `AsyncOutput` class) than the one the actual `get_output()`
method uses (`async_copy_ready_event` on `AsyncGPUModelRunnerOutput`).
The pre-sync completed instantly on the wrong event, then the full GPU→CPU
copy wait appeared inside the `fn()` call — incorrectly attributed to
`.tolist()`.

Phase 4 (`trace_tolist_patch.py`) corrected this by synchronizing the
actual `async_copy_ready_event` before calling `get_output()`, revealing
the true breakdown.

---

## Phase 4: `trace_tolist_patch.py` — Corrected Measurement

### The vLLM class hierarchy changed

Runtime introspection revealed the actual async output class is
`AsyncGPUModelRunnerOutput` (in `vllm/v1/worker/gpu_model_runner.py`), not
the `AsyncOutput` class (in `vllm/v1/worker/gpu/async_utils.py`) referenced
in the Phase 3 analysis.  Key attributes:

- `sampled_token_ids_cpu` — **torch.Tensor**, not numpy array
- `async_copy_ready_event` — **torch.Event**, not CUDA event
- `vocab_size` — int

### Corrected timing results

Synchronizing the **correct** event (`async_copy_ready_event`) before calling
`get_output()`:

```
  [get_output] sched=0.8ms   sync=690.2ms  tolist+rest=0.0ms  sampled_token_ids_cpu=(31, 1)
  [get_output] sched=673.8ms sync=702.0ms  tolist+rest=0.0ms  sampled_token_ids_cpu=(65, 1)
  [get_output] sched=0.0ms   sync=621.0ms  tolist+rest=0.0ms  sampled_token_ids_cpu=(99, 1)
  [get_output] sched=601.7ms sync=780.3ms  tolist+rest=0.0ms  sampled_token_ids_cpu=(121, 1)
```

| Step | Batch | Shape | sync (ms) | tolist+rest (ms) |
|------|-------|-------|-----------|-----------------|
| 2 | 65 | (31, 1) | 690 | 0.0 |
| 3 | 99 | (65, 1) | 702 | 0.0 |
| 4 | 121 | (99, 1) | 621 | 0.0 |
| 5 | 155 | (121, 1) | 780 | 0.0 |

### Key findings

1. **`.tolist()` takes 0.0ms.**  The tensor shape is `(batch, 1)` — not
   `(batch, 8192)` as previously assumed.  Converting a `(155, 1)` tensor
   to a nested Python list is trivially fast.

2. **The entire 600-780ms is `async_copy_ready_event.synchronize()`** —
   the async thread blocks waiting for the GPU→CPU copy stream to complete.
   The copy stream includes not just the `sampled_token_ids` DMA but also
   any preceding GPU work that hasn't finished on the copy stream.

3. **The shape `(batch, 1)` indicates non-speculative-decode mode.**
   `max_gen_len == 1` means each request generates one token per step.
   The sampler produces `[num_reqs, 1]`, not `[num_reqs, max_seq_len]`.

---

## Root Cause: GPU→CPU Copy Stream Synchronization

The 600-780ms per step is spent in `async_copy_ready_event.synchronize()`,
waiting for the CUDA copy stream to complete.  The copy stream transfers
`sampled_token_ids` (and optionally logprobs tensors) from GPU to CPU via
non-blocking copy on a separate stream.

The wait time is long because:
- The copy stream must wait for the **main stream** (forward pass) to
  complete before starting the copy (`copy_stream.wait_stream(main_stream)`)
- The async thread pool has **1 worker**, so step N+1's future can't start
  until step N's `get_output()` finishes — `sched=600-700ms` shows the
  queue wait

### Why it appeared as ".tolist()"

`trace_step_flow.py` synchronized a `copy_event` from the wrong class
(`AsyncOutput`), which completed instantly.  The remaining time in `fn()`
(which includes the real `async_copy_ready_event.synchronize()` + `.tolist()`)
was labelled `tolist+rest`, creating the false impression that `.tolist()`
was the bottleneck.

### Why dual-GPU threading doesn't help

Each engine has its own async thread pool and copy stream, but the
synchronization wait is per-engine GPU latency.  Two threads waiting on
separate GPU copy streams don't benefit from overlap — each still waits
for its own GPU work to complete.

---

## Summary of Hypotheses

| # | Hypothesis | Status |
|---|-----------|--------|
| H1 | `sample_tokens` accounts for most of the gap | **Confirmed** — but via async pipelining, the cost manifests as `future.result()` blocking |
| H2 | Gap scales with batch size | **Likely** — larger batches mean more GPU work before copy can start |
| H3 | `future.result()` is negligible (pre-set Future) | **Refuted** — with async scheduling, `future.result()` blocks for 600-780ms waiting for the async thread |
| H4 | Small gaps (A-F) < 1ms | **Confirmed** — `_process_aborts_queue` <0.02ms, all other gaps negligible |
| H5 | GPU→CPU DMA transfer is the bottleneck | **Confirmed** — `async_copy_ready_event.synchronize()` accounts for 100% of the 600-780ms |
| H6 | `.tolist()` is the bottleneck | **Refuted** — tensor shape is `(batch, 1)`, `.tolist()` takes 0.0ms |

---

## Implications and Next Steps

The bottleneck is **GPU→CPU copy stream synchronization**, not `.tolist()`
or CPython object allocation.  The async thread blocks waiting for the copy
stream to finish, which in turn waits for the main GPU stream (forward pass)
to complete.

### Theory: PCIe transfer latency, not bandwidth

The copy stream transfers tiny tensors — `(batch, 1)` sampled token IDs,
shape `(155, 1)` at most — so bandwidth is irrelevant.  But PCIe has
non-trivial **latency** per transfer (~5-15µs per round-trip on PCIe 3.0).
If the async pipeline doesn't overlap enough transfers, latency accumulates.

More importantly, the copy stream does `copy_stream.wait_stream(main_stream)`
before initiating the DMA.  This means the copy can't even start until the
entire forward pass completes on the main stream.  The 600-780ms sync wait
likely includes:

1. Waiting for the forward pass to finish on the main stream
2. The actual DMA transfer (tiny, ~µs)
3. Any other work queued on the copy stream

If the forward pass dominates (1), then the async pipelining is failing to
hide it — the copy for step N should overlap with the forward pass for
step N+1, but the single-worker async thread pool serializes the sync calls.

This may explain why `mp_static_generate.py` (multi-process) is faster:
each process gets its own CUDA context with independent stream scheduling.
The in-process threaded setup shares a CUDA context, and the copy stream's
`wait_stream(main_stream)` dependency creates a serial chain that the
pipelining can't break.  In multi-process mode, vLLM's async scheduling may
also behave differently (e.g., using ZMQ-based pipelining that naturally
decouples the copy from the step loop).

### Possible directions

1. **Profile the copy stream** — determine whether the wait is for the
   forward pass to finish (copy_stream.wait_stream) or for the actual DMA
   transfer, or for something else on the copy stream
2. **Increase async thread pool workers** — with more workers, multiple
   futures could be in-flight, though each still blocks on its own sync
3. **Overlap investigation** — understand why the pipelining doesn't hide
   the copy latency (the forward pass for step N+1 should overlap with
   the copy for step N, but the single-worker pool serializes everything)
4. **CUDA event profiling** — use `nsys` or CUDA events to measure the
   actual GPU timeline and identify where the copy stream stalls
5. **Compare in-process vs multi-process stream timelines** — instrument
   both `mp_engine_generate.py` and `threaded_static_generate.py` with
   CUDA events to see whether multi-process mode achieves better overlap
   of forward pass and copy stream
6. **Per-engine CUDA streams** — currently both engines share the default
   stream on their respective GPUs, but within each engine the main stream
   and copy stream are tightly coupled.  Giving each engine its own
   non-default CUDA stream could allow the driver to schedule transfers
   and compute more independently, hiding more CPU↔GPU latency through
   better overlap.  Earlier concern was that CUDA userspace driver locking
   would serialize kernel launches across threads anyway, but
   `cuda_contention_bench.py` results suggest driver lock contention is
   modest (~2-3ms overhead, not 600ms).  Separate streams per engine may
   let the hardware pipeline transfers from one engine while another
   engine's forward pass is running, rather than serializing through a
   shared default stream's dependency chain

### Scripts

- `scripts/trace_step_flow.py` — execution-order tracer with async thread
  instrumentation
- `scripts/trace_tolist_patch.py` — corrected diagnostics with proper event
  synchronization and tensor shape reporting

---

## Phase 5: `cuda_pipeline_bench.py` — Isolated CUDA Pipeline Benchmark

Phases 1-4 identified `async_copy_ready_event.synchronize()` as the bottleneck
(600-780ms), but couldn't distinguish whether the cost came from CUDA driver
lock contention, threading overhead, copy stream scheduling, or simply the
forward pass duration.  `cuda_pipeline_bench.py` isolates these factors using
pure PyTorch (no vLLM) with a synthetic workload that mimics the vLLM pipeline:
chain of matmuls → lm_head projection → argmax → async GPU→CPU copy.

Three configurations were tested: 16 layers / 200 steps, 16 layers / 100 steps,
and 32 layers / 200 steps.  All runs used `PYTHON_GIL=0` (free-threaded Python).

### Phase A Results: No CUDA Driver Lock Contention

| Config | Steps/s | vs single | vs subprocess |
|--------|---------|-----------|---------------|
| 1 GPU single (16L) | 833 | 1.00x | — |
| 2 GPU threaded (16L) | 2582 | 3.10x | 1.38x |
| 2 GPU subprocess (16L) | 1872 | 2.25x | 1.00x |
| 1 GPU single (32L) | 625 | 1.00x | — |
| 2 GPU threaded (32L) | 1846 | 2.95x | 1.61x |
| 2 GPU subprocess (32L) | 1144 | 1.83x | 1.00x |

**Finding: threaded is faster than subprocess**, consistently 1.38-1.61x.  No
driver lock contention.  The subprocess disadvantage comes from process startup
and CUDA context initialization overhead, which dominates at sub-millisecond
step times.  At vLLM's actual step durations (600-780ms), this startup cost
would be negligible.

### Phase B Results: Async Pipelining and Threading

| Config | 16L, 200s | 16L, 100s | 32L, 200s |
|--------|----------------|----------------|----------------|
| Async vs sync (1 GPU) | 1.06x | 2.19x | 1.01x |
| Async vs sync (2 GPU thr) | 1.01x | 1.31x | 1.01x |
| Threaded vs subprocess | 1.70x | 2.12x | 1.86x |
| 2-stream vs 1-stream | 1.01x | 1.09x | 1.00x |

Timing breakdown (representative, 16L / 200 steps):

| Config | launch p50 | copy_wait p50 |
|--------|-----------|---------------|
| Sync 1 GPU | 0.21 ms | 0.39 ms |
| Async 1 GPU | 0.20 ms | 0.34 ms |
| Async 2 GPU threaded | 0.21 ms | 0.33 ms |
| Async 2 GPU subprocess | 0.20 ms | 0.34 ms |

**Findings:**

1. **Async pipelining helps when compute is light** (16L/100s: 2.19x) — the copy
   overlaps with the next step's kernel launch.  With heavier compute (32L),
   the copy is already fully hidden behind the forward pass → 1.01x.
2. **Two copy streams provide no benefit** (1.00-1.09x) — the payload is tiny
   (`[batch_size]` int64 token IDs), so a single copy stream never saturates.
3. **Threaded >> subprocess** at these timescales — process startup dominates.
   Copy wait times are identical (0.33-0.35ms) regardless of threading model,
   confirming no copy-stream contention between threads.
4. **Kernel launch overhead is negligible** — p50 launch times are 0.20-0.37ms,
   unaffected by threading.

### Phase C Results: Same-GPU Stream Isolation (Thread Count Sweep)

Sweep of 1-8 threads on cuda:0, 16 layers, 200 steps (`PYTHON_GIL=0`):

```
threads     dedicated   vs base        shared   vs base  ded/shared
      1      1496.1/s     1.00x      1497.5/s     1.00x       1.00x
      2      2111.0/s     1.41x      1663.5/s     1.11x       1.27x
      3      2362.8/s     1.58x      1663.9/s     1.11x       1.42x
      4      2471.7/s     1.65x      1662.9/s     1.11x       1.49x
      5      2547.3/s     1.70x      1660.0/s     1.11x       1.53x
      6      2567.1/s     1.72x      1663.4/s     1.11x       1.54x
      7      2572.0/s     1.72x      1659.8/s     1.11x       1.55x
      8      2567.3/s     1.72x      1661.2/s     1.11x       1.55x
```

**Findings:**

1. **Shared default stream plateaus immediately at 1.11x** — the default
   stream serializes all GPU work regardless of thread count.  Adding threads
   2-8 produces no additional throughput; every thread's copy must wait for
   every other thread's forward pass on the same stream.

2. **Dedicated streams follow a diminishing-returns curve** — the per-step
   marginal gains:

   | Transition | Incremental gain |
   |------------|-----------------|
   | 1→2 threads | +41% |
   | 2→3 | +17% |
   | 3→4 | +7% |
   | 4→5 | +5% |
   | 5→6 | +2% |
   | 6→8 | flat (saturated) |

   The curve saturates at **1.72x** around 5-6 threads.  With dedicated
   streams the CUDA driver can interleave kernel launches from different
   threads, but once the GPU's SMs are fully occupied additional threads
   just queue behind the same hardware.  The ceiling at 1.72x means ~28%
   of compute capacity is lost to scheduling overhead and memory bandwidth
   contention even under ideal stream isolation.

3. **Dedicated vs shared gap widens with threads** — at 2 threads the
   advantage is 1.27x, growing to 1.55x at 8 threads.  Shared-stream
   throughput is completely thread-count-invariant while dedicated-stream
   throughput scales (sublinearly) with threads.

### Conclusions

The `cuda_pipeline_bench.py` results answer the three questions from the
Phase 4 "Possible directions":

1. **Is the CUDA driver lock serializing kernel launches?**  **No.**  Threaded
   execution is faster than subprocess, and kernel launch times are unaffected
   by threading.  The GIL-free build eliminates Python-side serialization, and
   the CUDA driver handles multi-threaded submissions efficiently.

2. **Does async copy pipelining help?**  **Yes, when compute doesn't dominate.**
   For lightweight workloads the pipeline provides up to 2.19x speedup.  For
   heavier compute (where the forward pass takes longer than the copy), the
   copy is naturally hidden → ~1.0x.  In vLLM's case (600-780ms forward pass,
   sub-ms copy), the async pipeline is working correctly — the long sync wait
   is simply the forward pass duration, not a pipelining failure.

3. **Does stream isolation help on the same GPU?**  **Partially.**  Dedicated
   streams scale sublinearly from 1.41x (2 threads) to a ceiling of 1.72x
   (5-6 threads), while shared streams are stuck at 1.11x regardless of
   thread count.  The 1.72x ceiling reflects GPU SM saturation — beyond
   that, threads queue behind hardware.  For multi-GPU setups (each engine
   on its own GPU), stream isolation is unnecessary — separate GPUs have
   independent SMs.

### Implication for the vLLM bottleneck

The 600-780ms `async_copy_ready_event.synchronize()` wait is **not** caused
by driver locking, threading overhead, or copy stream contention.  It is the
**forward pass duration itself**.  The copy stream does
`copy_stream.wait_stream(main_stream)` before initiating the DMA, so the
sync wait includes the full forward pass.  The async pipelining is working
as designed — `future.result()` blocks until the previous step's GPU work
completes, which takes 600-780ms because that's how long the model forward
pass takes on this hardware.

The remaining optimization opportunities are:
- Reduce forward pass latency (model optimization, quantization, smaller
  batch sizes)
- Multi-process mode (eliminates shared CUDA context, allows OS-level
  scheduling of independent GPU work)
- Same-GPU stream isolation only helps for same-GPU multi-engine setups
  and saturates at ~1.7x (5-6 dedicated streams)

### Scripts

- `scripts/cuda_pipeline_bench.py` — pure-PyTorch benchmark simulating vLLM
  async copy pipeline (four phases: kernel launch, pipeline throughput,
  same-GPU contention, saturation sweep)

---

## Phase 5D: Saturation Sweep & Sub-Batch Splitting

Extended `cuda_pipeline_bench.py` with Phase D to answer: does multi-stream
parallelism still help when the GPU is near saturation, and does splitting a
fixed batch across streams improve throughput?

### D1: Compute Intensity vs Multi-Stream Gain

Fixed 4 threads on cuda:0 with dedicated streams, swept `num_layers`:

| layers | 1-thread sps | 4-thread sps | speedup |
|--------|-------------|-------------|---------|
| 4 | 1042 | 3271 | 3.14x |
| 8 | 2035 | 3013 | 1.48x |
| 16 | 1560 | 2473 | 1.59x |
| 32 | 1063 | 1826 | 1.72x |
| 64 | 651 | 1228 | 1.89x |
| 128 | 366 | 695 | 1.90x |

**Finding:** Multi-stream gains do NOT disappear at higher compute — they
actually increase from 1.48x to 1.90x as layers grow.  The GPU is not
saturated even at 128 layers with batch=32, hidden=1024 on RTX 2060.

### D2: Fixed-Work Sub-Batch Splitting

Split total batch of 128 across N streams sharing the same weight tensors:

| splits | sub_batch | eff throughput | vs baseline |
|--------|-----------|---------------|-------------|
| 1 | 128 | 138k tok/s | 1.00x |
| 2 | 64 | 126k tok/s | 0.91x |
| 4 | 32 | 79k tok/s | 0.57x |
| 8 | 16 | 43k tok/s | 0.31x |

**Finding:** Sub-batch splitting is strictly worse.  The GPU processes one
large batch more efficiently than N smaller ones — the matmul hardware already
parallelizes across the batch dimension internally.

### D3: Per-Step Latency Distribution

| splits | step p50 (ms) | wall step p50 (ms) | vs baseline |
|--------|--------------|--------------------|----|
| 1 | 0.84 | 0.84 | 1.00x |
| 2 | 0.92 | 0.92 | 1.10x |
| 4 | 1.47 | 1.47 | 1.75x |
| 8 | 2.72 | 2.72 | 3.24x |

**Finding:** Individual sub-batches are slower, not faster.  No latency
benefit from splitting.

### Conclusion

Sub-batch stream parallelism would not help vLLM.  Multi-stream gains only
exist when each stream brings genuinely new work (D1 / Phase C), not when
redistributing a fixed workload (D2/D3).

### Related docs

- `LLM_INFERENCE_FLOW.md` — diagrams explaining autoregressive decoding,
  batching, and why sub-batch splitting hurts

---

## Phase 6: Per-Engine CUDA Stream Verification

Investigated whether vLLM engine threads already get their own dedicated CUDA
streams, or whether we need to monkey-patch stream assignment.

### Source analysis

vLLM's `current_stream()` in `vllm/utils/torch_utils.py` uses
`threading.local()` to store the current stream.  On first call from a new
thread, it lazily creates a new `torch.cuda.Stream()` via
`torch.cuda.set_stream(torch.cuda.Stream())`.  The comment says "per process"
but the `threading.local()` makes it per-thread.

### Diagnostic: `stream_diagnostic.py`

Created two engines (sequentially on main thread), then stepped them on
separate threads while logging stream identity:

```
Main thread:  Stream(id=67, ptr=0x40e53ed0)
engine-0:     Stream(id=67, ptr=0x4aa3d850)  — consistent across 100 steps
engine-1:     Stream(id=99, ptr=0x40f2dc00)  — consistent across 100 steps

vLLM streams same?    False
PyTorch streams same? False
Step stream overlap?  False
```

### Finding

**Each engine thread already gets its own dedicated CUDA stream** — no
monkey-patching needed.  The `threading.local()` in vLLM's `current_stream()`
accidentally supports multi-threaded engines correctly.  CUDA graph capture
also uses this path (`cuda_graph.py` calls `current_stream()`), so graphs are
captured and replayed on the per-thread stream.

This means the Phase C/D1 gains from dedicated streams should already be
showing up in the threaded vLLM benchmarks.  The throughput gaps measured
elsewhere are genuinely from other sources (Python overhead, event-loop
round-trips in V1_MP=0 mode), not from stream sharing.

### Scripts

- `scripts/stream_diagnostic.py` — diagnostic script
