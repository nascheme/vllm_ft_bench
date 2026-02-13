# Strategy and Direction

## The Core Question

Can single-process multi-threaded vLLM match or beat multi-process
data-parallel inference throughput? And if so, does it unlock capabilities
that are hard or impossible with multi-process designs?

Free-threaded Python 3.14t could be an enabler: it allows Python threads to truly
run in parallel on multiple CPU cores, removing the GIL bottleneck that made
Python threading unattractive for CPU-intensive work.

## Calibrating Expectations

For **offline batch throughput**, the likely answer is: not worth replacing
multi-process. vLLM's existing shared-memory IPC approach already works well
here — the overhead of process communication is negligible compared to GPU
compute time, and there's no scheduling complexity to exploit. This matches
feedback from practitioners who have explored similar ground: if we were
writing vLLM from scratch, free-threaded Python would be a natural fit, but
retrofitting threading into the existing architecture doesn't change the
throughput story much for batch workloads.

More likely is **latency reduction through targeted patches**: rather than
replacing the architecture wholesale, find the specific spots where GIL
contention or serial IPC add latency, and fix those incrementally. A collection
of smaller, focused improvements is more likely to land and matter in practice.

## Current Status

On 2x RTX 2060, threaded + CUDA graphs achieves ~95% of multi-process
throughput. The remaining ~5% gap is under investigation (see `PERFORMANCE.md`).
The short-term goal is to close that gap and understand the remaining
bottlenecks well enough to address them.

## Why Threading Could Be Better Than Multi-Process

For pure offline batch throughput, multi-process already works well. Threading
becomes compelling for **online serving** (requests arriving unpredictably)
because of zero-cost engine observability.

### The multi-process routing problem

In a multi-process setup, a load balancer routing requests to engine processes
must poll each process for its state over IPC. This means:
- Stats are stale (last poll, not real-time)
- Polling adds latency that scales with engine count
- No atomic "read state + route" decision — there's a race between reading
  stats and the engine state changing

### The threading advantage

In a single process, the dispatcher reads engine state as direct memory
accesses (e.g., `scheduler.kv_cache_manager.usage`). No IPC, no staleness.
Routing decisions are instantaneous and can be made atomically with respect to
a single step cycle.

This enables:

**Prefix cache deduplication across engines** — vLLM has automatic prefix
caching: repeated system prompts skip prefill for cached tokens. In
multi-process, each engine caches prefixes independently; both may store the
same 1000-token system prompt. A single-process scheduler can route requests
to the engine that already has their prefix cached, effectively doubling the
useful prefix cache diversity across the GPU fleet.

**Smarter KV cache management** — When one engine is memory-pressured, a
global scheduler can confidently evict its cached prefixes knowing the other
engine still has them. Can also migrate preempted requests to an engine with
headroom, avoiding wasted prefill computation.

**Heterogeneous request routing** — With direct visibility into each engine's
KV cache pressure, routing by queue length or load is easy. Multi-process
routing typically uses round-robin or coarse-grained polling; single-process
can route by real-time block availability.

### What free-threaded Python specifically enables for scheduling

GIL removal made no difference for the engine threads themselves (PyTorch
already releases the GIL during GPU work). But for a **global scheduler
thread** that does CPU-intensive routing decisions — reading stats from both
engines, computing prefix hashes, managing preemption — the GIL would force
it to contend with engine threads during their CPU phases (tokenization, output
processing). Free-threaded Python lets the scheduler thread run truly in
parallel, contributing zero latency to engine thread throughput.

### Parallel IPC processing at scale

A concrete example of a targeted FTP patch that pays off: when engines return
results, the response messages (containing token IDs, logprobs, finish reasons)
must be deserialized. In a multi-process setup this deserialization is serial
— one process handles one engine's messages at a time. With free-threaded
Python, messages arriving from multiple engines can be depickled in parallel on
separate threads.

On 2 GPUs the benefit is modest; on 8 GPUs the effect compounds — 8 streams
of output can be deserialized simultaneously rather than sequentially. This is
a good example of the general principle: **FTP patches have higher leverage on
larger GPU counts**, because that's where serial CPU bottlenecks become a
larger fraction of overall latency.

## Hardware Path

**Current (2x RTX 2060):** DP=2, TP=1. Experiments validate the threading
approach and scheduling logic on real hardware, even though the GPUs are
modest.

**Future (A10/A100):** With higher-end hardware, TP (tensor parallel) runs
become interesting. For example, 8× H100 = 2 engines with TP=4 each, managed
by a single-process threaded dispatcher. The scheduling logic is the same; only
engine initialization becomes more complex (thread-safe process groups, NCCL
communicators per engine).

The value of the global scheduler scales with GPU count and model size: more
GPUs means more prefix cache deduplication benefit, more complex routing
decisions, and a bigger payoff from zero-IPC observability.

## What to Try Next

**Closing the ~5% gap:**
- Profile with `samply` to identify what the remaining overhead is
- Investigate `queue.Queue` mutex as a contention point (replace with
  lock-free or per-engine pre-partitioned structures)
- Check biased reference counting overhead (shared objects touched by both
  engine threads)

**Targeted IPC / latency patches:**
- Identify serial deserialization bottlenecks (depickling engine responses) and
  parallelize with FTP threads — pays off most at 8+ GPUs
- Profile the CPU phases between GPU kernel launches; any serial Python work
  there creates inter-kernel gaps that waste GPU compute time
- Treat latency reduction as the primary metric: lower latency improves
  usability *and* keeps the GPU busier by shrinking idle gaps between kernels

**Online serving benchmark:**
- Build an arrival simulator (Poisson arrivals at varying rates)
- Measure throughput at a given P99 latency target — this is the metric
  production systems optimize for, and it favors real-time routing

**Prefix-aware routing:**
- Implement a dispatcher that routes requests sharing a system prompt to the
  same engine
- Measure KV cache hit rate improvement vs. round-robin routing

**TP>1 path (future):**
- Requires thread-safe parallel state in vLLM (process groups, NCCL)
- The scheduler and dispatch logic need no changes
- Needs multi-GPU hardware with enough cards for TP groups (e.g., 4× A10)
