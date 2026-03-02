# LLM Inference Flow: How Engine Steps Work

## The Core Loop: Autoregressive Decoding

LLM generation is inherently sequential — each token depends on the previous
one. A single prompt goes through many engine steps, each producing **one
token**:

```
Prompt: "The capital of France is"

Step 1:  GPU forward pass ["The capital of France is"]  →  token: "Paris"
Step 2:  GPU forward pass ["The capital of France is Paris"]  →  token: ","
Step 3:  GPU forward pass ["... is Paris,"]  →  token: " a"
Step 4:  GPU forward pass ["... Paris, a"]  →  token: " city"
  ...until stop condition (EOS token, max length, etc.)
```

Each step **depends on the previous step's output** — you can't skip ahead.
The GPU produces logits (probability scores for every vocab token), then the
CPU has to:

1. **Sample** a token from those probabilities (temperature, top-p, etc.)
2. **Check** stop conditions (EOS? max tokens? stop string?)
3. **Update** bookkeeping (KV cache metadata, output buffers, streaming)
4. **Schedule** what goes into the next step (new prompts arriving? finished
   ones to evict?)

## What One Engine Step Looks Like

```
         CPU                              GPU
          │                                │
          │  ┌─────────────────────┐       │
          │  │ Scheduler:          │       │
          │  │  - pick which reqs  │       │
          │  │    to run this step │       │
          │  │  - manage KV cache  │       │
          │  └────────┬────────────┘       │
          │           │                    │
          │     launch kernels ──────────► │  ┌──────────────────────┐
          │           │                    │  │ Forward pass:        │
          │    (CPU free while             │  │  Embed tokens        │
          │     GPU computes)              │  │  × N transformer     │
          │           │                    │  │    layers:           │
          │           │                    │  │    - Attention (QKV) │
          │           │                    │  │    - MLP (up/down)   │
          │           │                    │  │    - LayerNorm       │
          │           │                    │  │  LM head → logits   │
          │           │                    │  │  Argmax / sampling   │
          │           │                    │  └──────────┬───────────┘
          │           │                    │             │
          │     ◄──── token IDs (copy D→H) ─────────────┘
          │           │                    │
          │  ┌────────▼────────────┐       │
          │  │ Output processing:  │       │
          │  │  - detokenize       │       │
          │  │  - check stop conds │       │
          │  │  - stream to client │       │
          │  │  - free finished    │       │
          │  └────────┬────────────┘       │
          │           │                    │
          │     next step ─────────────►   │
          ▼                                ▼
```

## The Batching Trick: Multiple Prompts In One Step

You can't batch steps for the *same* prompt, but you **can** batch *different*
prompts together. If 32 users are all generating, one step processes all of
them at once:

```
Step N:
  ┌──────────────────────────────────────────────┐
  │ Batch (32 requests, all at different points): │
  │                                               │
  │  req 0: "The capital of France is Paris,"     │  → next token: " a"
  │  req 1: "How to bake a"                       │  → next token: " cake"
  │  req 2: "def fibonacci(n):\n    if n <="      │  → next token: " 1"
  │  req 3: "Once upon a time there was a small"  │  → next token: " village"
  │  ...                                          │
  │  req 31: "The weather today"                  │  → next token: " is"
  └──────────────────────────────────────────────┘
          │
          ▼  ONE forward pass, all 32 rows in parallel on GPU
          │
  ┌───────▼──────────┐
  │ 32 token IDs out │  (one per request)
  └──────────────────┘
```

The GPU does one big matrix multiply: `[32 × hidden] @ [hidden × hidden]` —
all 32 prompts in one operation. This is where batching gives you throughput:
the matmul hardware is underutilized with 1 row, but saturated with 32+.

## Why Sub-Batch Splitting Hurts (Phase D2 Result)

Splitting that batch of 32 across 4 streams:

```
WHAT WE TESTED (D2):                      WHAT THE GPU SEES:

Stream A: [8 × hidden] @ [hidden × hidden]  ─┐
Stream B: [8 × hidden] @ [hidden × hidden]   ├─ 4 smaller matmuls
Stream C: [8 × hidden] @ [hidden × hidden]   │  fighting for the
Stream D: [8 × hidden] @ [hidden × hidden]  ─┘  same SMs

          vs.

Default:  [32 × hidden] @ [hidden × hidden]  ── 1 big matmul
                                                  uses SMs efficiently
```

The GPU's tensor cores *already* parallelize across the batch dimension
internally. Splitting into sub-batches just adds synchronization overhead
without giving the hardware anything it couldn't already do.

Phase D2 confirmed this — splitting a batch of 128 across streams was strictly
worse:

| splits | sub_batch | eff throughput | vs baseline |
|--------|-----------|---------------|-------------|
| 1 | 128 | 138k tok/s | 1.00x |
| 2 | 64 | 126k tok/s | 0.91x |
| 4 | 32 | 79k tok/s | 0.57x |
| 8 | 16 | 43k tok/s | 0.31x |

## Where Multi-Stream *Does* Help (Phase D1 Result)

When each stream brings **new, additional work** — not subdivided work:

```
Thread 1: [32 × hidden] @ [hidden × hidden]  on Stream A  ─┐
Thread 2: [32 × hidden] @ [hidden × hidden]  on Stream B   ├─ 4× the total work
Thread 3: [32 × hidden] @ [hidden × hidden]  on Stream C   │  GPU can interleave
Thread 4: [32 × hidden] @ [hidden × hidden]  on Stream D  ─┘  if not saturated
```

This is the multi-engine scenario — 4 independent engines each serving their
own users. Phase D1 showed this gives ~1.9x throughput on the RTX 2060 even at
128 layers, meaning the GPU has spare capacity that a single stream doesn't
fully exploit:

| layers | 1-thread sps | 4-thread sps | speedup |
|--------|-------------|-------------|---------|
| 4 | 1042 | 3271 | 3.14x |
| 8 | 2035 | 3013 | 1.48x |
| 16 | 1560 | 2473 | 1.59x |
| 32 | 1063 | 1826 | 1.72x |
| 64 | 651 | 1228 | 1.89x |
| 128 | 366 | 695 | 1.90x |

## Implications for vLLM

The opportunity isn't sub-batch splitting within one engine. It's running
**multiple engines on the same GPU**, each handling its own batch of requests
with its own stream. The D1 result (1.9x at 128 layers) says there's real
headroom for that approach — the GPU isn't saturated by a single engine's
forward pass, and a second engine on a separate stream can fill the gaps.

---

## KV Cache and Prefix Sharing

The **KV cache** is what makes decoding fast: instead of re-running the entire
forward pass on the full sequence each step, we save the key and value tensors
from every processed token. Each new step, only the *new* token(s) need
computation — older tokens' K,V are loaded from cache:

```
Without KV cache — step 3 re-processes all previous tokens:
  Input: ["The", "capital", "of", "France", "is", "Paris"]
              ↑ attention over all 6 tokens again

With KV cache — step 3 only processes the new token:
  Input: ["Paris"]   (just the new token)
              ↑ Q for "Paris" attends to K,V loaded from cache for prior 5
```

**Prefix sharing** extends this further: if two requests start with the same
tokens (e.g. a shared system prompt), they can use the *same* cached K,V
blocks rather than each computing and storing a separate copy.

### Block Layout: Two Requests Sharing a System Prompt

The KV cache is divided into fixed-size blocks (16 tokens each in practice;
4 tokens here for clarity). Each full block gets a stable hash so it can be
looked up by any request that needs it.

```
System prompt: "You are a helpful assistant."   (tokens 0–7)
Request A:  [system prompt] + "Translate to French: Hello"
Request B:  [system prompt] + "What is 2+2?"

                     ┌── SHARED ── computed once, stored once ──┐
                     │                                          │
Tokens 0–3:   ┌────────────┐     Tokens 4–7:   ┌────────────┐  │
              │  Block 0   │                   │  Block 1   │  │
              │ [T0–T3]    │                   │ [T4–T7]    │  │
              │ hash: H0   │                   │ hash: H1   │  │
              │ ref_cnt: 2 │ ←── both reqs     │ ref_cnt: 2 │  │
              └────────────┘     share it      └────────────┘  │
                     │                                          │
                     └──────────────────────────────────────────┘

Tokens 8+:    ┌────────────┐                   ┌────────────┐
              │  Block 2A  │                   │  Block 2B  │
              │ (req A's   │                   │ (req B's   │
              │  tokens)   │                   │  tokens)   │
              │ hash: H2A  │                   │ hash: H2B  │
              │ ref_cnt: 1 │                   │ ref_cnt: 1 │
              └────────────┘                   └────────────┘
                 unique to A                      unique to B
```

Blocks 0 and 1 sit in GPU VRAM once. Both requests point to them via their
block tables. The ref_cnt tracks how many active requests are using each
block — a block with ref_cnt > 0 is never evicted.

**Only full blocks are cached.** A block that isn't yet completely filled with
tokens gets no hash and cannot be shared. As generation proceeds and the last
partial block fills up, it is hashed and added to the cache at that moment —
available to any future request with the same prefix.

### Block Hash Chain: How Matches Are Found

Each block's hash depends on the previous block's hash and the token IDs it
contains. This means two requests with identical prefixes produce identical
block hashes, regardless of which request computed them first:

```
Block 0:  hash( seed,  [T0, T1, T2, T3] )  =  H0
Block 1:  hash( H0,    [T4, T5, T6, T7] )  =  H1   ← any request with the
Block 2A: hash( H1,    [T8A, ..., T11A] )  =  H2A     same prefix gets H0, H1
Block 2B: hash( H1,    [T8B, ..., T11B] )  =  H2B

Request B arrives. Scheduler walks its block hash list:
  cache[H0]?  → HIT  → reuse Block 0, ref_cnt 1→2
  cache[H1]?  → HIT  → reuse Block 1, ref_cnt 1→2
  cache[H2B]? → MISS → allocate a new block, compute tokens 8B–11B
```

Scanning stops at the first miss — the prefix match is always a contiguous
leading segment.

### What the Attention Kernel Sees

When Request B's first step runs with 8 prefix tokens cached and 4 new tokens:

```
Request B's tokens:  [T0 T1 T2 T3 | T4 T5 T6 T7 | T8' T9' T10' T11']
                      ╰─────────────────────────╯   ╰──────────────────╯
                         cached prefix (8 tok)         new tokens (4 tok)
                         not recomputed                computed this step

  ┌──────────────────────────────────────────────────────────────────┐
  │ Context phase — 8 cached tokens:                                 │
  │   K, V loaded from Block 0 and Block 1 in KV cache              │
  │   no GPU compute — memory reads only                             │
  │                                                                  │
  │ Query phase — 4 new tokens:                                      │
  │   Q, K, V computed by attention projections on fresh embeddings  │
  │   new K, V written into Block 2B (freshly allocated)            │
  │   Q attends to all context K,V + its own K,V (causal mask)      │
  └──────────────────────────────────────────────────────────────────┘

Cost comparison for this prefill step:
  Without prefix cache:  forward pass over 12 tokens
  With prefix cache:     forward pass over  4 tokens   ← 3× less compute
```

The saving compounds with longer shared prefixes — a 1000-token system prompt
shared by 100 concurrent requests means each new request's prefill costs only
the tokens unique to it.

### LRU Eviction Under Memory Pressure

Blocks that no active request is currently using (ref_cnt == 0) sit in a
free list ordered from oldest to newest. When the pool runs out of space,
the oldest idle cached block is evicted to make room:

```
Free list (oldest → newest, all ref_cnt == 0):

  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │  B47 │──▶│  B12 │──▶│  B89 │──▶│  B03 │──▶│  B56 │
  │ H:x  │   │ H:y  │   │ H:z  │   │ H:w  │   │ H:v  │
  └──────┘   └──────┘   └──────┘   └──────┘   └──────┘
      ↑ evicted first                                ↑ evicted last

When a new block is needed:
  1. Pop B47 from the front of the free list
  2. Remove its hash (H:x) from the hash→block lookup table
  3. B47 is now a clean block available for new content

If a subsequent request needs a block with hash H:x again, it gets a cache
miss and must recompute — the eviction was irreversible.
```

Blocks pinned by active requests (ref_cnt > 0) are never touched by eviction.
The LRU ordering means recently-used cached prefixes survive longer, which
naturally favors hot system prompts over one-off prefix segments.

When a request finishes, its blocks are freed to the tail in **reverse** order
(last block freed first). The last block holds the most-unique tokens and is
least likely to be reused, so adding it closest to the eviction head is correct:
it gets evicted before earlier blocks that other requests are more likely to need.
