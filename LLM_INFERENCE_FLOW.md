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
