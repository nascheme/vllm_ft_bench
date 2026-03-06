# vLLM Free-Threaded Python Experiments

This repo contains experiment scripts for running vLLM under free-threaded
Python (CPython 3.14t, `--disable-gil`). The focus is on finding performance
optimizations with multi-threaded setups, enabled by true Python thread
parallelism.

## Relationship to the vLLM free-threaded build

Running vLLM on Python 3.14t requires changes to the vLLM source and build
system. That build can be done with the GitHub nascheme/vllm-ft-build repo.

We assume that build is already ready.  We assume you can run `python` directly
and it will use the correct venv.  Also note that some extension packages still
re-enable the GIL.  So you should run with `PYTHON_GIL=0` to force it to stay
disabled.

## Current Hardware

2x RTX 2060 (6 GB each), 64 CPU cores, Linux VM.

These GPUs are old enough to constrain what's possible:
- All experiments use **TP=1/PP=1** (one GPU per engine, data-parallel only)
- `enforce_eager=True` is the default (CUDA graphs can be enabled explicitly)
- No NVLink — GPUs communicate only through the host

Future experiments on A10/A100 hardware would allow TP (tensor parallel) runs.

## Project Goal

The primary goal is to understand whether single-process multi-threaded vLLM
can match or exceed multi-process data-parallel inference throughput. Because
free-threaded Python releases the GIL, threads can run Python code in parallel
on multiple CPU cores, unlike standard Python threads.

The intermediate target: **parity with multi-process DP** for offline batch
inference. Threaded + CUDA graphs now achieves parity with the equivalent
multi-process configuration (`mp_engine_generate.py`, same step-loop
methodology). The remaining ~3% gap to `mp_static_generate.py` (`LLM.generate()`
with `VLLM_ENABLE_V1_MULTIPROCESSING=1`) is under investigation and is not
caused by threading itself.

## Quick Start

```bash
# Activate your "vllm" build
$ source ~/src/vllm-ft-build/.venv/bin/activate

# Install vllm_ft package
$ python -m pip install -e .

# Single GPU baseline
$ python scripts/simple_generate.py

# Multi-process baseline (2 GPUs, separate processes)
$ python scripts/mp_generate.py

# Threaded dual-engine (2 GPUs, one process)
$ python scripts/threaded_static_generate.py --preload

# Threaded + CUDA graphs (parity with equivalent multi-process)
$ python scripts/threaded_pipelined_generate.py --cuda-graphs
```

Common flags (all scripts):
- `--model MODEL` — default: `HuggingFaceTB/SmolLM2-360M-Instruct`
- `--num-requests N` — number of requests (default: 1000)
- `--dataset PATH` — ShareGPT JSON file; see dataset resolution below
- `--input-len N` / `--output-len N` — token lengths for random prompts
- `--num-gpus N` — number of GPUs (default: 2)

### Dataset resolution

The prompt dataset is resolved in priority order (highest first):

| Source | How to set | Effect |
|---|---|---|
| `--dataset PATH` | CLI flag | Uses `PATH`; empty string forces random prompts |
| `PROMPT_DATASET` | Environment variable | Uses the path; empty string forces random prompts |
| Built-in default | _(automatic)_ | `ShareGPT_V3_unfiltered_cleaned_split.json` in cwd |

```bash
# Use the default dataset (file must exist in cwd)
python scripts/threaded_static_generate.py

# Override dataset for this run
python scripts/threaded_static_generate.py --dataset /data/my_dataset.json

# Force random prompts for this run (ignores env var and default)
python scripts/threaded_static_generate.py --dataset ''

# Set a project-wide default via env var
export PROMPT_DATASET=/data/my_dataset.json
python scripts/threaded_static_generate.py

# Force random prompts via env var
export PROMPT_DATASET=
python scripts/threaded_static_generate.py
```

## Documentation

- `SCRIPTS.md` — inventory of all scripts with descriptions
- `PERFORMANCE.md` — benchmark results, what's been ruled out, open questions
- `STRATEGY.md` — why this project matters and where it's going
