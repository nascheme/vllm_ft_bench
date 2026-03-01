# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Documentation

- `README.md` — project overview and quick start
- `SCRIPTS.md` — script inventory
- `PERFORMANCE.md` — benchmark results and open questions
- `STRATEGY.md` — project direction and rationale

## Layout

```
scripts/          Benchmark and experiment scripts (all standalone entrypoints)
src/vllm_ft/      Shared utility package
  util.py         CLI parsing, engine creation, monkey-patches, reporting helpers
pyproject.toml    Hatch build (package: vllm_ft)
```

## vLLM source

The vllm source tree is at **`/home/nas/pkg/vllm/vllm/`**.  Always read
files from this path when you need to inspect vllm internals — do not search
`site-packages`, `.venv`, or other locations.

### Async output pipeline (key classes)

The async output path is non-obvious.  When monkey-patching `get_output()`,
target the **concrete** class, not the abstract base:

| Class | File | Role |
|---|---|---|
| `AsyncModelRunnerOutput` | `v1/outputs.py` | ABC, defines `get_output()` interface |
| `AsyncGPUModelRunnerOutput` | `v1/worker/gpu_model_runner.py` | **Concrete class used at runtime.** Has `sampled_token_ids_cpu` (torch.Tensor), `async_copy_ready_event` (torch.Event), `_model_runner_output`, `_invalid_req_indices`, `_logprobs_tensors_cpu` |
| ~~`AsyncOutput`~~ | ~~`v1/worker/gpu/async_utils.py`~~ | Legacy/unused in current vllm — do NOT patch this class |

The executor submits `result.get_output` (a bound method on
`AsyncGPUModelRunnerOutput`) to the async output thread pool via
`UniProcExecutor.collective_rpc()` (`v1/executor/uniproc_executor.py`).

## Development hints

You cannot assume the local PC has `torch` or `vllm` installed. These are only
available on the GPU VM.  This means:

- **Do not** try to install packages or search for those packages to
  resolve import errors — they are intentionally absent locally.
- **Do not** treat LSP/pyright "import could not be resolved" diagnostics for
  `torch` or `vllm` as errors to fix.
- `uvx ruff check` and `uvx ruff format` work fine locally and are the right
  tools for linting and formatting.
- `uvx mypy` and pyright/LSP type checking are **not useful** locally because
  the key dependencies are missing — skip them.

## Key Utilities in `src/vllm_ft/util.py`

- `make_arg_parser()` — common CLI flags
- `build_request_items()` — random / hardcoded / ShareGPT dataset construction
- `apply_forward_context_monkey_patch()` — makes `vllm.forward_context`
  thread-safe via `threading.local()`; call after vllm import, before threads start
- `create_engine(engine_args, device_index, ...)` — creates `LLMEngine` pinned
  to a GPU, patching `device_config.device` and clearing `_ROPE_DICT` cache;
  supports `cuda_graphs=True` and `multiprocess_mode=True`
- `print_throughput_results()` / `print_prompt_length_histogram()` — reporting
