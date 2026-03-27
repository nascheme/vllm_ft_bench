# Profiling vLLM with Samply

[Samply](https://github.com/mstange/samply) is a sampling profiler that
produces Firefox Profiler format JSON. It captures both native (C++/CUDA) and
Python frames, making it ideal for profiling vLLM where performance-critical
code spans Python, PyTorch C++, and CUDA driver calls.

## Capturing a profile

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHON_GIL=0 \
PYTHONPERFSUPPORT=1 \
  samply record python scripts/your_benchmark.py [args...]
```

### Environment variables explained

| Variable | Why |
|----------|-----|
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Required. With `fork`, samply cannot symbolize function addresses because the child process's memory maps aren't properly recorded. |
| `PYTHON_GIL=0` | Prevents extension modules from re-enabling the GIL at import time. Only needed for free-threaded Python benchmarks — stock vLLM expects the GIL and uses multiprocessing for parallelism. |
| `PYTHONPERFSUPPORT=1` | Enables the perf trampoline so Python function names appear in the profile instead of opaque `PyEval_EvalFrameDefault` frames. |

Samply writes a `profile.json.gz` file when the process exits.

## Symbolizing the profile

The raw `profile.json.gz` from samply contains hex addresses rather than
readable symbol names. To symbolize it:

1. Open the profile in [Firefox Profiler](https://profiler.firefox.com/)
   (drag-and-drop or use the URL samply prints).
2. Click **Upload Profile** (the upload button in the top-right).
3. Once uploaded, **download** the profile from the upload page.

The download step is key — Firefox Profiler symbolizes the profile before
saving, so the downloaded file contains readable function names. Use this
symbolized file for all subsequent analysis.

## Analyzing with `profile_report.py`

`scripts/profile_report.py` parses symbolized Firefox Profiler JSON and
produces text-based reports. It is the primary tool for understanding where
time goes.

### Recommended workflow

Run these three commands against your symbolized profile. Adjust the
`--start` and `--end` values to isolate the steady-state inference window
(skip startup/shutdown noise — check the Firefox Profiler timeline to find
appropriate boundaries):

```bash
# 1. Summary — identify threads and their sample counts
python3 scripts/profile_report.py profile.json.gz \
  --report summary --start 25 --end 62

# 2. Hotspots — flat ranking of functions by self-time (most useful!)
python3 scripts/profile_report.py profile.json.gz \
  --report hotspots --frames smart --top 40 --start 25 --end 62

# 3. Stacks — top call chains with full context
python3 scripts/profile_report.py profile.json.gz \
  --report stacks --frames smart --top 20 --start 25 --end 62
```

These three reports should give you everything you need. The calltree report
exists but is best avoided — eager-mode PyTorch creates deeply nested
`call_method` → pybind11 → torch dispatch chains that produce hundreds of lines
of C++ template noise, obscuring the actual operations.

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `profile` (positional) | required | Path to `.json` or `.json.gz` profile |
| `--report` | `hotspots` | Report type: `summary`, `stacks`, `hotspots`, `calltree`, `all` |
| `--top` | 30 | Number of entries to display |
| `--min-samples` | 2 | Minimum samples to show a call chain |
| `--thread` | all | Filter by thread name (substring match, case-insensitive) |
| `--frames` | `all` | Frame filtering: `all`, `native`, `script`, `smart` |
| `--start` | – | Start of time window (seconds from profile start) |
| `--end` | – | End of time window (seconds from profile start) |
| `--sort` | `wall` | Sort by: `wall`, `cpu`, or `wait` |
| `--full-paths` | off | Disable path prefix shortening |
| `--min-pct` | 0.5 | Calltree: prune nodes below this % |
| `--max-depth` | 20 | Calltree: max render depth |

### Frame filter modes

| Mode | What it keeps | When to use |
|------|---------------|-------------|
| `smart` | Python frames + interesting native frames (CUDA APIs, PyTorch C++). Drops CPython interpreter noise and unresolved addresses. | **Default choice.** |
| `script` | Python frames only | Pure Python-level view |
| `native` | Non-Python frames only | C++/CUDA internals investigation |
| `all` | Everything | Complete raw picture (noisy) |

### Reading the hotspots report

```
Function                                    Self  Self%    Total Total%  CPU(ms)  Wait(ms)
cuEventSynchronize                         56473  54.4%    67012  64.6%  56441.9     675.4
_GI___clock_gettime                        10681  10.3%    10734  10.3%  10663.8      57.4
call_method                                 2275   2.2%    28874  27.8%   2273.9     341.3
```

- **Self / Self%** — samples where this function was at the top of the stack
  (actually executing). The most important column.
- **Total / Total%** — samples where this function appears anywhere in the
  stack. High Total with low Self means the function is a caller, not a leaf.
- **CPU(ms)** — actual CPU time. **Wait(ms)** — time blocked (I/O, futex, etc.)

### Reading the stacks report

Each entry shows a complete call chain, leaf-first:

```
#1  29224 samples (28.2%)  CPU: 29208.0ms  [WorkerAsyncOutp:3589716]
    cuEventSynchronize
    cudaEventSynchronize
    c10::cuda::impl::CUDAGuardImpl::synchronizeEvent(void*) const
    THPEvent_synchronize(_object*, _object*)
```

- Frames are listed deepest-first (leaf at top, caller at bottom)
- `[ThreadName:TID]` identifies the thread
- The `//N/` prefixes in paths map to a legend printed at the end of the output

### Reading the summary report

Lists every thread with sample counts and CPU/wall/wait totals. Sort by
sample count to find the threads that matter. Threads with very few samples
(< 10) are background noise.

## Interpreting results

### Common vLLM threads

| Thread name pattern | Role |
|---------------------|------|
| `VLLM::EngineCore` or `EngineCor` | Engine core — runs the inference loop |
| `WorkerAsyncOutput` or `WorkerAsyncOutp` | Async output thread — collects results from GPU |
| `MainThread` | Main process thread |

The same thread name with different TIDs means multiple processes (one per
GPU). Use `--thread <name>` to isolate a specific thread.

### Time budget categories

Use the hotspots Self% column to classify where time goes:

**GPU synchronization** — `cudaEventSynchronize` / `cuEventSynchronize`.
Typically the largest category (50%+) in GPU-bound workloads. The CPU
spin-waits for the GPU to finish. `_GI___clock_gettime` with high self-time
is almost always a child of the sync call (the clock read inside the
spin-wait loop) — group it with GPU sync.

**CUDA kernel launches** — `cudaLaunchKernel` / `cuLaunchKernel`. CPU-side
overhead of dispatching work to the GPU. The parent Python frames reveal which
operation triggered the launch (attention, MLP, etc.).

**Triton kernel loading** — `cuModuleLoadData` or
`CompiledKernel._init_handles`. One-time JIT compilation/loading cost per
kernel variant. Use time windowing to exclude this from steady-state analysis.

**Torch dispatch overhead** — `call_method` with high self-time. In eager
mode, PyTorch's dispatch mechanism (`call_method` → pybind11 →
`torch::jit::invokeOperatorFromPython` → `c10::Dispatcher::callBoxed`) wraps
every operation. The self-time in `call_method` is argument parsing and
dispatch key lookup, not the actual tensor work. Inside the dispatch chain you
find the real operations: `THPVariable_linear`, `fused_add_rms_norm`,
`rotary_embedding`, `silu_and_mul`, etc.

**Input preparation** — `GPUModelRunner._prepare_inputs`,
`_prepare_input_ids`. CPU work setting up tensors before model execution.

**Scheduler overhead** — `EngineCore.step_with_batch_queue`,
`Scheduler.schedule`. CPU-side batch scheduling decisions.

### Key questions to answer

- **GPU-bound or CPU-bound?** If GPU sync dominates (50%+), the workload is
  GPU-bound and the CPU is mostly waiting. CPU-bound profiles show more time
  in dispatch, scheduling, or input preparation.
- **Is the async output pattern working?** The async output thread should be
  the one waiting on GPU sync, not the engine core thread.
- **Any unexpected costs?** Large scheduler overhead, serialization costs, or
  Triton JIT outside the startup window are red flags.
- **One-time vs. steady-state costs?** Always use `--start`/`--end` to
  separate startup (model loading, Triton JIT) from steady-state inference.

## Using an LLM for analysis

For deeper analysis, you can hand the profile to an LLM. Give it the
symbolized `profile.json.gz` and `scripts/profile_report.py`, then instruct
it as follows:

### Instructions to give your LLM

> **Step 1: Generate reports.** Run these three commands against the profile
> (adjust the `--start`/`--end` window to cover steady-state inference only):
>
> ```bash
> python3 scripts/profile_report.py profile.json.gz --report summary --start START --end END
> python3 scripts/profile_report.py profile.json.gz --report hotspots --frames smart --top 40 --start START --end END
> python3 scripts/profile_report.py profile.json.gz --report stacks --frames smart --top 20 --start START --end END
> ```
>
> Do not run calltree or other exploratory variants — these three reports
> provide everything needed and the calltree output is very large and noisy
> with eager-mode PyTorch.
>
> **Step 2: Classify the time budget.** Group functions from the hotspots
> report into categories: GPU synchronization, CUDA kernel launches, Triton
> kernel loading, torch dispatch overhead, input preparation, and
> scheduler/engine overhead. Note that `_GI___clock_gettime` is typically
> part of the GPU sync spin-wait loop.
>
> **Step 3: Produce diagrams.** Create:
> - A process/thread architecture diagram showing threads and their roles
> - A dominant call path diagram from the top stacks entries
> - A vLLM engine call flow showing the path from the engine loop through
>   executor → worker → model runner → model forward
> - An ASCII time budget bar chart
>
> **Step 4: Answer these questions:**
> - Is the workload GPU-bound or CPU-bound?
> - What is the CPU actually spending time on?
> - Are there unexpected costs (scheduler overhead, serialization)?
> - Is the async output pattern working correctly?
> - Are there one-time costs (Triton JIT) inflating the profile?

### Tips for LLM analysis

- Provide the vLLM source tree so the LLM can look up function
  implementations when call chains are ambiguous.
- Always specify a time window — analyzing the full profile wastes context on
  import/startup noise.
- The `--frames smart` filter dramatically reduces noise. Without it, the
  output is cluttered with CPython interpreter frames (`PyEval_*`,
  `py_trampoline_*`, etc.) and unresolved hex addresses (`fun_XXXXXX`).

## Adapting to different setups

Your profile may look different depending on configuration:

- **Different executors**: `UniProcExecutor` (single GPU),
  `MultiProcExecutor` (multi-GPU TP), `RayDistributedExecutor` — thread
  structure and sync patterns vary.
- **Attention backends**: `TritonAttentionImpl`, `FlashAttentionImpl`,
  `FlashInferImpl` — look inside `Attention.forward` to see which is active.
- **Models**: You may see `MistralForCausalLM`, `Qwen2ForCausalLM`, etc.
  instead of `LlamaForCausalLM`. The overall pattern is the same.
- **torch.compile vs eager**: With compile, you'll see `torch._dynamo` and
  `torch._inductor` frames instead of direct model forward calls. In eager
  mode (the default for most profiling), you see `call_method` → torch
  dispatch chains.
- **Threaded vs multiprocess**: Threaded benchmarks run multiple engines as
  threads in one process; multiprocess benchmarks use separate processes per
  GPU. Thread interactions and sync patterns differ accordingly.
