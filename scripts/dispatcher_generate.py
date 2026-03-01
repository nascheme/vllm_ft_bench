# Threaded Request Dispatcher (TRD) — dual-engine multi-GPU benchmark.
#
# Architecture: two independent LLMEngine instances (one per GPU) with
# per-engine queues and explicit least-load routing via a Dispatcher.
#
#   Caller Thread(s)       Tokenizer Thread          Engine Thread 0 (cuda:0)    Engine Thread 1 (cuda:1)
#     submit(prompt, sp)     process_inputs()         engine_queues[0].get()      engine_queues[1].get()
#     -> Future              route(task) by load      add_request()               add_request()
#                            engine_queues[i].put()   step()                      step()
#                                                     future.set_result()         future.set_result()
#
# Key difference from threaded_generate.py: per-engine queues with explicit
# routing instead of a shared queue with implicit GIL-based balancing.

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import concurrent.futures
import queue
import threading
import time
import uuid
from dataclasses import dataclass

import torch

from vllm import EngineArgs, SamplingParams
from vllm.tokenizers import get_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm_ft.util import (
    apply_forward_context_monkey_patch,
    build_request_items,
    create_engine,
    make_arg_parser,
    print_throughput_results,
    render_request,
)

apply_forward_context_monkey_patch()

# ---------------------------------------------------------------------------

MAX_PULL_PER_STEP = 8

_SENTINEL = None  # Poison pill for shutdown


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InferenceTask:
    task_id: str
    prompt: str
    sampling_params: SamplingParams
    future: concurrent.futures.Future
    created_at: float
    proc_input: object = None
    engine_index: int = -1


class EngineStats:
    """Per-engine load counters.

    Shared between tokenizer thread (reads in_flight) and engine thread
    (updates on start/complete).  Lock is held only during counter updates.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.in_flight: int = 0
        self.completed: int = 0
        self.prompt_tokens: int = 0
        self.output_tokens: int = 0

    def increment_in_flight(self):
        with self._lock:
            self.in_flight += 1

    def record_completion(self, prompt_toks: int, output_toks: int):
        with self._lock:
            self.in_flight -= 1
            self.completed += 1
            self.prompt_tokens += prompt_toks
            self.output_tokens += output_toks

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "in_flight": self.in_flight,
                "completed": self.completed,
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
            }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class Dispatcher:
    def __init__(self, model: str, num_gpus: int = 2, **engine_kwargs):
        self.num_gpus = num_gpus

        # Create engines sequentially.
        engine_args = EngineArgs(
            model=model,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            async_scheduling=False,
            **engine_kwargs,
        )
        self.engines = []
        for i in range(num_gpus):
            print(f"Creating engine on cuda:{i} ...")
            self.engines.append(create_engine(engine_args, i, UsageContext.LLM_CLASS))
        print(f"All {num_gpus} engines created.")

        self.renderer = self.engines[0].renderer

        # Per-engine queues and stats.
        self.engine_queues: list[queue.Queue] = [queue.Queue() for _ in range(num_gpus)]
        self.stats: list[EngineStats] = [EngineStats() for _ in range(num_gpus)]

        # Tokenize queue: caller -> tokenizer thread.
        self._tokenize_queue: queue.Queue = queue.Queue()

        # Threads.
        self._shutdown = threading.Event()
        self._tok_thread = threading.Thread(
            target=self._tokenizer_loop, name="Dispatch::tok", daemon=True
        )
        self._engine_threads: list[threading.Thread] = []
        for i in range(num_gpus):
            t = threading.Thread(
                target=self._engine_loop,
                args=(i,),
                name=f"Dispatch::engine{i}",
                daemon=True,
            )
            self._engine_threads.append(t)

        # Start all threads.
        self._tok_thread.start()
        for t in self._engine_threads:
            t.start()

    # -- Public API --

    def submit(
        self, prompt: str, sampling_params: SamplingParams
    ) -> concurrent.futures.Future:
        """Submit a single request. Returns a Future resolved with RequestOutput."""
        future: concurrent.futures.Future = concurrent.futures.Future()
        task = InferenceTask(
            task_id=str(uuid.uuid4()),
            prompt=prompt,
            sampling_params=sampling_params,
            future=future,
            created_at=time.time(),
        )
        self._tokenize_queue.put(task)
        return future

    def submit_batch(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[concurrent.futures.Future]:
        """Submit multiple requests. Returns a list of Futures."""
        return [self.submit(p, sampling_params) for p in prompts]

    def shutdown(self, wait: bool = True):
        """Signal all threads to stop and optionally wait for them."""
        self._shutdown.set()
        # Poison pills for tokenizer.
        self._tokenize_queue.put(_SENTINEL)
        # Poison pills for engine threads.
        for eq in self.engine_queues:
            eq.put(_SENTINEL)
        if wait:
            self._tok_thread.join()
            for t in self._engine_threads:
                t.join()

    # -- Internal: routing --

    def _route(self) -> int:
        """Pick the engine with the lowest in_flight count."""
        best = 0
        best_load = self.stats[0].in_flight
        for i in range(1, self.num_gpus):
            load = self.stats[i].in_flight
            if load < best_load:
                best = i
                best_load = load
        return best

    # -- Internal: tokenizer thread --

    def _tokenizer_loop(self):
        while not self._shutdown.is_set():
            try:
                task = self._tokenize_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is _SENTINEL:
                break

            try:
                task.proc_input = render_request(self.renderer, task.prompt)

                # Route to least-loaded engine.
                idx = self._route()
                task.engine_index = idx
                self.stats[idx].increment_in_flight()
                self.engine_queues[idx].put(task)
            except Exception as e:
                task.future.set_exception(e)

    # -- Internal: engine threads --

    def _engine_loop(self, engine_index: int):
        torch.cuda.set_device(engine_index)
        engine = self.engines[engine_index]
        eq = self.engine_queues[engine_index]
        stats = self.stats[engine_index]

        # Map of task_id -> InferenceTask for in-flight requests.
        pending: dict[str, InferenceTask] = {}

        while not self._shutdown.is_set() or pending or not eq.empty():
            # Pull up to MAX_PULL_PER_STEP tasks from the per-engine queue.
            pulled = 0
            for _ in range(MAX_PULL_PER_STEP):
                try:
                    task = eq.get_nowait()
                except queue.Empty:
                    break
                if task is _SENTINEL:
                    # Drain remaining pending before exiting.
                    continue
                pulled += 1
                pending[task.task_id] = task
                engine.add_request(
                    task.task_id,
                    task.proc_input,
                    task.sampling_params,
                )

            if engine.has_unfinished_requests():
                request_outputs = engine.step()
                for output in request_outputs:
                    if output.finished:
                        task = pending.pop(output.request_id, None)
                        if task is not None:
                            prompt_toks = (
                                len(output.prompt_token_ids)
                                if output.prompt_token_ids
                                else 0
                            )
                            output_toks = sum(
                                len(o.token_ids) for o in output.outputs if o
                            )
                            stats.record_completion(prompt_toks, output_toks)
                            task.future.set_result(output)
            elif self._shutdown.is_set() and eq.empty() and not pending:
                break
            else:
                # No work yet — block briefly for new tasks.
                try:
                    task = eq.get(timeout=0.5)
                    if task is _SENTINEL:
                        if not pending:
                            break
                        continue
                    pending[task.task_id] = task
                    engine.add_request(
                        task.task_id,
                        task.proc_input,
                        task.sampling_params,
                    )
                except queue.Empty:
                    if self._shutdown.is_set() and not pending:
                        break


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser(
        "Threaded Request Dispatcher — dual-engine multi-GPU benchmark.",
    )
    args = parser.parse_args()

    # 1. Generate dataset.
    tokenizer = get_tokenizer(args.model)
    request_items = build_request_items(args, tokenizer)
    num_requests = len(request_items)

    # 2. Create dispatcher (creates engines internally).
    dispatcher = Dispatcher(model=args.model, num_gpus=args.num_gpus)

    # 3. Submit all requests.
    print(f"Submitting {num_requests} requests ...")
    start_time = time.time()
    futures = []
    for req, sp in request_items:
        f = dispatcher.submit(req.prompt, sp)
        futures.append(f)

    # 4. Collect results with progress updates.
    completed = 0
    last_report = time.time()
    for f in concurrent.futures.as_completed(futures):
        try:
            f.result()
        except Exception as e:
            print(f"Request failed: {e}")
        completed += 1
        now = time.time()
        if now - last_report >= 2.0:
            elapsed = now - start_time
            print(f"  [{elapsed:.1f}s] {completed}/{num_requests} completed ...")
            last_report = now

    elapsed = time.time() - start_time

    # 5. Shutdown dispatcher.
    dispatcher.shutdown(wait=True)

    # 6. Report results.
    engine_stats = [dispatcher.stats[i].snapshot() for i in range(args.num_gpus)]
    print_throughput_results(elapsed, engine_stats)


if __name__ == "__main__":
    main()
