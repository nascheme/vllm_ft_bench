"""Threaded tensor-parallel executor and helpers.

Provides ThreadedTPExecutor — a single-process, multi-threaded TP executor
that eliminates the SHM IPC overhead of vLLM's MultiprocExecutor.

Workers communicate via NCCL (GPU tensors) and Python object passing
(zero-copy within the same process).
"""

import copy
import queue as queue_mod
import sys
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError
from contextlib import contextmanager, suppress
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    _get_unique_name,
    _register_group,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Thread-local storage for parallel_state
# ---------------------------------------------------------------------------

_ps_tls = threading.local()


def _tls_get_tp_group() -> GroupCoordinator:
    g = getattr(_ps_tls, "_TP", None)
    assert g is not None, "tensor model parallel group is not initialized (TLS)"
    return g


def _tls_get_pp_group() -> GroupCoordinator:
    g = getattr(_ps_tls, "_PP", None)
    assert g is not None, "pipeline model parallel group is not initialized (TLS)"
    return g


def _tls_get_world_group() -> GroupCoordinator:
    g = getattr(_ps_tls, "_WORLD", None)
    assert g is not None, "world group is not initialized (TLS)"
    return g


def _tls_get_dcp_group() -> GroupCoordinator:
    g = getattr(_ps_tls, "_DCP", None)
    assert g is not None, "decode context parallel group is not initialized (TLS)"
    return g


def _tls_get_pcp_group() -> GroupCoordinator:
    g = getattr(_ps_tls, "_PCP", None)
    assert g is not None, "prefill context parallel group is not initialized (TLS)"
    return g


def apply_parallel_state_tls_patch():
    """Replace parallel_state getters with TLS-based versions.

    Must be called after vllm is imported but before worker threads start.
    Patches all modules that imported the original getter functions.
    """
    import vllm.distributed.parallel_state as _ps

    patches = [
        ("get_tp_group", _ps.get_tp_group, _tls_get_tp_group),
        ("get_pp_group", _ps.get_pp_group, _tls_get_pp_group),
        ("get_world_group", _ps.get_world_group, _tls_get_world_group),
        ("get_dcp_group", _ps.get_dcp_group, _tls_get_dcp_group),
        ("get_pcp_group", _ps.get_pcp_group, _tls_get_pcp_group),
        # Also patch the backward-compat alias.
        (
            "get_context_model_parallel_group",
            _ps.get_context_model_parallel_group,
            _tls_get_dcp_group,
        ),
    ]

    # Patch the canonical module first.
    for attr, _orig, new in patches:
        setattr(_ps, attr, new)

    # Patch every other module that imported the original.
    for mod in list(sys.modules.values()):
        if mod is None or mod is _ps:
            continue
        for attr, orig, new in patches:
            try:
                if getattr(mod, attr, None) is orig:
                    setattr(mod, attr, new)
            except TypeError, AttributeError:
                pass


# ---------------------------------------------------------------------------
# Thread-local vllm_config patch
# ---------------------------------------------------------------------------

_config_tls = threading.local()


def _tl_get_current_vllm_config():
    cfg = getattr(_config_tls, "config", None)
    if cfg is None:
        raise AssertionError(
            "Current vLLM config is not set (TLS). "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context."
        )
    return cfg


def _tl_get_current_vllm_config_or_none():
    return getattr(_config_tls, "config", None)


@contextmanager
def _tl_set_current_vllm_config(vllm_config, check_compile=False, prefix=None):
    old = getattr(_config_tls, "config", None)
    old_prefix = getattr(_config_tls, "prefix", None)
    try:
        _config_tls.config = vllm_config
        _config_tls.prefix = prefix
        yield
    finally:
        _config_tls.config = old
        _config_tls.prefix = old_prefix


def apply_vllm_config_tls_patch():
    """Replace global vllm_config accessors with thread-local versions.

    The default set/get_current_vllm_config uses a module-level global,
    which is not thread-safe.  This patch makes them per-thread.
    Must be called after vllm is imported but before worker threads start.
    """
    import vllm.config.vllm as _vc

    orig_set = _vc.set_current_vllm_config
    orig_get = _vc.get_current_vllm_config
    orig_get_or_none = _vc.get_current_vllm_config_or_none

    patches = [
        ("set_current_vllm_config", orig_set, _tl_set_current_vllm_config),
        ("get_current_vllm_config", orig_get, _tl_get_current_vllm_config),
        (
            "get_current_vllm_config_or_none",
            orig_get_or_none,
            _tl_get_current_vllm_config_or_none,
        ),
    ]

    # Also clear the lru_cache on get_cached_compilation_config since it
    # caches across threads.  We disable the cache entirely by replacing it
    # with a plain function.
    _vc.get_cached_compilation_config = lambda: (
        _tl_get_current_vllm_config().compilation_config
    )

    # Patch canonical module.
    for attr, _orig, new in patches:
        setattr(_vc, attr, new)

    # Patch every importer.
    for mod in list(sys.modules.values()):
        if mod is None or mod is _vc:
            continue
        for attr, orig, new in patches:
            try:
                if getattr(mod, attr, None) is orig:
                    setattr(mod, attr, new)
            except TypeError, AttributeError:
                pass



# ---------------------------------------------------------------------------
# InProcessGroup — in-memory substitute for StatelessProcessGroup
# ---------------------------------------------------------------------------


class InProcessGroup(StatelessProcessGroup):
    """In-process group for NCCL unique ID exchange between threads."""

    def __init__(
        self, rank: int, world_size: int, barrier: threading.Barrier, shared_state: dict
    ):
        # Skip StatelessProcessGroup.__init__ (needs TCP store).
        self.rank = rank
        self.world_size = world_size
        self._barrier = barrier
        self._shared = shared_state

    def broadcast_obj(self, obj: Any, src: int) -> Any:
        if self.rank == src:
            self._shared["bcast"] = obj
        self._barrier.wait()
        result = self._shared["bcast"]
        self._barrier.wait()  # ensure all read before next write
        return result


# ---------------------------------------------------------------------------
# Duck-typed GroupCoordinator construction
# ---------------------------------------------------------------------------


@contextmanager
def _nccl_graph_capture(self, graph_capture_context=None):
    """graph_capture for GroupCoordinators using _PyNcclDeviceCommunicator.

    Skips the CudaCommunicator isinstance check from the upstream method
    while preserving the stream-switching logic needed for CUDA graph capture.
    Also disables the CPU collective barrier during capture (ops are recorded,
    not executed, so the barrier would deadlock).
    """
    # Pin this thread to the correct device before any stream operations.
    torch.cuda.set_device(self.device)

    if graph_capture_context is None:
        stream = torch.cuda.Stream(device=self.device)
        from vllm.distributed.parallel_state import GraphCaptureContext

        graph_capture_context = GraphCaptureContext(stream)
    else:
        stream = graph_capture_context.stream

    # Synchronize the device to clear all pending work on all streams
    # before switching to the capture stream.  This prevents
    # "legacy stream depends on capturing stream" errors when ops like
    # cos_sin_cache.to() read buffers created on the default stream.
    torch.cuda.synchronize(self.device)

    dev_comm = self.device_communicator
    if dev_comm is not None:
        dev_comm._barrier_enabled = False

    # Use set_stream/restore instead of torch.cuda.stream() context
    # manager — the context manager may have thread-safety issues in
    # free-threaded Python.
    prev_stream = torch.cuda.current_stream(self.device)
    torch.cuda.set_stream(stream)
    # Initialize per-stream CUDA handles (mirrors vLLM ubatching pattern).
    _ = torch.cuda.current_blas_handle()
    try:
        yield graph_capture_context
    finally:
        torch.cuda.set_stream(prev_stream)
        if dev_comm is not None:
            dev_comm._barrier_enabled = True


def _make_group_coordinator(
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    group_name: str,
    device_communicator=None,
) -> GroupCoordinator:
    """Build a GroupCoordinator without calling __init__ (no torch.distributed)."""
    import types

    group = object.__new__(GroupCoordinator)
    group.rank = rank
    group.ranks = list(range(world_size))
    group.world_size = world_size
    group.rank_in_group = rank
    group.local_rank = local_rank
    group.device = torch.device(f"cuda:{local_rank}")
    group.unique_name = _get_unique_name(group_name)
    group.cpu_group = None
    group.device_group = None
    group.mq_broadcaster = None
    group.use_device_communicator = device_communicator is not None
    group.device_communicator = device_communicator
    group.use_custom_op_call = True
    group.use_cpu_custom_send_recv = False
    # Override graph_capture to skip CudaCommunicator isinstance check.
    group.graph_capture = types.MethodType(_nccl_graph_capture, group)
    _register_group(group)
    return group


class _PyNcclDeviceCommunicator:
    """Minimal device communicator wrapping PyNcclCommunicator.

    Implements the methods that model layers actually call via custom ops
    (all_reduce, all_gather, etc.).  Used with GroupCoordinator whose
    graph_capture is overridden by _nccl_graph_capture (which skips the
    CudaCommunicator isinstance check from upstream).
    """

    def __init__(self, rank, world_size, local_rank, pynccl_comm, unique_name,
                 collective_barrier: threading.Barrier | None = None):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.cpu_group = None
        self.device_group = None
        self.unique_name = unique_name
        self.pynccl_comm = pynccl_comm
        # Disable cross-process all-reduce variants.
        self.ca_comm = None
        self.qr_comm = None
        self.symm_mem_comm = None
        self.fi_ar_comm = None
        self.use_all2all = False
        # CPU barrier shared across all TP ranks.  After each NCCL collective
        # enqueue, every rank waits here so no rank races ahead and triggers
        # an implicit CUDA sync before the other rank has posted its side of
        # the collective (which would deadlock).  This is pure CPU
        # synchronisation — no GPU round-trip like torch.cuda.synchronize().
        self._collective_barrier = collective_barrier
        # Disabled during CUDA graph capture (ops are just recorded, not
        # executed, so the barrier would deadlock or stall capture).
        self._barrier_enabled = True

    def _rank_sync(self) -> None:
        """Wait for all TP ranks to reach this point (CPU-only)."""
        if self._barrier_enabled and self._collective_barrier is not None:
            self._collective_barrier.wait()

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        out = self.pynccl_comm.all_reduce(input_)
        # Ensure all ranks have enqueued this collective before any rank
        # continues — prevents one rank from racing ahead.
        self._rank_sync()
        if out is None:
            return input_.clone()
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        self.pynccl_comm.all_gather(output_tensor, input_)
        self._rank_sync()
        # Reshape to match DeviceCommunicatorBase convention.
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return self.pynccl_comm.reduce_scatter(input_, dim)

    def send(self, tensor, dst):
        self.pynccl_comm.send(tensor, dst)

    def recv(self, tensor, src):
        self.pynccl_comm.recv(tensor, src)


# ---------------------------------------------------------------------------
# FutureWrapper — ordered lazy resolution (mirrors MultiprocExecutor)
# ---------------------------------------------------------------------------


class _FutureWrapper(Future):
    """Future that lazily fetches its result, draining predecessors first."""

    def __init__(self, futures_deque: deque):
        self._futures_deque = futures_deque
        super().__init__()

    def result(self, timeout=None):
        # Drain any futures ahead of us in the deque.
        while not self.done():
            future, get_response = self._futures_deque.pop()
            future._resolve(get_response)
        return super().result()

    def _resolve(self, get_response: Callable):
        try:
            response = get_response()
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


# ---------------------------------------------------------------------------
# ThreadedTPExecutor
# ---------------------------------------------------------------------------


class ThreadedTPExecutor(Executor):
    """Single-process, multi-threaded tensor-parallel executor.

    Workers run as daemon threads. Communication uses NCCL for GPU tensors
    and direct Python object passing for control plane (zero IPC overhead).
    """

    def _init_executor(self) -> None:
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self._tp_size = tp_size
        self._workers: list = [None] * tp_size
        # Store worker init errors so the main thread can re-raise them.
        self._worker_errors: list = [None] * tp_size
        # Keep strong references to GroupCoordinators so _groups weak refs
        # stay alive across GC.
        self._group_coordinators: list = []

        # Per-worker command and result queues (replaces barrier dispatch).
        self._cmd_queues: list[queue_mod.SimpleQueue] = [
            queue_mod.SimpleQueue() for _ in range(tp_size)
        ]
        self._result_queues: list[queue_mod.SimpleQueue] = [
            queue_mod.SimpleQueue() for _ in range(tp_size)
        ]
        # Ordered deque of pending non-block futures (FIFO drain).
        self._futures_deque: deque[tuple[_FutureWrapper, Callable]] = deque()

        nccl_barrier = threading.Barrier(tp_size)
        nccl_shared: dict = {}
        ready_barrier = threading.Barrier(tp_size + 1)
        # Shared barrier for NCCL collective synchronisation (CPU-only).
        collective_barrier = threading.Barrier(tp_size)
        self._collective_barrier = collective_barrier

        # --- Patch graph pool to be per-thread ---
        # The default get_global_graph_pool() returns a process-wide
        # singleton.  Concurrent CUDA graph captures from different threads
        # sharing the same pool can conflict.  Patch it to return a
        # per-thread pool so each worker captures independently.
        from vllm.platforms import current_platform

        _pool_tls = threading.local()
        _orig_get_pool = type(current_platform).get_global_graph_pool

        def _per_thread_graph_pool(self_plat):
            pool = getattr(_pool_tls, "pool", None)
            if pool is None:
                pool = self_plat.graph_pool_handle()
                _pool_tls.pool = pool
            return pool

        type(current_platform).get_global_graph_pool = _per_thread_graph_pool

        # Also make set_graph_pool_id thread-local.
        from vllm.distributed.device_communicators import pynccl_allocator as _pa

        _gpi_tls = threading.local()
        _orig_set_gpi = _pa.set_graph_pool_id

        def _tls_set_graph_pool_id(pool_id):
            _gpi_tls.pool_id = pool_id

        _pa.set_graph_pool_id = _tls_set_graph_pool_id
        # Also patch any module that already imported set_graph_pool_id.
        for mod in list(sys.modules.values()):
            if mod is None or mod is _pa:
                continue
            try:
                if getattr(mod, "set_graph_pool_id", None) is _orig_set_gpi:
                    mod.set_graph_pool_id = _tls_set_graph_pool_id
            except (TypeError, AttributeError):
                pass

        # --- Patch torch.cuda.graph to use thread_local error mode ---
        # The default "global" mode means any CUDA error on ANY thread
        # invalidates graph captures on ALL threads.  "thread_local" keeps
        # errors isolated per-thread.
        _OrigCudaGraph = torch.cuda.graph

        class _ThreadLocalCudaGraph(_OrigCudaGraph):
            def __init__(self, *args, capture_error_mode="thread_local", **kwargs):
                super().__init__(
                    *args, capture_error_mode=capture_error_mode, **kwargs
                )

        torch.cuda.graph = _ThreadLocalCudaGraph

        # --- Make _ROPE_DICT thread-local ---
        # The global _ROPE_DICT caches RotaryEmbedding instances by key.
        # When workers load models concurrently, they share the same dict,
        # causing cross-device contamination (worker-0 gets RoPE on cuda:1).
        # Replace with a thread-local dict so each worker gets its own.
        import vllm.model_executor.layers.rotary_embedding as _rope_mod

        _rope_tls = threading.local()

        class _ThreadLocalRopeDict:
            """Dict-like proxy that stores per-thread _ROPE_DICT copies."""

            def __contains__(self, key):
                d = getattr(_rope_tls, "rope_dict", None)
                return d is not None and key in d

            def __getitem__(self, key):
                return getattr(_rope_tls, "rope_dict", {})[key]

            def __setitem__(self, key, value):
                d = getattr(_rope_tls, "rope_dict", None)
                if d is None:
                    d = {}
                    _rope_tls.rope_dict = d
                d[key] = value

            def clear(self):
                _rope_tls.rope_dict = {}

        _rope_mod._ROPE_DICT = _ThreadLocalRopeDict()

        # --- Patch distributed init to no-op ONCE from main thread ---
        # This avoids races where one worker's finally-restore undoes
        # another worker's patches mid-init.
        import vllm.distributed.parallel_state as _ps
        import vllm.v1.worker.gpu_worker as _gw

        self._orig_init_dist = _ps.init_distributed_environment
        self._orig_ensure_mp = _ps.ensure_model_parallel_initialized
        self._orig_init_worker_dist = _gw.init_worker_distributed_environment
        self._orig_set_car = _ps.set_custom_all_reduce
        _ps.init_distributed_environment = lambda *a, **kw: None
        _ps.ensure_model_parallel_initialized = lambda *a, **kw: None
        _gw.init_worker_distributed_environment = lambda *a, **kw: None
        _ps.set_custom_all_reduce = lambda *a, **kw: None

        self._threads = []
        for rank in range(tp_size):
            t = threading.Thread(
                target=self._worker_loop,
                args=(rank, tp_size, nccl_barrier, nccl_shared, ready_barrier),
                daemon=True,
                name=f"TPWorker-{rank}",
            )
            t.start()
            self._threads.append(t)

        ready_barrier.wait()

        # Restore patched functions now that all workers are initialized.
        _ps.init_distributed_environment = self._orig_init_dist
        _ps.ensure_model_parallel_initialized = self._orig_ensure_mp
        _gw.init_worker_distributed_environment = self._orig_init_worker_dist
        _ps.set_custom_all_reduce = self._orig_set_car

        # Check for worker init errors.
        for rank, err in enumerate(self._worker_errors):
            if err is not None:
                raise RuntimeError(
                    f"TPWorker-{rank} failed during initialization"
                ) from err

        logger.info("All %d TP worker threads initialized.", tp_size)

    @cached_property
    def max_concurrent_batches(self) -> int:
        return 2 if self.scheduler_config.async_scheduling else 1

    def _worker_loop(self, rank, tp_size, nccl_barrier, nccl_shared, ready_barrier):
        try:
            self._worker_init(rank, tp_size, nccl_barrier, nccl_shared)
        except Exception as e:
            logger.error("TPWorker-%d init failed: %s", rank, e, exc_info=True)
            self._worker_errors[rank] = e
            ready_barrier.wait()
            return

        ready_barrier.wait()

        cmd_q = self._cmd_queues[rank]
        result_q = self._result_queues[rank]
        worker = self._workers[rank]

        # --- Event loop: pull commands from queue ---
        while True:
            cmd = cmd_q.get()
            if cmd is None:  # shutdown sentinel
                break
            method, args, kwargs = cmd
            # Re-pin CUDA device before every command.  In free-threaded
            # Python, torch's per-thread device/stream state can drift;
            # re-setting ensures current_stream() returns this worker's
            # stream, not another thread's.
            torch.cuda.set_device(rank)
            try:
                result = run_method(worker, method, args, kwargs)
            except Exception as e:
                method_name = (
                    method
                    if isinstance(method, str)
                    else getattr(method, "__name__", repr(method))
                )
                logger.error(
                    "TPWorker-%d: %s failed: %s",
                    rank,
                    method_name,
                    e,
                    exc_info=True,
                )
                result = e
            result_q.put(result)

    def _worker_init(self, rank, tp_size, nccl_barrier, nccl_shared):
        """Initialize one TP worker thread (NCCL, parallel state, model)."""
        torch.cuda.set_device(rank)

        # Each worker needs its own vllm_config because mutable state
        # (e.g. compilation_config.static_forward_context) is populated
        # per-model during load_model and must not collide across workers.
        worker_config = copy.deepcopy(self.vllm_config)
        # Set the thread-local config directly (set_current_vllm_config is a
        # context manager, but we want it set for the thread's lifetime).
        _config_tls.config = worker_config

        # _ROPE_DICT is now thread-local (patched in _init_executor), so
        # each worker thread gets its own empty dict — no need to clear.

        # --- Set up NCCL communicator ---
        in_proc_group = InProcessGroup(rank, tp_size, nccl_barrier, nccl_shared)
        pynccl_comm = PyNcclCommunicator(
            group=in_proc_group, device=torch.device(f"cuda:{rank}")
        )

        # --- Build thread-local parallel state ---
        dev_comm = _PyNcclDeviceCommunicator(
            rank, tp_size, rank, pynccl_comm, unique_name="",
            collective_barrier=self._collective_barrier,
        )
        tp_group = _make_group_coordinator(
            rank=rank,
            world_size=tp_size,
            local_rank=rank,
            group_name="tp",
            device_communicator=dev_comm,
        )
        dev_comm.unique_name = tp_group.unique_name

        world_group = _make_group_coordinator(
            rank=rank,
            world_size=tp_size,
            local_rank=rank,
            group_name="world",
        )
        pp_group = _make_group_coordinator(
            rank=0,
            world_size=1,
            local_rank=rank,
            group_name="pp",
        )
        dcp_group = _make_group_coordinator(
            rank=0,
            world_size=1,
            local_rank=rank,
            group_name="dcp",
        )
        pcp_group = _make_group_coordinator(
            rank=0,
            world_size=1,
            local_rank=rank,
            group_name="pcp",
        )

        _ps_tls._TP = tp_group
        _ps_tls._PP = pp_group
        _ps_tls._WORLD = world_group
        _ps_tls._DCP = dcp_group
        _ps_tls._PCP = pcp_group

        # Keep strong references so _groups weak refs survive GC.
        self._group_coordinators.extend(
            [tp_group, world_group, pp_group, dcp_group, pcp_group]
        )

        # --- Create and initialize worker ---
        from vllm.utils.network_utils import (
            get_distributed_init_method,
            get_ip,
            get_open_port,
        )

        worker = WorkerWrapperBase(rpc_rank=0, global_rank=rank)
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        kwargs = dict(
            vllm_config=worker_config,
            local_rank=rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=(rank == 0),
            shared_worker_lock=Lock(),
        )
        worker.init_worker(all_kwargs=[kwargs])
        worker.init_device()
        worker.load_model()

        # Pre-convert RoPE cos_sin_cache buffers to the model dtype AND
        # correct device on the default stream.  The cos_sin_cache is
        # initially created on CPU (_compute_cos_sin_cache uses plain
        # torch.arange/einsum).  Moving it to the GPU here ensures
        # _match_cos_sin_cache_dtype() is a no-op during CUDA graph
        # capture (which would otherwise fail with a cross-stream error).
        try:
            model_dtype = worker_config.model_config.dtype
            target_device = torch.device(f"cuda:{rank}")
            inner_worker = worker.worker
            model_runner = inner_worker.model_runner
            model_obj = model_runner.model
            # Unwrap CudaGraphBatchRunner if present.
            inner = getattr(model_obj, "runnable", model_obj)
            count = 0
            for mod in inner.modules():
                cs = getattr(mod, "cos_sin_cache", None)
                if cs is not None and (
                    cs.dtype != model_dtype or cs.device != target_device
                ):
                    mod.cos_sin_cache = cs.to(target_device, dtype=model_dtype)
                    count += 1
            if count:
                logger.info(
                    "TPWorker-%d: pre-converted %d cos_sin_cache buffers "
                    "to %s on %s", rank, count, model_dtype, target_device,
                )
        except Exception as e:
            logger.warning(
                "TPWorker-%d: cos_sin_cache pre-conversion failed: %s",
                rank, e,
            )

        self._workers[rank] = worker

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # Enqueue command to all workers (non-blocking).
        cmd = (method, args, kwargs)
        for q in self._cmd_queues:
            q.put(cmd)

        def get_response():
            # Dequeue one result per worker (keeps queues aligned).
            results = []
            for i in range(self._tp_size):
                r = self._result_queues[i].get(timeout=120)
                if isinstance(r, Exception):
                    logger.error(
                        "TPWorker-%d returned exception for %s", i, method
                    )
                    raise r
                results.append(r)

            driver_result = results[0]
            # Unwrap AsyncModelRunnerOutput.
            if isinstance(driver_result, AsyncModelRunnerOutput):
                driver_result = driver_result.get_output()

            return driver_result if single_value else results

        if non_block:
            future = _FutureWrapper(self._futures_deque)
            self._futures_deque.appendleft((future, get_response))
            return future

        # Blocking: drain any pending non-block futures first.
        while self._futures_deque:
            fut, get_resp = self._futures_deque.pop()
            fut._resolve(get_resp)

        return get_response()

    def execute_model(self, scheduler_output, non_block=False):
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        if non_block and output.done():
            output.result()
        return output

    def sample_tokens(self, grammar_output=None, non_block=False):
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self):
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        return

    def shutdown(self) -> None:
        # Send shutdown sentinel to each worker.
        for q in self._cmd_queues:
            q.put(None)
        for t in self._threads:
            t.join(timeout=5)
        for w in self._workers:
            if w is not None:
                w.shutdown()
