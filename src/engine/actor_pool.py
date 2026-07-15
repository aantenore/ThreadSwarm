"""
Actor Swarm: process-based heterogeneous worker pool for DAG execution with shared memory.

ActorHypervisor can run a single homogeneous pool or a heterogeneous set of pools
(keyed by route key such as tool_name or model_type). Each pool has its own task
queue and run_inference_hook. Tasks carry context_metadata from
ContextMemoryManager; workers attach to the shared block instead of receiving the
context through task queues. NumPy arrays are reconstructed zero-copy; text and
bytes are materialized per worker. No threading is used for inference.
"""

from __future__ import annotations

import logging
import os
import pickle
import queue
import signal
import threading
import time
from multiprocessing import resource_tracker
from multiprocessing import Process, Queue
from multiprocessing.reduction import ForkingPickler
from typing import Any, Callable

from .shared_memory import attach_and_reconstruct

logger = logging.getLogger(__name__)

SHUTDOWN_SENTINEL = "__ThreadSwarm_SHUTDOWN__"

# Signature for inference hook: (context_payload, instruction, task_id, modality, model_type) -> result dict
InferenceHook = Callable[[Any, str, str, str, str | None], dict[str, Any]]


def _create_process_queue() -> Queue:
    """Create a queue whose feeder exits quietly if a forced teardown closes its reader."""
    queue_ = Queue()
    # multiprocessing.Queue has no public abort API. Setting this before the
    # first put is the stdlib-supported internal switch used by its feeder to
    # treat EPIPE as terminal when a killed consumer can no longer drain data.
    if hasattr(queue_, "_ignore_epipe"):
        queue_._ignore_epipe = True  # type: ignore[attr-defined]
    return queue_


class UnknownRouteError(ValueError):
    """Raised when a heterogeneous pool cannot resolve a requested route."""


class ConcurrentRunError(RuntimeError):
    """Raised when two orchestrators try to share one hypervisor concurrently."""


def _serialize_queue_payload(payload: Any, label: str) -> bytes:
    """Serialize before queue feeder threads can outlive borrowed buffers."""
    try:
        return bytes(ForkingPickler.dumps(payload))
    except Exception as exc:
        raise ValueError(f"{label} is not multiprocessing-serializable: {exc}") from exc


def _result_event(
    task: dict[str, Any],
    *,
    result: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a correlated worker result event."""
    return {
        "run_id": task.get("run_id"),
        "attempt_id": task.get("attempt_id"),
        "task_id": task.get("task_id", ""),
        "attempt": task.get("attempt"),
        "result": result,
        "error": error,
    }


def _put_result(result_queue: Queue, task: dict[str, Any], result: Any) -> None:
    """Publish a result or a correlated serialization error instead of hanging."""
    event = _result_event(task, result=result)
    try:
        serialized_event = _serialize_queue_payload(event, "Worker result")
    except ValueError as exc:
        event = _result_event(task, error=str(exc))
        serialized_event = _serialize_queue_payload(event, "Worker error result")
    result_queue.put(serialized_event)


def _build_execution_context(task: dict[str, Any], payload: Any) -> Any:
    """
    Return either the raw payload (backward-compatible) or a structured execution
    context for DAG orchestration.
    """
    if not task.get("context_envelope"):
        return payload
    return {
        "payload": payload,
        "dependency_results": task.get("dependency_results", {}),
        "task_id": task.get("task_id", ""),
        "modality": task.get("modality", "text"),
        "tool_name": task.get("tool_name"),
        "model_type": task.get("model_type"),
        "attempt": task.get("attempt"),
    }


def _default_inference(
    context: Any,
    instruction: str,
    task_id: str,
    modality: str,
    model_type: str | None,
) -> dict[str, Any]:
    """Stub when no hook is provided."""
    return {
        "task_id": task_id,
        "instruction": instruction[:80],
        "output": "stub",
        "modality": modality,
        "model_type": model_type,
        "context_preview": (
            getattr(context, "shape", None) if hasattr(context, "shape") else (str(context)[:100] if context else None)
        ),
    }


def _worker_loop(
    task_queue: Queue,
    result_queue: Queue,
    lifecycle_queue: Queue,
    worker_id: int,
    model_type: str,
    run_inference_hook: InferenceHook | None,
) -> None:
    """
    Worker process: read tasks, attach to context via ContextMemoryManager metadata,
    run inference hook, push result. Exits on SHUTDOWN_SENTINEL.
    """
    inference = run_inference_hook or _default_inference
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, AttributeError):
        pass

    while True:
        try:
            queued_item = task_queue.get()
        except Exception as e:
            logger.warning("Worker %s get error: %s", worker_id, e)
            continue

        if queued_item == SHUTDOWN_SENTINEL:
            logger.debug("Worker %s received shutdown sentinel", worker_id)
            break

        try:
            task = pickle.loads(queued_item)
        except Exception as exc:
            logger.error("Worker %s received an invalid task envelope: %s", worker_id, exc)
            continue

        task_id = task.get("task_id", "")
        instruction = task.get("instruction", "")
        modality = task.get("modality", "text")
        task_model_type = task.get("tool_name") or task.get("model_type") or model_type
        context_metadata = task.get("context_metadata") or task.get("image_metadata")
        shm_handle = None
        try:
            if context_metadata:
                shm_handle, payload = attach_and_reconstruct(context_metadata)
            else:
                payload = None
            lifecycle_queue.put(
                {
                    "event": "started",
                    "run_id": task.get("run_id"),
                    "attempt_id": task.get("attempt_id"),
                    "task_id": task_id,
                    "attempt": task.get("attempt"),
                    "worker_id": worker_id,
                }
            )
            result = inference(_build_execution_context(task, payload), instruction, task_id, modality, task_model_type)
            _put_result(result_queue, task, result)
        except Exception as e:
            logger.exception("Worker %s task %s failed: %s", worker_id, task_id, e)
            result_queue.put(
                _serialize_queue_payload(
                    _result_event(task, error=str(e)),
                    "Worker error result",
                )
            )
        finally:
            if shm_handle is not None:
                try:
                    shm_handle.close()
                except Exception:
                    pass

    logger.debug("Worker %s exiting", worker_id)


class ModelActor:
    """Single worker process for a given route key. Spawned by ActorHypervisor."""

    def __init__(self, process: Process, worker_id: int, model_type: str):
        self.process = process
        self.worker_id = worker_id
        self.model_type = model_type

    @property
    def is_alive(self) -> bool:
        return self.process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout=timeout)

    def terminate(self) -> None:
        self.process.terminate()

    def kill(self) -> None:
        kill = getattr(self.process, "kill", None)
        if kill is not None:
            kill()
        else:  # pragma: no cover - Python implementations without Process.kill
            self.process.terminate()

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode


class ActorHypervisor:
    """
    Heterogeneous actor pool: spawns one or more worker pools by route key, each
    with its own task queue and inference hook. Tasks carry context_metadata
    (from ContextMemoryManager); workers reconstruct and run the hook. Single
    result queue for all pools. Submit with task["tool_name"] or task["model_type"].
    """

    def __init__(
        self,
        num_workers: int | None = None,
        run_inference_hook: InferenceHook | None = None,
        worker_configs: list[dict[str, Any]] | None = None,
    ):
        """
        Either (num_workers, run_inference_hook) for one homogeneous pool, or
        worker_configs for heterogeneous pools. worker_configs: list of
        {"model_type": str, "num_workers": int, "run_inference_hook": callable}.
        The config key remains "model_type" for backward compatibility, but it
        may represent any route key, including a local tool name. Hooks must be
        picklable (e.g. module-level) on Windows.
        """
        self._result_queue: Queue | None = None
        self._lifecycle_queue: Queue | None = None
        self._started = False
        self._pools: list[
            tuple[str, Queue, list[ModelActor], int]
        ] = []  # (model_type, task_queue, actors, num_workers)
        self._run_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._active_run_id: str | None = None
        self._stopping = False
        self._generation = 0
        self._legacy_homogeneous = worker_configs is None

        if worker_configs is not None:
            if not worker_configs:
                raise ValueError("worker_configs must contain at least one worker pool")
            self._worker_configs = [dict(config) for config in worker_configs]
        else:
            n = max(1, (os.cpu_count() or 1)) if num_workers is None else num_workers
            self._worker_configs = [
                {"model_type": "default", "num_workers": n, "run_inference_hook": run_inference_hook}
            ]
        self._validate_worker_configs()
        self._num_workers = sum(config["num_workers"] for config in self._worker_configs)
        self._routes = frozenset(config["model_type"] for config in self._worker_configs)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def started(self) -> bool:
        """Whether worker processes have been started."""
        return self._started

    @property
    def active_run_id(self) -> str | None:
        """ID of the orchestrated run currently owning this hypervisor."""
        return self._active_run_id

    @property
    def routes(self) -> frozenset[str]:
        """Configured execution routes."""
        return self._routes

    @property
    def generation(self) -> int:
        """
        Monotonic pool generation, incremented after each successful start.

        Orchestrators can retain this value for the duration of a run and treat
        an unexpected change as invalidation of all attempts submitted to the
        previous process and queue generation.
        """
        return self._generation

    def _validate_worker_configs(self) -> None:
        seen_routes: set[str] = set()
        for config in self._worker_configs:
            route = config.get("model_type", "default")
            if not isinstance(route, str) or not route.strip():
                raise ValueError("Each worker config requires a non-empty string model_type route")
            if route in seen_routes:
                raise ValueError(f"Duplicate worker route: {route}")
            seen_routes.add(route)
            workers = config.get("num_workers", 1)
            if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
                raise ValueError(f"Worker route {route} requires num_workers to be a positive integer")
            hook = config.get("run_inference_hook")
            if hook is not None and not callable(hook):
                raise ValueError(f"Worker route {route} run_inference_hook must be callable")
            config["model_type"] = route
            config["num_workers"] = workers

    def acquire_run(self, run_id: str) -> None:
        """Exclusively bind one orchestrated run to this hypervisor."""
        if not self._run_lock.acquire(blocking=False):
            raise ConcurrentRunError(f"ActorHypervisor is already executing run {self._active_run_id or '<unknown>'}")
        self._active_run_id = run_id

    def release_run(self, run_id: str) -> None:
        """Release an orchestrated run binding."""
        if self._active_run_id != run_id:
            return
        self._active_run_id = None
        self._run_lock.release()

    def resolve_route(self, task: dict[str, Any], model_type: str | None = None) -> str:
        """Resolve a task route without executing it, failing closed for explicit pools."""
        requested_route = task.get("tool_name") or task.get("model_type") or model_type
        if self._legacy_homogeneous:
            return self._worker_configs[0]["model_type"]
        if not requested_route:
            available = ", ".join(sorted(self._routes))
            raise UnknownRouteError(f"Task requires an explicit route; available routes: {available}")
        if requested_route not in self._routes:
            available = ", ".join(sorted(self._routes))
            raise UnknownRouteError(f"Unknown worker route {requested_route!r}; available routes: {available}")
        return requested_route

    def start(self) -> None:
        """Spawn all worker processes (one pool per worker_config). Idempotent."""
        with self._lifecycle_lock:
            if self._started:
                return
            if os.name == "posix":
                # With fork, a prestarted worker that later attaches to the
                # parent's first SharedMemory block can otherwise launch an
                # independent tracker and unlink the parent's block on worker
                # termination. Start the parent's tracker before any fork so
                # every generation inherits the same ownership process.
                resource_tracker.ensure_running()
            self._result_queue = _create_process_queue()
            self._lifecycle_queue = _create_process_queue()
            self._pools = []
            try:
                for pool_index, cfg in enumerate(self._worker_configs):
                    model_type = cfg["model_type"]
                    n = cfg["num_workers"]
                    hook = cfg.get("run_inference_hook")
                    task_queue = _create_process_queue()
                    actors = []
                    self._pools.append((model_type, task_queue, actors, n))
                    for i in range(n):
                        worker_id = pool_index * 100 + i
                        process = Process(
                            target=_worker_loop,
                            args=(task_queue, self._result_queue, self._lifecycle_queue, worker_id, model_type, hook),
                            name=f"ModelActor-{model_type}-{i}",
                        )
                        process.start()
                        actors.append(ModelActor(process, worker_id, model_type))
            except Exception:
                self._dispose_pools(force=True, timeout=2.0)
                raise
            self._started = True
            self._generation += 1
            logger.info(
                "ActorHypervisor started with %s pools, %s total workers",
                len(self._pools),
                self._num_workers,
            )

    def submit(self, task: dict[str, Any], model_type: str | None = None) -> None:
        """
        Submit a sub-task. Task must have task_id, instruction; optionally
        context_metadata (from ContextMemoryManager), modality, tool_name, model_type.
        Routes to the pool for task.get("tool_name"), task.get("model_type"),
        model_type, or "default".
        """
        serialized_task = _serialize_queue_payload(task, "Submitted task")
        with self._lifecycle_lock:
            if not self._started or self._result_queue is None:
                raise RuntimeError("ActorHypervisor not started; call start() first")
            task_run_id = task.get("run_id")
            if self._active_run_id is not None and task_run_id != self._active_run_id:
                raise ConcurrentRunError(
                    f"ActorHypervisor is owned by run {self._active_run_id}; "
                    f"cannot submit work for {task_run_id or '<unscoped>'}"
                )
            route = self.resolve_route(task, model_type)
            for mt, tq, _actors, _ in self._pools:
                if mt == route:
                    tq.put(serialized_task)
                    return
            raise UnknownRouteError(f"Worker route {route!r} is configured but not running")

    def get_result(self, block: bool = True, timeout: float | None = None) -> dict[str, Any] | None:
        """Get one result from the shared result queue."""
        result_queue = self._result_queue
        if result_queue is None:
            return None
        try:
            serialized_result = result_queue.get(block=block, timeout=timeout)
            return pickle.loads(serialized_result)
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            # A concurrent restart may replace the queue and return the pool to
            # started=True before this consumer observes the old queue closing.
            # Treat closure of a superseded queue as lifecycle invalidation; the
            # orchestrator will diagnose the generation change on its next poll.
            if result_queue is not self._result_queue or self._stopping or not self._started:
                return None
            raise

    def unexpected_worker_deaths(self) -> list[dict[str, Any]]:
        """Return workers that exited while the hypervisor was expected to be live."""
        with self._lifecycle_lock:
            if not self._started or self._stopping:
                return []
            return [
                {
                    "worker_id": actor.worker_id,
                    "route": route,
                    "exitcode": actor.exitcode,
                }
                for route, _task_queue, actors, _count in self._pools
                for actor in actors
                if actor.exitcode is not None
            ]

    def get_lifecycle_event(self, block: bool = False, timeout: float | None = None) -> dict[str, Any] | None:
        """Get a worker lifecycle event such as an attempt-start acknowledgement."""
        lifecycle_queue = self._lifecycle_queue
        if lifecycle_queue is None:
            return None
        try:
            return lifecycle_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            if lifecycle_queue is not self._lifecycle_queue or self._stopping or not self._started:
                return None
            raise

    def shutdown(self, timeout: float = 10.0, *, force: bool = False) -> None:
        """Stop all workers, optionally terminating in-flight work immediately."""
        with self._lifecycle_lock:
            if not self._started and not self._pools:
                self._started = False
                return
            self._dispose_pools(force=force, timeout=timeout)
            logger.info("ActorHypervisor shutdown complete")

    def restart(self, timeout: float = 2.0) -> None:
        """Discard all queues and workers, then create a fresh pool generation."""
        with self._lifecycle_lock:
            self.shutdown(timeout=timeout, force=True)
            self.start()

    def _dispose_pools(self, *, force: bool, timeout: float) -> None:
        self._stopping = True
        actors = [actor for _route, _queue, pool_actors, _count in self._pools for actor in pool_actors]
        # A worker that was already gone before shutdown cannot consume its
        # queue. Treat that path as forced even if the caller requested a
        # graceful stop, otherwise Queue.join_thread() can block forever.
        forced_cleanup = force or any(actor.exitcode is not None for actor in actors)
        try:
            if force:
                for actor in actors:
                    if actor.is_alive:
                        actor.terminate()
            else:
                for _route, task_queue, _actors, count in self._pools:
                    for _ in range(count):
                        try:
                            task_queue.put(SHUTDOWN_SENTINEL)
                        except Exception:
                            break

            deadline = time.monotonic() + max(0.0, timeout)
            for actor in actors:
                actor.join(timeout=max(0.0, deadline - time.monotonic()))
            for actor in actors:
                if actor.is_alive:
                    forced_cleanup = True
                    logger.warning("Worker %s did not exit; terminating", actor.worker_id)
                    actor.terminate()
            for actor in actors:
                actor.join(timeout=0.5)
            for actor in actors:
                if actor.is_alive:
                    forced_cleanup = True
                    logger.error("Worker %s ignored termination; killing", actor.worker_id)
                    actor.kill()
                    actor.join(timeout=0.5)
            if any(actor.exitcode not in (None, 0) for actor in actors):
                forced_cleanup = True

            for _route, task_queue, _actors, _count in self._pools:
                # All consumers have exited, so a task queue is disposable even
                # after a graceful stop. Aborting it avoids a join race if a
                # worker died between the liveness checks above.
                self._abort_queue(task_queue)
            if self._result_queue is not None:
                self._close_queue(self._result_queue, force=forced_cleanup)
            if self._lifecycle_queue is not None:
                self._close_queue(self._lifecycle_queue, force=forced_cleanup)
        finally:
            self._pools = []
            self._result_queue = None
            self._lifecycle_queue = None
            self._started = False
            self._stopping = False

    @staticmethod
    def _close_queue(queue_: Queue, *, force: bool) -> None:
        if force:
            ActorHypervisor._abort_queue(queue_)
            return
        try:
            queue_.close()
            queue_.join_thread()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _abort_queue(queue_: Queue) -> None:
        """Boundedly release a queue whose consumer may have been killed."""
        ActorHypervisor._clear_queue_buffer(queue_)
        try:
            queue_.cancel_join_thread()
        except (OSError, ValueError):
            pass
        try:
            queue_.close()
        except (OSError, ValueError):
            pass

        feeder = getattr(queue_, "_thread", None)
        if feeder is not None:
            # Give an unblocked feeder a chance to consume close()'s sentinel.
            # Closing its reader concurrently with that normal path can race
            # the feeder's own reader_close() and produce EBADF.
            feeder.join(timeout=0.05)
        if feeder is not None and feeder.is_alive():
            # A feeder blocked on a full pipe cannot reach the sentinel. Closing
            # the local reader turns that write into EPIPE; _ignore_epipe (set
            # at construction) then ends the daemon feeder quietly.
            reader = getattr(queue_, "_reader", None)
            if reader is not None:
                try:
                    reader.close()
                except (OSError, ValueError):
                    pass
            if os.name == "nt":
                writer = getattr(queue_, "_writer", None)
                if writer is not None:
                    try:
                        writer.close()
                    except (OSError, ValueError):
                        pass
            feeder.join(timeout=0.45)
            if feeder.is_alive():
                logger.warning("Queue feeder did not exit during forced teardown")

        # Once the feeder is gone, release any serialized payloads it could not
        # send. This is important for reusable in-process hypervisors.
        buffer = getattr(queue_, "_buffer", None)
        if buffer is not None and (feeder is None or not feeder.is_alive()):
            buffer.clear()
        writer = getattr(queue_, "_writer", None)
        if writer is not None:
            try:
                writer.close()
            except (OSError, ValueError):
                pass

    @staticmethod
    def _clear_queue_buffer(queue_: Queue) -> None:
        """Drop unsent payload references from a queue that will never be reused."""
        buffer = getattr(queue_, "_buffer", None)
        not_empty = getattr(queue_, "_notempty", None)
        if buffer is None:
            return
        if not_empty is None:
            buffer.clear()
            return
        with not_empty:
            buffer.clear()

    def __enter__(self) -> ActorHypervisor:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
