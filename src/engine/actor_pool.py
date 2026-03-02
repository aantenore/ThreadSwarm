"""
Actor Swarm: process-based heterogeneous worker pool for DAG execution with zero-copy shared memory.

ActorHypervisor can run a single homogeneous pool or a heterogeneous set of pools
(keyed by model_type). Each pool has its own task queue and run_inference_hook
(e.g. Gemma-1B for text, SmolVLM for vision). Tasks carry context_metadata from
ContextMemoryManager; workers reconstruct payloads (ndarray, text, bytes) without
copying. No threading for inference; no pickle for context payloads.
"""

from __future__ import annotations

import logging
import os
import signal
from multiprocessing import Process, Queue
from typing import Any, Callable

from .shared_memory import attach_and_reconstruct

logger = logging.getLogger(__name__)

SHUTDOWN_SENTINEL = "__ThreadSwarm_SHUTDOWN__"

# Signature for inference hook: (context_payload, instruction, task_id, modality, model_type) -> result dict
InferenceHook = Callable[[Any, str, str, str, str | None], dict[str, Any]]


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
            task = task_queue.get()
        except Exception as e:
            logger.warning("Worker %s get error: %s", worker_id, e)
            continue

        if task == SHUTDOWN_SENTINEL:
            logger.debug("Worker %s received shutdown sentinel", worker_id)
            break

        task_id = task.get("task_id", "")
        instruction = task.get("instruction", "")
        modality = task.get("modality", "text")
        task_model_type = task.get("model_type") or model_type
        context_metadata = task.get("context_metadata") or task.get("image_metadata")
        shm_handle = None
        try:
            if context_metadata:
                shm_handle, payload = attach_and_reconstruct(context_metadata)
                result = inference(payload, instruction, task_id, modality, task_model_type)
            else:
                result = inference(None, instruction, task_id, modality, task_model_type)
            result_queue.put({"task_id": task_id, "result": result, "error": None})
        except Exception as e:
            logger.exception("Worker %s task %s failed: %s", worker_id, task_id, e)
            result_queue.put({"task_id": task_id, "result": None, "error": str(e)})
        finally:
            if shm_handle is not None:
                try:
                    shm_handle.close()
                except Exception:
                    pass

    logger.debug("Worker %s exiting", worker_id)


class ModelActor:
    """Single worker process for a given model type. Spawned by ActorHypervisor."""

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


class ActorHypervisor:
    """
    Heterogeneous actor pool: spawns one or more worker pools by model_type, each
    with its own task queue and inference hook. Tasks carry context_metadata
    (from ContextMemoryManager); workers reconstruct and run the hook. Single
    result queue for all pools. Submit with task["model_type"] to route.
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
        Hooks must be picklable (e.g. module-level) on Windows.
        """
        self._result_queue: Queue | None = None
        self._started = False
        self._pools: list[tuple[str, Queue, list[ModelActor], int]] = []  # (model_type, task_queue, actors, num_workers)

        if worker_configs:
            self._worker_configs = list(worker_configs)
            self._num_workers = sum(c.get("num_workers", 1) for c in worker_configs)
        else:
            n = num_workers or max(1, (os.cpu_count() or 1))
            self._worker_configs = [{"model_type": "default", "num_workers": n, "run_inference_hook": run_inference_hook}]
            self._num_workers = n

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def start(self) -> None:
        """Spawn all worker processes (one pool per worker_config). Idempotent."""
        if self._started:
            return
        self._result_queue = Queue()
        self._pools = []
        for cfg in self._worker_configs:
            model_type = cfg.get("model_type", "default")
            n = cfg.get("num_workers", 1)
            hook = cfg.get("run_inference_hook")
            task_queue = Queue()
            actors = []
            for i in range(n):
                p = Process(
                    target=_worker_loop,
                    args=(task_queue, self._result_queue, len(self._pools) * 100 + i, model_type, hook),
                    name=f"ModelActor-{model_type}-{i}",
                )
                p.start()
                actors.append(ModelActor(p, len(self._pools) * 100 + i, model_type))
            self._pools.append((model_type, task_queue, actors, n))
        self._started = True
        logger.info("ActorHypervisor started with %s pools, %s total workers", len(self._pools), self._num_workers)

    def submit(self, task: dict[str, Any], model_type: str | None = None) -> None:
        """
        Submit a sub-task. Task must have task_id, instruction; optionally
        context_metadata (from ContextMemoryManager), modality, model_type.
        Routes to the pool for task.get("model_type") or model_type or "default".
        """
        if not self._started or self._result_queue is None:
            raise RuntimeError("ActorHypervisor not started; call start() first")
        route = task.get("model_type") or model_type or "default"
        for mt, tq, _actors, _ in self._pools:
            if mt == route:
                tq.put(task)
                return
        if self._pools:
            self._pools[0][1].put(task)
            return
        raise RuntimeError("No worker pools available")

    def get_result(self, block: bool = True, timeout: float | None = None) -> dict[str, Any] | None:
        """Get one result from the shared result queue."""
        if self._result_queue is None:
            return None
        try:
            return self._result_queue.get(block=block, timeout=timeout)
        except Exception:
            return None

    def shutdown(self, timeout: float = 10.0) -> None:
        """Send shutdown sentinel to each pool, join all actors."""
        if not self._started:
            self._started = False
            return
        for _model_type, task_queue, actors, n in self._pools:
            for _ in range(n):
                try:
                    task_queue.put(SHUTDOWN_SENTINEL)
                except Exception:
                    break
        per_pool = max(0.1, timeout / max(1, len(self._pools)))
        for _model_type, _tq, actors, _ in self._pools:
            for actor in actors:
                actor.join(timeout=per_pool)
        for _model_type, _tq, actors, _ in self._pools:
            for actor in actors:
                if actor.is_alive:
                    logger.warning("Worker %s did not exit gracefully; terminating", actor.worker_id)
                    actor.terminate()
                    actor.join(timeout=2.0)
        self._pools = []
        self._result_queue = None
        self._started = False
        logger.info("ActorHypervisor shutdown complete")

    def __enter__(self) -> ActorHypervisor:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
