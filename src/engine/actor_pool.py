"""
Actor Swarm: process-based worker pool for executing DAG tasks with zero-copy shared memory.

ActorHypervisor spawns N worker processes (based on os.cpu_count()). Each ModelActor
runs an infinite loop: read sub-tasks from a multiprocessing.Queue, reconstruct
images from shared memory metadata, run inference, and push results to a result queue.
No threading for inference; no pickle for image data.
"""

from __future__ import annotations

import logging
import os
import signal
from multiprocessing import Process, Queue
from typing import Any, Callable

from .shared_memory import attach_and_reconstruct

logger = logging.getLogger(__name__)

# Value-based sentinel so it survives pickle/unpickle across process boundary.
SHUTDOWN_SENTINEL = "__CPU_LLM_SHUTDOWN__"


def _default_inference(image: Any, instruction: str, task_id: str) -> dict[str, Any]:
    """Stub inference when no model is provided. Override via run_inference_hook."""
    return {
        "task_id": task_id,
        "instruction": instruction[:80],
        "output": "stub",
        "shape": getattr(image, "shape", None),
    }


def _worker_loop(
    task_queue: Queue,
    result_queue: Queue,
    worker_id: int,
    run_inference_hook: Callable[[Any, str, str], dict[str, Any]] | None,
) -> None:
    """
    Single worker process: read tasks from task_queue, attach to shared memory,
    run inference, push result to result_queue. Exits on SHUTDOWN_SENTINEL (None).
    """
    inference = run_inference_hook or _default_inference
    # Ignore SIGINT in workers so only the main process handles Ctrl+C
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
        image_metadata = task.get("image_metadata")
        shm_handle = None
        try:
            if image_metadata:
                shm_handle, arr = attach_and_reconstruct(image_metadata)
                result = inference(arr, instruction, task_id)
            else:
                result = inference(None, instruction, task_id)
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
    """
    Represents a single worker process that runs inference on sub-tasks.
    Not instantiated directly by users; ActorHypervisor spawns these.
    """

    def __init__(
        self,
        process: Process,
        worker_id: int,
    ):
        self.process = process
        self.worker_id = worker_id

    @property
    def is_alive(self) -> bool:
        return self.process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout=timeout)

    def terminate(self) -> None:
        self.process.terminate()


class ActorHypervisor:
    """
    Spawns and manages a pool of worker processes. Each worker listens to
    task_queue for sub-tasks (containing shared memory metadata, not raw images),
    reconstructs the image via zero-copy, runs inference, and sends results
    to result_queue. Graceful shutdown: send one None per worker, then join.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        run_inference_hook: Callable[[Any, str, str], dict[str, Any]] | None = None,
    ):
        """
        run_inference_hook: (image, instruction, task_id) -> result dict.
        On Windows (spawn) this must be a picklable callable (e.g. module-level function).
        """
        self._num_workers = num_workers or max(1, (os.cpu_count() or 1))
        self._run_inference_hook = run_inference_hook
        self._task_queue: Queue | None = None
        self._result_queue: Queue | None = None
        self._actors: list[ModelActor] = []
        self._started = False

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def start(self) -> None:
        """Spawn N worker processes. Idempotent."""
        if self._started:
            return
        self._task_queue = Queue()
        self._result_queue = Queue()
        self._actors = []
        for i in range(self._num_workers):
            p = Process(
                target=_worker_loop,
                args=(self._task_queue, self._result_queue, i, self._run_inference_hook),
                name=f"ModelActor-{i}",
            )
            p.start()
            self._actors.append(ModelActor(p, i))
        self._started = True
        logger.info("ActorHypervisor started with %s workers", self._num_workers)

    def submit(self, task: dict[str, Any]) -> None:
        """
        Submit a sub-task to the pool. Task must be a dict with at least
        task_id and instruction. If the task involves an image, include
        image_metadata (name, shape, dtype) from VisionMemoryManager.
        """
        if not self._started or self._task_queue is None:
            raise RuntimeError("ActorHypervisor not started; call start() first")
        self._task_queue.put(task)

    def get_result(self, block: bool = True, timeout: float | None = None) -> dict[str, Any] | None:
        """Get one result from the result queue. Returns None if timeout and block=False."""
        if self._result_queue is None:
            return None
        try:
            return self._result_queue.get(block=block, timeout=timeout)
        except Exception:
            return None

    def shutdown(self, timeout: float = 10.0) -> None:
        """
        Graceful shutdown: send one None sentinel per worker, then join.
        If workers do not exit within timeout, terminate them.
        """
        if not self._started or self._task_queue is None:
            self._started = False
            return
        for _ in range(self._num_workers):
            try:
                self._task_queue.put(SHUTDOWN_SENTINEL)
            except Exception:
                break
        for actor in self._actors:
            actor.join(timeout=max(0, timeout / max(1, len(self._actors))))
        for actor in self._actors:
            if actor.is_alive:
                logger.warning("Worker %s did not exit gracefully; terminating", actor.worker_id)
                actor.terminate()
                actor.join(timeout=2.0)
        self._actors = []
        self._task_queue = None
        self._result_queue = None
        self._started = False
        logger.info("ActorHypervisor shutdown complete")

    def __enter__(self) -> ActorHypervisor:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
