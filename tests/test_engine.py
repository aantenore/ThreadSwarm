"""Tests for shared memory and actor pool (no real model)."""

import numpy as np
import pytest

from src.engine.shared_memory import (
    VisionMemoryManager,
    attach_and_reconstruct,
    load_image,
)
from src.engine.actor_pool import ActorHypervisor, SHUTDOWN_SENTINEL


# Module-level callables so they are picklable on Windows (spawn).
def _stub_inference(image, instruction, task_id):
    return {"task_id": task_id, "instruction": instruction, "done": True}


def _check_shape_inference(image, instruction, task_id):
    return {"shape": getattr(image, "shape", None), "task_id": task_id}


def test_vision_memory_manager_roundtrip():
    """Load array into shared memory and reconstruct in same process."""
    manager = VisionMemoryManager(name_prefix="test_")
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    meta = manager.load_and_share(arr)
    assert "name" in meta and "shape" in meta and "dtype" in meta
    assert meta["shape"] == (64, 64, 3)
    assert meta["dtype"] == "uint8"

    shm, view = attach_and_reconstruct(meta)
    np.testing.assert_array_equal(view, arr)
    shm.close()
    manager.close()


def test_actor_pool_submit_and_result():
    """Run hypervisor with stub inference; submit one task, get one result."""
    with ActorHypervisor(num_workers=2, run_inference_hook=_stub_inference) as hv:
        hv.submit({
            "task_id": "t1",
            "instruction": "Do something",
            "image_metadata": None,
        })
        out = hv.get_result(timeout=5.0)
        assert out is not None
        assert out.get("error") is None
        assert out.get("result", {}).get("task_id") == "t1"
        assert out["result"]["done"] is True


def test_actor_pool_with_shared_memory_image():
    """Submit a task that includes image_metadata; worker reconstructs and runs stub."""
    manager = VisionMemoryManager(name_prefix="test_")
    arr = np.ones((8, 8, 3), dtype=np.uint8) * 42
    meta = manager.load_and_share(arr)

    with ActorHypervisor(num_workers=1, run_inference_hook=_check_shape_inference) as hv:
        hv.submit({
            "task_id": "img_task",
            "instruction": "inspect",
            "image_metadata": meta,
        })
        out = hv.get_result(timeout=5.0)
        assert out is not None and out.get("error") is None
        assert out["result"]["shape"] == (8, 8, 3)

    manager.close()
