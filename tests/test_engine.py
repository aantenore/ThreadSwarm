"""Tests for shared memory and actor pool (no real model)."""

import numpy as np
import pytest

from src.engine.shared_memory import (
    ContextMemoryManager,
    VisionMemoryManager,
    attach_and_reconstruct,
    load_image,
)
from src.engine.actor_pool import ActorHypervisor, SHUTDOWN_SENTINEL


# Module-level callables, picklable on Windows. New hook signature: (context, instruction, task_id, modality, model_type).
def _stub_inference(context, instruction, task_id, modality, model_type):
    return {"task_id": task_id, "instruction": instruction, "done": True}


def _check_shape_inference(context, instruction, task_id, modality, model_type):
    return {"shape": getattr(context, "shape", None), "task_id": task_id}


def test_context_memory_manager_ndarray():
    """ContextMemoryManager: ndarray roundtrip (e.g. image/audio)."""
    manager = ContextMemoryManager(name_prefix="test_")
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    meta = manager.load_and_share(arr)
    assert "name" in meta and "shape" in meta and "dtype" in meta and meta.get("payload_type") == "ndarray"
    assert meta["shape"] == (64, 64, 3)
    shm, view = attach_and_reconstruct(meta)
    np.testing.assert_array_equal(view, arr)
    shm.close()
    manager.close()


def test_context_memory_manager_text():
    """ContextMemoryManager: text roundtrip."""
    manager = ContextMemoryManager(name_prefix="test_")
    text = "Hello world\nCode snippet: def foo(): pass"
    meta = manager.load_and_share(text)
    assert meta.get("payload_type") == "text"
    assert "size" in meta
    shm, reconstructed = attach_and_reconstruct(meta)
    assert reconstructed == text
    shm.close()
    manager.close()


def test_vision_memory_manager_alias():
    """VisionMemoryManager is alias for ContextMemoryManager; ndarray still works."""
    manager = VisionMemoryManager(name_prefix="test_")
    arr = np.ones((8, 8, 3), dtype=np.uint8) * 42
    meta = manager.load_and_share(arr)
    assert meta["shape"] == (8, 8, 3)
    shm, view = attach_and_reconstruct(meta)
    np.testing.assert_array_equal(view, arr)
    shm.close()
    manager.close()


def test_actor_pool_submit_and_result():
    """Homogeneous pool with stub hook; task can use context_metadata or image_metadata."""
    with ActorHypervisor(num_workers=2, run_inference_hook=_stub_inference) as hv:
        hv.submit({
            "task_id": "t1",
            "instruction": "Do something",
            "context_metadata": None,
        })
        out = hv.get_result(timeout=5.0)
        assert out is not None
        assert out.get("error") is None
        assert out.get("result", {}).get("task_id") == "t1"
        assert out["result"]["done"] is True


def test_actor_pool_with_shared_memory_context():
    """Submit task with context_metadata (ndarray); worker reconstructs and runs hook."""
    manager = ContextMemoryManager(name_prefix="test_")
    arr = np.ones((8, 8, 3), dtype=np.uint8) * 42
    meta = manager.load_and_share(arr)

    with ActorHypervisor(num_workers=1, run_inference_hook=_check_shape_inference) as hv:
        hv.submit({
            "task_id": "img_task",
            "instruction": "inspect",
            "context_metadata": meta,
        })
        out = hv.get_result(timeout=5.0)
        assert out is not None and out.get("error") is None
        assert out["result"]["shape"] == (8, 8, 3)

    manager.close()
