"""Regression tests for shared-context ownership and reconstruction semantics."""

import json

from multiprocessing import shared_memory

import numpy as np
import pytest

from threadswarm.engine.actor_pool import ActorHypervisor
from threadswarm.engine.shared_memory import (
    ContextMemoryManager,
    attach_and_reconstruct,
)


def _inspect_ndarray_mutability(context, instruction, task_id, modality, model_type):
    try:
        context[0, 0] = 99
    except ValueError:
        mutation_rejected = True
    else:
        mutation_rejected = False
    return {
        "mutation_rejected": mutation_rejected,
        "writeable": context.flags.writeable,
    }


def test_empty_text_roundtrip() -> None:
    with ContextMemoryManager(name_prefix="tet_") as manager:
        metadata = manager.load_and_share("")
        assert metadata["size"] == 0

        shm, reconstructed = attach_and_reconstruct(metadata)
        try:
            assert reconstructed == ""
        finally:
            shm.close()


def test_empty_bytes_roundtrip() -> None:
    with ContextMemoryManager(name_prefix="teb_") as manager:
        metadata = manager.load_and_share(b"")
        assert metadata["size"] == 0

        shm, reconstructed = attach_and_reconstruct(metadata)
        try:
            assert reconstructed == b""
        finally:
            shm.close()


def test_empty_ndarray_roundtrip_is_read_only() -> None:
    original = np.empty((0, 3), dtype=np.float32)

    with ContextMemoryManager(name_prefix="tea_") as manager:
        metadata = manager.load_and_share(original)
        assert metadata["size"] == 0

        shm, reconstructed = attach_and_reconstruct(metadata)
        try:
            assert isinstance(reconstructed, np.ndarray)
            assert reconstructed.shape == original.shape
            assert reconstructed.dtype == original.dtype
            assert reconstructed.nbytes == 0
            assert reconstructed.flags.writeable is False
        finally:
            shm.close()


def test_loading_a_second_payload_unlinks_the_previous_owned_block() -> None:
    manager = ContextMemoryManager(name_prefix="tr_")
    try:
        first_metadata = manager.load_and_share(b"first")
        second_metadata = manager.load_and_share(b"second")

        assert manager.get_metadata() == second_metadata
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=first_metadata["name"])

        shm, reconstructed = attach_and_reconstruct(second_metadata)
        try:
            assert reconstructed == b"second"
        finally:
            shm.close()
    finally:
        manager.close()


def test_generated_names_are_portable_and_unique_with_a_long_prefix() -> None:
    first_manager = ContextMemoryManager(name_prefix="prefix-" * 40)
    second_manager = ContextMemoryManager(name_prefix="prefix-" * 40)
    try:
        first = first_manager.load_and_share(b"first")
        second = second_manager.load_and_share(b"second")

        assert first["name"] != second["name"]
        assert len(first["name"].encode("utf-8")) <= 30
        assert len(second["name"].encode("utf-8")) <= 30
    finally:
        first_manager.close()
        second_manager.close()


def test_explicit_non_portable_shared_memory_name_is_rejected() -> None:
    with ContextMemoryManager() as manager:
        with pytest.raises(ValueError, match="1-30 ASCII"):
            manager.load_and_share(b"payload", name="x" * 31)


def test_reconstructed_ndarray_view_rejects_mutation() -> None:
    original = np.arange(6, dtype=np.int64).reshape(2, 3)

    with ContextMemoryManager(name_prefix="tro_") as manager:
        metadata = manager.load_and_share(original)
        shm, reconstructed = attach_and_reconstruct(metadata)
        try:
            assert isinstance(reconstructed, np.ndarray)
            np.testing.assert_array_equal(reconstructed, original)
            assert reconstructed.flags.writeable is False
            with pytest.raises(ValueError, match="read-only"):
                reconstructed[0, 0] = 99
        finally:
            shm.close()


def test_object_dtype_ndarray_is_rejected_before_publishing_shared_pointers() -> None:
    payload = np.array([{"unsafe": "python object"}], dtype=object)

    with ContextMemoryManager(name_prefix="too_") as manager:
        with pytest.raises(TypeError, match="Object-dtype ndarrays cannot be shared safely"):
            manager.load_and_share(payload)
        assert manager.get_metadata() is None


def test_invalid_replacement_preserves_the_current_shared_payload() -> None:
    invalid = np.array([{"unsafe": "python object"}], dtype=object)

    with ContextMemoryManager(name_prefix="tip_") as manager:
        original_metadata = manager.load_and_share(b"still-valid")

        with pytest.raises(TypeError, match="Object-dtype ndarrays cannot be shared safely"):
            manager.load_and_share(invalid)

        assert manager.get_metadata() == original_metadata
        shm, reconstructed = attach_and_reconstruct(original_metadata)
        try:
            assert reconstructed == b"still-valid"
        finally:
            shm.close()


@pytest.mark.parametrize(
    "forged_dtype",
    [
        "O",
        [("unsafe", "O")],
        {"names": ["unsafe"], "formats": ["O"]},
        {
            "threadswarm_dtype": 1,
            "kind": "structured",
            "fields": [{"name": "unsafe", "dtype": "|O", "offset": 0}],
            "itemsize": np.dtype(object).itemsize,
            "aligned": False,
        },
    ],
)
def test_attach_rejects_forged_object_dtype_metadata(forged_dtype) -> None:
    with ContextMemoryManager(name_prefix="tfo_") as manager:
        valid_metadata = manager.load_and_share(np.array([1], dtype=np.uint64))
        forged_metadata = {
            **valid_metadata,
            "dtype": forged_dtype,
        }

        with pytest.raises(TypeError, match="Object-dtype ndarrays cannot be shared safely"):
            attach_and_reconstruct(forged_metadata)

        # Rejection must not disturb the manager-owned block or prevent a valid
        # attachment to it afterwards.
        shm, reconstructed = attach_and_reconstruct(valid_metadata)
        try:
            np.testing.assert_array_equal(reconstructed, np.array([1], dtype=np.uint64))
        finally:
            shm.close()


def test_structured_dtype_roundtrip_is_lossless_after_json_serialization() -> None:
    nested_dtype = np.dtype([("code", "S3"), ("weight", "<f4")])
    structured_dtype = np.dtype(
        {
            "names": ["id", "position", "details"],
            "formats": ["<i8", ("<f8", (2,)), nested_dtype],
            "offsets": [0, 8, 24],
            "titles": ["identifier", None, None],
            "itemsize": 32,
        },
        align=True,
    )
    original = np.zeros(2, dtype=structured_dtype)
    original["id"] = [7, 11]
    original["position"] = [[1.25, 2.5], [3.75, 5.0]]
    original["details"]["code"] = [b"one", b"two"]
    original["details"]["weight"] = [0.5, 1.5]

    with ContextMemoryManager(name_prefix="tsd_") as manager:
        metadata = manager.load_and_share(original)
        assert isinstance(metadata["dtype"], dict)

        # Simulate transport through a configuration/API boundary where tuples
        # become lists; the dtype schema must remain fully reconstructable.
        transported_metadata = json.loads(json.dumps(metadata))
        shm, reconstructed = attach_and_reconstruct(transported_metadata)
        try:
            assert isinstance(reconstructed, np.ndarray)
            assert reconstructed.dtype == structured_dtype
            assert reconstructed.dtype.isalignedstruct is True
            assert reconstructed.dtype.fields["id"][2] == "identifier"
            np.testing.assert_array_equal(reconstructed, original)
            assert reconstructed.flags.writeable is False
        finally:
            shm.close()


def test_actor_worker_receives_read_only_ndarray_view() -> None:
    original = np.arange(4, dtype=np.int64).reshape(2, 2)

    with ContextMemoryManager(name_prefix="taw_") as manager:
        metadata = manager.load_and_share(original)
        with ActorHypervisor(num_workers=1, run_inference_hook=_inspect_ndarray_mutability) as hypervisor:
            hypervisor.submit(
                {
                    "task_id": "inspect_mutability",
                    "instruction": "Inspect shared context mutability",
                    "context_metadata": metadata,
                }
            )
            outcome = hypervisor.get_result(timeout=5.0)

    assert outcome is not None
    assert outcome["error"] is None
    assert outcome["result"] == {
        "mutation_rejected": True,
        "writeable": False,
    }
