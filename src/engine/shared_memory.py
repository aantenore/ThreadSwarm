"""
Shared-memory transport for generic multimodal context payloads.

ContextMemoryManager allocates multiprocessing.shared_memory.SharedMemory blocks
for large payloads (images, audio buffers, text, code, binary data). Workers
receive only metadata instead of the context over multiprocessing queues.
Reconstructed ndarray payloads are read-only zero-copy views; text and bytes are
copied from the shared block when reconstructed as Python objects.
"""

from __future__ import annotations

import logging
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# OpenCV optional for image loading
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Payload type identifiers for reconstruction
PAYLOAD_NDARRAY = "ndarray"
PAYLOAD_TEXT = "text"
PAYLOAD_BYTES = "bytes"

# SharedMemory rejects zero-byte allocations. Empty logical payloads use one
# physical byte while their metadata keeps the correct logical size of zero.
_MINIMUM_ALLOCATION_SIZE = 1

# macOS limits POSIX shared-memory names to 30 user-supplied bytes (Python adds
# the leading slash). Generated names stay within that portable boundary.
_PORTABLE_SHARED_MEMORY_NAME_BYTES = 30

# Structured dtypes cannot be represented losslessly with ``str(dtype)``.  In
# particular, NumPy's string form is not a stable input to ``np.dtype`` and can
# discard offsets/alignment.  Keep scalar dtypes as their backwards-compatible
# string representation and use a recursive, JSON-compatible schema only when
# structure needs to be preserved.
_DTYPE_SCHEMA_KEY = "threadswarm_dtype"
_DTYPE_SCHEMA_VERSION = 1
_DTYPE_KIND_STRUCTURED = "structured"
_DTYPE_KIND_SUBARRAY = "subarray"


def _validate_shareable_dtype(dtype: np.dtype[Any]) -> None:
    if dtype.hasobject:
        raise TypeError(
            "Object-dtype ndarrays cannot be shared safely; use a numeric, boolean, string, or structured dtype"
        )


def _serialize_dtype(dtype: np.dtype[Any]) -> str | dict[str, Any]:
    """Return a lossless metadata representation for a NumPy dtype."""
    if dtype.fields is not None:
        fields: list[dict[str, Any]] = []
        for field_name in dtype.names or ():
            field_info = dtype.fields[field_name]
            field: dict[str, Any] = {
                "name": field_name,
                "dtype": _serialize_dtype(field_info[0]),
                "offset": int(field_info[1]),
            }
            if len(field_info) > 2:
                field["title"] = field_info[2]
            fields.append(field)
        return {
            _DTYPE_SCHEMA_KEY: _DTYPE_SCHEMA_VERSION,
            "kind": _DTYPE_KIND_STRUCTURED,
            "fields": fields,
            "itemsize": int(dtype.itemsize),
            "aligned": bool(dtype.isalignedstruct),
        }

    if dtype.subdtype is not None:
        base_dtype, shape = dtype.subdtype
        return {
            _DTYPE_SCHEMA_KEY: _DTYPE_SCHEMA_VERSION,
            "kind": _DTYPE_KIND_SUBARRAY,
            "base": _serialize_dtype(base_dtype),
            "shape": [int(dimension) for dimension in shape],
        }

    return dtype.str


def _deserialize_dtype(encoded: Any) -> np.dtype[Any]:
    """Decode ThreadSwarm dtype metadata while accepting legacy dtype specs."""
    if not isinstance(encoded, dict) or _DTYPE_SCHEMA_KEY not in encoded:
        return np.dtype(encoded)

    if encoded.get(_DTYPE_SCHEMA_KEY) != _DTYPE_SCHEMA_VERSION:
        raise ValueError("Unsupported ThreadSwarm dtype metadata version")

    kind = encoded.get("kind")
    if kind == _DTYPE_KIND_SUBARRAY:
        base_dtype = _deserialize_dtype(encoded["base"])
        shape = tuple(int(dimension) for dimension in encoded["shape"])
        return np.dtype((base_dtype, shape))

    if kind == _DTYPE_KIND_STRUCTURED:
        encoded_fields = encoded["fields"]
        names: list[str] = []
        formats: list[np.dtype[Any]] = []
        offsets: list[int] = []
        titles: list[Any] = []
        has_titles = False
        for field in encoded_fields:
            names.append(field["name"])
            formats.append(_deserialize_dtype(field["dtype"]))
            offsets.append(int(field["offset"]))
            title = field.get("title")
            titles.append(title)
            has_titles = has_titles or title is not None

        dtype_spec: dict[str, Any] = {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(encoded["itemsize"]),
        }
        if has_titles:
            dtype_spec["titles"] = titles
        return np.dtype(dtype_spec, align=bool(encoded.get("aligned", False)))

    raise ValueError(f"Unsupported ThreadSwarm dtype metadata kind: {kind!r}")


def _load_image_cv2(path: str | Path) -> np.ndarray:
    if not HAS_CV2:
        raise RuntimeError("OpenCV is required for common image formats; install ThreadSwarm[vision]")
    arr = cv2.imread(str(path))
    if arr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(arr)


def _load_image_numpy(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    raise ValueError(f"Cannot load image from {path} without OpenCV; use .npy or install ThreadSwarm[vision]")


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk into a contiguous numpy array (RGB)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if HAS_CV2 and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return _load_image_cv2(path)
    return _load_image_numpy(path)


class ContextMemoryManager:
    """
    Manages shared-memory transport for generic large context payloads:
    images (ndarray), text, code, audio buffers, and binary blobs.

    - Allocates a SharedMemory block and copies payload into it.
    - Returns metadata (name, size, payload_type, and type-specific fields) so
      workers can attach without transferring the context through task queues.
    - Reconstructs ndarrays as read-only zero-copy views. Reconstructing text or
      bytes creates a per-worker Python object copy.
    - Owns at most one block; loading another payload closes and unlinks the
      previously owned block first.
    """

    def __init__(self, name_prefix: str = "ctx_"):
        self._name_prefix = name_prefix
        self._shm: shared_memory.SharedMemory | None = None
        self._name: str | None = None
        self._metadata: dict[str, Any] | None = None

    def _unique_name(self, size: int) -> str:
        safe_prefix = (
            "".join(c if c.isascii() and (c.isalnum() or c in "_-") else "_" for c in self._name_prefix)
            or "ctx"
        )
        suffix = f"_{uuid.uuid4().hex[:12]}_{size:x}"
        suffix = suffix[-(_PORTABLE_SHARED_MEMORY_NAME_BYTES - 1) :]
        prefix_budget = _PORTABLE_SHARED_MEMORY_NAME_BYTES - len(suffix.encode("utf-8"))
        return f"{safe_prefix[:max(1, prefix_budget)]}{suffix}"

    def load_and_share(
        self,
        payload: np.ndarray | str | bytes | bytearray | memoryview,
        name: str | None = None,
    ) -> dict[str, Any]:
        """
        Put a large payload into shared memory and return metadata for workers.

        :param payload: ndarray (e.g. image, audio), str (text/code), or bytes-like.
        :param name: Optional shared memory block name; if None, a unique name is generated.
        :return: Metadata dict: name, size, payload_type, and type-specific keys
                 (shape, dtype for ndarray; encoding for text). Use with
                 attach_and_reconstruct.
        """
        if isinstance(payload, np.ndarray):
            normalized_payload: np.ndarray | str | bytes = payload
        elif isinstance(payload, str):
            normalized_payload = payload
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            normalized_payload = bytes(payload) if isinstance(payload, (bytearray, memoryview)) else payload
        else:
            raise TypeError(f"Unsupported payload type: {type(payload)}")

        if isinstance(normalized_payload, np.ndarray):
            _validate_shareable_dtype(normalized_payload.dtype)

        self.close()
        if isinstance(normalized_payload, np.ndarray):
            return self._share_ndarray(normalized_payload, name)
        if isinstance(normalized_payload, str):
            return self._share_text(normalized_payload, name)
        return self._share_bytes(normalized_payload, name)

    def _create_shared_memory(
        self,
        logical_size: int,
        name: str | None,
    ) -> tuple[shared_memory.SharedMemory, str]:
        allocation_size = max(_MINIMUM_ALLOCATION_SIZE, logical_size)
        if name is not None:
            is_portable_name = all(character.isascii() and (character.isalnum() or character in "_-") for character in name)
            if (
                not name
                or len(name.encode("utf-8")) > _PORTABLE_SHARED_MEMORY_NAME_BYTES
                or not is_portable_name
            ):
                raise ValueError(
                    f"Shared-memory name must contain 1-{_PORTABLE_SHARED_MEMORY_NAME_BYTES} ASCII "
                    "letters, digits, underscores, or hyphens for cross-platform compatibility"
                )
        resolved_name = name or self._unique_name(logical_size)
        try:
            shm = shared_memory.SharedMemory(
                create=True,
                size=allocation_size,
                name=resolved_name,
            )
        except FileExistsError:
            resolved_name = self._unique_name(logical_size)
            shm = shared_memory.SharedMemory(
                create=True,
                size=allocation_size,
                name=resolved_name,
            )
        return shm, resolved_name

    def _share_ndarray(self, arr: np.ndarray, name: str | None) -> dict[str, Any]:
        arr = np.ascontiguousarray(arr)
        size = arr.nbytes
        shm, name = self._create_shared_memory(size, name)
        self._shm = shm
        self._name = name
        try:
            np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)
        except Exception:
            self.close()
            raise
        self._metadata = {
            "name": name,
            "size": size,
            "payload_type": PAYLOAD_NDARRAY,
            "shape": tuple(arr.shape),
            "dtype": _serialize_dtype(arr.dtype),
        }
        return dict(self._metadata)

    def _share_text(self, text: str, name: str | None) -> dict[str, Any]:
        data = text.encode("utf-8")
        return self._share_bytes(data, name, payload_type=PAYLOAD_TEXT, encoding="utf-8")

    def _share_bytes(
        self,
        data: bytes,
        name: str | None,
        payload_type: str = PAYLOAD_BYTES,
        encoding: str | None = None,
    ) -> dict[str, Any]:
        size = len(data)
        shm, name = self._create_shared_memory(size, name)
        self._shm = shm
        self._name = name
        try:
            shm.buf[:size] = data
        except Exception:
            self.close()
            raise
        self._metadata = {
            "name": name,
            "size": size,
            "payload_type": payload_type,
        }
        if encoding:
            self._metadata["encoding"] = encoding
        return dict(self._metadata)

    def get_metadata(self) -> dict[str, Any] | None:
        """Return metadata for the current shared block, if any."""
        return dict(self._metadata) if self._metadata else None

    def close(self) -> None:
        """Close and unlink the currently owned shared memory block."""
        shm = self._shm
        self._shm = None
        self._name = None
        self._metadata = None
        if shm is None:
            return

        # Unlink first so the name cannot be reused by new attachments even if
        # closing a platform handle encounters an exported-buffer error.
        try:
            shm.unlink()
        except (FileNotFoundError, OSError) as e:
            logger.debug("SharedMemory unlink: %s", e)
        try:
            shm.close()
        except (BufferError, OSError) as e:
            logger.debug("SharedMemory close: %s", e)

    def __enter__(self) -> ContextMemoryManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def attach_and_reconstruct(
    metadata: dict[str, Any],
) -> tuple[shared_memory.SharedMemory, np.ndarray | str | bytes]:
    """
    Attach to an existing SharedMemory block and reconstruct the payload.

    Ndarrays are returned as read-only zero-copy views. Text and bytes avoid
    queue transfer but are copied when materialized as Python objects. The caller
    must keep the SharedMemory reference until done, then close it.

    :param metadata: Dict from ContextMemoryManager.load_and_share
                     (name, size, payload_type, ...).
    :return: (shm, payload) — payload is ndarray (view), str (decoded text),
             or bytes (copy of buffer).
    """
    payload_type = metadata.get("payload_type", PAYLOAD_NDARRAY)

    if payload_type == PAYLOAD_NDARRAY:
        shape = tuple(metadata["shape"])
        dtype = _deserialize_dtype(metadata["dtype"])
        _validate_shareable_dtype(dtype)

    name = metadata["name"]
    size = metadata["size"]
    shm = shared_memory.SharedMemory(name=name)
    try:
        if payload_type == PAYLOAD_NDARRAY:
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            arr.setflags(write=False)
            return shm, arr
        if payload_type == PAYLOAD_TEXT:
            encoding = metadata.get("encoding", "utf-8")
            text = bytes(shm.buf[:size]).decode(encoding)
            return shm, text
        # PAYLOAD_BYTES or unknown: return bytes (one copy from shared buffer)
        data = bytes(shm.buf[:size])
        return shm, data
    except Exception:
        shm.close()
        raise


# Backwards compatibility alias
VisionMemoryManager = ContextMemoryManager
