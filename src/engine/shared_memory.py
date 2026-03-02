"""
Zero-copy shared memory for generic multimodal context payloads.

ContextMemoryManager allocates multiprocessing.shared_memory.SharedMemory blocks
for large payloads (images, audio buffers, text, code, binary data). Workers
receive only metadata and attach to the same block to reconstruct the payload
without copying across processes. No pickle for context data.
"""

from __future__ import annotations

import logging
import time
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


def _load_image_cv2(path: str | Path) -> np.ndarray:
    if not HAS_CV2:
        raise RuntimeError("opencv-python required for image loading; install opencv-python-headless")
    arr = cv2.imread(str(path))
    if arr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(arr)


def _load_image_numpy(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    raise ValueError(f"Cannot load image from {path} without OpenCV; use .npy or install opencv-python-headless")


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
    Manages zero-copy shared memory for generic large context payloads:
    images (ndarray), text, code, audio buffers, binary blobs.

    - Allocates a SharedMemory block and copies payload into it.
    - Returns metadata (name, size, payload_type, and type-specific fields) so
      workers can attach and reconstruct without copying the buffer across processes.
    """

    def __init__(self, name_prefix: str = "ctx_"):
        self._name_prefix = name_prefix
        self._shm: shared_memory.SharedMemory | None = None
        self._name: str | None = None
        self._metadata: dict[str, Any] | None = None

    def _unique_name(self, size: int) -> str:
        name = f"{self._name_prefix}{int(time.time() * 1e6)}_{size}"
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:64]

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
                 (shape, dtype for ndarray; encoding for text). Use with attach_and_reconstruct.
        """
        if isinstance(payload, np.ndarray):
            return self._share_ndarray(payload, name)
        if isinstance(payload, str):
            return self._share_text(payload, name)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return self._share_bytes(bytes(payload) if isinstance(payload, (bytearray, memoryview)) else payload, name)
        raise TypeError(f"Unsupported payload type: {type(payload)}")

    def _share_ndarray(self, arr: np.ndarray, name: str | None) -> dict[str, Any]:
        arr = np.ascontiguousarray(arr)
        size = arr.nbytes
        name = name or self._unique_name(size)
        try:
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            name = self._unique_name(size)
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)
        self._shm = shm
        self._name = name
        self._metadata = {
            "name": name,
            "size": size,
            "payload_type": PAYLOAD_NDARRAY,
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
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
        name = name or self._unique_name(size)
        try:
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            name = self._unique_name(size)
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        shm.buf[:size] = data
        self._shm = shm
        self._name = name
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
        """Unlink the shared memory block (call when the owner is done)."""
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except (FileNotFoundError, OSError) as e:
                logger.debug("SharedMemory close/unlink: %s", e)
            self._shm = None
            self._name = None
            self._metadata = None

    def __enter__(self) -> ContextMemoryManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def attach_and_reconstruct(metadata: dict[str, Any]) -> tuple[shared_memory.SharedMemory, np.ndarray | str | bytes]:
    """
    Attach to an existing SharedMemory block and reconstruct the payload (zero-copy
    where possible). Caller must keep the SharedMemory reference until done, then close it.

    :param metadata: Dict from ContextMemoryManager.load_and_share (name, size, payload_type, ...).
    :return: (shm, payload) — payload is ndarray (view), str (decoded text), or bytes (copy of buffer).
    """
    name = metadata["name"]
    size = metadata["size"]
    payload_type = metadata.get("payload_type", PAYLOAD_NDARRAY)
    shm = shared_memory.SharedMemory(name=name)

    if payload_type == PAYLOAD_NDARRAY:
        shape = tuple(metadata["shape"])
        dtype = np.dtype(metadata["dtype"])
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return shm, arr
    if payload_type == PAYLOAD_TEXT:
        encoding = metadata.get("encoding", "utf-8")
        text = bytes(shm.buf[:size]).decode(encoding)
        return shm, text
    # PAYLOAD_BYTES or unknown: return bytes (one copy from shared buffer)
    data = bytes(shm.buf[:size])
    return shm, data


# Backwards compatibility alias
VisionMemoryManager = ContextMemoryManager
