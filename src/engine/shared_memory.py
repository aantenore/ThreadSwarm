"""
Zero-Copy Shared Memory for vision payloads.

Images are loaded once, placed in a multiprocessing.shared_memory.SharedMemory
block, and workers receive only metadata (name, shape, dtype) to attach to the
same block and reconstruct the numpy array without copying the buffer.
"""

from __future__ import annotations

import logging
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# OpenCV is optional for file loading; numpy can still wrap raw buffers
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _load_image_cv2(path: str | Path) -> np.ndarray:
    path = str(path)
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required to load images from file; install opencv-python-headless")
    arr = cv2.imread(path)
    if arr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    # BGR -> RGB for typical ML pipelines; optional, can be made configurable
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(arr)


def _load_image_numpy(path: str | Path) -> np.ndarray:
    """Fallback: load as numpy .npy or raw (e.g. .bin with known shape)."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    raise ValueError(f"Cannot load image from {path} without OpenCV; use .npy or install opencv-python-headless")


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk into a contiguous numpy array (RGB, uint8 or float)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if HAS_CV2 and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return _load_image_cv2(path)
    return _load_image_numpy(path)


class VisionMemoryManager:
    """
    Manages zero-copy shared memory for image (and optionally tensor) data.

    - Allocates a SharedMemory block and copies image data into it.
    - Returns metadata (name, shape, dtype) so workers can attach and build
      a numpy array that views the same memory without copying.
    """

    def __init__(self, name_prefix: str = "vision_"):
        self._name_prefix = name_prefix
        self._shm: shared_memory.SharedMemory | None = None
        self._name: str | None = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None

    def load_and_share(self, image: np.ndarray | str | Path, name: str | None = None) -> dict[str, Any]:
        """
        Load an image (from array or path), put it in shared memory, and return
        metadata for workers to attach.

        :param image: Either a numpy array (contiguous) or a path to an image file.
        :param name: Optional name for the shared memory block; if None, a unique name is generated.
        :return: Dict with keys: name (str), shape (tuple), dtype (str), size (int).
                 Workers use this to reconstruct the array via attach_and_reconstruct.
        """
        if isinstance(image, (str, Path)):
            image = load_image(image)
        arr = np.asarray(image)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        size = arr.nbytes
        if name is None:
            name = f"{self._name_prefix}{id(arr)}_{size}"
        # SharedMemory names must be valid; strip any path chars
        name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:32]

        try:
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        except FileExistsError:
            # Name collision; use a more unique name
            import time
            name = f"{self._name_prefix}{int(time.time() * 1e6)}_{size}"
            name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:32]
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)

        np.copyto(np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf), arr)

        self._shm = shm
        self._name = name
        self._shape = arr.shape
        self._dtype = arr.dtype

        return {
            "name": name,
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "size": size,
        }

    def get_metadata(self) -> dict[str, Any] | None:
        """Return current shared block metadata if one has been created."""
        if self._name is None or self._shape is None or self._dtype is None:
            return None
        return {
            "name": self._name,
            "shape": self._shape,
            "dtype": str(self._dtype),
            "size": int(np.prod(self._shape) * self._dtype.itemsize),
        }

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
            self._shape = None
            self._dtype = None

    def __enter__(self) -> VisionMemoryManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def attach_and_reconstruct(metadata: dict[str, Any]) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    """
    Attach to an existing SharedMemory block using metadata from the manager
    and return the handle and a numpy array that views the same buffer (zero-copy).

    The caller must keep the SharedMemory reference until done with the array,
    then call shm.close() (and optionally unlink only from the creating process).

    :param metadata: Dict with "name", "shape", "dtype" (and optionally "size").
    :return: (shm, array) — keep shm referenced until you are done with the array.
    """
    name = metadata["name"]
    shape = tuple(metadata["shape"])
    dtype = np.dtype(metadata["dtype"])
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, arr
