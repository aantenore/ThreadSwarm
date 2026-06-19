"""ThreadSwarm public package exports."""

from importlib.metadata import PackageNotFoundError, version

from .config import ThreadSwarmConfig, ThreadSwarmConfigError

try:
    __version__ = version("ThreadSwarm")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ThreadSwarmConfig",
    "ThreadSwarmConfigError",
    "__version__",
]
