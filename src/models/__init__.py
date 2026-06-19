"""Optional model adapters for tasks that need more than local CPU-friendly tools."""

from .openai_compatible import OpenAICompatibleWorker, build_openai_compatible_worker

__all__ = ["OpenAICompatibleWorker", "build_openai_compatible_worker"]
