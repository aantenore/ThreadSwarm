"""Typed runtime configuration for ThreadSwarm."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_LLM_TIMEOUT = 60.0


class ThreadSwarmConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


@dataclass(frozen=True, slots=True)
class ThreadSwarmConfig:
    """Runtime settings that can vary by machine, provider, or deployment."""

    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout: float = DEFAULT_LLM_TIMEOUT
    default_workers: int | None = None

    def __post_init__(self) -> None:
        if not self.llm_base_url.strip():
            raise ThreadSwarmConfigError("llm_base_url cannot be empty")
        if not self.llm_model.strip():
            raise ThreadSwarmConfigError("llm_model cannot be empty")
        if self.llm_timeout <= 0:
            raise ThreadSwarmConfigError("llm_timeout must be greater than 0")
        if self.default_workers is not None and self.default_workers <= 0:
            raise ThreadSwarmConfigError("default_workers must be greater than 0 when set")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        prefix: str = "THREADSWARM_",
    ) -> "ThreadSwarmConfig":
        """Build config from environment-like mapping with safe local defaults."""
        source = os.environ if env is None else env
        return cls(
            llm_base_url=_get_text(source, f"{prefix}LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            llm_model=_get_text(source, f"{prefix}LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_timeout=_get_float(source, f"{prefix}LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT),
            default_workers=_get_optional_int(source, f"{prefix}DEFAULT_WORKERS"),
        )

    def compiler_kwargs(self) -> dict[str, str | float]:
        """Return keyword arguments accepted by SemanticCompiler."""
        return {
            "base_url": self.llm_base_url,
            "model": self.llm_model,
            "timeout": self.llm_timeout,
        }


def _get_text(source: Mapping[str, str], key: str, default: str) -> str:
    value = source.get(key, default)
    return value.strip()


def _get_float(source: Mapping[str, str], key: str, default: float) -> float:
    raw = source.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ThreadSwarmConfigError(f"{key} must be a number") from exc


def _get_optional_int(source: Mapping[str, str], key: str) -> int | None:
    raw = source.get(key)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ThreadSwarmConfigError(f"{key} must be an integer") from exc
