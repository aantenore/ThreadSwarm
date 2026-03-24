"""
Registry for local CPU-friendly tools that can execute DAG tasks.

This keeps the execution model generic: a task can be routed to a plain Python
tool that runs on a normal PC, or to a model-backed worker when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actor_pool import ActorHypervisor, InferenceHook


@dataclass(slots=True)
class LocalToolSpec:
    """Definition of one local executable tool."""

    name: str
    run: InferenceHook
    num_workers: int = 1
    description: str = ""
    modalities: tuple[str, ...] = field(default_factory=tuple)


class LocalToolRegistry:
    """Register local tools and expose them as worker configs for ActorHypervisor."""

    def __init__(self):
        self._tools: dict[str, LocalToolSpec] = {}

    def register(
        self,
        name: str,
        run: InferenceHook,
        *,
        num_workers: int = 1,
        description: str = "",
        modalities: tuple[str, ...] | None = None,
    ) -> LocalToolSpec:
        if not name.strip():
            raise ValueError("Tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        spec = LocalToolSpec(
            name=name,
            run=run,
            num_workers=max(1, num_workers),
            description=description,
            modalities=tuple(modalities or ()),
        )
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> LocalToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_worker_configs(self) -> list[dict[str, Any]]:
        return [
            {
                "model_type": spec.name,
                "num_workers": spec.num_workers,
                "run_inference_hook": spec.run,
            }
            for spec in self._tools.values()
        ]

    def create_hypervisor(self) -> ActorHypervisor:
        worker_configs = self.to_worker_configs()
        if not worker_configs:
            raise ValueError("No local tools registered")
        return ActorHypervisor(worker_configs=worker_configs)
