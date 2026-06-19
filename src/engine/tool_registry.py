"""
Registry for local CPU-friendly tools that can execute DAG tasks.

This keeps the execution model generic: a task can be routed to a plain Python
tool that runs on a normal PC, or to a model-backed worker when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .actor_pool import ActorHypervisor, InferenceHook

SchemaModel = type[BaseModel]


@dataclass(frozen=True, slots=True)
class ToolContract:
    """Metadata and optional schemas for one executable local tool."""

    input_schema: SchemaModel | None = None
    output_schema: SchemaModel | None = None
    risk_class: str = "compute_only"
    side_effect_class: str = "none"
    result_size_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_class": self.risk_class,
            "side_effect_class": self.side_effect_class,
            "result_size_limit": self.result_size_limit,
            "input_schema": self.input_schema.model_json_schema() if self.input_schema else None,
            "output_schema": self.output_schema.model_json_schema() if self.output_schema else None,
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionInput:
    """Structured input envelope used for optional tool input validation."""

    payload: Any
    dependency_results: dict[str, Any]
    task_id: str
    instruction: str
    modality: str
    tool_name: str | None
    model_type: str | None
    route_key: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "dependency_results": self.dependency_results,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "modality": self.modality,
            "tool_name": self.tool_name,
            "model_type": self.model_type,
            "route_key": self.route_key,
        }


@dataclass(frozen=True, slots=True)
class ValidatedToolRunner:
    """Picklable wrapper that validates tool input and output when schemas exist."""

    tool_name: str
    run: InferenceHook
    contract: ToolContract

    def __call__(
        self,
        context: Any,
        instruction: str,
        task_id: str,
        modality: str,
        route_key: str | None,
    ) -> dict[str, Any]:
        execution_input = _build_execution_input(context, instruction, task_id, modality, route_key)
        _validate_model(self.contract.input_schema, execution_input.to_dict(), f"Tool {self.tool_name} input")
        result = self.run(context, instruction, task_id, modality, route_key)
        _validate_result_size(self.tool_name, result, self.contract.result_size_limit)
        validated = _validate_model(self.contract.output_schema, result, f"Tool {self.tool_name} output")
        if validated is None:
            return result
        return validated.model_dump()


@dataclass(slots=True)
class LocalToolSpec:
    """Definition of one local executable tool."""

    name: str
    run: InferenceHook
    num_workers: int = 1
    description: str = ""
    modalities: tuple[str, ...] = field(default_factory=tuple)
    contract: ToolContract = field(default_factory=ToolContract)

    def to_contract_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "num_workers": self.num_workers,
            "modalities": list(self.modalities),
            **self.contract.to_dict(),
        }


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
        input_schema: SchemaModel | None = None,
        output_schema: SchemaModel | None = None,
        risk_class: str = "compute_only",
        side_effect_class: str = "none",
        result_size_limit: int | None = None,
    ) -> LocalToolSpec:
        if not name.strip():
            raise ValueError("Tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if not risk_class.strip():
            raise ValueError("Tool risk_class cannot be empty")
        if not side_effect_class.strip():
            raise ValueError("Tool side_effect_class cannot be empty")
        if result_size_limit is not None and result_size_limit <= 0:
            raise ValueError("Tool result_size_limit must be greater than 0 when set")
        spec = LocalToolSpec(
            name=name,
            run=run,
            num_workers=max(1, num_workers),
            description=description,
            modalities=tuple(modalities or ()),
            contract=ToolContract(
                input_schema=input_schema,
                output_schema=output_schema,
                risk_class=risk_class,
                side_effect_class=side_effect_class,
                result_size_limit=result_size_limit,
            ),
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

    def contracts(self) -> dict[str, dict[str, Any]]:
        return {name: spec.to_contract_dict() for name, spec in self._tools.items()}

    def to_worker_configs(self) -> list[dict[str, Any]]:
        return [
            {
                "model_type": spec.name,
                "num_workers": spec.num_workers,
                "run_inference_hook": ValidatedToolRunner(spec.name, spec.run, spec.contract),
            }
            for spec in self._tools.values()
        ]

    def create_hypervisor(self) -> ActorHypervisor:
        worker_configs = self.to_worker_configs()
        if not worker_configs:
            raise ValueError("No local tools registered")
        return ActorHypervisor(worker_configs=worker_configs)


def _build_execution_input(
    context: Any,
    instruction: str,
    task_id: str,
    modality: str,
    route_key: str | None,
) -> ToolExecutionInput:
    if isinstance(context, dict):
        return ToolExecutionInput(
            payload=context.get("payload"),
            dependency_results=dict(context.get("dependency_results", {})),
            task_id=task_id,
            instruction=instruction,
            modality=modality,
            tool_name=context.get("tool_name"),
            model_type=context.get("model_type"),
            route_key=route_key,
        )
    return ToolExecutionInput(
        payload=context,
        dependency_results={},
        task_id=task_id,
        instruction=instruction,
        modality=modality,
        tool_name=None,
        model_type=None,
        route_key=route_key,
    )


def _validate_model(schema: SchemaModel | None, payload: Any, label: str) -> BaseModel | None:
    if schema is None:
        return None
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{label} schema validation failed: {exc}") from exc


def _validate_result_size(tool_name: str, result: Any, limit: int | None) -> None:
    if limit is None:
        return
    size = len(repr(result))
    if size > limit:
        raise ValueError(f"Tool {tool_name} result exceeded size limit: {size} > {limit}")
