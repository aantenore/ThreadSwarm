"""Capability-bound semantic compilation and execution.

The language model may propose a DAG, but this module owns the executable
boundary: only explicitly registered, policy-admitted local tools can be bound
to a plan. Catalog and plan digests fence mutations between compilation and
execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .parser import TaskDAG, serialize_capability_catalog

if TYPE_CHECKING:
    from src.engine.orchestrator import DAGExecutionReport

CAPABILITY_CATALOG_SCHEMA_VERSION = 1
BOUND_PLAN_SCHEMA_VERSION = 1
_SAFE_TOOL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SAFE_MODALITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
_SAFE_POLICY_CLASS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


class CapabilityCompiler(Protocol):
    """Compiler port used by the capability-aware runtime."""

    def compile(
        self,
        user_prompt: str,
        *,
        capability_catalog: dict[str, Any] | None = None,
    ) -> TaskDAG: ...


class CapabilityRegistry(Protocol):
    """Minimal registry port required for catalog binding and execution."""

    def contracts(self) -> dict[str, dict[str, Any]]: ...

    def create_hypervisor(self, *, tool_names: list[str] | None = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class CapabilityPolicy:
    """Fail-closed policy controlling which registered tools reach the model."""

    allowed_risk_classes: frozenset[str] = field(
        default_factory=lambda: frozenset({"compute_only", "read_only", "search_only"})
    )
    allowed_side_effect_classes: frozenset[str] = field(default_factory=lambda: frozenset({"none"}))
    max_tools: int = 32
    max_description_chars: int = 512
    max_modalities_per_tool: int = 8
    max_modality_chars: int = 64
    max_output_fields_per_tool: int = 64
    max_schema_field_name_chars: int = 128
    max_catalog_bytes: int = 1_048_576
    max_prompt_catalog_bytes: int = 32_768
    max_compiler_response_chars: int = 131_072
    max_user_prompt_chars: int = 16_384
    max_tasks: int = 32
    max_dependencies_per_task: int = 16
    max_task_id_chars: int = 128
    max_task_description_chars: int = 512
    max_instruction_chars: int = 4_096
    max_total_instruction_chars: int = 32_768
    max_payload_hint_chars: int = 256
    max_retry_count: int = 2
    max_retry_delay_seconds: float = 30.0
    max_task_timeout_seconds: float = 300.0
    default_run_timeout_seconds: float = 300.0
    max_run_timeout_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_risk_classes",
            _normalize_policy_classes(self.allowed_risk_classes, "allowed_risk_classes"),
        )
        object.__setattr__(
            self,
            "allowed_side_effect_classes",
            _normalize_policy_classes(
                self.allowed_side_effect_classes,
                "allowed_side_effect_classes",
            ),
        )
        integer_limits = {
            "max_tools": self.max_tools,
            "max_description_chars": self.max_description_chars,
            "max_tasks": self.max_tasks,
            "max_modalities_per_tool": self.max_modalities_per_tool,
            "max_modality_chars": self.max_modality_chars,
            "max_output_fields_per_tool": self.max_output_fields_per_tool,
            "max_schema_field_name_chars": self.max_schema_field_name_chars,
            "max_catalog_bytes": self.max_catalog_bytes,
            "max_prompt_catalog_bytes": self.max_prompt_catalog_bytes,
            "max_compiler_response_chars": self.max_compiler_response_chars,
            "max_user_prompt_chars": self.max_user_prompt_chars,
            "max_dependencies_per_task": self.max_dependencies_per_task,
            "max_task_id_chars": self.max_task_id_chars,
            "max_task_description_chars": self.max_task_description_chars,
            "max_instruction_chars": self.max_instruction_chars,
            "max_total_instruction_chars": self.max_total_instruction_chars,
            "max_payload_hint_chars": self.max_payload_hint_chars,
        }
        for name, value in integer_limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_retry_count, bool) or not isinstance(self.max_retry_count, int):
            raise ValueError("max_retry_count must be a non-negative integer")
        if self.max_retry_count < 0:
            raise ValueError("max_retry_count must be a non-negative integer")
        positive_time_limits = {
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "max_task_timeout_seconds": self.max_task_timeout_seconds,
            "default_run_timeout_seconds": self.default_run_timeout_seconds,
            "max_run_timeout_seconds": self.max_run_timeout_seconds,
        }
        for name, value in positive_time_limits.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be greater than 0")
        if self.default_run_timeout_seconds > self.max_run_timeout_seconds:
            raise ValueError("default_run_timeout_seconds cannot exceed max_run_timeout_seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_risk_classes": sorted(self.allowed_risk_classes),
            "allowed_side_effect_classes": sorted(self.allowed_side_effect_classes),
            "max_tools": self.max_tools,
            "max_description_chars": self.max_description_chars,
            "max_modalities_per_tool": self.max_modalities_per_tool,
            "max_modality_chars": self.max_modality_chars,
            "max_output_fields_per_tool": self.max_output_fields_per_tool,
            "max_schema_field_name_chars": self.max_schema_field_name_chars,
            "max_catalog_bytes": self.max_catalog_bytes,
            "max_prompt_catalog_bytes": self.max_prompt_catalog_bytes,
            "max_compiler_response_chars": self.max_compiler_response_chars,
            "max_user_prompt_chars": self.max_user_prompt_chars,
            "max_tasks": self.max_tasks,
            "max_dependencies_per_task": self.max_dependencies_per_task,
            "max_task_id_chars": self.max_task_id_chars,
            "max_task_description_chars": self.max_task_description_chars,
            "max_instruction_chars": self.max_instruction_chars,
            "max_total_instruction_chars": self.max_total_instruction_chars,
            "max_payload_hint_chars": self.max_payload_hint_chars,
            "max_retry_count": self.max_retry_count,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "max_task_timeout_seconds": self.max_task_timeout_seconds,
            "default_run_timeout_seconds": self.default_run_timeout_seconds,
            "max_run_timeout_seconds": self.max_run_timeout_seconds,
        }

    def plan_limits(self) -> dict[str, Any]:
        """Return only model-relevant DAG limits for the prompt projection."""
        return {
            "max_tasks": self.max_tasks,
            "max_dependencies_per_task": self.max_dependencies_per_task,
            "max_task_id_chars": self.max_task_id_chars,
            "max_task_description_chars": self.max_task_description_chars,
            "max_instruction_chars": self.max_instruction_chars,
            "max_total_instruction_chars": self.max_total_instruction_chars,
            "max_payload_hint_chars": self.max_payload_hint_chars,
            "max_retry_count": self.max_retry_count,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "max_task_timeout_seconds": self.max_task_timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class ToolCapability:
    """Immutable catalog projection for one policy-admitted local tool."""

    name: str
    description: str
    num_workers: int
    modalities: tuple[str, ...]
    risk_class: str
    side_effect_class: str
    result_size_limit: int | None
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None

    def to_dict(self, *, concise: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "modalities": list(self.modalities),
            "risk_class": self.risk_class,
            "side_effect_class": self.side_effect_class,
            "result_size_limit": self.result_size_limit,
        }
        if concise:
            payload["output_contract"] = _schema_summary(self.output_schema)
        else:
            payload.update(
                {
                    "num_workers": self.num_workers,
                    "input_schema": copy.deepcopy(self.input_schema),
                    "output_schema": copy.deepcopy(self.output_schema),
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class CapabilityCatalog:
    """Deterministic catalog of the exact capabilities admitted for planning."""

    tools: tuple[ToolCapability, ...]
    policy: CapabilityPolicy
    digest: str

    @classmethod
    def from_registry(
        cls,
        registry: CapabilityRegistry,
        policy: CapabilityPolicy | None = None,
    ) -> "CapabilityCatalog":
        resolved_policy = policy or CapabilityPolicy()
        admitted: list[ToolCapability] = []

        for name, contract in sorted(registry.contracts().items()):
            if contract["risk_class"] not in resolved_policy.allowed_risk_classes:
                continue
            if contract["side_effect_class"] not in resolved_policy.allowed_side_effect_classes:
                continue
            description = str(contract["description"]).strip()
            if not _SAFE_TOOL_NAME.fullmatch(name):
                raise CapabilityCatalogError(
                    f"Tool name {name!r} is not catalog-safe; use 1-128 letters, digits, or ._:/- characters"
                )
            if not description:
                raise CapabilityCatalogError(
                    f"Tool {name!r} needs a description before capability-aware compilation"
                )
            if len(description) > resolved_policy.max_description_chars:
                raise CapabilityCatalogError(
                    f"Tool {name!r} description has {len(description)} characters; "
                    f"limit it to {resolved_policy.max_description_chars}"
                )
            modalities = tuple(str(item) for item in contract["modalities"])
            _validate_prompt_contract(name, modalities, contract["output_schema"], resolved_policy)
            admitted.append(
                ToolCapability(
                    name=name,
                    description=description,
                    num_workers=int(contract["num_workers"]),
                    modalities=modalities,
                    risk_class=str(contract["risk_class"]),
                    side_effect_class=str(contract["side_effect_class"]),
                    result_size_limit=contract["result_size_limit"],
                    input_schema=copy.deepcopy(contract["input_schema"]),
                    output_schema=copy.deepcopy(contract["output_schema"]),
                )
            )

        if not admitted:
            raise CapabilityCatalogError("No registered tools are admitted by the capability policy")
        if len(admitted) > resolved_policy.max_tools:
            raise CapabilityCatalogError(
                f"Capability catalog has {len(admitted)} tools; policy allows at most {resolved_policy.max_tools}"
            )

        digest_payload = {
            "schema_version": CAPABILITY_CATALOG_SCHEMA_VERSION,
            "policy": resolved_policy.to_dict(),
            "tools": [tool.to_dict() for tool in admitted],
        }
        catalog_bytes = _canonical_json_bytes(digest_payload)
        if len(catalog_bytes) > resolved_policy.max_catalog_bytes:
            raise CapabilityCatalogError(
                f"Capability catalog has {len(catalog_bytes)} bytes; policy allows at most "
                f"{resolved_policy.max_catalog_bytes}"
            )
        return cls(
            tools=tuple(admitted),
            policy=resolved_policy,
            digest=hashlib.sha256(catalog_bytes).hexdigest(),
        )

    def prompt_payload(self) -> dict[str, Any]:
        """Return the compact, stable projection supplied to the model."""
        payload = {
            "schema_version": CAPABILITY_CATALOG_SCHEMA_VERSION,
            "catalog_digest": self.digest,
            "compiler_limits": {
                "max_response_chars": self.policy.max_compiler_response_chars,
            },
            "plan_limits": self.policy.plan_limits(),
            "tools": [tool.to_dict(concise=True) for tool in self.tools],
        }
        size = len(serialize_capability_catalog(payload).encode("utf-8"))
        if size > self.policy.max_prompt_catalog_bytes:
            raise CapabilityCatalogError(
                f"Capability prompt catalog has {size} bytes; policy allows at most "
                f"{self.policy.max_prompt_catalog_bytes}"
            )
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAPABILITY_CATALOG_SCHEMA_VERSION,
            "catalog_digest": self.digest,
            "policy": self.policy.to_dict(),
            "tools": [tool.to_dict() for tool in self.tools],
        }

    def by_name(self) -> dict[str, ToolCapability]:
        return {tool.name: tool for tool in self.tools}


@dataclass(frozen=True, slots=True)
class CapabilityPlanIssue:
    """Structured, actionable reason why a proposed route was rejected."""

    code: str
    message: str
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "task_id": self.task_id,
        }


class CapabilityCatalogError(ValueError):
    """Raised when a registry cannot be projected into a safe bounded catalog."""


class CapabilityPlanError(ValueError):
    """Raised when a model-proposed plan is not bound to admitted capabilities."""

    def __init__(self, message: str, *, issues: list[CapabilityPlanIssue] | None = None):
        super().__init__(message)
        self.issues = tuple(issues or ())


class CapabilityExecutionError(RuntimeError):
    """Raised when a bound plan fails after execution has started."""

    def __init__(self, message: str, *, result: "CapabilityRunResult"):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class BoundTaskDAG:
    """A DAG bound to one capability catalog and protected from mutation."""

    dag: TaskDAG
    catalog_digest: str
    plan_digest: str
    capability_policy: CapabilityPolicy

    @classmethod
    def bind(cls, dag: TaskDAG, catalog: CapabilityCatalog) -> "BoundTaskDAG":
        bound_dag = dag.model_copy(deep=True)
        return cls(
            dag=bound_dag,
            catalog_digest=catalog.digest,
            plan_digest=_plan_digest(bound_dag, catalog.digest, catalog.policy),
            capability_policy=catalog.policy,
        )

    def verified_dag_snapshot(self) -> TaskDAG:
        """Return the verified execution snapshot, detached from caller mutations."""
        snapshot = self.dag.model_copy(deep=True)
        current = _plan_digest(snapshot, self.catalog_digest, self.capability_policy)
        if current != self.plan_digest:
            raise CapabilityPlanError(
                "Bound DAG changed after compilation; compile a fresh plan before execution",
                issues=[
                    CapabilityPlanIssue(
                        code="plan_digest_mismatch",
                        message="The task DAG no longer matches its bound plan digest",
                    )
                ],
            )
        return snapshot

    def verify_integrity(self) -> None:
        """Reject a caller-owned DAG that no longer matches its bound digest."""
        self.verified_dag_snapshot()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BOUND_PLAN_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "plan_digest": self.plan_digest,
            "capability_policy": self.capability_policy.to_dict(),
            "dag": self.dag.model_dump(mode="json"),
        }


@dataclass(frozen=True, slots=True)
class CapabilityRunResult:
    """Bound plan plus its structured execution report."""

    plan: BoundTaskDAG
    execution: DAGExecutionReport
    run_timeout_seconds: float

    def to_dict(self, *, include_dependency_results: bool = True) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "execution_policy": {"run_timeout_seconds": self.run_timeout_seconds},
            "execution": self.execution.to_dict(
                include_dependency_results=include_dependency_results,
            ),
        }


class CapabilityAwareRuntime:
    """Compile natural-language intent against live capabilities, then execute it."""

    def __init__(
        self,
        compiler: CapabilityCompiler,
        registry: CapabilityRegistry,
        *,
        policy: CapabilityPolicy | None = None,
    ):
        self.compiler = compiler
        self.registry = registry
        self.policy = policy or CapabilityPolicy()

    def compile(self, user_prompt: str) -> BoundTaskDAG:
        if not user_prompt.strip():
            raise CapabilityPlanError(
                "User prompt cannot be empty",
                issues=[
                    CapabilityPlanIssue(
                        code="empty_user_prompt",
                        message="Provide a non-empty intent to compile",
                    )
                ],
            )
        if len(user_prompt) > self.policy.max_user_prompt_chars:
            raise CapabilityPlanError(
                f"User prompt has {len(user_prompt)} characters; policy allows at most "
                f"{self.policy.max_user_prompt_chars}",
                issues=[
                    CapabilityPlanIssue(
                        code="user_prompt_too_large",
                        message=f"Limit the intent to {self.policy.max_user_prompt_chars} characters",
                    )
                ],
            )
        catalog = CapabilityCatalog.from_registry(self.registry, self.policy)
        dag = self.compiler.compile(user_prompt, capability_catalog=catalog.prompt_payload())
        validate_dag_capabilities(dag, catalog)
        return BoundTaskDAG.bind(dag, catalog)

    def execute(
        self,
        plan: BoundTaskDAG,
        *,
        context: Any = None,
        fail_fast: bool = True,
        timeout: float | None = None,
    ) -> DAGExecutionReport:
        from src.engine.orchestrator import DAGOrchestrator

        execution_dag = plan.verified_dag_snapshot()
        current_catalog = CapabilityCatalog.from_registry(self.registry, self.policy)
        if current_catalog.digest != plan.catalog_digest:
            raise CapabilityPlanError(
                "Registered capabilities changed after compilation; compile a fresh plan before execution",
                issues=[
                    CapabilityPlanIssue(
                        code="catalog_digest_mismatch",
                        message="The live capability catalog no longer matches the plan",
                    )
                ],
            )
        validate_dag_capabilities(execution_dag, current_catalog)
        resolved_timeout = self._resolve_run_timeout(timeout)
        selected_tool_names = sorted(
            {task.tool_name for task in execution_dag.tasks if task.tool_name is not None}
        )
        return DAGOrchestrator(
            self.registry.create_hypervisor(tool_names=selected_tool_names)
        ).run(
            execution_dag,
            context=context,
            fail_fast=fail_fast,
            timeout=resolved_timeout,
        )

    def compile_and_run(
        self,
        user_prompt: str,
        *,
        context: Any = None,
        fail_fast: bool = True,
        timeout: float | None = None,
    ) -> CapabilityRunResult:
        from src.engine.orchestrator import DAGExecutionError

        plan = self.compile(user_prompt)
        resolved_timeout = self._resolve_run_timeout(timeout)
        try:
            execution = self.execute(
                plan,
                context=context,
                fail_fast=fail_fast,
                timeout=resolved_timeout,
            )
        except DAGExecutionError as error:
            if error.report is None:
                raise
            result = CapabilityRunResult(
                plan=plan,
                execution=error.report,
                run_timeout_seconds=resolved_timeout,
            )
            raise CapabilityExecutionError(str(error), result=result) from error
        return CapabilityRunResult(
            plan=plan,
            execution=execution,
            run_timeout_seconds=resolved_timeout,
        )

    def _resolve_run_timeout(self, timeout: float | None) -> float:
        resolved = self.policy.default_run_timeout_seconds if timeout is None else timeout
        if (
            isinstance(resolved, bool)
            or not isinstance(resolved, (int, float))
            or not math.isfinite(resolved)
            or resolved <= 0
        ):
            raise CapabilityPlanError(
                "Run timeout must be a positive number",
                issues=[
                    CapabilityPlanIssue(
                        code="invalid_run_timeout",
                        message="Choose a positive run timeout in seconds",
                    )
                ],
            )
        if resolved > self.policy.max_run_timeout_seconds:
            raise CapabilityPlanError(
                f"Run timeout {resolved:g}s exceeds the policy maximum "
                f"of {self.policy.max_run_timeout_seconds:g}s",
                issues=[
                    CapabilityPlanIssue(
                        code="run_timeout_exceeded",
                        message=(
                            f"Choose at most {self.policy.max_run_timeout_seconds:g} seconds"
                        ),
                    )
                ],
            )
        return float(resolved)


def validate_dag_capabilities(dag: TaskDAG, catalog: CapabilityCatalog) -> None:
    """Reject unbounded, missing, invented, ambiguous, or incompatible routes."""
    issues: list[CapabilityPlanIssue] = []
    policy = catalog.policy
    if not dag.tasks:
        raise CapabilityPlanError(
            "Capability-bound plan rejected: the compiler produced no executable tasks",
            issues=[
                CapabilityPlanIssue(
                    code="empty_dag",
                    message="The compiler must produce at least one executable task",
                )
            ],
        )
    if len(dag.tasks) > policy.max_tasks:
        raise CapabilityPlanError(
            f"Capability-bound plan rejected: plan has {len(dag.tasks)} tasks; "
            f"policy allows at most {policy.max_tasks}",
            issues=[
                CapabilityPlanIssue(
                    code="task_count_exceeded",
                    message=f"Reduce the plan to at most {policy.max_tasks} tasks",
                )
            ],
        )
    total_instruction_chars = sum(len(task.instruction) for task in dag.tasks)
    if total_instruction_chars > policy.max_total_instruction_chars:
        issues.append(
            CapabilityPlanIssue(
                code="total_instruction_size_exceeded",
                message=(
                    f"Plan has {total_instruction_chars} instruction characters; policy allows at most "
                    f"{policy.max_total_instruction_chars}"
                ),
            )
        )

    capabilities = catalog.by_name()
    structurally_bounded = True
    for task in dag.tasks:
        task_label = _bounded_label(task.id, policy.max_task_id_chars)
        if len(task.id) > policy.max_task_id_chars or not _SAFE_TASK_ID.fullmatch(task.id):
            structurally_bounded = False
            issues.append(
                CapabilityPlanIssue(
                    code="invalid_task_id",
                    task_id=task_label,
                    message=(
                        f"Use a 1-{policy.max_task_id_chars} character task ID containing only "
                        "letters, digits, or ._:-"
                    ),
                )
            )
        if len(task.description) > policy.max_task_description_chars:
            issues.append(
                CapabilityPlanIssue(
                    code="task_description_too_large",
                    task_id=task_label,
                    message=f"Limit the task description to {policy.max_task_description_chars} characters",
                )
            )
        elif not task.description.strip():
            issues.append(
                CapabilityPlanIssue(
                    code="empty_task_description",
                    task_id=task_label,
                    message="Provide a non-empty task description",
                )
            )
        if len(task.instruction) > policy.max_instruction_chars:
            issues.append(
                CapabilityPlanIssue(
                    code="task_instruction_too_large",
                    task_id=task_label,
                    message=f"Limit the task instruction to {policy.max_instruction_chars} characters",
                )
            )
        elif not task.instruction.strip():
            issues.append(
                CapabilityPlanIssue(
                    code="empty_task_instruction",
                    task_id=task_label,
                    message="Provide a non-empty task instruction",
                )
            )
        if len(task.dependencies) > policy.max_dependencies_per_task:
            structurally_bounded = False
            issues.append(
                CapabilityPlanIssue(
                    code="dependency_count_exceeded",
                    task_id=task_label,
                    message=(
                        f"Task has {len(task.dependencies)} dependencies; policy allows at most "
                        f"{policy.max_dependencies_per_task}"
                    ),
                )
            )
        else:
            for dependency_id in task.dependencies:
                if (
                    len(dependency_id) > policy.max_task_id_chars
                    or not _SAFE_TASK_ID.fullmatch(dependency_id)
                ):
                    structurally_bounded = False
                    issues.append(
                        CapabilityPlanIssue(
                            code="invalid_dependency_id",
                            task_id=task_label,
                            message=(
                                "Every dependency ID must follow the same bounded syntax as a task ID"
                            ),
                        )
                    )
                    break
        if task.payload_hint is not None and len(task.payload_hint) > policy.max_payload_hint_chars:
            issues.append(
                CapabilityPlanIssue(
                    code="payload_hint_too_large",
                    task_id=task_label,
                    message=f"Limit payload_hint to {policy.max_payload_hint_chars} characters",
                )
            )
        if task.retry_count > policy.max_retry_count:
            issues.append(
                CapabilityPlanIssue(
                    code="retry_count_exceeded",
                    task_id=task_label,
                    message=f"Use at most {policy.max_retry_count} retries",
                )
            )
        if not math.isfinite(task.retry_delay_seconds) or task.retry_delay_seconds > policy.max_retry_delay_seconds:
            issues.append(
                CapabilityPlanIssue(
                    code="retry_delay_exceeded",
                    task_id=task_label,
                    message=f"Use a retry delay no greater than {policy.max_retry_delay_seconds:g} seconds",
                )
            )
        if task.timeout_seconds is not None and (
            not math.isfinite(task.timeout_seconds)
            or task.timeout_seconds > policy.max_task_timeout_seconds
        ):
            issues.append(
                CapabilityPlanIssue(
                    code="task_timeout_exceeded",
                    task_id=task_label,
                    message=f"Use a task timeout no greater than {policy.max_task_timeout_seconds:g} seconds",
                )
            )
        if (
            not task.modality
            or len(task.modality) > policy.max_modality_chars
            or not _SAFE_MODALITY.fullmatch(task.modality)
        ):
            issues.append(
                CapabilityPlanIssue(
                    code="invalid_modality",
                    task_id=task_label,
                    message=(
                        f"Use a 1-{policy.max_modality_chars} character modality containing only "
                        "letters, digits, or ._:/-"
                    ),
                )
            )
        if task.tool_name is None:
            issues.append(
                CapabilityPlanIssue(
                    code="missing_tool_route",
                    task_id=task_label,
                    message="Set tool_name to one exact name from the supplied catalog",
                )
            )
            continue
        if task.model_type is not None:
            issues.append(
                CapabilityPlanIssue(
                    code="ambiguous_route",
                    task_id=task_label,
                    message="Capability-bound local tasks must set tool_name only; remove model_type",
                )
            )
        capability = capabilities.get(task.tool_name)
        if capability is None:
            issues.append(
                CapabilityPlanIssue(
                    code="unknown_or_disallowed_tool",
                    task_id=task_label,
                    message=(
                        f"Tool {_bounded_label(task.tool_name, 128)!r} is not admitted; choose one of: "
                        f"{', '.join(sorted(capabilities))}"
                    ),
                )
            )
            continue
        if capability.side_effect_class != "none" and task.retry_count > 0:
            issues.append(
                CapabilityPlanIssue(
                    code="side_effect_retry_forbidden",
                    task_id=task_label,
                    message=(
                        f"Tool {task.tool_name!r} has side effects; set retry_count to 0 unless "
                        "the host adds a separate idempotency contract"
                    ),
                )
            )
        if capability.modalities and task.modality not in capability.modalities:
            issues.append(
                CapabilityPlanIssue(
                    code="unsupported_modality",
                    task_id=task_label,
                    message=(
                        f"Tool {task.tool_name!r} does not support modality "
                        f"{_bounded_label(task.modality, 64)!r}; "
                        f"choose one of: {', '.join(capability.modalities)}"
                    ),
                )
            )

    if structurally_bounded:
        validation_error = dag.validation_error()
        if validation_error:
            issues.append(CapabilityPlanIssue(code="invalid_dag", message=validation_error))

    if issues:
        details = "; ".join(
            f"{issue.task_id + ': ' if issue.task_id else ''}{issue.message}" for issue in issues
        )
        raise CapabilityPlanError(f"Capability-bound plan rejected: {details}", issues=issues)


def _schema_summary(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    properties = schema.get("properties")
    required = set(schema.get("required", []))
    summary: dict[str, Any] = {"type": _schema_type(schema)}
    if isinstance(properties, dict):
        summary["fields"] = [
            {
                "name": name,
                "type": _schema_type(definition),
                "required": name in required,
            }
            for name, definition in sorted(properties.items())
        ]
    return summary


def _bounded_label(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _normalize_policy_classes(value: Any, name: str) -> frozenset[str]:
    if isinstance(value, str) or not isinstance(value, (frozenset, set, tuple, list)):
        raise ValueError(f"{name} must be a collection of class names")
    normalized: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} must contain only strings")
        candidate = item.strip()
        if not _SAFE_POLICY_CLASS.fullmatch(candidate):
            raise ValueError(f"{name} contains an invalid class name")
        normalized.add(candidate)
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return frozenset(normalized)


def _validate_prompt_contract(
    tool_name: str,
    modalities: tuple[str, ...],
    output_schema: Any,
    policy: CapabilityPolicy,
) -> None:
    if len(modalities) > policy.max_modalities_per_tool:
        raise CapabilityCatalogError(
            f"Tool {tool_name!r} declares {len(modalities)} modalities; policy allows at most "
            f"{policy.max_modalities_per_tool}"
        )
    for modality in modalities:
        if (
            not modality
            or len(modality) > policy.max_modality_chars
            or not _SAFE_MODALITY.fullmatch(modality)
        ):
            raise CapabilityCatalogError(
                f"Tool {tool_name!r} has an invalid modality; use 1-{policy.max_modality_chars} "
                "letters, digits, or ._:/- characters"
            )
    if not isinstance(output_schema, dict):
        return
    properties = output_schema.get("properties")
    if not isinstance(properties, dict):
        return
    if len(properties) > policy.max_output_fields_per_tool:
        raise CapabilityCatalogError(
            f"Tool {tool_name!r} output schema has {len(properties)} fields; policy allows at most "
            f"{policy.max_output_fields_per_tool}"
        )
    for field_name in properties:
        if not isinstance(field_name, str) or not field_name or len(field_name) > policy.max_schema_field_name_chars:
            raise CapabilityCatalogError(
                f"Tool {tool_name!r} output schema has an invalid field name; use 1-"
                f"{policy.max_schema_field_name_chars} characters"
            )


def _schema_type(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "unknown"
    value = schema.get("type")
    if value in {"array", "boolean", "integer", "null", "number", "object", "string"}:
        return value
    return "unknown"


def _plan_digest(
    dag: TaskDAG,
    catalog_digest: str,
    capability_policy: CapabilityPolicy,
) -> str:
    return _sha256_json(
        {
            "schema_version": BOUND_PLAN_SCHEMA_VERSION,
            "catalog_digest": catalog_digest,
            "capability_policy": capability_policy.to_dict(),
            "dag": dag.model_dump(mode="json"),
        }
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
