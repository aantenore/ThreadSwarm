"""
Minimal DAG orchestrator for executing TaskDAGs on local tools or workers.

The orchestrator is intentionally lightweight:
- validates the DAG before execution,
- submits ready tasks to the execution pool,
- waits for results and unlocks dependent tasks,
- blocks downstream work when an upstream task fails,
- returns a structured execution report with a default leaf-node reducer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.compiler.parser import TaskDAG

from .actor_pool import ActorHypervisor
from .shared_memory import ContextMemoryManager

ContextPayload = Any
ReducerHook = Callable[[TaskDAG, dict[str, "TaskExecutionRecord"]], Any]


@dataclass(slots=True)
class TaskExecutionRecord:
    """Execution state and output for one task in the DAG."""

    task_id: str
    status: str = "pending"
    result: Any = None
    error: str | None = None
    submitted_at: float | None = None
    completed_at: float | None = None
    submitted_instruction: str | None = None
    dependency_results: dict[str, Any] = field(default_factory=dict)
    modality: str = "text"
    tool_name: str | None = None
    model_type: str | None = None
    dependencies: list[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 1
    retry_delay_seconds: float = 0.0
    timeout_seconds: float | None = None
    timed_out: bool = False
    attempt_errors: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.error is None

    @property
    def duration_seconds(self) -> float | None:
        if self.submitted_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.submitted_at

    def to_dict(
        self,
        *,
        include_results: bool = True,
        include_dependency_results: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-friendly execution record for tracing and debugging."""
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "error": self.error,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "submitted_instruction": self.submitted_instruction,
            "modality": self.modality,
            "tool_name": self.tool_name,
            "model_type": self.model_type,
            "dependencies": list(self.dependencies),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "timed_out": self.timed_out,
            "attempt_errors": list(self.attempt_errors),
        }
        if include_results:
            payload["result"] = _json_safe(self.result)
        if include_dependency_results:
            payload["dependency_results"] = _json_safe(self.dependency_results)
        return payload


@dataclass(slots=True)
class DAGExecutionReport:
    """Structured report for one DAG execution."""

    task_results: dict[str, TaskExecutionRecord]
    execution_order: list[str]
    final_result: Any
    leaf_task_ids: list[str]
    started_at: float
    completed_at: float
    context_metadata: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return all(record.status == "completed" for record in self.task_results.values())

    @property
    def failed_task_ids(self) -> list[str]:
        return [task_id for task_id, record in self.task_results.items() if record.status == "failed"]

    @property
    def blocked_task_ids(self) -> list[str]:
        return [task_id for task_id, record in self.task_results.items() if record.status == "blocked"]

    @property
    def duration_seconds(self) -> float:
        return self.completed_at - self.started_at

    def summary(self) -> dict[str, Any]:
        """Return compact run metadata suitable for logs, dashboards, and evals."""
        total_tasks = len(self.task_results)
        completed = sum(1 for record in self.task_results.values() if record.status == "completed")
        failed = len(self.failed_task_ids)
        blocked = len(self.blocked_task_ids)
        return {
            "succeeded": self.succeeded,
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "blocked_tasks": blocked,
            "execution_order": list(self.execution_order),
            "leaf_task_ids": list(self.leaf_task_ids),
            "duration_seconds": self.duration_seconds,
        }

    def to_dict(
        self,
        *,
        include_results: bool = True,
        include_dependency_results: bool = False,
        include_context_metadata: bool = False,
    ) -> dict[str, Any]:
        """Return a JSON-friendly report for trace export and post-run analysis."""
        payload: dict[str, Any] = {
            "summary": self.summary(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "final_result": _json_safe(self.final_result) if include_results else None,
            "task_results": {
                task_id: record.to_dict(
                    include_results=include_results,
                    include_dependency_results=include_dependency_results,
                )
                for task_id, record in self.task_results.items()
            },
        }
        if include_context_metadata:
            payload["context_metadata"] = self.context_metadata
        return payload


def _json_safe(value: Any) -> Any:
    """Convert common Python/scientific objects into JSON-safe structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {
            "type": "bytes",
            "size": len(data),
            "preview": data[:120].decode("utf-8", errors="replace"),
        }
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return repr(value)


class DAGExecutionError(RuntimeError):
    """Raised when DAG execution cannot complete successfully."""

    def __init__(self, message: str, report: DAGExecutionReport | None = None):
        super().__init__(message)
        self.report = report


def default_result_reducer(dag: TaskDAG, task_results: dict[str, TaskExecutionRecord]) -> Any:
    """
    Reduce the execution result to the leaf task outputs.

    When the DAG has a single leaf node, return that task's result directly.
    For multiple leaves, return a mapping from leaf task ID to task result.
    """
    depended_on = {dep for task in dag.tasks for dep in task.dependencies}
    leaf_ids = [task.id for task in dag.tasks if task.id not in depended_on]
    leaf_results = {task_id: task_results[task_id].result for task_id in leaf_ids}
    if not leaf_ids:
        return None
    if len(leaf_ids) == 1:
        return leaf_results[leaf_ids[0]]
    return leaf_results


class DAGOrchestrator:
    """Execute a validated TaskDAG using an ActorHypervisor."""

    def __init__(
        self,
        actor_hypervisor: ActorHypervisor,
        reducer: ReducerHook | None = None,
    ):
        self.actor_hypervisor = actor_hypervisor
        self.reducer = reducer or default_result_reducer

    def run(
        self,
        dag: TaskDAG,
        *,
        context: ContextPayload | None = None,
        context_metadata: dict[str, Any] | None = None,
        fail_fast: bool = True,
        timeout: float | None = None,
    ) -> DAGExecutionReport:
        """
        Execute a TaskDAG and return a structured report.

        :param dag: Valid TaskDAG to execute.
        :param context: Optional shared context payload (ndarray, str, bytes) to place in shared memory.
        :param context_metadata: Precomputed shared-memory metadata. Mutually exclusive with context.
        :param fail_fast: Raise DAGExecutionError when any task fails.
        :param timeout: Maximum execution time in seconds for the full DAG.
        """
        validation_error = dag.validation_error()
        if validation_error:
            raise DAGExecutionError(f"Invalid DAG: {validation_error}")
        if context is not None and context_metadata is not None:
            raise ValueError("Provide either context or context_metadata, not both")

        owned_context_manager: ContextMemoryManager | None = None
        if context is not None:
            owned_context_manager = ContextMemoryManager()
            context_metadata = owned_context_manager.load_and_share(context)

        started_at = time.monotonic()
        started_here = False
        if not self.actor_hypervisor.started:
            self.actor_hypervisor.start()
            started_here = True

        task_lookup = {task.id: task for task in dag.tasks}
        dependency_counts = {task.id: len(task.dependencies) for task in dag.tasks}
        dependents: dict[str, list[str]] = {task.id: [] for task in dag.tasks}
        task_results = {
            task.id: TaskExecutionRecord(
                task_id=task.id,
                modality=task.modality,
                tool_name=task.tool_name,
                model_type=task.model_type,
                dependencies=list(task.dependencies),
                max_attempts=task.retry_count + 1,
                retry_delay_seconds=task.retry_delay_seconds,
                timeout_seconds=task.timeout_seconds,
            )
            for task in dag.tasks
        }
        execution_order: list[str] = []

        for task in dag.tasks:
            for dep in task.dependencies:
                dependents[dep].append(task.id)

        failures_seen = False
        retry_ready_at: dict[str, float] = {}
        active_attempts: set[tuple[str, int]] = set()
        attempt_deadlines: dict[tuple[str, int], float] = {}

        def submit_task(task_id: str) -> None:
            task = task_lookup[task_id]
            record = task_results[task_id]
            if record.status != "pending":
                return

            dependency_results = {dependency_id: task_results[dependency_id].result for dependency_id in task.dependencies}
            instruction = task.instruction
            record.attempts += 1
            attempt = record.attempts
            self.actor_hypervisor.submit(
                {
                    "task_id": task.id,
                    "instruction": instruction,
                    "context_metadata": context_metadata,
                    "context_envelope": True,
                    "dependency_results": dependency_results,
                    "modality": task.modality,
                    "tool_name": task.tool_name,
                    "model_type": task.model_type,
                    "payload_hint": task.payload_hint,
                    "attempt": attempt,
                }
            )
            submitted_at = time.monotonic()
            record.status = "running"
            record.error = None
            record.timed_out = False
            record.submitted_at = submitted_at
            record.completed_at = None
            record.submitted_instruction = instruction
            record.dependency_results = dependency_results
            attempt_key = (task_id, attempt)
            active_attempts.add(attempt_key)
            if task.timeout_seconds is not None:
                attempt_deadlines[attempt_key] = submitted_at + task.timeout_seconds

        def fail_attempt(task_id: str, reason: str, *, timed_out: bool = False) -> None:
            nonlocal failures_seen
            record = task_results[task_id]
            record.status = "failed"
            record.error = reason
            record.timed_out = timed_out
            record.attempt_errors.append(reason)
            if record.attempts <= task_lookup[task_id].retry_count:
                schedule_retry(task_id, reason)
                return
            failures_seen = True
            block_dependents(task_id, reason)
            if fail_fast:
                block_pending_tasks(f"Execution stopped after task {task_id} failed: {reason}")

        def schedule_retry(task_id: str, reason: str) -> None:
            task = task_lookup[task_id]
            record = task_results[task_id]
            record.status = "pending"
            record.error = reason
            retry_ready_at[task_id] = time.monotonic() + task.retry_delay_seconds

        def expire_timed_out_attempts() -> None:
            now = time.monotonic()
            expired_attempts = [
                attempt_key
                for attempt_key, deadline in attempt_deadlines.items()
                if deadline <= now and attempt_key in active_attempts
            ]
            for task_id, attempt in expired_attempts:
                attempt_key = (task_id, attempt)
                active_attempts.discard(attempt_key)
                attempt_deadlines.pop(attempt_key, None)
                record = task_results[task_id]
                if record.status != "running" or record.attempts != attempt:
                    continue
                record.completed_at = now
                fail_attempt(
                    task_id,
                    _format_timeout_error(task_id, attempt, task_lookup[task_id].timeout_seconds),
                    timed_out=True,
                )

        def submit_due_retries() -> None:
            now = time.monotonic()
            due_task_ids = [
                task_id
                for task_id, ready_at in retry_ready_at.items()
                if ready_at <= now and task_results[task_id].status == "pending"
            ]
            for task_id in due_task_ids:
                retry_ready_at.pop(task_id, None)
                submit_task(task_id)

        def block_dependents(task_id: str, reason: str) -> None:
            timestamp = time.monotonic()
            for dependent_id in dependents[task_id]:
                dependent_record = task_results[dependent_id]
                if dependent_record.status != "pending":
                    continue
                dependent_record.status = "blocked"
                dependent_record.error = f"Blocked by failed dependency {task_id}: {reason}"
                dependent_record.completed_at = timestamp
                block_dependents(dependent_id, dependent_record.error)

        def block_pending_tasks(reason: str) -> None:
            timestamp = time.monotonic()
            for record in task_results.values():
                if record.status == "pending":
                    record.status = "blocked"
                    record.error = reason
                    record.completed_at = timestamp

        try:
            for task in dag.tasks:
                if dependency_counts[task.id] == 0:
                    submit_task(task.id)

            if not dag.tasks:
                completed_at = time.monotonic()
                return DAGExecutionReport(
                    task_results=task_results,
                    execution_order=[],
                    final_result=None,
                    leaf_task_ids=[],
                    started_at=started_at,
                    completed_at=completed_at,
                    context_metadata=context_metadata,
                )

            while True:
                submit_due_retries()
                expire_timed_out_attempts()
                finished = sum(
                    1 for record in task_results.values() if record.status in {"completed", "failed", "blocked"}
                )
                if finished == len(dag.tasks):
                    break

                remaining_timeout = self._remaining_timeout(started_at, timeout)
                if remaining_timeout is not None and remaining_timeout <= 0:
                    raise DAGExecutionError("DAG execution timed out")

                if not active_attempts:
                    if retry_ready_at:
                        wait_for_retry = max(0.0, min(retry_ready_at.values()) - time.monotonic())
                        if remaining_timeout is not None:
                            wait_for_retry = min(wait_for_retry, remaining_timeout)
                        time.sleep(min(0.05, wait_for_retry))
                        continue
                    pending = [task_id for task_id, record in task_results.items() if record.status == "pending"]
                    raise DAGExecutionError(
                        f"DAG execution stalled with pending tasks: {', '.join(pending)}",
                    )

                result = self.actor_hypervisor.get_result(
                    timeout=self._next_wait_timeout(remaining_timeout, retry_ready_at, attempt_deadlines)
                )
                if result is None:
                    continue

                task_id = result["task_id"]
                record = task_results[task_id]
                attempt = int(result.get("attempt") or record.attempts)
                attempt_key = (task_id, attempt)
                if attempt_key not in active_attempts:
                    continue
                active_attempts.remove(attempt_key)
                attempt_deadlines.pop(attempt_key, None)
                record.completed_at = time.monotonic()
                if record.status != "running" or record.attempts != attempt:
                    continue

                if result.get("error"):
                    fail_attempt(task_id, result["error"])
                    continue

                record.status = "completed"
                record.error = None
                record.timed_out = False
                record.result = result.get("result")
                execution_order.append(task_id)

                if failures_seen and fail_fast:
                    continue

                for dependent_id in dependents[task_id]:
                    dependent_record = task_results[dependent_id]
                    if dependent_record.status != "pending":
                        continue
                    dependency_counts[dependent_id] -= 1
                    if dependency_counts[dependent_id] == 0:
                        submit_task(dependent_id)

            completed_at = time.monotonic()
            leaf_task_ids = [task.id for task in dag.tasks if not dependents[task.id]]
            report = DAGExecutionReport(
                task_results=task_results,
                execution_order=execution_order,
                final_result=self.reducer(dag, task_results),
                leaf_task_ids=leaf_task_ids,
                started_at=started_at,
                completed_at=completed_at,
                context_metadata=context_metadata,
            )

            if report.failed_task_ids:
                message = f"DAG execution failed for tasks: {', '.join(report.failed_task_ids)}"
                raise DAGExecutionError(message, report=report)
            if fail_fast and report.blocked_task_ids:
                message = f"DAG execution blocked downstream tasks: {', '.join(report.blocked_task_ids)}"
                raise DAGExecutionError(message, report=report)
            return report
        except DAGExecutionError as exc:
            if exc.report is None:
                completed_at = time.monotonic()
                leaf_task_ids = [task.id for task in dag.tasks if not dependents[task.id]]
                exc.report = DAGExecutionReport(
                    task_results=task_results,
                    execution_order=execution_order,
                    final_result=self.reducer(dag, task_results),
                    leaf_task_ids=leaf_task_ids,
                    started_at=started_at,
                    completed_at=completed_at,
                    context_metadata=context_metadata,
                )
            if fail_fast:
                raise
            return exc.report
        finally:
            if started_here:
                self.actor_hypervisor.shutdown()
            if owned_context_manager is not None:
                owned_context_manager.close()

    @staticmethod
    def _remaining_timeout(started_at: float, timeout: float | None) -> float | None:
        if timeout is None:
            return None
        return timeout - (time.monotonic() - started_at)

    @staticmethod
    def _next_wait_timeout(
        remaining_timeout: float | None,
        retry_ready_at: dict[str, float],
        attempt_deadlines: dict[tuple[str, int], float],
    ) -> float:
        wait_timeout = 0.5
        if remaining_timeout is not None:
            wait_timeout = min(wait_timeout, max(0.0, remaining_timeout))
        now = time.monotonic()
        if retry_ready_at:
            wait_timeout = min(wait_timeout, max(0.0, min(retry_ready_at.values()) - now))
        if attempt_deadlines:
            wait_timeout = min(wait_timeout, max(0.0, min(attempt_deadlines.values()) - now))
        return wait_timeout


def _format_timeout_error(task_id: str, attempt: int, timeout_seconds: float | None) -> str:
    timeout_label = "unknown" if timeout_seconds is None else f"{timeout_seconds:g}"
    return f"Task {task_id} attempt {attempt} timed out after {timeout_label} seconds"
