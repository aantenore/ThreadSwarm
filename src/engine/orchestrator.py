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
import uuid
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
    attempt_ids: list[str] = field(default_factory=list)
    current_attempt_id: str | None = None
    run_id: str | None = None

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
            "run_id": self.run_id,
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
            "attempt_ids": list(self.attempt_ids),
            "current_attempt_id": self.current_attempt_id,
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
    run_id: str | None = None
    stop_reason: str | None = None

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
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
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
            "run_id": self.run_id,
            "stop_reason": self.stop_reason,
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
        run_id = uuid.uuid4().hex
        started_at = time.monotonic()
        self.actor_hypervisor.acquire_run(run_id)

        owned_context_manager: ContextMemoryManager | None = None
        started_here = False
        pool_requires_reset = False
        run_generation: int | None = None
        stop_reason: str | None = None
        task_lookup = {task.id: task for task in dag.tasks}
        dependency_counts = {task.id: len(task.dependencies) for task in dag.tasks}
        dependents: dict[str, list[str]] = {task.id: [] for task in dag.tasks}
        task_results: dict[str, TaskExecutionRecord] = {}
        execution_order: list[str] = []
        retry_ready_at: dict[str, float] = {}
        active_attempts: dict[str, tuple[str, int]] = {}
        attempt_deadlines: dict[str, float] = {}
        fail_fast_reason: str | None = None

        def build_report() -> DAGExecutionReport:
            completed_at = time.monotonic()
            leaf_task_ids = [task.id for task in dag.tasks if not dependents[task.id]]
            return DAGExecutionReport(
                task_results=task_results,
                execution_order=execution_order,
                final_result=self.reducer(dag, task_results),
                leaf_task_ids=leaf_task_ids,
                started_at=started_at,
                completed_at=completed_at,
                context_metadata=context_metadata,
                run_id=run_id,
                stop_reason=stop_reason,
            )

        def submit_task(task_id: str) -> None:
            task = task_lookup[task_id]
            record = task_results[task_id]
            if record.status != "pending":
                return

            dependency_results = {
                dependency_id: task_results[dependency_id].result for dependency_id in task.dependencies
            }
            instruction = task.instruction
            record.attempts += 1
            attempt = record.attempts
            attempt_id = uuid.uuid4().hex
            self.actor_hypervisor.submit(
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
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
            record.current_attempt_id = attempt_id
            record.attempt_ids.append(attempt_id)
            active_attempts[attempt_id] = (task_id, attempt)

        def record_started_attempts() -> None:
            while True:
                event = self.actor_hypervisor.get_lifecycle_event()
                if event is None:
                    return
                if event.get("event") != "started" or event.get("run_id") != run_id:
                    continue
                attempt_id = event.get("attempt_id")
                if not isinstance(attempt_id, str) or attempt_id not in active_attempts:
                    continue
                task_id, attempt = active_attempts[attempt_id]
                record = task_results[task_id]
                if (
                    event.get("task_id") != task_id
                    or record.attempts != attempt
                    or record.current_attempt_id != attempt_id
                ):
                    continue
                timeout_seconds = task_lookup[task_id].timeout_seconds
                if timeout_seconds is not None and attempt_id not in attempt_deadlines:
                    attempt_deadlines[attempt_id] = time.monotonic() + timeout_seconds

        def fail_attempt(task_id: str, reason: str, *, timed_out: bool = False) -> None:
            nonlocal fail_fast_reason
            record = task_results[task_id]
            record.status = "failed"
            record.error = reason
            record.timed_out = timed_out
            record.attempt_errors.append(reason)
            if record.attempts <= task_lookup[task_id].retry_count:
                schedule_retry(task_id, reason)
                return
            block_dependents(task_id, reason)
            if fail_fast:
                fail_fast_reason = f"Execution stopped after task {task_id} failed: {reason}"

        def schedule_retry(task_id: str, reason: str) -> None:
            task = task_lookup[task_id]
            record = task_results[task_id]
            record.status = "pending"
            record.error = reason
            retry_ready_at[task_id] = time.monotonic() + task.retry_delay_seconds

        def expire_timed_out_attempts() -> None:
            nonlocal pool_requires_reset, run_generation, stop_reason
            now = time.monotonic()
            expired_attempts = [
                attempt_id
                for attempt_id, deadline in attempt_deadlines.items()
                if deadline <= now and attempt_id in active_attempts
            ]
            if not expired_attempts:
                return

            timed_out_reasons: dict[str, str] = {}
            for attempt_id in expired_attempts:
                task_id, attempt = active_attempts.pop(attempt_id)
                attempt_deadlines.pop(attempt_id, None)
                record = task_results[task_id]
                if record.status != "running" or record.attempts != attempt or record.current_attempt_id != attempt_id:
                    continue
                record.completed_at = now
                reason = _format_timeout_error(task_id, attempt, task_lookup[task_id].timeout_seconds)
                timed_out_reasons[task_id] = reason
                fail_attempt(task_id, reason, timed_out=True)

            # A logical timeout means at least one worker may still be executing
            # against this run's shared context. Recycle immediately so retries
            # cannot sit behind a hung attempt. If unrelated attempts were also
            # in flight, fail them instead of replaying potentially side-effectful
            # work after the reset.
            concurrent_work_was_cancelled = bool(active_attempts)
            if concurrent_work_was_cancelled:
                stop_reason = "Pool reset after task timeout cancelled concurrent in-flight work"
                mark_outstanding_terminal(stop_reason, running_status="failed")
                for task_id, reason in timed_out_reasons.items():
                    record = task_results[task_id]
                    record.status = "failed"
                    record.error = reason
                    record.timed_out = True
                    record.completed_at = now

            pool_requires_reset = True
            self.actor_hypervisor.restart(timeout=2.0)
            run_generation = self.actor_hypervisor.generation
            pool_requires_reset = False

            if concurrent_work_was_cancelled:
                raise DAGExecutionError(stop_reason)

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

        def mark_outstanding_terminal(
            reason: str,
            *,
            running_status: str = "blocked",
            timed_out: bool = False,
        ) -> None:
            timestamp = time.monotonic()
            for record in task_results.values():
                if record.status == "running":
                    record.status = running_status
                    record.error = reason
                    record.timed_out = timed_out
                    record.completed_at = timestamp
                    if running_status == "failed":
                        record.attempt_errors.append(reason)
                elif record.status == "pending":
                    record.status = "blocked"
                    record.error = reason
                    record.completed_at = timestamp
            retry_ready_at.clear()
            active_attempts.clear()
            attempt_deadlines.clear()

        try:
            # Validate every route before allocating context or submitting any work.
            for task in dag.tasks:
                self.actor_hypervisor.resolve_route(
                    {
                        "tool_name": task.tool_name,
                        "model_type": task.model_type,
                    }
                )

            task_results.update(
                {
                    task.id: TaskExecutionRecord(
                        task_id=task.id,
                        run_id=run_id,
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
            )
            for task in dag.tasks:
                for dep in task.dependencies:
                    dependents[dep].append(task.id)

            if not dag.tasks:
                return build_report()

            if context is not None:
                owned_context_manager = ContextMemoryManager()
                context_metadata = owned_context_manager.load_and_share(context)

            if not self.actor_hypervisor.started:
                self.actor_hypervisor.start()
                started_here = True
            run_generation = self.actor_hypervisor.generation

            for task in dag.tasks:
                if dependency_counts[task.id] == 0:
                    submit_task(task.id)

            while True:
                observed_generation = self.actor_hypervisor.generation
                if run_generation is not None and observed_generation != run_generation:
                    stop_reason = (
                        "ActorHypervisor pool generation changed unexpectedly while tasks were in flight "
                        f"(expected {run_generation}, observed {observed_generation})"
                    )
                    mark_outstanding_terminal(stop_reason, running_status="failed")
                    pool_requires_reset = True
                    raise DAGExecutionError(stop_reason)

                submit_due_retries()
                record_started_attempts()
                expire_timed_out_attempts()

                if fail_fast_reason is not None:
                    had_active_attempts = bool(active_attempts)
                    mark_outstanding_terminal(f"Cancelled by fail-fast: {fail_fast_reason}")
                    pool_requires_reset = pool_requires_reset or had_active_attempts
                    stop_reason = fail_fast_reason
                    raise DAGExecutionError(fail_fast_reason)

                if active_attempts and not self.actor_hypervisor.started:
                    stop_reason = "ActorHypervisor stopped unexpectedly while tasks were in flight"
                    mark_outstanding_terminal(stop_reason, running_status="failed")
                    pool_requires_reset = True
                    raise DAGExecutionError(stop_reason)

                dead_workers = self.actor_hypervisor.unexpected_worker_deaths()
                if dead_workers:
                    details = "; ".join(
                        f"worker_id={item['worker_id']} route={item['route']!r} exitcode={item['exitcode']}"
                        for item in dead_workers
                    )
                    stop_reason = f"Worker process died unexpectedly: {details}"
                    mark_outstanding_terminal(stop_reason, running_status="failed")
                    pool_requires_reset = True
                    raise DAGExecutionError(stop_reason)

                finished = sum(
                    1 for record in task_results.values() if record.status in {"completed", "failed", "blocked"}
                )
                if finished == len(dag.tasks):
                    break

                remaining_timeout = self._remaining_timeout(started_at, timeout)
                if remaining_timeout is not None and remaining_timeout <= 0:
                    stop_reason = "DAG execution timed out"
                    mark_outstanding_terminal(stop_reason, running_status="failed", timed_out=True)
                    pool_requires_reset = True
                    raise DAGExecutionError(stop_reason)

                if not active_attempts:
                    if retry_ready_at:
                        wait_for_retry = max(0.0, min(retry_ready_at.values()) - time.monotonic())
                        if remaining_timeout is not None:
                            wait_for_retry = min(wait_for_retry, remaining_timeout)
                        time.sleep(min(0.05, wait_for_retry))
                        continue
                    pending = [task_id for task_id, record in task_results.items() if record.status == "pending"]
                    stop_reason = f"DAG execution stalled with pending tasks: {', '.join(pending)}"
                    mark_outstanding_terminal(stop_reason)
                    raise DAGExecutionError(stop_reason)

                result = self.actor_hypervisor.get_result(
                    timeout=self._next_wait_timeout(remaining_timeout, retry_ready_at, attempt_deadlines)
                )
                if result is None:
                    continue

                # A shared pool may still surface an old generation result. Never
                # correlate it by task ID alone.
                if result.get("run_id") != run_id:
                    continue
                attempt_id = result.get("attempt_id")
                if not isinstance(attempt_id, str) or attempt_id not in active_attempts:
                    continue
                task_id, attempt = active_attempts.pop(attempt_id)
                if result.get("task_id") != task_id:
                    stop_reason = (
                        f"Worker result correlation failed for attempt {attempt_id}: "
                        f"expected task {task_id}, got {result.get('task_id')!r}"
                    )
                    mark_outstanding_terminal(stop_reason, running_status="failed")
                    pool_requires_reset = True
                    raise DAGExecutionError(stop_reason)
                record = task_results[task_id]
                attempt_deadlines.pop(attempt_id, None)
                record.completed_at = time.monotonic()
                if record.status != "running" or record.attempts != attempt or record.current_attempt_id != attempt_id:
                    continue

                if result.get("error"):
                    fail_attempt(task_id, result["error"])
                    continue

                record.status = "completed"
                record.error = None
                record.timed_out = False
                record.result = result.get("result")
                execution_order.append(task_id)

                for dependent_id in dependents[task_id]:
                    dependent_record = task_results[dependent_id]
                    if dependent_record.status != "pending":
                        continue
                    dependency_counts[dependent_id] -= 1
                    if dependency_counts[dependent_id] == 0:
                        submit_task(dependent_id)

            report = build_report()

            if report.failed_task_ids:
                message = f"DAG execution failed for tasks: {', '.join(report.failed_task_ids)}"
                raise DAGExecutionError(message, report=report)
            if fail_fast and report.blocked_task_ids:
                message = f"DAG execution blocked downstream tasks: {', '.join(report.blocked_task_ids)}"
                raise DAGExecutionError(message, report=report)
            return report
        except DAGExecutionError as exc:
            if exc.report is None:
                exc.report = build_report()
            if fail_fast:
                raise
            return exc.report
        except Exception:
            if active_attempts:
                pool_requires_reset = True
                mark_outstanding_terminal("Execution aborted by an unexpected orchestrator error")
            raise
        finally:
            try:
                if pool_requires_reset:
                    if started_here:
                        if self.actor_hypervisor.started:
                            self.actor_hypervisor.shutdown(timeout=2.0, force=True)
                    else:
                        self.actor_hypervisor.restart(timeout=2.0)
                elif started_here:
                    self.actor_hypervisor.shutdown()
            finally:
                try:
                    if owned_context_manager is not None:
                        owned_context_manager.close()
                finally:
                    self.actor_hypervisor.release_run(run_id)

    @staticmethod
    def _remaining_timeout(started_at: float, timeout: float | None) -> float | None:
        if timeout is None:
            return None
        return timeout - (time.monotonic() - started_at)

    @staticmethod
    def _next_wait_timeout(
        remaining_timeout: float | None,
        retry_ready_at: dict[str, float],
        attempt_deadlines: dict[str, float],
    ) -> float:
        wait_timeout = 0.1
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
