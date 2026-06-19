"""Tests for shared memory and actor pool (no real model)."""

import time

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from threadswarm.compiler.parser import SubTask, TaskDAG
from threadswarm.engine.actor_pool import ActorHypervisor, SHUTDOWN_SENTINEL
from threadswarm.engine.orchestrator import DAGExecutionError, DAGOrchestrator
from threadswarm.engine.shared_memory import (
    ContextMemoryManager,
    VisionMemoryManager,
    attach_and_reconstruct,
    load_image,
)
from threadswarm.engine.tool_registry import LocalToolRegistry


class TextToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: str
    dependency_results: dict
    task_id: str
    instruction: str
    modality: str
    tool_name: str | None
    model_type: str | None
    route_key: str | None
    attempt: int | None


class NormalizedTextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    normalized: str
    modality: str


class RequiredValueOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


# Module-level callables, picklable on Windows. New hook signature: (context, instruction, task_id, modality, model_type).
def _stub_inference(context, instruction, task_id, modality, model_type):
    return {"task_id": task_id, "instruction": instruction, "done": True}


def _check_shape_inference(context, instruction, task_id, modality, model_type):
    return {"shape": getattr(context, "shape", None), "task_id": task_id}


def _local_tool_inference(context, instruction, task_id, modality, model_type):
    payload = context["payload"]
    dependency_results = context["dependency_results"]

    if model_type == "normalize-text":
        return {
            "instruction": instruction,
            "normalized": payload.strip().upper(),
            "modality": modality,
        }
    if model_type == "summarize-text":
        previous = dependency_results["task_1"]["normalized"]
        return {
            "instruction": instruction,
            "summary": f"{previous} -> {instruction}",
            "model_type": model_type,
        }
    if model_type == "finalize-report":
        previous = dependency_results["task_2"]["summary"]
        return {
            "instruction": instruction,
            "final": previous.lower(),
        }
    raise RuntimeError(f"Unexpected tool: {model_type}")


def _branching_tool_inference(context, instruction, task_id, modality, model_type):
    payload = context["payload"]
    dependency_results = context["dependency_results"]

    if model_type == "uppercase-root":
        return {"base": payload.upper()}
    if model_type == "left-branch":
        return {"value": f"L:{dependency_results['root']['base']}"}
    if model_type == "right-branch":
        return {"value": f"R:{dependency_results['root']['base']}"}
    raise RuntimeError(f"Unexpected tool: {model_type}")


def _failing_local_tool(context, instruction, task_id, modality, model_type):
    if model_type == "fail-tool":
        raise RuntimeError("boom")
    return {"task_id": task_id, "instruction": instruction}


def _binary_result_tool(context, instruction, task_id, modality, model_type):
    return {
        "payload": b"hello bytes",
        "shape": (2, 3),
        "array_like": np.array([1, 2, 3], dtype=np.int64),
    }


def _invalid_contract_tool(context, instruction, task_id, modality, model_type):
    return {"unexpected": "shape"}


_FLAKY_TOOL_ATTEMPTS: dict[str, int] = {}


def _flaky_once_tool(context, instruction, task_id, modality, model_type):
    attempts = _FLAKY_TOOL_ATTEMPTS.get(task_id, 0) + 1
    _FLAKY_TOOL_ATTEMPTS[task_id] = attempts
    if attempts == 1:
        raise RuntimeError("transient failure")
    return {"attempts": attempts, "instruction": instruction}


def _always_fail_tool(context, instruction, task_id, modality, model_type):
    raise RuntimeError("still failing")


def _slow_tool(context, instruction, task_id, modality, model_type):
    time.sleep(0.2)
    return {"task_id": task_id, "attempt": context["attempt"]}


def _slow_first_attempt_tool(context, instruction, task_id, modality, model_type):
    attempt = context["attempt"]
    if attempt == 1:
        time.sleep(0.4)
    return {"task_id": task_id, "attempt": attempt}


def test_context_memory_manager_ndarray():
    """ContextMemoryManager: ndarray roundtrip (e.g. image/audio)."""
    manager = ContextMemoryManager(name_prefix="test_")
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    meta = manager.load_and_share(arr)
    assert "name" in meta and "shape" in meta and "dtype" in meta and meta.get("payload_type") == "ndarray"
    assert meta["shape"] == (64, 64, 3)
    shm, view = attach_and_reconstruct(meta)
    np.testing.assert_array_equal(view, arr)
    shm.close()
    manager.close()


def test_context_memory_manager_text():
    """ContextMemoryManager: text roundtrip."""
    manager = ContextMemoryManager(name_prefix="test_")
    text = "Hello world\nCode snippet: def foo(): pass"
    meta = manager.load_and_share(text)
    assert meta.get("payload_type") == "text"
    assert "size" in meta
    shm, reconstructed = attach_and_reconstruct(meta)
    assert reconstructed == text
    shm.close()
    manager.close()


def test_vision_memory_manager_alias():
    """VisionMemoryManager is alias for ContextMemoryManager; ndarray still works."""
    manager = VisionMemoryManager(name_prefix="test_")
    arr = np.ones((8, 8, 3), dtype=np.uint8) * 42
    meta = manager.load_and_share(arr)
    assert meta["shape"] == (8, 8, 3)
    shm, view = attach_and_reconstruct(meta)
    np.testing.assert_array_equal(view, arr)
    shm.close()
    manager.close()


def test_actor_pool_submit_and_result():
    """Homogeneous pool with stub hook; task can use context_metadata or image_metadata."""
    with ActorHypervisor(num_workers=2, run_inference_hook=_stub_inference) as hv:
        hv.submit({
            "task_id": "t1",
            "instruction": "Do something",
            "context_metadata": None,
        })
        out = hv.get_result(timeout=5.0)
        assert out is not None
        assert out.get("error") is None
        assert out.get("result", {}).get("task_id") == "t1"
        assert out["result"]["done"] is True


def test_actor_pool_with_shared_memory_context():
    """Submit task with context_metadata (ndarray); worker reconstructs and runs hook."""
    manager = ContextMemoryManager(name_prefix="test_")
    arr = np.ones((8, 8, 3), dtype=np.uint8) * 42
    meta = manager.load_and_share(arr)

    with ActorHypervisor(num_workers=1, run_inference_hook=_check_shape_inference) as hv:
        hv.submit({
            "task_id": "img_task",
            "instruction": "inspect",
            "context_metadata": meta,
        })
        out = hv.get_result(timeout=5.0)
        assert out is not None and out.get("error") is None
        assert out["result"]["shape"] == (8, 8, 3)

    manager.close()


def test_dag_orchestrator_runs_linear_workflow():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Normalize", instruction="Normalize input", dependencies=[], tool_name="normalize-text"),
            SubTask(id="task_2", description="Summarize", instruction="Summarize normalized text", dependencies=["task_1"], tool_name="summarize-text"),
            SubTask(id="task_3", description="Finalize", instruction="Finalize answer", dependencies=["task_2"], tool_name="finalize-report"),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("normalize-text", _local_tool_inference, description="Normalize text")
    registry.register("summarize-text", _local_tool_inference, description="Summarize normalized text")
    registry.register("finalize-report", _local_tool_inference, description="Finalize report")

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="  ciao swarm  ")

    assert report.succeeded is True
    assert report.execution_order == ["task_1", "task_2", "task_3"]
    assert report.leaf_task_ids == ["task_3"]
    assert report.task_results["task_2"].dependency_results == {
        "task_1": {
            "instruction": "Normalize input",
            "normalized": "CIAO SWARM",
            "modality": "text",
        }
    }
    assert report.final_result == {
        "instruction": "Finalize answer",
        "final": "ciao swarm -> summarize normalized text",
    }

    exported = report.to_dict(include_dependency_results=True)
    assert exported["summary"]["succeeded"] is True
    assert exported["summary"]["total_tasks"] == 3
    assert exported["summary"]["completed_tasks"] == 3
    assert exported["summary"]["failed_tasks"] == 0
    assert exported["summary"]["blocked_tasks"] == 0
    assert exported["summary"]["duration_seconds"] >= 0
    assert exported["task_results"]["task_2"]["tool_name"] == "summarize-text"
    assert exported["task_results"]["task_2"]["dependency_results"]["task_1"]["normalized"] == "CIAO SWARM"


def test_local_tool_contracts_validate_output_and_expose_metadata():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Normalize", instruction="Normalize input", dependencies=[], tool_name="normalize-text"),
        ]
    )

    registry = LocalToolRegistry()
    registry.register(
        "normalize-text",
        _local_tool_inference,
        description="Normalize text",
        modalities=("text",),
        input_schema=TextToolInput,
        output_schema=NormalizedTextOutput,
        risk_class="compute_only",
        side_effect_class="none",
        result_size_limit=500,
    )

    contracts = registry.contracts()
    assert contracts["normalize-text"]["risk_class"] == "compute_only"
    assert contracts["normalize-text"]["side_effect_class"] == "none"
    assert contracts["normalize-text"]["output_schema"]["properties"]["normalized"]["type"] == "string"

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="  contract  ")

    assert report.succeeded is True
    assert report.final_result == {
        "instruction": "Normalize input",
        "normalized": "CONTRACT",
        "modality": "text",
    }


def test_local_tool_contract_validation_failure_blocks_task():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Bad", instruction="Return invalid shape", dependencies=[], tool_name="bad-tool"),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("bad-tool", _invalid_contract_tool, output_schema=RequiredValueOutput)

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="ignored", fail_fast=False)

    assert report.succeeded is False
    assert report.failed_task_ids == ["task_1"]
    assert "Tool bad-tool output schema validation failed" in (report.task_results["task_1"].error or "")


def test_dag_orchestrator_retries_transient_task_failure():
    dag = TaskDAG(
        tasks=[
            SubTask(
                id="flaky_retry_task",
                description="Flaky",
                instruction="Retry once",
                dependencies=[],
                tool_name="flaky-tool",
                retry_count=1,
            ),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("flaky-tool", _flaky_once_tool, num_workers=1)

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="ignored")
    record = report.task_results["flaky_retry_task"]

    assert report.succeeded is True
    assert record.attempts == 2
    assert record.attempt_errors == ["transient failure"]
    assert report.final_result == {"attempts": 2, "instruction": "Retry once"}
    exported = report.to_dict()
    assert exported["task_results"]["flaky_retry_task"]["attempts"] == 2
    assert exported["task_results"]["flaky_retry_task"]["attempt_errors"] == ["transient failure"]


def test_dag_orchestrator_reports_exhausted_retries():
    dag = TaskDAG(
        tasks=[
            SubTask(
                id="always_fail_task",
                description="Always fail",
                instruction="Exhaust retries",
                dependencies=[],
                tool_name="always-fail-tool",
                retry_count=2,
            ),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("always-fail-tool", _always_fail_tool, num_workers=1)

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="ignored", fail_fast=False)
    record = report.task_results["always_fail_task"]

    assert report.succeeded is False
    assert record.status == "failed"
    assert record.attempts == 3
    assert record.attempt_errors == ["still failing", "still failing", "still failing"]


def test_dag_orchestrator_marks_task_timeout_and_blocks_dependents():
    dag = TaskDAG(
        tasks=[
            SubTask(
                id="slow_task",
                description="Slow",
                instruction="Run slowly",
                dependencies=[],
                tool_name="slow-tool",
                timeout_seconds=0.05,
            ),
            SubTask(
                id="blocked_task",
                description="Blocked",
                instruction="Never runs",
                dependencies=["slow_task"],
                tool_name="slow-tool",
            ),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("slow-tool", _slow_tool, num_workers=1)

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="ignored", fail_fast=False)
    record = report.task_results["slow_task"]

    assert report.succeeded is False
    assert report.failed_task_ids == ["slow_task"]
    assert report.blocked_task_ids == ["blocked_task"]
    assert record.status == "failed"
    assert record.timed_out is True
    assert record.timeout_seconds == 0.05
    assert "timed out after 0.05 seconds" in (record.error or "")


def test_dag_orchestrator_retries_after_timeout_and_ignores_late_result():
    dag = TaskDAG(
        tasks=[
            SubTask(
                id="slow_once_task",
                description="Slow once",
                instruction="Retry after timeout",
                dependencies=[],
                tool_name="slow-first-attempt-tool",
                retry_count=1,
                timeout_seconds=0.15,
            ),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("slow-first-attempt-tool", _slow_first_attempt_tool, num_workers=2)

    hypervisor = registry.create_hypervisor()
    hypervisor.start()
    try:
        orchestrator = DAGOrchestrator(hypervisor)
        report = orchestrator.run(dag, context="ignored")
    finally:
        hypervisor.shutdown()
    record = report.task_results["slow_once_task"]

    assert report.succeeded is True
    assert record.status == "completed"
    assert record.error is None
    assert record.timed_out is False
    assert record.attempts == 2
    assert record.attempt_errors == ["Task slow_once_task attempt 1 timed out after 0.15 seconds"]
    assert report.final_result == {"task_id": "slow_once_task", "attempt": 2}


def test_dag_orchestrator_default_reducer_returns_leaf_mapping_for_branches():
    dag = TaskDAG(
        tasks=[
            SubTask(id="root", description="Root", instruction="Root task", dependencies=[], tool_name="uppercase-root"),
            SubTask(id="left", description="Left", instruction="Left branch", dependencies=["root"], tool_name="left-branch"),
            SubTask(id="right", description="Right", instruction="Right branch", dependencies=["root"], tool_name="right-branch"),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("uppercase-root", _branching_tool_inference, description="Uppercase root")
    registry.register("left-branch", _branching_tool_inference, description="Build left branch")
    registry.register("right-branch", _branching_tool_inference, description="Build right branch")

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="edge")

    assert report.succeeded is True
    assert report.leaf_task_ids == ["left", "right"]
    assert report.final_result == {
        "left": {"value": "L:EDGE"},
        "right": {"value": "R:EDGE"},
    }


def test_execution_report_export_is_json_safe_for_common_payloads():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Binary", instruction="Return binary result", dependencies=[], tool_name="binary-tool"),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("binary-tool", _binary_result_tool)

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="ignored")
    exported = report.to_dict()

    assert exported["final_result"]["payload"] == {
        "type": "bytes",
        "size": 11,
        "preview": "hello bytes",
    }
    assert exported["final_result"]["shape"] == [2, 3]
    assert exported["final_result"]["array_like"] == [1, 2, 3]


def test_dag_orchestrator_returns_report_on_failure_when_fail_fast_disabled():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Fail", instruction="Explode", dependencies=[], tool_name="fail-tool"),
            SubTask(id="task_2", description="Blocked", instruction="Never runs", dependencies=["task_1"], tool_name="safe-tool"),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("fail-tool", _failing_local_tool)
    registry.register("safe-tool", _failing_local_tool)

    orchestrator = DAGOrchestrator(registry.create_hypervisor())
    report = orchestrator.run(dag, context="ignored", fail_fast=False)

    assert report.succeeded is False
    assert report.failed_task_ids == ["task_1"]
    assert report.blocked_task_ids == ["task_2"]
    assert report.task_results["task_1"].status == "failed"
    assert "boom" in (report.task_results["task_1"].error or "")
    assert report.task_results["task_2"].status == "blocked"


def test_dag_orchestrator_raises_with_report_when_fail_fast_enabled():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Fail", instruction="Explode", dependencies=[], tool_name="fail-tool"),
            SubTask(id="task_2", description="Blocked", instruction="Never runs", dependencies=["task_1"], tool_name="safe-tool"),
        ]
    )

    registry = LocalToolRegistry()
    registry.register("fail-tool", _failing_local_tool)
    registry.register("safe-tool", _failing_local_tool)

    orchestrator = DAGOrchestrator(registry.create_hypervisor())

    with pytest.raises(DAGExecutionError) as exc_info:
        orchestrator.run(dag, context="ignored", fail_fast=True)

    report = exc_info.value.report
    assert report is not None
    assert report.failed_task_ids == ["task_1"]
    assert report.blocked_task_ids == ["task_2"]
