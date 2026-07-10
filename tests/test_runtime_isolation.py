"""Adversarial tests for run isolation and actor lifecycle hardening."""

from __future__ import annotations

import os
import multiprocessing
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from threadswarm.compiler import SubTask, TaskDAG
from threadswarm.engine import (
    ActorHypervisor,
    ConcurrentRunError,
    DAGExecutionError,
    DAGOrchestrator,
    UnknownRouteError,
)


def _echo_payload(context, instruction, task_id, modality, route_key):
    payload = context.get("payload") if isinstance(context, dict) else context
    return {"payload": payload, "route_key": route_key, "task_id": task_id}


def _delay_first_payload(context, instruction, task_id, modality, route_key):
    payload = context["payload"]
    if payload == "run-a":
        time.sleep(0.5)
    return {"payload": payload, "task_id": task_id}


def _crash_worker(context, instruction, task_id, modality, route_key):
    os._exit(23)


def _fail_immediately(context, instruction, task_id, modality, route_key):
    raise RuntimeError("intentional failure")


def _sleep_for_fail_fast_probe(context, instruction, task_id, modality, route_key):
    time.sleep(5.0)
    return {"task_id": task_id}


def _return_unpicklable_result(context, instruction, task_id, modality, route_key):
    return {"callback": lambda: None}


def _return_shared_payload_view(context, instruction, task_id, modality, route_key):
    return context["payload"]


def _always_hangs(context, instruction, task_id, modality, route_key):
    time.sleep(5.0)
    return {"task_id": task_id}


def _single_task_dag(*, tool_name: str | None = "echo") -> TaskDAG:
    return TaskDAG(
        tasks=[
            SubTask(
                id="same-task",
                description="Echo",
                instruction="Echo the payload",
                tool_name=tool_name,
            )
        ]
    )


def test_homogeneous_pool_rejects_non_positive_worker_count():
    with pytest.raises(ValueError, match="positive integer"):
        ActorHypervisor(num_workers=0, run_inference_hook=_echo_payload)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="fork is unavailable")
def test_prestarted_fork_pool_keeps_parent_shared_memory_across_timeout_retry():
    probe = textwrap.dedent(
        """
        import multiprocessing
        import time

        from threadswarm.compiler import SubTask, TaskDAG
        from threadswarm.engine import ActorHypervisor, DAGOrchestrator

        multiprocessing.set_start_method("fork", force=True)

        def retry_after_timeout(context, instruction, task_id, modality, route_key):
            if context["attempt"] == 1:
                time.sleep(2.0)
            return {"attempt": context["attempt"], "payload": context["payload"]}

        hypervisor = ActorHypervisor(
            worker_configs=[
                {
                    "model_type": "retry",
                    "num_workers": 1,
                    "run_inference_hook": retry_after_timeout,
                }
            ]
        )
        hypervisor.start()
        try:
            dag = TaskDAG(
                tasks=[
                    SubTask(
                        id="retry",
                        description="Retry after timeout",
                        instruction="Retry",
                        tool_name="retry",
                        retry_count=1,
                        timeout_seconds=0.1,
                    )
                ]
            )
            report = DAGOrchestrator(hypervisor).run(dag, context="still-owned", fail_fast=False)
            record = report.task_results["retry"]
            assert record.status == "completed", record.error
            assert record.attempts == 2
            assert record.result == {"attempt": 2, "payload": "still-owned"}
        finally:
            hypervisor.shutdown(force=True)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "resource_tracker" not in completed.stderr


def test_worker_result_and_report_propagate_run_and_attempt_ids():
    # Homogeneous mode intentionally accepts arbitrary routing hints for legacy callers.
    with ActorHypervisor(num_workers=1, run_inference_hook=_echo_payload) as hypervisor:
        hypervisor.submit(
            {
                "run_id": "direct-run",
                "attempt_id": "direct-attempt",
                "task_id": "direct-task",
                "instruction": "Echo",
                "tool_name": "legacy-arbitrary-route",
            }
        )
        worker_result = hypervisor.get_result(timeout=5.0)

    assert worker_result is not None
    assert worker_result["run_id"] == "direct-run"
    assert worker_result["attempt_id"] == "direct-attempt"

    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "echo", "num_workers": 1, "run_inference_hook": _echo_payload},
        ]
    )
    report = DAGOrchestrator(hypervisor).run(_single_task_dag(), context="payload")
    record = report.task_results["same-task"]

    assert report.run_id
    assert record.run_id == report.run_id
    assert len(record.attempt_ids) == 1
    assert record.current_attempt_id == record.attempt_ids[0]
    assert report.to_dict()["task_results"]["same-task"]["attempt_ids"] == record.attempt_ids


@pytest.mark.parametrize("tool_name", [None, "unknown-route"])
def test_explicit_worker_routes_fail_closed_before_start(tool_name):
    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "echo", "num_workers": 1, "run_inference_hook": _echo_payload},
        ]
    )
    dag = TaskDAG(
        tasks=[
            SubTask(id="valid", description="Valid", instruction="Valid", tool_name="echo"),
            SubTask(id="invalid", description="Invalid", instruction="Invalid", tool_name=tool_name),
        ]
    )

    with pytest.raises(UnknownRouteError):
        DAGOrchestrator(hypervisor).run(dag)

    assert hypervisor.started is False
    assert hypervisor.active_run_id is None


def test_timed_out_run_is_isolated_and_prestarted_pool_is_recreated():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "echo", "num_workers": 1, "run_inference_hook": _delay_first_payload},
        ]
    )
    hypervisor.start()
    first_generation = hypervisor.generation
    orchestrator = DAGOrchestrator(hypervisor)

    try:
        run_a = orchestrator.run(
            _single_task_dag(),
            context="run-a",
            timeout=0.05,
            fail_fast=False,
        )
        second_generation = hypervisor.generation
        run_b = orchestrator.run(
            _single_task_dag(),
            context="run-b",
            timeout=5.0,
        )
    finally:
        hypervisor.shutdown(force=True)

    assert run_a.run_id != run_b.run_id
    assert run_a.stop_reason == "DAG execution timed out"
    assert run_a.task_results["same-task"].status == "failed"
    assert all(record.status in {"completed", "failed", "blocked"} for record in run_a.task_results.values())
    assert second_generation == first_generation + 1
    assert run_b.final_result["payload"] == "run-b"
    assert run_a.task_results["same-task"].attempt_ids != run_b.task_results["same-task"].attempt_ids


def test_only_one_orchestrated_run_can_own_a_hypervisor():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "echo", "num_workers": 1, "run_inference_hook": _delay_first_payload},
        ]
    )
    hypervisor.start()
    orchestrator = DAGOrchestrator(hypervisor)
    thread_errors: list[BaseException] = []

    def run_slow_task() -> None:
        try:
            orchestrator.run(_single_task_dag(), context="run-a", timeout=5.0)
        except BaseException as exc:  # pragma: no cover - asserted below
            thread_errors.append(exc)

    thread = threading.Thread(target=run_slow_task)
    thread.start()
    deadline = time.monotonic() + 2.0
    while hypervisor.active_run_id is None and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert hypervisor.active_run_id is not None
        with pytest.raises(ConcurrentRunError):
            orchestrator.run(_single_task_dag(), context="run-b", timeout=1.0)
        thread.join(timeout=5.0)
    finally:
        hypervisor.shutdown(force=True)

    assert thread.is_alive() is False
    assert thread_errors == []


def test_unexpected_worker_death_becomes_bounded_dag_error():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "crash", "num_workers": 1, "run_inference_hook": _crash_worker},
        ]
    )
    dag = _single_task_dag(tool_name="crash")
    started_at = time.monotonic()

    with pytest.raises(DAGExecutionError) as exc_info:
        DAGOrchestrator(hypervisor).run(dag)

    elapsed = time.monotonic() - started_at
    report = exc_info.value.report
    assert elapsed < 5.0
    assert report is not None
    assert "Worker process died unexpectedly" in str(exc_info.value)
    assert report.task_results["same-task"].status == "failed"
    assert "exitcode=23" in (report.task_results["same-task"].error or "")
    assert hypervisor.started is False


def test_external_pool_shutdown_cannot_strand_an_active_run():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "echo", "num_workers": 1, "run_inference_hook": _delay_first_payload},
        ]
    )
    hypervisor.start()
    orchestrator = DAGOrchestrator(hypervisor)
    thread_errors: list[BaseException] = []

    def run_slow_task() -> None:
        try:
            orchestrator.run(_single_task_dag(), context="run-a")
        except BaseException as exc:
            thread_errors.append(exc)

    thread = threading.Thread(target=run_slow_task)
    thread.start()
    deadline = time.monotonic() + 2.0
    while hypervisor.active_run_id is None and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    hypervisor.shutdown(force=True)
    thread.join(timeout=3.0)

    try:
        assert thread.is_alive() is False
        assert len(thread_errors) == 1
        assert isinstance(thread_errors[0], DAGExecutionError)
        assert "stopped unexpectedly" in str(thread_errors[0])
        assert hypervisor.started is True
    finally:
        hypervisor.shutdown(force=True)


def test_external_pool_restart_cannot_strand_an_active_run():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "echo", "num_workers": 1, "run_inference_hook": _delay_first_payload},
        ]
    )
    hypervisor.start()
    initial_generation = hypervisor.generation
    orchestrator = DAGOrchestrator(hypervisor)
    thread_errors: list[BaseException] = []

    def run_slow_task() -> None:
        try:
            orchestrator.run(_single_task_dag(), context="run-a")
        except BaseException as exc:
            thread_errors.append(exc)

    thread = threading.Thread(target=run_slow_task)
    thread.start()
    deadline = time.monotonic() + 2.0
    while hypervisor.active_run_id is None and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)

    try:
        hypervisor.restart(timeout=1.0)
        assert hypervisor.generation >= initial_generation + 1
        thread.join(timeout=3.0)

        assert thread.is_alive() is False
        assert len(thread_errors) == 1
        error = thread_errors[0]
        assert isinstance(error, DAGExecutionError)
        assert any(
            reason in str(error)
            for reason in ("generation changed unexpectedly", "stopped unexpectedly")
        )
        assert error.report is not None
        assert error.report.task_results["same-task"].status == "failed"
        assert hypervisor.active_run_id is None
        assert hypervisor.started is True
    finally:
        hypervisor.shutdown(force=True)
        thread.join(timeout=3.0)


def test_fail_fast_terminates_independent_in_flight_work():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {"model_type": "fail", "num_workers": 1, "run_inference_hook": _fail_immediately},
            {"model_type": "slow", "num_workers": 1, "run_inference_hook": _sleep_for_fail_fast_probe},
        ]
    )
    dag = TaskDAG(
        tasks=[
            SubTask(id="fails", description="Fail", instruction="Fail", tool_name="fail"),
            SubTask(id="slow", description="Slow", instruction="Sleep", tool_name="slow"),
        ]
    )
    started_at = time.monotonic()

    with pytest.raises(DAGExecutionError) as exc_info:
        DAGOrchestrator(hypervisor).run(dag, fail_fast=True)

    elapsed = time.monotonic() - started_at
    report = exc_info.value.report
    assert elapsed < 4.0
    assert report is not None
    assert report.task_results["fails"].status == "failed"
    assert report.task_results["slow"].status == "blocked"
    assert "Cancelled by fail-fast" in (report.task_results["slow"].error or "")
    assert hypervisor.started is False


def test_unpicklable_task_is_rejected_before_queue_submission():
    with ActorHypervisor(num_workers=1, run_inference_hook=_echo_payload) as hypervisor:
        with pytest.raises(ValueError, match="Submitted task is not multiprocessing-serializable"):
            hypervisor.submit(
                {
                    "task_id": "bad-task",
                    "instruction": "Reject before feeder thread",
                    "callback": lambda: None,
                }
            )


def test_unpicklable_worker_result_becomes_correlated_task_failure():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {
                "model_type": "unpicklable",
                "num_workers": 1,
                "run_inference_hook": _return_unpicklable_result,
            },
        ]
    )
    report = DAGOrchestrator(hypervisor).run(
        _single_task_dag(tool_name="unpicklable"),
        context="payload",
        fail_fast=False,
        timeout=5.0,
    )

    record = report.task_results["same-task"]
    assert record.status == "failed"
    assert record.run_id == report.run_id
    assert "Worker result is not multiprocessing-serializable" in (record.error or "")


def test_returned_shared_ndarray_is_detached_before_worker_unmaps_context():
    hypervisor = ActorHypervisor(
        worker_configs=[
            {
                "model_type": "return-array",
                "num_workers": 1,
                "run_inference_hook": _return_shared_payload_view,
            },
        ]
    )
    payload = np.arange(1_000, dtype=np.int64)

    report = DAGOrchestrator(hypervisor).run(
        _single_task_dag(tool_name="return-array"),
        context=payload,
        timeout=5.0,
    )

    np.testing.assert_array_equal(report.final_result, payload)
    assert report.succeeded is True


def test_task_timeout_recycles_hung_pool_before_retrying() -> None:
    hypervisor = ActorHypervisor(
        worker_configs=[
            {
                "model_type": "hang",
                "num_workers": 1,
                "run_inference_hook": _always_hangs,
            },
        ]
    )
    dag = TaskDAG(
        tasks=[
            SubTask(
                id="same-task",
                description="Bound a hung worker",
                instruction="Hang",
                tool_name="hang",
                retry_count=1,
                timeout_seconds=0.05,
            )
        ]
    )
    started_at = time.monotonic()

    report = DAGOrchestrator(hypervisor).run(dag, fail_fast=False)

    assert time.monotonic() - started_at < 2.0
    record = report.task_results["same-task"]
    assert record.status == "failed"
    assert record.attempts == 2
    assert len(record.attempt_errors) == 2
    assert hypervisor.started is False


def test_concurrent_start_creates_only_one_pool_generation() -> None:
    hypervisor = ActorHypervisor(num_workers=1, run_inference_hook=_echo_payload)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def start_pool() -> None:
        try:
            barrier.wait()
            hypervisor.start()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=start_pool) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    try:
        assert all(thread.is_alive() is False for thread in threads)
        assert errors == []
        assert hypervisor.generation == 1
    finally:
        hypervisor.shutdown(force=True)


def test_graceful_shutdown_is_bounded_with_a_saturated_task_queue() -> None:
    hypervisor = ActorHypervisor(
        worker_configs=[
            {
                "model_type": "hang",
                "num_workers": 1,
                "run_inference_hook": _always_hangs,
            },
        ]
    )
    hypervisor.start()
    for index in range(12):
        hypervisor.submit(
            {
                "task_id": f"queued-{index}",
                "instruction": "Saturate the queue",
                "tool_name": "hang",
                "blob": b"x" * 1_000_000,
            }
        )
    task_queue = hypervisor._pools[0][1]
    feeder = task_queue._thread
    assert feeder is not None and feeder.is_alive()
    started_at = time.monotonic()

    hypervisor.shutdown(timeout=0.05)

    assert time.monotonic() - started_at < 3.0
    assert hypervisor.started is False
    assert feeder.is_alive() is False
    assert len(task_queue._buffer) == 0


def test_shutdown_aborts_a_saturated_queue_after_worker_crash() -> None:
    hypervisor = ActorHypervisor(
        worker_configs=[
            {
                "model_type": "crash",
                "num_workers": 1,
                "run_inference_hook": _crash_worker,
            },
        ]
    )
    hypervisor.start()
    hypervisor.submit(
        {
            "task_id": "trigger-crash",
            "instruction": "Crash",
            "tool_name": "crash",
        }
    )
    deadline = time.monotonic() + 3.0
    while not hypervisor.unexpected_worker_deaths() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert hypervisor.unexpected_worker_deaths()

    for index in range(12):
        hypervisor.submit(
            {
                "task_id": f"orphaned-{index}",
                "instruction": "No worker can consume this",
                "tool_name": "crash",
                "blob": b"x" * 1_000_000,
            }
        )
    task_queue = hypervisor._pools[0][1]
    feeder = task_queue._thread
    assert feeder is not None and feeder.is_alive()
    started_at = time.monotonic()

    hypervisor.shutdown(timeout=0.05)

    assert time.monotonic() - started_at < 3.0
    assert hypervisor.started is False
    assert feeder.is_alive() is False
    assert len(task_queue._buffer) == 0
