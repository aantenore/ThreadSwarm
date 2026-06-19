"""File-backed golden evals for deterministic local DAG behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.compiler import parse_task_dag_json
from src.engine import DAGExecutionReport, DAGOrchestrator
from src.tools import build_text_tool_registry


@dataclass(slots=True)
class GoldenEvalResult:
    """Outcome for one golden eval case."""

    name: str
    path: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    report: DAGExecutionReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "passed": self.passed,
            "errors": list(self.errors),
            "summary": self.report.summary() if self.report is not None else None,
        }


def evaluate_golden_path(path: Path) -> list[GoldenEvalResult]:
    """Run one golden case JSON file or every JSON file in a directory."""
    if path.is_dir():
        case_paths = sorted(item for item in path.glob("*.json") if item.is_file())
    else:
        case_paths = [path]
    if not case_paths:
        raise ValueError(f"No golden eval JSON files found at {path}")
    return [run_golden_case(case_path) for case_path in case_paths]


def run_golden_case(path: Path) -> GoldenEvalResult:
    """Run a deterministic golden eval case and compare expected report fields."""
    try:
        raw_case = json.loads(path.read_text(encoding="utf-8"))
        name = str(raw_case.get("name") or path.stem)
        dag_payload = raw_case.get("dag", raw_case.get("tasks"))
        if dag_payload is None:
            raise ValueError("Golden eval case must include 'dag' or 'tasks'")

        dag = parse_task_dag_json(json.dumps(dag_payload))
        toolkit = str(raw_case.get("toolkit", "text"))
        registry = _build_registry(toolkit)
        report = DAGOrchestrator(registry.create_hypervisor()).run(
            dag,
            context=str(raw_case.get("payload", "")),
            fail_fast=False,
        )
        report_payload = report.to_dict(include_dependency_results=True)
        errors = _compare_expectations(raw_case.get("expect", {}), report_payload)
        return GoldenEvalResult(name=name, path=str(path), passed=not errors, errors=errors, report=report)
    except Exception as exc:
        return GoldenEvalResult(name=path.stem, path=str(path), passed=False, errors=[str(exc)])


def _build_registry(toolkit: str):
    if toolkit == "text":
        return build_text_tool_registry()
    raise ValueError(f"Unknown golden eval toolkit: {toolkit}")


def _compare_expectations(expected: Any, actual: Any) -> list[str]:
    errors: list[str] = []
    _assert_subset(expected, actual, "$", errors)
    return errors


def _assert_subset(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected object, got {type(actual).__name__}")
            return
        for key, expected_value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing key")
                continue
            _assert_subset(expected_value, actual[key], f"{path}.{key}", errors)
        return

    if isinstance(expected, list):
        if expected != actual:
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")
        return

    if expected != actual:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")
