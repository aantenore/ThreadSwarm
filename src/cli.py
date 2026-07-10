"""Command line interface for ThreadSwarm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from src.compiler import SemanticCompilationError, SemanticCompiler, parse_task_dag_json
from src.config import ThreadSwarmConfig, ThreadSwarmConfigError
from src.demos.incident_triage import load_bundle_text, run_demo
from src.engine import DAGExecutionError, DAGExecutionReport, DAGOrchestrator
from src.evals import default_golden_path, evaluate_golden_path
from src.tools import build_text_tool_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="threadswarm", description="CPU-first DAG runtime utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-dag", help="Validate a TaskDAG JSON file.")
    validate_parser.add_argument("dag_file", type=Path, help="Path to a JSON DAG array or object with a tasks key.")
    validate_parser.set_defaults(handler=_handle_validate_dag)

    compile_parser = subparsers.add_parser("compile", help="Compile a natural language prompt into a TaskDAG.")
    compile_parser.add_argument("prompt", help="High-level intent to decompose.")
    compile_parser.add_argument("--base-url", help="OpenAI-compatible API base URL.")
    compile_parser.add_argument("--model", help="Model name to request from the compiler provider.")
    compile_parser.add_argument("--timeout", type=float, help="Compiler HTTP timeout in seconds.")
    compile_parser.set_defaults(handler=_handle_compile)

    run_parser = subparsers.add_parser("run-dag", help="Run a TaskDAG JSON file with a built-in local toolkit.")
    run_parser.add_argument("dag_file", type=Path, help="Path to a JSON DAG array or object with a tasks key.")
    run_parser.add_argument("--toolkit", choices=("text",), default="text", help="Built-in local toolkit to use.")
    run_parser.add_argument("--payload", default=None, help="Inline text payload for the DAG.")
    run_parser.add_argument("--input-file", type=Path, default=None, help="Read text payload from this file.")
    run_parser.add_argument("--json", action="store_true", help="Print final result as JSON.")
    run_parser.add_argument("--report-file", type=Path, help="Write the full execution report JSON to this path.")
    run_parser.add_argument(
        "--no-fail-fast", action="store_true", help="Return a report instead of raising on task failure."
    )
    run_parser.set_defaults(handler=_handle_run_dag)

    eval_parser = subparsers.add_parser("eval-golden", help="Run deterministic golden eval JSON cases.")
    eval_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Golden eval JSON file or directory. Defaults to the packaged deterministic fixtures.",
    )
    eval_parser.add_argument("--json", action="store_true", help="Print eval results as JSON.")
    eval_parser.set_defaults(handler=_handle_eval_golden)

    demo_parser = subparsers.add_parser("demo", help="Run packaged demos.")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_name", required=True)
    incident_parser = demo_subparsers.add_parser("incident-triage", help="Run the local-tool incident triage demo.")
    incident_parser.add_argument("--input-file", type=Path, default=None, help="Path to an incident bundle text file.")
    incident_parser.add_argument("--json", action="store_true", help="Print final report payload as JSON.")
    incident_parser.add_argument("--report-file", type=Path, help="Write the full execution report JSON to this path.")
    incident_parser.set_defaults(handler=_handle_incident_triage_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, DAGExecutionError, ThreadSwarmConfigError, SemanticCompilationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _handle_validate_dag(args: argparse.Namespace) -> int:
    dag = parse_task_dag_json(args.dag_file.read_text(encoding="utf-8"))
    validation_error = dag.validation_error()
    if validation_error:
        print(f"error: invalid DAG: {validation_error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "task_count": len(dag.tasks),
                "task_ids": [task.id for task in dag.tasks],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _handle_compile(args: argparse.Namespace) -> int:
    config = ThreadSwarmConfig.from_env(os.environ)
    if args.base_url:
        config = ThreadSwarmConfig(
            llm_base_url=args.base_url,
            llm_model=config.llm_model,
            llm_timeout=config.llm_timeout,
            default_workers=config.default_workers,
        )
    if args.model:
        config = ThreadSwarmConfig(
            llm_base_url=config.llm_base_url,
            llm_model=args.model,
            llm_timeout=config.llm_timeout,
            default_workers=config.default_workers,
        )
    if args.timeout is not None:
        config = ThreadSwarmConfig(
            llm_base_url=config.llm_base_url,
            llm_model=config.llm_model,
            llm_timeout=args.timeout,
            default_workers=config.default_workers,
        )

    dag = SemanticCompiler.from_config(config).compile(args.prompt)
    print(dag.model_dump_json(indent=2))
    return 0


def _handle_run_dag(args: argparse.Namespace) -> int:
    dag = parse_task_dag_json(args.dag_file.read_text(encoding="utf-8"))
    validation_error = dag.validation_error()
    if validation_error:
        raise ValueError(f"invalid DAG: {validation_error}")

    payload = _read_payload(args.payload, args.input_file)
    config = ThreadSwarmConfig.from_env(os.environ)
    registry = _build_registry(args.toolkit, default_workers=config.default_workers)
    report = DAGOrchestrator(registry.create_hypervisor()).run(
        dag,
        context=payload,
        fail_fast=not args.no_fail_fast,
    )
    if args.report_file:
        _write_report_file(report, args.report_file)
    if args.json:
        print(json.dumps(report.final_result, indent=2, sort_keys=True))
    else:
        print(report.final_result)
    return 0


def _handle_eval_golden(args: argparse.Namespace) -> int:
    results = evaluate_golden_path(args.path if args.path is not None else default_golden_path())
    passed = all(result.passed for result in results)
    payload = {
        "passed": passed,
        "total_cases": len(results),
        "failed_cases": sum(1 for result in results if not result.passed),
        "cases": [result.to_dict() for result in results],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} {result.name}")
            for error in result.errors:
                print(f"  - {error}")
        print(f"{payload['total_cases'] - payload['failed_cases']}/{payload['total_cases']} golden evals passed")
    return 0 if passed else 1


def _handle_incident_triage_demo(args: argparse.Namespace) -> int:
    report = run_demo(load_bundle_text(args.input_file))
    if args.report_file:
        _write_report_file(report, args.report_file)
    print("Execution order:", report.execution_order)
    print()
    if args.json:
        print(json.dumps(report.final_result, indent=2, sort_keys=True))
    else:
        print(report.final_result["report_markdown"])
    return 0


def _write_report_file(report: DAGExecutionReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(include_dependency_results=True), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_payload(payload: str | None, input_file: Path | None) -> str:
    if payload is not None and input_file is not None:
        raise ValueError("Provide either --payload or --input-file, not both")
    if input_file is not None:
        return input_file.read_text(encoding="utf-8")
    return payload or ""


def _build_registry(toolkit: str, *, default_workers: int | None = None):
    if toolkit == "text":
        return build_text_tool_registry(default_workers=default_workers)
    raise ValueError(f"Unknown toolkit: {toolkit}")


if __name__ == "__main__":
    raise SystemExit(main())
