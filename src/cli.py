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

    demo_parser = subparsers.add_parser("demo", help="Run packaged demos.")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_name", required=True)
    incident_parser = demo_subparsers.add_parser("incident-triage", help="Run the local-tool incident triage demo.")
    incident_parser.add_argument("--input-file", type=Path, default=None, help="Path to an incident bundle text file.")
    incident_parser.add_argument("--json", action="store_true", help="Print final report payload as JSON.")
    incident_parser.set_defaults(handler=_handle_incident_triage_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, ThreadSwarmConfigError, SemanticCompilationError, ValueError) as exc:
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


def _handle_incident_triage_demo(args: argparse.Namespace) -> int:
    report = run_demo(load_bundle_text(args.input_file))
    print("Execution order:", report.execution_order)
    print()
    if args.json:
        print(json.dumps(report.final_result, indent=2, sort_keys=True))
    else:
        print(report.final_result["report_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
