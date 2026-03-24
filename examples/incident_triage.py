"""
Runnable local-tool demo for ThreadSwarm.

This example analyzes a small incident bundle with a DAG of CPU-friendly tools:
- extract services
- extract error signatures
- extract incident notes
- infer a probable cause
- build a final report
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.compiler.parser import SubTask, TaskDAG
from src.engine import DAGExecutionReport, DAGOrchestrator, LocalToolRegistry

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = EXAMPLES_DIR / "data" / "incident_bundle.txt"


def _extract_section(bundle_text: str, header: str, next_headers: tuple[str, ...]) -> str:
    start_match = re.search(rf"^{re.escape(header)}\s*$", bundle_text, re.MULTILINE)
    if not start_match:
        return ""

    start = start_match.end()
    end = len(bundle_text)
    for candidate in next_headers:
        next_match = re.search(rf"^{re.escape(candidate)}\s*$", bundle_text[start:], re.MULTILINE)
        if next_match:
            end = min(end, start + next_match.start())
    return bundle_text[start:end].strip()


def incident_tool(context: dict[str, Any], instruction: str, task_id: str, modality: str, route_key: str | None) -> dict[str, Any]:
    """Module-level worker hook, picklable on Windows."""
    bundle_text = str(context["payload"] or "")
    dependency_results = context["dependency_results"]
    tool_name = context.get("tool_name") or route_key or ""

    if tool_name == "extract-services":
        services_match = re.search(r"^Services:\s*(.+)$", bundle_text, re.MULTILINE)
        services = []
        if services_match:
            services = [part.strip() for part in services_match.group(1).split(",") if part.strip()]
        return {
            "instruction": instruction,
            "services": services,
        }

    if tool_name == "extract-error-signatures":
        log_block = _extract_section(bundle_text, "LOG EXCERPTS", ("NOTES",))
        signatures: list[str] = []
        for line in log_block.splitlines():
            normalized = re.sub(r"^\S+\s+", "", line.strip())
            if "ERROR" in normalized or "WARN" in normalized:
                signatures.append(normalized)
        counts = Counter(signatures)
        ranked = [
            {"signature": signature, "count": count}
            for signature, count in counts.most_common()
        ]
        return {
            "instruction": instruction,
            "error_count": sum(item["count"] for item in ranked),
            "top_signatures": ranked[:3],
        }

    if tool_name == "extract-notes":
        notes_block = _extract_section(bundle_text, "NOTES", ())
        notes = [line.lstrip("- ").strip() for line in notes_block.splitlines() if line.strip()]
        return {
            "instruction": instruction,
            "notes": notes,
        }

    if tool_name == "infer-probable-cause":
        services = dependency_results["task_1"]["services"]
        top_signatures = dependency_results["task_2"]["top_signatures"]
        notes = dependency_results["task_3"]["notes"]

        signature_texts = [item["signature"] for item in top_signatures]
        primary_signature = top_signatures[0]["signature"] if top_signatures else "No dominant signature"
        probable_service = "payments" if any("payments" in service for service in services) else (services[0] if services else "unknown")

        has_db_pool_exhaustion = any("database connection pool exhausted" in signature.lower() for signature in signature_texts)
        has_upstream_timeout = any("timed out" in signature.lower() for signature in signature_texts)
        notes_point_to_database = any("cpu" in note.lower() for note in notes)

        if has_db_pool_exhaustion and notes_point_to_database:
            probable_cause = "payments database saturation caused connection pool exhaustion"
            primary_signal = "ERROR payments database connection pool exhausted"
        elif has_upstream_timeout:
            probable_cause = "upstream timeout between api-gateway and payments"
            primary_signal = "ERROR api-gateway upstream payments timed out after 3000ms"
        else:
            probable_cause = "undetermined from local heuristics"
            primary_signal = primary_signature

        contributing_factors = []
        if any("rollback reduced" in note.lower() for note in notes):
            contributing_factors.append("rollback only partially reduced symptoms")
        if any("cpu" in note.lower() for note in notes):
            contributing_factors.append("database CPU saturation during incident")
        if any("deploy" in note.lower() for note in notes):
            contributing_factors.append("incident started after deploy-4812 rollout")

        return {
            "instruction": instruction,
            "primary_service": probable_service,
            "probable_cause": probable_cause,
            "contributing_factors": contributing_factors,
            "primary_signal": primary_signal,
        }

    if tool_name == "build-incident-report":
        services = dependency_results["task_1"]["services"]
        errors = dependency_results["task_2"]
        diagnosis = dependency_results["task_4"]
        top_signatures = errors["top_signatures"]

        lines = [
            "Incident Triage Report",
            "=====================",
            f"Services in scope: {', '.join(services) if services else 'unknown'}",
            f"Primary service: {diagnosis['primary_service']}",
            f"Probable cause: {diagnosis['probable_cause']}",
            f"Primary signal: {diagnosis['primary_signal']}",
            f"Observed log events: {errors['error_count']}",
            "Top signatures:",
        ]
        for item in top_signatures:
            lines.append(f"- {item['signature']} ({item['count']})")
        if diagnosis["contributing_factors"]:
            lines.append("Contributing factors:")
            for factor in diagnosis["contributing_factors"]:
                lines.append(f"- {factor}")
        return {
            "instruction": instruction,
            "report_markdown": "\n".join(lines),
        }

    raise RuntimeError(f"Unknown tool route: {tool_name}")


def build_registry() -> LocalToolRegistry:
    registry = LocalToolRegistry()
    registry.register("extract-services", incident_tool, description="Extract impacted services from the bundle")
    registry.register("extract-error-signatures", incident_tool, description="Extract and rank error signatures")
    registry.register("extract-notes", incident_tool, description="Extract operator notes from the bundle")
    registry.register("infer-probable-cause", incident_tool, description="Infer a likely cause from upstream results")
    registry.register("build-incident-report", incident_tool, description="Build the final markdown report")
    return registry


def build_dag() -> TaskDAG:
    return TaskDAG(
        tasks=[
            SubTask(
                id="task_1",
                description="Extract services",
                instruction="Extract the impacted services from the incident bundle",
                dependencies=[],
                tool_name="extract-services",
                modality="text",
            ),
            SubTask(
                id="task_2",
                description="Extract error signatures",
                instruction="Extract and rank the dominant error signatures from the logs",
                dependencies=[],
                tool_name="extract-error-signatures",
                modality="text",
            ),
            SubTask(
                id="task_3",
                description="Extract notes",
                instruction="Extract the operator notes and deployment clues",
                dependencies=[],
                tool_name="extract-notes",
                modality="text",
            ),
            SubTask(
                id="task_4",
                description="Infer probable cause",
                instruction="Infer the most likely cause from services, log signatures, and notes",
                dependencies=["task_1", "task_2", "task_3"],
                tool_name="infer-probable-cause",
                modality="text",
            ),
            SubTask(
                id="task_5",
                description="Build report",
                instruction="Create a short incident triage report",
                dependencies=["task_1", "task_2", "task_4"],
                tool_name="build-incident-report",
                modality="text",
            ),
        ]
    )


def run_demo(bundle_text: str) -> DAGExecutionReport:
    orchestrator = DAGOrchestrator(build_registry().create_hypervisor())
    return orchestrator.run(build_dag(), context=bundle_text)


def load_bundle_text(input_file: Path | None = None) -> str:
    source = input_file or DEFAULT_INPUT_FILE
    return source.read_text(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ThreadSwarm local-tool incident triage demo.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the incident bundle text file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final report payload as JSON instead of markdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_demo(load_bundle_text(args.input_file))

    print("Execution order:", report.execution_order)
    print()
    if args.json:
        print(json.dumps(report.final_result, indent=2, sort_keys=True))
        return
    print(report.final_result["report_markdown"])


if __name__ == "__main__":
    main()
