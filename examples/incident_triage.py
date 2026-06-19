"""Compatibility wrapper for the packaged incident triage demo."""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threadswarm.demos.incident_triage import (
    build_dag,
    build_registry,
    incident_tool,
    load_bundle_text,
    main,
    run_demo,
)

__all__ = [
    "build_dag",
    "build_registry",
    "incident_tool",
    "load_bundle_text",
    "main",
    "run_demo",
]


if __name__ == "__main__":
    main()
