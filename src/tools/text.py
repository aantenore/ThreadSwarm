"""Small deterministic text toolkit for local DAG execution."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.engine import LocalToolRegistry


class TextToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: str
    dependency_results: dict[str, Any]
    task_id: str
    instruction: str
    modality: str
    tool_name: str | None
    model_type: str | None
    route_key: str | None
    attempt: int | None


class NormalizeTextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class WordCountOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word_count: int


class ExtractRegexOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str
    matches: list[str]


class CollectJsonOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    dependency_results: dict[str, Any]


def text_tool(context: dict[str, Any], instruction: str, task_id: str, modality: str, route_key: str | None) -> dict[str, Any]:
    """Run one deterministic text tool by route key."""
    payload = str(context["payload"] or "")
    dependency_results = context["dependency_results"]
    tool_name = context.get("tool_name") or route_key or ""

    if tool_name == "normalize-text":
        return {"text": payload.strip().upper()}

    if tool_name == "word-count":
        text = _latest_text(dependency_results) or payload
        return {"word_count": len(re.findall(r"\b\w+\b", text))}

    if tool_name == "extract-regex":
        pattern = instruction.strip()
        return {
            "pattern": pattern,
            "matches": re.findall(pattern, payload),
        }

    if tool_name == "collect-json":
        return {"dependency_results": dependency_results}

    raise RuntimeError(f"Unknown text tool route: {tool_name}")


def build_text_tool_registry() -> LocalToolRegistry:
    registry = LocalToolRegistry()
    registry.register(
        "normalize-text",
        text_tool,
        description="Trim text and convert it to uppercase",
        modalities=("text",),
        input_schema=TextToolInput,
        output_schema=NormalizeTextOutput,
        risk_class="compute_only",
        side_effect_class="none",
        result_size_limit=50_000,
    )
    registry.register(
        "word-count",
        text_tool,
        description="Count words in the latest upstream text result or original payload",
        modalities=("text",),
        input_schema=TextToolInput,
        output_schema=WordCountOutput,
        risk_class="compute_only",
        side_effect_class="none",
        result_size_limit=5_000,
    )
    registry.register(
        "extract-regex",
        text_tool,
        description="Extract regex matches from the original text payload. The task instruction is the regex pattern.",
        modalities=("text",),
        input_schema=TextToolInput,
        output_schema=ExtractRegexOutput,
        risk_class="compute_only",
        side_effect_class="none",
        result_size_limit=50_000,
    )
    registry.register(
        "collect-json",
        text_tool,
        description="Collect dependency results into one JSON object",
        modalities=("text",),
        input_schema=TextToolInput,
        output_schema=CollectJsonOutput,
        risk_class="compute_only",
        side_effect_class="none",
        result_size_limit=100_000,
    )
    return registry


def _latest_text(dependency_results: dict[str, Any]) -> str | None:
    for result in reversed(list(dependency_results.values())):
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            return result["text"]
    return None
