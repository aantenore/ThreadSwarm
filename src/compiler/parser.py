"""
Semantic Compiler: translates natural-language multimodal intent into a DAG of SubTasks.

Connects to a local OpenAI-compatible LLM to perform intent analysis and task
decomposition. Output is a heterogeneous DAG: each SubTask specifies modality,
an optional local tool hint, and an optional model type for the execution runtime.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from src.config import ThreadSwarmConfig

logger = logging.getLogger(__name__)

# Modality hints for heterogeneous DAG tasks.
MODALITY_TEXT = "text"
MODALITY_CODE = "code"
MODALITY_VISION = "vision"
MODALITY_AUDIO = "audio"
MODALITY_MULTIMODAL = "multimodal"
MODALITIES = (MODALITY_TEXT, MODALITY_CODE, MODALITY_VISION, MODALITY_AUDIO, MODALITY_MULTIMODAL)


# ---------------------------------------------------------------------------
# DAG schema (Pydantic)
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """A single micro-task in the execution DAG with routing hints for local tools or models."""

    id: str = Field(..., description="Unique task identifier (e.g. task_1, task_2)")
    description: str = Field(..., description="Human-readable description of the task")
    instruction: str = Field(..., description="Precise directive for the model/actor")
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of task IDs that must complete before this task runs",
    )
    payload_hint: str | None = Field(
        default=None,
        description="Hint for payload (e.g. 'frame', 'chapter', 'log_file', 'audio_chunk')",
    )
    modality: str = Field(
        default=MODALITY_TEXT,
        description="Required modality for this task: text, code, vision, audio, or multimodal",
    )
    tool_name: str | None = Field(
        default=None,
        description="Optional local tool hint (e.g. keyword-extractor, file-search, regex-parser) for CPU-friendly execution",
    )
    model_type: str | None = Field(
        default=None,
        description="Optional model type hint (e.g. gemma-1b, deepseek-r1-1.5b, smolvlm) for actor selection",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries after the first failed execution attempt",
    )
    retry_delay_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Delay before retrying a failed task",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Optional logical timeout for each execution attempt of this task",
    )


class TaskDAG(BaseModel):
    """Directed Acyclic Graph of SubTasks. Tasks are ordered; dependencies refer to prior task IDs."""

    tasks: list[SubTask] = Field(default_factory=list, description="Ordered list of sub-tasks")

    def get_task_by_id(self, task_id: str) -> SubTask | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def validation_error(self) -> str | None:
        """Return a descriptive validation error, or None when the DAG is valid."""
        id_to_index: dict[str, int] = {}
        dependency_graph: dict[str, list[str]] = {}

        for index, task in enumerate(self.tasks):
            if task.id in id_to_index:
                return f"Duplicate task id: {task.id}"
            id_to_index[task.id] = index
            dependency_graph[task.id] = list(task.dependencies)

        for task in self.tasks:
            seen_dependencies: set[str] = set()
            for dep in task.dependencies:
                if dep in seen_dependencies:
                    return f"Task {task.id} repeats dependency {dep}"
                seen_dependencies.add(dep)

                if dep == task.id:
                    return f"Task {task.id} cannot depend on itself"
                if dep not in id_to_index:
                    return f"Task {task.id} references missing dependency {dep}"

        visited: set[str] = set()
        visiting: set[str] = set()
        path: list[str] = []

        def visit(task_id: str) -> str | None:
            if task_id in visiting:
                cycle_start = path.index(task_id)
                cycle = path[cycle_start:] + [task_id]
                return f"Cycle detected: {' -> '.join(cycle)}"
            if task_id in visited:
                return None

            visiting.add(task_id)
            path.append(task_id)
            for dep in dependency_graph[task_id]:
                error = visit(dep)
                if error:
                    return error
            path.pop()
            visiting.remove(task_id)
            visited.add(task_id)
            return None

        for task in self.tasks:
            error = visit(task.id)
            if error:
                return error

        for index, task in enumerate(self.tasks):
            for dep in task.dependencies:
                if id_to_index[dep] >= index:
                    return f"Task {task.id} depends on {dep}, but dependencies must appear earlier in the DAG"

        return None

    def validate_dag(self) -> bool:
        """Ensure no cycles and all dependency IDs exist."""
        return self.validation_error() is None


# ---------------------------------------------------------------------------
# Semantic Compiler
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a Semantic Compiler for a local, CPU-first multimodal runtime. Your job is to take a user's high-level intent (involving text, code, images, audio, video, or mixed inputs) and decompose it into a strict Directed Acyclic Graph (DAG) of micro-tasks that can be executed in parallel on CPU by local tools or lightweight specialized workers.

Rules:
- Output ONLY a valid JSON array of task objects. No markdown, no explanation outside the JSON.
- Each task must have: "id" (string, e.g. "task_1"), "description" (string), "instruction" (string, precise directive), "dependencies" (array of task ids that must complete first). Optionally: "payload_hint" (string), "modality" (one of: text, code, vision, audio, multimodal), "tool_name" (string, for local CPU-friendly tools), "model_type" (string, e.g. "gemma-1b", "deepseek-r1-1.5b", "smolvlm" for actor selection).
- Optional reliability fields: "retry_count" (integer >= 0), "retry_delay_seconds" (number >= 0), and "timeout_seconds" (number > 0).
- IDs must be unique. Dependencies must reference earlier task IDs. No cycles.
- Keep tasks atomic. Prefer local CPU-friendly tools when possible, and use "tool_name" to name them. Set "modality" so the orchestrator can assign the right executor: use "text" for summarization/QA, "code" for logic/code analysis, "vision" for images/frames, "audio" for audio, "multimodal" when input mixes modalities.
- First tasks often prepare data (e.g. OCR logs, split document, extract frames). Later tasks depend on them and do the actual analysis or synthesis.
"""


class SemanticCompiler:
    """
    Compiler that calls a local LLM to turn natural-language (multimodal) intent
    into a heterogeneous DAG of SubTasks. Uses an OpenAI-compatible API (e.g. Ollama).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.2",
        system_prompt: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout = timeout

    @classmethod
    def from_config(
        cls,
        config: ThreadSwarmConfig,
        *,
        system_prompt: str | None = None,
    ) -> "SemanticCompiler":
        """Create a compiler from typed runtime configuration."""
        return cls(**config.compiler_kwargs(), system_prompt=system_prompt)

    def compile(self, user_prompt: str) -> TaskDAG:
        """
        Analyze the user's intent and return a validated heterogeneous TaskDAG.

        :param user_prompt: Natural language description (can reference code, docs, images, audio).
        :return: TaskDAG of SubTasks with modality and optional tool/model routing hints.
        :raises SemanticCompilationError: On API or parsing failure.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        with httpx.Client(timeout=self.timeout) as client:
            url = f"{self.base_url}/chat/completions"
            try:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise SemanticCompilationError(f"LLM API request failed: {e}") from e

            data = resp.json()
            choices = data.get("choices")
            if not choices:
                raise SemanticCompilationError("LLM returned no choices")

            content = choices[0].get("message", {}).get("content") or ""
            if not content.strip():
                raise SemanticCompilationError("LLM returned empty content")

        dag = self._parse_llm_output(content)
        validation_error = dag.validation_error()
        if validation_error:
            raise SemanticCompilationError(f"DAG validation failed: {validation_error}")
        return dag

    def _parse_llm_output(self, content: str) -> TaskDAG:
        """Extract a JSON array from LLM output and parse into TaskDAG."""
        return parse_task_dag_json(content)


class SemanticCompilationError(Exception):
    """Raised when the Semantic Compiler fails to produce a valid DAG."""

    pass


def parse_task_dag_json(content: str) -> TaskDAG:
    """Parse a TaskDAG from JSON text, accepting arrays, objects, and JSON fences."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content)
    content = content.strip()

    try:
        raw = json.loads(content)
    except json.JSONDecodeError as e:
        raise SemanticCompilationError(f"Invalid JSON DAG: {e}") from e

    try:
        if isinstance(raw, list):
            tasks = [SubTask.model_validate(item) for item in raw]
            return TaskDAG(tasks=tasks)
        if isinstance(raw, dict) and "tasks" in raw:
            return TaskDAG.model_validate(raw)
    except ValidationError as e:
        raise SemanticCompilationError(f"Invalid DAG schema: {e}") from e
    raise SemanticCompilationError("DAG JSON must be an array of tasks or an object with a 'tasks' key")
