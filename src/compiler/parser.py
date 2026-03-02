"""
Semantic Compiler: translates natural-language intent into a DAG of SubTasks.

Connects to a local OpenAI-compatible LLM (e.g. Ollama, LM Studio) to perform
intent analysis and task decomposition. Output is a strict JSON array of
SubTask objects (Pydantic-validated) representing a Directed Acyclic Graph.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DAG schema (Pydantic)
# ---------------------------------------------------------------------------


class SubTask(BaseModel):
    """A single micro-task in the execution DAG."""

    id: str = Field(..., description="Unique task identifier (e.g. task_1, task_2)")
    description: str = Field(..., description="Human-readable description of the task")
    instruction: str = Field(..., description="Precise directive for the model/actor")
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of task IDs that must complete before this task runs",
    )
    payload_hint: str | None = Field(
        default=None,
        description="Optional hint for payload (e.g. 'frame', 'region', 'text')",
    )


class TaskDAG(BaseModel):
    """Directed Acyclic Graph of SubTasks. Tasks are ordered; dependencies refer to prior task IDs."""

    tasks: list[SubTask] = Field(default_factory=list, description="Ordered list of sub-tasks")

    def get_task_by_id(self, task_id: str) -> SubTask | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def validate_dag(self) -> bool:
        """Ensure no cycles and all dependency IDs exist."""
        ids = {t.id for t in self.tasks}
        for t in self.tasks:
            for dep in t.dependencies:
                if dep not in ids:
                    return False
        # Simple cycle check: toposort
        seen: set[str] = set()
        for t in self.tasks:
            for dep in t.dependencies:
                if dep not in seen:
                    pass  # dependency is earlier in list
            seen.add(t.id)
        return True


# ---------------------------------------------------------------------------
# Semantic Compiler
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a Semantic Compiler for a distributed Vision AI system. Your job is to take a user's high-level intent and decompose it into a strict Directed Acyclic Graph (DAG) of micro-tasks that can be executed in parallel on CPU by small vision/language models.

Rules:
- Output ONLY a valid JSON array of task objects. No markdown, no explanation outside the JSON.
- Each task must have: "id" (string, e.g. "task_1"), "description" (string), "instruction" (string, precise directive for the model), "dependencies" (array of task ids that must complete first), and optionally "payload_hint" (string).
- IDs must be unique. Dependencies must reference earlier task IDs. No cycles.
- Keep tasks atomic and small (e.g. "Extract text from this region", "Classify if person wears hard hat in this crop").
- The first tasks often handle splitting input (e.g. split video into frames, split image into regions). Later tasks depend on them and do the actual analysis.
"""


class SemanticCompiler:
    """
    Compiler that calls a local LLM to turn natural-language intent into a
    DAG of SubTasks. Uses an OpenAI-compatible API (e.g. Ollama at localhost:11434).
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

    def compile(self, user_prompt: str) -> TaskDAG:
        """
        Analyze the user's intent and return a validated TaskDAG.

        :param user_prompt: Natural language description of what the user wants.
        :return: TaskDAG of SubTasks.
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
        if not dag.validate_dag():
            raise SemanticCompilationError("DAG validation failed (missing deps or cycle)")
        return dag

    def _parse_llm_output(self, content: str) -> TaskDAG:
        """Extract a JSON array from LLM output and parse into TaskDAG."""
        content = content.strip()
        # Strip markdown code block if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```\s*$", "", content)
        content = content.strip()

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as e:
            raise SemanticCompilationError(f"Invalid JSON from LLM: {e}") from e

        if isinstance(raw, list):
            tasks = [SubTask.model_validate(item) for item in raw]
            return TaskDAG(tasks=tasks)
        if isinstance(raw, dict) and "tasks" in raw:
            return TaskDAG.model_validate(raw)
        raise SemanticCompilationError("LLM output must be a JSON array of tasks or object with 'tasks' key")


class SemanticCompilationError(Exception):
    """Raised when the Semantic Compiler fails to produce a valid DAG."""

    pass
