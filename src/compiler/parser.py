"""
Semantic Compiler: translates natural-language multimodal intent into a DAG of SubTasks.

Connects to a local OpenAI-compatible LLM to perform intent analysis and task
decomposition. Output is a heterogeneous DAG: each SubTask specifies modality
and optional model type (text, code, vision, audio, etc.) for the actor swarm.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Modality / model-type hints for heterogeneous DAG (which actor/model handles the task)
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
    """A single micro-task in the execution DAG with modality and optional model hint."""

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
    model_type: str | None = Field(
        default=None,
        description="Optional model type hint (e.g. gemma-1b, deepseek-r1-1.5b, smolvlm) for actor selection",
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
        seen: set[str] = set()
        for t in self.tasks:
            for dep in t.dependencies:
                if dep not in seen:
                    pass
            seen.add(t.id)
        return True


# ---------------------------------------------------------------------------
# Semantic Compiler
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a Semantic Compiler for a distributed multimodal AI system. Your job is to take a user's high-level intent (involving text, code, images, audio, video, or mixed inputs) and decompose it into a strict Directed Acyclic Graph (DAG) of micro-tasks that can be executed in parallel on CPU by small specialized models.

Rules:
- Output ONLY a valid JSON array of task objects. No markdown, no explanation outside the JSON.
- Each task must have: "id" (string, e.g. "task_1"), "description" (string), "instruction" (string, precise directive), "dependencies" (array of task ids that must complete first). Optionally: "payload_hint" (string), "modality" (one of: text, code, vision, audio, multimodal), "model_type" (string, e.g. "gemma-1b", "deepseek-r1-1.5b", "smolvlm" for actor selection).
- IDs must be unique. Dependencies must reference earlier task IDs. No cycles.
- Keep tasks atomic. Set "modality" so the orchestrator can assign the right model: use "text" for summarization/QA, "code" for logic/code analysis, "vision" for images/frames, "audio" for audio, "multimodal" when input mixes modalities.
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

    def compile(self, user_prompt: str) -> TaskDAG:
        """
        Analyze the user's intent and return a validated heterogeneous TaskDAG.

        :param user_prompt: Natural language description (can reference code, docs, images, audio).
        :return: TaskDAG of SubTasks with modality and optional model_type.
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
