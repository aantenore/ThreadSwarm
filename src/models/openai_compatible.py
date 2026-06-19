"""OpenAI-compatible model worker adapter for DAG tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from src.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_TIMEOUT, ThreadSwarmConfig

DEFAULT_WORKER_SYSTEM_PROMPT = (
    "You are a ThreadSwarm task worker. Follow the task instruction, use dependency results as evidence, "
    "and return a concise result for downstream DAG tasks."
)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleWorker:
    """Picklable worker hook for OpenAI-compatible chat completion APIs."""

    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    timeout: float = DEFAULT_LLM_TIMEOUT
    system_prompt: str = DEFAULT_WORKER_SYSTEM_PROMPT
    max_context_chars: int = 12_000
    temperature: float | None = None
    headers: Mapping[str, str] | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url cannot be empty")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than 0")

    @classmethod
    def from_config(cls, config: ThreadSwarmConfig, **overrides: Any) -> "OpenAICompatibleWorker":
        """Create a model worker from typed ThreadSwarm runtime configuration."""
        base_url = overrides.pop("base_url", config.llm_base_url)
        model = overrides.pop("model", config.llm_model)
        timeout = overrides.pop("timeout", config.llm_timeout)
        return cls(base_url=base_url, model=model, timeout=timeout, **overrides)

    def to_worker_config(self, *, route_key: str | None = None, num_workers: int = 1) -> dict[str, Any]:
        """Return an ActorHypervisor worker config for this model adapter."""
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than 0")
        return {
            "model_type": route_key or self.model,
            "num_workers": num_workers,
            "run_inference_hook": self,
        }

    def __call__(
        self,
        context: Any,
        instruction: str,
        task_id: str,
        modality: str,
        route_key: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _build_messages(
                context=context,
                instruction=instruction,
                task_id=task_id,
                modality=modality,
                route_key=route_key,
                system_prompt=self.system_prompt,
                max_context_chars=self.max_context_chars,
            ),
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(dict(self.extra_body))

        with httpx.Client(timeout=self.timeout, headers=dict(self.headers or {})) as client:
            response = client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

        choice = _first_choice(data)
        return {
            "task_id": task_id,
            "model": self.model,
            "route_key": route_key,
            "modality": modality,
            "content": _message_content_to_text(choice.get("message", {}).get("content")),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
        }


def build_openai_compatible_worker(
    config: ThreadSwarmConfig | None = None,
    **overrides: Any,
) -> OpenAICompatibleWorker:
    """Factory for callers that prefer functions over direct class construction."""
    return OpenAICompatibleWorker.from_config(config or ThreadSwarmConfig.from_env(), **overrides)


def _build_messages(
    *,
    context: Any,
    instruction: str,
    task_id: str,
    modality: str,
    route_key: str | None,
    system_prompt: str,
    max_context_chars: int,
) -> list[dict[str, str]]:
    task_payload = {
        "task_id": task_id,
        "modality": modality,
        "route_key": route_key,
        "instruction": instruction,
        "context": _context_preview(context, max_context_chars=max_context_chars),
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(task_payload, ensure_ascii=True, indent=2, sort_keys=True)},
    ]


def _context_preview(context: Any, *, max_context_chars: int) -> dict[str, Any]:
    if isinstance(context, dict):
        return {
            "payload": _preview_value(context.get("payload"), max_context_chars=max_context_chars),
            "dependency_results": _preview_json(context.get("dependency_results", {}), max_context_chars=max_context_chars),
            "tool_name": context.get("tool_name"),
            "model_type": context.get("model_type"),
            "attempt": context.get("attempt"),
        }
    return {"payload": _preview_value(context, max_context_chars=max_context_chars)}


def _preview_value(value: Any, *, max_context_chars: int) -> dict[str, Any]:
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, str):
        return {
            "type": "text",
            "truncated": len(value) > max_context_chars,
            "value": value[:max_context_chars],
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        preview = data[:max_context_chars].decode("utf-8", errors="replace")
        return {"type": "bytes", "size": len(data), "truncated": len(data) > max_context_chars, "preview": preview}
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(getattr(value, "shape", ())),
            "dtype": str(getattr(value, "dtype", "")),
        }
    return _preview_json(value, max_context_chars=max_context_chars)


def _preview_json(value: Any, *, max_context_chars: int) -> dict[str, Any]:
    serialized = json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True)
    return {
        "type": "json",
        "truncated": len(serialized) > max_context_chars,
        "value": serialized[:max_context_chars],
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return repr(value)


def _first_choice(data: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI-compatible response did not include choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("OpenAI-compatible response choice is not an object")
    return choice


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)
