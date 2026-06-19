import json

import pytest

from src.config import ThreadSwarmConfig
from src.models import OpenAICompatibleWorker, build_openai_compatible_worker
from src.models import openai_compatible


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_openai_compatible_worker_calls_chat_completions(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *, timeout, headers):
            self.timeout = timeout
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            calls.append({"url": url, "json": json, "timeout": self.timeout, "headers": self.headers})
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "model answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 12},
                }
            )

    monkeypatch.setattr(openai_compatible.httpx, "Client", FakeClient)
    worker = OpenAICompatibleWorker(
        base_url="http://local-llm/v1/",
        model="small-model",
        timeout=7.5,
        headers={"Authorization": "Bearer test"},
        temperature=0.1,
        max_context_chars=200,
    )

    result = worker(
        {
            "payload": "raw payload",
            "dependency_results": {"task_1": {"summary": "upstream"}},
            "model_type": "small-model",
            "attempt": 2,
        },
        "Summarize the upstream result",
        "task_2",
        "text",
        "small-model",
    )

    assert result == {
        "task_id": "task_2",
        "model": "small-model",
        "route_key": "small-model",
        "modality": "text",
        "content": "model answer",
        "finish_reason": "stop",
        "usage": {"total_tokens": 12},
    }
    assert calls[0]["url"] == "http://local-llm/v1/chat/completions"
    assert calls[0]["timeout"] == 7.5
    assert calls[0]["headers"] == {"Authorization": "Bearer test"}
    request_payload = calls[0]["json"]
    assert request_payload["model"] == "small-model"
    assert request_payload["stream"] is False
    assert request_payload["temperature"] == 0.1
    user_content = json.loads(request_payload["messages"][1]["content"])
    assert user_content["instruction"] == "Summarize the upstream result"
    assert user_content["context"]["payload"]["value"] == "raw payload"
    assert '"summary": "upstream"' in user_content["context"]["dependency_results"]["value"]
    assert user_content["context"]["attempt"] == 2


def test_openai_compatible_worker_factory_and_worker_config():
    config = ThreadSwarmConfig(
        llm_base_url="http://configured/v1",
        llm_model="configured-model",
        llm_timeout=3.0,
    )

    worker = build_openai_compatible_worker(config, max_context_chars=50)
    worker_config = worker.to_worker_config(route_key="summarizer", num_workers=2)

    assert worker.base_url == "http://configured/v1"
    assert worker.model == "configured-model"
    assert worker.timeout == 3.0
    assert worker.max_context_chars == 50
    assert worker_config == {
        "model_type": "summarizer",
        "num_workers": 2,
        "run_inference_hook": worker,
    }


def test_openai_compatible_worker_rejects_response_without_choices(monkeypatch):
    class FakeClient:
        def __init__(self, *, timeout, headers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            return _FakeResponse({"choices": []})

    monkeypatch.setattr(openai_compatible.httpx, "Client", FakeClient)
    worker = OpenAICompatibleWorker()

    with pytest.raises(ValueError, match="did not include choices"):
        worker("payload", "Run task", "task_1", "text", None)
