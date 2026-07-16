"""Tests for the Semantic Compiler (parser and DAG)."""

import pytest

from src.compiler import parser as compiler_parser
from threadswarm.compiler.parser import (
    SemanticCompiler,
    SemanticCompilationError,
    SubTask,
    TaskDAG,
)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_subtask_and_dag_validation():
    t1 = SubTask(id="task_1", description="Split frames", instruction="Split video into frames", dependencies=[])
    t2 = SubTask(id="task_2", description="Analyze", instruction="Classify frame", dependencies=["task_1"])
    dag = TaskDAG(tasks=[t1, t2])
    assert dag.validate_dag()
    assert dag.validation_error() is None
    assert dag.get_task_by_id("task_1") is t1
    assert dag.get_task_by_id("task_2") is t2


def test_parse_llm_output_array():
    compiler = SemanticCompiler(base_url="http://localhost:11434/v1", model="dummy")
    raw_json = '''[
        {"id": "task_1", "description": "Split", "instruction": "Split input", "dependencies": []},
        {"id": "task_2", "description": "Run", "instruction": "Process frame", "dependencies": ["task_1"]}
    ]'''
    dag = compiler._parse_llm_output(raw_json)
    assert len(dag.tasks) == 2
    assert dag.tasks[0].id == "task_1"
    assert dag.tasks[1].dependencies == ["task_1"]


def test_parse_llm_output_with_markdown():
    compiler = SemanticCompiler(base_url="http://localhost:11434/v1", model="dummy")
    raw = '```json\n[{"id": "t1", "description": "D", "instruction": "I", "dependencies": []}]\n```'
    dag = compiler._parse_llm_output(raw)
    assert len(dag.tasks) == 1
    assert dag.tasks[0].id == "t1"


def test_parse_llm_output_with_modality():
    """Parse DAG with modality, tool_name, and model_type hints for heterogeneous executors."""
    compiler = SemanticCompiler(base_url="http://localhost:11434/v1", model="dummy")
    raw = '''[
        {"id": "t1", "description": "OCR", "instruction": "Extract text", "dependencies": [], "modality": "vision", "tool_name": "ocr-local", "model_type": "smolvlm"},
        {"id": "t2", "description": "Summarize", "instruction": "Summarize log", "dependencies": ["t1"], "modality": "text", "tool_name": "keyword-extractor", "model_type": "gemma-1b"}
    ]'''
    dag = compiler._parse_llm_output(raw)
    assert len(dag.tasks) == 2
    assert dag.tasks[0].modality == "vision" and dag.tasks[0].tool_name == "ocr-local" and dag.tasks[0].model_type == "smolvlm"
    assert dag.tasks[1].modality == "text" and dag.tasks[1].tool_name == "keyword-extractor" and dag.tasks[1].model_type == "gemma-1b"


def test_parse_llm_output_with_retry_policy():
    compiler = SemanticCompiler(base_url="http://localhost:11434/v1", model="dummy")
    raw = '''[
        {"id": "t1", "description": "Fetch", "instruction": "Fetch data", "dependencies": [], "tool_name": "fetch-local", "retry_count": 2, "retry_delay_seconds": 0.25, "timeout_seconds": 5.0}
    ]'''
    dag = compiler._parse_llm_output(raw)
    assert dag.tasks[0].retry_count == 2
    assert dag.tasks[0].retry_delay_seconds == 0.25
    assert dag.tasks[0].timeout_seconds == 5.0


def test_semantic_compiler_sends_catalog_only_in_system_message(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            calls.append({"url": url, "payload": json, "timeout": self.timeout})
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '[{"id":"task_1","description":"Normalize",'
                                    '"instruction":"Normalize input","dependencies":[],'
                                    '"tool_name":"normalize-text"}]'
                                )
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr(compiler_parser.httpx, "Client", FakeClient)
    compiler = SemanticCompiler(base_url="http://local-compiler/v1", model="planner", timeout=4.0)
    catalog = {
        "schema_version": 1,
        "catalog_digest": "abc",
        "tools": [{"name": "normalize-text", "description": "Normalize text"}],
    }

    dag = compiler.compile("Prepare my payload", capability_catalog=catalog)

    assert dag.tasks[0].tool_name == "normalize-text"
    assert calls[0]["url"] == "http://local-compiler/v1/chat/completions"
    assert calls[0]["timeout"] == 4.0
    messages = calls[0]["payload"]["messages"]
    assert '"catalog_digest":"abc"' in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Prepare my payload"}


def test_semantic_compiler_rejects_oversized_response_before_dag_validation(monkeypatch):
    class FakeClient:
        def __init__(self, *, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '[{"id":"task_1","description":"Normalize",'
                                    '"instruction":"Normalize input","dependencies":[],"tool_name":"normalize-text"}]'
                                )
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr(compiler_parser.httpx, "Client", FakeClient)
    compiler = SemanticCompiler(base_url="http://local-compiler/v1", model="planner")
    catalog = {
        "compiler_limits": {"max_response_chars": 8},
        "plan_limits": {"max_tasks": 1},
        "tools": [{"name": "normalize-text"}],
    }

    with pytest.raises(SemanticCompilationError, match="LLM output has"):
        compiler.compile("Prepare my payload", capability_catalog=catalog)


def test_semantic_compiler_rejects_task_count_before_model_validation():
    compiler = SemanticCompiler(base_url="http://localhost:11434/v1", model="dummy")
    raw = "[{}, {}]"

    with pytest.raises(SemanticCompilationError, match="allows at most 1"):
        compiler._parse_llm_output(raw, max_tasks=1)


def test_parse_llm_output_invalid_raises():
    compiler = SemanticCompiler(base_url="http://localhost:11434/v1", model="dummy")
    with pytest.raises(SemanticCompilationError):
        compiler._parse_llm_output("not json at all")
    with pytest.raises(SemanticCompilationError):
        compiler._parse_llm_output('{"wrong": "structure"}')


def test_validate_dag_rejects_missing_dependency():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Analyze", instruction="Run analysis", dependencies=["task_999"]),
        ]
    )

    assert dag.validate_dag() is False
    assert dag.validation_error() == "Task task_1 references missing dependency task_999"


def test_validate_dag_rejects_duplicate_task_ids():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="First", instruction="Run first", dependencies=[]),
            SubTask(id="task_1", description="Second", instruction="Run second", dependencies=[]),
        ]
    )

    assert dag.validate_dag() is False
    assert dag.validation_error() == "Duplicate task id: task_1"


def test_validate_dag_rejects_cycles():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="First", instruction="Run first", dependencies=["task_2"]),
            SubTask(id="task_2", description="Second", instruction="Run second", dependencies=["task_1"]),
        ]
    )

    assert dag.validate_dag() is False
    assert dag.validation_error() == "Cycle detected: task_1 -> task_2 -> task_1"


def test_validate_dag_rejects_future_dependencies():
    dag = TaskDAG(
        tasks=[
            SubTask(id="task_1", description="Merge", instruction="Merge results", dependencies=["task_2"]),
            SubTask(id="task_2", description="Prepare", instruction="Prepare data", dependencies=[]),
        ]
    )

    assert dag.validate_dag() is False
    assert dag.validation_error() == "Task task_1 depends on task_2, but dependencies must appear earlier in the DAG"
