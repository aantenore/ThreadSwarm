"""Tests for the Semantic Compiler (parser and DAG)."""

import pytest

from src.compiler.parser import (
    SemanticCompiler,
    SemanticCompilationError,
    SubTask,
    TaskDAG,
)


def test_subtask_and_dag_validation():
    t1 = SubTask(id="task_1", description="Split frames", instruction="Split video into frames", dependencies=[])
    t2 = SubTask(id="task_2", description="Analyze", instruction="Classify frame", dependencies=["task_1"])
    dag = TaskDAG(tasks=[t1, t2])
    assert dag.validate_dag()
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


def test_parse_llm_output_invalid_raises():
    compiler = SemanticCompiler(base_url="http://localhost:11434/v1", model="dummy")
    with pytest.raises(SemanticCompilationError):
        compiler._parse_llm_output("not json at all")
    with pytest.raises(SemanticCompilationError):
        compiler._parse_llm_output('{"wrong": "structure"}')
