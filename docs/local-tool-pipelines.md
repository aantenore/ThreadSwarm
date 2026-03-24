# Local Tool Pipelines

This guide shows how to use ThreadSwarm for the main goal of the project:

build complex workflows out of small tasks that can run on almost any PC by
preferring local CPU-friendly tools over heavyweight model calls.

## Mental Model

Use ThreadSwarm like this:

1. Break the job into tiny tasks.
2. Give each task a `tool_name` when a normal local executor can do the work.
3. Keep the large shared payload in RAM once.
4. Let the orchestrator unlock tasks as dependencies finish.
5. Use `model_type` only for the tasks that truly need a specialized worker.

In practice:
- `tool_name` is the default path
- `model_type` is the escape hatch

## Core Pieces

- `SubTask`: one task in the DAG
- `TaskDAG`: ordered task list with dependencies
- `LocalToolRegistry`: maps local tool names to worker callables
- `DAGOrchestrator`: runs the DAG
- `ContextMemoryManager`: stores large payloads once in shared memory

## Worker Contract

Local tool executors use the same worker hook signature as the actor pool:

```python
def run_tool(context, instruction, task_id, modality, model_type):
    ...
```

For orchestrated DAG runs:
- `context` is a dictionary with:
  - `payload`
  - `dependency_results`
  - `task_id`
  - `modality`
  - `tool_name`
  - `model_type`
- the fifth positional argument still carries the route key used by the pool
  for backward compatibility

If the task was routed by `tool_name`, you should usually read the executor name
from `context["tool_name"]`.

## End-to-End Example

The example below builds a tiny text workflow:

- normalize the input
- summarize the normalized text
- finalize a report

```python
from src.compiler.parser import SubTask, TaskDAG
from src.engine import DAGOrchestrator, LocalToolRegistry


def text_tool(context, instruction, task_id, modality, route_key):
    payload = context["payload"]
    dependency_results = context["dependency_results"]
    tool_name = context["tool_name"]

    if tool_name == "normalize-text":
        return {
            "normalized": payload.strip().upper(),
            "instruction": instruction,
        }

    if tool_name == "summarize-text":
        normalized = dependency_results["task_1"]["normalized"]
        return {
            "summary": f"{normalized} -> {instruction}",
        }

    if tool_name == "finalize-report":
        summary = dependency_results["task_2"]["summary"]
        return {
            "final": summary.lower(),
        }

    raise RuntimeError(f"Unknown tool: {tool_name}")


registry = LocalToolRegistry()
registry.register("normalize-text", text_tool, description="Normalize raw text")
registry.register("summarize-text", text_tool, description="Summarize normalized text")
registry.register("finalize-report", text_tool, description="Finalize the answer")

dag = TaskDAG(
    tasks=[
        SubTask(
            id="task_1",
            description="Normalize input",
            instruction="Normalize the raw input text",
            dependencies=[],
            tool_name="normalize-text",
        ),
        SubTask(
            id="task_2",
            description="Summarize normalized text",
            instruction="Summarize the normalized text",
            dependencies=["task_1"],
            tool_name="summarize-text",
        ),
        SubTask(
            id="task_3",
            description="Finalize report",
            instruction="Produce the final report",
            dependencies=["task_2"],
            tool_name="finalize-report",
        ),
    ]
)

orchestrator = DAGOrchestrator(registry.create_hypervisor())
report = orchestrator.run(dag, context="  hello threadswarm  ")

print(report.execution_order)
print(report.final_result)
```

Expected shape of the result:

```python
["task_1", "task_2", "task_3"]
{"final": "hello threadswarm -> summarize the normalized text"}
```

## When To Use tool_name

Use `tool_name` for tasks like:
- regex extraction
- file search
- CSV parsing
- JSON validation
- OCR wrappers around local binaries
- simple ranking or rule-based filtering
- deterministic transforms

These are ideal because they:
- run cheaply on CPU
- are portable
- are easier to test than model prompts
- fail more predictably

## When To Use model_type

Use `model_type` only when the task actually needs a specialized worker, for example:
- open-ended summarization where rules are not enough
- semantic comparison across messy text
- vision reasoning beyond basic OCR
- audio interpretation

The principle is:

cheap deterministic tools first, specialized models second.

## Dependency Results

Every downstream task receives a small `dependency_results` mapping.

Example:

```python
{
    "task_1": {"normalized": "HELLO THREADSWARM"},
    "task_2": {"summary": "HELLO THREADSWARM -> Summarize the normalized text"},
}
```

Use this for small intermediate artifacts.
Do not put large blobs there.
Large shared inputs should stay in shared memory as the main `payload`.

## Design Rules

- Keep tasks atomic.
- Prefer one clear responsibility per task.
- Keep dependencies explicit.
- Store large shared inputs once, not per task.
- Pass only small derived outputs through `dependency_results`.
- Register tools with stable names.
- Keep worker functions module-level so they stay picklable on Windows.

## Suggested Workflow For New Features

1. Start with a hand-written `TaskDAG`.
2. Implement the local tools and register them.
3. Run the DAG through the orchestrator.
4. Add tests for success and failure paths.
5. Only then teach the semantic compiler to emit the same `tool_name` values.

That sequence keeps the runtime testable before any prompt-dependent planning is introduced.
