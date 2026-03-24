# ThreadSwarm

ThreadSwarm is a CPU-first runtime for breaking complex work into a DAG of small tasks that can run on almost any PC.

The key idea is simple:
- a semantic compiler turns a high-level request into a `TaskDAG`
- large shared context lives once in RAM
- each task is routed to a cheap local tool or, when needed, a specialized worker
- an orchestrator respects dependencies and reduces leaf results into a final output

This repo is aimed at practical portability, not GPU-heavy orchestration. The happy path is: use local tools first, use models only when they actually add value.

## Why

Complex tasks often get pushed into one giant model, which creates three problems:
- hardware cost
- latency
- wasted intelligence for work that could be done by simpler tools

ThreadSwarm takes the opposite approach: decompose work into tiny steps and run those steps with the cheapest executor that can do the job on CPU hardware.

## Architecture

```mermaid
flowchart LR
    A["User intent"] --> B["SemanticCompiler"]
    B --> C["TaskDAG"]
    D["Shared context in RAM"] --> E["DAGOrchestrator"]
    C --> E
    E --> F["Local tool or worker pool"]
    F --> G["Task results"]
    G --> E
    E --> H["Reduced final result"]
```

The runtime is split into a few small pieces:
- `src/compiler/`: planning only
- `src/engine/shared_memory.py`: zero-copy context sharing for `ndarray`, `str`, and `bytes`
- `src/engine/actor_pool.py`: process-based execution pool
- `src/engine/orchestrator.py`: dependency-aware DAG execution
- `src/engine/tool_registry.py`: registration of local CPU-friendly tools

## Current Capabilities

What the repo can do today:
- validate DAGs with clear errors for duplicate IDs, missing dependencies, future dependencies, self-dependencies, and cycles
- store large context once in shared memory and reconstruct it in worker processes
- route tasks by `tool_name` or `model_type`
- execute a DAG end-to-end with dependency tracking
- block downstream tasks after upstream failures
- reduce leaf task results into a final result

What is still intentionally lightweight:
- real model adapters in `src/models/`
- retries, persistence, and richer scheduling policies
- a polished CLI or end-user app layer

## Task Schema

`SubTask` supports:
- `id`
- `description`
- `instruction`
- `dependencies`
- `payload_hint`
- `modality`
- `tool_name`
- `model_type`

Use `tool_name` when a task should run on a local CPU-friendly executor.
Use `model_type` when a specialized worker or model is actually required.

## Execution Model

1. `SemanticCompiler.compile(...)` produces a `TaskDAG`.
2. `ContextMemoryManager.load_and_share(...)` stores the large payload once in RAM.
3. `DAGOrchestrator.run(...)` submits every ready task.
4. `ActorHypervisor` routes each task to the right pool by `tool_name` or `model_type`.
5. Workers receive:
   - the shared payload
   - small `dependency_results`
   - task metadata like `modality`, `tool_name`, and `model_type`
6. The orchestrator waits for completions, unlocks dependents, and reduces leaf outputs.

## Practical Guide

The most useful doc right now is the hands-on guide for local tool pipelines:

[docs/local-tool-pipelines.md](docs/local-tool-pipelines.md)

It covers:
- when to use `tool_name`
- how to register local tools
- how to build a small DAG
- how to run it through the orchestrator
- what the worker context looks like

## Repository Structure

```text
docs/
  local-tool-pipelines.md   Practical guide for local tool DAGs
  rfcs/                     RFC folder for architectural proposals
src/
  compiler/                 Semantic compiler and DAG schema
  engine/                   Shared memory, actor pool, orchestrator, tool registry
  models/                   Optional model adapters
tests/                      Compiler and engine tests
```

## Constraints

- CPU-first execution
- `multiprocessing`, not threads, for inference/execution workers
- no large payload transfer over normal queues
- Windows-compatible picklable worker hooks

## Install

Requirements:
- Python 3.10+

Dependencies live in:
- `requirements.txt`
- `pyproject.toml`

Typical setup:

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

Typical test run:

```bash
pytest -q
```

## Contribution Notes

- Keep planning concerns in `src/compiler`
- Keep execution concerns in `src/engine`
- Prefer local tools when they can solve the task well
- Add model-backed executors only where they materially improve outcomes
- Write RFCs in `docs/rfcs/` for meaningful architectural changes

## Status

The core runtime is implemented and tested. The next useful layer is a richer library of local tools and a stronger story for mixed tool-plus-model workflows.
