# Quickstart

This quickstart gets you from clone to a working local-tool DAG in a few minutes.

## 1. Install

From the repo root:

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

## 2. Run The Demo

Run the incident-triage example:

```bash
python -m examples.incident_triage
```

What it does:
- loads a sample incident bundle from `examples/data/incident_bundle.txt`
- runs a DAG of local CPU-friendly tools
- extracts services, log signatures, and notes
- infers a probable cause
- builds a final incident report

Because the first three tasks are independent, their relative execution order may vary.
A typical output looks like this:

```text
Execution order: ['task_2', 'task_1', 'task_3', 'task_4', 'task_5']

Incident Triage Report
=====================
Services in scope: api-gateway, payments, catalog
Primary service: payments
Probable cause: payments database saturation caused connection pool exhaustion
Primary signal: ERROR payments database connection pool exhausted
...
```

## 3. Inspect The DAG

The demo DAG lives in:

- `examples/incident_triage.py`

Look at:
- `build_dag()` for the task graph
- `build_registry()` for local tool registration
- `incident_tool(...)` for the actual CPU-friendly executors
- `run_demo(...)` for the orchestrated run

## 4. Try Your Own Input

You can point the demo at another text bundle:

```bash
python -m examples.incident_triage --input-file path/to/your_bundle.txt
```

Or print the final payload as JSON:

```bash
python -m examples.incident_triage --json
```

## 5. Use The Pattern For Your Own Workflow

The basic recipe is:

1. Define a `TaskDAG`.
2. Register local tools in `LocalToolRegistry`.
3. Run the DAG with `DAGOrchestrator`.
4. Keep large input in shared memory.
5. Pass only small intermediate results through `dependency_results`.

## Next Reads

- `docs/local-tool-pipelines.md` for the deeper design guide
- `tests/test_engine.py` for compact execution examples
