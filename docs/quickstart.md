# Quickstart

This quickstart gets you from clone to a working local-tool DAG in a few minutes.

## 1. Install

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The base install supports text, bytes, NumPy arrays, and `.npy` image payloads.
Install the optional vision dependency only when common image formats are needed:

```bash
python -m pip install -e ".[vision]"
```

## 2. Run The Demo

Run the packaged incident-triage demo:

```bash
threadswarm demo incident-triage
```

What it does:
- loads a packaged sample incident bundle
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

- `src/demos/incident_triage.py`

Look at:
- `build_dag()` for the task graph
- `build_registry()` for local tool registration
- `incident_tool(...)` for the actual CPU-friendly executors
- `run_demo(...)` for the orchestrated run

`examples/incident_triage.py` remains as a compatibility wrapper if you prefer running the demo as a Python module.

## 4. Try Your Own Input

You can point the demo at another text bundle:

```bash
threadswarm demo incident-triage --input-file path/to/your_bundle.txt
```

Or print the final payload as JSON:

```bash
threadswarm demo incident-triage --json
```

Or write the full execution report for debugging/evals:

```bash
threadswarm demo incident-triage --json --report-file reports/incident.json
```

You can also validate a DAG JSON file before wiring execution:

```bash
threadswarm validate-dag path/to/dag.json
```

Or run a DAG JSON file with the built-in text toolkit:

```bash
threadswarm run-dag path/to/dag.json --payload "hello local dag" --json
```

Run the deterministic fixture bundled with ThreadSwarm from any working directory:

```bash
threadswarm eval-golden --json
```

Pass a file or directory when you want to run your own fixtures:

```bash
threadswarm eval-golden evals/golden --json
```

## 5. Use The Pattern For Your Own Workflow

The basic recipe is:

1. Define a `TaskDAG`.
2. Register local tools in `LocalToolRegistry`.
3. Run the DAG with `DAGOrchestrator`.
4. Keep large input in shared memory.
5. Pass only small intermediate results through `dependency_results`.

## Next Reads

- [How ThreadSwarm Works](how-it-works.md) for the exact runtime lifecycle and diagrams
- [Local Tool Pipelines](local-tool-pipelines.md) for the hands-on design guide
- [Configuration](configuration.md) and [`.env.example`](../.env.example) for compiler and worker settings
- [Product Strategy](product-strategy.md) for positioning and the capability roadmap
- [`tests/test_engine.py`](../tests/test_engine.py) for compact execution examples
- [README](../README.md) to return to the project overview
