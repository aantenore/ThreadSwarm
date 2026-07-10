# Product Strategy

This note answers the practical question: does ThreadSwarm make sense as a project?

## Verdict

Yes, but only with a narrow wedge.

ThreadSwarm should not try to become a general-purpose agent framework. That space is already crowded by mature projects and platforms. The useful positioning is:

> CPU-first DAG execution for local, testable tools and large shared payloads, with model calls used only where they add value.

That makes ThreadSwarm a good fit for:
- local incident/log triage;
- document, code, image, audio, and data preprocessing pipelines;
- agentic workflows where most steps are deterministic tools;
- privacy-sensitive or low-resource environments;
- experiments that need a small embeddable runtime instead of a platform.

It is a weak fit for:
- hosted enterprise agent platforms;
- long-running durable workflows with external side effects;
- data platform lineage and scheduling;
- broad multi-agent collaboration UX.

## Current Landscape

Sources checked on 2026-07-10:
- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- MCP specification: https://modelcontextprotocol.io/specification/2025-11-25
- Prefect task runners: https://docs.prefect.io/v3/concepts/task-runners
- Dask local scheduling: https://docs.dask.org/en/stable/scheduling.html
- Ray object serialization: https://docs.ray.io/en/latest/ray-core/objects/serialization.html

The market already covers several adjacent jobs:

| Alternative | Strong at | Why ThreadSwarm should not clone it |
|---|---|---|
| LangGraph | long-running stateful agent orchestration | much richer state, persistence, and production runtime story |
| Prefect/Dask | workflow orchestration and parallel task execution | mature scheduling, integrations, and operational controls |
| Ray | distributed Python execution and object store | production-grade distributed scheduling and shared objects |

## Differentiation

ThreadSwarm can still be useful because its design center is different:

- local-first execution rather than managed orchestration;
- process-based CPU workers rather than cloud/distributed runtime first;
- read-only zero-copy `ndarray` views, plus shared-memory transport for text and bytes on one machine;
- deterministic local tools as the default path;
- small dependency footprint and embeddable code;
- explicit DAG validation before execution.

Novelty score: 2/4 today.

It is not novel as "agent orchestration". It is meaningfully differentiated as a small local runtime for CPU-friendly tool DAGs over shared multimodal context. It can become 3/4 if it adds strong observability, tool contracts, local model adapters, and repeatable evals around that wedge.

Shared-memory NumPy transport is not novel by itself: Ray also documents
same-node, read-only zero-copy deserialization for NumPy arrays. The defensible
wedge is therefore the combination of a small embeddable runtime, explicit
local-tool routing, deterministic evals, and minimal infrastructure—not the
shared-memory primitive in isolation.

## Capability Roadmap

Must have:
- structured execution reports and trace export (implemented);
- stable tool contracts with input/output schemas (implemented);
- failure reports that are useful for regression tests;
- packaged local tool examples beyond incident triage;
- provider adapters kept behind configuration.
- a capability-aware compiler that receives the configured tool/worker catalog
  and cannot silently invent executable routes;
- one supported compile-and-run path that binds planning to those capabilities.

Should have:
- retry policies per task (implemented);
- timeout policies per task (implemented as logical attempt deadlines);
- JSON/YAML DAG loading from CLI (JSON implemented);
- OpenAI-compatible model worker adapter (implemented);
- golden-case eval fixtures for local-tool DAG behavior (implemented);
- optional OpenTelemetry or LangSmith/OpenInference adapter.

Could have:
- local persistent run store;
- MCP adapter for tool discovery;
- visual DAG export;
- Ray/Prefect bridge for users who outgrow the local runtime;
- physical package move away from internal `src.*` modules.

## Current Product Gap

The compiler and executor are deliberately separate today. `SemanticCompiler`
can produce a validated DAG, and the runtime can execute an explicitly routed
DAG, but the compiler does not yet receive the live tool/worker catalog and the
CLI has no single compile-and-run command. An LLM-generated route can therefore
be rejected by the runtime—which is safer than falling back to the wrong pool,
but means ThreadSwarm is currently an execution-runtime MVP rather than a
complete autonomous agent product.

## Implementation Decision

The first capabilities implemented from this strategy are structured execution report export and optional local tool contracts.

Why:
- agentic systems need traces before they need more autonomy;
- report JSON enables debugging, evals, dashboards, and CI regression snapshots;
- tool contracts keep deterministic local tools auditable and safer to compose;
- it strengthens the existing runtime without changing the architecture;
- it does not compete with bigger frameworks.

The CLI now supports:

```bash
threadswarm demo incident-triage --json --report-file reports/incident.json
```

The report includes:
- run summary;
- task status and routing metadata;
- task durations;
- execution order;
- leaf tasks;
- final result;
- dependency results when exported through CLI.

Tool registration now supports:
- input and output schemas through Pydantic models;
- risk class and side-effect class metadata;
- result size limits;
- contract metadata export through `LocalToolRegistry.contracts()`.

Task execution now supports:
- `retry_count`;
- `retry_delay_seconds`;
- `timeout_seconds`;
- run/attempt correlation IDs and retry errors in execution reports;
- fail-closed route validation;
- worker-death detection and tainted-pool recycling.

Timeout behavior is attempt-scoped and begins after a worker start
acknowledgement. When a task exceeds its logical deadline, the orchestrator
marks that attempt failed, can retry it, and ignores late results using run and
attempt IDs. Hard-killing only the specific in-flight worker remains out of
scope; the lightweight runtime immediately recycles the whole tainted pool.
Concurrent in-flight work is not replayed automatically because its side-effect
state may be unknown.

CLI execution now supports:
- `threadswarm run-dag` for JSON DAG files;
- a built-in deterministic `text` toolkit;
- payloads from `--payload` or `--input-file`;
- report export through `--report-file`.

Model-backed execution now supports:
- `OpenAICompatibleWorker`;
- construction from `ThreadSwarmConfig`;
- `to_worker_config()` for `ActorHypervisor`;
- compact JSON task envelopes with payload previews and dependency results.

Golden evals now support:
- file-backed JSON cases under `evals/golden`;
- deterministic local toolkit execution;
- subset comparison against report JSON;
- `threadswarm eval-golden` for local regression checks.

Packaging now supports:
- public `threadswarm.*` imports;
- `threadswarm.cli:main` as the console entrypoint;
- compatibility wrappers over the current internal `src.*` modules.

## Build-Vs-Buy Recommendation

Build ThreadSwarm if the target is local, CPU-first, deterministic-tool-heavy workflows.

Integrate instead of competing when users need:
- long-running durable orchestration: use LangGraph or a workflow engine;
- enterprise data orchestration: use Dagster or Prefect;
- distributed compute: use Ray;
- standardized external tool ecosystems: add MCP compatibility rather than inventing a private protocol.
