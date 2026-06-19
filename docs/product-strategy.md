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

Sources checked on 2026-06-19:
- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- Microsoft Agent Framework: https://learn.microsoft.com/en-us/agent-framework/overview/
- CrewAI: https://docs.crewai.com/en/introduction
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- MCP specification: https://modelcontextprotocol.io/specification/2025-11-25
- Prefect: https://docs.prefect.io/v3/get-started
- Dagster: https://docs.dagster.io/
- Ray objects: https://docs.ray.io/en/latest/ray-core/key-concepts.html

The market already covers several adjacent jobs:

| Alternative | Strong at | Why ThreadSwarm should not clone it |
|---|---|---|
| LangGraph | long-running stateful agent orchestration | much richer state, persistence, and production runtime story |
| Microsoft Agent Framework | enterprise agent SDK and telemetry | stronger ecosystem and provider integration |
| CrewAI | high-level multi-agent automation | better role/crew abstraction and user-facing automation story |
| OpenAI Agents SDK | provider-native agents, tracing, guardrails | best when OpenAI-hosted primitives are the desired runtime |
| Prefect/Dagster | workflow/data orchestration | scheduling, observability, lineage, retries, and operations are mature |
| Ray | distributed Python execution and object store | production-grade distributed scheduling and shared objects |

## Differentiation

ThreadSwarm can still be useful because its design center is different:

- local-first execution rather than managed orchestration;
- process-based CPU workers rather than cloud/distributed runtime first;
- zero-copy shared memory for large payloads on one machine;
- deterministic local tools as the default path;
- small dependency footprint and embeddable code;
- explicit DAG validation before execution.

Novelty score: 2/4 today.

It is not novel as "agent orchestration". It is meaningfully differentiated as a small local runtime for CPU-friendly tool DAGs over shared multimodal context. It can become 3/4 if it adds strong observability, tool contracts, local model adapters, and repeatable evals around that wedge.

## Capability Roadmap

Must have:
- structured execution reports and trace export (implemented);
- stable tool contracts with input/output schemas (implemented);
- failure reports that are useful for regression tests;
- packaged local tool examples beyond incident triage;
- provider adapters kept behind configuration.

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
- package rename from `src.*` imports to a public `threadswarm.*` namespace.

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
- attempt counts and retry errors in execution reports.

Timeout behavior is attempt-scoped: when a task exceeds its logical deadline,
the orchestrator marks that attempt failed, can retry it, and ignores late
results from expired attempts. Hard-killing only the specific in-flight worker is
still out of scope for the current lightweight local pool.

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

## Build-Vs-Buy Recommendation

Build ThreadSwarm if the target is local, CPU-first, deterministic-tool-heavy workflows.

Integrate instead of competing when users need:
- long-running durable orchestration: use LangGraph, Microsoft Agent Framework, or workflow engines;
- enterprise data orchestration: use Dagster or Prefect;
- distributed compute: use Ray;
- standardized external tool ecosystems: add MCP compatibility rather than inventing a private protocol.
