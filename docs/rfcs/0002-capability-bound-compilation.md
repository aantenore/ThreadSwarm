# RFC 0002: Capability-bound compilation

- Status: accepted
- Date: 2026-07-17

## Context

`SemanticCompiler` could produce a valid `TaskDAG`, and `DAGOrchestrator` could
execute an explicitly routed graph, but no contract connected planning to the
live tool registry. A generated route failed safely when unknown, yet callers
still had to join compilation and execution themselves and could accidentally
run a plan after the registry changed.

ThreadSwarm should remain a small local runtime. Distributed compute, durable
workflow engines, and broad agent frameworks already solve different layers.
The missing product layer is therefore a narrow binding transaction, not a new
scheduler or protocol.

## Decision

Add `CapabilityAwareRuntime` around the existing compiler, registry, and
orchestrator.

The transaction is:

```text
live registry
  -> policy admission
  -> deterministic full-catalog digest
  -> compact prompt projection
  -> model-proposed TaskDAG
  -> deterministic route validation
  -> bound-plan digest
  -> pre-execution catalog and plan fences
  -> existing DAGOrchestrator
```

### Application-owned policy

The default `CapabilityPolicy` admits only `compute_only`, `read_only`, and
`search_only` risk classes whose side-effect class is `none`. Applications may
construct another policy explicitly; prompt text cannot expand it. The policy
also caps the full and prompt catalog, tool metadata, user intent, compiler
response, task count and fields, dependency fan-in, retries, task timeouts, and
the complete run. An explicitly admitted side-effecting tool cannot auto-retry
until the host provides a separate idempotency contract.

### Prompt projection

The model receives tool name, description, modalities, risk, side-effect class,
result-size limit, and a compact output-contract summary. Repeated full input
envelope schemas are excluded from the prompt but remain part of the full catalog
digest. Catalog descriptions and schemas are labelled untrusted data.

### Deterministic enforcement

Every task must set exactly one admitted `tool_name`. `model_type` is rejected in
this local-tool path because dual routing is ambiguous even though the legacy
runtime would prioritize `tool_name`. Missing routes, aliases, unknown names, and
unsupported modalities produce structured `CapabilityPlanIssue` records.

`BoundTaskDAG` stores two SHA-256 fences:

- `catalog_digest` binds the plan to the admitted registry contracts and policy;
- `plan_digest` binds the complete DAG to that catalog digest.

Immediately before worker creation, the runtime deep-copies the caller-owned DAG,
checks the digest of that snapshot, and then validates, routes, and executes only
the snapshot. This closes the check-to-use mutation window. It also creates
worker pools only for tool names referenced by the verified snapshot. The
digests provide mutation integrity, not signer identity or cross-process
provenance.

## Consequences

Benefits:

- one supported compile-and-run path without implicit fallback;
- smaller and more stable model context than full contract injection;
- deterministic safety checks independent of model behavior;
- machine-readable rejection reasons and evidence artifacts;
- bounded compiler/runtime resources and failure evidence;
- least-privilege process startup for the verified plan;
- no changes to the existing manual DAG path.

Costs and limits:

- the model still determines decomposition quality;
- a registry change requires recompilation even when the changed detail is not
  used by the DAG;
- local tools only; model workers and discovered MCP tools require separate
  typed adapters and policy decisions;
- digests detect mutation but do not authenticate an artifact producer.

## Rejected alternatives

- **Prompt-only tool instructions:** cannot enforce authorization or prevent
  runtime drift.
- **Silent fallback to a default worker:** hides invalid plans and may execute
  work in the wrong trust boundary.
- **Expose every registry entry:** expands context and lets risky tools compete
  for model attention before policy admission.
- **Build a distributed scheduler:** duplicates mature systems and abandons the
  local CPU-first wedge.
