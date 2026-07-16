# Delivery contract: capability-aware compile-and-run

- Date: 2026-07-17
- Delivery mode: feature branch, tested pull request, then fast-forward `main`
- Release impact: none; no tag or package publication in this tranche
- Decision record: [RFC 0002](rfcs/0002-capability-bound-compilation.md)

## Objective

Turn natural-language intent into an executable local-tool DAG without allowing
the model to invent a route, select an unapproved side effect, or execute a plan
against capabilities different from the ones it saw during compilation.

## Scope

Must:

- derive a deterministic catalog from the live `LocalToolRegistry`;
- admit capabilities through application-owned risk and side-effect policy;
- give the compiler a compact catalog projection with an explicit untrusted-data
  boundary;
- require every compiled task to name one exact admitted tool;
- reject missing, unknown, ambiguous, or modality-incompatible routes before
  worker startup;
- bind the DAG to the catalog digest and detect later registry or plan mutation;
- enforce finite application-owned budgets before recursive validation or worker
  startup;
- execute a deep verified snapshot and start only the selected tool pools;
- expose one library transaction and one CLI command;
- export a bound plan and combined execution evidence without requiring a model
  provider during deterministic tests.

Out of scope:

- model-worker capability catalogs, MCP discovery, remote workers, persistence,
  distributed scheduling, approvals, provider installation, releases, and tags;
- treating prompt/schema adherence as an authorization control;
- executing tools with side effects under the default policy.

## Acceptance matrix

| ID | Requirement | Acceptance criteria | Verification |
|---|---|---|---|
| C1 | Deterministic catalog | Same registry and policy produce the same digest and sorted prompt projection | Catalog unit test |
| C2 | Bounded context | Full and prompt catalogs, descriptions, schema metadata, user intent, and compiler response have explicit byte/character/count caps | Unit and adversarial limit tests |
| C3 | Trust boundary | Catalog fields are labelled untrusted data and cannot override the compiler contract | Prompt test and documentation review |
| C4 | Route binding | Missing, invented, dual, and wrong-modality routes fail before execution | Parameterized adversarial tests |
| C5 | Side-effect default | Non-`none` side effects are absent unless explicitly admitted; admitted side effects cannot auto-retry without an idempotency contract | Policy tests |
| C6 | Registry fence | Registry drift after compilation produces a typed `catalog_digest_mismatch` | Mutation test |
| C7 | Plan fence | Pre-snapshot mutation is rejected and post-snapshot caller mutation cannot change executed instructions or routes | Mutation and TOCTOU regression tests |
| C8 | End-to-end path | Safe natural-language plan reaches deterministic local execution | Runtime and CLI integration tests |
| C9 | Evidence | CLI writes the exact bound plan and combined execution report on success and after an execution failure | CLI integration tests |
| C10 | Regression safety | Full existing test suite, lint, package build, and wheel checks pass | Local and CI gates |
| C11 | Resource safety | Task count, dependencies, text fields, retries, per-task timeout, and whole-run timeout remain within finite policy limits | Boundary and non-finite-value tests |
| C12 | Least privilege | Runtime creates workers only for policy-admitted tools referenced by the verified DAG snapshot | Worker-subset integration test |

## Rollback

The change is additive. Reverting the feature commit removes the catalog,
binding layer, CLI command, tests, and documentation without migrating stored
data or changing the existing `compile` and `run-dag` contracts.

## Evidence

Local pre-publication evidence:

- Ruff: passed;
- pytest: 121 passed;
- independent review: safe to ship, no residual P0/P1/P2 finding;
- `git diff --check`: passed;
- source distribution and wheel build: passed;
- Twine metadata checks for both artifacts: passed;
- isolated wheel import and `compile-run --help` smoke test: passed.

Remote CI, canonical Git identity, and clean `main` are recorded during
publication. Acceptance requires all C1-C12 checks; no tag or release is part
of this delivery.
