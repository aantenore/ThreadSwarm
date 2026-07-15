# RFC 0001: Run Isolation And Pool Recovery

Status: accepted

## Context

`ActorHypervisor` may be kept alive across multiple DAG runs. Previously, worker
results were correlated only by task ID and retry number. A late result from a
timed-out run could therefore satisfy a later run that reused the same task ID.
Timeouts could also unlink owned shared memory while queued work still referred
to it, and an exited worker could leave the orchestrator waiting forever.

## Decision

- Every orchestrated run gets a unique `run_id`; every submission gets a unique
  `attempt_id`. Both values travel through worker lifecycle and result events.
- One orchestrated run owns a hypervisor at a time. Concurrent ownership fails
  immediately instead of competing for shared result queues.
- Explicit heterogeneous pools resolve routes before allocating context or
  submitting work. Missing and unknown routes fail closed. The legacy
  homogeneous constructor remains a deliberate generic fallback.
- Workers acknowledge `started` only after attaching shared context. Per-task
  execution deadlines begin from that event, not queue submission.
- Worker death, external pool shutdown, fail-fast cancellation, and global
  timeout terminalize every record. A task-level timeout immediately discards
  the current generation and starts a fresh generation of every route pool before
  retrying; unrelated in-flight attempts fail rather
  than being replayed with uncertain side effects.
- A pool that was already running before the DAG is recreated after recovery;
  a pool created for the DAG remains stopped.
- Each pool start advances a generation token. An external restart invalidates
  in-flight orchestration explicitly, so work lost with the old queues becomes
  a terminal report instead of an indefinite wait.
- Task and result payloads are checked for multiprocessing serializability before
  relying on queue feeder threads. Queue envelopes contain detached serialized
  bytes, so a result may safely reference its input shared-memory view. An
  invalid result becomes a correlated task failure instead of a silent hang.
- Queue teardown treats task queues as disposable after their consumers exit.
  Backlog is released and a blocked feeder is interrupted, keeping shutdown
  bounded even after worker death or a saturated pipe.
- On POSIX, the parent starts its shared-memory resource tracker before forking
  a pool. Prestarted workers therefore inherit the parent's tracker and cannot
  unlink parent-owned context when a timed-out generation is terminated.

## Consequences

Correctness is favored over preserving warm workers after an abnormal run. A
timeout may reload every worker in the affected hypervisor, including healthy
workers. Selective cancellation would require assignment tracking and a richer
supervisor protocol; that complexity is deferred until measurements justify it.

The hypervisor is reusable across sequential runs, but it is not a concurrent
multi-tenant dispatcher. Callers that need concurrent DAGs should create
separate hypervisors or introduce a higher-level scheduler.

## Verification

Regression coverage proves:

- late work from run A cannot complete run B;
- missing and unknown routes execute nothing;
- task timeouts begin after worker start acknowledgement;
- worker death and external shutdown end within a bounded interval;
- external restart invalidates the active generation without stranding the run;
- saturated queues release feeder threads and buffered task payloads;
- a prestarted `fork` pool can recycle a timed-out attempt and reattach the
  parent-owned context without tracker warnings;
- fail-fast cancels independent in-flight work;
- all timeout reports contain only terminal task states;
- unpicklable tasks fail synchronously and unpicklable results become task errors.
