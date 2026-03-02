# ThreadSwarm — The Semantic Compiler & Distributed AI on CPUs

**Software 3.0**: Natural-language intent is compiled into a Directed Acyclic Graph (DAG) of micro-tasks, executed by a swarm of small vision/language models on **local CPUs**—no GPUs required.

## Philosophy

- **Problem**: Monolithic models are expensive, slow, and wasteful for simple sub-tasks.
- **Solution**: A **Semantic Compiler** turns user intent into a DAG of sub-problems; a swarm of **Actors** (small models on CPU cores) executes them in parallel via **zero-copy shared memory**, avoiding Python's GIL and serialization bottlenecks.

## Architecture

1. **Semantic Compiler** (`src/compiler/`)  
   Connects to a local LLM (OpenAI-compatible, e.g. Ollama at `localhost:11434`). Takes a user prompt and outputs a strict JSON DAG of `SubTask` objects (Pydantic-validated).

2. **Zero-Copy Memory** (`src/engine/shared_memory.py`)  
   `VisionMemoryManager` loads images (e.g. via numpy/cv2), allocates `multiprocessing.shared_memory.SharedMemory`, and exposes metadata (name, shape, dtype) so workers can reconstruct numpy arrays without copying the buffer.

3. **Actor Swarm** (`src/engine/actor_pool.py`)  
   `ActorHypervisor` spawns N worker processes (based on `os.cpu_count()`). Workers listen on a task queue, reconstruct images from shared memory, run inference, and send results to a result queue. Graceful shutdown is supported.

## Constraints

- **No threading for AI inference** — use `multiprocessing` only.
- **No pickle / IPC queues for image tensors** — images only via shared memory; only metadata over queues.

## Engineering & contribution guidelines

- **No threading for inference** — Do not use Python's `threading` for AI/model inference (GIL). Use `multiprocessing` (Process, Queue, Pipe, etc.).
- **No pickle/IPC for image data** — Do not pass image tensors or large buffers via pickle or queues. Use `multiprocessing.shared_memory.SharedMemory`; only send metadata (name, shape, dtype) over queues.
- **Windows (spawn)** — When using `ActorHypervisor` with a custom `run_inference_hook`, the hook must be picklable (e.g. a module-level function), not a lambda or nested function.
- **Code** — Use type hints and Pydantic for public APIs and task schemas. Keep `src/compiler` for intent → DAG logic, `src/engine` for execution, `src/models` for model adapters.
- **RFCs** — New architectural features require an RFC in `docs/rfcs/` before code; implementation must follow the spec.
- **Quality** — No placeholders or mocks in the core engine; contributions should be production-ready.

## Structure

```
docs/rfcs/     — RFCs for architectural changes
src/compiler/  — Semantic Compiler (parser, DAG generation)
src/engine/    — Shared memory, actor pool, execution
src/models/    — Model adapters (e.g. SLM loaders)
tests/         — Tests
```

## Requirements

- Python 3.10+
- Dependencies: see `requirements.txt` or `pyproject.toml`

## License

Open source; see repository license file.
