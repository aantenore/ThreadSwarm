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

See `.cursorrules` for full engineering and contribution guidelines.

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
