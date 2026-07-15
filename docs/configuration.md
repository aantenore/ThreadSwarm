# Configuration

ThreadSwarm keeps machine-specific and provider-specific values in `ThreadSwarmConfig`.

The local defaults target an OpenAI-compatible Ollama endpoint:

```bash
THREADSWARM_LLM_BASE_URL=http://localhost:11434/v1
THREADSWARM_LLM_MODEL=llama3.2
THREADSWARM_LLM_TIMEOUT=60
THREADSWARM_DEFAULT_WORKERS=
```

Use `.env.example` as the canonical list of supported keys.

In code:

```python
from threadswarm.compiler import SemanticCompiler
from threadswarm.config import ThreadSwarmConfig
from threadswarm.engine import LocalToolRegistry
from threadswarm.models import OpenAICompatibleWorker

config = ThreadSwarmConfig.from_env()
compiler = SemanticCompiler.from_config(config)
model_worker = OpenAICompatibleWorker.from_config(config)
registry = LocalToolRegistry.from_config(config)
```

`default_workers` is used by `LocalToolRegistry.from_config(...)` for tools that
do not specify their own `num_workers`; explicit per-tool values still win. The
`run-dag` CLI also applies it to the selected built-in toolkit.

At the CLI:

```bash
threadswarm compile "Analyze this incident bundle and produce a triage DAG" \
  --base-url http://localhost:11434/v1 \
  --model llama3.2
```

`compile` and `OpenAICompatibleWorker` need a running OpenAI-compatible provider.
The local-tool demo, JSON DAG runner, and DAG validator do not.

## Optional Vision Dependency

The base package keeps OpenCV optional. It can share NumPy arrays and load `.npy`
payloads without OpenCV. Install the `vision` extra to load JPEG, PNG, BMP, or
WebP files directly:

```bash
python -m pip install "ThreadSwarm[vision]"
```

For an editable development checkout, use `python -m pip install -e ".[dev,vision]"`.
