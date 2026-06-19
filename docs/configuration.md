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
from src.compiler import SemanticCompiler
from src.config import ThreadSwarmConfig

config = ThreadSwarmConfig.from_env()
compiler = SemanticCompiler.from_config(config)
```

At the CLI:

```bash
threadswarm compile "Analyze this incident bundle and produce a triage DAG" \
  --base-url http://localhost:11434/v1 \
  --model llama3.2
```

`compile` needs a running OpenAI-compatible provider. The local-tool demo and DAG validator do not.
