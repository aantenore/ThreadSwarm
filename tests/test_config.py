import pytest

from threadswarm.config import ThreadSwarmConfig, ThreadSwarmConfigError


def test_config_from_env_defaults():
    config = ThreadSwarmConfig.from_env({})

    assert config.llm_base_url == "http://localhost:11434/v1"
    assert config.llm_model == "llama3.2"
    assert config.llm_timeout == 60.0
    assert config.default_workers is None


def test_config_from_env_overrides():
    config = ThreadSwarmConfig.from_env(
        {
            "THREADSWARM_LLM_BASE_URL": "http://localhost:9999/v1",
            "THREADSWARM_LLM_MODEL": "qwen2.5",
            "THREADSWARM_LLM_TIMEOUT": "12.5",
            "THREADSWARM_DEFAULT_WORKERS": "3",
        }
    )

    assert config.llm_base_url == "http://localhost:9999/v1"
    assert config.llm_model == "qwen2.5"
    assert config.llm_timeout == 12.5
    assert config.default_workers == 3


def test_config_from_process_environment(monkeypatch):
    monkeypatch.setenv("THREADSWARM_LLM_MODEL", "local-test-model")

    config = ThreadSwarmConfig.from_env()

    assert config.llm_model == "local-test-model"


def test_config_rejects_invalid_numbers():
    with pytest.raises(ThreadSwarmConfigError, match="THREADSWARM_LLM_TIMEOUT"):
        ThreadSwarmConfig.from_env({"THREADSWARM_LLM_TIMEOUT": "soon"})

    with pytest.raises(ThreadSwarmConfigError, match="default_workers"):
        ThreadSwarmConfig(default_workers=0)
