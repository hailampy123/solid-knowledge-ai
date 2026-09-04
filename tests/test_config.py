import pytest

from skai.config import Settings, resolve_model


def test_code_defaults():
    # assert the code defaults from the schema, immune to any developer .env
    f = Settings.model_fields
    assert f["model"].default == "anthropic/claude-haiku-4-5"
    assert f["chroma_dir"].default == "./.chroma"
    assert f["collection"].default == "knowledge"
    assert f["top_k"].default == 3
    assert f["max_retries"].default == 2
    # generation knobs default to None => provider default, not a forced value
    assert f["max_tokens"].default is None
    assert f["top_p"].default is None
    assert f["api_base"].default is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("SKAI_MODEL", "anthropic/claude-3-haiku-latest")
    monkeypatch.setenv("SKAI_TOP_K", "3")
    s = Settings(_env_file=None)
    assert s.anthropic_api_key == "sk-test"
    assert s.model == "anthropic/claude-3-haiku-latest"
    assert s.top_k == 3


def test_langfuse_enabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    s = Settings(_env_file=None)
    assert s.langfuse_enabled is True


def test_resolve_model_aliases():
    assert resolve_model("haiku") == "anthropic/claude-haiku-4-5"
    assert resolve_model("sonnet") == "anthropic/claude-sonnet-4-5"
    # cross-provider aliases
    assert resolve_model("gemini-flash") == "gemini/gemini-2.5-flash"
    assert resolve_model("gpt-4o-mini") == "openai/gpt-4o-mini"
    # a full id (any provider) passes through unchanged
    assert resolve_model("anthropic/claude-sonnet-4-5") == "anthropic/claude-sonnet-4-5"
    assert resolve_model("gemini/gemini-2.5-pro") == "gemini/gemini-2.5-pro"


def test_provider_keys_and_knobs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    monkeypatch.setenv("SKAI_MAX_TOKENS", "512")
    monkeypatch.setenv("SKAI_API_BASE", "http://localhost:11434")
    s = Settings(_env_file=None)
    assert s.openai_api_key == "sk-openai"
    assert s.gemini_api_key == "sk-gemini"
    assert s.max_tokens == 512
    assert s.api_base == "http://localhost:11434"


def test_resolve_model_blocks_opus():
    with pytest.raises(ValueError, match="Opus"):
        resolve_model("anthropic/claude-opus-4-5")
    with pytest.raises(ValueError, match="Opus"):
        resolve_model("opus")  # not an alias, falls through, then blocked
