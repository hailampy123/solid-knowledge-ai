from skai.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.model == "anthropic/claude-3-5-sonnet-latest"
    assert s.chroma_dir == "./.chroma"
    assert s.collection == "knowledge"
    assert s.top_k == 5
    assert s.max_retries == 2
    assert s.langfuse_enabled is False


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
