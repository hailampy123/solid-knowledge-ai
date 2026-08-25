import pytest

from skai.agent.llm import make_llm
from skai.config import Settings
from skai.observability import get_callbacks


def test_no_langfuse_returns_empty_callbacks():
    s = Settings(_env_file=None)  # no langfuse keys
    assert get_callbacks(s) == []


def test_make_llm_requires_key_for_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings(_env_file=None, ANTHROPIC_API_KEY=None, SKAI_MODEL="anthropic/claude-3-5-sonnet-latest")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_llm(s)


def test_make_llm_builds_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    s = Settings(_env_file=None, SKAI_MODEL="anthropic/claude-3-5-sonnet-latest")
    llm = make_llm(s, callbacks=[])
    assert llm.model == "anthropic/claude-3-5-sonnet-latest"
