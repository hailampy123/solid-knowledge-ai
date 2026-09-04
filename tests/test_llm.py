"""make_llm provider routing: key export, provider-aware key check, knob passthrough.

No network — ChatLiteLLM construction is offline; we only assert how it was built.
"""
import pytest

from skai.agent.llm import make_llm
from skai.config import Settings


def _settings(monkeypatch, **over):
    # start from a clean env so a developer's real keys don't leak into asserts
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None, **over)


def test_missing_key_for_chosen_provider_raises(monkeypatch):
    s = _settings(monkeypatch, SKAI_MODEL="gpt-4o")  # openai, no OPENAI_API_KEY
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_llm(s, callbacks=[])


def test_key_exported_and_model_resolved(monkeypatch):
    s = _settings(monkeypatch, SKAI_MODEL="gemini-flash", GEMINI_API_KEY="sk-gem")
    llm = make_llm(s, callbacks=[])
    assert llm.model == "gemini/gemini-2.5-flash"
    import os
    assert os.environ["GEMINI_API_KEY"] == "sk-gem"
    assert os.environ["GOOGLE_API_KEY"] == "sk-gem"  # mirrored for Gemini


def test_api_base_skips_key_check_and_passes_knobs(monkeypatch):
    s = _settings(
        monkeypatch, SKAI_MODEL="openai/gpt-4o", SKAI_API_BASE="http://localhost:11434",
        SKAI_MAX_TOKENS="256", SKAI_TOP_P="0.9",
    )
    llm = make_llm(s, callbacks=[])  # no key, but api_base set => no raise
    assert llm.api_base == "http://localhost:11434"
    assert llm.max_tokens == 256
    assert llm.top_p == 0.9
