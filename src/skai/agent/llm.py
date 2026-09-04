"""LiteLLM-backed chat model factory. Provider-agnostic (Anthropic/OpenAI/Gemini/…)."""
from __future__ import annotations

import os

from langchain_litellm import ChatLiteLLM

from skai.config import Settings, resolve_model
from skai.observability import get_callbacks

# LiteLLM provider prefix -> the env var it reads for that provider's key.
_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def make_llm(settings: Settings, callbacks: list | None = None) -> ChatLiteLLM:
    """Build a ChatLiteLLM for the configured model, whatever the provider.

    Exports any configured provider key to the env LiteLLM reads, resolves model
    aliases (blocks Opus), and raises a clear error if the chosen model's
    provider has no key (unless a custom `api_base` is set).
    """
    model = resolve_model(settings.model)

    # Export whichever keys are configured; GOOGLE_API_KEY mirrors GEMINI_API_KEY
    # so either name in the environment works for Gemini.
    for env_var, value in (
        ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("GEMINI_API_KEY", settings.gemini_api_key),
        ("GOOGLE_API_KEY", settings.gemini_api_key),
    ):
        if value:
            os.environ.setdefault(env_var, value)

    provider = model.split("/", 1)[0] if "/" in model else ""
    key_env = _PROVIDER_KEY_ENV.get(provider)
    if key_env and not os.environ.get(key_env) and not settings.api_base:
        raise RuntimeError(
            f"{key_env} is not set for model '{model}'. Add it to .env or the "
            f"environment, set SKAI_MODEL to a provider you have a key for, or "
            f"set SKAI_API_BASE for a local/proxy endpoint."
        )

    kwargs: dict = {
        "model": model,
        "temperature": settings.temperature,
        "callbacks": callbacks if callbacks is not None else get_callbacks(settings),
    }
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    if settings.top_p is not None:
        kwargs["top_p"] = settings.top_p
    if settings.api_base:
        kwargs["api_base"] = settings.api_base

    return ChatLiteLLM(**kwargs)
