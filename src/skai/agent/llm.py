"""LiteLLM-backed chat model factory (routes to Claude by default)."""
from __future__ import annotations

import os

from langchain_litellm import ChatLiteLLM

from skai.config import Settings, resolve_model
from skai.observability import get_callbacks


def make_llm(settings: Settings, callbacks: list | None = None) -> ChatLiteLLM:
    """Build a ChatLiteLLM. Exports the Anthropic key to the env LiteLLM reads.

    Resolves model aliases (haiku/sonnet), blocks Opus, and raises a clear error
    if an Anthropic model is requested without a key.
    """
    model = resolve_model(settings.model)

    if settings.anthropic_api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

    if model.startswith("anthropic/") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or the environment "
            "(or set SKAI_MODEL to a provider you have a key for)."
        )

    return ChatLiteLLM(
        model=model,
        temperature=settings.temperature,
        callbacks=callbacks if callbacks is not None else get_callbacks(settings),
    )
