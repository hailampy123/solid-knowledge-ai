"""Langfuse tracing wiring. No keys => empty callback list (tracing disabled)."""
from __future__ import annotations

import logging

from skai.config import Settings

logger = logging.getLogger(__name__)


def get_callbacks(settings: Settings) -> list:
    """Return LangChain callbacks for graph/LLM invokes.

    A Langfuse handler when keys are configured, otherwise an empty list so the
    graph runs untraced with zero setup.
    """
    if not settings.langfuse_enabled:
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Configure the global client from settings (also honours env vars).
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host or "https://cloud.langfuse.com",
        )
        return [CallbackHandler()]
    except Exception as e:  # noqa: BLE001 - tracing must never break the app
        logger.warning("Langfuse disabled: %s", e)
        return []
