"""Runtime settings loaded from environment / .env.

Mixed naming on purpose:
- Provider vars keep their standard names (`ANTHROPIC_API_KEY`, `LANGFUSE_*`)
  so LiteLLM and Langfuse pick them up from the environment directly too.
- App-owned knobs use a `SKAI_` prefix to avoid colliding with generic names.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Short aliases -> LiteLLM model ids. Opus is intentionally excluded.
MODEL_ALIASES = {
    "haiku": "anthropic/claude-haiku-4-5",
    "sonnet": "anthropic/claude-sonnet-4-5",
}


def resolve_model(name: str) -> str:
    """Map a short alias (haiku/sonnet) or a full id to a LiteLLM id.

    Opus is blocked for this assistant (cost/latency choice for a Q&A agent).
    """
    resolved = MODEL_ALIASES.get(name.lower().strip(), name)
    if "opus" in resolved.lower():
        raise ValueError(
            "Opus models are disabled for this assistant. Use 'haiku', 'sonnet', "
            "or another non-Opus model id."
        )
    return resolved


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )

    # LLM (provider-standard names)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # App-owned knobs (SKAI_ prefix)
    model: str = Field(default="anthropic/claude-haiku-4-5", alias="SKAI_MODEL")
    # DeepEval judge model; falls back to `model` when unset. A stronger judge
    # (e.g. sonnet) grades more reliably than the agent's own model.
    judge_model: str | None = Field(default=None, alias="SKAI_JUDGE_MODEL")
    temperature: float = Field(default=0.0, alias="SKAI_TEMPERATURE")
    chroma_dir: str = Field(default="./.chroma", alias="SKAI_CHROMA_DIR")
    collection: str = Field(default="knowledge", alias="SKAI_COLLECTION")
    memory_db: str = Field(default="./.skai/memory.sqlite", alias="SKAI_MEMORY_DB")

    @property
    def feedback_db(self) -> str:
        """User-feedback + retrieval-gap store; sits beside the memory db."""
        return str(Path(self.memory_db).with_name("feedback.sqlite"))

    # Lower k keeps retrieved context focused: fewer off-topic chunks dilute the
    # answer and score better on contextual-relevancy. Raise for broader recall.
    top_k: int = Field(default=3, alias="SKAI_TOP_K")
    max_retries: int = Field(default=2, alias="SKAI_MAX_RETRIES")

    # Guardrails. Redaction + injection scan default on; refusal topics off.
    pii_redaction: bool = Field(default=True, alias="SKAI_PII_REDACTION")
    injection_scan: bool = Field(default=True, alias="SKAI_INJECTION_SCAN")
    refusal_topics_raw: str = Field(default="", alias="SKAI_REFUSAL_TOPICS")

    @property
    def refusal_topics(self) -> list[str]:
        """Comma-separated denied output-policy topics; empty => policy off."""
        return [t.strip() for t in self.refusal_topics_raw.split(",") if t.strip()]

    # Observability (Langfuse-standard names). Absent keys => tracing disabled.
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, alias="LANGFUSE_HOST")

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
