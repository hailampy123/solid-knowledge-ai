"""Runtime settings loaded from environment / .env.

Mixed naming on purpose:
- Provider vars keep their standard names (`ANTHROPIC_API_KEY`, `LANGFUSE_*`)
  so LiteLLM and Langfuse pick them up from the environment directly too.
- App-owned knobs use a `SKAI_` prefix to avoid colliding with generic names.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )

    # LLM (provider-standard names)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # App-owned knobs (SKAI_ prefix)
    model: str = Field(default="anthropic/claude-3-5-sonnet-latest", alias="SKAI_MODEL")
    temperature: float = Field(default=0.0, alias="SKAI_TEMPERATURE")
    chroma_dir: str = Field(default="./.chroma", alias="SKAI_CHROMA_DIR")
    collection: str = Field(default="knowledge", alias="SKAI_COLLECTION")
    memory_db: str = Field(default="./.skai/memory.sqlite", alias="SKAI_MEMORY_DB")
    top_k: int = Field(default=5, alias="SKAI_TOP_K")
    max_retries: int = Field(default=2, alias="SKAI_MAX_RETRIES")

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
