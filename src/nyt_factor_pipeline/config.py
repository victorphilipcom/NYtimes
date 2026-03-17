"""Typed configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys ---
    nyt_api_key: str = ""
    openai_api_key: str = ""

    # --- Database ---
    db_path: Path = Path("data/nyt_pipeline.duckdb")

    # --- Embeddings ---
    embedding_backend: str = "local"  # "local" or "openai"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # --- NYT rate limiting ---
    nyt_requests_per_minute: int = 3
    nyt_archive_daily_budget: int = 1500
    nyt_article_search_daily_budget: int = 500
    nyt_concurrency: int = 1
    nyt_max_retries: int = 5

    # --- Clustering ---
    clustering_min_cluster_size: int = 5
    clustering_min_samples: int = 3
    theme_similarity_threshold: float = 0.65
    theme_merge_threshold: float = 0.80

    # --- Scoring ---
    importance_min_word_count: int = 200
    importance_word_count_boost_threshold: int = 800

    # --- Logging ---
    log_level: str = "INFO"

    # --- LLM token budget ---
    llm_max_daily_tokens: int = Field(default=100_000, description="Max tokens/day for LLM calls")

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_nyt(self) -> bool:
        return bool(self.nyt_api_key)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
