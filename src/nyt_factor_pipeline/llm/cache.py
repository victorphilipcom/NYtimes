"""LLM prompt/response cache — avoid redundant API calls."""

from __future__ import annotations

from datetime import datetime

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.utils.hashing import prompt_hash

log = get_logger(__name__)


class LLMCache:
    """Cache LLM responses by prompt hash in DuckDB."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def get(self, prompt_text: str) -> str | None:
        """Look up a cached response by prompt hash."""
        ph = prompt_hash(prompt_text)
        result = self.conn.execute(
            "SELECT response_text FROM llm_cache WHERE prompt_hash = ?", [ph]
        ).fetchone()
        if result:
            log.debug("llm_cache_hit", hash=ph[:8])
            return result[0]
        return None

    def put(
        self,
        prompt_text: str,
        response_text: str,
        model: str = "",
        tokens_used: int = 0,
    ) -> None:
        """Store a response in the cache."""
        ph = prompt_hash(prompt_text)
        now = datetime.utcnow()
        self.conn.execute(
            """INSERT INTO llm_cache (prompt_hash, prompt_text, response_text, model, tokens_used, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT (prompt_hash) DO UPDATE SET
                   response_text = ?, model = ?, tokens_used = ?, created_at = ?""",
            [ph, prompt_text, response_text, model, tokens_used, now,
             response_text, model, tokens_used, now],
        )

    def count(self) -> int:
        result = self.conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return result[0] if result else 0
