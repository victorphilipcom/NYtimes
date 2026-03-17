"""Tests for LLM prompt caching."""

import duckdb

from nyt_factor_pipeline.db import init_schema


def _get_test_db():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    return conn


class TestLLMCache:
    def test_put_and_get(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.llm.cache import LLMCache

        cache = LLMCache(conn)
        prompt = "What is the theme about?"
        response = "This theme covers economic policy."

        cache.put(prompt, response, model="gpt-4o-mini", tokens_used=50)
        result = cache.get(prompt)
        assert result == response

    def test_cache_miss(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.llm.cache import LLMCache

        cache = LLMCache(conn)
        assert cache.get("nonexistent prompt") is None

    def test_cache_overwrite(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.llm.cache import LLMCache

        cache = LLMCache(conn)
        prompt = "Same prompt"

        cache.put(prompt, "First response")
        cache.put(prompt, "Updated response")
        assert cache.get(prompt) == "Updated response"

    def test_cache_count(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.llm.cache import LLMCache

        cache = LLMCache(conn)
        assert cache.count() == 0

        cache.put("prompt1", "response1")
        cache.put("prompt2", "response2")
        assert cache.count() == 2

    def test_different_prompts_different_hashes(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.llm.cache import LLMCache

        cache = LLMCache(conn)
        cache.put("prompt A", "response A")
        cache.put("prompt B", "response B")

        assert cache.get("prompt A") == "response A"
        assert cache.get("prompt B") == "response B"
