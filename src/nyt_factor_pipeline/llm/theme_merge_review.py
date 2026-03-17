"""LLM-based merge review for candidate theme pairs."""

from __future__ import annotations

import duckdb

from nyt_factor_pipeline.clustering.merge_split import execute_merge, find_merge_candidates
from nyt_factor_pipeline.llm.cache import LLMCache
from nyt_factor_pipeline.llm.openai_client import OpenAIClient
from nyt_factor_pipeline.llm.prompts import MERGE_REVIEW_SYSTEM, format_merge_review_prompt
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.themes.theme_store import get_theme_representative_data

log = get_logger(__name__)


def review_merge_candidates(
    conn: duckdb.DuckDBPyConnection,
    max_reviews: int = 10,
    auto_execute: bool = False,
) -> list[dict]:
    """Review merge candidates using the LLM.

    Args:
        conn: DuckDB connection
        max_reviews: Max number of pairs to review
        auto_execute: If True, automatically execute approved merges

    Returns:
        List of review results
    """
    candidates = find_merge_candidates(conn)[:max_reviews]

    if not candidates:
        log.info("no_merge_candidates")
        return []

    client = OpenAIClient()
    cache = LLMCache(conn)
    results = []

    for cand in candidates:
        data_a = get_theme_representative_data(conn, cand["theme_id_a"])
        data_b = get_theme_representative_data(conn, cand["theme_id_b"])

        user_prompt = format_merge_review_prompt(
            label_a=cand["label_a"],
            keywords_a=data_a.get("top_keywords", []),
            headlines_a=data_a.get("representative_headlines", []),
            label_b=cand["label_b"],
            keywords_b=data_b.get("top_keywords", []),
            headlines_b=data_b.get("representative_headlines", []),
        )
        full_prompt = MERGE_REVIEW_SYSTEM + "\n\n" + user_prompt

        cached = cache.get(full_prompt)
        if cached:
            response = cached
        else:
            try:
                response = client.chat(MERGE_REVIEW_SYSTEM, user_prompt)
                cache.put(full_prompt, response, model=client.model)
            except Exception as e:
                log.error("merge_review_failed", error=str(e))
                continue

        decision, rationale = _parse_merge_response(response)
        result = {
            **cand,
            "decision": decision,
            "rationale": rationale,
        }
        results.append(result)

        if decision == "MERGE" and auto_execute:
            execute_merge(conn, cand["theme_id_a"], cand["theme_id_b"])
            log.info("merge_executed", a=cand["theme_id_a"], b=cand["theme_id_b"])

    return results


def _parse_merge_response(response: str) -> tuple[str, str]:
    decision = "NO_MERGE"
    rationale = ""
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().upper()
            if "MERGE" in val and "NO" not in val:
                decision = "MERGE"
        elif line.upper().startswith("RATIONALE:"):
            rationale = line.split(":", 1)[1].strip()
    return decision, rationale
