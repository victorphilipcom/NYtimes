"""LLM-based theme naming — called only for new/stable themes."""

from __future__ import annotations

import re

import duckdb

from nyt_factor_pipeline.llm.cache import LLMCache
from nyt_factor_pipeline.llm.openai_client import OpenAIClient
from nyt_factor_pipeline.llm.prompts import (
    THEME_NAMING_SYSTEM,
    format_theme_naming_prompt,
)
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.themes.theme_store import (
    get_theme_representative_data,
    get_unlabeled_themes,
    update_theme_label,
)

log = get_logger(__name__)


def label_new_themes(
    conn: duckdb.DuckDBPyConnection,
    max_themes: int = 20,
) -> int:
    """Label unlabeled themes using the LLM. Returns count of themes labeled."""
    unlabeled = get_unlabeled_themes(conn)[:max_themes]

    if not unlabeled:
        log.info("no_unlabeled_themes")
        return 0

    client = OpenAIClient()
    cache = LLMCache(conn)
    labeled = 0

    for theme_info in unlabeled:
        theme_id = theme_info["theme_id"]
        rep_data = get_theme_representative_data(conn, theme_id)

        if not rep_data["top_keywords"] and not rep_data["representative_headlines"]:
            continue

        user_prompt = format_theme_naming_prompt(rep_data)
        full_prompt = THEME_NAMING_SYSTEM + "\n\n" + user_prompt

        # Check cache
        cached = cache.get(full_prompt)
        if cached:
            response = cached
        else:
            try:
                response = client.chat(THEME_NAMING_SYSTEM, user_prompt)
                cache.put(full_prompt, response, model=client.model)
            except Exception as e:
                log.error("theme_labeling_failed", theme_id=theme_id, error=str(e))
                continue

        # Parse response
        label, description, parent = _parse_naming_response(response)

        if label:
            update_theme_label(conn, theme_id, label, description, parent)
            labeled += 1
            log.info("theme_labeled", theme_id=theme_id, label=label)

    return labeled


def _parse_naming_response(response: str) -> tuple[str, str, str]:
    """Parse the structured naming response."""
    label = ""
    description = ""
    parent = ""

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("LABEL:"):
            label = line.split(":", 1)[1].strip()
        elif line.upper().startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PARENT_CATEGORY:"):
            parent = line.split(":", 1)[1].strip()

    return label, description, parent
