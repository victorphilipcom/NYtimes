"""LLM-based RBICS industry mapping for stable themes."""

from __future__ import annotations

import re

import duckdb

from nyt_factor_pipeline.llm.cache import LLMCache
from nyt_factor_pipeline.llm.openai_client import OpenAIClient
from nyt_factor_pipeline.llm.prompts import RBICS_MAPPING_SYSTEM, format_rbics_mapping_prompt
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.themes.theme_store import get_active_themes, get_theme_representative_data

log = get_logger(__name__)


def map_themes_to_rbics(
    conn: duckdb.DuckDBPyConnection,
    max_themes: int = 20,
    min_articles: int = 10,
) -> int:
    """Map stable themes to RBICS industries using the LLM.

    Only maps themes that:
    - Are active and labeled
    - Have no existing RBICS mapping
    - Have at least min_articles

    Returns count of themes mapped.
    """
    # Get themes needing mapping
    themes = get_active_themes(conn)
    unmapped_themes = []
    for t in themes:
        existing = conn.execute(
            "SELECT COUNT(*) FROM theme_rbics_exposure WHERE theme_id = ?",
            [t["theme_id"]],
        ).fetchone()
        if existing and existing[0] > 0:
            continue
        if t.get("llm_labeled_at") is None:
            continue
        unmapped_themes.append(t)

    unmapped_themes = unmapped_themes[:max_themes]

    if not unmapped_themes:
        log.info("no_themes_needing_rbics_mapping")
        return 0

    client = OpenAIClient()
    cache = LLMCache(conn)
    mapped = 0

    for theme in unmapped_themes:
        rep_data = get_theme_representative_data(conn, theme["theme_id"])
        if rep_data["total_articles"] < min_articles:
            continue

        rep_data["label"] = theme["current_label"]
        rep_data["description"] = theme.get("description", "")

        user_prompt = format_rbics_mapping_prompt(rep_data)
        full_prompt = RBICS_MAPPING_SYSTEM + "\n\n" + user_prompt

        cached = cache.get(full_prompt)
        if cached:
            response = cached
        else:
            try:
                response = client.chat(RBICS_MAPPING_SYSTEM, user_prompt)
                cache.put(full_prompt, response, model=client.model)
            except Exception as e:
                log.error("rbics_mapping_failed", theme_id=theme["theme_id"], error=str(e))
                continue

        exposures = _parse_rbics_response(response)
        _save_exposures(conn, theme["theme_id"], exposures)
        mapped += 1

    log.info("rbics_mapping_done", mapped=mapped)
    return mapped


def _parse_rbics_response(response: str) -> list[dict]:
    """Parse RBICS mapping response into list of exposure dicts."""
    exposures = []
    current_direction = "ambiguous"

    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("POSITIVE"):
            current_direction = "positive"
            continue
        elif line.upper().startswith("NEGATIVE"):
            current_direction = "negative"
            continue
        elif line.upper().startswith("AMBIGUOUS"):
            current_direction = "ambiguous"
            continue

        if line.startswith("- "):
            parts = line[2:].split(":", 1)
            industry = parts[0].strip()
            rationale = parts[1].strip() if len(parts) > 1 else ""
            if industry:
                exposures.append({
                    "industry": industry,
                    "direction": current_direction,
                    "rationale": rationale,
                    "strength": 0.7 if current_direction != "ambiguous" else 0.3,
                })

    return exposures


def _save_exposures(
    conn: duckdb.DuckDBPyConnection, theme_id: str, exposures: list[dict]
) -> None:
    """Save RBICS exposures to the database."""
    for exp in exposures:
        # Use industry name as a placeholder RBICS code
        rbics_code = exp["industry"].lower().replace(" ", "_")[:50]
        conn.execute(
            """INSERT INTO theme_rbics_exposure
               (theme_id, rbics_code, rbics_level, exposure_direction,
                exposure_strength, rationale, source)
               VALUES (?, ?, 'industry', ?, ?, ?, 'llm')
               ON CONFLICT (theme_id, rbics_code, exposure_direction)
               DO UPDATE SET exposure_strength = ?, rationale = ?""",
            [
                theme_id, rbics_code, exp["direction"],
                exp["strength"], exp["rationale"],
                exp["strength"], exp["rationale"],
            ],
        )
