"""Article filtering for embedding/clustering — remove noise before expensive ops."""

from __future__ import annotations

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)

# Material types to exclude from embedding/clustering
EXCLUDE_MATERIAL_TYPES: set[str] = {
    "Letter",
    "Obituary",
    "Obituary (Obit)",
    "Correction",
    "Paid Notice",
    "Review",
    "Schedule",
    "List",
    "Caption",
    "Summary",
    "Recipe",
    "Interactive Feature",
    "Paid Death Notice",
    "Marriage Announcement",
}

DEFAULT_MIN_IMPORTANCE = 0.35
DEFAULT_MIN_WORD_COUNT = 150


def get_embeddable_articles(
    conn: duckdb.DuckDBPyConnection,
    min_importance: float = DEFAULT_MIN_IMPORTANCE,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
    start_date: str | None = None,
    end_date: str | None = None,
    exclude_already_embedded: bool = True,
) -> list[dict]:
    """Get articles that pass quality filters for embedding.

    Returns list of dicts with article_id, normalized_text, pub_date, importance_score.
    """
    excluded_types = ", ".join(f"'{t}'" for t in EXCLUDE_MATERIAL_TYPES)

    conditions = [
        f"type_of_material NOT IN ({excluded_types})",
        f"importance_score >= {min_importance}",
        f"word_count >= {min_word_count}",
        "normalized_text IS NOT NULL",
        "LENGTH(normalized_text) > 50",
    ]

    if start_date:
        conditions.append(f"pub_date >= '{start_date}'")
    if end_date:
        conditions.append(f"pub_date <= '{end_date}'")
    if exclude_already_embedded:
        conditions.append(
            "article_id NOT IN (SELECT article_id FROM article_embeddings)"
        )

    where = " AND ".join(conditions)
    query = f"""
        SELECT article_id, normalized_text, pub_date, importance_score
        FROM articles
        WHERE {where}
        ORDER BY pub_date DESC
    """

    rows = conn.execute(query).fetchall()
    return [
        {
            "article_id": r[0],
            "normalized_text": r[1],
            "pub_date": r[2],
            "importance_score": r[3],
        }
        for r in rows
    ]


def compute_macro_relevance(conn: duckdb.DuckDBPyConnection) -> int:
    """Set macro_relevance_score based on filtering criteria.

    This is a binary 0/1 score indicating whether the article passes
    the quality/relevance filter for the embedding pipeline.
    """
    excluded_types = ", ".join(f"'{t}'" for t in EXCLUDE_MATERIAL_TYPES)

    conn.execute(
        f"""UPDATE articles SET macro_relevance_score = CASE
            WHEN type_of_material IN ({excluded_types}) THEN 0.0
            WHEN importance_score < {DEFAULT_MIN_IMPORTANCE} THEN 0.0
            WHEN word_count < {DEFAULT_MIN_WORD_COUNT} THEN 0.0
            WHEN normalized_text IS NULL OR LENGTH(normalized_text) < 50 THEN 0.0
            ELSE 1.0
        END,
        updated_at = current_timestamp
        WHERE macro_relevance_score = 0"""
    )

    result = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE macro_relevance_score > 0"
    ).fetchone()
    return result[0] if result else 0
