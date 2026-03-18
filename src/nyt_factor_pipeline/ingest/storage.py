"""Batch article storage for DuckDB — shared by archive and article search ingest."""

from __future__ import annotations

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)

_ARTICLE_COLUMNS = [
    "article_id", "source_api", "web_url", "uri", "pub_date", "year", "month", "day",
    "headline_main", "abstract", "snippet", "lead_paragraph", "source",
    "section_name", "subsection_name", "news_desk", "type_of_material",
    "document_type", "print_section", "print_page", "word_count",
    "byline_original", "keywords_json", "multimedia_json", "normalized_text",
    "importance_score", "macro_relevance_score",
]


def upsert_articles(conn: duckdb.DuckDBPyConnection, articles: list) -> None:
    """Batch-insert articles into DuckDB using a temp table for performance.

    This is orders of magnitude faster than row-by-row inserts because DuckDB
    is an OLAP engine optimised for bulk operations.
    """
    if not articles:
        return

    rows = [
        (
            a.article_id, a.source_api, a.web_url, a.uri, a.pub_date,
            a.year, a.month, a.day, a.headline_main, a.abstract,
            a.snippet, a.lead_paragraph, a.source, a.section_name,
            a.subsection_name, a.news_desk, a.type_of_material,
            a.document_type, a.print_section, a.print_page, a.word_count,
            a.byline_original, a.keywords_json, a.multimedia_json,
            a.normalized_text, a.importance_score, a.macro_relevance_score,
        )
        for a in articles
    ]

    cols = ", ".join(_ARTICLE_COLUMNS)
    placeholders = ", ".join(["?"] * len(_ARTICLE_COLUMNS))

    conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS _staging_articles AS SELECT * FROM articles WHERE 1=0")
    conn.execute("DELETE FROM _staging_articles")

    conn.executemany(
        f"INSERT INTO _staging_articles ({cols}) VALUES ({placeholders})",
        rows,
    )

    conn.execute(f"""
        INSERT INTO articles ({cols})
        SELECT {cols} FROM _staging_articles
        ON CONFLICT (article_id) DO UPDATE SET updated_at = now()
    """)

    conn.execute("DELETE FROM _staging_articles")
