"""Export DuckDB data to Supabase/PostgreSQL via DATABASE_URL."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)

# PostgreSQL schema — mirrors the DuckDB schema but uses PG types.
_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    source_api TEXT,
    web_url TEXT,
    uri TEXT,
    pub_date TIMESTAMPTZ,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    headline_main TEXT,
    abstract TEXT,
    snippet TEXT,
    lead_paragraph TEXT,
    source TEXT,
    section_name TEXT,
    subsection_name TEXT,
    news_desk TEXT,
    type_of_material TEXT,
    document_type TEXT,
    print_section TEXT,
    print_page TEXT,
    word_count INTEGER,
    byline_original TEXT,
    keywords_json JSONB DEFAULT '[]',
    multimedia_json JSONB DEFAULT '[]',
    normalized_text TEXT,
    importance_score DOUBLE PRECISION,
    macro_relevance_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS themes (
    theme_id TEXT PRIMARY KEY,
    current_label TEXT,
    description TEXT,
    parent_theme_id TEXT,
    first_seen DATE,
    last_seen DATE,
    active_flag BOOLEAN DEFAULT true,
    llm_labeled_at TIMESTAMPTZ,
    metadata_json JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS theme_timeseries (
    theme_id TEXT,
    date DATE,
    article_count INTEGER,
    weighted_article_count DOUBLE PRECISION,
    intensity DOUBLE PRECISION,
    burst_zscore DOUBLE PRECISION,
    PRIMARY KEY (theme_id, date)
);

CREATE TABLE IF NOT EXISTS cluster_theme_link (
    cluster_id TEXT,
    window_start DATE,
    window_end DATE,
    theme_id TEXT,
    similarity_score DOUBLE PRECISION,
    action TEXT,
    PRIMARY KEY (cluster_id, window_start, window_end, theme_id)
);

CREATE TABLE IF NOT EXISTS theme_rbics_exposure (
    theme_id TEXT,
    rbics_code TEXT,
    rbics_level TEXT,
    exposure_direction TEXT,
    exposure_strength DOUBLE PRECISION,
    rationale TEXT,
    source TEXT,
    PRIMARY KEY (theme_id, rbics_code, exposure_direction)
);

CREATE TABLE IF NOT EXISTS companies (
    company_id TEXT PRIMARY KEY,
    ticker TEXT,
    company_name TEXT,
    rbics_code TEXT,
    rbics_name TEXT,
    metadata_json JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS company_theme_exposure (
    company_id TEXT,
    theme_id TEXT,
    exposure_strength DOUBLE PRECISION,
    exposure_direction TEXT,
    source TEXT,
    PRIMARY KEY (company_id, theme_id)
);

CREATE TABLE IF NOT EXISTS company_theme_scores (
    company_id TEXT,
    date DATE,
    theme_id TEXT,
    score DOUBLE PRECISION,
    PRIMARY KEY (company_id, date, theme_id)
);

CREATE TABLE IF NOT EXISTS sync_metadata (
    table_name TEXT PRIMARY KEY,
    last_synced_at TIMESTAMPTZ DEFAULT now(),
    rows_synced INTEGER DEFAULT 0
);
"""

# Tables to sync and their DuckDB queries.  Ordered so that foreign-key-like
# references resolve (themes before theme_timeseries, etc.).
_SYNC_TABLES: list[dict[str, Any]] = [
    {
        "name": "articles",
        "query": "SELECT article_id, source_api, web_url, uri, pub_date, year, month, day, "
                 "headline_main, abstract, snippet, lead_paragraph, source, section_name, "
                 "subsection_name, news_desk, type_of_material, document_type, print_section, "
                 "print_page, word_count, byline_original, keywords_json, multimedia_json, "
                 "normalized_text, importance_score, macro_relevance_score, created_at, updated_at "
                 "FROM articles",
        "pk": "article_id",
        "columns": [
            "article_id", "source_api", "web_url", "uri", "pub_date", "year", "month", "day",
            "headline_main", "abstract", "snippet", "lead_paragraph", "source", "section_name",
            "subsection_name", "news_desk", "type_of_material", "document_type", "print_section",
            "print_page", "word_count", "byline_original", "keywords_json", "multimedia_json",
            "normalized_text", "importance_score", "macro_relevance_score", "created_at",
            "updated_at",
        ],
    },
    {
        "name": "themes",
        "query": "SELECT theme_id, current_label, description, parent_theme_id, "
                 "first_seen, last_seen, active_flag, llm_labeled_at, metadata_json FROM themes",
        "pk": "theme_id",
        "columns": [
            "theme_id", "current_label", "description", "parent_theme_id",
            "first_seen", "last_seen", "active_flag", "llm_labeled_at", "metadata_json",
        ],
    },
    {
        "name": "theme_timeseries",
        "query": "SELECT theme_id, date, article_count, weighted_article_count, intensity, "
                 "burst_zscore FROM theme_timeseries",
        "pk": "(theme_id, date)",
        "columns": [
            "theme_id", "date", "article_count", "weighted_article_count", "intensity",
            "burst_zscore",
        ],
    },
    {
        "name": "cluster_theme_link",
        "query": "SELECT cluster_id, window_start, window_end, theme_id, similarity_score, "
                 "action FROM cluster_theme_link",
        "pk": "(cluster_id, window_start, window_end, theme_id)",
        "columns": [
            "cluster_id", "window_start", "window_end", "theme_id", "similarity_score", "action",
        ],
    },
    {
        "name": "theme_rbics_exposure",
        "query": "SELECT theme_id, rbics_code, rbics_level, exposure_direction, "
                 "exposure_strength, rationale, source FROM theme_rbics_exposure",
        "pk": "(theme_id, rbics_code, exposure_direction)",
        "columns": [
            "theme_id", "rbics_code", "rbics_level", "exposure_direction",
            "exposure_strength", "rationale", "source",
        ],
    },
    {
        "name": "companies",
        "query": "SELECT company_id, ticker, company_name, rbics_code, rbics_name, "
                 "metadata_json FROM companies",
        "pk": "company_id",
        "columns": [
            "company_id", "ticker", "company_name", "rbics_code", "rbics_name", "metadata_json",
        ],
    },
    {
        "name": "company_theme_exposure",
        "query": "SELECT company_id, theme_id, exposure_strength, exposure_direction, "
                 "source FROM company_theme_exposure",
        "pk": "(company_id, theme_id)",
        "columns": [
            "company_id", "theme_id", "exposure_strength", "exposure_direction", "source",
        ],
    },
    {
        "name": "company_theme_scores",
        "query": "SELECT company_id, date, theme_id, score FROM company_theme_scores",
        "pk": "(company_id, date, theme_id)",
        "columns": ["company_id", "date", "theme_id", "score"],
    },
]

BATCH_SIZE = 500


def _coerce_value(val: Any) -> Any:
    """Coerce DuckDB-native types to psycopg2-friendly ones."""
    if val is None:
        return None
    if isinstance(val, bytes):
        # Skip blobs (centroids) — not synced to PG
        return None
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return val


def _get_pg_connection(database_url: str):
    """Create a psycopg2 connection from a DATABASE_URL."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 is required for Supabase sync. "
            "Install it with: pip install psycopg2-binary"
        )
    return psycopg2.connect(database_url)


def init_pg_schema(database_url: str) -> None:
    """Create PostgreSQL tables if they don't exist."""
    pg = _get_pg_connection(database_url)
    try:
        with pg.cursor() as cur:
            cur.execute(_PG_SCHEMA)
        pg.commit()
        log.info("pg_schema_initialized")
    finally:
        pg.close()


def sync_table(
    duck_conn: duckdb.DuckDBPyConnection,
    database_url: str,
    table_spec: dict[str, Any],
    *,
    incremental: bool = True,
    since: datetime | None = None,
) -> int:
    """Sync a single table from DuckDB to PostgreSQL using upsert.

    Returns number of rows synced.
    """
    pg = _get_pg_connection(database_url)
    name = table_spec["name"]
    columns = table_spec["columns"]
    query = table_spec["query"]
    pk = table_spec["pk"]

    # For incremental sync, only push rows newer than last sync
    if incremental and since is None:
        try:
            with pg.cursor() as cur:
                cur.execute(
                    "SELECT last_synced_at FROM sync_metadata WHERE table_name = %s", (name,)
                )
                row = cur.fetchone()
                if row:
                    since = row[0]
        except Exception:
            pass  # Table may not exist yet

    # Add time filter for tables with timestamp columns
    if since and name == "articles":
        query += f" WHERE updated_at > '{since.isoformat()}'"
    elif since and name == "theme_timeseries":
        query += f" WHERE date > '{since.date().isoformat() if hasattr(since, 'date') else since}'"

    rows = duck_conn.execute(query).fetchall()
    if not rows:
        log.info("sync_skip_empty", table=name)
        pg.close()
        return 0

    # Build upsert SQL
    col_list = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    # ON CONFLICT ... DO UPDATE for all non-PK columns
    pk_cols = [c.strip() for c in pk.strip("()").split(",")]
    update_cols = [c for c in columns if c.strip() not in pk_cols]
    if update_cols:
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        upsert_sql = (
            f"INSERT INTO {name} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {update_clause}"
        )
    else:
        upsert_sql = (
            f"INSERT INTO {name} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING"
        )

    total = 0
    try:
        with pg.cursor() as cur:
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                coerced = [tuple(_coerce_value(v) for v in row) for row in batch]
                cur.executemany(upsert_sql, coerced)
                total += len(batch)

            # Update sync metadata
            cur.execute(
                "INSERT INTO sync_metadata (table_name, last_synced_at, rows_synced) "
                "VALUES (%s, now(), %s) "
                "ON CONFLICT (table_name) DO UPDATE SET last_synced_at = now(), rows_synced = %s",
                (name, total, total),
            )
        pg.commit()
        log.info("sync_table_done", table=name, rows=total)
    finally:
        pg.close()

    return total


def sync_all(
    duck_conn: duckdb.DuckDBPyConnection,
    database_url: str,
    *,
    incremental: bool = True,
    tables: list[str] | None = None,
) -> dict[str, int]:
    """Sync all (or selected) tables from DuckDB to PostgreSQL.

    Returns dict of table_name -> rows synced.
    """
    init_pg_schema(database_url)

    results: dict[str, int] = {}
    for spec in _SYNC_TABLES:
        if tables and spec["name"] not in tables:
            continue
        count = sync_table(duck_conn, database_url, spec, incremental=incremental)
        results[spec["name"]] = count

    return results
