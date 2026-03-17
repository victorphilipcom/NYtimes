"""DuckDB database management — schema creation and connection helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    source_api TEXT,
    web_url TEXT,
    uri TEXT,
    pub_date TIMESTAMP,
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
    keywords_json JSON,
    multimedia_json JSON,
    normalized_text TEXT,
    importance_score DOUBLE,
    macro_relevance_score DOUBLE,
    created_at TIMESTAMP DEFAULT current_timestamp,
    updated_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS article_embeddings (
    article_id TEXT PRIMARY KEY,
    embedding BLOB,
    embedding_model TEXT,
    embedded_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS clusters_raw (
    cluster_id TEXT,
    window_start DATE,
    window_end DATE,
    article_count INTEGER,
    centroid BLOB,
    top_keywords_json JSON,
    representative_article_ids JSON,
    created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (cluster_id, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS article_cluster_membership (
    article_id TEXT,
    cluster_id TEXT,
    window_start DATE,
    window_end DATE,
    membership_score DOUBLE
);

CREATE TABLE IF NOT EXISTS themes (
    theme_id TEXT PRIMARY KEY,
    current_label TEXT,
    description TEXT,
    parent_theme_id TEXT,
    centroid BLOB,
    first_seen DATE,
    last_seen DATE,
    active_flag BOOLEAN DEFAULT true,
    llm_labeled_at TIMESTAMP,
    metadata_json JSON
);

CREATE TABLE IF NOT EXISTS cluster_theme_link (
    cluster_id TEXT,
    window_start DATE,
    window_end DATE,
    theme_id TEXT,
    similarity_score DOUBLE,
    action TEXT,
    PRIMARY KEY (cluster_id, window_start, window_end, theme_id)
);

CREATE TABLE IF NOT EXISTS theme_timeseries (
    theme_id TEXT,
    date DATE,
    article_count INTEGER,
    weighted_article_count DOUBLE,
    intensity DOUBLE,
    burst_zscore DOUBLE,
    PRIMARY KEY (theme_id, date)
);

CREATE TABLE IF NOT EXISTS theme_rbics_exposure (
    theme_id TEXT,
    rbics_code TEXT,
    rbics_level TEXT,
    exposure_direction TEXT,
    exposure_strength DOUBLE,
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
    metadata_json JSON
);

CREATE TABLE IF NOT EXISTS company_theme_exposure (
    company_id TEXT,
    theme_id TEXT,
    exposure_strength DOUBLE,
    exposure_direction TEXT,
    source TEXT,
    PRIMARY KEY (company_id, theme_id)
);

CREATE TABLE IF NOT EXISTS company_theme_scores (
    company_id TEXT,
    date DATE,
    theme_id TEXT,
    score DOUBLE,
    PRIMARY KEY (company_id, date, theme_id)
);

CREATE TABLE IF NOT EXISTS api_request_log (
    request_id TEXT PRIMARY KEY,
    api_name TEXT,
    requested_at TIMESTAMP,
    url TEXT,
    params_json JSON,
    response_status INTEGER,
    retry_count INTEGER,
    elapsed_ms INTEGER
);

CREATE TABLE IF NOT EXISTS ingest_checkpoints (
    source_api TEXT,
    checkpoint_key TEXT,
    checkpoint_value_json JSON,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (source_api, checkpoint_key)
);

CREATE TABLE IF NOT EXISTS rate_limit_state (
    api_name TEXT,
    date DATE,
    requests_today INTEGER DEFAULT 0,
    minute_bucket TEXT,
    requests_this_minute INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (api_name, date, minute_bucket)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT PRIMARY KEY,
    prompt_text TEXT,
    response_text TEXT,
    model TEXT,
    tokens_used INTEGER,
    created_at TIMESTAMP DEFAULT current_timestamp
);
"""


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection. Creates parent directories if needed."""
    if db_path is None:
        db_path = get_settings().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all tables if they don't exist."""
    for statement in _SCHEMA_SQL.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
    log.info("schema_initialized")


def init_db(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open connection and ensure schema exists."""
    conn = get_connection(db_path)
    init_schema(conn)
    return conn
