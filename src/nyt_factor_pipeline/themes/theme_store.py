"""Theme CRUD operations and queries."""

from __future__ import annotations

import json
import pickle
from datetime import date

import duckdb
import numpy as np

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def get_active_themes(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Get all active themes with metadata."""
    rows = conn.execute(
        """SELECT theme_id, current_label, description, first_seen, last_seen,
                  llm_labeled_at, metadata_json
           FROM themes WHERE active_flag = true
           ORDER BY last_seen DESC"""
    ).fetchall()

    return [
        {
            "theme_id": r[0],
            "current_label": r[1],
            "description": r[2],
            "first_seen": r[3],
            "last_seen": r[4],
            "llm_labeled_at": r[5],
            "metadata": json.loads(r[6]) if r[6] else {},
        }
        for r in rows
    ]


def get_theme_details(conn: duckdb.DuckDBPyConnection, theme_id: str) -> dict | None:
    """Get full details for a theme including linked clusters."""
    row = conn.execute(
        """SELECT theme_id, current_label, description, parent_theme_id,
                  first_seen, last_seen, active_flag, llm_labeled_at, metadata_json
           FROM themes WHERE theme_id = ?""",
        [theme_id],
    ).fetchone()

    if not row:
        return None

    # Get linked clusters
    clusters = conn.execute(
        """SELECT cluster_id, window_start, window_end, similarity_score, action
           FROM cluster_theme_link
           WHERE theme_id = ?
           ORDER BY window_start""",
        [theme_id],
    ).fetchall()

    # Get article count
    article_count = conn.execute(
        """SELECT COUNT(DISTINCT acm.article_id)
           FROM article_cluster_membership acm
           JOIN cluster_theme_link ctl ON acm.cluster_id = ctl.cluster_id
               AND acm.window_start = ctl.window_start
               AND acm.window_end = ctl.window_end
           WHERE ctl.theme_id = ?""",
        [theme_id],
    ).fetchone()

    return {
        "theme_id": row[0],
        "current_label": row[1],
        "description": row[2],
        "parent_theme_id": row[3],
        "first_seen": row[4],
        "last_seen": row[5],
        "active_flag": row[6],
        "llm_labeled_at": row[7],
        "metadata": json.loads(row[8]) if row[8] else {},
        "cluster_count": len(clusters),
        "article_count": article_count[0] if article_count else 0,
        "clusters": [
            {
                "cluster_id": c[0],
                "window_start": c[1],
                "window_end": c[2],
                "similarity": c[3],
                "action": c[4],
            }
            for c in clusters
        ],
    }


def get_unlabeled_themes(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Get active themes that have no LLM label yet."""
    rows = conn.execute(
        """SELECT t.theme_id, t.current_label, t.first_seen, t.last_seen,
                  COUNT(DISTINCT ctl.cluster_id) as cluster_count
           FROM themes t
           LEFT JOIN cluster_theme_link ctl ON t.theme_id = ctl.theme_id
           WHERE t.active_flag = true AND t.llm_labeled_at IS NULL
           GROUP BY t.theme_id, t.current_label, t.first_seen, t.last_seen
           HAVING cluster_count >= 1
           ORDER BY cluster_count DESC"""
    ).fetchall()

    return [
        {
            "theme_id": r[0],
            "current_label": r[1],
            "first_seen": r[2],
            "last_seen": r[3],
            "cluster_count": r[4],
        }
        for r in rows
    ]


def get_theme_representative_data(
    conn: duckdb.DuckDBPyConnection, theme_id: str
) -> dict:
    """Get representative data for a theme for LLM labeling.

    Returns top keywords, representative headlines, and abstracts.
    """
    # Get all cluster keywords for this theme
    rows = conn.execute(
        """SELECT cr.top_keywords_json, cr.representative_article_ids
           FROM clusters_raw cr
           JOIN cluster_theme_link ctl ON cr.cluster_id = ctl.cluster_id
               AND cr.window_start = ctl.window_start
               AND cr.window_end = ctl.window_end
           WHERE ctl.theme_id = ?""",
        [theme_id],
    ).fetchall()

    all_keywords: list[str] = []
    all_rep_ids: list[str] = []
    for kw_json, rep_json in rows:
        try:
            all_keywords.extend(json.loads(kw_json) if kw_json else [])
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            all_rep_ids.extend(json.loads(rep_json) if rep_json else [])
        except (json.JSONDecodeError, TypeError):
            pass

    # Deduplicate keywords, keep top 20
    seen = set()
    unique_kw = []
    for kw in all_keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique_kw.append(kw)
    top_keywords = unique_kw[:20]

    # Get representative articles
    unique_rep_ids = list(dict.fromkeys(all_rep_ids))[:15]
    articles = []
    if unique_rep_ids:
        placeholders = ", ".join(["?"] * len(unique_rep_ids))
        art_rows = conn.execute(
            f"""SELECT headline_main, abstract, pub_date
                FROM articles WHERE article_id IN ({placeholders})
                ORDER BY pub_date DESC""",
            unique_rep_ids,
        ).fetchall()
        articles = [
            {"headline": r[0], "abstract": r[1], "pub_date": r[2]}
            for r in art_rows
        ]

    # Get date range and count
    meta = conn.execute(
        """SELECT MIN(ctl.window_start), MAX(ctl.window_end),
                  COUNT(DISTINCT acm.article_id)
           FROM cluster_theme_link ctl
           LEFT JOIN article_cluster_membership acm ON ctl.cluster_id = acm.cluster_id
               AND ctl.window_start = acm.window_start AND ctl.window_end = acm.window_end
           WHERE ctl.theme_id = ?""",
        [theme_id],
    ).fetchone()

    return {
        "theme_id": theme_id,
        "top_keywords": top_keywords,
        "representative_headlines": [a["headline"] for a in articles[:10]],
        "representative_abstracts": [a["abstract"] for a in articles[:5] if a["abstract"]],
        "date_range_start": meta[0] if meta else None,
        "date_range_end": meta[1] if meta else None,
        "total_articles": meta[2] if meta else 0,
    }


def update_theme_label(
    conn: duckdb.DuckDBPyConnection,
    theme_id: str,
    label: str,
    description: str,
    parent_category: str = "",
) -> None:
    """Update a theme with an LLM-generated label."""
    from datetime import datetime

    metadata = {"parent_category": parent_category} if parent_category else {}
    conn.execute(
        """UPDATE themes SET current_label = ?, description = ?,
                  llm_labeled_at = ?, metadata_json = ?
           WHERE theme_id = ?""",
        [label, description, datetime.utcnow(), json.dumps(metadata), theme_id],
    )
    log.info("theme_labeled", theme_id=theme_id, label=label)
