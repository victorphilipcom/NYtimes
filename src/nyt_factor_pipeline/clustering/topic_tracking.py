"""Track themes across clustering windows by comparing centroids."""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import date, datetime

import duckdb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def track_themes(
    conn: duckdb.DuckDBPyConnection,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    """Match new clusters to existing themes or create new themes.

    For each unlinked cluster:
    1. Compare cluster centroid to all active theme centroids
    2. If similarity > threshold: link to existing theme, update centroid
    3. If similarity < threshold: create new candidate theme

    Returns summary of actions taken.
    """
    settings = get_settings()
    threshold = settings.theme_similarity_threshold

    # Get unlinked clusters (not yet associated with a theme)
    conditions = []
    params = []
    if start_date:
        conditions.append("cr.window_start >= ?")
        params.append(str(start_date))
    if end_date:
        conditions.append("cr.window_end <= ?")
        params.append(str(end_date))

    where = " AND ".join(["1=1"] + conditions)

    unlinked = conn.execute(
        f"""SELECT cr.cluster_id, cr.window_start, cr.window_end,
                   cr.centroid, cr.article_count, cr.top_keywords_json
            FROM clusters_raw cr
            LEFT JOIN cluster_theme_link ctl
                ON cr.cluster_id = ctl.cluster_id
                AND cr.window_start = ctl.window_start
                AND cr.window_end = ctl.window_end
            WHERE ctl.theme_id IS NULL AND {where}
            ORDER BY cr.window_start""",
        params,
    ).fetchall()

    # Load active themes
    theme_rows = conn.execute(
        "SELECT theme_id, centroid FROM themes WHERE active_flag = true"
    ).fetchall()

    theme_centroids: dict[str, np.ndarray] = {}
    for tid, centroid_blob in theme_rows:
        if centroid_blob:
            theme_centroids[tid] = pickle.loads(centroid_blob)

    stats = {"linked": 0, "new_themes": 0, "skipped": 0}

    for cluster_id, w_start, w_end, centroid_blob, article_count, kw_json in unlinked:
        if not centroid_blob:
            stats["skipped"] += 1
            continue

        cluster_centroid = pickle.loads(centroid_blob)

        # Find best matching theme
        best_theme_id = None
        best_sim = 0.0

        if theme_centroids:
            for tid, t_centroid in theme_centroids.items():
                sim = cosine_similarity(
                    cluster_centroid.reshape(1, -1),
                    t_centroid.reshape(1, -1),
                )[0, 0]
                if sim > best_sim:
                    best_sim = sim
                    best_theme_id = tid

        if best_theme_id and best_sim >= threshold:
            # Link to existing theme
            _link_cluster_to_theme(
                conn, cluster_id, w_start, w_end, best_theme_id, best_sim, "linked"
            )
            # Update theme centroid with exponential moving average
            _update_theme_centroid(
                conn, best_theme_id, cluster_centroid, w_end, theme_centroids
            )
            stats["linked"] += 1
            log.debug(
                "cluster_linked",
                cluster=cluster_id,
                theme=best_theme_id,
                similarity=round(best_sim, 3),
            )
        else:
            # Create new theme
            theme_id = f"theme_{uuid.uuid4().hex[:10]}"
            keywords = []
            try:
                keywords = json.loads(kw_json) if kw_json else []
            except (json.JSONDecodeError, TypeError):
                pass

            auto_label = _generate_auto_label(keywords)
            _create_theme(
                conn, theme_id, cluster_centroid, w_start, w_end, auto_label
            )
            _link_cluster_to_theme(
                conn, cluster_id, w_start, w_end, theme_id, 1.0, "created"
            )
            theme_centroids[theme_id] = cluster_centroid
            stats["new_themes"] += 1
            log.info(
                "new_theme_created",
                theme=theme_id,
                label=auto_label,
                articles=article_count,
            )

    log.info("theme_tracking_done", **stats)
    return stats


def _generate_auto_label(keywords: list[str]) -> str:
    """Generate a temporary label from top keywords (before LLM naming)."""
    if not keywords:
        return "unlabeled"
    return " / ".join(keywords[:3])


def _link_cluster_to_theme(
    conn: duckdb.DuckDBPyConnection,
    cluster_id: str,
    window_start: date,
    window_end: date,
    theme_id: str,
    similarity: float,
    action: str,
) -> None:
    conn.execute(
        """INSERT INTO cluster_theme_link
           (cluster_id, window_start, window_end, theme_id, similarity_score, action)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT (cluster_id, window_start, window_end, theme_id)
           DO UPDATE SET similarity_score = ?, action = ?""",
        [cluster_id, window_start, window_end, theme_id, float(similarity), action, float(similarity), action],
    )


def _create_theme(
    conn: duckdb.DuckDBPyConnection,
    theme_id: str,
    centroid: np.ndarray,
    first_seen: date,
    last_seen: date,
    label: str,
) -> None:
    centroid_blob = pickle.dumps(centroid)
    conn.execute(
        """INSERT INTO themes
           (theme_id, current_label, centroid, first_seen, last_seen, active_flag, metadata_json)
           VALUES (?, ?, ?, ?, ?, true, '{}')""",
        [theme_id, label, centroid_blob, first_seen, last_seen],
    )


def _update_theme_centroid(
    conn: duckdb.DuckDBPyConnection,
    theme_id: str,
    new_centroid: np.ndarray,
    last_seen: date,
    theme_centroids: dict[str, np.ndarray],
    alpha: float = 0.3,
) -> None:
    """Update theme centroid with exponential moving average."""
    old_centroid = theme_centroids[theme_id]
    updated = alpha * new_centroid + (1 - alpha) * old_centroid
    updated = updated / np.linalg.norm(updated)  # re-normalize
    theme_centroids[theme_id] = updated

    centroid_blob = pickle.dumps(updated)
    conn.execute(
        "UPDATE themes SET centroid = ?, last_seen = ? WHERE theme_id = ?",
        [centroid_blob, last_seen, theme_id],
    )
