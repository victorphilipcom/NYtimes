"""Detect theme merge candidates by comparing theme centroids."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def find_merge_candidates(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Find pairs of active themes that are close enough to potentially merge.

    Returns list of dicts with theme_id_a, theme_id_b, similarity, labels.
    """
    settings = get_settings()
    threshold = settings.theme_merge_threshold

    rows = conn.execute(
        "SELECT theme_id, current_label, centroid FROM themes WHERE active_flag = true"
    ).fetchall()

    themes = []
    for tid, label, centroid_blob in rows:
        if centroid_blob:
            themes.append((tid, label, pickle.loads(centroid_blob)))

    if len(themes) < 2:
        return []

    candidates = []
    for i in range(len(themes)):
        for j in range(i + 1, len(themes)):
            sim = cosine_similarity(
                themes[i][2].reshape(1, -1),
                themes[j][2].reshape(1, -1),
            )[0, 0]
            if sim >= threshold:
                candidates.append({
                    "theme_id_a": themes[i][0],
                    "label_a": themes[i][1],
                    "theme_id_b": themes[j][0],
                    "label_b": themes[j][1],
                    "similarity": round(float(sim), 4),
                })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    log.info("merge_candidates_found", count=len(candidates))
    return candidates


def execute_merge(
    conn: duckdb.DuckDBPyConnection,
    keep_theme_id: str,
    merge_theme_id: str,
) -> None:
    """Merge merge_theme_id into keep_theme_id.

    - Repoints cluster links
    - Deactivates merged theme
    - Updates centroid as average
    """
    # Update cluster links
    conn.execute(
        "UPDATE cluster_theme_link SET theme_id = ? WHERE theme_id = ?",
        [keep_theme_id, merge_theme_id],
    )

    # Average centroids
    rows = conn.execute(
        "SELECT theme_id, centroid FROM themes WHERE theme_id IN (?, ?)",
        [keep_theme_id, merge_theme_id],
    ).fetchall()

    centroids = {}
    for tid, blob in rows:
        if blob:
            centroids[tid] = pickle.loads(blob)

    if keep_theme_id in centroids and merge_theme_id in centroids:
        new_centroid = (centroids[keep_theme_id] + centroids[merge_theme_id]) / 2
        new_centroid = new_centroid / np.linalg.norm(new_centroid)
        conn.execute(
            "UPDATE themes SET centroid = ? WHERE theme_id = ?",
            [pickle.dumps(new_centroid), keep_theme_id],
        )

    # Deactivate merged theme
    conn.execute(
        "UPDATE themes SET active_flag = false, parent_theme_id = ? WHERE theme_id = ?",
        [keep_theme_id, merge_theme_id],
    )

    log.info("themes_merged", kept=keep_theme_id, merged=merge_theme_id)
