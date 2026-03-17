"""Weekly clustering of article embeddings using HDBSCAN."""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import date, datetime

import duckdb
import numpy as np

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.clustering.keywords import extract_top_keywords
from nyt_factor_pipeline.clustering.representatives import select_representatives
from nyt_factor_pipeline.embeddings.cache import get_embeddings_for_articles
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.utils.dates import week_windows

log = get_logger(__name__)


def cluster_window(
    conn: duckdb.DuckDBPyConnection,
    window_start: date,
    window_end: date,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> list[dict]:
    """Cluster embedded articles within a date window using HDBSCAN.

    Returns list of cluster info dicts.
    """
    settings = get_settings()
    min_cluster_size = min_cluster_size or settings.clustering_min_cluster_size
    min_samples = min_samples or settings.clustering_min_samples

    # Get articles with embeddings in this window
    rows = conn.execute(
        """SELECT a.article_id, a.headline_main, a.normalized_text, a.importance_score
           FROM articles a
           JOIN article_embeddings ae ON a.article_id = ae.article_id
           WHERE a.pub_date >= ? AND a.pub_date < ?
             AND a.macro_relevance_score > 0
           ORDER BY a.pub_date""",
        [str(window_start), str(window_end)],
    ).fetchall()

    if len(rows) < min_cluster_size:
        log.info(
            "insufficient_articles_for_clustering",
            window=f"{window_start}_{window_end}",
            count=len(rows),
        )
        return []

    article_ids = [r[0] for r in rows]
    headlines = {r[0]: r[1] for r in rows}
    texts = {r[0]: r[2] for r in rows}
    scores = {r[0]: r[3] for r in rows}

    # Load embeddings
    embeddings_map = get_embeddings_for_articles(conn, article_ids)
    valid_ids = [aid for aid in article_ids if aid in embeddings_map]

    if len(valid_ids) < min_cluster_size:
        return []

    X = np.array([embeddings_map[aid] for aid in valid_ids])

    # Run HDBSCAN
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    # Build cluster results
    clusters = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    for label in sorted(unique_labels):
        mask = labels == label
        cluster_ids = [valid_ids[i] for i in range(len(valid_ids)) if mask[i]]
        cluster_embeddings = X[mask]

        centroid = cluster_embeddings.mean(axis=0)
        cluster_id = f"c_{window_start}_{label}_{uuid.uuid4().hex[:6]}"

        # Top keywords
        cluster_texts = [texts[aid] for aid in cluster_ids if aid in texts]
        top_keywords = extract_top_keywords(cluster_texts, n=20)

        # Representative articles (closest to centroid)
        rep_ids = select_representatives(
            cluster_ids, cluster_embeddings, centroid, n=10
        )

        cluster_info = {
            "cluster_id": cluster_id,
            "window_start": window_start,
            "window_end": window_end,
            "article_count": len(cluster_ids),
            "centroid": centroid,
            "top_keywords": top_keywords,
            "representative_article_ids": rep_ids,
            "article_ids": cluster_ids,
            "membership_scores": {
                aid: float(1.0 - (np.linalg.norm(embeddings_map[aid] - centroid)))
                for aid in cluster_ids
            },
        }
        clusters.append(cluster_info)

    log.info(
        "clustering_done",
        window=f"{window_start}_{window_end}",
        articles=len(valid_ids),
        clusters=len(clusters),
        noise=int((labels == -1).sum()),
    )

    # Persist clusters
    _save_clusters(conn, clusters, window_start, window_end)

    return clusters


def cluster_date_range(
    conn: duckdb.DuckDBPyConnection,
    start_date: date,
    end_date: date,
    window_type: str = "weekly",
) -> list[dict]:
    """Cluster articles over a date range using rolling windows."""
    if window_type == "weekly":
        windows = week_windows(start_date, end_date)
    else:
        from nyt_factor_pipeline.utils.dates import date_windows
        windows = date_windows(start_date, end_date, window_days=30)

    all_clusters = []
    for w_start, w_end in windows:
        clusters = cluster_window(conn, w_start, w_end)
        all_clusters.extend(clusters)

    return all_clusters


def _save_clusters(
    conn: duckdb.DuckDBPyConnection,
    clusters: list[dict],
    window_start: date,
    window_end: date,
) -> None:
    """Persist cluster results to DuckDB."""
    now = datetime.utcnow()

    for c in clusters:
        centroid_blob = pickle.dumps(c["centroid"])
        conn.execute(
            """INSERT INTO clusters_raw
               (cluster_id, window_start, window_end, article_count, centroid,
                top_keywords_json, representative_article_ids, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (cluster_id, window_start, window_end)
               DO UPDATE SET article_count = ?, centroid = ?,
                             top_keywords_json = ?, representative_article_ids = ?""",
            [
                c["cluster_id"],
                window_start,
                window_end,
                c["article_count"],
                centroid_blob,
                json.dumps(c["top_keywords"]),
                json.dumps(c["representative_article_ids"]),
                now,
                c["article_count"],
                centroid_blob,
                json.dumps(c["top_keywords"]),
                json.dumps(c["representative_article_ids"]),
            ],
        )

        # Save membership
        for aid in c["article_ids"]:
            mscore = c["membership_scores"].get(aid, 0.0)
            conn.execute(
                """INSERT INTO article_cluster_membership
                   (article_id, cluster_id, window_start, window_end, membership_score)
                   VALUES (?, ?, ?, ?, ?)""",
                [aid, c["cluster_id"], window_start, window_end, mscore],
            )
