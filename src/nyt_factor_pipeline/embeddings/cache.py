"""Embedding cache — avoid recomputing embeddings for already-embedded articles."""

from __future__ import annotations

import pickle

import duckdb
import numpy as np


def get_cached_embedding(conn: duckdb.DuckDBPyConnection, article_id: str) -> np.ndarray | None:
    """Retrieve a cached embedding for an article."""
    result = conn.execute(
        "SELECT embedding FROM article_embeddings WHERE article_id = ?",
        [article_id],
    ).fetchone()
    if result and result[0]:
        return pickle.loads(result[0])
    return None


def get_embeddings_for_articles(
    conn: duckdb.DuckDBPyConnection, article_ids: list[str]
) -> dict[str, np.ndarray]:
    """Retrieve cached embeddings for a list of article IDs."""
    if not article_ids:
        return {}
    placeholders = ", ".join(["?"] * len(article_ids))
    rows = conn.execute(
        f"SELECT article_id, embedding FROM article_embeddings WHERE article_id IN ({placeholders})",
        article_ids,
    ).fetchall()
    result = {}
    for aid, emb_blob in rows:
        if emb_blob:
            result[aid] = pickle.loads(emb_blob)
    return result


def count_embedded(conn: duckdb.DuckDBPyConnection) -> int:
    result = conn.execute("SELECT COUNT(*) FROM article_embeddings").fetchone()
    return result[0] if result else 0
