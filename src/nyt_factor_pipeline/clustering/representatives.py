"""Select representative articles from a cluster."""

from __future__ import annotations

import numpy as np


def select_representatives(
    article_ids: list[str],
    embeddings: np.ndarray,
    centroid: np.ndarray,
    n: int = 10,
) -> list[str]:
    """Select the n articles closest to the cluster centroid.

    Args:
        article_ids: List of article IDs in the cluster
        embeddings: (k, dim) array of embeddings for the cluster
        centroid: (dim,) centroid vector
        n: Number of representatives to select

    Returns:
        List of article IDs closest to the centroid
    """
    if len(article_ids) <= n:
        return article_ids

    distances = np.linalg.norm(embeddings - centroid, axis=1)
    top_indices = np.argsort(distances)[:n]
    return [article_ids[i] for i in top_indices]
