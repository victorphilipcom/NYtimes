"""Extract top keywords from a cluster of article texts using TF-IDF."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_top_keywords(texts: list[str], n: int = 20) -> list[str]:
    """Extract top-n keywords from a set of texts using TF-IDF.

    Returns keywords sorted by aggregate TF-IDF score (highest first).
    """
    if not texts:
        return []

    try:
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    # Sum TF-IDF scores across all documents
    scores = tfidf_matrix.sum(axis=0).A1
    top_indices = scores.argsort()[::-1][:n]
    return [feature_names[i] for i in top_indices]
