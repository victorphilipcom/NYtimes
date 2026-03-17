"""Build text inputs for the embedding model from article data."""

from __future__ import annotations


def build_embedding_text(
    headline_main: str,
    abstract: str,
    snippet: str,
    lead_paragraph: str,
    keywords_json: str,
    section_name: str = "",
    news_desk: str = "",
) -> str:
    """Build compact text for embedding.

    This uses the pre-built normalized_text when available, but can also
    rebuild from components. The output is what gets sent to the embedding model.
    """
    from nyt_factor_pipeline.utils.text import build_normalized_text

    base = build_normalized_text(
        headline=headline_main,
        abstract=abstract,
        snippet=snippet,
        lead_paragraph=lead_paragraph,
        keywords_json=keywords_json,
    )

    # Optionally prepend section context for better topic separation
    prefix_parts = []
    if section_name:
        prefix_parts.append(f"[{section_name}]")
    if news_desk and news_desk != section_name:
        prefix_parts.append(f"[{news_desk}]")

    if prefix_parts:
        return " ".join(prefix_parts) + " " + base
    return base
