"""Normalize NYT API responses into the unified Article schema.

Handles both Archive API and Article Search API payloads.
"""

from __future__ import annotations

import json
from datetime import datetime

from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.models import Article
from nyt_factor_pipeline.utils.hashing import stable_article_id
from nyt_factor_pipeline.utils.text import build_normalized_text

log = get_logger(__name__)


def normalize_article(raw: dict, source_api: str) -> Article | None:
    """Normalize a single raw article dict from either NYT API into an Article.

    Returns None if the article lacks essential identifiers.
    """
    uri = raw.get("uri", "") or ""
    web_url = raw.get("web_url", "") or ""

    if not uri and not web_url:
        return None

    article_id = stable_article_id(uri, web_url)

    # Headline
    headline_obj = raw.get("headline", {})
    if isinstance(headline_obj, dict):
        headline_main = headline_obj.get("main", "") or ""
    else:
        headline_main = str(headline_obj) if headline_obj else ""

    # Date parsing
    pub_date_str = raw.get("pub_date", "") or ""
    pub_date = _parse_date(pub_date_str)
    year = pub_date.year if pub_date else None
    month = pub_date.month if pub_date else None
    day = pub_date.day if pub_date else None

    # Keywords
    keywords_raw = raw.get("keywords", []) or []
    keywords_json = json.dumps(keywords_raw)

    # Multimedia
    multimedia_raw = raw.get("multimedia", []) or []
    multimedia_json = json.dumps(multimedia_raw)

    # Byline
    byline_obj = raw.get("byline", {})
    if isinstance(byline_obj, dict):
        byline_original = byline_obj.get("original", "") or ""
    elif isinstance(byline_obj, str):
        byline_original = byline_obj
    else:
        byline_original = ""

    abstract = raw.get("abstract", "") or ""
    snippet = raw.get("snippet", "") or ""
    lead_paragraph = raw.get("lead_paragraph", "") or ""

    normalized_text = build_normalized_text(
        headline=headline_main,
        abstract=abstract,
        snippet=snippet,
        lead_paragraph=lead_paragraph,
        keywords_json=keywords_json,
    )

    return Article(
        article_id=article_id,
        source_api=source_api,
        web_url=web_url,
        uri=uri,
        pub_date=pub_date,
        year=year,
        month=month,
        day=day,
        headline_main=headline_main,
        abstract=abstract,
        snippet=snippet,
        lead_paragraph=lead_paragraph,
        source=raw.get("source", "") or "",
        section_name=raw.get("section_name", "") or "",
        subsection_name=raw.get("subsection_name", "") or "",
        news_desk=raw.get("news_desk", "") or "",
        type_of_material=raw.get("type_of_material", "") or "",
        document_type=raw.get("document_type", "") or "",
        print_section=raw.get("print_section", "") or "",
        print_page=str(raw.get("print_page", "") or ""),
        word_count=int(raw.get("word_count", 0) or 0),
        byline_original=byline_original,
        keywords_json=keywords_json,
        multimedia_json=multimedia_json,
        normalized_text=normalized_text,
    )


def normalize_batch(raw_articles: list[dict], source_api: str) -> list[Article]:
    """Normalize a batch of raw articles, skipping any that fail."""
    articles = []
    for raw in raw_articles:
        try:
            article = normalize_article(raw, source_api)
            if article:
                articles.append(article)
        except Exception as e:
            log.warning(
                "normalize_failed",
                error=str(e),
                uri=raw.get("uri", "unknown"),
            )
    return articles


def _parse_date(s: str) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S+0000", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.replace("+0000", "+00:00")[:25], fmt[:25])
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
