"""Text normalization and deduplication utilities."""

from __future__ import annotations

import json
import re


def build_normalized_text(
    headline: str,
    abstract: str,
    snippet: str,
    lead_paragraph: str,
    keywords_json: str,
) -> str:
    """Build a compact, deduplicated text representation for embedding.

    Combines headline, abstract, snippet, lead_paragraph, and flattened keywords.
    Deduplicates overlapping text segments and handles nulls.
    """
    parts: list[str] = []
    seen_lower: set[str] = set()

    for text in [headline, abstract, snippet, lead_paragraph]:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        lower = cleaned.lower().strip()
        # Skip if this text is a substring of something already added
        if any(lower in s for s in seen_lower):
            continue
        # Remove previously added texts that are substrings of this one
        seen_lower = {s for s in seen_lower if s not in lower}
        seen_lower.add(lower)
        parts.append(cleaned)

    # Flatten keywords
    kw_text = flatten_keywords(keywords_json)
    if kw_text:
        parts.append(kw_text)

    return " | ".join(parts)


def flatten_keywords(keywords_json: str) -> str:
    """Extract keyword values from NYT keywords JSON."""
    try:
        keywords = json.loads(keywords_json)
    except (json.JSONDecodeError, TypeError):
        return ""

    values = []
    for kw in keywords:
        if isinstance(kw, dict):
            val = kw.get("value", "")
        elif isinstance(kw, str):
            val = kw
        else:
            continue
        val = val.strip()
        if val:
            values.append(val)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in values:
        low = v.lower()
        if low not in seen:
            seen.add(low)
            unique.append(v)
    return ", ".join(unique)


def clean_text(text: str | None) -> str:
    """Clean and normalize a text field."""
    if not text:
        return ""
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text
