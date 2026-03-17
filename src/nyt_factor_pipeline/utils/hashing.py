"""Hashing utilities for deduplication and caching."""

from __future__ import annotations

import hashlib
import json


def stable_article_id(uri: str, web_url: str) -> str:
    """Derive a stable article ID from NYT URI or URL."""
    key = uri or web_url
    if not key:
        raise ValueError("Both uri and web_url are empty; cannot generate article ID")
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def prompt_hash(prompt_text: str) -> str:
    """Hash a prompt for caching LLM responses."""
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:32]


def dict_hash(d: dict) -> str:
    """Hash a dictionary deterministically."""
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:20]
