"""Pydantic models for the unified article schema and pipeline entities."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class Article(BaseModel):
    """Unified article representation from both Archive and Article Search APIs."""

    article_id: str
    source_api: str  # "archive" or "article_search"
    web_url: str = ""
    uri: str = ""
    pub_date: datetime | None = None
    year: int | None = None
    month: int | None = None
    day: int | None = None
    headline_main: str = ""
    abstract: str = ""
    snippet: str = ""
    lead_paragraph: str = ""
    source: str = ""
    section_name: str = ""
    subsection_name: str = ""
    news_desk: str = ""
    type_of_material: str = ""
    document_type: str = ""
    print_section: str = ""
    print_page: str = ""
    word_count: int = 0
    byline_original: str = ""
    keywords_json: str = "[]"
    multimedia_json: str = "[]"
    normalized_text: str = ""
    importance_score: float = 0.0
    macro_relevance_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ClusterRaw(BaseModel):
    """A raw cluster from a single clustering window."""

    cluster_id: str
    window_start: date
    window_end: date
    article_count: int = 0
    centroid: bytes = b""
    top_keywords_json: str = "[]"
    representative_article_ids: str = "[]"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Theme(BaseModel):
    """A tracked theme that persists across clustering windows."""

    theme_id: str
    current_label: str = ""
    description: str = ""
    parent_theme_id: str = ""
    centroid: bytes = b""
    first_seen: date | None = None
    last_seen: date | None = None
    active_flag: bool = True
    llm_labeled_at: datetime | None = None
    metadata_json: str = "{}"


class ThemeTimeseries(BaseModel):
    theme_id: str
    date: date
    article_count: int = 0
    weighted_article_count: float = 0.0
    intensity: float = 0.0
    burst_zscore: float = 0.0


class Company(BaseModel):
    company_id: str
    ticker: str = ""
    company_name: str = ""
    rbics_code: str = ""
    rbics_name: str = ""
    metadata_json: str = "{}"


class ThemeRBICSExposure(BaseModel):
    theme_id: str
    rbics_code: str
    rbics_level: str = ""
    exposure_direction: str = ""  # "positive", "negative", "ambiguous"
    exposure_strength: float = 0.0
    rationale: str = ""
    source: str = ""  # "llm" or "manual"
