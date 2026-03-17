"""Tests for NYT article normalization."""

import json

from nyt_factor_pipeline.ingest.normalize import normalize_article, normalize_batch


def _make_raw_article(**overrides) -> dict:
    base = {
        "uri": "nyt://article/test-123",
        "web_url": "https://www.nytimes.com/2024/01/15/business/test-article.html",
        "headline": {"main": "Test Headline"},
        "abstract": "Test abstract text.",
        "snippet": "Test snippet.",
        "lead_paragraph": "Test lead paragraph with more detail.",
        "pub_date": "2024-01-15T10:30:00+0000",
        "source": "The New York Times",
        "section_name": "Business Day",
        "subsection_name": "",
        "news_desk": "Business/Financial Desk",
        "type_of_material": "News",
        "document_type": "article",
        "print_section": "A",
        "print_page": "1",
        "word_count": 1200,
        "byline": {"original": "By Test Author"},
        "keywords": [
            {"name": "subject", "value": "Economy"},
            {"name": "organizations", "value": "Federal Reserve"},
        ],
        "multimedia": [],
    }
    base.update(overrides)
    return base


class TestNormalizeArticle:
    def test_basic_normalization(self):
        raw = _make_raw_article()
        article = normalize_article(raw, "archive")

        assert article is not None
        assert article.source_api == "archive"
        assert article.headline_main == "Test Headline"
        assert article.abstract == "Test abstract text."
        assert article.section_name == "Business Day"
        assert article.news_desk == "Business/Financial Desk"
        assert article.word_count == 1200
        assert article.print_page == "1"
        assert article.year == 2024
        assert article.month == 1
        assert article.day == 15

    def test_normalized_text_deduplication(self):
        raw = _make_raw_article(
            abstract="Identical text here.",
            snippet="Identical text here.",  # duplicate
        )
        article = normalize_article(raw, "archive")
        assert article is not None
        # The identical snippet should be deduplicated
        assert article.normalized_text.count("Identical text here") == 1

    def test_stable_article_id(self):
        raw = _make_raw_article()
        a1 = normalize_article(raw, "archive")
        a2 = normalize_article(raw, "article_search")
        assert a1 is not None and a2 is not None
        assert a1.article_id == a2.article_id  # Same URI = same ID

    def test_missing_uri_uses_web_url(self):
        raw = _make_raw_article(uri="")
        article = normalize_article(raw, "archive")
        assert article is not None
        assert len(article.article_id) == 20

    def test_missing_both_ids_returns_none(self):
        raw = _make_raw_article(uri="", web_url="")
        article = normalize_article(raw, "archive")
        assert article is None

    def test_keywords_in_normalized_text(self):
        raw = _make_raw_article()
        article = normalize_article(raw, "archive")
        assert article is not None
        assert "Economy" in article.normalized_text
        assert "Federal Reserve" in article.normalized_text

    def test_null_handling(self):
        raw = _make_raw_article(
            abstract=None,
            snippet=None,
            lead_paragraph=None,
            byline=None,
            keywords=None,
        )
        article = normalize_article(raw, "archive")
        assert article is not None
        assert article.abstract == ""
        assert article.byline_original == ""

    def test_article_search_format(self):
        """Article Search API has slightly different field names but same structure."""
        raw = _make_raw_article()
        article = normalize_article(raw, "article_search")
        assert article is not None
        assert article.source_api == "article_search"


class TestNormalizeBatch:
    def test_batch_skips_failures(self):
        good = _make_raw_article()
        bad = {"uri": "", "web_url": ""}  # will be skipped

        articles = normalize_batch([good, bad], "archive")
        assert len(articles) == 1
        assert articles[0].headline_main == "Test Headline"

    def test_empty_batch(self):
        assert normalize_batch([], "archive") == []
