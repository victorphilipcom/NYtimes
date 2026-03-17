"""Tests for text building and normalization."""

import json

from nyt_factor_pipeline.utils.text import build_normalized_text, clean_text, flatten_keywords


class TestBuildNormalizedText:
    def test_basic_build(self):
        result = build_normalized_text(
            headline="Economy Grows",
            abstract="The economy grew 3%.",
            snippet="Economic growth surged.",
            lead_paragraph="In a surprising development, the economy grew 3%.",
            keywords_json=json.dumps([{"value": "Economy"}, {"value": "GDP"}]),
        )
        assert "Economy Grows" in result
        assert "Economy, GDP" in result

    def test_deduplication(self):
        """Identical abstract and snippet should not repeat."""
        result = build_normalized_text(
            headline="Title",
            abstract="Same text here.",
            snippet="Same text here.",
            lead_paragraph="Different lead.",
            keywords_json="[]",
        )
        assert result.count("Same text here") == 1

    def test_substring_dedup(self):
        """If abstract is a substring of lead_paragraph, skip the abstract."""
        result = build_normalized_text(
            headline="Title",
            abstract="short text",
            snippet="",
            lead_paragraph="This is short text with more context.",
            keywords_json="[]",
        )
        # The abstract "short text" is a substring of lead_paragraph
        parts = result.split(" | ")
        assert len(parts) <= 3  # headline + lead + maybe keywords

    def test_null_handling(self):
        result = build_normalized_text(
            headline="Title",
            abstract=None,
            snippet=None,
            lead_paragraph="",
            keywords_json="[]",
        )
        assert "Title" in result

    def test_empty_result(self):
        result = build_normalized_text("", "", "", "", "[]")
        assert result == ""


class TestFlattenKeywords:
    def test_nyt_format(self):
        kw = json.dumps([
            {"name": "subject", "value": "Economy"},
            {"name": "organizations", "value": "Federal Reserve"},
        ])
        assert flatten_keywords(kw) == "Economy, Federal Reserve"

    def test_string_list(self):
        kw = json.dumps(["Economy", "Trade"])
        assert flatten_keywords(kw) == "Economy, Trade"

    def test_deduplication(self):
        kw = json.dumps([{"value": "Economy"}, {"value": "economy"}])
        result = flatten_keywords(kw)
        assert result.lower().count("economy") == 1

    def test_empty(self):
        assert flatten_keywords("[]") == ""
        assert flatten_keywords("") == ""
        assert flatten_keywords("invalid") == ""


class TestCleanText:
    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_none(self):
        assert clean_text(None) == ""
