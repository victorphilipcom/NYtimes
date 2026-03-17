"""Tests for article importance scoring."""

from nyt_factor_pipeline.scoring.article_importance import compute_importance_score


class TestImportanceScoring:
    def test_front_page_business_news_high_score(self):
        score = compute_importance_score(
            headline_main="Major Economic Shift",
            section_name="Business Day",
            news_desk="Business/Financial Desk",
            type_of_material="News",
            print_section="A",
            print_page="1",
            word_count=1500,
        )
        assert score > 0.7

    def test_letter_to_editor_low_score(self):
        score = compute_importance_score(
            headline_main="Reader Comment",
            section_name="Opinion",
            news_desk="Editorial Desk",
            type_of_material="Letter",
            print_section="",
            print_page="",
            word_count=200,
        )
        assert score < 0.5

    def test_obituary_low_score(self):
        score = compute_importance_score(
            headline_main="Famous Person Dies",
            section_name="Obituaries",
            news_desk="Obituary",
            type_of_material="Obituary",
            print_section="B",
            print_page="5",
            word_count=800,
        )
        assert score < 0.5

    def test_score_range(self):
        """All scores should be in [0, 1]."""
        test_cases = [
            ("", "", "", "", "", "", 0),
            ("Headline", "World", "Foreign Desk", "News", "A", "1", 2000),
            ("Short", "Style", "Society Desk", "Review", "", "", 100),
        ]
        for hl, sec, desk, mat, psec, pp, wc in test_cases:
            score = compute_importance_score(hl, sec, desk, mat, psec, pp, wc)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {hl}"

    def test_word_count_boost(self):
        base = compute_importance_score(
            "Headline", "Business Day", "Business", "News", "", "", 100
        )
        boosted = compute_importance_score(
            "Headline", "Business Day", "Business", "News", "", "", 1000
        )
        assert boosted > base

    def test_empty_inputs(self):
        score = compute_importance_score("", "", "", "", "", "", 0)
        assert 0.0 <= score <= 1.0
