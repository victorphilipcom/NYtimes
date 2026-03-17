"""Article importance scoring using NYT metadata only (no LLM)."""

from __future__ import annotations

from nyt_factor_pipeline.config import get_settings

# Configurable weight maps — higher = more important
SECTION_WEIGHTS: dict[str, float] = {
    "Business Day": 1.0,
    "Business": 1.0,
    "World": 0.9,
    "U.S.": 0.8,
    "Technology": 0.9,
    "Science": 0.7,
    "Politics": 0.8,
    "Opinion": 0.5,
    "Health": 0.6,
    "Climate": 0.7,
    "DealBook": 1.0,
    "Economy": 1.0,
}

NEWS_DESK_WEIGHTS: dict[str, float] = {
    "Business/Financial Desk": 1.0,
    "Business": 1.0,
    "Financial Desk": 1.0,
    "Foreign Desk": 0.9,
    "Foreign": 0.9,
    "Washington": 0.8,
    "Washington Desk": 0.8,
    "National Desk": 0.7,
    "National": 0.7,
    "Science Desk": 0.7,
    "Technology": 0.9,
    "Climate": 0.7,
    "Economy": 1.0,
}

# Low-value material types to penalize
LOW_VALUE_TYPES: set[str] = {
    "Letter",
    "Obituary",
    "Obituary (Obit)",
    "Correction",
    "Paid Notice",
    "Review",
    "Schedule",
    "List",
    "Caption",
    "Summary",
    "Recipe",
    "Interactive Feature",
}


def compute_importance_score(
    headline_main: str,
    section_name: str,
    news_desk: str,
    type_of_material: str,
    print_section: str,
    print_page: str,
    word_count: int,
) -> float:
    """Compute an importance score in [0, 1] using NYT metadata.

    Higher scores indicate articles more likely relevant for macro/business analysis.
    """
    score = 0.0
    weights_sum = 0.0

    # 1. Print placement (strong signal)
    if print_page == "1" or print_page == "A1":
        score += 1.0 * 2.0
    elif print_section == "A":
        score += 0.7 * 2.0
    elif print_section == "B":
        score += 0.4 * 2.0
    weights_sum += 2.0

    # 2. Section relevance
    section_w = SECTION_WEIGHTS.get(section_name, 0.3)
    score += section_w * 1.5
    weights_sum += 1.5

    # 3. News desk relevance
    desk_w = NEWS_DESK_WEIGHTS.get(news_desk, 0.3)
    score += desk_w * 1.5
    weights_sum += 1.5

    # 4. Material type
    if type_of_material in LOW_VALUE_TYPES:
        score += 0.0
    elif type_of_material == "News":
        score += 1.0 * 1.0
    elif type_of_material in ("News Analysis", "An Analysis"):
        score += 0.9 * 1.0
    else:
        score += 0.4 * 1.0
    weights_sum += 1.0

    # 5. Word count — longer articles tend to be more substantial
    settings = get_settings()
    if word_count >= settings.importance_word_count_boost_threshold:
        score += 0.8 * 0.5
    elif word_count >= settings.importance_min_word_count:
        score += 0.5 * 0.5
    else:
        score += 0.2 * 0.5
    weights_sum += 0.5

    # 6. Headline presence (sanity check)
    if headline_main.strip():
        score += 0.5 * 0.5
    weights_sum += 0.5

    return round(min(score / weights_sum, 1.0), 4) if weights_sum > 0 else 0.0


def score_articles_in_db(conn) -> int:
    """Score all articles in the database that haven't been scored yet."""
    rows = conn.execute(
        """SELECT article_id, headline_main, section_name, news_desk,
                  type_of_material, print_section, print_page, word_count
           FROM articles
           WHERE importance_score = 0"""
    ).fetchall()

    count = 0
    for row in rows:
        article_id, headline, section, desk, mat_type, psec, ppage, wc = row
        score = compute_importance_score(
            headline_main=headline or "",
            section_name=section or "",
            news_desk=desk or "",
            type_of_material=mat_type or "",
            print_section=psec or "",
            print_page=str(ppage or ""),
            word_count=wc or 0,
        )
        conn.execute(
            "UPDATE articles SET importance_score = ?, updated_at = now() WHERE article_id = ?",
            [score, article_id],
        )
        count += 1

    return count
