"""Prompt templates for sparse LLM tasks."""

from __future__ import annotations

THEME_NAMING_SYSTEM = """You are a news theme analyst. Given evidence about a news theme cluster,
provide a concise, descriptive label and brief description. Be specific and factual."""

THEME_NAMING_USER = """Here is evidence about a news theme cluster:

Top Keywords: {keywords}
Representative Headlines:
{headlines}

Representative Abstracts:
{abstracts}

Date Range: {date_start} to {date_end}
Article Count: {article_count}

Please provide:
1. LABEL: A concise theme label (3-8 words)
2. DESCRIPTION: A one-sentence description of what this theme covers
3. PARENT_CATEGORY: A broader category this belongs to (e.g., "Geopolitics", "Technology", "Trade", "Energy", "Finance", "Healthcare")

Format your response exactly as:
LABEL: <label>
DESCRIPTION: <description>
PARENT_CATEGORY: <category>"""


MERGE_REVIEW_SYSTEM = """You are a news theme analyst. Decide whether two news themes
should be merged into one, or kept separate. Be conservative — only merge if they
clearly cover the same underlying topic."""

MERGE_REVIEW_USER = """Theme A: "{label_a}"
Keywords: {keywords_a}
Headlines: {headlines_a}

Theme B: "{label_b}"
Keywords: {keywords_b}
Headlines: {headlines_b}

Should these themes be merged? Respond exactly as:
DECISION: MERGE or NO_MERGE
RATIONALE: <brief explanation>"""


RBICS_MAPPING_SYSTEM = """You are a financial analyst. Given a stable news theme,
suggest which industries (using RBICS-style classifications) are likely positively
or negatively affected. Be specific about industry names and brief about rationale."""

RBICS_MAPPING_USER = """Theme: "{label}"
Description: {description}
Top Keywords: {keywords}
Representative Headlines:
{headlines}

Date Range: {date_start} to {date_end}
Article Count: {article_count}

List industries likely affected by this theme:

POSITIVE (industries that benefit):
- <industry_name>: <brief rationale>

NEGATIVE (industries that are hurt):
- <industry_name>: <brief rationale>

AMBIGUOUS (could go either way):
- <industry_name>: <brief rationale>

Use specific industry names like "Semiconductors", "Oil & Gas Exploration",
"Commercial Banking", "Electric Utilities", etc."""


def format_theme_naming_prompt(data: dict) -> str:
    """Format theme naming prompt from representative data."""
    headlines = "\n".join(f"- {h}" for h in data.get("representative_headlines", [])[:10])
    abstracts = "\n".join(f"- {a}" for a in data.get("representative_abstracts", [])[:5])
    keywords = ", ".join(data.get("top_keywords", [])[:20])

    return THEME_NAMING_USER.format(
        keywords=keywords,
        headlines=headlines or "None available",
        abstracts=abstracts or "None available",
        date_start=data.get("date_range_start", "unknown"),
        date_end=data.get("date_range_end", "unknown"),
        article_count=data.get("total_articles", 0),
    )


def format_merge_review_prompt(
    label_a: str, keywords_a: list, headlines_a: list,
    label_b: str, keywords_b: list, headlines_b: list,
) -> str:
    return MERGE_REVIEW_USER.format(
        label_a=label_a,
        keywords_a=", ".join(keywords_a[:10]),
        headlines_a="\n".join(f"- {h}" for h in headlines_a[:5]),
        label_b=label_b,
        keywords_b=", ".join(keywords_b[:10]),
        headlines_b="\n".join(f"- {h}" for h in headlines_b[:5]),
    )


def format_rbics_mapping_prompt(data: dict) -> str:
    headlines = "\n".join(f"- {h}" for h in data.get("representative_headlines", [])[:10])
    keywords = ", ".join(data.get("top_keywords", [])[:20])

    return RBICS_MAPPING_USER.format(
        label=data.get("label", "Unknown"),
        description=data.get("description", ""),
        keywords=keywords,
        headlines=headlines or "None available",
        date_start=data.get("date_range_start", "unknown"),
        date_end=data.get("date_range_end", "unknown"),
        article_count=data.get("total_articles", 0),
    )
