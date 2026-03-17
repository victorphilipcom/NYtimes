"""Map themes to companies via RBICS codes."""

from __future__ import annotations

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def build_company_theme_exposures(conn: duckdb.DuckDBPyConnection) -> int:
    """Build company_theme_exposure by joining theme_rbics_exposure with companies.

    For each (theme, rbics_code) exposure, find companies with matching rbics_code
    and create a company_theme_exposure entry.

    Returns count of exposure rows created.
    """
    # Match via exact rbics_code or via fuzzy name matching
    conn.execute("DELETE FROM company_theme_exposure")

    count_result = conn.execute(
        """INSERT INTO company_theme_exposure (company_id, theme_id, exposure_strength, exposure_direction, source)
           SELECT c.company_id, tre.theme_id, tre.exposure_strength, tre.exposure_direction, 'rbics_match'
           FROM theme_rbics_exposure tre
           JOIN companies c ON c.rbics_code = tre.rbics_code
           WHERE tre.exposure_strength > 0"""
    )

    # Also try fuzzy matching by name similarity
    # Get all theme-rbics pairs with no exact match
    unmatched = conn.execute(
        """SELECT DISTINCT tre.theme_id, tre.rbics_code, tre.exposure_direction,
                  tre.exposure_strength, tre.rationale
           FROM theme_rbics_exposure tre
           LEFT JOIN companies c ON c.rbics_code = tre.rbics_code
           WHERE c.company_id IS NULL"""
    ).fetchall()

    fuzzy_matched = 0
    for theme_id, rbics_code, direction, strength, rationale in unmatched:
        # Try matching by rbics_name containing the industry keyword
        industry_name = rbics_code.replace("_", " ")
        matches = conn.execute(
            """SELECT company_id FROM companies
               WHERE LOWER(rbics_name) LIKE ?""",
            [f"%{industry_name}%"],
        ).fetchall()

        for (company_id,) in matches:
            conn.execute(
                """INSERT INTO company_theme_exposure
                   (company_id, theme_id, exposure_strength, exposure_direction, source)
                   VALUES (?, ?, ?, ?, 'fuzzy_rbics_match')
                   ON CONFLICT (company_id, theme_id) DO UPDATE SET
                       exposure_strength = EXCLUDED.exposure_strength""",
                [company_id, theme_id, strength * 0.8, direction],
            )
            fuzzy_matched += 1

    total = conn.execute("SELECT COUNT(*) FROM company_theme_exposure").fetchone()
    count = total[0] if total else 0
    log.info("company_exposures_built", total=count, fuzzy=fuzzy_matched)
    return count
