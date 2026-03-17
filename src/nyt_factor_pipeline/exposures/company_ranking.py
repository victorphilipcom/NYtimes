"""Compute daily company-theme scores for factor investing research."""

from __future__ import annotations

from datetime import date

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def score_companies_for_date(
    conn: duckdb.DuckDBPyConnection,
    score_date: date,
) -> int:
    """Compute company_theme_scores for a given date.

    Formula:
    score = theme_intensity(date) * company_exposure_strength * direction_sign

    where direction_sign = +1 for positive, -1 for negative, 0 for ambiguous

    Returns count of scores computed.
    """
    rows = conn.execute(
        """SELECT cte.company_id, cte.theme_id, cte.exposure_strength, cte.exposure_direction,
                  COALESCE(ts.intensity, 0) as intensity
           FROM company_theme_exposure cte
           LEFT JOIN theme_timeseries ts
               ON cte.theme_id = ts.theme_id AND ts.date = ?
           WHERE cte.exposure_strength > 0""",
        [score_date],
    ).fetchall()

    count = 0
    for company_id, theme_id, strength, direction, intensity in rows:
        direction_sign = _direction_to_sign(direction)
        score = intensity * strength * direction_sign

        conn.execute(
            """INSERT INTO company_theme_scores (company_id, date, theme_id, score)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (company_id, date, theme_id) DO UPDATE SET score = ?""",
            [company_id, score_date, theme_id, score, score],
        )
        count += 1

    log.info("company_scores_computed", date=str(score_date), scores=count)
    return count


def score_companies_range(
    conn: duckdb.DuckDBPyConnection,
    start_date: date,
    end_date: date,
) -> int:
    """Score companies for a date range."""
    # Get all dates with theme timeseries data
    rows = conn.execute(
        """SELECT DISTINCT date FROM theme_timeseries
           WHERE date >= ? AND date <= ? ORDER BY date""",
        [start_date, end_date],
    ).fetchall()

    total = 0
    for (d,) in rows:
        total += score_companies_for_date(conn, d)

    return total


def get_company_scores(
    conn: duckdb.DuckDBPyConnection,
    company_id: str | None = None,
    score_date: date | None = None,
    top_n: int = 50,
) -> list[dict]:
    """Query company scores with optional filters."""
    conditions = []
    params: list = []

    if company_id:
        conditions.append("cts.company_id = ?")
        params.append(company_id)
    if score_date:
        conditions.append("cts.date = ?")
        params.append(score_date)

    where = " AND ".join(conditions) if conditions else "1=1"

    rows = conn.execute(
        f"""SELECT cts.company_id, c.ticker, c.company_name, cts.date, cts.theme_id,
                   t.current_label, cts.score
            FROM company_theme_scores cts
            JOIN companies c ON cts.company_id = c.company_id
            JOIN themes t ON cts.theme_id = t.theme_id
            WHERE {where}
            ORDER BY ABS(cts.score) DESC
            LIMIT ?""",
        params + [top_n],
    ).fetchall()

    return [
        {
            "company_id": r[0],
            "ticker": r[1],
            "company_name": r[2],
            "date": r[3],
            "theme_id": r[4],
            "theme_label": r[5],
            "score": r[6],
        }
        for r in rows
    ]


def get_company_total_scores(
    conn: duckdb.DuckDBPyConnection,
    score_date: date,
    top_n: int = 50,
) -> list[dict]:
    """Get aggregated company scores for a date.

    Returns total, positive-only, and negative-only scores.
    """
    rows = conn.execute(
        """SELECT cts.company_id, c.ticker, c.company_name,
                  SUM(cts.score) as total_score,
                  SUM(CASE WHEN cts.score > 0 THEN cts.score ELSE 0 END) as positive_score,
                  SUM(CASE WHEN cts.score < 0 THEN cts.score ELSE 0 END) as negative_score,
                  COUNT(*) as theme_count
           FROM company_theme_scores cts
           JOIN companies c ON cts.company_id = c.company_id
           WHERE cts.date = ?
           GROUP BY cts.company_id, c.ticker, c.company_name
           ORDER BY ABS(SUM(cts.score)) DESC
           LIMIT ?""",
        [score_date, top_n],
    ).fetchall()

    return [
        {
            "company_id": r[0],
            "ticker": r[1],
            "company_name": r[2],
            "total_score": r[3],
            "positive_score": r[4],
            "negative_score": r[5],
            "theme_count": r[6],
        }
        for r in rows
    ]


def _direction_to_sign(direction: str) -> float:
    if direction == "positive":
        return 1.0
    elif direction == "negative":
        return -1.0
    return 0.0
