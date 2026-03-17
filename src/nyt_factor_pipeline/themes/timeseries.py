"""Build theme intensity time series."""

from __future__ import annotations

from datetime import date

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def build_theme_timeseries(
    conn: duckdb.DuckDBPyConnection,
    start_date: date | None = None,
    end_date: date | None = None,
) -> int:
    """Build daily theme_timeseries from cluster membership and article scores.

    For each theme and date:
    - article_count: number of articles assigned to the theme that day
    - weighted_article_count: sum of importance_score for those articles
    - intensity: normalized weighted count (scaled by rolling mean)
    - burst_zscore: computed later by burst_detection

    Returns count of rows inserted/updated.
    """
    date_filter = ""
    params: list = []
    if start_date:
        date_filter += " AND a.pub_date >= ?"
        params.append(str(start_date))
    if end_date:
        date_filter += " AND a.pub_date <= ?"
        params.append(str(end_date))

    # Compute daily counts per theme
    query = f"""
        SELECT ctl.theme_id,
               CAST(a.pub_date AS DATE) as day,
               COUNT(DISTINCT acm.article_id) as article_count,
               COALESCE(SUM(a.importance_score), 0) as weighted_count
        FROM cluster_theme_link ctl
        JOIN article_cluster_membership acm
            ON ctl.cluster_id = acm.cluster_id
            AND ctl.window_start = acm.window_start
            AND ctl.window_end = acm.window_end
        JOIN articles a ON acm.article_id = a.article_id
        WHERE 1=1 {date_filter}
        GROUP BY ctl.theme_id, CAST(a.pub_date AS DATE)
        ORDER BY ctl.theme_id, day
    """

    rows = conn.execute(query, params).fetchall()

    if not rows:
        return 0

    # Compute global daily article counts for normalization
    global_daily = conn.execute(
        f"""SELECT CAST(pub_date AS DATE) as day, COUNT(*) as cnt
            FROM articles
            WHERE macro_relevance_score > 0 {date_filter}
            GROUP BY CAST(pub_date AS DATE)""",
        params,
    ).fetchall()
    global_counts = {str(r[0]): r[1] for r in global_daily}

    count = 0
    for theme_id, day, article_count, weighted_count in rows:
        global_count = global_counts.get(str(day), 1)
        intensity = weighted_count / max(global_count, 1)

        conn.execute(
            """INSERT INTO theme_timeseries
               (theme_id, date, article_count, weighted_article_count, intensity, burst_zscore)
               VALUES (?, ?, ?, ?, ?, 0.0)
               ON CONFLICT (theme_id, date) DO UPDATE SET
                   article_count = ?,
                   weighted_article_count = ?,
                   intensity = ?""",
            [
                theme_id, day, article_count, weighted_count, intensity,
                article_count, weighted_count, intensity,
            ],
        )
        count += 1

    log.info("timeseries_built", rows=count)
    return count


def get_theme_timeseries(
    conn: duckdb.DuckDBPyConnection,
    theme_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    """Get time series for a specific theme."""
    conditions = ["theme_id = ?"]
    params: list = [theme_id]
    if start_date:
        conditions.append("date >= ?")
        params.append(str(start_date))
    if end_date:
        conditions.append("date <= ?")
        params.append(str(end_date))

    where = " AND ".join(conditions)
    rows = conn.execute(
        f"""SELECT date, article_count, weighted_article_count, intensity, burst_zscore
            FROM theme_timeseries WHERE {where} ORDER BY date""",
        params,
    ).fetchall()

    return [
        {
            "date": r[0],
            "article_count": r[1],
            "weighted_article_count": r[2],
            "intensity": r[3],
            "burst_zscore": r[4],
        }
        for r in rows
    ]
