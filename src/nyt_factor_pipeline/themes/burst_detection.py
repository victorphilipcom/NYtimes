"""Burst detection — identify sudden spikes in theme intensity."""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def compute_burst_zscores(
    conn: duckdb.DuckDBPyConnection,
    lookback_days: int = 30,
    min_history: int = 7,
) -> int:
    """Compute burst z-scores for all theme-date pairs.

    For each (theme, date) the z-score measures how unusual the intensity is
    compared to the rolling lookback window.

    burst_zscore = (intensity - rolling_mean) / rolling_std

    Returns count of updated rows.
    """
    # Get all themes with timeseries
    themes = conn.execute(
        "SELECT DISTINCT theme_id FROM theme_timeseries"
    ).fetchall()

    count = 0
    for (theme_id,) in themes:
        rows = conn.execute(
            """SELECT date, intensity FROM theme_timeseries
               WHERE theme_id = ? ORDER BY date""",
            [theme_id],
        ).fetchall()

        if len(rows) < min_history:
            continue

        dates = [r[0] for r in rows]
        intensities = np.array([r[1] for r in rows], dtype=np.float64)

        zscores = np.zeros(len(intensities))
        for i in range(min_history, len(intensities)):
            window = intensities[max(0, i - lookback_days) : i]
            mean = window.mean()
            std = window.std()
            if std > 1e-10:
                zscores[i] = (intensities[i] - mean) / std
            else:
                zscores[i] = 0.0

        for i in range(len(dates)):
            conn.execute(
                "UPDATE theme_timeseries SET burst_zscore = ? WHERE theme_id = ? AND date = ?",
                [float(zscores[i]), theme_id, dates[i]],
            )
            count += 1

    log.info("burst_zscores_computed", themes=len(themes), updated_rows=count)
    return count


def get_bursting_themes(
    conn: duckdb.DuckDBPyConnection,
    min_zscore: float = 2.0,
    recent_days: int = 7,
) -> list[dict]:
    """Get themes that are currently bursting (high recent z-score)."""
    rows = conn.execute(
        """SELECT ts.theme_id, t.current_label, ts.date, ts.intensity, ts.burst_zscore
           FROM theme_timeseries ts
           JOIN themes t ON ts.theme_id = t.theme_id
           WHERE ts.burst_zscore >= ?
             AND ts.date >= current_date - ?
           ORDER BY ts.burst_zscore DESC""",
        [min_zscore, recent_days],
    ).fetchall()

    return [
        {
            "theme_id": r[0],
            "label": r[1],
            "date": r[2],
            "intensity": r[3],
            "burst_zscore": r[4],
        }
        for r in rows
    ]
