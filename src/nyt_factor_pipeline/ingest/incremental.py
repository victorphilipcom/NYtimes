"""Incremental ingestion logic — determine what to fetch next."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb

from nyt_factor_pipeline.ingest.checkpoints import CheckpointManager
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.utils.dates import month_range

log = get_logger(__name__)


def get_latest_archive_month(conn: duckdb.DuckDBPyConnection) -> tuple[int, int] | None:
    """Get the latest completed archive month from checkpoints."""
    cm = CheckpointManager(conn, "archive")
    checkpoints = cm.list_checkpoints()
    completed = [
        cp["key"] for cp in checkpoints if cp["value"].get("completed")
    ]
    if not completed:
        return None
    latest = max(completed)
    parts = latest.split("-")
    return int(parts[0]), int(parts[1])


def get_latest_article_date(conn: duckdb.DuckDBPyConnection) -> date | None:
    """Get the latest pub_date from ingested articles."""
    result = conn.execute(
        "SELECT MAX(pub_date) FROM articles"
    ).fetchone()
    if result and result[0]:
        return result[0].date() if hasattr(result[0], "date") else result[0]
    return None


def compute_resume_plan(conn: duckdb.DuckDBPyConnection) -> dict:
    """Determine what to fetch next.

    Returns a plan dict with 'archive' and 'article_search' keys
    indicating what ranges need fetching.
    """
    latest_archive = get_latest_archive_month(conn)
    today = date.today()

    plan: dict = {"archive": None, "article_search": None}

    # Archive: the latest full month we can fetch is last month
    last_full_month = today.replace(day=1) - timedelta(days=1)
    archive_end = (last_full_month.year, last_full_month.month)

    if latest_archive is None:
        # No archive data yet — suggest starting from a reasonable default
        plan["archive"] = {
            "status": "no_data",
            "suggestion": "Run ingest-archive with desired start/end months",
        }
    else:
        next_year, next_month = latest_archive
        next_month += 1
        if next_month > 12:
            next_month = 1
            next_year += 1

        if (next_year, next_month) <= archive_end:
            remaining = month_range(
                f"{next_year:04d}-{next_month:02d}",
                f"{archive_end[0]:04d}-{archive_end[1]:02d}",
            )
            plan["archive"] = {
                "status": "behind",
                "next_month": f"{next_year:04d}-{next_month:02d}",
                "end_month": f"{archive_end[0]:04d}-{archive_end[1]:02d}",
                "months_remaining": len(remaining),
            }
        else:
            plan["archive"] = {"status": "up_to_date"}

    # Article search: cover from end of latest archive through today
    if latest_archive:
        # Start article search from the 1st of the month after last archive
        as_year, as_month = latest_archive
        as_month += 1
        if as_month > 12:
            as_month = 1
            as_year += 1
        as_start = date(as_year, as_month, 1)
    else:
        latest_date = get_latest_article_date(conn)
        if latest_date:
            as_start = latest_date + timedelta(days=1)
        else:
            as_start = today  # nothing to do

    if as_start <= today:
        plan["article_search"] = {
            "status": "behind",
            "start_date": str(as_start),
            "end_date": str(today),
            "days": (today - as_start).days + 1,
        }
    else:
        plan["article_search"] = {"status": "up_to_date"}

    return plan
