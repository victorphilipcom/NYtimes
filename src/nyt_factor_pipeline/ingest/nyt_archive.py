"""NYT Archive API ingestion — fetch historical months of articles."""

from __future__ import annotations

import duckdb

from nyt_factor_pipeline.ingest.checkpoints import CheckpointManager
from nyt_factor_pipeline.ingest.nyt_client import NYTClient, NYT_BASE_URL
from nyt_factor_pipeline.ingest.normalize import normalize_batch
from nyt_factor_pipeline.ingest.request_budget import BudgetExhaustedError
from nyt_factor_pipeline.ingest.storage import upsert_articles
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.utils.dates import format_month, month_range

log = get_logger(__name__)

ARCHIVE_URL_TEMPLATE = f"{NYT_BASE_URL}/archive/v1/{{year}}/{{month}}.json"


def estimate_archive_requests(start_month: str, end_month: str) -> int:
    """Estimate the number of API requests needed for an archive backfill.

    Each month = 1 request.
    """
    months = month_range(start_month, end_month)
    return len(months)


def ingest_archive(
    conn: duckdb.DuckDBPyConnection,
    start_month: str,
    end_month: str,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Ingest NYT Archive API data for a range of months.

    Args:
        conn: DuckDB connection
        start_month: Start month in YYYY-MM format
        end_month: End month in YYYY-MM format
        force: If True, re-fetch even completed months
        dry_run: If True, only estimate cost without fetching

    Returns:
        Summary dict with counts
    """
    months = month_range(start_month, end_month)
    total_requests = len(months)

    if dry_run:
        return {
            "months": len(months),
            "estimated_requests": total_requests,
            "action": "dry_run",
        }

    client = NYTClient(conn, api_name="archive")
    checkpoints = CheckpointManager(conn, source_api="archive")
    stats = {"months_fetched": 0, "articles_ingested": 0, "months_skipped": 0, "errors": 0}

    for year, month in months:
        key = format_month(year, month)

        if not force and checkpoints.is_completed(key):
            log.info("archive_month_skipped", month=key, reason="already_completed")
            stats["months_skipped"] += 1
            continue

        try:
            url = ARCHIVE_URL_TEMPLATE.format(year=year, month=month)
            log.info("archive_fetching", month=key)
            data = client.get(url)

            response_obj = data.get("response", {})
            docs = response_obj.get("docs", [])

            if not docs:
                log.warning("archive_empty_response", month=key)
                checkpoints.mark_completed(key, {"article_count": 0})
                stats["months_fetched"] += 1
                continue

            articles = normalize_batch(docs, source_api="archive")
            upsert_articles(conn, articles)

            checkpoints.mark_completed(key, {"article_count": len(articles)})
            stats["months_fetched"] += 1
            stats["articles_ingested"] += len(articles)

            log.info(
                "archive_month_done",
                month=key,
                articles=len(articles),
                raw_docs=len(docs),
            )

        except BudgetExhaustedError:
            log.warning("archive_budget_exhausted", last_month=key)
            break
        except Exception as e:
            log.error("archive_month_error", month=key, error=str(e))
            stats["errors"] += 1
            continue

    client.close()
    return stats
