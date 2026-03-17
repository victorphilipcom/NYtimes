"""NYT Article Search API ingestion — fetch recent articles with pagination."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb

from nyt_factor_pipeline.ingest.checkpoints import CheckpointManager
from nyt_factor_pipeline.ingest.nyt_client import NYTClient, NYT_BASE_URL
from nyt_factor_pipeline.ingest.normalize import normalize_batch
from nyt_factor_pipeline.ingest.request_budget import BudgetExhaustedError
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.utils.dates import date_windows

log = get_logger(__name__)

ARTICLE_SEARCH_URL = f"{NYT_BASE_URL}/search/v2/articlesearch.json"
MAX_PAGE = 100  # NYT practical limit
RESULTS_PER_PAGE = 10


def estimate_article_search_requests(
    start_date: date,
    end_date: date,
    avg_articles_per_day: int = 200,
) -> int:
    """Estimate requests needed. Each day ~ avg_articles_per_day / 10 pages."""
    days = (end_date - start_date).days + 1
    pages_per_day = max(1, avg_articles_per_day // RESULTS_PER_PAGE)
    return days * pages_per_day


def ingest_article_search(
    conn: duckdb.DuckDBPyConnection,
    start_date: date,
    end_date: date,
    force: bool = False,
    dry_run: bool = False,
    window_days: int = 1,
    fq: str | None = None,
) -> dict:
    """Ingest articles from NYT Article Search API.

    Splits the date range into windows and paginates each.
    Adaptively splits windows if too many hits.

    Args:
        conn: DuckDB connection
        start_date: Start date
        end_date: End date
        force: Re-fetch completed windows
        dry_run: Only estimate
        window_days: Size of each date window in days
        fq: Optional filter query for NYT API
    """
    windows = date_windows(start_date, end_date, window_days)

    if dry_run:
        est = estimate_article_search_requests(start_date, end_date)
        return {
            "windows": len(windows),
            "estimated_requests": est,
            "action": "dry_run",
        }

    client = NYTClient(conn, api_name="article_search")
    checkpoints = CheckpointManager(conn, source_api="article_search")
    stats = {"windows_fetched": 0, "articles_ingested": 0, "windows_skipped": 0, "errors": 0}

    for w_start, w_end in windows:
        _ingest_window(conn, client, checkpoints, w_start, w_end, force, fq, stats)

    client.close()
    return stats


def _ingest_window(
    conn: duckdb.DuckDBPyConnection,
    client: NYTClient,
    checkpoints: CheckpointManager,
    w_start: date,
    w_end: date,
    force: bool,
    fq: str | None,
    stats: dict,
) -> None:
    """Ingest a single date window with pagination and adaptive splitting."""
    key = f"{w_start}_{w_end}"

    if not force and checkpoints.is_completed(key):
        stats["windows_skipped"] += 1
        return

    try:
        # Page 0 to get hit count
        params = _build_params(w_start, w_end, page=0, fq=fq)
        data = client.get(ARTICLE_SEARCH_URL, params=params)

        response_obj = data.get("response", {})
        meta = response_obj.get("meta", {})
        total_hits = meta.get("hits", 0)
        docs = response_obj.get("docs", [])

        # If too many hits, split the window
        if total_hits > MAX_PAGE * RESULTS_PER_PAGE and (w_end - w_start).days > 0:
            log.info(
                "splitting_window",
                start=str(w_start),
                end=str(w_end),
                hits=total_hits,
            )
            mid = w_start + (w_end - w_start) // 2
            _ingest_window(conn, client, checkpoints, w_start, mid, force, fq, stats)
            _ingest_window(conn, client, checkpoints, mid + timedelta(days=1), w_end, force, fq, stats)
            return

        # Process page 0
        articles = normalize_batch(docs, source_api="article_search")
        _upsert_articles(conn, articles)
        total_ingested = len(articles)

        # Paginate remaining pages
        total_pages = min((total_hits + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE, MAX_PAGE)

        for page in range(1, total_pages):
            page_key = f"{key}_p{page}"
            cp = checkpoints.get_value(page_key)
            if cp and cp.get("completed"):
                continue

            try:
                params = _build_params(w_start, w_end, page=page, fq=fq)
                data = client.get(ARTICLE_SEARCH_URL, params=params)
                page_docs = data.get("response", {}).get("docs", [])

                if not page_docs:
                    break

                page_articles = normalize_batch(page_docs, source_api="article_search")
                _upsert_articles(conn, page_articles)
                total_ingested += len(page_articles)

                checkpoints.mark_completed(page_key, {"article_count": len(page_articles)})

            except BudgetExhaustedError:
                raise
            except Exception as e:
                log.warning("page_error", window=key, page=page, error=str(e))
                stats["errors"] += 1

        checkpoints.mark_completed(key, {"article_count": total_ingested, "total_hits": total_hits})
        stats["windows_fetched"] += 1
        stats["articles_ingested"] += total_ingested

        log.info(
            "window_done",
            window=key,
            articles=total_ingested,
            hits=total_hits,
        )

    except BudgetExhaustedError:
        log.warning("article_search_budget_exhausted", window=key)
        raise
    except Exception as e:
        log.error("window_error", window=key, error=str(e))
        stats["errors"] += 1


def _build_params(
    start: date, end: date, page: int = 0, fq: str | None = None
) -> dict:
    params: dict = {
        "begin_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "page": page,
        "sort": "newest",
    }
    if fq:
        params["fq"] = fq
    return params


def _upsert_articles(conn: duckdb.DuckDBPyConnection, articles: list) -> None:
    for article in articles:
        conn.execute(
            """INSERT INTO articles (
                article_id, source_api, web_url, uri, pub_date, year, month, day,
                headline_main, abstract, snippet, lead_paragraph, source,
                section_name, subsection_name, news_desk, type_of_material,
                document_type, print_section, print_page, word_count,
                byline_original, keywords_json, multimedia_json, normalized_text,
                importance_score, macro_relevance_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (article_id) DO UPDATE SET
                updated_at = current_timestamp""",
            [
                article.article_id,
                article.source_api,
                article.web_url,
                article.uri,
                article.pub_date,
                article.year,
                article.month,
                article.day,
                article.headline_main,
                article.abstract,
                article.snippet,
                article.lead_paragraph,
                article.source,
                article.section_name,
                article.subsection_name,
                article.news_desk,
                article.type_of_material,
                article.document_type,
                article.print_section,
                article.print_page,
                article.word_count,
                article.byline_original,
                article.keywords_json,
                article.multimedia_json,
                article.normalized_text,
                article.importance_score,
                article.macro_relevance_score,
            ],
        )
