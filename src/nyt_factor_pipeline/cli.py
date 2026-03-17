"""CLI entrypoint for the NYT Factor Pipeline."""

from __future__ import annotations

import json
from datetime import date, datetime

import click

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.db import init_db
from nyt_factor_pipeline.logging_utils import get_logger, setup_logging

log = get_logger(__name__)


@click.group()
def cli():
    """NYT Factor Pipeline — theme discovery for factor investing research."""
    setup_logging()


@cli.command()
def init_db_cmd():
    """Initialize the DuckDB database schema."""
    conn = init_db()
    click.echo(f"Database initialized at {get_settings().db_path}")
    conn.close()


@cli.command()
@click.option("--start", required=True, help="Start month (YYYY-MM)")
@click.option("--end", required=True, help="End month (YYYY-MM)")
def estimate_backfill_cost(start: str, end: str):
    """Estimate the number of API requests for a backfill."""
    from nyt_factor_pipeline.ingest.nyt_archive import estimate_archive_requests
    from nyt_factor_pipeline.ingest.nyt_article_search import estimate_article_search_requests
    from nyt_factor_pipeline.utils.dates import parse_date

    archive_reqs = estimate_archive_requests(start, end)

    # Estimate article search for the remaining tail
    end_parts = end.split("-")
    end_year, end_month = int(end_parts[0]), int(end_parts[1])
    tail_start = date(end_year, end_month, 1)
    tail_end = date.today()

    if tail_start < tail_end:
        search_reqs = estimate_article_search_requests(tail_start, tail_end)
    else:
        search_reqs = 0

    click.echo(f"Archive API requests:        {archive_reqs}")
    click.echo(f"Article Search API requests:  ~{search_reqs}")
    click.echo(f"Total estimated requests:     ~{archive_reqs + search_reqs}")
    click.echo(f"\nArchive daily budget:         {get_settings().nyt_archive_daily_budget}")
    click.echo(f"Estimated archive days:       {max(1, archive_reqs // get_settings().nyt_archive_daily_budget)}")


@cli.command()
@click.option("--start", required=True, help="Start month (YYYY-MM)")
@click.option("--end", required=True, help="End month (YYYY-MM)")
@click.option("--force", is_flag=True, help="Re-fetch completed months")
@click.option("--dry-run", is_flag=True, help="Only estimate, don't fetch")
def ingest_archive(start: str, end: str, force: bool, dry_run: bool):
    """Ingest historical articles via the NYT Archive API."""
    conn = init_db()
    from nyt_factor_pipeline.ingest.nyt_archive import ingest_archive as _ingest

    stats = _ingest(conn, start, end, force=force, dry_run=dry_run)
    click.echo(json.dumps(stats, indent=2, default=str))
    conn.close()


@cli.command()
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD), default: today")
@click.option("--force", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--window-days", default=1, help="Days per query window")
def ingest_recent(start_date: str, end_date: str | None, force: bool, dry_run: bool, window_days: int):
    """Ingest recent articles via the NYT Article Search API."""
    conn = init_db()
    from nyt_factor_pipeline.ingest.nyt_article_search import ingest_article_search
    from nyt_factor_pipeline.utils.dates import parse_date

    sd = parse_date(start_date)
    ed = parse_date(end_date) if end_date else date.today()

    stats = ingest_article_search(conn, sd, ed, force=force, dry_run=dry_run, window_days=window_days)
    click.echo(json.dumps(stats, indent=2, default=str))
    conn.close()


@cli.command()
def resume_ingest():
    """Show what needs to be fetched next and resume ingestion."""
    conn = init_db()
    from nyt_factor_pipeline.ingest.incremental import compute_resume_plan

    plan = compute_resume_plan(conn)
    click.echo(json.dumps(plan, indent=2, default=str))
    conn.close()


@cli.command()
def show_rate_limit_state():
    """Show current rate-limit state for all APIs."""
    conn = init_db()
    from nyt_factor_pipeline.ingest.request_budget import RequestBudget

    for api in ["archive", "article_search"]:
        budget = RequestBudget(conn, api)
        state = budget.get_state_summary()
        click.echo(f"\n{api}:")
        for k, v in state.items():
            click.echo(f"  {k}: {v}")
    conn.close()


@cli.command()
def score_articles():
    """Score all unscored articles for importance."""
    conn = init_db()
    from nyt_factor_pipeline.scoring.article_importance import score_articles_in_db
    from nyt_factor_pipeline.scoring.article_filtering import compute_macro_relevance

    scored = score_articles_in_db(conn)
    relevant = compute_macro_relevance(conn)
    click.echo(f"Scored {scored} articles")
    click.echo(f"Macro-relevant articles: {relevant}")
    conn.close()


@cli.command()
def rebuild_normalized_text():
    """Rebuild normalized_text for all articles."""
    conn = init_db()
    from nyt_factor_pipeline.utils.text import build_normalized_text

    rows = conn.execute(
        "SELECT article_id, headline_main, abstract, snippet, lead_paragraph, keywords_json FROM articles"
    ).fetchall()

    count = 0
    for article_id, headline, abstract, snippet, lead, kw_json in rows:
        nt = build_normalized_text(headline or "", abstract or "", snippet or "", lead or "", kw_json or "[]")
        conn.execute(
            "UPDATE articles SET normalized_text = ?, updated_at = current_timestamp WHERE article_id = ?",
            [nt, article_id],
        )
        count += 1

    click.echo(f"Rebuilt normalized_text for {count} articles")
    conn.close()


@cli.command()
@click.option("--batch-size", default=64, help="Embedding batch size")
def embed_articles(batch_size: int):
    """Embed articles that pass quality filters."""
    conn = init_db()
    from nyt_factor_pipeline.embeddings.embedder import embed_articles as _embed
    from nyt_factor_pipeline.scoring.article_filtering import get_embeddable_articles

    articles = get_embeddable_articles(conn)
    click.echo(f"Found {len(articles)} articles to embed")

    if articles:
        count = _embed(conn, articles, batch_size=batch_size)
        click.echo(f"Embedded {count} articles")
    conn.close()


@cli.command()
@click.option("--window", default="weekly", type=click.Choice(["weekly", "monthly"]))
@click.option("--start-date", required=True)
@click.option("--end-date", required=True)
def cluster_window(window: str, start_date: str, end_date: str):
    """Run clustering for a date range."""
    conn = init_db()
    from nyt_factor_pipeline.clustering.cluster_weekly import cluster_date_range
    from nyt_factor_pipeline.utils.dates import parse_date

    sd = parse_date(start_date)
    ed = parse_date(end_date)
    clusters = cluster_date_range(conn, sd, ed, window_type=window)
    click.echo(f"Created {len(clusters)} clusters")
    conn.close()


@cli.command()
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
def track_themes(start_date: str | None, end_date: str | None):
    """Match clusters to themes (create or link)."""
    conn = init_db()
    from nyt_factor_pipeline.clustering.topic_tracking import track_themes as _track
    from nyt_factor_pipeline.utils.dates import parse_date

    sd = parse_date(start_date) if start_date else None
    ed = parse_date(end_date) if end_date else None
    stats = _track(conn, sd, ed)
    click.echo(json.dumps(stats, indent=2))
    conn.close()


@cli.command()
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
def build_theme_timeseries(start_date: str | None, end_date: str | None):
    """Build daily theme intensity time series."""
    conn = init_db()
    from nyt_factor_pipeline.themes.timeseries import build_theme_timeseries as _build
    from nyt_factor_pipeline.themes.burst_detection import compute_burst_zscores
    from nyt_factor_pipeline.utils.dates import parse_date

    sd = parse_date(start_date) if start_date else None
    ed = parse_date(end_date) if end_date else None
    count = _build(conn, sd, ed)
    zcount = compute_burst_zscores(conn)
    click.echo(f"Built {count} timeseries rows, computed {zcount} burst z-scores")
    conn.close()


@cli.command()
@click.option("--max-themes", default=20)
def label_new_themes(max_themes: int):
    """Label unlabeled themes using the LLM (requires OPENAI_API_KEY)."""
    conn = init_db()
    from nyt_factor_pipeline.llm.theme_labeling import label_new_themes as _label

    count = _label(conn, max_themes=max_themes)
    click.echo(f"Labeled {count} themes")
    conn.close()


@cli.command()
@click.option("--max-reviews", default=10)
@click.option("--auto-execute", is_flag=True)
def propose_theme_merges(max_reviews: int, auto_execute: bool):
    """Review merge candidates using the LLM (requires OPENAI_API_KEY)."""
    conn = init_db()
    from nyt_factor_pipeline.llm.theme_merge_review import review_merge_candidates

    results = review_merge_candidates(conn, max_reviews=max_reviews, auto_execute=auto_execute)
    for r in results:
        click.echo(f"  {r['label_a']} + {r['label_b']}: {r['decision']} — {r['rationale']}")
    conn.close()


@cli.command()
@click.option("--max-themes", default=20)
def map_themes_rbics(max_themes: int):
    """Map themes to RBICS industries using the LLM (requires OPENAI_API_KEY)."""
    conn = init_db()
    from nyt_factor_pipeline.llm.rbics_mapping import map_themes_to_rbics

    count = map_themes_to_rbics(conn, max_themes=max_themes)
    click.echo(f"Mapped {count} themes to RBICS industries")
    conn.close()


@cli.command()
@click.option("--csv", "csv_path", required=True, help="Path to companies CSV")
def ingest_companies(csv_path: str):
    """Ingest company-RBICS mapping from CSV."""
    conn = init_db()
    from nyt_factor_pipeline.exposures.company_ingest import ingest_companies_csv

    count = ingest_companies_csv(conn, csv_path)
    click.echo(f"Ingested {count} companies")
    conn.close()


@cli.command()
@click.option("--date", "score_date", default=None, help="Date to score (YYYY-MM-DD)")
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
def score_companies(score_date: str | None, start_date: str | None, end_date: str | None):
    """Compute company-theme scores."""
    conn = init_db()
    from nyt_factor_pipeline.exposures.company_ranking import score_companies_for_date, score_companies_range
    from nyt_factor_pipeline.exposures.theme_to_rbics import build_company_theme_exposures
    from nyt_factor_pipeline.utils.dates import parse_date

    # Build exposures first
    exp_count = build_company_theme_exposures(conn)
    click.echo(f"Built {exp_count} company-theme exposures")

    if score_date:
        count = score_companies_for_date(conn, parse_date(score_date))
    elif start_date and end_date:
        count = score_companies_range(conn, parse_date(start_date), parse_date(end_date))
    else:
        click.echo("Provide --date or both --start-date and --end-date")
        conn.close()
        return

    click.echo(f"Computed {count} company-theme scores")
    conn.close()


@cli.command()
@click.option("--output", default="data/artifacts/theme_dashboard.html", help="Output HTML path")
@click.option("--top-n", default=25, help="Max themes to display")
@click.option("--start-date", default=None)
@click.option("--end-date", default=None)
def generate_dashboard(output: str, top_n: int, start_date: str | None, end_date: str | None):
    """Generate an interactive HTML dashboard visualizing themes through time."""
    conn = init_db()
    from nyt_factor_pipeline.utils.dates import parse_date
    from nyt_factor_pipeline.viz.dashboard import generate_dashboard as _gen

    sd = parse_date(start_date) if start_date else None
    ed = parse_date(end_date) if end_date else None

    path = _gen(conn, output_path=output, top_n_themes=top_n, start_date=sd, end_date=ed)
    click.echo(f"Dashboard generated: {path}")
    click.echo(f"Open in browser: file://{path.resolve()}")
    conn.close()


@cli.command()
@click.option("--start", required=True, help="Start month for archive (YYYY-MM)")
@click.option("--end", required=True, help="End month for archive (YYYY-MM)")
@click.option("--force", is_flag=True)
def run_backfill(start: str, end: str, force: bool):
    """Run full backfill: archive ingest -> score -> embed -> cluster -> themes."""
    conn = init_db()

    from nyt_factor_pipeline.clustering.cluster_weekly import cluster_date_range
    from nyt_factor_pipeline.clustering.topic_tracking import track_themes as _track
    from nyt_factor_pipeline.embeddings.embedder import embed_articles as _embed
    from nyt_factor_pipeline.ingest.nyt_archive import ingest_archive as _ingest_archive
    from nyt_factor_pipeline.scoring.article_filtering import compute_macro_relevance, get_embeddable_articles
    from nyt_factor_pipeline.scoring.article_importance import score_articles_in_db
    from nyt_factor_pipeline.themes.burst_detection import compute_burst_zscores
    from nyt_factor_pipeline.themes.timeseries import build_theme_timeseries as _build_ts
    from nyt_factor_pipeline.utils.dates import parse_date

    click.echo("Step 1/7: Ingesting archive data...")
    stats = _ingest_archive(conn, start, end, force=force)
    click.echo(f"  Archive: {stats}")

    click.echo("Step 2/7: Scoring articles...")
    scored = score_articles_in_db(conn)
    relevant = compute_macro_relevance(conn)
    click.echo(f"  Scored: {scored}, Relevant: {relevant}")

    click.echo("Step 3/7: Embedding articles...")
    articles = get_embeddable_articles(conn)
    if articles:
        embedded = _embed(conn, articles)
        click.echo(f"  Embedded: {embedded}")
    else:
        click.echo("  No articles to embed")

    # Determine date range from ingested data
    date_range = conn.execute(
        "SELECT MIN(pub_date), MAX(pub_date) FROM articles WHERE macro_relevance_score > 0"
    ).fetchone()

    if date_range and date_range[0]:
        sd = date_range[0].date() if hasattr(date_range[0], 'date') else date_range[0]
        ed = date_range[1].date() if hasattr(date_range[1], 'date') else date_range[1]

        click.echo("Step 4/7: Clustering...")
        clusters = cluster_date_range(conn, sd, ed)
        click.echo(f"  Clusters: {len(clusters)}")

        click.echo("Step 5/7: Tracking themes...")
        theme_stats = _track(conn)
        click.echo(f"  Themes: {theme_stats}")

        click.echo("Step 6/7: Building timeseries...")
        ts_count = _build_ts(conn, sd, ed)
        click.echo(f"  Timeseries rows: {ts_count}")

        click.echo("Step 7/7: Computing burst detection...")
        burst_count = compute_burst_zscores(conn)
        click.echo(f"  Burst z-scores: {burst_count}")
        # Generate dashboard
        click.echo("Generating dashboard...")
        from nyt_factor_pipeline.viz.dashboard import generate_dashboard as _gen_dash
        path = _gen_dash(conn, start_date=sd, end_date=ed)
        click.echo(f"  Dashboard: {path}")
    else:
        click.echo("  No articles found for clustering")

    click.echo("Backfill complete!")
    conn.close()


@cli.command()
@click.option("--end-date", default=None, help="End date (default: today)")
@click.option("--sync-supabase", is_flag=True, help="Push results to Supabase/PostgreSQL after update")
def run_incremental_update(end_date: str | None, sync_supabase: bool):
    """Run incremental update: ingest recent -> score -> embed -> cluster -> track."""
    conn = init_db()

    from nyt_factor_pipeline.clustering.cluster_weekly import cluster_date_range
    from nyt_factor_pipeline.clustering.topic_tracking import track_themes as _track
    from nyt_factor_pipeline.embeddings.embedder import embed_articles as _embed
    from nyt_factor_pipeline.ingest.incremental import compute_resume_plan
    from nyt_factor_pipeline.ingest.nyt_article_search import ingest_article_search
    from nyt_factor_pipeline.scoring.article_filtering import compute_macro_relevance, get_embeddable_articles
    from nyt_factor_pipeline.scoring.article_importance import score_articles_in_db
    from nyt_factor_pipeline.themes.burst_detection import compute_burst_zscores
    from nyt_factor_pipeline.themes.timeseries import build_theme_timeseries as _build_ts
    from nyt_factor_pipeline.utils.dates import parse_date

    ed = parse_date(end_date) if end_date else date.today()

    plan = compute_resume_plan(conn)
    click.echo(f"Resume plan: {json.dumps(plan, indent=2, default=str)}")

    # Ingest recent tail
    as_plan = plan.get("article_search", {})
    if as_plan.get("status") == "behind":
        sd = parse_date(as_plan["start_date"])
        click.echo(f"Ingesting articles from {sd} to {ed}...")
        stats = ingest_article_search(conn, sd, ed)
        click.echo(f"  {stats}")

    click.echo("Scoring articles...")
    score_articles_in_db(conn)
    compute_macro_relevance(conn)

    click.echo("Embedding new articles...")
    articles = get_embeddable_articles(conn)
    if articles:
        _embed(conn, articles)

    # Cluster recent data
    recent_start = conn.execute(
        """SELECT MIN(pub_date) FROM articles
           WHERE macro_relevance_score > 0
             AND article_id NOT IN (SELECT article_id FROM article_cluster_membership)"""
    ).fetchone()

    if recent_start and recent_start[0]:
        rs = recent_start[0].date() if hasattr(recent_start[0], 'date') else recent_start[0]
        click.echo(f"Clustering from {rs} to {ed}...")
        cluster_date_range(conn, rs, ed)
        _track(conn, rs, ed)
        _build_ts(conn, rs, ed)
        compute_burst_zscores(conn)

        # Regenerate dashboard
        from nyt_factor_pipeline.viz.dashboard import generate_dashboard as _gen_dash
        path = _gen_dash(conn)
        click.echo(f"Dashboard updated: {path}")

    # Sync to Supabase/PostgreSQL if requested
    if sync_supabase:
        settings = get_settings()
        if not settings.database_url:
            click.echo("ERROR: DATABASE_URL not set. Skipping Supabase sync.")
        else:
            click.echo("Syncing to Supabase/PostgreSQL...")
            from nyt_factor_pipeline.export.supabase_sync import sync_all
            results = sync_all(conn, settings.database_url)
            for table, count in results.items():
                click.echo(f"  {table}: {count} rows")

    click.echo("Incremental update complete!")
    conn.close()


@cli.command()
def sync_supabase():
    """Push all DuckDB data to Supabase/PostgreSQL (DATABASE_URL)."""
    settings = get_settings()
    if not settings.database_url:
        click.echo("ERROR: Set DATABASE_URL in .env (e.g. postgresql://user:pass@host:5432/db)")
        return

    conn = init_db()
    from nyt_factor_pipeline.export.supabase_sync import sync_all

    click.echo(f"Syncing to PostgreSQL...")
    results = sync_all(conn, settings.database_url, incremental=True)
    for table, count in results.items():
        click.echo(f"  {table}: {count} rows")

    click.echo("Sync complete!")
    conn.close()


@cli.command()
@click.option("--full", is_flag=True, help="Full sync (ignore last_synced_at)")
@click.option("--tables", default=None, help="Comma-separated table names to sync")
def sync_supabase_full(full: bool, tables: str | None):
    """Full re-sync of DuckDB data to Supabase/PostgreSQL."""
    settings = get_settings()
    if not settings.database_url:
        click.echo("ERROR: Set DATABASE_URL in .env")
        return

    conn = init_db()
    from nyt_factor_pipeline.export.supabase_sync import sync_all

    table_list = [t.strip() for t in tables.split(",")] if tables else None
    click.echo("Full sync to PostgreSQL...")
    results = sync_all(conn, settings.database_url, incremental=not full, tables=table_list)
    for table, count in results.items():
        click.echo(f"  {table}: {count} rows")

    click.echo("Sync complete!")
    conn.close()


if __name__ == "__main__":
    cli()
