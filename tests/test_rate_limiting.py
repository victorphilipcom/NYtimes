"""Tests for rate limiting and request budget management."""

from datetime import date

import duckdb

from nyt_factor_pipeline.db import init_schema


def _get_test_db():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    return conn


class TestRequestBudget:
    def test_initial_state(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.request_budget import RequestBudget

        budget = RequestBudget(conn, "archive", daily_budget=100, requests_per_minute=5)
        assert budget.get_requests_today() == 0
        assert not budget.budget_exhausted()
        assert budget.can_make_request()

    def test_record_request(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.request_budget import RequestBudget

        budget = RequestBudget(conn, "archive", daily_budget=100, requests_per_minute=5)
        budget.record_request()
        assert budget.get_requests_today() == 1

    def test_budget_exhaustion(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.request_budget import RequestBudget

        budget = RequestBudget(conn, "archive", daily_budget=3, requests_per_minute=100)

        for _ in range(3):
            budget.record_request()

        assert budget.budget_exhausted()
        assert not budget.can_make_request()

    def test_state_summary(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.request_budget import RequestBudget

        budget = RequestBudget(conn, "archive", daily_budget=100, requests_per_minute=5)
        budget.record_request()
        budget.record_request()

        summary = budget.get_state_summary()
        assert summary["api_name"] == "archive"
        assert summary["requests_today"] == 2
        assert summary["daily_budget"] == 100
        assert summary["budget_remaining"] == 98


class TestCheckpointManager:
    def test_checkpoint_lifecycle(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.checkpoints import CheckpointManager

        cm = CheckpointManager(conn, "archive")

        assert not cm.is_completed("2024-01")
        cm.mark_completed("2024-01", {"article_count": 500})
        assert cm.is_completed("2024-01")

    def test_get_value(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.checkpoints import CheckpointManager

        cm = CheckpointManager(conn, "archive")
        cm.mark_completed("2024-01", {"article_count": 500})

        val = cm.get_value("2024-01")
        assert val is not None
        assert val["completed"] is True
        assert val["article_count"] == 500

    def test_clear_checkpoint(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.checkpoints import CheckpointManager

        cm = CheckpointManager(conn, "archive")
        cm.mark_completed("2024-01")
        assert cm.is_completed("2024-01")

        cm.clear("2024-01")
        assert not cm.is_completed("2024-01")

    def test_list_checkpoints(self):
        conn = _get_test_db()
        from nyt_factor_pipeline.ingest.checkpoints import CheckpointManager

        cm = CheckpointManager(conn, "archive")
        cm.mark_completed("2024-01")
        cm.mark_completed("2024-02")

        checkpoints = cm.list_checkpoints()
        assert len(checkpoints) == 2


class TestAdaptiveWindowSplitter:
    def test_date_windows(self):
        from nyt_factor_pipeline.utils.dates import date_windows

        windows = date_windows(date(2024, 1, 1), date(2024, 1, 10), window_days=3)
        assert len(windows) == 4
        assert windows[0] == (date(2024, 1, 1), date(2024, 1, 3))
        assert windows[-1] == (date(2024, 1, 10), date(2024, 1, 10))

    def test_week_windows(self):
        from nyt_factor_pipeline.utils.dates import week_windows

        windows = week_windows(date(2024, 1, 1), date(2024, 1, 31))
        assert len(windows) >= 4
        for start, end in windows:
            assert start <= end
            assert (end - start).days <= 6

    def test_month_range(self):
        from nyt_factor_pipeline.utils.dates import month_range

        months = month_range("2024-01", "2024-06")
        assert len(months) == 6
        assert months[0] == (2024, 1)
        assert months[-1] == (2024, 6)

    def test_estimate_archive_requests(self):
        from nyt_factor_pipeline.ingest.nyt_archive import estimate_archive_requests

        est = estimate_archive_requests("2020-01", "2024-12")
        assert est == 60  # 5 years * 12 months
