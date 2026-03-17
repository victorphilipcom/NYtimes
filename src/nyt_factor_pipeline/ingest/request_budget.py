"""Request budget and rate-limit tracking for NYT API calls."""

from __future__ import annotations

import time
from datetime import date, datetime

import duckdb

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


class RequestBudget:
    """Tracks and enforces rate limits and daily budgets for an API.

    Persists state in DuckDB so limits survive restarts.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        api_name: str,
        daily_budget: int | None = None,
        requests_per_minute: int | None = None,
    ):
        self.conn = conn
        self.api_name = api_name
        settings = get_settings()

        if api_name == "archive":
            self.daily_budget = daily_budget or settings.nyt_archive_daily_budget
        elif api_name == "article_search":
            self.daily_budget = daily_budget or settings.nyt_article_search_daily_budget
        else:
            self.daily_budget = daily_budget or 1000

        self.requests_per_minute = requests_per_minute or settings.nyt_requests_per_minute
        self._last_request_time: float = 0.0

    def _today(self) -> date:
        return date.today()

    def _minute_bucket(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M")

    def get_requests_today(self) -> int:
        """Load total requests today from persistent state."""
        today = self._today()
        result = self.conn.execute(
            """SELECT COALESCE(SUM(requests_this_minute), 0)
               FROM rate_limit_state
               WHERE api_name = ? AND date = ?""",
            [self.api_name, today],
        ).fetchone()
        return result[0] if result else 0

    def get_requests_this_minute(self) -> int:
        bucket = self._minute_bucket()
        today = self._today()
        result = self.conn.execute(
            """SELECT requests_this_minute FROM rate_limit_state
               WHERE api_name = ? AND date = ? AND minute_bucket = ?""",
            [self.api_name, today, bucket],
        ).fetchone()
        return result[0] if result else 0

    def budget_exhausted(self) -> bool:
        return self.get_requests_today() >= self.daily_budget

    def can_make_request(self) -> bool:
        if self.budget_exhausted():
            return False
        if self.get_requests_this_minute() >= self.requests_per_minute:
            return False
        return True

    def wait_if_needed(self) -> None:
        """Block until a request is allowed by rate limits."""
        while not self.can_make_request():
            if self.budget_exhausted():
                log.warning("daily_budget_exhausted", api=self.api_name, budget=self.daily_budget)
                raise BudgetExhaustedError(
                    f"{self.api_name} daily budget of {self.daily_budget} exhausted"
                )
            # Wait for next minute bucket
            now = time.time()
            seconds_until_next_minute = 60 - (now % 60)
            wait = min(seconds_until_next_minute + 0.5, 20)
            log.debug("rate_limit_wait", seconds=round(wait, 1))
            time.sleep(wait)

        # Also enforce minimum inter-request delay
        min_interval = 60.0 / self.requests_per_minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def record_request(self) -> None:
        """Record that a request was made. Updates persistent state."""
        self._last_request_time = time.time()
        today = self._today()
        bucket = self._minute_bucket()
        now = datetime.utcnow()

        # Upsert into rate_limit_state
        self.conn.execute(
            """INSERT INTO rate_limit_state (api_name, date, requests_today, minute_bucket, requests_this_minute, updated_at)
               VALUES (?, ?, 1, ?, 1, ?)
               ON CONFLICT (api_name, date, minute_bucket)
               DO UPDATE SET
                   requests_this_minute = rate_limit_state.requests_this_minute + 1,
                   requests_today = rate_limit_state.requests_today + 1,
                   updated_at = ?""",
            [self.api_name, today, bucket, now, now],
        )

    def get_state_summary(self) -> dict:
        """Return a summary of current rate-limit state."""
        return {
            "api_name": self.api_name,
            "date": str(self._today()),
            "requests_today": self.get_requests_today(),
            "daily_budget": self.daily_budget,
            "requests_per_minute": self.requests_per_minute,
            "budget_remaining": max(0, self.daily_budget - self.get_requests_today()),
        }


class BudgetExhaustedError(Exception):
    pass
