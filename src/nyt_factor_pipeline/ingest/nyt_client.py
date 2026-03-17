"""Reusable NYT API HTTP client with rate limiting, retries, and request logging."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

import duckdb
import httpx

from nyt_factor_pipeline.config import get_settings
from nyt_factor_pipeline.ingest.request_budget import BudgetExhaustedError, RequestBudget
from nyt_factor_pipeline.logging_utils import get_logger
from nyt_factor_pipeline.utils.retry import RetryableError, retry_with_backoff

log = get_logger(__name__)

NYT_BASE_URL = "https://api.nytimes.com/svc"


class NYTClient:
    """HTTP client for NYT APIs with built-in rate limiting, retries, and logging."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        api_name: str,
        daily_budget: int | None = None,
        requests_per_minute: int | None = None,
    ):
        self.conn = conn
        self.api_name = api_name
        self.settings = get_settings()
        self.budget = RequestBudget(
            conn, api_name,
            daily_budget=daily_budget,
            requests_per_minute=requests_per_minute,
        )
        self._http = httpx.Client(timeout=30.0)

    def get(self, url: str, params: dict | None = None) -> dict:
        """Make a GET request with rate limiting, retries, and logging.

        Returns the parsed JSON response.
        Raises BudgetExhaustedError if daily budget is used up.
        """
        params = dict(params or {})
        params["api-key"] = self.settings.nyt_api_key

        self.budget.wait_if_needed()

        request_id = uuid.uuid4().hex[:16]
        start = time.time()
        retry_count = 0

        def _do_request() -> dict:
            nonlocal retry_count
            response = self._http.get(url, params=params)
            elapsed_ms = int((time.time() - start) * 1000)

            # Log the request
            self._log_request(
                request_id=request_id,
                url=url,
                params={k: v for k, v in params.items() if k != "api-key"},
                status=response.status_code,
                retry_count=retry_count,
                elapsed_ms=elapsed_ms,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                ra_val = float(retry_after) if retry_after else None
                retry_count += 1
                raise RetryableError(
                    f"Rate limited (429) from {self.api_name}",
                    status_code=429,
                    retry_after=ra_val,
                )
            if response.status_code >= 500:
                retry_count += 1
                raise RetryableError(
                    f"Server error ({response.status_code}) from {self.api_name}",
                    status_code=response.status_code,
                )
            response.raise_for_status()
            return response.json()

        try:
            result = retry_with_backoff(
                _do_request,
                max_retries=self.settings.nyt_max_retries,
            )
            self.budget.record_request()
            return result
        except BudgetExhaustedError:
            raise
        except RetryableError:
            # Exhausted retries
            self.budget.record_request()
            raise

    def _log_request(
        self,
        request_id: str,
        url: str,
        params: dict,
        status: int,
        retry_count: int,
        elapsed_ms: int,
    ) -> None:
        now = datetime.utcnow()
        try:
            self.conn.execute(
                """INSERT INTO api_request_log
                   (request_id, api_name, requested_at, url, params_json,
                    response_status, retry_count, elapsed_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    request_id,
                    self.api_name,
                    now,
                    url,
                    json.dumps(params),
                    status,
                    retry_count,
                    elapsed_ms,
                ],
            )
        except Exception as e:
            log.warning("request_log_failed", error=str(e))

    def close(self) -> None:
        self._http.close()
