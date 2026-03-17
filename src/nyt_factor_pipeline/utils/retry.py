"""Retry helpers with exponential backoff and jitter."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    retry_after_header: str | None = None,
) -> T:
    """Execute fn with retries on transient errors.

    fn should raise RetryableError with status_code and optional retry_after
    for retryable failures.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RetryableError as e:
            last_exc = e
            if e.status_code not in retryable_statuses:
                raise
            if attempt == max_retries:
                raise

            if e.retry_after and e.retry_after > 0:
                delay = e.retry_after
            else:
                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, delay * 0.25)  # jitter

            log.warning(
                "retrying_request",
                attempt=attempt + 1,
                max_retries=max_retries,
                status=e.status_code,
                delay=round(delay, 1),
            )
            time.sleep(delay)
        except Exception:
            raise
    raise last_exc  # type: ignore[misc]


class RetryableError(Exception):
    """Raised for HTTP errors that should trigger a retry."""

    def __init__(self, message: str, status_code: int, retry_after: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
