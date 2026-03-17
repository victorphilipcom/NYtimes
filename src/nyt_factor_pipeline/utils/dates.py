"""Date utilities for the pipeline."""

from __future__ import annotations

from datetime import date, datetime, timedelta


def parse_month(s: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month)."""
    parts = s.strip().split("-")
    return int(parts[0]), int(parts[1])


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Generate list of (year, month) from start to end inclusive. Format: 'YYYY-MM'."""
    sy, sm = parse_month(start)
    ey, em = parse_month(end)
    result = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        result.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return result


def date_windows(
    start: date, end: date, window_days: int = 1
) -> list[tuple[date, date]]:
    """Split a date range into windows of window_days size."""
    windows = []
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=window_days - 1), end)
        windows.append((current, window_end))
        current = window_end + timedelta(days=1)
    return windows


def week_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into ISO week windows (Mon-Sun)."""
    windows = []
    current = start - timedelta(days=start.weekday())  # back to Monday
    while current <= end:
        w_end = current + timedelta(days=6)
        w_start = max(current, start)
        w_end = min(w_end, end)
        if w_start <= w_end:
            windows.append((w_start, w_end))
        current += timedelta(days=7)
    return windows


def parse_date(s: str) -> date:
    """Parse YYYY-MM-DD string to date."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def format_month(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"
