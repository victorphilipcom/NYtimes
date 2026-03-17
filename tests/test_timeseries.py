"""Tests for theme timeseries and burst detection."""

import pickle
from datetime import date, datetime

import duckdb
import numpy as np

from nyt_factor_pipeline.db import init_schema


def _get_test_db():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    return conn


def _seed_data(conn):
    """Seed test data for timeseries tests."""
    centroid = pickle.dumps(np.random.randn(10).astype(np.float32))

    # Theme
    conn.execute(
        """INSERT INTO themes (theme_id, current_label, centroid, first_seen, last_seen, active_flag, metadata_json)
           VALUES ('t1', 'Test Theme', ?, '2024-01-01', '2024-01-14', true, '{}')""",
        [centroid],
    )

    # Clusters
    for i, (ws, we) in enumerate([
        (date(2024, 1, 1), date(2024, 1, 7)),
        (date(2024, 1, 8), date(2024, 1, 14)),
    ]):
        cid = f"c{i}"
        conn.execute(
            """INSERT INTO clusters_raw
               (cluster_id, window_start, window_end, article_count, centroid, top_keywords_json, representative_article_ids)
               VALUES (?, ?, ?, 5, ?, '[]', '[]')""",
            [cid, ws, we, centroid],
        )
        conn.execute(
            """INSERT INTO cluster_theme_link
               (cluster_id, window_start, window_end, theme_id, similarity_score, action)
               VALUES (?, ?, ?, 't1', 0.9, 'linked')""",
            [cid, ws, we],
        )

    # Articles
    for day_offset in range(14):
        d = date(2024, 1, 1 + day_offset)
        aid = f"a{day_offset}"
        conn.execute(
            """INSERT INTO articles
               (article_id, source_api, pub_date, year, month, day,
                headline_main, importance_score, macro_relevance_score, normalized_text, word_count)
               VALUES (?, 'test', ?, 2024, 1, ?, 'Test', 0.8, 1.0, 'text', 500)""",
            [aid, datetime(2024, 1, 1 + day_offset), 1 + day_offset],
        )

        # Assign to appropriate cluster
        cid = "c0" if day_offset < 7 else "c1"
        ws = date(2024, 1, 1) if day_offset < 7 else date(2024, 1, 8)
        we = date(2024, 1, 7) if day_offset < 7 else date(2024, 1, 14)
        conn.execute(
            """INSERT INTO article_cluster_membership
               (article_id, cluster_id, window_start, window_end, membership_score)
               VALUES (?, ?, ?, ?, 0.9)""",
            [aid, cid, ws, we],
        )


class TestTimeseries:
    def test_build_timeseries(self):
        conn = _get_test_db()
        _seed_data(conn)

        from nyt_factor_pipeline.themes.timeseries import build_theme_timeseries

        count = build_theme_timeseries(conn)
        assert count > 0

        rows = conn.execute(
            "SELECT * FROM theme_timeseries WHERE theme_id = 't1' ORDER BY date"
        ).fetchall()
        assert len(rows) == 14  # 14 days

    def test_get_theme_timeseries(self):
        conn = _get_test_db()
        _seed_data(conn)

        from nyt_factor_pipeline.themes.timeseries import build_theme_timeseries, get_theme_timeseries

        build_theme_timeseries(conn)
        ts = get_theme_timeseries(conn, "t1")
        assert len(ts) == 14
        assert all("intensity" in row for row in ts)


class TestBurstDetection:
    def test_compute_burst_zscores(self):
        conn = _get_test_db()
        _seed_data(conn)

        from nyt_factor_pipeline.themes.timeseries import build_theme_timeseries
        from nyt_factor_pipeline.themes.burst_detection import compute_burst_zscores

        build_theme_timeseries(conn)
        count = compute_burst_zscores(conn)
        assert count > 0

        # Check that z-scores were written
        rows = conn.execute(
            "SELECT burst_zscore FROM theme_timeseries WHERE theme_id = 't1'"
        ).fetchall()
        assert any(r[0] is not None for r in rows)
