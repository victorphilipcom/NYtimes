"""Tests for topic tracking and theme management."""

import pickle
from datetime import date

import duckdb
import numpy as np

from nyt_factor_pipeline.db import init_schema


def _get_test_db():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    return conn


class TestTopicTracking:
    def test_new_theme_creation(self):
        """When no existing themes, a cluster should create a new theme."""
        conn = _get_test_db()
        centroid = np.random.randn(10).astype(np.float32)

        # Insert a cluster
        conn.execute(
            """INSERT INTO clusters_raw
               (cluster_id, window_start, window_end, article_count, centroid,
                top_keywords_json, representative_article_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["c1", date(2024, 1, 1), date(2024, 1, 7), 10,
             pickle.dumps(centroid), '["economy", "gdp"]', '["a1", "a2"]'],
        )

        from nyt_factor_pipeline.clustering.topic_tracking import track_themes
        stats = track_themes(conn)

        assert stats["new_themes"] == 1
        assert stats["linked"] == 0

        # Verify theme was created
        themes = conn.execute("SELECT theme_id, current_label FROM themes").fetchall()
        assert len(themes) == 1
        assert "economy" in themes[0][1].lower()

    def test_linking_to_existing_theme(self):
        """A cluster similar to an existing theme should link to it."""
        conn = _get_test_db()
        base_centroid = np.random.randn(10).astype(np.float32)
        base_centroid = base_centroid / np.linalg.norm(base_centroid)

        # Create existing theme
        conn.execute(
            """INSERT INTO themes
               (theme_id, current_label, centroid, first_seen, last_seen, active_flag, metadata_json)
               VALUES (?, ?, ?, ?, ?, true, '{}')""",
            ["t1", "Economy", pickle.dumps(base_centroid),
             date(2024, 1, 1), date(2024, 1, 7)],
        )

        # Insert a very similar cluster
        similar_centroid = base_centroid + np.random.randn(10).astype(np.float32) * 0.01
        similar_centroid = similar_centroid / np.linalg.norm(similar_centroid)

        conn.execute(
            """INSERT INTO clusters_raw
               (cluster_id, window_start, window_end, article_count, centroid,
                top_keywords_json, representative_article_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["c2", date(2024, 1, 8), date(2024, 1, 14), 8,
             pickle.dumps(similar_centroid), '["economy", "growth"]', '["a3"]'],
        )

        from nyt_factor_pipeline.clustering.topic_tracking import track_themes
        stats = track_themes(conn)

        assert stats["linked"] == 1
        assert stats["new_themes"] == 0

    def test_dissimilar_cluster_creates_new_theme(self):
        """A very different cluster should create a new theme."""
        conn = _get_test_db()
        centroid_a = np.zeros(10, dtype=np.float32)
        centroid_a[0] = 1.0

        conn.execute(
            """INSERT INTO themes
               (theme_id, current_label, centroid, first_seen, last_seen, active_flag, metadata_json)
               VALUES (?, ?, ?, ?, ?, true, '{}')""",
            ["t1", "Economy", pickle.dumps(centroid_a),
             date(2024, 1, 1), date(2024, 1, 7)],
        )

        # Very different centroid
        centroid_b = np.zeros(10, dtype=np.float32)
        centroid_b[9] = 1.0

        conn.execute(
            """INSERT INTO clusters_raw
               (cluster_id, window_start, window_end, article_count, centroid,
                top_keywords_json, representative_article_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ["c2", date(2024, 1, 8), date(2024, 1, 14), 5,
             pickle.dumps(centroid_b), '["technology", "ai"]', '["a4"]'],
        )

        from nyt_factor_pipeline.clustering.topic_tracking import track_themes
        stats = track_themes(conn)

        assert stats["new_themes"] == 1
        themes = conn.execute("SELECT COUNT(*) FROM themes").fetchone()
        assert themes[0] == 2
