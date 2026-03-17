"""Persistent checkpoint management for resumable ingestion."""

from __future__ import annotations

import json
from datetime import datetime

import duckdb

from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


class CheckpointManager:
    """Manages checkpoints in DuckDB for resumable ingestion."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, source_api: str):
        self.conn = conn
        self.source_api = source_api

    def is_completed(self, key: str) -> bool:
        """Check if a checkpoint key has been marked complete."""
        result = self.conn.execute(
            """SELECT checkpoint_value_json FROM ingest_checkpoints
               WHERE source_api = ? AND checkpoint_key = ?""",
            [self.source_api, key],
        ).fetchone()
        if result is None:
            return False
        try:
            val = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            return val.get("completed", False)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    def mark_completed(self, key: str, metadata: dict | None = None) -> None:
        """Mark a checkpoint as completed."""
        value = {"completed": True, **(metadata or {})}
        now = datetime.utcnow()
        self.conn.execute(
            """INSERT INTO ingest_checkpoints (source_api, checkpoint_key, checkpoint_value_json, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (source_api, checkpoint_key)
               DO UPDATE SET checkpoint_value_json = ?, updated_at = ?""",
            [self.source_api, key, json.dumps(value), now, json.dumps(value), now],
        )
        log.info("checkpoint_saved", source=self.source_api, key=key)

    def get_value(self, key: str) -> dict | None:
        """Get checkpoint value."""
        result = self.conn.execute(
            """SELECT checkpoint_value_json FROM ingest_checkpoints
               WHERE source_api = ? AND checkpoint_key = ?""",
            [self.source_api, key],
        ).fetchone()
        if result is None:
            return None
        try:
            val = result[0]
            return json.loads(val) if isinstance(val, str) else val
        except (json.JSONDecodeError, TypeError):
            return None

    def set_value(self, key: str, value: dict) -> None:
        """Set checkpoint value (partial progress)."""
        now = datetime.utcnow()
        self.conn.execute(
            """INSERT INTO ingest_checkpoints (source_api, checkpoint_key, checkpoint_value_json, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (source_api, checkpoint_key)
               DO UPDATE SET checkpoint_value_json = ?, updated_at = ?""",
            [self.source_api, key, json.dumps(value), now, json.dumps(value), now],
        )

    def list_checkpoints(self) -> list[dict]:
        """List all checkpoints for this source."""
        rows = self.conn.execute(
            """SELECT checkpoint_key, checkpoint_value_json, updated_at
               FROM ingest_checkpoints WHERE source_api = ?
               ORDER BY checkpoint_key""",
            [self.source_api],
        ).fetchall()
        results = []
        for key, val_json, updated in rows:
            try:
                val = json.loads(val_json) if isinstance(val_json, str) else val_json
            except (json.JSONDecodeError, TypeError):
                val = {}
            results.append({"key": key, "value": val, "updated_at": updated})
        return results

    def clear(self, key: str) -> None:
        """Remove a checkpoint (for force re-ingestion)."""
        self.conn.execute(
            "DELETE FROM ingest_checkpoints WHERE source_api = ? AND checkpoint_key = ?",
            [self.source_api, key],
        )
