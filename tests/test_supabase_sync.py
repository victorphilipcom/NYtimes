"""Tests for the Supabase/PostgreSQL sync module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nyt_factor_pipeline.export.supabase_sync import (
    BATCH_SIZE,
    _SYNC_TABLES,
    _coerce_value,
)


class TestCoerceValue:
    def test_none(self):
        assert _coerce_value(None) is None

    def test_string(self):
        assert _coerce_value("hello") == "hello"

    def test_int(self):
        assert _coerce_value(42) == 42

    def test_float(self):
        assert _coerce_value(3.14) == 3.14

    def test_dict_to_json(self):
        result = _coerce_value({"key": "value"})
        assert result == '{"key": "value"}'

    def test_list_to_json(self):
        result = _coerce_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_bytes_returns_none(self):
        assert _coerce_value(b"\x00\x01\x02") is None

    def test_bool(self):
        assert _coerce_value(True) is True
        assert _coerce_value(False) is False


class TestSyncTableSpecs:
    def test_all_tables_have_required_keys(self):
        for spec in _SYNC_TABLES:
            assert "name" in spec
            assert "query" in spec
            assert "pk" in spec
            assert "columns" in spec

    def test_table_names_unique(self):
        names = [s["name"] for s in _SYNC_TABLES]
        assert len(names) == len(set(names))

    def test_articles_table_present(self):
        names = [s["name"] for s in _SYNC_TABLES]
        assert "articles" in names
        assert "themes" in names
        assert "theme_timeseries" in names

    def test_columns_are_lists(self):
        for spec in _SYNC_TABLES:
            assert isinstance(spec["columns"], list)
            assert len(spec["columns"]) > 0


class TestGetPgConnection:
    def test_missing_psycopg2_raises(self):
        with patch.dict("sys.modules", {"psycopg2": None}):
            from nyt_factor_pipeline.export.supabase_sync import _get_pg_connection
            with pytest.raises(ImportError, match="psycopg2"):
                _get_pg_connection("postgresql://localhost/test")


class TestBatchSize:
    def test_batch_size_reasonable(self):
        assert BATCH_SIZE > 0
        assert BATCH_SIZE <= 10000
