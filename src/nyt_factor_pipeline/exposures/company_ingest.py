"""Ingest company-RBICS mapping from user-provided CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb

from nyt_factor_pipeline.exposures.rbics_schema import validate_csv_columns
from nyt_factor_pipeline.logging_utils import get_logger

log = get_logger(__name__)


def ingest_companies_csv(
    conn: duckdb.DuckDBPyConnection,
    csv_path: str | Path,
) -> int:
    """Ingest companies and RBICS mappings from a CSV file.

    Expected columns: company_id, ticker, company_name, rbics_code, rbics_name

    Returns count of companies ingested.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])

        missing = validate_csv_columns(columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")

        count = 0
        for row in reader:
            conn.execute(
                """INSERT INTO companies
                   (company_id, ticker, company_name, rbics_code, rbics_name, metadata_json)
                   VALUES (?, ?, ?, ?, ?, '{}')
                   ON CONFLICT (company_id) DO UPDATE SET
                       ticker = ?, company_name = ?, rbics_code = ?, rbics_name = ?""",
                [
                    row["company_id"],
                    row["ticker"],
                    row["company_name"],
                    row["rbics_code"],
                    row["rbics_name"],
                    row["ticker"],
                    row["company_name"],
                    row["rbics_code"],
                    row["rbics_name"],
                ],
            )
            count += 1

    log.info("companies_ingested", count=count, path=str(csv_path))
    return count
