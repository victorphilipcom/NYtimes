"""RBICS schema definition and helpers.

RBICS (FactSet Revere Business Industry Classification System) data is proprietary.
This module defines the schema for user-provided RBICS mappings and provides
helpers for matching LLM-suggested industries to actual RBICS codes.
"""

from __future__ import annotations

# Example RBICS hierarchy levels:
# L1: Economy, L2: Sector, L3: Sub-Sector, L4: Industry, L5: Sub-Industry, L6: Activity
RBICS_LEVELS = ["economy", "sector", "sub_sector", "industry", "sub_industry", "activity"]

# Example CSV schema for user-provided company-RBICS mapping:
EXPECTED_CSV_COLUMNS = [
    "company_id",   # unique identifier
    "ticker",       # stock ticker
    "company_name", # full company name
    "rbics_code",   # RBICS classification code
    "rbics_name",   # human-readable RBICS name
]

EXAMPLE_CSV = """company_id,ticker,company_name,rbics_code,rbics_name
AAPL,AAPL,Apple Inc,tech_consumer_electronics,Consumer Electronics
MSFT,MSFT,Microsoft Corp,tech_software_infrastructure,Software - Infrastructure
JPM,JPM,JPMorgan Chase,fin_commercial_banking,Commercial Banking
XOM,XOM,Exxon Mobil,energy_oil_gas_integrated,Oil & Gas - Integrated
"""


def validate_csv_columns(columns: list[str]) -> list[str]:
    """Check that required columns are present. Returns list of missing columns."""
    return [c for c in EXPECTED_CSV_COLUMNS if c not in columns]
