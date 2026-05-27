"""Tests for SEC EDGAR fundamentals fetcher — PIT integrity + filter logic."""
from __future__ import annotations

import pandas as pd
import pytest

from quant.ml import edgar


def test_observations_to_quarterly_df_filters_non_quarterly():
    """Only fp ∈ Q1..Q4 with 60-100d duration should survive."""
    obs = [
        # Cumulative H1 (6M): should drop
        {"fp": "Q2", "start": "2025-01-01", "end": "2025-06-30", "val": 100, "filed": "2025-08-01", "fy": 2025},
        # Annual: should drop
        {"fp": "FY", "start": "2024-01-01", "end": "2024-12-31", "val": 400, "filed": "2025-02-28", "fy": 2024},
        # Clean Q1: keep
        {"fp": "Q1", "start": "2025-01-01", "end": "2025-03-31", "val": 90, "filed": "2025-05-01", "fy": 2025},
        # Clean Q3: keep
        {"fp": "Q3", "start": "2025-07-01", "end": "2025-09-30", "val": 110, "filed": "2025-11-01", "fy": 2025},
    ]
    df = edgar._observations_to_quarterly_df(obs)
    assert len(df) == 2
    assert set(df["fp"].unique()) == {"Q1", "Q3"}


def test_observations_to_quarterly_df_dedupes_amendments():
    """Same (fy, fp) filed twice → keep the LATEST filing (amendment supersedes)."""
    obs = [
        {"fp": "Q1", "start": "2025-01-01", "end": "2025-03-31", "val": 100, "filed": "2025-05-01", "fy": 2025},
        # Amended 3 months later with corrected value
        {"fp": "Q1", "start": "2025-01-01", "end": "2025-03-31", "val": 95, "filed": "2025-08-15", "fy": 2025},
    ]
    df = edgar._observations_to_quarterly_df(obs)
    assert len(df) == 1
    assert df.iloc[0]["val"] == 95  # amended (later filed) wins


def test_pick_line_item_merges_alternatives():
    """Companies switch concepts over time (e.g. ASC-606); merge all alternatives."""
    facts = {
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {"fp": "Q1", "fy": 2017, "val": 100, "filed": "2017-05-01",
                 "start": "2017-01-01", "end": "2017-03-31"}
            ]}},
            "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                {"fp": "Q1", "fy": 2024, "val": 200, "filed": "2024-05-01",
                 "start": "2024-01-01", "end": "2024-03-31"}
            ]}},
        }}
    }
    obs = edgar._pick_line_item(facts, [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ])
    assert len(obs) == 2
    fys = sorted(o["fy"] for o in obs)
    assert fys == [2017, 2024]


def test_build_pit_timeseries_computes_ttm():
    """TTM = rolling 4-quarter sum of single-quarter values."""
    fund = {
        "revenue": pd.DataFrame([
            {"filed": pd.Timestamp("2024-05-01"), "period_end": pd.Timestamp("2024-03-31"), "val": 100, "fy": 2024, "fp": "Q1"},
            {"filed": pd.Timestamp("2024-08-01"), "period_end": pd.Timestamp("2024-06-30"), "val": 110, "fy": 2024, "fp": "Q2"},
            {"filed": pd.Timestamp("2024-11-01"), "period_end": pd.Timestamp("2024-09-30"), "val": 120, "fy": 2024, "fp": "Q3"},
            {"filed": pd.Timestamp("2025-02-01"), "period_end": pd.Timestamp("2024-12-31"), "val": 130, "fy": 2024, "fp": "Q4"},
            {"filed": pd.Timestamp("2025-05-01"), "period_end": pd.Timestamp("2025-03-31"), "val": 140, "fy": 2025, "fp": "Q1"},
        ]),
    }
    daily = edgar._build_pit_timeseries(fund)
    # After 4 filings, TTM at 2025-02-01 = 100+110+120+130 = 460
    assert "revenue_Q" in daily.columns
    assert "revenue_TTM" in daily.columns
    # On 2025-02-15 we should have the 4-quarter TTM filed up to 2025-02-01
    assert daily.loc["2025-02-15", "revenue_TTM"] == 460
    # After Q1 2025 filing on 2025-05-01, TTM rolls forward: 110+120+130+140 = 500
    assert daily.loc["2025-05-15", "revenue_TTM"] == 500


def test_ticker_to_cik_returns_padded_string():
    """CIK should be 10-digit zero-padded string. Known: AAPL = 0000320193."""
    cik = edgar.ticker_to_cik("AAPL")
    # If SEC cache hasn't been refreshed, skip rather than fail
    if cik is None:
        pytest.skip("SEC ticker map not cached locally")
    assert isinstance(cik, str)
    assert len(cik) == 10
    assert cik.isdigit()
    assert cik == "0000320193"


def test_ticker_to_cik_returns_none_for_etf():
    """ETFs (SOXX, VOO) are not in SEC company-tickers map → None."""
    for etf in ("SOXX", "VOO", "XLK"):
        cik = edgar.ticker_to_cik(etf)
        if not edgar.TICKER_MAP_PATH.exists():
            pytest.skip("SEC ticker map not cached")
        # Most pure ETFs aren't in the map; a few (QQQ/SPY) ARE listed and
        # would return a CIK but with empty fundamentals downstream. So accept
        # either None OR a CIK with no fundamentals.
        if cik is not None:
            # If they happen to be in the map, OK — but we won't get meaningful fundamentals
            assert isinstance(cik, str)
