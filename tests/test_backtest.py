"""Unit tests for backtest.py — strategy backtests with vectorbt.

Critical: lookahead bias guards, slippage handling, period slicing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from quant import backtest


@pytest.fixture
def synthetic_prices():
    """Create deterministic price series with known up-down cycles."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    # Sinusoidal pattern: cycles between 90 and 110
    closes = 100 + 10 * np.sin(np.linspace(0, 8 * np.pi, 500))
    return pd.DataFrame({
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.full(500, 1_000_000),
    }, index=dates)


def test_slice_returns_correct_window(synthetic_prices):
    """_slice should return ~5 years' worth of data when called with period_years=5."""
    sliced = backtest._slice(synthetic_prices, period_years=2)
    # 2 years × 252 trading days ≈ 504; but our dataset is 500
    assert len(sliced) <= 500
    # Verify last date preserved
    assert sliced.index.max() == synthetic_prices.index.max()


def test_slice_with_as_of_truncates_future_data(synthetic_prices):
    cutoff = "2020-12-31"
    sliced = backtest._slice(synthetic_prices, period_years=10, as_of=cutoff)
    assert sliced.index.max() <= pd.Timestamp(cutoff)


def test_split_params_extracts_as_of():
    p, as_of = backtest._split_params({"short": 5, "long": 20, "as_of": "2025-01-01"})
    assert "as_of" not in p
    assert p == {"short": 5, "long": 20}
    assert as_of == "2025-01-01"


def test_split_params_no_as_of():
    p, as_of = backtest._split_params({"short": 5, "long": 20})
    assert as_of is None


def test_dual_ma_metrics_required_fields(synthetic_prices, monkeypatch):
    """Verify dual_ma backtest returns all required metrics keys."""
    # Monkey-patch _load to return our synthetic data
    monkeypatch.setattr(backtest, "_load", lambda sym: synthetic_prices)
    out = backtest.dual_ma("FAKE", {"short": 5, "long": 20}, period_years=2)
    required = {"total_return", "sharpe", "max_drawdown", "n_trades",
                "win_rate", "profit_factor", "annual_return", "sortino"}
    assert required.issubset(out.keys())
    # Sharpe should be a finite number
    assert np.isfinite(out["sharpe"])


def test_dual_ma_invalid_short_long_raises(synthetic_prices, monkeypatch):
    monkeypatch.setattr(backtest, "_load", lambda sym: synthetic_prices)
    with pytest.raises(ValueError, match="short must be"):
        backtest.dual_ma("FAKE", {"short": 50, "long": 5}, period_years=2)


def test_dual_ma_insufficient_history_raises(monkeypatch):
    """A symbol with <250 rows should raise insufficient history error."""
    short_df = pd.DataFrame({
        "close": np.linspace(100, 110, 50),
    }, index=pd.date_range("2020-01-01", periods=50, freq="B"))
    monkeypatch.setattr(backtest, "_load", lambda sym: short_df)
    with pytest.raises(ValueError, match="insufficient"):
        backtest.dual_ma("X", {"short": 5, "long": 20})


def test_strategy_registry_has_all_4():
    assert "dual_ma" in backtest.REGISTRY
    assert "rsi_meanrev" in backtest.REGISTRY
    assert "bb_breakout" in backtest.REGISTRY
    assert "macd_cross" in backtest.REGISTRY


def test_run_dispatches_to_correct_strategy(synthetic_prices, monkeypatch):
    monkeypatch.setattr(backtest, "_load", lambda sym: synthetic_prices)
    out = backtest.run("dual_ma", "FAKE", {"short": 5, "long": 20}, period_years=2)
    assert "sharpe" in out


def test_run_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        backtest.run("nonexistent_strategy", "X", {})


def test_cost_model_default_round_trip_is_20bps():
    """Default round-trip = 2*(commission + slippage) = 2*(5+5)bp = 20bp = 0.002"""
    c = backtest.DEFAULT_COSTS
    round_trip = 2 * (c.commission_pct + c.slippage_pct)
    assert abs(round_trip - 0.002) < 1e-9


def test_higher_costs_reduce_total_return(synthetic_prices, monkeypatch):
    """Sanity: doubling fees should not increase total_return."""
    monkeypatch.setattr(backtest, "_load", lambda sym: synthetic_prices)
    base = backtest.dual_ma("FAKE", {"short": 5, "long": 20}, period_years=2)

    # Patch DEFAULT_COSTS to 10x
    expensive = backtest.CostModel(commission_pct=0.005, slippage_pct=0.005)
    monkeypatch.setattr(backtest, "DEFAULT_COSTS", expensive)
    high = backtest.dual_ma("FAKE", {"short": 5, "long": 20}, period_years=2)
    assert high["total_return"] <= base["total_return"]


def test_walk_forward_returns_per_fold_metrics(synthetic_prices, monkeypatch):
    monkeypatch.setattr(backtest, "_load", lambda sym: synthetic_prices)
    out = backtest.walk_forward("dual_ma", "FAKE", {"short": 5, "long": 20},
                                  period_years=2, n_folds=4, min_fold_days=60)
    assert out["n_folds"] >= 2
    assert "sharpe_median" in out
    assert "consistency_rate" in out
    assert 0 <= out["consistency_rate"] <= 1
    assert len(out["folds"]) == out["n_folds"]
    # Each fold has start/end/days metadata
    for f in out["folds"]:
        assert "start" in f and "end" in f and "days" in f


def test_walk_forward_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        backtest.walk_forward("nope", "X", {})
