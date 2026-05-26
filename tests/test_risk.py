"""Unit tests for risk.py — VaR / CVaR / max drawdown."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from quant import risk


def test_parametric_var_negative_for_normal_returns():
    """Returns with mean 0, sigma 0.01 → VaR 95% should be negative ~-1.65σ."""
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0, 0.01, 252))
    out = risk.parametric_var(rets, conf=0.95)
    assert out["var_pct_1d"] < 0
    # Should be roughly -1.65% ± buffer
    assert -2.0 < out["var_pct_1d"] < -1.0


def test_parametric_var_higher_conf_more_extreme():
    np.random.seed(1)
    rets = pd.Series(np.random.normal(0, 0.02, 500))
    var95 = risk.parametric_var(rets, conf=0.95)
    var99 = risk.parametric_var(rets, conf=0.99)
    assert var99["var_pct_1d"] < var95["var_pct_1d"]


def test_historical_var_matches_quantile():
    """Historical VaR == 5th percentile of returns."""
    rets = pd.Series([-0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05] * 12)
    out = risk.historical_var(rets, conf=0.95)
    expected = float(rets.quantile(0.05) * 100)
    assert out["var_pct_1d"] == round(expected, 2)


def test_cvar_lower_than_var():
    """CVaR (expected shortfall) is the mean of returns below VaR — should be more extreme."""
    np.random.seed(7)
    rets = pd.Series(np.random.normal(0, 0.02, 500))
    out = risk.historical_var(rets, conf=0.95)
    assert out["cvar_pct_1d"] <= out["var_pct_1d"]


def test_max_drawdown_on_known_curve():
    """Manually compute drawdown for a known series."""
    # 100 -> 110 -> 99 -> 105: peak 110, trough 99, mdd = (99-110)/110 = -10%
    rets = pd.Series([0.10, -0.10, 0.0606060606])
    out = risk.max_drawdown(rets)
    assert out["max_drawdown_pct"] < 0
    # Roughly -10%
    assert -11 < out["max_drawdown_pct"] < -9


def test_empty_series_returns_empty_dict():
    assert risk.parametric_var(pd.Series([], dtype=float)) == {}
    assert risk.historical_var(pd.Series([], dtype=float)) == {}
    assert risk.max_drawdown(pd.Series([], dtype=float)) == {}
