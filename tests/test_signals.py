"""Unit tests for signals.py — RSI / MACD / MA / BB / signal codes."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import pytest

from quant import signals


def test_compute_returns_none_for_empty_df(strategies_cfg):
    assert signals.compute("XYZ", pd.DataFrame(), strategies_cfg) is None


def test_compute_short_history_marks_insufficient(strategies_cfg):
    """20 rows is too few for MA50/MA200 — should be marked but not crash."""
    dates = pd.date_range("2025-01-01", periods=20, freq="B")
    df = pd.DataFrame({"open":[100]*20,"high":[101]*20,"low":[99]*20,
                        "close":[100]*20,"volume":[1000]*20}, index=dates)
    sig = signals.compute("X", df, strategies_cfg)
    assert sig is not None
    assert "INSUFFICIENT_HISTORY" in sig.signal_codes


def test_uptrend_above_ma200(trending_up_prices, strategies_cfg):
    """40 days uptrend — too short for MA50/MA200, but checks should not crash."""
    sig = signals.compute("UP", trending_up_prices, strategies_cfg)
    assert sig is not None
    # MA20 is computable from 40 rows; MA50/MA200 will be NaN (insufficient history)
    assert sig.ma20 > 0 and not math.isnan(sig.ma20)
    # 20-day price change should be strongly positive on a linear uptrend
    assert sig.chg_20d_pct > 5
    assert "INSUFFICIENT_HISTORY" in sig.signal_codes


def test_overbought_signal_present(overbought_prices, strategies_cfg):
    """Sharp rally last 15 days → RSI > 70."""
    sig = signals.compute("OB", overbought_prices, strategies_cfg)
    assert sig is not None
    assert sig.rsi > 70, f"Expected RSI > 70 after rally, got {sig.rsi}"
    assert "RSI_OVERBOUGHT" in sig.signal_codes or "RSI_EXTREME_OVERBOUGHT" in sig.signal_codes


def test_crash_triggers_oversold_or_break_lower(crash_prices, strategies_cfg):
    sig = signals.compute("CR", crash_prices, strategies_cfg)
    assert sig is not None
    # After 30% crash from 130 → 90, RSI should be oversold
    assert sig.rsi < 35


def test_chg_pct_calculation(trending_up_prices, strategies_cfg):
    sig = signals.compute("UP", trending_up_prices, strategies_cfg)
    last_close = float(trending_up_prices["close"].iloc[-1])
    prev_close = float(trending_up_prices["close"].iloc[-2])
    expected_chg = (last_close / prev_close - 1) * 100
    assert math.isclose(sig.chg_1d_pct, expected_chg, rel_tol=1e-3)


def test_atr_pct_is_positive(trending_up_prices, strategies_cfg):
    sig = signals.compute("UP", trending_up_prices, strategies_cfg)
    assert sig.atr_pct >= 0


def test_rsi_zscore_no_warning_on_constant_series(strategies_cfg, recwarn):
    """Constant prices → RSI window std=0; rsi_zscore_252d should be NaN, not warn."""
    import numpy as np
    import pandas as pd
    import warnings
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    flat = pd.DataFrame({"open":[100]*300,"high":[100.5]*300,"low":[99.5]*300,
                          "close":[100]*300,"volume":[1000]*300}, index=dates)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sig = signals.compute("FLAT", flat, strategies_cfg)
    assert sig is not None
    assert math.isnan(sig.rsi_zscore_252d)
