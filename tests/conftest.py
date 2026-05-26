"""Shared pytest fixtures + helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def trending_up_prices():
    """40 days of monotonic uptrend, useful for testing trend signals."""
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    closes = np.linspace(100, 140, 40)
    df = pd.DataFrame({
        "open": closes - 0.5,
        "high": closes + 0.5,
        "low":  closes - 1.0,
        "close": closes,
        "volume": np.full(40, 1_000_000),
    }, index=dates)
    return df


@pytest.fixture
def overbought_prices():
    """Sharp recent rally → RSI overbought."""
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    closes = np.concatenate([np.linspace(100, 100, 25), np.linspace(100, 140, 15)])
    df = pd.DataFrame({
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.full(40, 1_000_000),
    }, index=dates)
    return df


@pytest.fixture
def crash_prices():
    """Trending up then sharp crash."""
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    closes = np.concatenate([np.linspace(100, 130, 40), np.linspace(130, 90, 20)])
    df = pd.DataFrame({
        "open": closes,
        "high": closes + 1,
        "low": closes - 1,
        "close": closes,
        "volume": np.full(60, 1_000_000),
    }, index=dates)
    return df


@pytest.fixture
def strategies_cfg():
    return {
        "indicators": {
            "rsi":       {"overbought": 70, "oversold": 30,
                           "extreme_overbought": 80, "extreme_oversold": 20},
            "macd":      {"enabled": True},
            "bollinger": {"period": 20, "std": 2},
            "ma":        {"short": 20, "medium": 50, "long": 200},
            "atr":       {"period": 14},
        },
    }
