"""Sanity tests for the Alpha158-style feature builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.ml import features as f


def _synthetic(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=dates)


def test_build_features_returns_alpha158_subset():
    feats = f.build_features(_synthetic())
    # KBAR 9 + price 4 + volume 1 + rolling 26 ops × 5 windows = 144
    assert len(feats.columns) == 144, f"unexpected col count: {len(feats.columns)}"
    # KBAR features must exist
    for name in ["KMID", "KLEN", "KMID2", "KUP", "KLOW", "KSFT"]:
        assert name in feats.columns
    # Rolling features at all default windows
    for w in (5, 10, 20, 30, 60):
        for op in ("MA", "STD", "ROC", "MAX", "MIN", "RSV", "CORR"):
            assert f"{op}{w}" in feats.columns, f"missing {op}{w}"


def test_build_features_no_explosion():
    feats = f.build_features(_synthetic())
    # After warm-up (60 bars), features should be mostly finite
    warm = feats.iloc[80:]
    nan_ratio = warm.isna().sum().sum() / (warm.shape[0] * warm.shape[1])
    assert nan_ratio < 0.05, f"too many NaN after warmup: {nan_ratio:.1%}"
    # No infinities
    assert not warm.replace([np.inf, -np.inf], np.nan).isna().sum().sum() > warm.isna().sum().sum() * 2


def test_build_features_empty_input():
    assert f.build_features(pd.DataFrame()).empty
    assert f.build_features(None).empty


def test_build_features_missing_columns():
    bad = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="missing columns"):
        f.build_features(bad)


def test_forward_return_label_aligns_to_decision_day():
    df = _synthetic(50)
    label = f.forward_return_label(df, horizon_days=5)
    # Last few rows should be NaN (no forward bars)
    assert label.iloc[-5:].isna().all()
    # Label index matches input index
    assert (label.index == df.index).all()
    # Spot-check: label at day t = (close[t+5] - close[t+1]) / close[t+1]
    t = 10
    expected = (df["close"].iloc[t + 5] - df["close"].iloc[t + 1]) / df["close"].iloc[t + 1]
    assert abs(label.iloc[t] - expected) < 1e-9


def test_kmid_matches_qlib_formula():
    """KMID = (close - open) / open."""
    df = _synthetic(50)
    feats = f.build_features(df)
    expected = (df["close"] - df["open"]) / df["open"]
    assert (feats["KMID"] - expected).abs().max() < 1e-9


def test_ma5_matches_qlib_formula():
    """MA5 = rolling_mean(close, 5) / close."""
    df = _synthetic(50)
    feats = f.build_features(df)
    expected = df["close"].rolling(5).mean() / df["close"]
    # Compare where both are non-NaN
    both = pd.concat([feats["MA5"], expected], axis=1).dropna()
    assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() < 1e-9
