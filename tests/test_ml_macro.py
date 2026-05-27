"""Tests for macro feature builder — stationarity + no calendar leak."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.ml import macro as m


def test_load_macro_features_returns_only_stationary_cols():
    """Macro features must be deltas / ratios / pct-ranks, never raw levels.

    Raw VIX / Y10 / DXY levels would let LightGBM memorize calendar periods,
    destroying cross-sectional generalization (this regressed median IC from
    +0.058 to +0.002 in our 2026-05 ablation).
    """
    df = m.load_macro_features()
    if df.empty:
        pytest.skip("no cached macro data — run `python -m quant.ml.macro` first")

    banned = {"VIX", "Y10", "Y3M", "DXY", "GOLD", "OIL", "VIX_X_Y10"}
    found = banned & set(df.columns)
    assert not found, f"raw-level macro features detected (cross-section poison): {found}"


def test_load_macro_features_has_expected_features():
    """Sanity check that the rate-of-change and regime features are present."""
    df = m.load_macro_features()
    if df.empty:
        pytest.skip("no cached macro data")

    expected = {"VIX_PCT60", "VIX_TREND", "Y10_CHG20_BPS", "GOLD_CHG60", "OIL_CHG60"}
    missing = expected - set(df.columns)
    assert not missing, f"missing expected macro features: {missing}"


def test_macro_features_finite_after_warmup():
    df = m.load_macro_features()
    if df.empty:
        pytest.skip("no cached macro data")

    warm = df.iloc[100:]  # past the 60-day rolling warmup
    nan_ratio = warm.isna().sum().sum() / (warm.shape[0] * warm.shape[1])
    assert nan_ratio < 0.10, f"too many NaN in warm macro features: {nan_ratio:.1%}"
    no_inf = warm.replace([np.inf, -np.inf], np.nan)
    assert no_inf.isna().sum().sum() == warm.isna().sum().sum(), "infinities found in macro features"


def test_vix_pct60_bounded_zero_to_one():
    df = m.load_macro_features()
    if df.empty:
        pytest.skip("no cached macro data")
    pct = df["VIX_PCT60"].dropna()
    if pct.empty:
        pytest.skip("no VIX_PCT60 values")
    assert pct.min() >= 0.0 - 1e-9
    assert pct.max() <= 1.0 + 1e-9
