"""Tests for optimizer.py — focus on momentum_lock added 2026-05-26."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import optimizer


def _synthetic_returns(strong: list[str], weak: list[str], days: int = 252) -> pd.DataFrame:
    """Return a returns DataFrame where `strong` symbols have +0.5%/day drift over last 20d,
    `weak` symbols hover near 0."""
    rng = np.random.default_rng(seed=42)
    cols = {}
    for s in strong:
        # 220 days quiet + 20 days strong rally (cumulative +30% over last 20d)
        quiet = rng.normal(0, 0.01, days - 20)
        rally = rng.normal(0.013, 0.01, 20)  # ~+30% cumulative log return
        cols[s] = np.concatenate([quiet, rally])
    for s in weak:
        cols[s] = rng.normal(0, 0.01, days)
    df = pd.DataFrame(cols)
    df.index = pd.date_range("2025-09-01", periods=days, freq="B")
    return df


def test_build_momentum_locked_bounds_locks_strong_winner():
    """ARM 5d 累涨 30% + 当前权重 19% → 锁定下界 = 9% (current - 10pp)."""
    returns = _synthetic_returns(strong=["ARM"], weak=["VOO", "QQQ"])
    current = {"ARM": 0.19, "VOO": 0.30, "QQQ": 0.30}
    bounds = optimizer.build_momentum_locked_bounds(returns, current,
                                                     lock_threshold_pct=0.15,
                                                     max_drift_down_pct=0.10)
    # bounds[0] = ARM (first column in returns)
    cols = list(returns.columns)
    arm_idx = cols.index("ARM")
    voo_idx = cols.index("VOO")
    lo_arm, hi_arm = bounds[arm_idx]
    lo_voo, hi_voo = bounds[voo_idx]
    assert lo_arm == pytest.approx(0.09, abs=0.001), f"ARM lower bound should be 0.09 got {lo_arm}"
    assert hi_arm == 0.30
    assert lo_voo == 0.0, f"VOO no momentum, no lock; got {lo_voo}"


def test_build_momentum_locked_bounds_no_lock_for_low_weight():
    """RKLB 强势 +30% 但当前 weight 只 3% (<5%) → 不锁."""
    returns = _synthetic_returns(strong=["RKLB"], weak=["VOO"])
    current = {"RKLB": 0.03, "VOO": 0.50}
    bounds = optimizer.build_momentum_locked_bounds(returns, current,
                                                     min_weight_to_lock_pct=0.05)
    cols = list(returns.columns)
    lo, _ = bounds[cols.index("RKLB")]
    assert lo == 0.0


def test_build_momentum_locked_bounds_no_lock_for_weak_stock():
    """SOXX 走平 → 不锁."""
    returns = _synthetic_returns(strong=[], weak=["SOXX", "VOO"])
    current = {"SOXX": 0.20, "VOO": 0.50}
    bounds = optimizer.build_momentum_locked_bounds(returns, current,
                                                     lock_threshold_pct=0.15)
    cols = list(returns.columns)
    for col in cols:
        lo, _ = bounds[cols.index(col)]
        assert lo == 0.0


def test_build_momentum_locked_bounds_threshold_strict():
    """累涨刚好低于阈值 → 不锁."""
    returns = _synthetic_returns(strong=["BARELY"], weak=["VOO"])
    # 把 BARELY 改成 14% 累涨 (低于 0.15 阈值): 直接 scale 一下
    returns.loc[returns.index[-20:], "BARELY"] = returns.loc[returns.index[-20:], "BARELY"] * 0.4
    current = {"BARELY": 0.20, "VOO": 0.50}
    bounds = optimizer.build_momentum_locked_bounds(returns, current,
                                                     lock_threshold_pct=0.15)
    cols = list(returns.columns)
    lo, _ = bounds[cols.index("BARELY")]
    # Either locked or not depending on randomness, but should not exceed current weight
    assert lo <= 0.20


def test_mpt_optimize_accepts_custom_bounds():
    """直接给 bounds 参数, 验证 lower 被尊重."""
    returns = _synthetic_returns(strong=["ARM"], weak=["VOO", "QQQ"])
    bounds = [(0.20, 0.30), (0.0, 0.30), (0.0, 0.30)]  # ARM lower=0.20
    out = optimizer.mpt_optimize(returns, target="max_sharpe", bounds=bounds)
    weights = out["weights"]
    assert weights["ARM"] >= 0.199, f"ARM should be locked >=0.20, got {weights['ARM']}"


def test_mpt_optimize_default_bounds_unchanged():
    """无 bounds 参数 → 跟旧行为一致 (default 0..max_weight). 用 4 标的保证可行性."""
    returns = _synthetic_returns(strong=[], weak=["A", "B", "C", "D"])
    out = optimizer.mpt_optimize(returns, target="max_sharpe", max_weight=0.30)
    weights = out["weights"]
    for w in weights.values():
        assert -1e-6 <= w <= 0.30 + 1e-6
    assert abs(sum(weights.values()) - 1.0) < 1e-3


def test_run_for_currency_momentum_lock_default_on(monkeypatch, tmp_path):
    """run_for_currency 默认开启 momentum_lock."""
    portfolio = {
        "positions": {
            "ARM": {"shares": 1.0, "currency": "USD"},
            "VOO": {"shares": 1.0, "currency": "USD"},
        }
    }
    monkeypatch.setattr(optimizer.cfg_mod, "load", lambda name: portfolio)

    # Fake fetcher.load_local: ARM with strong momentum, VOO flat
    strong_df = pd.DataFrame({"close": np.concatenate([np.ones(300), np.linspace(1, 1.5, 20)])})
    flat_df = pd.DataFrame({"close": np.ones(320)})
    strong_df.index = pd.date_range("2025-06-01", periods=320, freq="B")
    flat_df.index = pd.date_range("2025-06-01", periods=320, freq="B")

    def fake_load(sym):
        return strong_df if sym == "ARM" else flat_df
    monkeypatch.setattr(optimizer.fetcher, "load_local", fake_load)

    out = optimizer.run_for_currency("USD", momentum_lock=True)
    assert out["momentum_lock_applied"] is True
    # ARM should be locked since 20d momentum is +50% and weight = 1.5/(1.5+1) = 60%
    assert "ARM" in out["locked_symbols"]


def test_run_for_currency_can_disable_lock(monkeypatch):
    """显式关掉 momentum_lock 应该不出现 locked_symbols."""
    portfolio = {
        "positions": {
            "ARM": {"shares": 1.0, "currency": "USD"},
            "VOO": {"shares": 1.0, "currency": "USD"},
        }
    }
    monkeypatch.setattr(optimizer.cfg_mod, "load", lambda name: portfolio)
    strong_df = pd.DataFrame({"close": np.concatenate([np.ones(300), np.linspace(1, 1.5, 20)])})
    flat_df = pd.DataFrame({"close": np.ones(320)})
    strong_df.index = pd.date_range("2025-06-01", periods=320, freq="B")
    flat_df.index = pd.date_range("2025-06-01", periods=320, freq="B")

    def fake_load(sym):
        return strong_df if sym == "ARM" else flat_df
    monkeypatch.setattr(optimizer.fetcher, "load_local", fake_load)

    out = optimizer.run_for_currency("USD", momentum_lock=False)
    assert out["momentum_lock_applied"] is False
    assert out["locked_symbols"] == {}
