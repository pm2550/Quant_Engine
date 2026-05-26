"""组合优化 - MPT 均值方差 / 风险平价 / 最大分散化."""
from __future__ import annotations
import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg_mod
from . import fetcher

log = logging.getLogger(__name__)


def _build_returns(symbols: list[str], lookback_days: int = 252) -> pd.DataFrame:
    """Daily log returns for each symbol over lookback window. NaN-aligned."""
    end = date.today()
    start = end - timedelta(days=int(lookback_days * 1.6))
    cols: dict[str, pd.Series] = {}
    for s in symbols:
        df = fetcher.load_local(s)
        if df.empty or "close" not in df.columns:
            continue
        c = df["close"].astype(float).copy()
        c.index = pd.to_datetime(c.index)
        c = c.loc[c.index >= pd.Timestamp(start)]
        cols[s] = np.log(c / c.shift(1))
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols, axis=1).dropna(how="all").tail(lookback_days)
    return df.dropna(axis=1, how="all")


def covariance(returns: pd.DataFrame, *, annualize: int = 252) -> pd.DataFrame:
    return returns.cov() * annualize


def correlation(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def expected_returns(returns: pd.DataFrame, *, annualize: int = 252) -> pd.Series:
    return returns.mean() * annualize


# ---- MPT Mean-Variance Optimisation ----
def mpt_optimize(returns: pd.DataFrame, *, target: str = "max_sharpe",
                 rf: float = 0.04, max_weight: float = 0.30,
                 bounds: list[tuple[float, float]] | None = None) -> dict:
    """Long-only MPT with weight caps. Returns weights + metrics.

    target: 'max_sharpe' | 'min_var' | 'risk_parity'
    bounds: optional per-symbol (lo, hi) tuple list aligned to returns.columns.
            If None, uses uniform (0, max_weight).
    """
    from scipy.optimize import minimize
    n = len(returns.columns)
    if n == 0:
        return {}
    mu = expected_returns(returns).values
    cov = covariance(returns).values

    def portfolio_metrics(w: np.ndarray) -> tuple[float, float, float]:
        ret = float(w @ mu)
        var = float(w @ cov @ w)
        vol = np.sqrt(max(var, 1e-12))
        sharpe = (ret - rf) / vol
        return ret, vol, sharpe

    if target == "max_sharpe":
        def neg_sharpe(w):
            return -portfolio_metrics(w)[2]
        objective = neg_sharpe
    elif target == "min_var":
        def variance_obj(w):
            return float(w @ cov @ w)
        objective = variance_obj
    elif target == "risk_parity":
        def risk_parity_obj(w):
            port_vol = np.sqrt(max(float(w @ cov @ w), 1e-12))
            mrc = cov @ w
            rc = w * mrc / port_vol
            target_rc = port_vol / n
            return float(np.sum((rc - target_rc) ** 2))
        objective = risk_parity_obj
    else:
        raise ValueError(f"unknown target: {target}")

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if bounds is None:
        bounds = [(0.0, max_weight)] * n
    elif len(bounds) != n:
        raise ValueError(f"bounds length {len(bounds)} != n_symbols {n}")
    # 安全保护: lower 不能超过 max_weight 否则 SLSQP infeasible
    safe_bounds = [(max(0.0, min(lo, hi)), max(lo, hi)) for lo, hi in bounds]
    # 用 bounds 中点 (而非均值) 作初始 — 尊重 momentum_lock 的下界提示
    w0 = np.array([(lo + hi) / 2 for lo, hi in safe_bounds])
    s = w0.sum()
    w0 = w0 / s if s > 0 else np.ones(n) / n

    res = minimize(objective, w0, method="SLSQP", bounds=safe_bounds,
                   constraints=constraints, options={"maxiter": 200, "ftol": 1e-9})
    if not res.success:
        log.warning("MPT optimization failed: %s", res.message)
    w = np.array(res.x)
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else w
    ret, vol, sharpe = portfolio_metrics(w)
    return {
        "target": target,
        "weights": dict(zip(returns.columns, [round(float(x), 4) for x in w])),
        "expected_return_annual": round(ret, 4),
        "volatility_annual": round(vol, 4),
        "sharpe": round(sharpe, 3),
        "rf": rf,
    }


def build_momentum_locked_bounds(
    returns: pd.DataFrame,
    current_weights: dict[str, float],
    *,
    momentum_lookback: int = 20,
    lock_threshold_pct: float = 0.15,
    min_weight_to_lock_pct: float = 0.05,
    max_drift_down_pct: float = 0.10,
    max_weight: float = 0.30,
) -> list[tuple[float, float]]:
    """Build per-symbol bounds that protect winners from MPT's clear-the-winner bias.

    规则 (全部满足才锁):
      1. 过去 N 天累涨 >= lock_threshold_pct (默认 +15%)
      2. 当前权重 >= min_weight_to_lock_pct (默认 5%) — 否则没什么可锁的
    锁定后下界 = max(0, current_weight - max_drift_down_pct)
    即周报不会建议 "ARM 19% → 0%" 这种刚涨完就清仓.

    Why: max-Sharpe MPT 对方差敏感, 强势股波动放大反被压低目标权重 → 数学正确, 行为反趋势.
    """
    bounds = []
    locked: list[str] = []
    for sym in returns.columns:
        cur_w = current_weights.get(sym, 0.0)
        # 用对数收益累计计算 momentum
        if len(returns) >= momentum_lookback:
            log_r_window = returns[sym].dropna().tail(momentum_lookback)
            if len(log_r_window) >= momentum_lookback // 2:
                cum_log = log_r_window.sum()
                momentum = float(np.exp(cum_log) - 1)
            else:
                momentum = 0.0
        else:
            momentum = 0.0

        if momentum >= lock_threshold_pct and cur_w >= min_weight_to_lock_pct:
            lower = max(0.0, cur_w - max_drift_down_pct)
            bounds.append((lower, max_weight))
            locked.append(f"{sym} (momentum {momentum * 100:+.1f}% cur {cur_w * 100:.1f}%, locked>={lower * 100:.1f}%)")
        else:
            bounds.append((0.0, max_weight))
    if locked:
        log.info("momentum_lock applied to: %s", "; ".join(locked))
    return bounds


def compare_to_current(portfolio: dict, opt_result: dict) -> dict:
    """Compute drift between current weights and optimal weights."""
    held = portfolio.get("positions", {})
    # current weights — same-currency only since we don't FX-merge
    market_values: dict[str, float] = {}
    for sym, info in held.items():
        df = fetcher.load_local(sym)
        if df.empty:
            continue
        price = float(df["close"].iloc[-1])
        market_values[sym] = price * info["shares"]
    by_ccy_total: dict[str, float] = {}
    for s, mv in market_values.items():
        c = held[s].get("currency", "USD")
        by_ccy_total[c] = by_ccy_total.get(c, 0) + mv
    cur_w = {
        s: mv / by_ccy_total[held[s].get("currency", "USD")]
        for s, mv in market_values.items()
        if by_ccy_total.get(held[s].get("currency", "USD"))
    }

    diffs = []
    for sym, opt_w in opt_result["weights"].items():
        c_w = cur_w.get(sym, 0.0)
        diff = opt_w - c_w
        if abs(diff) > 0.005:  # only flag drift > 0.5pp
            diffs.append({
                "symbol": sym,
                "current_pct": round(c_w * 100, 2),
                "optimal_pct": round(opt_w * 100, 2),
                "drift_pct": round(diff * 100, 2),
                "action": "加" if diff > 0 else "减",
            })
    diffs.sort(key=lambda d: abs(d["drift_pct"]), reverse=True)
    return {"drifts": diffs}


def _current_weights_for_currency(portfolio: dict, currency: str) -> dict[str, float]:
    """Snapshot current weights of positions in given currency bucket."""
    held = portfolio.get("positions", {})
    market_values: dict[str, float] = {}
    for sym, info in held.items():
        if info.get("currency", "USD") != currency:
            continue
        df = fetcher.load_local(sym)
        if df.empty:
            continue
        price = float(df["close"].iloc[-1])
        market_values[sym] = price * info["shares"]
    total = sum(market_values.values())
    if total <= 0:
        return {}
    return {s: mv / total for s, mv in market_values.items()}


def run_for_currency(currency: str = "USD", *, target: str = "max_sharpe",
                     lookback_days: int = 252, momentum_lock: bool = True,
                     lock_threshold_pct: float = 0.15,
                     max_drift_down_pct: float = 0.10) -> dict:
    """Optimize within one currency bucket of the user's portfolio.

    momentum_lock (default True, 2026-05-26+): 强势股 (累涨 >= lock_threshold_pct)
    且 当前 weight >= 5% 的标的, 下界 = current - max_drift_down_pct, 防止 MPT 强制清仓.
    """
    portfolio = cfg_mod.load("portfolio")
    held = portfolio.get("positions", {})
    syms = [s for s, info in held.items() if info.get("currency", "USD") == currency]
    if len(syms) < 2:
        return {"error": f"need at least 2 symbols in {currency} bucket; got {syms}"}
    returns = _build_returns(syms, lookback_days=lookback_days)
    if returns.empty:
        return {"error": "no return data"}

    bounds = None
    locked_info: dict[str, dict] = {}
    if momentum_lock:
        current_w = _current_weights_for_currency(portfolio, currency)
        bounds = build_momentum_locked_bounds(
            returns, current_w,
            lock_threshold_pct=lock_threshold_pct,
            max_drift_down_pct=max_drift_down_pct,
        )
        # Capture which symbols got locked for downstream transparency
        for sym, (lo, hi) in zip(returns.columns, bounds):
            if lo > 0:
                locked_info[sym] = {
                    "current_weight_pct": round(current_w.get(sym, 0) * 100, 2),
                    "lower_bound_pct": round(lo * 100, 2),
                    "upper_bound_pct": round(hi * 100, 2),
                }

    opt = mpt_optimize(returns, target=target, bounds=bounds)
    drift = compare_to_current(portfolio, opt)
    cov_df = covariance(returns)
    return {
        "currency": currency,
        "lookback_days": lookback_days,
        "n_symbols": len(returns.columns),
        "optimization": opt,
        "drift_vs_current": drift,
        "annual_vol_per_symbol": {s: round(float(np.sqrt(cov_df.loc[s, s])), 3)
                                  for s in returns.columns},
        "momentum_lock_applied": momentum_lock,
        "locked_symbols": locked_info,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="USD", choices=["USD", "CNY"])
    ap.add_argument("--target", default="max_sharpe",
                    choices=["max_sharpe", "min_var", "risk_parity"])
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--no-momentum-lock", action="store_true",
                    help="禁用 momentum_lock; 默认开启防止强势股被清仓")
    ap.add_argument("--lock-threshold-pct", type=float, default=0.15,
                    help="累涨触发锁定的阈值 (默认 0.15 = 20 日 +15%%)")
    ap.add_argument("--max-drift-down-pct", type=float, default=0.10,
                    help="锁定后允许下行漂移的上限 (默认 0.10 = -10pp)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = run_for_currency(args.currency, target=args.target,
                           lookback_days=args.lookback,
                           momentum_lock=not args.no_momentum_lock,
                           lock_threshold_pct=args.lock_threshold_pct,
                           max_drift_down_pct=args.max_drift_down_pct)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
