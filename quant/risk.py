"""组合风险预算 - VaR / CVaR / 最大回撤 / 压力测试."""
from __future__ import annotations
import argparse
import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config as cfg_mod
from . import fetcher

log = logging.getLogger(__name__)


def _portfolio_returns(portfolio: dict, currency: str = "USD",
                      lookback_days: int = 252) -> pd.Series:
    """Daily portfolio returns based on current weights, in given currency bucket."""
    held = portfolio.get("positions", {})
    syms = [s for s, info in held.items() if info.get("currency", "USD") == currency]
    if not syms:
        return pd.Series(dtype=float)

    # current weights within bucket
    market_values = {}
    for s in syms:
        df = fetcher.load_local(s)
        if df.empty:
            continue
        price = float(df["close"].iloc[-1])
        market_values[s] = price * held[s]["shares"]
    total = sum(market_values.values())
    if total == 0:
        return pd.Series(dtype=float)
    weights = {s: mv / total for s, mv in market_values.items()}

    # daily simple returns
    cols = {}
    for s in weights:
        df = fetcher.load_local(s)
        if df.empty:
            continue
        c = df["close"].astype(float).copy()
        c.index = pd.to_datetime(c.index)
        cols[s] = c.pct_change()
    if not cols:
        return pd.Series(dtype=float)
    rets = pd.concat(cols, axis=1).dropna(how="any").tail(lookback_days)
    if rets.empty:
        return pd.Series(dtype=float)
    weight_arr = np.array([weights[c] for c in rets.columns])
    return (rets * weight_arr).sum(axis=1)


def parametric_var(returns: pd.Series, *, conf: float = 0.95) -> dict:
    """Normal-distribution VaR (mean - z*sigma)."""
    if returns.empty:
        return {}
    from scipy.stats import norm
    z = norm.ppf(1 - conf)
    mu, sigma = returns.mean(), returns.std()
    var_pct = mu + z * sigma  # negative number = loss
    return {
        "method": "parametric",
        "confidence": conf,
        "var_pct_1d": round(float(var_pct) * 100, 2),
        "mu_pct": round(float(mu) * 100, 3),
        "sigma_pct": round(float(sigma) * 100, 3),
    }


def historical_var(returns: pd.Series, *, conf: float = 0.95) -> dict:
    if returns.empty:
        return {}
    q = returns.quantile(1 - conf)
    cvar = returns[returns <= q].mean()
    return {
        "method": "historical",
        "confidence": conf,
        "var_pct_1d": round(float(q) * 100, 2),
        "cvar_pct_1d": round(float(cvar) * 100, 2),  # expected shortfall = mean of tail
    }


def max_drawdown(returns: pd.Series) -> dict:
    if returns.empty:
        return {}
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()
    mdd_idx = dd.idxmin()
    peak_idx = peak.loc[:mdd_idx].idxmax()
    return {
        "max_drawdown_pct": round(float(mdd) * 100, 2),
        "peak_date": str(peak_idx)[:10],
        "trough_date": str(mdd_idx)[:10],
    }


def stress_test(portfolio: dict, currency: str = "USD") -> list[dict]:
    """What if scenarios on portfolio value (USD bucket)."""
    rets = _portfolio_returns(portfolio, currency=currency, lookback_days=252)
    if rets.empty:
        return []
    sigma = rets.std()
    scenarios = [
        ("黑天鹅日 (2008-style -7%)", -0.07),
        ("Fed 鸽派转折日 (+3%)", 0.03),
        ("加息惊讶日 (-3%)", -0.03),
        ("Daily 1-σ down move", float(-sigma)),
        ("Daily 3-σ down move", float(-3 * sigma)),
        ("Daily 1-σ up move", float(sigma)),
    ]
    held = portfolio.get("positions", {})
    market_values = {}
    for s, info in held.items():
        if info.get("currency", "USD") != currency:
            continue
        df = fetcher.load_local(s)
        if df.empty:
            continue
        price = float(df["close"].iloc[-1])
        market_values[s] = price * info["shares"]
    total = sum(market_values.values())

    out = []
    for name, pct in scenarios:
        out.append({
            "scenario": name,
            "return_pct": round(pct * 100, 2),
            "value_change": round(total * pct, 2),
            "new_value": round(total * (1 + pct), 2),
        })
    return out


def report(currency: str = "USD") -> dict:
    portfolio = cfg_mod.load("portfolio")
    rets = _portfolio_returns(portfolio, currency=currency)
    if rets.empty:
        return {"error": f"no data for {currency} bucket"}

    return {
        "currency": currency,
        "lookback_days": len(rets),
        "annualized_volatility_pct": round(float(rets.std() * np.sqrt(252) * 100), 2),
        "annualized_return_pct": round(float(rets.mean() * 252 * 100), 2),
        "sharpe_naive": round(float(rets.mean() * 252 / (rets.std() * np.sqrt(252))), 3),
        "var_95": parametric_var(rets, conf=0.95),
        "var_99": parametric_var(rets, conf=0.99),
        "historical_var_95": historical_var(rets, conf=0.95),
        "max_drawdown": max_drawdown(rets),
        "stress_tests": stress_test(portfolio, currency=currency),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="USD")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = report(args.currency)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
