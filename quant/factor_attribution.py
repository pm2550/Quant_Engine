"""收益因子归因 - 板块/主题 + 市场 beta + 个股 alpha 拆解."""
from __future__ import annotations
import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from . import config as cfg_mod
from . import fetcher

log = logging.getLogger(__name__)


def _holdings_with_weights(portfolio: dict, currency: str = "USD") -> tuple[dict, dict, float]:
    """Return (weights{sym→pct}, themes{sym→theme}, total_value) for one ccy bucket."""
    held = portfolio.get("positions", {})
    market_values = {}
    themes = {}
    for sym, info in held.items():
        if info.get("currency", "USD") != currency:
            continue
        df = fetcher.load_local(sym)
        if df.empty:
            continue
        price = float(df["close"].iloc[-1])
        market_values[sym] = price * info["shares"]
        themes[sym] = info.get("theme", "other")
    total = sum(market_values.values())
    weights = {s: mv / total for s, mv in market_values.items()} if total else {}
    return weights, themes, total


def _daily_returns(symbols: list[str], days: int) -> pd.DataFrame:
    cols = {}
    for s in symbols:
        df = fetcher.load_local(s)
        if df.empty:
            continue
        c = df["close"].astype(float).copy()
        c.index = pd.to_datetime(c.index)
        cols[s] = c.pct_change()
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).dropna(how="any").tail(days)


def attribute_today(portfolio: dict, *, currency: str = "USD",
                    benchmark: str = "VOO") -> dict:
    """Decompose today's portfolio return into themes + market beta + alpha."""
    weights, themes, total = _holdings_with_weights(portfolio, currency)
    if not weights:
        return {"error": f"empty {currency} bucket"}

    # Latest 1-day return per holding
    rets_today = {}
    for sym in weights:
        df = fetcher.load_local(sym)
        if len(df) < 2:
            continue
        c = df["close"].astype(float)
        rets_today[sym] = float(c.iloc[-1] / c.iloc[-2] - 1)

    if not rets_today:
        return {"error": "no return data"}

    # Per-holding contribution
    contributions = []
    portfolio_ret = 0.0
    for sym, r in rets_today.items():
        w = weights.get(sym, 0)
        contrib = w * r
        portfolio_ret += contrib
        contributions.append({
            "symbol": sym,
            "weight_pct": round(w * 100, 2),
            "ret_pct": round(r * 100, 2),
            "contribution_pct": round(contrib * 100, 3),
            "theme": themes.get(sym, "other"),
        })

    # Theme bucket attribution
    theme_attr: dict[str, dict] = {}
    for c in contributions:
        t = c["theme"]
        theme_attr.setdefault(t, {"contribution": 0.0, "weight": 0.0, "members": []})
        theme_attr[t]["contribution"] += c["contribution_pct"]
        theme_attr[t]["weight"] += c["weight_pct"]
        theme_attr[t]["members"].append(c["symbol"])

    # Market beta decomposition (USD only — needs benchmark daily series)
    beta_decomp = {}
    if currency == "USD":
        try:
            rets_252 = _daily_returns(list(weights.keys()) + [benchmark], days=252)
            if benchmark in rets_252 and not rets_252.empty:
                # weighted portfolio returns history
                w_arr = np.array([weights.get(c, 0) for c in rets_252.columns if c != benchmark])
                cols_no_bm = [c for c in rets_252.columns if c != benchmark]
                port_hist = (rets_252[cols_no_bm] * w_arr).sum(axis=1)
                bm_hist = rets_252[benchmark]
                cov = np.cov(port_hist.values, bm_hist.values)
                bm_var = float(np.var(bm_hist.values))
                beta = float(cov[0, 1] / bm_var) if bm_var > 0 else 1.0
                bm_today_ret = float(rets_252[benchmark].iloc[-1])
                market_part = beta * bm_today_ret
                alpha_part = portfolio_ret - market_part
                beta_decomp = {
                    "benchmark": benchmark,
                    "beta_252d": round(beta, 3),
                    "benchmark_today_pct": round(bm_today_ret * 100, 2),
                    "market_part_pct": round(market_part * 100, 3),
                    "alpha_part_pct": round(alpha_part * 100, 3),
                }
        except Exception as e:  # noqa: BLE001
            log.debug("beta decomp failed: %s", e)

    return {
        "currency": currency,
        "portfolio_total": round(total, 2),
        "portfolio_ret_today_pct": round(portfolio_ret * 100, 3),
        "by_holding": sorted(contributions, key=lambda x: abs(x["contribution_pct"]), reverse=True),
        "by_theme": [
            {"theme": t, "contribution_pct": round(d["contribution"], 3),
             "weight_pct": round(d["weight"], 2), "members": d["members"]}
            for t, d in sorted(theme_attr.items(), key=lambda x: abs(x[1]["contribution"]), reverse=True)
        ],
        "market_beta": beta_decomp,
    }


def render(attr: dict) -> str:
    if attr.get("error"):
        return f"_attribution: {attr['error']}_"
    lines = [f"📊 *归因 ({attr['currency']} 组合 {attr['portfolio_ret_today_pct']:+.2f}%):*"]
    if attr.get("market_beta"):
        b = attr["market_beta"]
        lines.append(f"  • 市场部分 (β={b['beta_252d']}): {b['market_part_pct']:+.2f}% (基准 {b['benchmark']} {b['benchmark_today_pct']:+.2f}%)")
        lines.append(f"  • 个股 α: {b['alpha_part_pct']:+.2f}%")
    lines.append("  *按主题:*")
    for t in attr.get("by_theme", []):
        lines.append(f"    {t['theme']}: {t['contribution_pct']:+.2f}% ({', '.join(t['members'][:3])})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="USD")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    pf = cfg_mod.load("portfolio")
    out = attribute_today(pf, currency=args.currency)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print()
    print(render(out))


if __name__ == "__main__":
    main()
