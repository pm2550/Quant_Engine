"""操作模拟 - 假设按优化建议/手动调仓做了, 历史上 N 天后会怎样.

用法:
  # 比较当前权重 vs MPT 优化权重在过去 30/90/365 天的回报
  python -m quant.dry_run --target max_sharpe --windows 30 90 365

  # 模拟一组手动调整 (--delta 'SOXX:-10,QQQ:+10' = SOXX -10pp, QQQ +10pp)
  python -m quant.dry_run --delta 'SOXX:-10,QQQ:+10' --windows 30 90
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config as cfg_mod
from . import fetcher, optimizer

log = logging.getLogger(__name__)


def _current_weights(portfolio: dict, currency: str = "USD") -> dict[str, float]:
    held = portfolio.get("positions", {})
    market_values = {}
    for s, info in held.items():
        if info.get("currency", "USD") != currency:
            continue
        df = fetcher.load_local(s)
        if df.empty:
            continue
        market_values[s] = float(df["close"].iloc[-1]) * info["shares"]
    total = sum(market_values.values())
    return {s: mv / total for s, mv in market_values.items()} if total else {}


def _apply_delta(weights: dict, delta_str: str) -> dict:
    """Apply 'SYM:+X,SYM2:-Y' percentage-point deltas, then renormalise."""
    if not delta_str:
        return weights
    new_w = dict(weights)
    for kv in delta_str.split(","):
        sym, pp = kv.strip().split(":")
        delta = float(pp) / 100.0
        new_w[sym] = max(0.0, new_w.get(sym, 0) + delta)
    s = sum(new_w.values())
    return {k: v / s for k, v in new_w.items()} if s else new_w


def _portfolio_return(weights: dict, lookback_days: int) -> dict:
    """Compute total return of weighted portfolio over the past N days (held-constant rebalance)."""
    end = date.today()
    start = end - timedelta(days=int(lookback_days * 1.6))
    cols = {}
    for s in weights:
        df = fetcher.load_local(s)
        if df.empty:
            continue
        c = df["close"].astype(float).copy()
        c.index = pd.to_datetime(c.index)
        cols[s] = c.loc[c.index >= pd.Timestamp(start)]
    if not cols:
        return {"error": "no data"}
    aligned = pd.concat(cols, axis=1).dropna(how="any").tail(lookback_days)
    if aligned.empty or len(aligned) < 2:
        return {"error": "insufficient data"}
    rets = aligned.pct_change().dropna()
    w_arr = np.array([weights.get(c, 0) for c in rets.columns])
    port_rets = (rets * w_arr).sum(axis=1)
    cum = (1 + port_rets).prod() - 1
    vol = port_rets.std() * np.sqrt(252)
    sharpe = (port_rets.mean() * 252 - 0.04) / vol if vol > 0 else 0
    cum_curve = (1 + port_rets).cumprod()
    peak = cum_curve.cummax()
    mdd = float(((cum_curve - peak) / peak).min())
    return {
        "lookback_days": int(len(rets)),
        "total_return_pct": round(float(cum) * 100, 2),
        "annualized_vol_pct": round(float(vol) * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(mdd * 100, 2),
    }


def compare(current: dict, proposed: dict, *, windows: list[int]) -> dict:
    out = {"current_weights": {k: round(v, 4) for k, v in current.items()},
           "proposed_weights": {k: round(v, 4) for k, v in proposed.items()},
           "comparisons": []}
    for w in windows:
        cur = _portfolio_return(current, lookback_days=w)
        prop = _portfolio_return(proposed, lookback_days=w)
        diff = None
        if "total_return_pct" in cur and "total_return_pct" in prop:
            diff = round(prop["total_return_pct"] - cur["total_return_pct"], 2)
        out["comparisons"].append({
            "lookback_days": w,
            "current": cur,
            "proposed": prop,
            "delta_return_pct": diff,
        })
    return out


def render(out: dict) -> str:
    lines = ["🔮 *Dry-run 历史回放对比*", ""]
    cur_w = out["current_weights"]
    prop_w = out["proposed_weights"]
    lines.append("*权重变化:*")
    syms = sorted(set(list(cur_w.keys()) + list(prop_w.keys())))
    for s in syms:
        c = cur_w.get(s, 0) * 100
        p = prop_w.get(s, 0) * 100
        diff = p - c
        if abs(diff) > 0.5:
            arrow = "📈" if diff > 0 else "📉"
            lines.append(f"  {arrow} `{s}` {c:.1f}% → {p:.1f}% ({diff:+.1f}pp)")
    lines.append("")
    lines.append("*历史回报对比 (持仓常驻):*")
    for c in out["comparisons"]:
        n = c["lookback_days"]
        cur_r = c["current"].get("total_return_pct", "?")
        prop_r = c["proposed"].get("total_return_pct", "?")
        diff = c.get("delta_return_pct", 0) or 0
        emoji = "✅" if diff > 0 else "❌" if diff < 0 else "➖"
        lines.append(f"  {n} 天: 当前 {cur_r}% vs 推荐 {prop_r}% {emoji} {diff:+.2f}pp")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--target", default="max_sharpe",
                    choices=["max_sharpe", "min_var", "risk_parity"],
                    help="if no --delta given, use optimizer's recommendation")
    ap.add_argument("--delta", help="manual deltas, e.g. 'SOXX:-10,QQQ:+10'")
    ap.add_argument("--windows", type=int, nargs="+", default=[30, 90, 365])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pf = cfg_mod.load("portfolio")
    current = _current_weights(pf, currency=args.currency)
    if not current:
        print(f"empty {args.currency} bucket")
        return

    if args.delta:
        proposed = _apply_delta(current, args.delta)
    else:
        opt = optimizer.run_for_currency(args.currency, target=args.target)
        proposed = opt.get("optimization", {}).get("weights", current)

    out = compare(current, proposed, windows=args.windows)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print()
    print(render(out))


if __name__ == "__main__":
    main()
