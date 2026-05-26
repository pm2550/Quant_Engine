"""交易/订单层信号 - 成交量异动 + 期权异动 + 做空利息 + 内部人净流."""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import db, fetcher

log = logging.getLogger(__name__)


def volume_anomaly(symbol: str) -> dict:
    """Today's volume vs 20-day avg + price-volume divergence."""
    df = fetcher.load_local(symbol)
    if df.empty or "volume" not in df.columns or len(df) < 21:
        return {"available": False}
    df = df.tail(40).copy()
    avg_20 = float(df["volume"].iloc[-21:-1].mean())
    today_vol = float(df["volume"].iloc[-1])
    today_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    chg_pct = (today_close / prev_close - 1) * 100
    ratio = today_vol / avg_20 if avg_20 else 1.0

    # Direction signal
    direction = "neutral"
    if ratio >= 2 and chg_pct > 1.5:
        direction = "bullish"  # 放量上涨 = 主动买盘强劲
    elif ratio >= 2 and chg_pct < -1.5:
        direction = "bearish"  # 放量下跌 = 抛压
    elif ratio >= 1.5 and chg_pct > 0:
        direction = "weak_bullish"
    elif ratio >= 1.5 and chg_pct < 0:
        direction = "weak_bearish"

    # 价量背离: 价格创新高但量没跟上 = 顶部预警
    price_high_20 = float(df["close"].iloc[-21:-1].max())
    new_high = today_close >= price_high_20
    divergence_warning = new_high and ratio < 0.8

    return {
        "available": True,
        "today_volume": int(today_vol),
        "avg_20d_volume": int(avg_20),
        "volume_ratio": round(ratio, 2),
        "chg_pct": round(chg_pct, 2),
        "direction": direction,
        "divergence_warning": divergence_warning,
        "factors": (
            ([f"放量{'上涨' if direction in ('bullish','weak_bullish') else '下跌'} {ratio:.1f}x"]
             if ratio >= 1.5 else []) +
            (["⚠️ 价创新高但量萎缩 (顶部预警)"] if divergence_warning else [])
        ),
    }


def options_unusual(symbol: str) -> dict:
    """Detect unusual options activity. Uses yfinance options chain."""
    if fetcher.is_a_share(symbol):
        return {"available": False, "reason": "A 股期权数据缺"}
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        # Take nearest 2 expirations
        exps = t.options[:2]
        if not exps:
            return {"available": False}

        total_call_oi = total_put_oi = 0
        total_call_vol = total_put_vol = 0
        max_call_iv_vs_oi = []
        max_put_iv_vs_oi = []
        spot = float(t.history(period="1d")["Close"].iloc[-1])

        for exp in exps:
            try:
                chain = t.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
            except Exception:
                continue
            # Sum OI/volume
            total_call_oi += int(calls["openInterest"].fillna(0).sum())
            total_put_oi += int(puts["openInterest"].fillna(0).sum())
            total_call_vol += int(calls["volume"].fillna(0).sum())
            total_put_vol += int(puts["volume"].fillna(0).sum())
            # ATM strikes
            calls = calls.dropna(subset=["strike", "openInterest"])
            puts = puts.dropna(subset=["strike", "openInterest"])
            atm_calls = calls[(calls["strike"] >= spot * 0.95) & (calls["strike"] <= spot * 1.10)]
            atm_puts = puts[(puts["strike"] >= spot * 0.90) & (puts["strike"] <= spot * 1.05)]
            # Volume / OI ratio (>0.5 = unusual)
            for _, row in atm_calls.iterrows():
                oi = row.get("openInterest", 0) or 0
                vol = row.get("volume", 0) or 0
                if oi > 1000 and vol / oi > 0.5:
                    max_call_iv_vs_oi.append({
                        "strike": float(row["strike"]),
                        "exp": exp,
                        "vol_oi_ratio": round(vol / oi, 2),
                        "iv": float(row.get("impliedVolatility", 0)),
                    })
            for _, row in atm_puts.iterrows():
                oi = row.get("openInterest", 0) or 0
                vol = row.get("volume", 0) or 0
                if oi > 1000 and vol / oi > 0.5:
                    max_put_iv_vs_oi.append({
                        "strike": float(row["strike"]),
                        "exp": exp,
                        "vol_oi_ratio": round(vol / oi, 2),
                        "iv": float(row.get("impliedVolatility", 0)),
                    })

        put_call_ratio = total_put_oi / total_call_oi if total_call_oi else 0
        # Direction signal
        direction = "neutral"
        if put_call_ratio < 0.6 and total_call_vol > total_call_oi * 0.3:
            direction = "bullish"  # call dominant + 高交投
        elif put_call_ratio > 1.5 and total_put_vol > total_put_oi * 0.3:
            direction = "bearish"  # put dominant
        elif total_call_vol > 5 * total_put_vol:
            direction = "speculative_bullish"
        elif total_put_vol > 5 * total_call_vol:
            direction = "hedging_bearish"

        factors = []
        if put_call_ratio < 0.5:
            factors.append(f"Call/Put OI 极偏多 (PC ratio {put_call_ratio:.2f})")
        elif put_call_ratio > 2:
            factors.append(f"Put/Call OI 极偏空 (PC ratio {put_call_ratio:.2f})")
        if max_call_iv_vs_oi:
            top = max(max_call_iv_vs_oi, key=lambda x: x["vol_oi_ratio"])
            factors.append(f"异常 Call: {top['exp']} ${top['strike']} vol/oi={top['vol_oi_ratio']}x IV={top['iv']:.2f}")
        if max_put_iv_vs_oi:
            top = max(max_put_iv_vs_oi, key=lambda x: x["vol_oi_ratio"])
            factors.append(f"异常 Put: {top['exp']} ${top['strike']} vol/oi={top['vol_oi_ratio']}x IV={top['iv']:.2f}")

        return {
            "available": True,
            "put_call_ratio": round(put_call_ratio, 2),
            "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
            "total_call_volume": total_call_vol, "total_put_volume": total_put_vol,
            "unusual_calls": max_call_iv_vs_oi[:5],
            "unusual_puts": max_put_iv_vs_oi[:5],
            "direction": direction,
            "factors": factors,
        }
    except Exception as e:  # noqa: BLE001
        log.debug("options %s: %s", symbol, e)
        return {"available": False, "error": str(e)}


def short_interest(symbol: str) -> dict:
    if fetcher.is_a_share(symbol):
        return {"available": False, "reason": "A 股做空数据缺"}
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        sp = info.get("shortPercentOfFloat") or info.get("sharesPercentSharesOut")
        sr = info.get("shortRatio")  # days to cover
        sso = info.get("sharesShort")
        factors = []
        direction = "neutral"
        if sp:
            if sp > 0.20:
                factors.append(f"高做空比 {sp*100:.1f}% (squeeze 候选)")
                direction = "potential_squeeze"
            elif sp > 0.10:
                factors.append(f"做空比 {sp*100:.1f}% (偏高)")
            elif sp < 0.02:
                factors.append(f"做空比 {sp*100:.1f}% (机构看多)")
        if sr and sr > 5:
            factors.append(f"shorts 平仓需 {sr:.1f} 天 (压力大)")
        return {
            "available": True,
            "short_pct_of_float": sp,
            "short_ratio_days": sr,
            "shares_short": sso,
            "direction": direction,
            "factors": factors,
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)}


def insider_net_flow(symbol: str, *, days: int = 90) -> dict:
    """Recent Form 4 filings — net buy vs net sell (count + dollar 估算)."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        rows = conn.execute(
            """SELECT title FROM news_archive
            WHERE source = ? AND fetched_at >= ?""",
            (f"sec-form4-{symbol.lower()}", cutoff),
        ).fetchall()
    n = len(rows)
    # Without parsing each Form 4 in detail, give count-based heuristic
    # (full parsing of XML transactions is complex)
    return {
        "available": n > 0,
        "filing_count_90d": n,
        "factors": [f"过去 {days} 天 {n} 个 Form 4 备案"] if n else [],
        "note": "粗粒度: 仅统计备案数, 未拆解买/卖 (那需 XML 解析)",
    }


def aggregate(symbol: str) -> dict:
    out = {
        "symbol": symbol,
        "volume": volume_anomaly(symbol),
        "options": options_unusual(symbol),
        "short": short_interest(symbol),
        "insider": insider_net_flow(symbol),
    }
    # Composite trade signal direction
    score = 0.0
    factors_all = []
    n_signals = 0
    for k in ["volume", "options", "short"]:
        sig = out.get(k, {})
        if not sig.get("available"):
            continue
        n_signals += 1
        d = sig.get("direction", "neutral")
        weight = {"volume": 0.4, "options": 0.4, "short": 0.2}[k]
        sign = {"bullish": 1, "weak_bullish": 0.5, "speculative_bullish": 0.7,
                "bearish": -1, "weak_bearish": -0.5, "hedging_bearish": -0.7,
                "potential_squeeze": 0.3, "neutral": 0}.get(d, 0)
        score += sign * weight
        factors_all += sig.get("factors", [])
    out["composite_direction_score"] = round(score, 3)
    out["all_factors"] = factors_all
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = aggregate(args.symbol)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
