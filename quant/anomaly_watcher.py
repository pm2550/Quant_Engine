"""主动盯持仓异动 — 价格行为本身就是事件.

设计 (2026-05-08 重构, 解决"002624 涨停 8h 静默"):
  - 每 5min 扫一遍持仓 + 关注池
  - 对每只算: 1d/5d 涨跌, RSI 突变, MA200 上下穿, 量能放大
  - 任一异动触发 → 写 events 表 (category=price_action)
  - severity ≥ push_threshold 推 TG (24h 同 (symbol,type,bucket) cooldown)
  - 留 impact_json.investigation_status=pending 给 investigator daemon 接力

跟 newswatch 互补: newswatch 看新闻, 这个看价格. 两者都写 events 表.
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd

from . import config as cfg_mod, db, fetcher, signals, telegram

log = logging.getLogger(__name__)

CATEGORY = "price_action"

# magnitude bucket → severity (one-day move)
def _sev_for_1d(pct: float) -> int:
    a = abs(pct)
    if a >= 15: return 8
    if a >= 10: return 7
    if a >= 7:  return 6
    if a >= 5:  return 5
    return 0

# 5d cumulative
def _sev_for_5d(pct: float) -> int:
    a = abs(pct)
    if a >= 30: return 8
    if a >= 20: return 7
    if a >= 15: return 6
    return 0

def _bucket_pct(pct: float) -> int:
    a = abs(pct)
    for b in (15, 10, 7, 5):
        if a >= b: return b
    return 0


def _recently_seen(symbol: str, anom_type: str, bucket: int, hours: int = 24) -> bool:
    """24h cooldown on same (symbol, type, bucket) — regardless of push status.

    Bug fix 2026-05-08: previous version required pushed_at IS NOT NULL,
    but sev=5 events never push, so cooldown never matched and we wrote 22
    duplicate QCOM rows in 50min. Now: any prior event in window blocks.
    """
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    needle = f'"anomaly_type": "{anom_type}"'
    bucket_needle = f'"bucket": {bucket}'
    with db.conn() as c:
        row = c.execute(
            """SELECT 1 FROM events
               WHERE category = ?
                 AND affected_symbols = ?
                 AND fired_at >= ?
                 AND impact_json LIKE ?
                 AND impact_json LIKE ?
               LIMIT 1""",
            (CATEGORY, symbol, cutoff, f"%{needle}%", f"%{bucket_needle}%"),
        ).fetchone()
    return row is not None


def _detect(symbol: str, df: pd.DataFrame, strategies_cfg: dict) -> list[dict]:
    """Return list of triggered anomaly dicts.

    Each: {anomaly_type, severity, magnitude_pct, bucket, summary, ccy_symbol}
    """
    if df is None or df.empty or len(df) < 2:
        return []
    df = df.sort_index()
    last = float(df["close"].iloc[-1])
    is_cn = fetcher.is_a_share(symbol)
    ccy = "¥" if is_cn else "$"
    triggered: list[dict] = []

    # 1d move
    if len(df) >= 2:
        prev = float(df["close"].iloc[-2])
        if prev > 0:
            chg = (last / prev - 1) * 100
            sev = _sev_for_1d(chg)
            if sev >= 5:
                direction = "涨" if chg > 0 else "跌"
                triggered.append({
                    "anomaly_type": "price_1d",
                    "severity": sev,
                    "magnitude_pct": round(chg, 2),
                    "bucket": _bucket_pct(chg),
                    "summary": f"{symbol} 单日{direction} {chg:+.2f}% (现 {ccy}{last:.2f}, 昨 {ccy}{prev:.2f})",
                })

    # 5d cumulative
    if len(df) >= 6:
        c5 = float(df["close"].iloc[-6])
        if c5 > 0:
            chg = (last / c5 - 1) * 100
            sev = _sev_for_5d(chg)
            if sev >= 6:
                direction = "累涨" if chg > 0 else "累跌"
                triggered.append({
                    "anomaly_type": "price_5d",
                    "severity": sev,
                    "magnitude_pct": round(chg, 2),
                    "bucket": _bucket_pct(chg),
                    "summary": f"{symbol} 5 日{direction} {chg:+.2f}% (现 {ccy}{last:.2f})",
                })

    # RSI / MA200 — need signals
    sig = None
    if len(df) >= 15:
        try:
            sig = signals.compute(symbol, df, strategies_cfg)
        except Exception as e:  # noqa: BLE001
            log.debug("signals.compute %s failed: %s", symbol, e)

    if sig is not None and len(df) >= 16:
        try:
            sig_prev = signals.compute(symbol, df.iloc[:-1], strategies_cfg)
        except Exception:
            sig_prev = None
        if sig_prev is not None:
            r_now, r_prev = sig.rsi, sig_prev.rsi
            if pd.notna(r_now) and pd.notna(r_prev):
                if r_prev < 30 and r_now > 50:
                    triggered.append({
                        "anomaly_type": "rsi_flip_up",
                        "severity": 5,
                        "magnitude_pct": round(r_now - r_prev, 1),
                        "bucket": 0,
                        "summary": f"{symbol} RSI {r_prev:.0f}(超卖)→{r_now:.0f} 反弹",
                    })
                elif r_prev > 70 and r_now < 30:
                    triggered.append({
                        "anomaly_type": "rsi_flip_dn",
                        "severity": 6,
                        "magnitude_pct": round(r_now - r_prev, 1),
                        "bucket": 0,
                        "summary": f"{symbol} RSI {r_prev:.0f}(超买)→{r_now:.0f} 暴跌",
                    })

            # MA200 cross
            ma_now, ma_prev = sig.ma200, sig_prev.ma200
            if pd.notna(ma_now) and pd.notna(ma_prev) and ma_now > 0 and ma_prev > 0:
                prev_close = float(df["close"].iloc[-2])
                crossed_up = (prev_close < ma_prev) and (last > ma_now)
                crossed_dn = (prev_close > ma_prev) and (last < ma_now)
                if crossed_up:
                    triggered.append({
                        "anomaly_type": "ma200_breakout",
                        "severity": 6,
                        "magnitude_pct": round((last / ma_now - 1) * 100, 2),
                        "bucket": 0,
                        "summary": f"{symbol} 上穿 MA200 ({ccy}{ma_now:.2f}→{ccy}{last:.2f}) 长期翻多",
                    })
                elif crossed_dn:
                    triggered.append({
                        "anomaly_type": "ma200_breakdown",
                        "severity": 7,
                        "magnitude_pct": round((last / ma_now - 1) * 100, 2),
                        "bucket": 0,
                        "summary": f"{symbol} 跌破 MA200 ({ccy}{ma_now:.2f}→{ccy}{last:.2f}) 长期翻空",
                    })

    # Volume spike
    if "volume" in df.columns and len(df) >= 21:
        vol_today = float(df["volume"].iloc[-1])
        vol_avg = float(df["volume"].iloc[-21:-1].mean())
        if vol_avg > 0 and vol_today > 0:
            ratio = vol_today / vol_avg
            if ratio >= 2.0:
                sev = 6 if ratio >= 3 else 5
                triggered.append({
                    "anomaly_type": "volume_spike",
                    "severity": sev,
                    "magnitude_pct": round(ratio, 2),
                    "bucket": 3 if ratio >= 3 else 2,
                    "summary": f"{symbol} 量能放大 {ratio:.1f}x (今 {vol_today:,.0f} vs 20 日均 {vol_avg:,.0f})",
                })

    return triggered


def _write_event(symbol: str, anom: dict) -> int:
    impact = {
        "anomaly_type": anom["anomaly_type"],
        "magnitude_pct": anom["magnitude_pct"],
        "bucket": anom["bucket"],
        "investigation_status": "pending",   # investigator daemon picks up
    }
    with db.conn() as c:
        cur = c.execute(
            "INSERT INTO events(news_id, severity, category, summary, impact_json, "
            "                   affected_symbols, fired_at) "
            "VALUES (NULL, ?, ?, ?, ?, ?, ?)",
            (anom["severity"], CATEGORY, anom["summary"],
             json.dumps(impact, ensure_ascii=False),
             symbol,
             datetime.utcnow().isoformat() + "Z"),
        )
        c.commit()
        return cur.lastrowid


def _mark_pushed(event_id: int) -> None:
    """Mark event as 'will be pushed by investigator' to prevent later re-push."""
    with db.conn() as c:
        c.execute("UPDATE events SET pushed_at=? WHERE id=?",
                  (datetime.utcnow().isoformat() + "Z", event_id))
        c.commit()


def run_once(*, push_threshold: int = 6, dry_run: bool = False) -> dict:
    portfolio = cfg_mod.load("portfolio")
    strategies = cfg_mod.load("strategies")
    chat_id = str(portfolio.get("telegram_target", "6213084357"))
    held = list(portfolio.get("positions", {}).keys())
    watch = [w["symbol"] for w in portfolio.get("watchlist", [])]
    symbols = list(dict.fromkeys(held + watch))   # dedup, preserve order

    triggered = pushed = skipped_cooldown = 0
    triggered_list: list[dict] = []

    for sym in symbols:
        try:
            df = fetcher.load_local(sym)
            if df.empty:
                df = fetcher.fetch_symbol(sym)
            if df is None or df.empty:
                continue
            anomalies = _detect(sym, df, strategies)
            for a in anomalies:
                if _recently_seen(sym, a["anomaly_type"], a["bucket"]):
                    skipped_cooldown += 1
                    continue
                if dry_run:
                    triggered_list.append({"symbol": sym, **a})
                    triggered += 1
                    continue
                eid = _write_event(sym, a)
                triggered += 1
                a_with_id = {"symbol": sym, "event_id": eid, **a}
                triggered_list.append(a_with_id)
                # Don't push from anomaly_watcher anymore — investigator owns
                # the consolidated "异动 + 原因" push so user gets ONE message
                # per real event, not two (anomaly first, cause later).
                # We still mark sev<push_threshold events as suppressed so they
                # never push (investigator will skip them too).
                if a["severity"] < push_threshold:
                    _mark_pushed(eid)  # signal: suppress, don't TG-push
                    continue
                # else: leave pushed_at NULL → investigator will pick up + push
        except Exception as e:  # noqa: BLE001
            log.warning("anomaly check %s failed: %s", sym, e)

    return {
        "checked": len(symbols),
        "triggered": triggered,
        "pushed": pushed,
        "skipped_cooldown": skipped_cooldown,
        "triggered_list": triggered_list,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push-threshold", type=int, default=6,
                    help="severity ≥ this → push Telegram (default 6)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true",
                    help="detect only, no DB write / no TG push")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    db.init()

    if args.once:
        result = run_once(push_threshold=args.push_threshold, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    log.info("anomaly_watcher loop started, interval=%ds threshold=%d",
             args.interval, args.push_threshold)
    while True:
        try:
            r = run_once(push_threshold=args.push_threshold)
            log.info("checked=%d triggered=%d pushed=%d cooldown=%d",
                     r["checked"], r["triggered"], r["pushed"], r["skipped_cooldown"])
        except Exception as e:  # noqa: BLE001
            log.exception("anomaly_watcher iteration failed: %s", e)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
