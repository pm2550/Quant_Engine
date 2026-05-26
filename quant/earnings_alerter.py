"""财报预警 - 持仓/关注池标的财报前 N 天推送提醒.

工作流:
  1. 每天 13:00 UTC (北京 21:00, 美股开盘前) 跑一次
  2. 拉持仓 + watchlist 所有标的的 earnings_calendar
  3. 找到 7 天内的 + 当天的财报
  4. 对每个 即将财报标的:
     - 拉历史 4 次财报后 1 天/5 天的股价反应
     - 对照 EPS surprise % vs 股价反应 (是否常超预期)
     - LLM 总结 + 提醒
  5. 推 TG (一条汇总, 多只股共一条)
  6. 状态记录避免重复推送
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd

from . import config as cfg_mod, db, fetcher, llm_router, telegram, earnings_calendar

log = logging.getLogger(__name__)

# Track which (symbol, report_date) pairs have already been alerted
ALERT_STATE_FILE = cfg_mod.RESULTS_DIR / "earnings_alerts.json"


def _load_state() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"sent": []}


def _save_state(state: dict) -> None:
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2))


def _historical_reactions(symbol: str, n: int = 4) -> list[dict]:
    """For each past earnings date, compute next-day and 5-day stock reaction."""
    out: list[dict] = []
    df = fetcher.load_local(symbol)
    if df.empty:
        return out
    df = df.sort_index()
    with sqlite3.connect(db.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM earnings_calendar WHERE symbol=? AND eps_actual IS NOT NULL "
            "ORDER BY report_date DESC LIMIT ?",
            (symbol, n),
        ).fetchall()
    for r in rows:
        rd = pd.Timestamp(r["report_date"])
        # next trading day after report
        forward = df[df.index > rd]
        if forward.empty:
            continue
        next_close = float(forward["close"].iloc[0])
        # the close on report day or last close before
        prior = df[df.index <= rd]
        if prior.empty:
            continue
        report_close = float(prior["close"].iloc[-1])
        d1 = (next_close / report_close - 1) * 100
        # 5 trading days later
        if len(forward) >= 5:
            d5_close = float(forward["close"].iloc[4])
            d5 = (d5_close / report_close - 1) * 100
        else:
            d5 = None
        out.append({
            "report_date": r["report_date"],
            "eps_estimate": r["eps_estimate"],
            "eps_actual": r["eps_actual"],
            "surprise_pct": r["surprise_pct"],
            "reaction_1d_pct": round(d1, 2),
            "reaction_5d_pct": round(d5, 2) if d5 is not None else None,
        })
    return out


def upcoming_earnings(*, days: int = 7) -> list[dict]:
    """Find any held/watch symbols with earnings in next N days."""
    today = date.today()
    end = today + timedelta(days=days)
    portfolio = cfg_mod.load("portfolio")
    held = set(portfolio.get("positions", {}).keys())
    watch = {w["symbol"] for w in portfolio.get("watchlist", [])}
    relevant = held | watch

    with sqlite3.connect(db.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM earnings_calendar WHERE report_date BETWEEN ? AND ? "
            "AND eps_actual IS NULL ORDER BY report_date",
            (today.isoformat(), end.isoformat()),
        ).fetchall()
    out = []
    for r in rows:
        if r["symbol"] not in relevant:
            continue
        info = portfolio["positions"].get(r["symbol"]) or next(
            (w for w in portfolio["watchlist"] if w["symbol"] == r["symbol"]), {}
        )
        days_until = (pd.Timestamp(r["report_date"]).date() - today).days
        out.append({
            "symbol": r["symbol"],
            "name": info.get("name", r["symbol"]),
            "is_held": r["symbol"] in held,
            "report_date": r["report_date"],
            "days_until": days_until,
            "eps_estimate": r["eps_estimate"],
            "revenue_estimate": r["revenue_estimate"],
            "history": _historical_reactions(r["symbol"], n=4),
        })
    return out


def _alert_key(symbol: str, report_date: str, days_bucket: str) -> str:
    return f"{symbol}:{report_date}:{days_bucket}"


def _bucket(days_until: int) -> str:
    if days_until <= 0:
        return "today"
    if days_until <= 1:
        return "1day"
    if days_until <= 3:
        return "3day"
    if days_until <= 7:
        return "7day"
    return "later"


def render_alert(items: list[dict]) -> str:
    if not items:
        return ""
    lines = [f"📅 *财报提醒 — 未来 7 天*", ""]
    for it in sorted(items, key=lambda x: x["days_until"]):
        sym = it["symbol"]
        nm = it["name"]
        d = it["days_until"]
        when = "**今天 ⚡**" if d == 0 else f"**{d} 天后**" if d > 0 else f"**{abs(d)} 天前 (已过)**"
        tag = "持仓" if it["is_held"] else "关注"
        eps_est = it.get("eps_estimate")
        rev_est = it.get("revenue_estimate")
        lines.append(f"• `{sym}` {nm} ({tag}) - {when} {it['report_date']}")
        bits = []
        if eps_est:
            bits.append(f"EPS 一致预期 ${eps_est:.2f}")
        if rev_est:
            bits.append(f"营收预期 ${rev_est/1e9:.2f}B")
        if bits:
            lines.append(f"   {' | '.join(bits)}")

        # Historical reaction
        hist = it.get("history", [])
        if hist:
            avg_d1 = sum(h["reaction_1d_pct"] for h in hist) / len(hist)
            beats = sum(1 for h in hist if (h.get("surprise_pct") or 0) > 0)
            lines.append(f"   过去 {len(hist)} 次: 财报次日平均 {avg_d1:+.1f}%, {beats}/{len(hist)} 次超预期")
            # Show last 1
            if hist:
                h = hist[0]
                surp = h.get("surprise_pct", 0) or 0
                lines.append(f"   上次 {h['report_date']}: 实际 ${h['eps_actual']} (预期 ${h['eps_estimate']}, surprise {surp:+.1f}%) → 次日 {h['reaction_1d_pct']:+.1f}%")
        lines.append("")
    return "\n".join(lines).rstrip()


def run_once(*, dry_run: bool = False) -> int:
    earnings_calendar.refresh_all()  # always refresh first
    upcoming = upcoming_earnings(days=7)
    if not upcoming:
        log.info("no upcoming earnings in 7 days")
        return 0

    state = _load_state()
    sent = set(state.get("sent", []))

    new_items = []
    for it in upcoming:
        key = _alert_key(it["symbol"], it["report_date"], _bucket(it["days_until"]))
        if key in sent:
            continue
        new_items.append(it)
        sent.add(key)

    if not new_items:
        log.info("all upcoming earnings already alerted in current bucket")
        return 0

    text = render_alert(new_items)
    if dry_run:
        print(text)
        return len(new_items)

    portfolio = cfg_mod.load("portfolio")
    try:
        telegram.send(text, chat_id=portfolio["telegram_target"])
        log.info("pushed earnings alert: %d items", len(new_items))
        state["sent"] = sorted(sent)[-200:]
        _save_state(state)
    except Exception:
        log.exception("push failed")
    return len(new_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    n = run_once(dry_run=args.dry_run)
    print(f"earnings_alerter: {n} new alerts")


if __name__ == "__main__":
    main()
