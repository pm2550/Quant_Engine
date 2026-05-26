"""统一的"重要日期" 聚合器 - 财报 + 除权日 + 宏观事件 + 行业活动.

Pull: earnings_calendar + corporate_events + macro_events 表
Render + push 一条汇总到 TG.
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta

from . import config as cfg_mod, db, telegram, earnings_alerter, corporate_events, macro_events

log = logging.getLogger(__name__)

ALERT_STATE_FILE = cfg_mod.RESULTS_DIR / "important_dates_alerts.json"


def _load_state() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"sent": []}


def _save_state(s: dict) -> None:
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_FILE.write_text(json.dumps(s, indent=2))


def aggregate(*, days: int = 7) -> dict:
    earnings = earnings_alerter.upcoming_earnings(days=days)
    corp = corporate_events.upcoming(days=days)
    macro = macro_events.upcoming(days=days)
    # filter corp by held/watch
    portfolio = cfg_mod.load("portfolio")
    relevant = set(portfolio.get("positions", {}).keys()) | {
        w["symbol"] for w in portfolio.get("watchlist", [])
    }
    corp = [c for c in corp if c["symbol"] in relevant]
    return {"earnings": earnings, "corporate": corp, "macro": macro}


def render(agg: dict) -> str:
    if not (agg["earnings"] or agg["corporate"] or agg["macro"]):
        return ""
    lines = [f"📅 *未来 7 天重要日期 — {date.today().isoformat()}*", ""]

    if agg["earnings"]:
        lines.append("*📊 财报:*")
        for e in agg["earnings"][:6]:
            d_until = e["days_until"]
            when = "**今天 ⚡**" if d_until == 0 else f"**{d_until} 天后**"
            tag = "持仓" if e["is_held"] else "关注"
            lines.append(f"  • `{e['symbol']}` ({tag}) {when} - {e['report_date']}")
            if e.get("eps_estimate"):
                lines.append(f"    EPS 预期 ${e['eps_estimate']:.2f}")
            hist = e.get("history", [])
            if hist:
                avg = sum(h["reaction_1d_pct"] for h in hist) / len(hist)
                beats = sum(1 for h in hist if (h.get("surprise_pct") or 0) > 0)
                lines.append(f"    历史 {len(hist)} 次: 平均次日 {avg:+.1f}%, {beats}/{len(hist)} 超预期")
        lines.append("")

    if agg["corporate"]:
        lines.append("*💰 除权除息/公司行为:*")
        for c in agg["corporate"][:5]:
            ev_d = c["event_date"]
            d_until = (datetime.fromisoformat(ev_d).date() - date.today()).days
            when = "今天 ⚡" if d_until == 0 else f"{d_until} 天后"
            type_zh = {"ex_dividend": "除权日 (持有股息)",
                       "split": "股票分拆"}.get(c["event_type"], c["event_type"])
            amt = f"${c['amount']:.2f}/股" if c.get("amount") else ""
            lines.append(f"  • `{c['symbol']}` {type_zh} - {when} {ev_d} {amt}")
        lines.append("")

    if agg["macro"]:
        lines.append("*🌍 宏观事件:*")
        for m in agg["macro"][:8]:
            ev_d = m["event_date"]
            d_until = (datetime.fromisoformat(ev_d).date() - date.today()).days
            when = "今天 ⚡" if d_until == 0 else f"{d_until} 天后"
            t_str = f" {m['event_time_utc']} UTC" if m.get("event_time_utc") else ""
            type_zh = {"FOMC": "美联储议息", "CPI": "CPI", "NFP": "非农就业",
                       "PPI": "PPI", "GDP": "GDP", "PMI": "PMI",
                       "ECB": "欧央行", "BOJ": "日央行"}.get(m["event_type"], m["event_type"])
            lines.append(f"  • {m['region']} {type_zh} - {when} {ev_d}{t_str}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _alert_keys(agg: dict) -> list[str]:
    keys = []
    today = date.today().isoformat()
    for e in agg["earnings"]:
        d_until = e["days_until"]
        bucket = "today" if d_until == 0 else "1day" if d_until == 1 else "week"
        keys.append(f"E:{e['symbol']}:{e['report_date']}:{bucket}")
    for c in agg["corporate"]:
        keys.append(f"C:{c['symbol']}:{c['event_type']}:{c['event_date']}")
    for m in agg["macro"]:
        d_until = (datetime.fromisoformat(m["event_date"]).date() - date.today()).days
        bucket = "today" if d_until == 0 else "1day" if d_until == 1 else "week"
        keys.append(f"M:{m['event_type']}:{m['region']}:{m['event_date']}:{bucket}")
    return keys


def run_once(*, dry_run: bool = False, days: int = 7) -> int:
    # Refresh data sources
    try:
        earnings_alerter.earnings_calendar.refresh_all()
    except Exception:
        log.exception("earnings refresh failed")
    try:
        corporate_events.refresh_all()
    except Exception:
        log.exception("corp events refresh failed")
    try:
        macro_events.refresh()
    except Exception:
        log.exception("macro events refresh failed")

    agg = aggregate(days=days)
    if not (agg["earnings"] or agg["corporate"] or agg["macro"]):
        log.info("no upcoming dates in next %d days", days)
        return 0

    state = _load_state()
    sent = set(state.get("sent", []))
    keys = _alert_keys(agg)
    new_keys = [k for k in keys if k not in sent]
    if not new_keys:
        log.info("all upcoming events already alerted")
        return 0

    text = render(agg)
    if dry_run:
        print(text)
        return len(new_keys)

    portfolio = cfg_mod.load("portfolio")
    try:
        telegram.send(text, chat_id=portfolio["telegram_target"])
        state["sent"] = sorted(set(sent) | set(new_keys))[-300:]
        _save_state(state)
        log.info("pushed important_dates alert: %d new keys", len(new_keys))
    except Exception:
        log.exception("push failed")
    return len(new_keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    n = run_once(dry_run=args.dry_run, days=args.days)
    print(f"important_dates: {n} new alerts")


if __name__ == "__main__":
    main()
