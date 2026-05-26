"""组合视角告警 — 不是单股 spam, 看主题板块层异动.

为什么: 单股 intraday alerter 在板块齐跌时会喷大量孤立告警, 用户看到的是
"为什么今天又跌了" 而不是 "今天 AI 主题 -X%". portfolio_pulse 补这条
组合层视角.

设计:
  - US close (21:30 UTC, Mon-Fri) 跑一次
  - CN close (07:30 UTC, Mon-Fri) 跑一次
  - 每次按 theme 分桶 (ai_compute / ai_memory / ai_power / broad_market /
    cn_gaming / 其它) 算:
      - 桶内今日加权涨跌 (按 share*price 权重)
      - 桶内最大涨/跌的标的
      - 是否"齐动" (≥3 只同向 ≥3% = systemic)
  - 输出一条 TG: "🇺🇸 美股 close: 今日 -2.5%, AI 主题领跌 -3.5% (ARM -10 / VRT -5 主导)"
  - 写一行进 events 表 (category=portfolio_pulse) 留痕

只推有意义的:
  - 桶内总变动 ≥ 2% 或 单只 ≥ 5% 才进 TG
  - "齐动" 加 ⚠️ 标记
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from . import config as cfg_mod, db, fetcher, telegram

log = logging.getLogger(__name__)

CATEGORY = "portfolio_pulse"


def _last_two_closes(symbol: str) -> Optional[tuple[float, float, str]]:
    """Return (today_close, yesterday_close, today_date) or None."""
    df = fetcher.load_local(symbol)
    if df is None or df.empty or len(df) < 2:
        return None
    df = df.sort_index()
    today_close = float(df["close"].iloc[-1])
    yest_close = float(df["close"].iloc[-2])
    today_date = str(df.index[-1])[:10]
    return today_close, yest_close, today_date


def _classify_market(symbol: str) -> str:
    return "cn" if fetcher.is_a_share(symbol) else "us"


def _build_pulse(market: str) -> dict:
    """Compute pulse for a single market (us|cn)."""
    portfolio = cfg_mod.load("portfolio")
    held = portfolio.get("positions", {})
    rows = []
    for sym, info in held.items():
        if _classify_market(sym) != market:
            continue
        d = _last_two_closes(sym)
        if not d:
            continue
        today_p, yest_p, dt = d
        chg_pct = (today_p / yest_p - 1) * 100 if yest_p else 0.0
        shares = float(info.get("shares", 0))
        value_today = shares * today_p
        rows.append({
            "symbol": sym,
            "name": info.get("name", sym),
            "theme": info.get("theme", "other"),
            "chg_pct": round(chg_pct, 2),
            "value_today": round(value_today, 2),
            "value_yest": round(shares * yest_p, 2),
            "shares": shares,
            "today_close": today_p,
            "yest_close": yest_p,
            "date": dt,
        })
    if not rows:
        return {"market": market, "n": 0, "msg": None}

    total_today = sum(r["value_today"] for r in rows)
    total_yest = sum(r["value_yest"] for r in rows)
    portfolio_chg_pct = (total_today / total_yest - 1) * 100 if total_yest else 0.0

    # Group by theme
    theme_buckets: dict[str, dict] = {}
    for r in rows:
        t = r["theme"]
        b = theme_buckets.setdefault(t, {
            "value_today": 0.0, "value_yest": 0.0,
            "members": [], "n_up_3pct": 0, "n_dn_3pct": 0,
        })
        b["value_today"] += r["value_today"]
        b["value_yest"] += r["value_yest"]
        b["members"].append(r)
        if r["chg_pct"] >= 3: b["n_up_3pct"] += 1
        if r["chg_pct"] <= -3: b["n_dn_3pct"] += 1

    theme_lines = []
    for t, b in theme_buckets.items():
        bucket_chg = (b["value_today"] / b["value_yest"] - 1) * 100 if b["value_yest"] else 0.0
        # systemic = ≥3 members same direction ≥3%
        systemic_up = b["n_up_3pct"] >= 3
        systemic_dn = b["n_dn_3pct"] >= 3
        # representative: biggest contributor by abs $ change
        contribs = sorted(
            b["members"],
            key=lambda x: abs(x["value_today"] - x["value_yest"]),
            reverse=True,
        )
        top3 = contribs[:3]
        leaders = " / ".join(f"{m['symbol']} {m['chg_pct']:+.1f}%" for m in top3)
        flag = ""
        if systemic_dn: flag = " ⚠️齐跌"
        elif systemic_up: flag = " 🚀齐涨"
        theme_lines.append({
            "theme": t,
            "chg_pct": round(bucket_chg, 2),
            "value": round(b["value_today"], 2),
            "n": len(b["members"]),
            "leaders": leaders,
            "flag": flag,
            "systemic": systemic_up or systemic_dn,
        })

    # sort themes by abs change
    theme_lines.sort(key=lambda x: abs(x["chg_pct"]), reverse=True)

    ccy = "¥" if market == "cn" else "$"
    market_label = "🇨🇳 A 股" if market == "cn" else "🇺🇸 美股"
    header = (
        f"{market_label} *组合 close 视角* ({datetime.utcnow().strftime('%Y-%m-%d')})\n"
        f"组合: {ccy}{total_today:,.0f} (今日 {portfolio_chg_pct:+.2f}%, "
        f"{ccy}{total_today - total_yest:+,.0f})"
    )
    body_lines = []
    for tl in theme_lines:
        body_lines.append(
            f"  • *{tl['theme']}* {tl['chg_pct']:+.2f}%{tl['flag']} "
            f"({ccy}{tl['value']:,.0f}, n={tl['n']}): {tl['leaders']}"
        )
    msg = header + "\n\n" + "\n".join(body_lines)

    # Decide: push only if portfolio move ≥ 2% OR any systemic theme OR any single ≥ 5%
    extreme_single = any(abs(r["chg_pct"]) >= 5 for r in rows)
    any_systemic = any(tl["systemic"] for tl in theme_lines)
    should_push = abs(portfolio_chg_pct) >= 2.0 or any_systemic or extreme_single

    return {
        "market": market,
        "n": len(rows),
        "portfolio_chg_pct": round(portfolio_chg_pct, 2),
        "total_today": round(total_today, 2),
        "total_yest": round(total_yest, 2),
        "themes": theme_lines,
        "rows": rows,
        "msg": msg,
        "should_push": should_push,
        "extreme_single": extreme_single,
        "systemic": any_systemic,
    }


def _write_event(pulse: dict) -> int:
    """Persist pulse summary into events table."""
    payload = {
        "market": pulse["market"],
        "portfolio_chg_pct": pulse["portfolio_chg_pct"],
        "themes": [{"theme": t["theme"], "chg_pct": t["chg_pct"],
                     "systemic": t["systemic"]} for t in pulse["themes"]],
        "extreme_single": pulse["extreme_single"],
    }
    sev = 6 if pulse["systemic"] else 5
    if abs(pulse["portfolio_chg_pct"]) >= 3: sev = 7
    affected = ",".join(r["symbol"] for r in pulse["rows"])
    summary = (
        f"[{pulse['market'].upper()}] 组合 {pulse['portfolio_chg_pct']:+.2f}%, "
        f"主题: " + ", ".join(
            f"{t['theme']} {t['chg_pct']:+.1f}%" for t in pulse["themes"][:3]
        )
    )
    with db.conn() as c:
        cur = c.execute(
            "INSERT INTO events(news_id, severity, category, summary, impact_json, "
            "                   affected_symbols, fired_at) "
            "VALUES (NULL, ?, ?, ?, ?, ?, ?)",
            (sev, CATEGORY, summary, json.dumps(payload, ensure_ascii=False),
             affected, datetime.utcnow().isoformat() + "Z"),
        )
        c.commit()
        return cur.lastrowid


def _push_tg(pulse: dict, chat_id: str) -> bool:
    msg = pulse["msg"]
    try:
        telegram.send(msg, chat_id=chat_id)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("portfolio_pulse TG push failed: %s", e)
        return False


def run_market(market: str, *, dry_run: bool = False) -> dict:
    pulse = _build_pulse(market)
    if pulse.get("n", 0) == 0:
        log.info("portfolio_pulse %s: empty (no holdings or no price data)", market)
        return pulse

    if dry_run:
        return pulse

    eid = _write_event(pulse)
    pulse["event_id"] = eid

    if pulse.get("should_push"):
        portfolio = cfg_mod.load("portfolio")
        chat_id = str(portfolio.get("telegram_target", ""))
        if _push_tg(pulse, chat_id):
            with db.conn() as c:
                c.execute(
                    "UPDATE events SET pushed_at=? WHERE id=?",
                    (datetime.utcnow().isoformat() + "Z", eid),
                )
                c.commit()
            pulse["pushed"] = True
        else:
            pulse["pushed"] = False
    else:
        pulse["pushed"] = False
        log.info("portfolio_pulse %s: written event %d, but below push threshold "
                  "(portfolio %.2f%%, systemic=%s, extreme=%s)",
                  market, eid, pulse["portfolio_chg_pct"],
                  pulse["systemic"], pulse["extreme_single"])
    return pulse


def run_all(*, dry_run: bool = False) -> dict:
    results = {}
    for market in ("us", "cn"):
        try:
            results[market] = run_market(market, dry_run=dry_run)
        except Exception as e:  # noqa: BLE001
            log.exception("portfolio_pulse %s failed: %s", market, e)
            results[market] = {"error": str(e)[:200]}
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "cn", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    db.init()

    if args.market == "all":
        r = run_all(dry_run=args.dry_run)
    else:
        r = run_market(args.market, dry_run=args.dry_run)

    # Avoid dumping rows (verbose); show summaries
    if isinstance(r, dict) and "msg" in r:
        print(r["msg"])
        print()
        print(json.dumps({k: v for k, v in r.items()
                          if k not in {"msg", "rows", "themes"}},
                         ensure_ascii=False, indent=2, default=str))
    else:
        for m, mr in r.items():
            print(f"=== {m} ===")
            if isinstance(mr, dict) and mr.get("msg"):
                print(mr["msg"])
                print()
            print(json.dumps({k: v for k, v in (mr or {}).items()
                              if k not in {"msg", "rows", "themes"}},
                             ensure_ascii=False, indent=2, default=str))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
