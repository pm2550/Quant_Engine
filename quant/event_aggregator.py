"""事件统一视图 - 给定 symbol, 列出所有相关事件的"窗口位置 + 方向 + 量级".

来源:
  • earnings_calendar  (财报)
  • corporate_events    (除权/分红/分拆)
  • macro_events        (FOMC/CPI/NFP/conferences)
  • events              (newswatch LLM 评级 + impact 推演的事件)
  • news_archive (近 N 天 SEC Form 4 / 8-K / 13F 进入的)

每个事件标记:
  • days_offset: -N (已过 N 天) ~ +M (M 天后), 0 = 今天
  • event_type: earnings / ex_dividend / product_launch / policy / insider / 8k / 13f / news
  • direction: bullish / bearish / neutral / two_sided (重大事件 binary outcome)
  • magnitude_pct: 估计幅度 (如有, 来自历史 + LLM impact)
  • window_phase: pre_event / event_day / post_event / cooling
  • relevance_score: 0..1 (这个事件对该 symbol 影响多大)
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Optional

from . import config as cfg_mod, db

log = logging.getLogger(__name__)


# Event direction priors per type (覆盖未推演的)
DIRECTION_PRIORS = {
    "earnings_beat":     ("two_sided",   8.0),   # ±8% but skewed by surprise
    "earnings_miss":     ("bearish",    -6.0),
    "ex_dividend":       ("neutral",     0.0),
    "split":             ("bullish",     2.0),
    "product_launch":    ("two_sided",   5.0),
    "investor_day":      ("two_sided",   4.0),
    "industry_conference": ("bullish",   2.0),
    "fda_approval":      ("two_sided",  20.0),
    "buyback":           ("bullish",     3.0),
    "dividend_raise":    ("bullish",     2.0),
    "dividend_cut":      ("bearish",    -5.0),
    "ceo_change":        ("two_sided",   3.0),
    "lawsuit":           ("bearish",    -2.0),
    "acquisition_target": ("bullish",   15.0),
    "acquisition_acquirer": ("bearish", -3.0),
    "regulatory":        ("two_sided",   3.0),
    "form4_buy":         ("bullish",     1.5),
    "form4_sell":        ("bearish",    -1.0),
    "8k":                ("two_sided",   2.0),
    "13f":               ("neutral",     0.0),
    "fomc":              ("two_sided",   3.0),
    "cpi":               ("two_sided",   2.0),
    "nfp":               ("two_sided",   2.0),
}


def _window_phase(days_offset: int) -> str:
    if days_offset > 7:
        return "future"
    if 1 <= days_offset <= 7:
        return "pre_event"
    if days_offset == 0:
        return "event_day"
    if -3 <= days_offset <= -1:
        return "post_event"   # 即时反应窗口
    if -10 <= days_offset < -3:
        return "cooling"
    return "stale"


def _relevance(event_type: str, days_offset: int, magnitude: float) -> float:
    """How much should multi-factor scoring weight this event?"""
    # Time decay: most relevant at 0, falls off both sides
    if days_offset > 7 or days_offset < -10:
        return 0.0
    time_factor = 1.0 - min(abs(days_offset), 7) / 7
    # Magnitude scaling
    mag_factor = min(abs(magnitude) / 10, 1.0) if magnitude else 0.3
    # Type-specific base weights
    type_weights = {
        "earnings": 1.0, "earnings_beat": 1.0, "earnings_miss": 1.0,
        "fomc": 0.9, "fda_approval": 1.0,
        "product_launch": 0.7, "investor_day": 0.6,
        "ex_dividend": 0.3, "split": 0.4,
        "form4_buy": 0.5, "form4_sell": 0.5,
        "8k": 0.7, "13f": 0.3,
        "cpi": 0.5, "nfp": 0.5,
    }
    type_w = type_weights.get(event_type, 0.5)
    return round(time_factor * mag_factor * type_w, 3)


def _earnings_events(symbol: str, *, days: int = 14) -> list[dict]:
    today = date.today()
    out = []
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        # Upcoming earnings
        upcoming = conn.execute(
            """SELECT * FROM earnings_calendar
            WHERE symbol = ? AND eps_actual IS NULL
            AND report_date BETWEEN ? AND ?""",
            (symbol, (today - timedelta(days=10)).isoformat(),
             (today + timedelta(days=days)).isoformat()),
        ).fetchall()
        # Recent past earnings (post-event window)
        past = conn.execute(
            """SELECT * FROM earnings_calendar
            WHERE symbol = ? AND eps_actual IS NOT NULL
            AND report_date >= ?
            ORDER BY report_date DESC LIMIT 4""",
            (symbol, (today - timedelta(days=10)).isoformat()),
        ).fetchall()

    for r in upcoming:
        d_off = (datetime.fromisoformat(r["report_date"]).date() - today).days
        out.append({
            "type": "earnings",
            "label": "财报",
            "date": r["report_date"],
            "days_offset": d_off,
            "direction": "two_sided",
            "magnitude_pct": 8.0,  # typical earnings move
            "window_phase": _window_phase(d_off),
            "details": {
                "eps_estimate": r["eps_estimate"],
                "revenue_estimate": r["revenue_estimate"],
            },
        })

    for r in past:
        d_off = (datetime.fromisoformat(r["report_date"]).date() - today).days
        if d_off > 0:
            continue  # not actually past
        surprise = r["surprise_pct"] or 0
        direction = "bullish" if surprise > 0 else "bearish"
        out.append({
            "type": "earnings_beat" if surprise > 0 else "earnings_miss",
            "label": f"财报已出 ({'超' if surprise>0 else '不及'}预期 {surprise:+.1f}%)",
            "date": r["report_date"],
            "days_offset": d_off,
            "direction": direction,
            "magnitude_pct": min(abs(surprise) * 0.8, 12),
            "window_phase": _window_phase(d_off),
            "details": {
                "surprise_pct": surprise,
                "eps_actual": r["eps_actual"],
                "eps_estimate": r["eps_estimate"],
            },
        })
    return out


def _corporate_events_for(symbol: str, *, days: int = 14) -> list[dict]:
    today = date.today()
    out = []
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM corporate_events
            WHERE symbol = ? AND event_date BETWEEN ? AND ?""",
            (symbol, (today - timedelta(days=14)).isoformat(),
             (today + timedelta(days=days)).isoformat()),
        ).fetchall()
    for r in rows:
        d_off = (datetime.fromisoformat(r["event_date"]).date() - today).days
        et = r["event_type"]
        direction, mag = DIRECTION_PRIORS.get(et, ("neutral", 0))
        out.append({
            "type": et,
            "label": {"ex_dividend": "除权除息", "split": "股票分拆"}.get(et, et),
            "date": r["event_date"],
            "days_offset": d_off,
            "direction": direction,
            "magnitude_pct": mag,
            "window_phase": _window_phase(d_off),
            "details": {"amount": r["amount"], "notes": r["notes"]},
        })
    return out


def _macro_events_for(symbol: str, *, days: int = 14) -> list[dict]:
    """Macro events affect all symbols, but with different relevance."""
    today = date.today()
    out = []
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM macro_events
            WHERE event_date BETWEEN ? AND ?
            ORDER BY event_date""",
            ((today - timedelta(days=3)).isoformat(),
             (today + timedelta(days=days)).isoformat()),
        ).fetchall()

    for r in rows:
        d_off = (datetime.fromisoformat(r["event_date"]).date() - today).days
        et = r["event_type"]
        # Symbol-specific check: if conference event_type embeds the symbol
        if "-" in et:
            base, sym_in_event = et.split("-", 1)
            if sym_in_event != symbol and sym_in_event != "MACRO":
                continue
            display_type = base
        else:
            display_type = et.lower()

        direction, mag = DIRECTION_PRIORS.get(display_type, ("two_sided", 2.0))
        label_map = {"FOMC": "美联储议息", "CPI": "CPI", "NFP": "非农", "ECB": "欧央行"}
        label = label_map.get(display_type.upper(), display_type)
        out.append({
            "type": display_type,
            "label": label,
            "date": r["event_date"],
            "days_offset": d_off,
            "direction": direction,
            "magnitude_pct": mag,
            "window_phase": _window_phase(d_off),
            "details": {"region": r["region"], "notes": r["notes"]},
        })
    return out


def _news_events_for(symbol: str, *, hours: int = 96) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    today = date.today()
    out = []
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT severity, category, summary, impact_json, fired_at
            FROM events WHERE fired_at >= ? AND affected_symbols LIKE ?
            ORDER BY fired_at DESC LIMIT 10""",
            (cutoff, f"%{symbol}%"),
        ).fetchall()
    for r in rows:
        try:
            imp = json.loads(r["impact_json"]) if r["impact_json"] else {}
        except Exception:
            imp = {}
        # Extract direction for this symbol from impact
        direction, mag = "neutral", 0
        for i in imp.get("impacts", []):
            if i.get("symbol") == symbol:
                direction = i.get("direction", "neutral")
                mag = i.get("magnitude_pct", 0) or 0
                break
        try:
            fired_d = datetime.fromisoformat(r["fired_at"].replace("Z", "")).date()
            d_off = (fired_d - today).days
        except Exception:
            d_off = -1
        out.append({
            "type": "news",
            "label": f"新闻 ({r['category']})",
            "date": r["fired_at"][:10],
            "days_offset": d_off,
            "direction": direction,
            "magnitude_pct": mag,
            "window_phase": _window_phase(d_off),
            "details": {"severity": r["severity"], "summary": r["summary"][:200]},
        })
    return out


def _sec_events_for(symbol: str, *, hours: int = 168) -> list[dict]:
    """SEC filings already in news_archive via sec_edgar.py."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    today = date.today()
    out = []
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT title, source, published_at FROM news_archive
            WHERE source LIKE ? AND fetched_at >= ?
            ORDER BY published_at DESC LIMIT 10""",
            (f"sec-%-{symbol.lower()}", cutoff),
        ).fetchall()
    for r in rows:
        try:
            pd_d = datetime.fromisoformat(r["published_at"].replace("Z", "+00:00")).date()
            d_off = (pd_d - today).days
        except Exception:
            d_off = -1
        title = r["title"]
        if "Form 4" in title:
            et, label, direction, mag = "form4_sell", "Form 4 内部人交易", "neutral", 1.5
        elif "8-K" in title:
            et, label, direction, mag = "8k", "8-K 重大事件", "two_sided", 3.0
        elif "13F" in title:
            et, label, direction, mag = "13f", "13F 机构持仓变化", "neutral", 1.0
        else:
            et, label, direction, mag = "filing", "SEC 备案", "neutral", 1.0
        out.append({
            "type": et,
            "label": label,
            "date": r["published_at"][:10] if r["published_at"] else "?",
            "days_offset": d_off,
            "direction": direction,
            "magnitude_pct": mag,
            "window_phase": _window_phase(d_off),
            "details": {"title": title[:120]},
        })
    return out


def aggregate(symbol: str, *, future_days: int = 14, past_days: int = 7) -> dict:
    """Return all events affecting symbol, with phase + direction + relevance."""
    events = []
    events += _earnings_events(symbol, days=future_days)
    events += _corporate_events_for(symbol, days=future_days)
    events += _macro_events_for(symbol, days=future_days)
    events += _news_events_for(symbol, hours=past_days * 24 + 24)
    events += _sec_events_for(symbol, hours=(past_days + 7) * 24)

    for e in events:
        e["relevance_score"] = _relevance(e["type"], e["days_offset"], e["magnitude_pct"])

    # sort by absolute relevance descending
    events.sort(key=lambda x: -x["relevance_score"])

    # Compute aggregate event tilt: 加权方向偏移
    direction_score = 0.0
    total_weight = 0.0
    for e in events:
        if e["relevance_score"] < 0.05:
            continue
        sign = {"bullish": 1, "bearish": -1, "neutral": 0, "two_sided": 0}[e["direction"]]
        direction_score += sign * e["relevance_score"] * (abs(e["magnitude_pct"]) / 10)
        total_weight += e["relevance_score"]

    direction_tilt = direction_score / total_weight if total_weight else 0
    has_imminent_high_mag = any(
        e["window_phase"] in ("event_day", "pre_event")
        and abs(e["magnitude_pct"]) >= 5
        and e["relevance_score"] > 0.3
        for e in events
    )

    return {
        "symbol": symbol,
        "n_events": len(events),
        "events": events[:15],
        "direction_tilt": round(direction_tilt, 3),    # -1..+1
        "has_imminent_high_mag": has_imminent_high_mag,
        "summary_top3": [
            f"{e['label']} ({e['date']}, {e['days_offset']:+d}d, "
            f"{e['direction']}, mag {e['magnitude_pct']:+.1f}%, rel {e['relevance_score']})"
            for e in events[:3]
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = aggregate(args.symbol, future_days=args.days)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
