"""Today's notable events — consolidated summary for the daily TG digest.

Replaces per-event LLM push spam. Pulls the last 24h of news+price events
from the `events` table, filters to ones that materially affect held
positions, and renders a compact markdown section.

No LLM direction / confidence shown — those proved 50/50 noise in the
2026-06-01 calibration audit (239 predictions, hit-rate ≈ baseline).
We show only: severity, category, summary, affected portfolio symbols,
and historical base rate from similar past events (real data only).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Iterable

from . import config as cfg_mod
from . import db


def _portfolio_symbols() -> set[str]:
    p = cfg_mod.load("portfolio") or {}
    held = set((p.get("positions") or {}).keys())
    watch = {w.get("symbol") for w in (p.get("watchlist") or []) if w.get("symbol")}
    return held | watch


def _recent_events(*, hours: int = 24,
                   categories: Iterable[str] = ("geopolitical", "policy", "macro",
                                                  "industry", "company", "price_action"),
                   min_severity: int = 6) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    cats = ",".join(f"'{c}'" for c in categories)
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""SELECT id, category, severity, summary, impact_json, affected_symbols,
                       fired_at, news_id
            FROM events
            WHERE fired_at >= ?
              AND category IN ({cats})
              AND severity >= ?
              AND archived = 0
            ORDER BY severity DESC, fired_at DESC""",
            (cutoff, min_severity),
        ).fetchall()
    return [dict(r) for r in rows]


def _events_with_holding_overlap(events: list[dict], holdings: set[str],
                                  min_overlap: int = 2) -> list[dict]:
    """Keep only events that touch >= min_overlap held symbols."""
    out = []
    for e in events:
        sym_csv = e.get("affected_symbols") or ""
        affected = {s for s in sym_csv.split(",") if s}
        overlap = affected & holdings
        if len(overlap) >= min_overlap:
            e["overlap"] = sorted(overlap)
            out.append(e)
    return out


def render_section(*, hours: int = 24, top_k: int = 6,
                    min_overlap: int = 2) -> str:
    """Render the consolidated events section. Empty string if nothing meaningful."""
    holdings = _portfolio_symbols()
    if not holdings:
        return ""

    events = _recent_events(hours=hours)
    notable = _events_with_holding_overlap(events, holdings, min_overlap=min_overlap)
    if not notable:
        return ""

    notable = sorted(notable, key=lambda e: (e["severity"], e["fired_at"]),
                      reverse=True)[:top_k]

    lines = [
        f"📰 *今日重大事件 ({len(notable)} 条)*",
        f"_过去 {hours}h, sev≥6, 至少命中 {min_overlap} 只持仓; LLM 方向预测已隐藏 (经审计 hit≈50%)_",
        "",
    ]
    for e in notable:
        sev = e["severity"]
        cat = e["category"]
        summary = (e.get("summary") or "")[:120]
        when = (e.get("fired_at") or "")[5:16].replace("T", " ")  # MM-DD HH:MM
        overlap = e.get("overlap") or []
        ov_str = ", ".join(overlap[:5]) + (" 等" if len(overlap) > 5 else "")
        lines.append(f"• [{sev}] {cat} `{when}` — {summary}")
        lines.append(f"    持仓命中: {ov_str}")

        # If impact_json has base_rate (real historical data), surface the
        # 20d median return for the FIRST overlapping symbol. Skip LLM
        # direction/confidence entirely.
        try:
            ij = json.loads(e.get("impact_json") or "{}")
        except Exception:
            ij = {}
        for imp in (ij.get("impacts") or []):
            sym = imp.get("symbol")
            if sym not in overlap:
                continue
            br = imp.get("base_rate") or {}
            fr20 = br.get("fwd_20d_pct") or {}
            n = (fr20 or {}).get("n", 0)
            if n >= 3:
                med = fr20.get("median")
                lo = fr20.get("min")
                hi = fr20.get("max")
                lines.append(
                    f"    {sym} 历史 n={n}: 20d 中位 {med:+.1f}% [{lo:+.1f},{hi:+.1f}]"
                )
                break  # one example per event is enough
    return "\n".join(lines)


if __name__ == "__main__":
    out = render_section()
    print(out if out else "(no notable events in last 24h)")
