"""Track candidates that scored conviction >= 3 in opportunity_scanner.

Phase D (2026-05-26): 主动跟踪机制. Scanner 推过的 conv>=3/5 候选自动入 30 天 tracked list,
每天对 tracked 也跑 multi_factor 记录 conviction history.
连续 3 天 conviction >= 4 → 推 "建议正式加入 watchlist" 升级提示.

State file: results/tracked_candidates.json
Schema:
  {
    "SYM": {
        "first_added_at": "2026-05-26T...",
        "last_seen_at":   "2026-05-27T...",
        "conviction_history": [3, 4, 4],  # last 30 entries
        "score_history":      [0.31, 0.42, 0.45],
        "sources": ["events", "theme_etf"],
        "promoted_at": null   # set when promote signal emitted
    }
  }
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from . import config as cfg_mod

log = logging.getLogger(__name__)

STATE_FILE = cfg_mod.RESULTS_DIR / "tracked_candidates.json"

TTL_DAYS = 30                # tracked entries pruned after this many days no-touch
ADD_CONVICTION_THRESHOLD = 3  # conviction >= 3 → start tracking
PROMOTE_WINDOW = 3            # require last N daily scores
PROMOTE_CONVICTION_THRESHOLD = 4  # all of last N must be >= this
HISTORY_CAP = 30              # cap each entry's history to last N days


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def tracked_symbols(state: dict) -> set[str]:
    return set(state.keys())


def update_for_scored(
    state: dict,
    scored: list[dict],
    *,
    now: datetime | None = None,
) -> dict:
    """Process today's scoring results.

    For each scored entry (dict with at least symbol + conviction + composite_score):
      - If symbol in state: append to history (cap at HISTORY_CAP), update last_seen_at.
      - If symbol NOT in state AND conviction >= ADD_CONVICTION_THRESHOLD: start tracking.
      - If symbol NOT in state AND conviction < threshold: ignore.

    Returns updated state.
    """
    now = now or datetime.utcnow()
    now_iso = now.isoformat() + "Z"
    for entry in scored:
        sym = entry.get("symbol")
        if not sym:
            continue
        conv = int(entry.get("conviction") or 0)
        score = float(entry.get("composite_score") or 0)
        sources = entry.get("sources") or []
        if sym in state:
            st = state[sym]
            st["conviction_history"].append(conv)
            st["score_history"].append(round(score, 3))
            st["conviction_history"] = st["conviction_history"][-HISTORY_CAP:]
            st["score_history"] = st["score_history"][-HISTORY_CAP:]
            st["last_seen_at"] = now_iso
            # merge sources
            existing_src = set(st.get("sources") or [])
            existing_src.update(sources)
            st["sources"] = sorted(existing_src)
        elif conv >= ADD_CONVICTION_THRESHOLD:
            state[sym] = {
                "first_added_at": now_iso,
                "last_seen_at": now_iso,
                "conviction_history": [conv],
                "score_history": [round(score, 3)],
                "sources": list(sources),
                "promoted_at": None,
            }
    return state


def find_promotion_candidates(state: dict) -> list[dict]:
    """Return entries where last PROMOTE_WINDOW conviction values all >= threshold
    AND not yet promoted."""
    out = []
    for sym, st in state.items():
        if st.get("promoted_at"):
            continue
        hist = st.get("conviction_history") or []
        if len(hist) < PROMOTE_WINDOW:
            continue
        recent = hist[-PROMOTE_WINDOW:]
        if all(c >= PROMOTE_CONVICTION_THRESHOLD for c in recent):
            out.append({
                "symbol": sym,
                "conviction_recent": recent,
                "score_recent": (st.get("score_history") or [])[-PROMOTE_WINDOW:],
                "sources": st.get("sources") or [],
                "tracking_since": st.get("first_added_at"),
            })
    return out


def mark_promoted(state: dict, symbol: str, *, now: datetime | None = None) -> None:
    if symbol in state:
        state[symbol]["promoted_at"] = ((now or datetime.utcnow()).isoformat() + "Z")


def prune_expired(state: dict, *, now: datetime | None = None,
                   ttl_days: int = TTL_DAYS) -> tuple[dict, list[str]]:
    """Remove entries last_seen_at older than ttl_days.

    Returns (state, removed_symbols).
    """
    now = now or datetime.utcnow()
    cutoff = now - timedelta(days=ttl_days)
    removed = []
    for sym in list(state.keys()):
        last = state[sym].get("last_seen_at")
        if not last:
            continue
        try:
            last_dt = datetime.fromisoformat(last.replace("Z", ""))
        except ValueError:
            continue
        if last_dt < cutoff:
            del state[sym]
            removed.append(sym)
    return state, removed


def format_promotion_message(promotions: list[dict]) -> str:
    """Markdown for TG: 几只 tracked candidates 连续 N 天 conviction>=4."""
    parts = [
        f"🚀 *候选股升级建议 — {datetime.utcnow().date().isoformat()}*",
        f"以下 {len(promotions)} 只在 tracked 中连续 {PROMOTE_WINDOW} 天 conviction>={PROMOTE_CONVICTION_THRESHOLD}/5\n"
    ]
    for p in promotions:
        recent = p["conviction_recent"]
        scores = p["score_recent"]
        sources = ", ".join(p["sources"] or [])
        parts.append(f"*{p['symbol']}* — 跟踪起始 {p['tracking_since'][:10]}")
        parts.append(f"  最近 {len(recent)} 天 conviction: {recent} (composite {scores})")
        parts.append(f"  发现源: {sources}")
        parts.append(f"  💡 建议: 回复 `/watch {p['symbol']}` 加入 watchlist 正式跟踪\n")
    return "\n".join(parts)
