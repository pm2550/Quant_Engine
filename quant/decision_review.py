"""30-day review of past decisions: compute hit_rate by conviction tier, push TG.

Triggered monthly by quant-decision-review.timer (Phase B-2, 2026-05-26).
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from . import config as cfg_mod, db, decision_log, fetcher, telegram

log = logging.getLogger(__name__)


# Direction expected from each action (1 = should go up, -1 = should go down, 0 = neutral/skip).
ACTION_TO_EXPECTED_DIRECTION: dict[str, int] = {
    "ADD": 1,
    "WATCH_BUY": 1,
    "REDUCE": -1,
    "STOP_LOSS": -1,
    "WATCH_SKIP": -1,    # 期望: 不买后这只股没大涨 (即 <0% 或基本持平算对)
    "DEFER_TO_LLM": 0,
}


def _current_price(symbol: str) -> float | None:
    df = fetcher.load_local(symbol)
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def _was_correct(expected: int, return_pct: float) -> int | None:
    """1 if direction matched. WATCH_SKIP 反着算: < +5% 算对 (avoided)."""
    if expected == 0:
        return None
    if expected == 1:
        return 1 if return_pct > 0 else 0
    if expected == -1:
        # 包含 WATCH_SKIP: 不买后该股不应该大涨; 5% 以下当作 "没明显错过"
        return 1 if return_pct < 5 else 0
    return None


def run_review(*, dry_run: bool = False, push: bool = True) -> dict:
    rows = decision_log.pending_reviews()
    if not rows:
        log.info("no pending reviews")
        return {"pending": 0, "reviewed": 0, "hit_rate": None}

    by_action: dict[str, list[tuple[sqlite3.Row, float, int | None]]] = defaultdict(list)
    by_conviction: dict[int, list[tuple[sqlite3.Row, float, int | None]]] = defaultdict(list)
    all_returns: list[float] = []

    reviewed = 0
    for row in rows:
        sym = row["symbol"]
        entry = row["entry_price"]
        if entry is None or entry <= 0:
            log.warning("skip %s id=%s no entry_price", sym, row["id"])
            continue
        cur = _current_price(sym)
        if cur is None:
            log.warning("skip %s id=%s no current price", sym, row["id"])
            continue
        return_pct = (cur - entry) / entry * 100
        expected = ACTION_TO_EXPECTED_DIRECTION.get(row["action"], 0)
        ok = _was_correct(expected, return_pct)
        if not dry_run:
            decision_log.mark_reviewed(row["id"], actual_return_pct=return_pct, was_correct=ok)
        by_action[row["action"]].append((row, return_pct, ok))
        by_conviction[row["conviction"] or 0].append((row, return_pct, ok))
        all_returns.append(return_pct)
        reviewed += 1

    # Aggregate
    summary: dict = {
        "pending": len(rows),
        "reviewed": reviewed,
        "avg_return_pct": round(sum(all_returns) / len(all_returns), 2) if all_returns else None,
        "by_action": {},
        "by_conviction": {},
    }

    for action, items in by_action.items():
        with_ok = [ok for _, _, ok in items if ok is not None]
        rets = [r for _, r, _ in items]
        summary["by_action"][action] = {
            "n": len(items),
            "hit_rate": round(sum(with_ok) / len(with_ok), 3) if with_ok else None,
            "avg_return_pct": round(sum(rets) / len(rets), 2) if rets else None,
        }

    for conv, items in by_conviction.items():
        with_ok = [ok for _, _, ok in items if ok is not None]
        rets = [r for _, r, _ in items]
        summary["by_conviction"][conv] = {
            "n": len(items),
            "hit_rate": round(sum(with_ok) / len(with_ok), 3) if with_ok else None,
            "avg_return_pct": round(sum(rets) / len(rets), 2) if rets else None,
        }

    # Worst miss
    worst: tuple[sqlite3.Row, float, int | None] | None = None
    for items in by_action.values():
        for triple in items:
            row, r, ok = triple
            if ok == 0 and (worst is None or abs(r) > abs(worst[1])):
                worst = triple
    if worst:
        wrow, wret, _ = worst
        summary["worst_miss"] = {
            "symbol": wrow["symbol"],
            "action": wrow["action"],
            "decided_at": wrow["decided_at"],
            "entry_price": wrow["entry_price"],
            "actual_return_pct": round(wret, 2),
            "conviction": wrow["conviction"],
        }

    # Markdown report
    md = _format_review_md(summary, rows)
    rpt_dir = cfg_mod.ROOT / "reports"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    rpt_path = rpt_dir / f"decision_review_{date.today().strftime('%Y%m')}.md"
    rpt_path.write_text(md, encoding="utf-8")
    log.info("wrote %s", rpt_path)

    if push and not dry_run and reviewed > 0:
        try:
            portfolio = cfg_mod.load("portfolio")
            telegram.send(md, chat_id=portfolio["telegram_target"])
        except Exception:
            log.exception("review push failed")

    if dry_run:
        print(md)

    return summary


def _format_review_md(summary: dict, rows: list[sqlite3.Row]) -> str:
    parts = [
        f"📊 *决策复盘 — {date.today().isoformat()}*",
        f"过去 30 天共 *{summary['reviewed']}* 条建议已到期复盘",
    ]
    if summary.get("avg_return_pct") is not None:
        parts.append(f"平均回报 (entry→今): *{summary['avg_return_pct']:+.2f}%*")

    if summary.get("by_action"):
        parts.append("\n*按 action 分组:*")
        for action, stats in summary["by_action"].items():
            hr = f"{stats['hit_rate'] * 100:.0f}%" if stats["hit_rate"] is not None else "N/A"
            r = f"{stats['avg_return_pct']:+.2f}%" if stats["avg_return_pct"] is not None else "?"
            parts.append(f"  • {action:12s} n={stats['n']:2d}  命中 {hr:>4s}  avg {r}")

    if summary.get("by_conviction"):
        parts.append("\n*按 conviction 分组:*")
        for conv in sorted(summary["by_conviction"].keys(), reverse=True):
            stats = summary["by_conviction"][conv]
            hr = f"{stats['hit_rate'] * 100:.0f}%" if stats["hit_rate"] is not None else "N/A"
            r = f"{stats['avg_return_pct']:+.2f}%" if stats["avg_return_pct"] is not None else "?"
            stars = "★" * conv + "☆" * (5 - conv) if conv > 0 else "-"
            parts.append(f"  • {stars}  n={stats['n']:2d}  命中 {hr:>4s}  avg {r}")

    if summary.get("worst_miss"):
        w = summary["worst_miss"]
        parts.append(f"\n*🚨 最大踏空/误判:* `{w['symbol']}` {w['action']} "
                     f"(信心 {w['conviction']}/5) → 实际 {w['actual_return_pct']:+.2f}%")

    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="run review but don't send TG")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = run_review(dry_run=args.dry_run, push=not args.no_push)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
