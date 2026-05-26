"""Persist each non-HOLD recommendation so 30-day review can score accuracy.

Written by daily.py at end of run.
Reviewed by decision_review.py monthly (or ad-hoc).
"""
from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from . import db

log = logging.getLogger(__name__)

REVIEW_HORIZON_DAYS = 30
LOGGABLE_ACTIONS = {"ADD", "WATCH_BUY", "REDUCE", "WATCH_SKIP", "STOP_LOSS", "DEFER_TO_LLM"}


def log_decision(
    *,
    symbol: str,
    action: str,
    composite_score: float | None,
    conviction: int | None,
    entry_price: float | None,
    currency: str | None,
    top_factors: list[dict] | None,
    counter_factors: list[dict] | None,
    decided_at: datetime | None = None,
    review_horizon_days: int = REVIEW_HORIZON_DAYS,
) -> int | None:
    """Insert a decision row. Returns inserted id or None if action not loggable."""
    if action not in LOGGABLE_ACTIONS:
        return None
    decided_at = decided_at or datetime.utcnow()
    review_due = decided_at + timedelta(days=review_horizon_days)
    with db.conn() as c:
        cur = c.execute(
            """INSERT INTO decision_log
               (decided_at, symbol, action, composite_score, conviction, entry_price,
                currency, top_factors_json, counter_factors_json, review_due_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                decided_at.isoformat(),
                symbol,
                action,
                composite_score,
                conviction,
                entry_price,
                currency,
                json.dumps(top_factors or [], ensure_ascii=False, default=str),
                json.dumps(counter_factors or [], ensure_ascii=False, default=str),
                review_due.isoformat(),
            ),
        )
        return cur.lastrowid


def log_from_raw(raw: dict[str, Any]) -> dict[str, int]:
    """Bulk-log all non-HOLD recommendations from orchestrator.run() output.

    Returns {'logged': N, 'skipped': M} counts.
    """
    logged = 0
    skipped = 0
    decided_at = datetime.utcnow()
    multi_scores = raw.get("multi_factor", {}) or {}
    for rec in raw.get("recommendations") or []:
        sym = rec.get("symbol")
        action = rec.get("action")
        if not sym or action not in LOGGABLE_ACTIONS:
            skipped += 1
            continue
        multi = multi_scores.get(sym) or {}
        notes = rec.get("notes") or {}
        entry_price = None
        # 1st try notes.price (sometimes injected by orchestrator), else fall back to signals
        if isinstance(notes, dict):
            entry_price = notes.get("price")
        sig = (raw.get("signals") or {}).get(sym) or {}
        if entry_price is None:
            entry_price = sig.get("price") or sig.get("close")

        try:
            log_decision(
                symbol=sym,
                action=action,
                composite_score=multi.get("composite_score"),
                conviction=multi.get("conviction"),
                entry_price=float(entry_price) if entry_price is not None else None,
                currency=rec.get("currency"),
                top_factors=multi.get("top_factors"),
                counter_factors=multi.get("counter_factors"),
                decided_at=decided_at,
            )
            logged += 1
        except Exception:
            log.exception("decision_log insert failed for %s", sym)
            skipped += 1
    return {"logged": logged, "skipped": skipped}


def pending_reviews(now: datetime | None = None) -> list[sqlite3.Row]:
    """Decisions whose review_due_at has passed and haven't been reviewed yet."""
    now = (now or datetime.utcnow()).isoformat()
    with db.conn() as c:
        rows = c.execute(
            "SELECT * FROM decision_log WHERE reviewed_at IS NULL AND review_due_at <= ? "
            "ORDER BY review_due_at ASC",
            (now,),
        ).fetchall()
    return rows


def mark_reviewed(decision_id: int, *, actual_return_pct: float | None, was_correct: int | None,
                  reviewed_at: datetime | None = None) -> None:
    reviewed_at = (reviewed_at or datetime.utcnow()).isoformat()
    with db.conn() as c:
        c.execute(
            "UPDATE decision_log SET reviewed_at=?, actual_return_pct=?, was_correct=? WHERE id=?",
            (reviewed_at, actual_return_pct, was_correct, decision_id),
        )


def recent_decisions(*, days: int = 30, action: str | None = None) -> list[sqlite3.Row]:
    """For ad-hoc / weekly review."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    sql = "SELECT * FROM decision_log WHERE decided_at >= ?"
    args: list[Any] = [cutoff]
    if action:
        sql += " AND action=?"
        args.append(action)
    sql += " ORDER BY decided_at DESC"
    with db.conn() as c:
        return c.execute(sql, args).fetchall()
