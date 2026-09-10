"""Persist每条 recommendation (含 HOLD) 供 30 天后复盘打分。

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
# D6 (2026-09-10): HOLD 原本被排除, 结果 4 个月的 392 条决策只覆盖 6 只 watchlist 标的,
# 11 只实际持仓一条都没有 —— 而"继续持有"本身就是一个决策, 也该被复盘。更要紧的是:
# 没有 HOLD 就没有横截面, 而 composite 唯一被证实有效的正是横截面排序 (rank IC +0.44)。
LOGGABLE_ACTIONS = {"ADD", "WATCH_BUY", "REDUCE", "WATCH_SKIP", "STOP_LOSS",
                    "DEFER_TO_LLM", "HOLD"}


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
    """Bulk-log all recommendations (HOLD included since D6) from orchestrator.run().

    同时把 composite 写进 model_predictions, 让 quant.calibration 能像评估
    challenger 一样评估 composite 的横截面预测力。

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
            _log_composite_prediction(
                symbol=sym, decided_at=decided_at,
                composite=multi.get("composite_score"),
                anchor=float(entry_price) if entry_price is not None else None,
            )
        except Exception:
            log.exception("decision_log insert failed for %s", sym)
            skipped += 1
    return {"logged": logged, "skipped": skipped}


def _log_composite_prediction(*, symbol: str, decided_at: datetime,
                               composite: float | None, anchor: float | None) -> None:
    """把 composite 当成一条"预测"存进 model_predictions, 供 calibration 统一评估。

    composite 不是收益预测而是一个排序分, 但 rank IC 只关心序 —— 存原值即可。
    """
    if composite is None:
        return
    try:
        with db.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO model_predictions
                   (snapshot_date, model, symbol, horizon_days, pred_value,
                    as_of, anchor_close, stale_days, extra_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (decided_at.strftime("%Y-%m-%d"), "multi_factor_composite", symbol,
                 REVIEW_HORIZON_DAYS, float(composite),
                 decided_at.strftime("%Y-%m-%d"), anchor, 0,
                 json.dumps({"note": "composite is a rank score, not a return forecast"},
                            ensure_ascii=False)),
            )
    except Exception:  # noqa: BLE001
        log.exception("model_predictions insert failed for %s", symbol)


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
