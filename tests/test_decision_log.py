"""Tests for decision_log + decision_review (Phase B-2, 2026-05-26)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

from quant import decision_log, decision_review


# ---- DB fixture: temp SQLite with decision_log table ----

@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_quant.sqlite"
    monkeypatch.setattr(decision_log.db, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decided_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                composite_score REAL,
                conviction INTEGER,
                entry_price REAL,
                currency TEXT,
                top_factors_json TEXT,
                counter_factors_json TEXT,
                review_due_at TEXT NOT NULL,
                reviewed_at TEXT,
                actual_return_pct REAL,
                was_correct INTEGER
            );
            CREATE INDEX idx_dl_pending ON decision_log(reviewed_at, review_due_at);
            """
        )
    return db_path


# ---- log_decision basics ----

def test_log_decision_writes_row(temp_db):
    rid = decision_log.log_decision(
        symbol="AMD", action="ADD", composite_score=0.45, conviction=3,
        entry_price=100.0, currency="USD",
        top_factors=[{"name": "events", "contribution": 0.16, "evidence": "财报超预期"}],
        counter_factors=[],
    )
    assert rid is not None
    with sqlite3.connect(temp_db) as conn:
        row = conn.execute("SELECT * FROM decision_log WHERE id=?", (rid,)).fetchone()
    assert row[2] == "AMD"  # symbol
    assert row[3] == "ADD"  # action


def test_log_decision_skips_hold(temp_db):
    """HOLD 不入库 (review 时没意义)."""
    rid = decision_log.log_decision(
        symbol="AMD", action="HOLD", composite_score=0.05, conviction=0,
        entry_price=100.0, currency="USD",
        top_factors=[], counter_factors=[],
    )
    assert rid is None


def test_log_decision_review_due_30_days(temp_db):
    """review_due_at 默认 = decided_at + 30 天."""
    now = datetime.utcnow()
    decision_log.log_decision(
        symbol="AMD", action="ADD", composite_score=0.4, conviction=4,
        entry_price=100.0, currency="USD",
        top_factors=[], counter_factors=[], decided_at=now,
    )
    with sqlite3.connect(temp_db) as conn:
        review_due = conn.execute("SELECT review_due_at FROM decision_log LIMIT 1").fetchone()[0]
    delta = (datetime.fromisoformat(review_due) - now).total_seconds() / 86400
    assert 29.5 < delta < 30.5


# ---- log_from_raw bulk ----

def test_log_from_raw_filters_hold(temp_db):
    raw = {
        "recommendations": [
            {"symbol": "AMD", "action": "ADD", "currency": "USD", "notes": {"price": 200}},
            {"symbol": "VOO", "action": "HOLD", "currency": "USD", "notes": {"price": 500}},
            {"symbol": "TSLA", "action": "WATCH_SKIP", "currency": "USD", "notes": {"price": 280}},
            {"symbol": "002624.SZ", "action": "ADD", "currency": "CNY", "notes": {"price": 15}},
        ],
        "multi_factor": {
            "AMD": {"composite_score": 0.45, "conviction": 3, "top_factors": [], "counter_factors": []},
            "TSLA": {"composite_score": -0.15, "conviction": 1, "top_factors": [], "counter_factors": []},
            "002624.SZ": {"composite_score": 0.32, "conviction": 2, "top_factors": [], "counter_factors": []},
        },
        "signals": {},
    }
    counts = decision_log.log_from_raw(raw)
    # AMD ADD + TSLA WATCH_SKIP + 002624 ADD = 3 logged; VOO HOLD skipped
    assert counts == {"logged": 3, "skipped": 1}


# ---- pending_reviews ----

def test_pending_reviews_returns_due_unreviewed(temp_db):
    now = datetime.utcnow()
    with sqlite3.connect(temp_db) as conn:
        # 1 due + not reviewed
        conn.execute(
            "INSERT INTO decision_log (decided_at, symbol, action, entry_price, conviction, review_due_at) "
            "VALUES (?, 'A', 'ADD', 100, 3, ?)",
            ((now - timedelta(days=35)).isoformat(), (now - timedelta(days=5)).isoformat()),
        )
        # 1 due + already reviewed (should skip)
        conn.execute(
            "INSERT INTO decision_log (decided_at, symbol, action, entry_price, conviction, review_due_at, reviewed_at, actual_return_pct, was_correct) "
            "VALUES (?, 'B', 'ADD', 100, 3, ?, ?, 5.0, 1)",
            ((now - timedelta(days=40)).isoformat(), (now - timedelta(days=10)).isoformat(), now.isoformat()),
        )
        # 1 not yet due
        conn.execute(
            "INSERT INTO decision_log (decided_at, symbol, action, entry_price, conviction, review_due_at) "
            "VALUES (?, 'C', 'ADD', 100, 3, ?)",
            ((now - timedelta(days=5)).isoformat(), (now + timedelta(days=25)).isoformat()),
        )
    pending = decision_log.pending_reviews()
    syms = [r["symbol"] for r in pending]
    assert syms == ["A"]


# ---- review run_review end-to-end ----

def test_run_review_marks_was_correct(monkeypatch, temp_db):
    now = datetime.utcnow()
    with sqlite3.connect(temp_db) as conn:
        # ADD at $100, current is $120 → +20% → correct (action=ADD expected up)
        conn.execute(
            "INSERT INTO decision_log (decided_at, symbol, action, entry_price, conviction, review_due_at) "
            "VALUES (?, 'AMD', 'ADD', 100.0, 3, ?)",
            ((now - timedelta(days=35)).isoformat(), (now - timedelta(days=5)).isoformat()),
        )
        # WATCH_SKIP at $100, current is $98 → -2% < 5% → correct (avoided)
        conn.execute(
            "INSERT INTO decision_log (decided_at, symbol, action, entry_price, conviction, review_due_at) "
            "VALUES (?, 'TSLA', 'WATCH_SKIP', 100.0, 2, ?)",
            ((now - timedelta(days=35)).isoformat(), (now - timedelta(days=5)).isoformat()),
        )

    # Mock fetcher.load_local
    def fake_load(sym):
        if sym == "AMD":
            return pd.DataFrame({"close": [120.0]})
        if sym == "TSLA":
            return pd.DataFrame({"close": [98.0]})
        return pd.DataFrame()
    monkeypatch.setattr(decision_review.fetcher, "load_local", fake_load)

    out = decision_review.run_review(dry_run=True, push=False)
    assert out["reviewed"] == 2
    assert out["by_action"]["ADD"]["hit_rate"] == 1.0
    assert out["by_action"]["ADD"]["avg_return_pct"] == 20.0
    assert out["by_action"]["WATCH_SKIP"]["hit_rate"] == 1.0


def test_run_review_records_worst_miss(monkeypatch, temp_db):
    now = datetime.utcnow()
    with sqlite3.connect(temp_db) as conn:
        # ADD at $100, current $70 → -30% → incorrect (expected +)
        conn.execute(
            "INSERT INTO decision_log (decided_at, symbol, action, entry_price, conviction, review_due_at) "
            "VALUES (?, 'X', 'ADD', 100.0, 4, ?)",
            ((now - timedelta(days=35)).isoformat(), (now - timedelta(days=5)).isoformat()),
        )

    monkeypatch.setattr(decision_review.fetcher, "load_local",
                        lambda s: pd.DataFrame({"close": [70.0]}))
    out = decision_review.run_review(dry_run=True, push=False)
    assert out["worst_miss"]["symbol"] == "X"
    assert out["worst_miss"]["actual_return_pct"] == -30.0


def test_run_review_handles_empty_pending(monkeypatch, temp_db):
    """没有 pending 时不爆."""
    out = decision_review.run_review(dry_run=True, push=False)
    assert out == {"pending": 0, "reviewed": 0, "hit_rate": None}


# ---- _was_correct logic ----

def test_was_correct_add_action_up_wins():
    assert decision_review._was_correct(expected=1, return_pct=5.0) == 1
    assert decision_review._was_correct(expected=1, return_pct=-2.0) == 0


def test_was_correct_reduce_action_down_wins():
    assert decision_review._was_correct(expected=-1, return_pct=-5.0) == 1
    assert decision_review._was_correct(expected=-1, return_pct=10.0) == 0


def test_was_correct_defer_returns_none():
    """DEFER_TO_LLM 没有方向期望."""
    assert decision_review._was_correct(expected=0, return_pct=15) is None


def test_was_correct_watch_skip_small_move_still_correct():
    """WATCH_SKIP +3% 不算大踏空."""
    assert decision_review._was_correct(expected=-1, return_pct=3.0) == 1
    assert decision_review._was_correct(expected=-1, return_pct=10.0) == 0
