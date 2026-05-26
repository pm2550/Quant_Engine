"""Tests for tracked_candidates (Phase D, 2026-05-26)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from quant import tracked_candidates as tc


@pytest.fixture
def tmp_state(monkeypatch, tmp_path):
    f = tmp_path / "tracked.json"
    monkeypatch.setattr(tc, "STATE_FILE", f)
    return f


# ---- update_for_scored ----

def test_update_adds_new_when_conviction_meets_threshold():
    state = {}
    scored = [
        {"symbol": "META", "conviction": 3, "composite_score": 0.32, "sources": ["events"]},
        {"symbol": "GOOG", "conviction": 1, "composite_score": 0.05, "sources": ["theme_etf"]},
    ]
    out = tc.update_for_scored(state, scored)
    assert "META" in out  # conviction 3 >= threshold
    assert "GOOG" not in out  # conviction 1 < threshold (won't start new tracking)


def test_update_appends_history_for_existing():
    state = {
        "META": {
            "first_added_at": "2026-05-20T00:00:00Z",
            "last_seen_at": "2026-05-25T00:00:00Z",
            "conviction_history": [3, 4],
            "score_history": [0.31, 0.42],
            "sources": ["events"],
            "promoted_at": None,
        }
    }
    scored = [{"symbol": "META", "conviction": 4, "composite_score": 0.45,
               "sources": ["theme_etf"]}]
    out = tc.update_for_scored(state, scored)
    assert out["META"]["conviction_history"] == [3, 4, 4]
    assert out["META"]["score_history"] == [0.31, 0.42, 0.45]
    # sources merged (events + theme_etf)
    assert set(out["META"]["sources"]) == {"events", "theme_etf"}


def test_update_appends_even_low_conviction_for_existing():
    """已 tracked 的, conviction 跌到 0 也要继续记 (用于复盘走向)."""
    state = {
        "META": {
            "first_added_at": "2026-05-20T00:00:00Z",
            "last_seen_at": "2026-05-25T00:00:00Z",
            "conviction_history": [3],
            "score_history": [0.31],
            "sources": ["events"],
            "promoted_at": None,
        }
    }
    out = tc.update_for_scored(state, [{"symbol": "META", "conviction": 0,
                                          "composite_score": -0.05, "sources": []}])
    assert out["META"]["conviction_history"] == [3, 0]


def test_update_caps_history_at_30():
    state = {
        "META": {
            "first_added_at": "2026-04-25T00:00:00Z",
            "last_seen_at": "2026-05-25T00:00:00Z",
            "conviction_history": list(range(30)),  # already at cap
            "score_history": [0.1] * 30,
            "sources": [],
            "promoted_at": None,
        }
    }
    out = tc.update_for_scored(state, [{"symbol": "META", "conviction": 5,
                                          "composite_score": 0.6}])
    assert len(out["META"]["conviction_history"]) == 30
    assert out["META"]["conviction_history"][-1] == 5
    assert out["META"]["conviction_history"][0] == 1  # 0 was dropped


# ---- find_promotion_candidates ----

def test_find_promotion_when_last_3_all_meet():
    state = {
        "META": {
            "first_added_at": "2026-05-20T00:00:00Z",
            "last_seen_at": "2026-05-26T00:00:00Z",
            "conviction_history": [3, 4, 4, 4],   # last 3 = [4, 4, 4]
            "score_history": [0.31, 0.42, 0.45, 0.50],
            "sources": ["events"],
            "promoted_at": None,
        },
        "OTHER": {
            "first_added_at": "2026-05-23T00:00:00Z",
            "last_seen_at": "2026-05-26T00:00:00Z",
            "conviction_history": [3, 4, 3],   # last 3 = [3, 4, 3] - not all >=4
            "score_history": [0.30, 0.40, 0.35],
            "sources": [],
            "promoted_at": None,
        },
    }
    promotions = tc.find_promotion_candidates(state)
    syms = {p["symbol"] for p in promotions}
    assert syms == {"META"}


def test_find_promotion_excludes_already_promoted():
    state = {
        "META": {
            "first_added_at": "2026-05-20T00:00:00Z",
            "last_seen_at": "2026-05-26T00:00:00Z",
            "conviction_history": [5, 5, 5],
            "score_history": [0.5, 0.5, 0.5],
            "sources": [],
            "promoted_at": "2026-05-25T00:00:00Z",
        }
    }
    promotions = tc.find_promotion_candidates(state)
    assert promotions == []


def test_find_promotion_needs_min_history_length():
    """少于 PROMOTE_WINDOW 个值不应该 promotee."""
    state = {
        "META": {
            "first_added_at": "2026-05-20T00:00:00Z",
            "last_seen_at": "2026-05-26T00:00:00Z",
            "conviction_history": [5, 5],  # only 2 vs window=3
            "score_history": [0.5, 0.5],
            "sources": [],
            "promoted_at": None,
        }
    }
    assert tc.find_promotion_candidates(state) == []


def test_mark_promoted_sets_timestamp():
    state = {"META": {"promoted_at": None}}
    tc.mark_promoted(state, "META")
    assert state["META"]["promoted_at"] is not None


# ---- prune ----

def test_prune_removes_old_entries():
    state = {
        "OLD": {
            "first_added_at": "2026-03-01T00:00:00Z",
            "last_seen_at": "2026-04-10T00:00:00Z",   # ~46 days ago vs today
            "conviction_history": [3],
            "score_history": [0.3],
            "sources": [],
            "promoted_at": None,
        },
        "FRESH": {
            "first_added_at": "2026-05-20T00:00:00Z",
            "last_seen_at": "2026-05-25T00:00:00Z",
            "conviction_history": [3],
            "score_history": [0.3],
            "sources": [],
            "promoted_at": None,
        },
    }
    now = datetime(2026, 5, 26)
    out, removed = tc.prune_expired(state, now=now, ttl_days=30)
    assert "OLD" not in out
    assert "FRESH" in out
    assert removed == ["OLD"]


# ---- format ----

def test_format_promotion_message_renders():
    promotions = [{
        "symbol": "META",
        "conviction_recent": [4, 4, 5],
        "score_recent": [0.4, 0.45, 0.50],
        "sources": ["events", "theme_etf"],
        "tracking_since": "2026-05-20T00:00:00Z",
    }]
    md = tc.format_promotion_message(promotions)
    assert "META" in md
    assert "[4, 4, 5]" in md
    assert "events" in md and "theme_etf" in md
    assert "/watch META" in md


# ---- save/load roundtrip ----

def test_save_and_load_roundtrip(tmp_state):
    tc.save_state({"META": {"foo": "bar"}})
    loaded = tc.load_state()
    assert loaded == {"META": {"foo": "bar"}}


def test_load_missing_file_returns_empty(tmp_state):
    assert tc.load_state() == {}


def test_load_corrupt_file_returns_empty(tmp_state):
    tmp_state.write_text("{not json")
    assert tc.load_state() == {}
