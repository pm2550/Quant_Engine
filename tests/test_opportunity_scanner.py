"""Tests for opportunity_scanner (Phase C, 2026-05-26)."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from quant import opportunity_scanner


@pytest.fixture
def tmp_state(monkeypatch, tmp_path):
    """Patch both opportunity_scanner.STATE_FILE and tracked_candidates.STATE_FILE
    so tests don't leak writes into the real /data2/quant/results files."""
    from quant import tracked_candidates as tc
    state_file = tmp_path / "opp_state.json"
    tracked_file = tmp_path / "tracked.json"
    monkeypatch.setattr(opportunity_scanner, "STATE_FILE", state_file)
    monkeypatch.setattr(tc, "STATE_FILE", tracked_file)
    return state_file


# ---- ignore list logic ----

def test_is_ignored_inside_window():
    today = date.today()
    ignore = {"NET": (today + timedelta(days=10)).isoformat()}
    assert opportunity_scanner._is_ignored("NET", ignore) is True


def test_is_ignored_expired():
    today = date.today()
    ignore = {"NET": (today - timedelta(days=1)).isoformat()}
    assert opportunity_scanner._is_ignored("NET", ignore) is False


def test_is_ignored_not_in_list():
    assert opportunity_scanner._is_ignored("NET", {}) is False


def test_is_ignored_malformed():
    assert opportunity_scanner._is_ignored("NET", {"NET": "not-a-date"}) is False


# ---- cooldown ----

def test_cooldown_active_recent(tmp_state):
    state = {"last_pushed": {"AAPL": (datetime.utcnow() - timedelta(hours=2)).isoformat()}}
    assert opportunity_scanner._cooldown_active("AAPL", state, hours=24) is True


def test_cooldown_expired(tmp_state):
    state = {"last_pushed": {"AAPL": (datetime.utcnow() - timedelta(hours=25)).isoformat()}}
    assert opportunity_scanner._cooldown_active("AAPL", state, hours=24) is False


def test_cooldown_no_state():
    assert opportunity_scanner._cooldown_active("AAPL", {"last_pushed": {}}, hours=24) is False


# ---- portfolio exclusion ----

def test_portfolio_symbols_includes_positions_and_watchlist(monkeypatch):
    fake = {
        "positions": {"AMD": {}, "VOO": {}},
        "watchlist": [{"symbol": "NVDA"}, {"symbol": "PLTR"}],
    }
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load", lambda name: fake)
    syms = opportunity_scanner._portfolio_symbols()
    assert syms == {"AMD", "VOO", "NVDA", "PLTR"}


# ---- run_scan end-to-end ----

def test_run_scan_filters_portfolio_and_returns_high_composite(monkeypatch, tmp_state, tmp_path):
    # Mock universe: META (will be candidate) + AMD (already in portfolio)
    universe_yaml = {"universe": [
        {"symbol": "META", "theme": "ai_software", "reason": "Llama AI"},
        {"symbol": "AMD", "theme": "ai_compute", "reason": "已在持仓"},
    ], "ignore": {}}
    monkeypatch.setattr(opportunity_scanner, "_load_universe", lambda: universe_yaml)
    monkeypatch.setattr(opportunity_scanner, "_portfolio_symbols", lambda: {"AMD"})
    # Phase D: also stub dynamic universe + tracked to keep test deterministic
    monkeypatch.setattr(opportunity_scanner.universe_discovery, "load_dynamic_universe", lambda: [])
    monkeypatch.setattr(opportunity_scanner.tracked_candidates, "load_state", lambda: {})

    # Mock strategies config + signals + multi_factor.score
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load",
                        lambda name: {"telegram_target": "fake"} if name == "portfolio" else {})
    # Inject a fake DataFrame for fetcher.load_local (>=50 rows)
    df = pd.DataFrame({"close": list(range(100, 200))})
    monkeypatch.setattr(opportunity_scanner.fetcher, "load_local", lambda s: df)

    class _Sig:
        def as_dict(self):
            return {"signal_codes": [], "rsi": 50, "above_ma50": True, "above_ma200": True,
                    "chg_20d_pct": 5}
    monkeypatch.setattr(opportunity_scanner.signals_mod, "compute", lambda *a, **kw: _Sig())

    monkeypatch.setattr(opportunity_scanner.multi_factor, "score",
                        lambda sym, sd, **kw: {
                            "composite_score": 0.42,
                            "conviction": 2,
                            "action": "ADD",
                            "top_factors": [{"name": "events", "contribution": 0.16, "evidence": "..."}],
                            "counter_factors": [],
                        })

    out = opportunity_scanner.run_scan(dry_run=True, push=False)
    assert out["n_candidates"] == 1
    assert out["candidates"][0]["symbol"] == "META"
    # AMD filtered out (in portfolio)


def test_run_scan_respects_threshold(monkeypatch, tmp_state):
    universe_yaml = {"universe": [
        {"symbol": "META", "theme": "ai_software", "reason": "x"},
    ], "ignore": {}}
    monkeypatch.setattr(opportunity_scanner, "_load_universe", lambda: universe_yaml)
    monkeypatch.setattr(opportunity_scanner, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load", lambda name: {})
    monkeypatch.setattr(opportunity_scanner.universe_discovery, "load_dynamic_universe", lambda: [])
    monkeypatch.setattr(opportunity_scanner.tracked_candidates, "load_state", lambda: {})

    df = pd.DataFrame({"close": list(range(100, 200))})
    monkeypatch.setattr(opportunity_scanner.fetcher, "load_local", lambda s: df)

    class _Sig:
        def as_dict(self): return {}
    monkeypatch.setattr(opportunity_scanner.signals_mod, "compute", lambda *a, **kw: _Sig())
    # composite = 0.25 < 0.30 threshold → no candidate
    monkeypatch.setattr(opportunity_scanner.multi_factor, "score",
                        lambda sym, sd, **kw: {"composite_score": 0.25, "conviction": 1,
                                                "top_factors": [], "counter_factors": []})
    out = opportunity_scanner.run_scan(threshold=0.30, dry_run=True, push=False)
    assert out["n_candidates"] == 0


def test_run_scan_respects_cooldown(monkeypatch, tmp_state, tmp_path):
    universe_yaml = {"universe": [{"symbol": "META", "theme": "ai", "reason": "x"}], "ignore": {}}
    monkeypatch.setattr(opportunity_scanner, "_load_universe", lambda: universe_yaml)
    monkeypatch.setattr(opportunity_scanner, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load", lambda name: {})
    monkeypatch.setattr(opportunity_scanner.universe_discovery, "load_dynamic_universe", lambda: [])
    monkeypatch.setattr(opportunity_scanner.tracked_candidates, "load_state", lambda: {})

    # 2 小时前推过 → 仍在 cooldown
    tmp_state.write_text(json.dumps({
        "last_pushed": {"META": (datetime.utcnow() - timedelta(hours=2)).isoformat()}
    }))
    # Even if everything else green, META should be skipped
    df = pd.DataFrame({"close": list(range(100, 200))})
    monkeypatch.setattr(opportunity_scanner.fetcher, "load_local", lambda s: df)

    class _Sig:
        def as_dict(self): return {}
    monkeypatch.setattr(opportunity_scanner.signals_mod, "compute", lambda *a, **kw: _Sig())
    monkeypatch.setattr(opportunity_scanner.multi_factor, "score",
                        lambda sym, sd, **kw: {"composite_score": 0.7, "conviction": 4,
                                                "top_factors": [], "counter_factors": []})

    out = opportunity_scanner.run_scan(cooldown_hours=24, dry_run=True, push=False)
    assert out["n_candidates"] == 0


def test_run_scan_ranks_by_composite(monkeypatch, tmp_state):
    universe_yaml = {"universe": [
        {"symbol": "META", "theme": "x", "reason": ""},
        {"symbol": "GOOG", "theme": "x", "reason": ""},
        {"symbol": "AAPL", "theme": "x", "reason": ""},
    ], "ignore": {}}
    monkeypatch.setattr(opportunity_scanner, "_load_universe", lambda: universe_yaml)
    monkeypatch.setattr(opportunity_scanner, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load", lambda name: {})
    monkeypatch.setattr(opportunity_scanner.universe_discovery, "load_dynamic_universe", lambda: [])
    monkeypatch.setattr(opportunity_scanner.tracked_candidates, "load_state", lambda: {})
    df = pd.DataFrame({"close": list(range(100, 200))})
    monkeypatch.setattr(opportunity_scanner.fetcher, "load_local", lambda s: df)

    class _Sig:
        def as_dict(self): return {}
    monkeypatch.setattr(opportunity_scanner.signals_mod, "compute", lambda *a, **kw: _Sig())

    score_by_sym = {"META": 0.35, "GOOG": 0.60, "AAPL": 0.45}
    monkeypatch.setattr(opportunity_scanner.multi_factor, "score",
                        lambda sym, sd, **kw: {"composite_score": score_by_sym[sym],
                                                "conviction": 3,
                                                "top_factors": [], "counter_factors": []})

    out = opportunity_scanner.run_scan(dry_run=True, push=False)
    assert [c["symbol"] for c in out["candidates"]] == ["GOOG", "AAPL", "META"]


# ============================================================================
# Phase D integration: dynamic universe + tracked candidates
# ============================================================================

@pytest.fixture
def tmp_tracked(monkeypatch, tmp_path):
    from quant import tracked_candidates as tc
    f = tmp_path / "tracked.json"
    monkeypatch.setattr(tc, "STATE_FILE", f)
    return f


def test_build_combined_universe_merges_static_dynamic_tracked(monkeypatch):
    from quant import universe_discovery, tracked_candidates
    monkeypatch.setattr(opportunity_scanner, "_load_universe",
                        lambda: {"universe": [{"symbol": "META", "theme": "static"}]})
    monkeypatch.setattr(universe_discovery, "load_dynamic_universe",
                        lambda: [{"symbol": "GOOG", "sources": ["events"], "reason": "..."}])
    monkeypatch.setattr(tracked_candidates, "load_state",
                        lambda: {"AAPL": {"first_added_at": "2026-05-20"}})

    entries, sources_per_sym = opportunity_scanner._build_combined_universe()
    syms = {e["symbol"] for e in entries}
    assert syms == {"META", "GOOG", "AAPL"}
    assert "static" in sources_per_sym["META"]
    assert "dynamic" in sources_per_sym["GOOG"]
    assert "tracked" in sources_per_sym["AAPL"]


def test_run_scan_tracks_low_conviction_scored(monkeypatch, tmp_state, tmp_tracked):
    """Conviction 1 (低于阈值) — 不推 TG 但若在 tracked 里要更新历史."""
    from quant import universe_discovery, tracked_candidates as tc

    # META is already being tracked (3 days history)
    initial_state = {
        "META": {
            "first_added_at": "2026-05-20T00:00:00Z",
            "last_seen_at": "2026-05-24T00:00:00Z",
            "conviction_history": [3, 3, 3],
            "score_history": [0.31, 0.32, 0.30],
            "sources": ["dynamic"],
            "promoted_at": None,
        }
    }
    tc.save_state(initial_state)

    monkeypatch.setattr(opportunity_scanner, "_load_universe", lambda: {"universe": [], "ignore": {}})
    monkeypatch.setattr(universe_discovery, "load_dynamic_universe",
                        lambda: [{"symbol": "META", "sources": ["events"]}])
    monkeypatch.setattr(opportunity_scanner, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load",
                        lambda name: {"telegram_target": "fake"} if name == "portfolio" else {})

    df = pd.DataFrame({"close": list(range(100, 200))})
    monkeypatch.setattr(opportunity_scanner.fetcher, "load_local", lambda s: df)

    class _Sig:
        def as_dict(self): return {}
    monkeypatch.setattr(opportunity_scanner.signals_mod, "compute", lambda *a, **kw: _Sig())
    # composite 0.05 → conviction 0 → 不入 push, 但 tracked 应继续记
    monkeypatch.setattr(opportunity_scanner.multi_factor, "score",
                        lambda sym, sd, **kw: {"composite_score": 0.05, "conviction": 0,
                                                "top_factors": [], "counter_factors": []})

    monkeypatch.setattr(opportunity_scanner.telegram, "send", lambda *a, **kw: {"ok": True})

    out = opportunity_scanner.run_scan(dry_run=False, push=False)
    assert out["n_candidates"] == 0  # below threshold
    # tracked state for META should have 4 conviction entries now
    state = tc.load_state()
    assert state["META"]["conviction_history"] == [3, 3, 3, 0]


def test_run_scan_emits_promotion(monkeypatch, tmp_state, tmp_tracked):
    """META 已 tracked 2 天 conviction>=4, 今天再得 4 → promotion 触发."""
    from quant import universe_discovery, tracked_candidates as tc

    initial_state = {
        "META": {
            "first_added_at": "2026-05-23T00:00:00Z",
            "last_seen_at": "2026-05-24T00:00:00Z",
            "conviction_history": [4, 4],
            "score_history": [0.45, 0.48],
            "sources": ["events"],
            "promoted_at": None,
        }
    }
    tc.save_state(initial_state)

    monkeypatch.setattr(opportunity_scanner, "_load_universe", lambda: {"universe": [], "ignore": {}})
    monkeypatch.setattr(universe_discovery, "load_dynamic_universe", lambda: [])
    monkeypatch.setattr(opportunity_scanner, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load",
                        lambda name: {"telegram_target": "fake"} if name == "portfolio" else {})

    df = pd.DataFrame({"close": list(range(100, 200))})
    monkeypatch.setattr(opportunity_scanner.fetcher, "load_local", lambda s: df)

    class _Sig:
        def as_dict(self): return {}
    monkeypatch.setattr(opportunity_scanner.signals_mod, "compute", lambda *a, **kw: _Sig())
    monkeypatch.setattr(opportunity_scanner.multi_factor, "score",
                        lambda sym, sd, **kw: {"composite_score": 0.50, "conviction": 4,
                                                "top_factors": [], "counter_factors": []})

    sent: list[str] = []
    monkeypatch.setattr(opportunity_scanner.telegram, "send",
                        lambda text, chat_id=None: sent.append(text) or {"ok": True})

    out = opportunity_scanner.run_scan(dry_run=False, push=True)
    assert out["n_promotions"] >= 1
    assert any("META" in s and "升级" in s for s in sent)
    # mark_promoted should have been set
    state = tc.load_state()
    assert state["META"]["promoted_at"] is not None


def test_run_scan_writes_state_after_push(monkeypatch, tmp_state):
    universe_yaml = {"universe": [{"symbol": "META", "theme": "x", "reason": ""}],
                     "ignore": {}}
    monkeypatch.setattr(opportunity_scanner, "_load_universe", lambda: universe_yaml)
    monkeypatch.setattr(opportunity_scanner, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(opportunity_scanner.cfg_mod, "load",
                        lambda name: {"telegram_target": "fake"} if name == "portfolio" else {})
    monkeypatch.setattr(opportunity_scanner.universe_discovery, "load_dynamic_universe", lambda: [])
    monkeypatch.setattr(opportunity_scanner.tracked_candidates, "load_state", lambda: {})
    df = pd.DataFrame({"close": list(range(100, 200))})
    monkeypatch.setattr(opportunity_scanner.fetcher, "load_local", lambda s: df)

    class _Sig:
        def as_dict(self): return {}
    monkeypatch.setattr(opportunity_scanner.signals_mod, "compute", lambda *a, **kw: _Sig())
    monkeypatch.setattr(opportunity_scanner.multi_factor, "score",
                        lambda sym, sd, **kw: {"composite_score": 0.5, "conviction": 3,
                                                "top_factors": [], "counter_factors": []})
    sent: list[str] = []
    monkeypatch.setattr(opportunity_scanner.telegram, "send",
                        lambda text, chat_id=None: sent.append(text) or {"ok": True})

    out = opportunity_scanner.run_scan(dry_run=False, push=True)
    assert out["n_candidates"] == 1
    assert sent  # telegram was called
    state = json.loads(tmp_state.read_text())
    assert "META" in state["last_pushed"]
