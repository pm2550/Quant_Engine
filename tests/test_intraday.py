"""Unit tests for intraday.py — market hour gating, move bucketing, alert rules.

The pure functions are the high-value targets: getting market hours wrong
means we ping yfinance off-hours (waste/throttle); getting alert rules
wrong means false-positive Telegram pings.  Network-touching paths
(`_fetch_us_intraday`, `_fetch_cn_intraday`) are integration territory and
left out here.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quant import intraday


# ---- Market hour gating ----


def test_us_market_hours_inside_window():
    # Tuesday 18:00 UTC = NYSE 13:00 ET (mid-session)
    t = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    assert intraday._is_us_market_hours(t) is True


def test_us_market_hours_off_hours():
    t = datetime(2026, 5, 5, 5, 0, tzinfo=timezone.utc)  # 1AM ET, closed
    assert intraday._is_us_market_hours(t) is False


def test_us_market_hours_weekend():
    t = datetime(2026, 5, 9, 18, 0, tzinfo=timezone.utc)  # Saturday
    assert intraday._is_us_market_hours(t) is False


def test_cn_market_hours_inside_window():
    t = datetime(2026, 5, 5, 3, 0, tzinfo=timezone.utc)  # ~11AM Shanghai
    assert intraday._is_cn_market_hours(t) is True


def test_cn_market_hours_weekend():
    t = datetime(2026, 5, 10, 3, 0, tzinfo=timezone.utc)  # Sunday
    assert intraday._is_cn_market_hours(t) is False


def test_active_markets_disjoint():
    """US and CN sessions don't overlap (US: 13-21:30 UTC, CN: 01-07:30 UTC)."""
    us_time = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)
    cn_time = datetime(2026, 5, 5, 3, 0, tzinfo=timezone.utc)
    assert intraday._active_markets(us_time) == {"US"}
    assert intraday._active_markets(cn_time) == {"CN"}


def test_active_markets_off_hours_returns_empty():
    t = datetime(2026, 5, 5, 11, 0, tzinfo=timezone.utc)  # gap between sessions
    assert intraday._active_markets(t) == set()


# ---- Move bucketing ----


def test_bucket_for_move_thresholds():
    assert intraday._bucket_for_move(0.5) == "0pct"
    assert intraday._bucket_for_move(5.5) == "5pct"
    assert intraday._bucket_for_move(7.0) == "7pct"
    assert intraday._bucket_for_move(-12.3) == "10pct"
    assert intraday._bucket_for_move(-15.0) == "15pct"
    # Symmetry: sign should not affect bucket
    assert intraday._bucket_for_move(8) == intraday._bucket_for_move(-8)


# ---- Alert detection ----


def _make_intraday(symbol="AMD", *, last=100, open_=100, prev_close=100,
                    currency="USD"):
    chg_open = (last / open_ - 1) * 100
    chg_prev = (last / prev_close - 1) * 100
    return {
        "symbol": symbol, "currency": currency,
        "open": open_, "last": last, "high": max(open_, last) + 1,
        "low": min(open_, last) - 1, "volume": 1_000_000,
        "prev_close": prev_close,
        "chg_from_open_pct": chg_open,
        "chg_from_prev_close_pct": chg_prev,
        "ts": "2026-05-05T18:00:00+00:00",
    }


def _portfolio(held=None, watch=None, threshold=0.05):
    return {
        "positions": {h: {"shares": 1, "currency": "USD", "name": h} for h in (held or [])},
        "watchlist": [{"symbol": w} for w in (watch or [])],
        "risk": {"intraday_move_alert_pct": threshold},
    }


def test_detect_alerts_intraday_move_triggers_above_threshold(monkeypatch):
    monkeypatch.setattr(intraday.fetcher, "load_local", lambda s: pd.DataFrame())
    intraday_data = {"AMD": _make_intraday("AMD", last=108, open_=100, prev_close=100)}
    alerts = intraday.detect_alerts(intraday_data, _portfolio(held=["AMD"]), {})
    move = [a for a in alerts if a["rule"] == "INTRADAY_MOVE"]
    assert len(move) == 1
    assert move[0]["bucket"] == "7pct"
    assert move[0]["is_held"] is True
    assert move[0]["value_pct"] == pytest.approx(8.0)


def test_detect_alerts_intraday_move_below_threshold_silent(monkeypatch):
    """3% move with 5% threshold — no alert."""
    monkeypatch.setattr(intraday.fetcher, "load_local", lambda s: pd.DataFrame())
    intraday_data = {"AMD": _make_intraday("AMD", last=103, open_=100, prev_close=100)}
    alerts = intraday.detect_alerts(intraday_data, _portfolio(held=["AMD"]), {})
    assert [a for a in alerts if a["rule"] == "INTRADAY_MOVE"] == []


def test_detect_alerts_break_ma200_only_on_crossing(monkeypatch):
    """Should fire only when prev_close was above MA200 and last is below."""
    # Construct 200 days of close=100 → ma200 = 100
    history = pd.DataFrame({
        "open": [100]*200, "high": [101]*200, "low": [99]*200,
        "close": [100]*200, "volume": [1_000_000]*200,
    }, index=pd.date_range("2025-01-01", periods=200, freq="B"))
    monkeypatch.setattr(intraday.fetcher, "load_local", lambda s: history)

    # last=99 < ma200 (100), prev_close=101 >= ma200 → crossing
    crossing = {"AMD": _make_intraday("AMD", last=99, open_=99, prev_close=101)}
    alerts = intraday.detect_alerts(crossing, _portfolio(held=["AMD"], threshold=0.99), {})
    assert any(a["rule"] == "BREAK_MA200" for a in alerts)

    # already below previously: prev_close=98, last=97 — should NOT fire BREAK_MA200
    already_below = {"AMD": _make_intraday("AMD", last=97, open_=97, prev_close=98)}
    alerts = intraday.detect_alerts(already_below, _portfolio(held=["AMD"], threshold=0.99), {})
    assert not any(a["rule"] == "BREAK_MA200" for a in alerts)


def test_detect_alerts_watch_breakout(monkeypatch):
    """Watchlist symbol crosses above 20-day high → WATCH_BREAKOUT."""
    history = pd.DataFrame({
        "open": [100]*20, "high": [105]*20, "low": [95]*20,
        "close": [100]*20, "volume": [1_000_000]*20,
    }, index=pd.date_range("2025-01-01", periods=20, freq="B"))
    monkeypatch.setattr(intraday.fetcher, "load_local", lambda s: history)

    breakout = {"NVDA": _make_intraday("NVDA", last=106, open_=100, prev_close=100)}
    alerts = intraday.detect_alerts(breakout, _portfolio(watch=["NVDA"], threshold=0.99), {})
    bo = [a for a in alerts if a["rule"] == "WATCH_BREAKOUT"]
    assert len(bo) == 1
    assert bo[0]["high20"] == 105


def test_detect_alerts_watch_below_high_no_breakout(monkeypatch):
    history = pd.DataFrame({
        "open": [100]*20, "high": [105]*20, "low": [95]*20,
        "close": [100]*20, "volume": [1_000_000]*20,
    }, index=pd.date_range("2025-01-01", periods=20, freq="B"))
    monkeypatch.setattr(intraday.fetcher, "load_local", lambda s: history)

    intraday_data = {"NVDA": _make_intraday("NVDA", last=104, open_=100, prev_close=100)}
    alerts = intraday.detect_alerts(intraday_data, _portfolio(watch=["NVDA"], threshold=0.99), {})
    assert not any(a["rule"] == "WATCH_BREAKOUT" for a in alerts)


# ---- Rendering ----


def test_render_empty_alerts_returns_empty_string():
    assert intraday.render([]) == ""


def test_render_intraday_move_uses_correct_currency_symbol():
    alert = {
        "symbol": "002624.SZ", "rule": "INTRADAY_MOVE",
        "bucket": "5pct", "value_pct": 6.5, "is_held": True,
        "data": {"currency": "CNY", "last": 16.2, "open": 15.2, "prev_close": 15.2},
    }
    out = intraday.render([alert])
    assert "¥" in out
    assert "$" not in out
    assert "002624.SZ" in out
    assert "+6.5%" in out


def test_render_break_ma200_includes_levels():
    alert = {
        "symbol": "AMD", "rule": "BREAK_MA200", "bucket": "below",
        "ma200": 150.5, "is_held": True,
        "data": {"currency": "USD", "last": 148.2, "open": 149,
                 "prev_close": 151, "chg_from_open_pct": -0.5},
    }
    out = intraday.render([alert])
    assert "MA200" in out and "148.2" in out and "150.5" in out
