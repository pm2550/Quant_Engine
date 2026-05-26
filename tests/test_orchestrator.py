"""Tests for daily orchestrator recommendation plumbing."""
from __future__ import annotations

from quant import orchestrator
from quant.signals import SymbolSignals


def _sig(symbol: str, *, codes=None) -> SymbolSignals:
    return SymbolSignals(
        symbol=symbol, last_date="2026-05-15", price=100.0,
        chg_1d_pct=0.0, chg_5d_pct=0.0, chg_20d_pct=0.0,
        rsi=50.0, rsi_zscore_252d=0.0,
        macd=0.0, macd_signal=0.0, macd_hist=0.0,
        bb_upper=110.0, bb_lower=90.0, bb_pct=0.5,
        ma20=100.0, ma50=100.0, ma200=100.0,
        above_ma50=True, above_ma200=True,
        atr_14=2.0, atr_pct=2.0,
        vol_20d_avg=1000.0, vol_today=1000.0, vol_ratio=1.0,
        signal_codes=codes or [],
    )


def _neutral_multi(symbol, signals_dict, fundamentals_data=None, current_price=None):
    return {
        "composite_score": 0.0,
        "action": "HOLD",
        "rationale": "neutral test score",
        "catalyst_imminent": False,
        "factor_breakdown": {
            "technical": {"score": 0.0, "factors": ["中性"], "weight": 1.0}
        },
    }


def test_orchestrator_skips_duplicate_watchlist_symbol(monkeypatch, tmp_path):
    portfolio = {
        "positions": {"AMD": {"shares": 1.0, "currency": "USD"}},
        "watchlist": [
            {"symbol": "AMD", "currency": "USD"},
            {"symbol": "NVDA", "currency": "USD"},
        ],
        "default_currency": "USD",
        "risk": {},
    }
    monkeypatch.setattr(
        orchestrator.cfg_mod,
        "load",
        lambda name: portfolio if name == "portfolio" else {"indicators": {}},
    )
    monkeypatch.setattr(orchestrator.cfg_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(orchestrator.fetcher, "fetch_all", lambda symbols, full_refresh=False: {s: object() for s in symbols})
    monkeypatch.setattr(orchestrator.fetcher, "is_a_share", lambda sym: False)
    monkeypatch.setattr(orchestrator.signals, "compute", lambda sym, df, cfg: _sig(sym))
    monkeypatch.setattr(orchestrator.fundamentals, "latest", lambda sym: {})
    monkeypatch.setattr(orchestrator.multi_factor, "score", _neutral_multi)
    monkeypatch.setattr(orchestrator, "_recent_audio_highlights", lambda: [])
    monkeypatch.setattr(orchestrator, "_upcoming_earnings_for_report", lambda: [])
    monkeypatch.setattr(orchestrator, "_important_dates_for_report", lambda: {"earnings": [], "corporate": [], "macro": []})

    out = orchestrator.run()

    amd_recs = [r for r in out["recommendations"] if r["symbol"] == "AMD"]
    assert len(amd_recs) == 1
    assert amd_recs[0]["action"] == "HOLD"
    assert out["multi_factor"]["AMD"]["composite_score"] == 0.0


def test_orchestrator_watch_skip_has_zero_trade_values(monkeypatch, tmp_path):
    portfolio = {
        "positions": {"AMD": {"shares": 1.0, "currency": "USD"}},
        "watchlist": [{"symbol": "NVDA", "currency": "USD"}],
        "default_currency": "USD",
        "risk": {},
    }
    monkeypatch.setattr(
        orchestrator.cfg_mod,
        "load",
        lambda name: portfolio if name == "portfolio" else {"indicators": {}},
    )
    monkeypatch.setattr(orchestrator.cfg_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(orchestrator.fetcher, "fetch_all", lambda symbols, full_refresh=False: {s: object() for s in symbols})
    monkeypatch.setattr(orchestrator.fetcher, "is_a_share", lambda sym: False)
    monkeypatch.setattr(orchestrator.signals, "compute", lambda sym, df, cfg: _sig(sym))
    monkeypatch.setattr(orchestrator.fundamentals, "latest", lambda sym: {})
    monkeypatch.setattr(orchestrator.multi_factor, "score", _neutral_multi)
    monkeypatch.setattr(orchestrator, "_recent_audio_highlights", lambda: [])
    monkeypatch.setattr(orchestrator, "_upcoming_earnings_for_report", lambda: [])
    monkeypatch.setattr(orchestrator, "_important_dates_for_report", lambda: {"earnings": [], "corporate": [], "macro": []})

    out = orchestrator.run()

    nvda = next(r for r in out["recommendations"] if r["symbol"] == "NVDA")
    assert nvda["action"] == "WATCH_SKIP"
    assert nvda["current_value"] == 0.0
    assert nvda["target_value"] == 0.0
    assert nvda["delta_value"] == 0.0
    assert nvda["delta_shares"] == 0.0
