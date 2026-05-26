"""Unit tests for recommender.py — deterministic action mapping."""
from __future__ import annotations
import pytest

from quant import recommender, signals


def _make_sig(**kwargs):
    """Build SymbolSignals with overrides."""
    defaults = dict(
        symbol="X", last_date="2025-01-01", price=100.0,
        chg_1d_pct=0, chg_5d_pct=0, chg_20d_pct=0,
        rsi=50, rsi_zscore_252d=0,
        macd=0, macd_signal=0, macd_hist=0,
        bb_upper=110, bb_lower=90, bb_pct=0.5,
        ma20=100, ma50=100, ma200=100,
        above_ma50=True, above_ma200=True,
        atr_14=2.0, atr_pct=2.0,
        vol_20d_avg=1000, vol_today=1000, vol_ratio=1.0,
        signal_codes=[],
    )
    defaults.update(kwargs)
    return signals.SymbolSignals(**defaults)


def test_held_neutral_returns_hold():
    sig = _make_sig(signal_codes=[])
    rec = recommender.for_held(sig, current_weight=0.10, total_value=1000)
    assert rec.action == "HOLD"


def test_held_oversold_returns_add():
    sig = _make_sig(signal_codes=["RSI_EXTREME_OVERSOLD", "MACD_GOLDEN_CROSS_ABOVE_ZERO"])
    rec = recommender.for_held(sig, current_weight=0.10, total_value=1000)
    assert rec.action == "ADD"
    assert rec.target_weight > rec.current_weight


def test_held_overbought_with_macd_death_returns_reduce():
    sig = _make_sig(signal_codes=["RSI_EXTREME_OVERBOUGHT", "MACD_DEATH_CROSS_ABOVE_ZERO", "BB_BREAK_UPPER"])
    rec = recommender.for_held(sig, current_weight=0.20, total_value=1000)
    assert rec.action == "REDUCE"
    assert rec.target_weight < rec.current_weight


def test_held_below_ma200_triggers_stop_loss():
    sig = _make_sig(
        signal_codes=["CROSS_BELOW_MA200"],
        above_ma200=False,
    )
    rec = recommender.for_held(sig, current_weight=0.10, total_value=1000)
    assert rec.action == "STOP_LOSS"


def test_watch_skip_when_no_signals():
    sig = _make_sig(signal_codes=[])
    rec = recommender.for_watch(sig)
    assert rec.action == "WATCH_SKIP"


def test_watch_buy_on_strong_signal():
    sig = _make_sig(signal_codes=[
        "MACD_GOLDEN_CROSS_ABOVE_ZERO",
        "CROSS_ABOVE_MA200",
        "RSI_OVERSOLD",
    ])
    rec = recommender.for_watch(sig)
    assert rec.action == "WATCH_BUY"
    assert rec.confidence > 0.5


def test_below_ma50_filter_halves_buy_score():
    """Below-MA50 filter should weaken bullish MACD signal — turns ADD into HOLD."""
    sig_above = _make_sig(
        signal_codes=["MACD_GOLDEN_CROSS_ABOVE_ZERO"],
        above_ma50=True,
    )
    sig_below = _make_sig(
        signal_codes=["MACD_GOLDEN_CROSS_ABOVE_ZERO"],
        above_ma50=False,
    )
    rec_above = recommender.for_held(sig_above, current_weight=0.1, total_value=1000)
    rec_below = recommender.for_held(sig_below, current_weight=0.1, total_value=1000)
    # Above MA50 → bullish action; below MA50 → no action upgrade
    assert rec_above.action == "ADD"
    assert rec_below.action == "HOLD"


def _multi(score, action="HOLD", factors=None):
    return {
        "composite_score": score,
        "action": action,
        "rationale": "test",
        "catalyst_imminent": False,
        "factor_breakdown": {
            "technical": {"score": score, "factors": factors or [], "weight": 1.0}
        },
    }


def test_multi_factor_neutral_ignores_single_macd_death_cross():
    sig = _make_sig(signal_codes=["MACD_DEATH_CROSS_ABOVE_ZERO"])
    rec = recommender.for_held_multi_factor(
        sig, _multi(0.0), current_weight=0.20, total_value=1000
    )
    assert rec.action == "HOLD"
    assert rec.target_weight == rec.current_weight


def test_multi_factor_neutral_ignores_single_breakout_buy_signal():
    sig = _make_sig(signal_codes=["CROSS_ABOVE_MA200", "VOLUME_SPIKE_2X"], chg_1d_pct=5)
    rec = recommender.for_held_multi_factor(
        sig, _multi(0.0), current_weight=0.10, total_value=1000
    )
    assert rec.action == "HOLD"


def test_multi_factor_ma200_break_without_confirmation_is_hold():
    sig = _make_sig(signal_codes=["CROSS_BELOW_MA200"], above_ma200=False)
    rec = recommender.for_held_multi_factor(
        sig, _multi(-0.10), current_weight=0.10, total_value=1000
    )
    assert rec.action == "HOLD"


def test_multi_factor_thresholds_drive_add_reduce_and_stop_loss():
    sig = _make_sig()
    add = recommender.for_held_multi_factor(
        sig, _multi(0.40), current_weight=0.10, total_value=1000
    )
    reduce = recommender.for_held_multi_factor(
        sig, _multi(-0.40), current_weight=0.10, total_value=1000
    )
    broken = _make_sig(signal_codes=["CROSS_BELOW_MA200"], above_ma200=False)
    stop = recommender.for_held_multi_factor(
        broken, _multi(-0.40), current_weight=0.10, total_value=1000
    )
    assert add.action == "ADD"
    assert reduce.action == "REDUCE"
    assert stop.action == "STOP_LOSS"


def test_to_dict_serializable():
    sig = _make_sig(signal_codes=["RSI_OVERSOLD"])
    rec = recommender.for_held(sig, current_weight=0.1, total_value=1000)
    d = recommender.to_dict(rec)
    assert "action" in d and "confidence" in d
    # Should be JSON-serializable
    import json
    json.dumps(d)
