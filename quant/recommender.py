"""Translate raw signals + portfolio context into concrete action recommendations.

This is the deterministic decision layer (no LLM). The LLM only reformats
into human-readable Chinese in the next step.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

from .signals import SymbolSignals

# Aligned with multi_factor.score() thresholds (lowered 2026-05-26 from 0.40 → 0.30).
# Rationale: 11 factors are seldom all aligned at >=0.4; 0.30 still requires多数因子同向.
ADD_THRESHOLD = 0.30
REDUCE_THRESHOLD = -0.30
MAX_TARGET_WEIGHT = 0.30
DEFAULT_WATCH_TARGET_WEIGHT = 0.05


@dataclass
class Recommendation:
    symbol: str
    action: str               # BUY | ADD | REDUCE | SELL | HOLD | WATCH_BUY | WATCH_SKIP | STOP_LOSS | TAKE_PROFIT
    current_weight: float     # 0..1, only for held positions
    target_weight: float      # 0..1
    reason_codes: list[str]   # which signals triggered this
    confidence: float         # 0..1
    notes: dict[str, Any]


def _multi_notes(multi: dict[str, Any]) -> dict[str, Any]:
    """Flatten multi_factor.score() output for downstream (LLM packager) consumption.

    Passes through *structured* top_factors / counter_factors (list of dicts with
    name/score/contribution/evidence) so the report can render conviction-based
    evidence rather than re-scanning factor_breakdown.
    """
    breakdown = multi.get("factor_breakdown") or {}
    factor_scores = {
        k: v.get("score")
        for k, v in breakdown.items()
        if isinstance(v, dict) and v.get("score") is not None
    }

    # Backwards-compat flat list of strings, derived from structured top_factors.
    top_evidence: list[str] = []
    for tf in multi.get("top_factors") or []:
        ev = (tf or {}).get("evidence")
        if ev and ev not in top_evidence:
            top_evidence.append(str(ev))
        if len(top_evidence) >= 5:
            break

    return {
        "composite_score": multi.get("composite_score"),
        "conviction": multi.get("conviction", 0),
        "top_factors": multi.get("top_factors") or [],          # structured list[dict]
        "counter_factors": multi.get("counter_factors") or [],  # structured list[dict]
        "top_factor_evidence": top_evidence,                    # flat string list for legacy callers
        "factor_scores": factor_scores,
        "catalyst_imminent": bool(multi.get("catalyst_imminent", False)),
        "multi_factor_action": multi.get("action"),
        "multi_factor_rationale": multi.get("rationale"),
    }


def _confidence_from_composite(composite: float, *, floor: float = 0.45) -> float:
    return max(floor, min(1.0, abs(composite)))


def for_held_multi_factor(
    sig: SymbolSignals,
    multi: dict[str, Any],
    *,
    current_weight: float,
    total_value: float,
    position_cost: float | None = None,
) -> Recommendation:
    """Map multi-factor score to a deterministic held-position recommendation.

    Price action and technical codes are only context inside multi_factor.score.
    This layer deliberately defaults to HOLD unless composite evidence is strong.
    """
    try:
        composite = float(multi.get("composite_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        composite = 0.0
    notes = _multi_notes(multi)
    notes["thresholds"] = {"add": ADD_THRESHOLD, "reduce": REDUCE_THRESHOLD}

    if "CROSS_BELOW_MA200" in sig.signal_codes and composite <= REDUCE_THRESHOLD:
        delta = min(0.05, 0.02 + abs(composite) * 0.05)
        return Recommendation(
            symbol=sig.symbol, action="STOP_LOSS",
            current_weight=current_weight,
            target_weight=max(current_weight - delta, 0.0),
            reason_codes=["RISK_CONFIRMED_BREAKDOWN", "CROSS_BELOW_MA200"],
            confidence=_confidence_from_composite(composite, floor=0.65),
            notes=notes | {"price": sig.price, "ma200": sig.ma200},
        )

    if composite >= ADD_THRESHOLD:
        delta = min(0.05, 0.02 + composite * 0.05)
        return Recommendation(
            symbol=sig.symbol, action="ADD",
            current_weight=current_weight,
            target_weight=min(current_weight + delta, MAX_TARGET_WEIGHT),
            reason_codes=["MULTI_FACTOR_BULLISH"],
            confidence=_confidence_from_composite(composite),
            notes=notes,
        )

    if composite <= REDUCE_THRESHOLD:
        delta = min(0.05, 0.02 + abs(composite) * 0.05)
        return Recommendation(
            symbol=sig.symbol, action="REDUCE",
            current_weight=current_weight,
            target_weight=max(current_weight - delta, 0.0),
            reason_codes=["MULTI_FACTOR_BEARISH"],
            confidence=_confidence_from_composite(composite),
            notes=notes,
        )

    return Recommendation(
        symbol=sig.symbol, action="HOLD",
        current_weight=current_weight, target_weight=current_weight,
        reason_codes=[], confidence=_confidence_from_composite(composite),
        notes=notes,
    )


def for_watch_multi_factor(sig: SymbolSignals, multi: dict[str, Any]) -> Recommendation:
    try:
        composite = float(multi.get("composite_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        composite = 0.0
    notes = _multi_notes(multi)
    notes["thresholds"] = {"watch_buy": ADD_THRESHOLD}

    if composite >= ADD_THRESHOLD:
        return Recommendation(
            symbol=sig.symbol, action="WATCH_BUY",
            current_weight=0.0, target_weight=DEFAULT_WATCH_TARGET_WEIGHT,
            reason_codes=["MULTI_FACTOR_BULLISH"],
            confidence=_confidence_from_composite(composite),
            notes=notes,
        )
    return Recommendation(
        symbol=sig.symbol, action="WATCH_SKIP",
        current_weight=0.0, target_weight=0.0,
        reason_codes=[], confidence=_confidence_from_composite(composite, floor=0.3),
        notes=notes,
    )


def _score_buy(sig: SymbolSignals) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    codes = set(sig.signal_codes)

    if "MACD_GOLDEN_CROSS_ABOVE_ZERO" in codes:
        score += 0.4; reasons.append("MACD_GOLDEN_CROSS_ABOVE_ZERO")
    elif "MACD_GOLDEN_CROSS_BELOW_ZERO" in codes:
        score += 0.2; reasons.append("MACD_GOLDEN_CROSS_BELOW_ZERO")
    if "CROSS_ABOVE_MA200" in codes:
        score += 0.3; reasons.append("CROSS_ABOVE_MA200")
    if "RSI_EXTREME_OVERSOLD" in codes:
        score += 0.3; reasons.append("RSI_EXTREME_OVERSOLD")
    elif "RSI_OVERSOLD" in codes:
        score += 0.15; reasons.append("RSI_OVERSOLD")
    if "BB_BREAK_LOWER" in codes:
        score += 0.15; reasons.append("BB_BREAK_LOWER")
    if "VOLUME_SPIKE_2X" in codes and sig.chg_1d_pct > 0:
        score += 0.1; reasons.append("VOLUME_SPIKE_2X_UP")

    if not sig.above_ma50:
        score *= 0.5; reasons.append("FILTER_BELOW_MA50_HALVED")
    return score, reasons


def _score_reduce(sig: SymbolSignals) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    codes = set(sig.signal_codes)

    if "RSI_EXTREME_OVERBOUGHT" in codes:
        score += 0.4; reasons.append("RSI_EXTREME_OVERBOUGHT")
    elif "RSI_OVERBOUGHT" in codes:
        score += 0.2; reasons.append("RSI_OVERBOUGHT")
    if "BB_BREAK_UPPER" in codes:
        score += 0.3; reasons.append("BB_BREAK_UPPER")
    if "MACD_DEATH_CROSS_ABOVE_ZERO" in codes:
        score += 0.4; reasons.append("MACD_DEATH_CROSS_ABOVE_ZERO")
    elif "MACD_DEATH_CROSS_BELOW_ZERO" in codes:
        score += 0.2; reasons.append("MACD_DEATH_CROSS_BELOW_ZERO")
    if "CROSS_BELOW_MA200" in codes:
        score += 0.5; reasons.append("CROSS_BELOW_MA200")
    if "VOLUME_SPIKE_2X" in codes and sig.chg_1d_pct < 0:
        score += 0.1; reasons.append("VOLUME_SPIKE_2X_DOWN")
    return score, reasons


def for_held(sig: SymbolSignals, *, current_weight: float, total_value: float, position_cost: float | None = None) -> Recommendation:
    buy, buy_r = _score_buy(sig)
    red, red_r = _score_reduce(sig)

    # Hard rules first (止损 / 止盈)
    if "CROSS_BELOW_MA200" in sig.signal_codes:
        return Recommendation(
            symbol=sig.symbol, action="STOP_LOSS",
            current_weight=current_weight, target_weight=max(current_weight - 0.05, 0.0),
            reason_codes=["CROSS_BELOW_MA200"], confidence=0.85,
            notes={"price": sig.price, "ma200": sig.ma200},
        )

    if buy > red and buy >= 0.4:
        delta = min(0.05, 0.02 + buy * 0.05)
        return Recommendation(
            symbol=sig.symbol, action="ADD",
            current_weight=current_weight,
            target_weight=min(current_weight + delta, 0.30),
            reason_codes=buy_r, confidence=min(buy, 1.0),
            notes={"score": round(buy, 2)},
        )

    if red > buy and red >= 0.4:
        delta = min(0.05, 0.02 + red * 0.05)
        return Recommendation(
            symbol=sig.symbol, action="REDUCE",
            current_weight=current_weight,
            target_weight=max(current_weight - delta, 0.0),
            reason_codes=red_r, confidence=min(red, 1.0),
            notes={"score": round(red, 2)},
        )

    return Recommendation(
        symbol=sig.symbol, action="HOLD",
        current_weight=current_weight, target_weight=current_weight,
        reason_codes=[], confidence=0.5,
        notes={"buy_score": round(buy, 2), "reduce_score": round(red, 2)},
    )


def for_watch(sig: SymbolSignals) -> Recommendation:
    buy, buy_r = _score_buy(sig)
    if buy >= 0.5:
        return Recommendation(
            symbol=sig.symbol, action="WATCH_BUY",
            current_weight=0.0, target_weight=0.05,
            reason_codes=buy_r, confidence=min(buy, 1.0),
            notes={"score": round(buy, 2)},
        )
    return Recommendation(
        symbol=sig.symbol, action="WATCH_SKIP",
        current_weight=0.0, target_weight=0.0,
        reason_codes=[], confidence=0.3,
        notes={"buy_score": round(buy, 2)},
    )


def to_dict(r: Recommendation) -> dict:
    return asdict(r)
