"""Compute per-symbol indicators and derive raw signal codes."""
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

log = logging.getLogger(__name__)


@dataclass
class SymbolSignals:
    symbol: str
    last_date: str
    price: float
    chg_1d_pct: float
    chg_5d_pct: float
    chg_20d_pct: float
    rsi: float
    rsi_zscore_252d: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_lower: float
    bb_pct: float           # position within bands, 0..1 above 1 = above upper
    ma20: float
    ma50: float
    ma200: float
    above_ma50: bool
    above_ma200: bool
    atr_14: float
    atr_pct: float          # ATR as % of price
    vol_20d_avg: float
    vol_today: float
    vol_ratio: float        # today / 20d avg
    signal_codes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _last(s: pd.Series, default=float("nan")) -> float:
    if s.empty:
        return default
    v = s.iloc[-1]
    return float(v) if pd.notna(v) else default


def _bool(b) -> bool:
    return bool(b) if pd.notna(b) else False


def compute(symbol: str, df: pd.DataFrame, cfg: dict) -> SymbolSignals | None:
    if df is None or df.empty:
        log.warning("no data for %s", symbol)
        return None

    df = df.sort_index().copy()
    n = len(df)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0, index=df.index)

    ind = cfg["indicators"]

    # Indicators degrade gracefully when history is short
    rsi = RSIIndicator(close, window=14).rsi() if n >= 15 else pd.Series([float("nan")] * n, index=close.index)
    macd = MACD(close) if n >= 35 else None
    bb = BollingerBands(close, window=ind["bollinger"]["period"], window_dev=ind["bollinger"]["std"]) if n >= ind["bollinger"]["period"] else None
    atr = AverageTrueRange(high, low, close, window=ind["atr"]["period"]).average_true_range() if n >= ind["atr"]["period"] else pd.Series([float("nan")] * n, index=close.index)
    ma20 = SMAIndicator(close, window=ind["ma"]["short"]).sma_indicator() if n >= ind["ma"]["short"] else pd.Series([float("nan")] * n, index=close.index)
    ma50 = SMAIndicator(close, window=ind["ma"]["medium"]).sma_indicator() if n >= ind["ma"]["medium"] else pd.Series([float("nan")] * n, index=close.index)
    ma200 = SMAIndicator(close, window=ind["ma"]["long"]).sma_indicator() if n >= ind["ma"]["long"] else pd.Series([float("nan")] * n, index=close.index)

    if bb is not None:
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower)
    else:
        nan_s = pd.Series([float("nan")] * n, index=close.index)
        bb_upper, bb_lower, bb_pct = nan_s, nan_s, nan_s

    rsi_window = rsi.tail(252).dropna()
    rsi_std = rsi_window.std() if len(rsi_window) > 30 else float("nan")
    rsi_z = ((rsi.iloc[-1] - rsi_window.mean()) / rsi_std
             if len(rsi_window) > 30 and pd.notna(rsi.iloc[-1]) and rsi_std and rsi_std > 0
             else float("nan"))

    chg_1d = close.pct_change(1).iloc[-1] * 100
    chg_5d = close.pct_change(5).iloc[-1] * 100
    chg_20d = close.pct_change(20).iloc[-1] * 100

    vol_avg = vol.tail(20).mean()
    vol_today = vol.iloc[-1]
    vol_ratio = vol_today / vol_avg if vol_avg else float("nan")

    last_close = _last(close)
    codes: list[str] = []

    if n < 50:
        codes.append("INSUFFICIENT_HISTORY")

    # MACD signals
    if macd is not None and len(macd.macd()) >= 2:
        m_now, m_prev = macd.macd().iloc[-1], macd.macd().iloc[-2]
        s_now, s_prev = macd.macd_signal().iloc[-1], macd.macd_signal().iloc[-2]
        if pd.notna(m_now) and pd.notna(s_now):
            if m_prev <= s_prev and m_now > s_now:
                codes.append("MACD_GOLDEN_CROSS_ABOVE_ZERO" if m_now > 0 else "MACD_GOLDEN_CROSS_BELOW_ZERO")
            elif m_prev >= s_prev and m_now < s_now:
                codes.append("MACD_DEATH_CROSS_ABOVE_ZERO" if m_now > 0 else "MACD_DEATH_CROSS_BELOW_ZERO")

    # RSI signals
    rsi_now = _last(rsi)
    if rsi_now >= ind["rsi"]["extreme_overbought"]:
        codes.append("RSI_EXTREME_OVERBOUGHT")
    elif rsi_now >= ind["rsi"]["overbought"]:
        codes.append("RSI_OVERBOUGHT")
    elif rsi_now <= ind["rsi"]["extreme_oversold"]:
        codes.append("RSI_EXTREME_OVERSOLD")
    elif rsi_now <= ind["rsi"]["oversold"]:
        codes.append("RSI_OVERSOLD")

    # Bollinger band breakouts
    bb_pct_now = _last(bb_pct)
    if bb_pct_now >= 1.0:
        codes.append("BB_BREAK_UPPER")
    elif bb_pct_now <= 0.0:
        codes.append("BB_BREAK_LOWER")

    # MA cross — only meaningful when MA values exist
    ma50_last, ma200_last = _last(ma50), _last(ma200)
    above50 = _bool(last_close > ma50_last) if pd.notna(ma50_last) else False
    above200 = _bool(last_close > ma200_last) if pd.notna(ma200_last) else False
    if pd.notna(ma200_last) and len(ma200.dropna()) >= 2:
        prev_ma200 = ma200.iloc[-2]
        if pd.notna(prev_ma200):
            prev_above = close.iloc[-2] > prev_ma200
            if not above200 and prev_above:
                codes.append("CROSS_BELOW_MA200")
            elif above200 and not prev_above:
                codes.append("CROSS_ABOVE_MA200")

    # Volume spike
    if pd.notna(vol_ratio) and vol_ratio >= 2:
        codes.append("VOLUME_SPIKE_2X")

    nan = float("nan")
    return SymbolSignals(
        symbol=symbol,
        last_date=str(df.index[-1].date()),
        price=last_close,
        chg_1d_pct=float(chg_1d) if pd.notna(chg_1d) else 0.0,
        chg_5d_pct=float(chg_5d) if pd.notna(chg_5d) else 0.0,
        chg_20d_pct=float(chg_20d) if pd.notna(chg_20d) else 0.0,
        rsi=float(rsi_now) if pd.notna(rsi_now) else nan,
        rsi_zscore_252d=float(rsi_z) if pd.notna(rsi_z) else nan,
        macd=_last(macd.macd()) if macd is not None else nan,
        macd_signal=_last(macd.macd_signal()) if macd is not None else nan,
        macd_hist=_last(macd.macd_diff()) if macd is not None else nan,
        bb_upper=_last(bb_upper),
        bb_lower=_last(bb_lower),
        bb_pct=float(bb_pct_now) if pd.notna(bb_pct_now) else nan,
        ma20=_last(ma20),
        ma50=_last(ma50),
        ma200=_last(ma200),
        above_ma50=above50,
        above_ma200=above200,
        atr_14=_last(atr),
        atr_pct=float(_last(atr) / last_close * 100) if last_close else 0.0,
        vol_20d_avg=float(vol_avg) if pd.notna(vol_avg) else 0.0,
        vol_today=float(vol_today) if pd.notna(vol_today) else 0.0,
        vol_ratio=float(vol_ratio) if pd.notna(vol_ratio) else 0.0,
        signal_codes=codes,
    )
