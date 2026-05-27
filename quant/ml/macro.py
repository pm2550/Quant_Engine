"""Macro features for the ML challenger — VIX / yields / DXY / commodities.

All series are observed daily so there's no point-in-time alignment problem.
We broadcast the same macro values across all symbols for each date.

Source: yfinance index tickers (no API key needed). Cached as parquet under
/data2/quant/data/macro/ so we don't re-download every run.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)
CACHE = Path("/data2/quant/data/macro")
CACHE.mkdir(parents=True, exist_ok=True)

# yfinance ticker → cache file name
TICKERS = {
    "^VIX":     "vix.parquet",       # CBOE volatility
    "^TNX":     "y10.parquet",       # 10-year Treasury yield
    "^IRX":     "y3m.parquet",       # 13-week T-bill (3M proxy)
    "^FVX":     "y5.parquet",        # 5-year Treasury
    "DX-Y.NYB": "dxy.parquet",       # US Dollar Index
    "GC=F":     "gold.parquet",      # Gold futures
    "CL=F":     "oil.parquet",       # Crude oil futures
}

LOOKBACK_YEARS = 20


def refresh_all(force: bool = False) -> dict[str, int]:
    """Download (or incrementally update) all macro tickers; return per-ticker row counts."""
    import yfinance as yf
    out = {}
    start = date.today() - timedelta(days=365 * LOOKBACK_YEARS)
    for ticker, fname in TICKERS.items():
        p = CACHE / fname
        if p.exists() and not force:
            existing = pd.read_parquet(p)
            if not existing.empty:
                last = pd.to_datetime(existing.index.max()).date()
                fetch_start = last - timedelta(days=5)
            else:
                fetch_start = start
        else:
            existing = pd.DataFrame()
            fetch_start = start
        try:
            df = yf.Ticker(ticker).history(start=fetch_start.isoformat(), auto_adjust=True)
        except Exception as e:  # noqa: BLE001
            log.warning("macro fetch failed for %s: %s", ticker, e)
            out[ticker] = -1
            continue
        if df is None or df.empty:
            out[ticker] = 0
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[["Close"]].rename(columns={"Close": ticker})
        if not existing.empty:
            keep = existing[~existing.index.isin(df.index)]
            df = pd.concat([keep, df]).sort_index()
        df.to_parquet(p)
        out[ticker] = len(df)
    return out


def load_macro_features() -> pd.DataFrame:
    """Load all cached macro series, compute features, return single DataFrame indexed by date.

    Returned columns are broadcast features (same value applies to every symbol
    on a given date):
      VIX, VIX_PCT60, VIX_TREND
      Y10, Y10_CHG20, Y10_3M_CURVE
      DXY, DXY_CHG60
      GOLD_CHG60, OIL_CHG60
      VIX_X_Y10 (regime interaction)
    """
    series = {}
    for ticker, fname in TICKERS.items():
        p = CACHE / fname
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        series[ticker] = df[ticker]
    if not series:
        return pd.DataFrame()

    # Inner-join on index of VIX (the densest series, daily US market days)
    if "^VIX" not in series:
        log.warning("VIX not cached, returning empty macro features")
        return pd.DataFrame()
    idx = series["^VIX"].index
    aligned = pd.DataFrame({k: v.reindex(idx).ffill() for k, v in series.items()})

    vix = aligned.get("^VIX")
    y10 = aligned.get("^TNX")
    y3m = aligned.get("^IRX")
    dxy = aligned.get("DX-Y.NYB")
    gold = aligned.get("GC=F")
    oil = aligned.get("CL=F")

    # IMPORTANT: only include rate-of-change / regime features, NOT raw levels.
    # Raw VIX/Y10/DXY levels let LightGBM memorize calendar periods (every train
    # year has a roughly characteristic VIX range), which destroys cross-sectional
    # generalization. Stick to ratios and rolling-pct changes that are dimensionless
    # and roughly stationary across decades.
    feat = pd.DataFrame(index=aligned.index)
    if vix is not None:
        feat["VIX_PCT60"] = vix.rolling(60, min_periods=20).rank(pct=True)
        feat["VIX_TREND"] = vix.rolling(20).mean() / vix.rolling(60).mean()
    if y10 is not None:
        feat["Y10_CHG20_BPS"] = (y10 - y10.shift(20)) * 100  # bps move in 20d
        if y3m is not None:
            # Curve as PCT60 rank: 0=most inverted in last 60d, 1=steepest
            curve = y10 - y3m
            feat["CURVE_PCT60"] = curve.rolling(60, min_periods=20).rank(pct=True)
    if dxy is not None:
        feat["DXY_CHG60"] = dxy.pct_change(60)
    if gold is not None:
        feat["GOLD_CHG60"] = gold.pct_change(60)
    if oil is not None:
        feat["OIL_CHG60"] = oil.pct_change(60)

    return feat.replace([np.inf, -np.inf], np.nan)


if __name__ == "__main__":
    print("refreshing macro tickers...")
    counts = refresh_all()
    for k, v in counts.items():
        print(f"  {k:12} {v} bars")
    print()
    feats = load_macro_features()
    print(f"macro features: {len(feats.columns)} cols × {len(feats)} dates")
    print(f"range: {feats.index.min().date()} → {feats.index.max().date()}")
    print(feats.tail(3))
