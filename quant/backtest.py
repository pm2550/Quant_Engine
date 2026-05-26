"""Backtest strategy implementations using vectorbt.

Cost model: commission + slippage applied symmetrically per side via
vectorbt's `fees` and `slippage` kwargs.  Defaults (commission 5 bps,
slippage 5 bps total = 10 bps round trip) match IBKR fixed-tier on
liquid US/HK stocks for retail account sizes.

Walk-forward: `walk_forward()` splits the period into K non-overlapping
out-of-sample folds and reports per-fold sharpe + median/std across folds.
A strategy that looks strong in aggregate but fails on multiple folds is
overfit to a regime, not robust.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import vectorbt as vbt

from . import fetcher

log = logging.getLogger(__name__)


@dataclass
class CostModel:
    """Symmetric per-side cost. Round-trip = 2 * (commission_pct + slippage_pct)."""
    commission_pct: float = 0.0005   # 5 bps — IBKR fixed tier on liquid stocks
    slippage_pct: float = 0.0005     # 5 bps — bid/ask + market impact for retail size

    @property
    def fees(self) -> float:
        return self.commission_pct

    @property
    def slippage(self) -> float:
        return self.slippage_pct


DEFAULT_COSTS = CostModel()


def _slice(df: pd.DataFrame, period_years: int, *, as_of: str | None = None) -> pd.DataFrame:
    """Slice the trailing `period_years` window. If `as_of` (YYYY-MM-DD) given, end at that date."""
    if as_of:
        end = pd.Timestamp(as_of)
        df = df.loc[df.index <= end]
    end = df.index.max() if len(df) else None
    if end is None:
        return df
    start = end - pd.DateOffset(years=period_years)
    return df.loc[df.index >= start]


def _split_params(params: dict) -> tuple[dict, str | None]:
    """Pop optional 'as_of' from params, return (clean_params, as_of)."""
    if "as_of" in params:
        p = dict(params)
        return p, p.pop("as_of")
    return params, None


def _metrics(pf: vbt.Portfolio) -> dict:
    stats = pf.stats()
    sharpe = float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0.0
    sortino = float(pf.sortino_ratio()) if not np.isnan(pf.sortino_ratio()) else 0.0
    total_ret = float(pf.total_return())
    n_years = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days / 365.25
    annual = float(((1 + total_ret) ** (1 / max(n_years, 0.001))) - 1) if total_ret > -1 else -1.0
    mdd = float(pf.max_drawdown())
    trades = pf.trades
    n_trades = int(trades.count())
    win_rate = float(trades.win_rate()) if n_trades else 0.0
    pf_ratio = float(trades.profit_factor()) if n_trades and not np.isnan(trades.profit_factor()) else 0.0
    return {
        "total_return": total_ret,
        "annual_return": annual,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "profit_factor": pf_ratio,
    }


def _load(symbol: str) -> pd.DataFrame | None:
    df = fetcher.load_local(symbol)
    return df if not df.empty else None


# ---- Strategy: dual moving average ----
def dual_ma(symbol: str, params: dict, period_years: int = 5) -> dict:
    df = _load(symbol)
    if df is None or len(df) < 250:
        raise ValueError(f"insufficient history for {symbol}")
    params, as_of = _split_params(params)
    df = _slice(df, period_years, as_of=as_of)
    close = df["close"].astype(float)

    short = int(params["short"])
    long = int(params["long"])
    if short >= long:
        raise ValueError("short must be < long")

    fast = close.rolling(short).mean()
    slow = close.rolling(long).mean()
    entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10_000,
                                     fees=DEFAULT_COSTS.fees, slippage=DEFAULT_COSTS.slippage, freq="1D")
    return _metrics(pf)


# ---- Strategy: RSI mean reversion ----
def rsi_meanrev(symbol: str, params: dict, period_years: int = 5) -> dict:
    df = _load(symbol)
    if df is None or len(df) < 100:
        raise ValueError(f"insufficient history for {symbol}")
    params, as_of = _split_params(params)
    df = _slice(df, period_years, as_of=as_of)
    close = df["close"].astype(float)

    period = int(params["period"])
    oversold = float(params["oversold"])
    overbought = float(params["overbought"])

    rsi = vbt.RSI.run(close, window=period).rsi
    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10_000,
                                     fees=DEFAULT_COSTS.fees, slippage=DEFAULT_COSTS.slippage, freq="1D")
    return _metrics(pf)


# ---- Strategy: Bollinger band breakout ----
def bb_breakout(symbol: str, params: dict, period_years: int = 5) -> dict:
    df = _load(symbol)
    if df is None or len(df) < 100:
        raise ValueError(f"insufficient history for {symbol}")
    params, as_of = _split_params(params)
    df = _slice(df, period_years, as_of=as_of)
    close = df["close"].astype(float)

    period = int(params["period"])
    std = float(params["std"])

    bb = vbt.BBANDS.run(close, window=period, alpha=std)
    entries = close > bb.upper
    exits = close < bb.middle  # exit when reverting to middle

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10_000,
                                     fees=DEFAULT_COSTS.fees, slippage=DEFAULT_COSTS.slippage, freq="1D")
    return _metrics(pf)


# ---- Strategy: MACD crossover ----
def macd_cross(symbol: str, params: dict, period_years: int = 5) -> dict:
    df = _load(symbol)
    if df is None or len(df) < 100:
        raise ValueError(f"insufficient history for {symbol}")
    params, as_of = _split_params(params)
    df = _slice(df, period_years, as_of=as_of)
    close = df["close"].astype(float)

    fast = int(params["fast"])
    slow = int(params["slow"])
    sig = int(params["signal"])
    if fast >= slow:
        raise ValueError("fast must be < slow")

    macd = vbt.MACD.run(close, fast_window=fast, slow_window=slow, signal_window=sig)
    line = macd.macd
    signal = macd.signal
    entries = (line > signal) & (line.shift(1) <= signal.shift(1))
    exits = (line < signal) & (line.shift(1) >= signal.shift(1))

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10_000,
                                     fees=DEFAULT_COSTS.fees, slippage=DEFAULT_COSTS.slippage, freq="1D")
    return _metrics(pf)


REGISTRY = {
    "dual_ma": dual_ma,
    "rsi_meanrev": rsi_meanrev,
    "bb_breakout": bb_breakout,
    "macd_cross": macd_cross,
}


def run(strategy: str, symbol: str, params: dict, *, period_years: int = 5) -> dict:
    fn = REGISTRY.get(strategy)
    if not fn:
        raise ValueError(f"unknown strategy: {strategy}")
    return fn(symbol, params, period_years=period_years)


# ---- Walk-forward validation ----
def walk_forward(strategy: str, symbol: str, params: dict, *,
                  period_years: int = 5, n_folds: int = 4,
                  min_fold_days: int = 90) -> dict:
    """Split history into n_folds non-overlapping out-of-sample windows.

    Reports median/std/min sharpe across folds and the consistency rate
    (folds with sharpe > 0).  A strategy that's robust survives most folds;
    one overfit to a regime collapses on at least one fold.

    Returns: {
        n_folds, folds: [{start, end, days, ...metrics}, ...],
        sharpe_median, sharpe_std, sharpe_min, sharpe_max,
        consistency_rate (fraction of folds with sharpe>0),
    }
    """
    if strategy not in REGISTRY:
        raise ValueError(f"unknown strategy: {strategy}")
    df = _load(symbol)
    if df is None or len(df) < min_fold_days * n_folds:
        raise ValueError(
            f"insufficient history for {symbol}: need >= {min_fold_days * n_folds} rows, got {0 if df is None else len(df)}"
        )
    params, as_of = _split_params(params)
    df = _slice(df, period_years, as_of=as_of)
    n = len(df)
    if n < min_fold_days * n_folds:
        raise ValueError(f"sliced window has only {n} rows, need >= {min_fold_days * n_folds}")
    fold_size = n // n_folds

    folds = []
    sharpes = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else n
        fold_df = df.iloc[start_idx:end_idx]
        if len(fold_df) < min_fold_days:
            continue
        fold_metrics = _run_on_df(strategy, fold_df, params)
        fold_metrics["fold_index"] = i
        fold_metrics["start"] = str(fold_df.index[0].date())
        fold_metrics["end"] = str(fold_df.index[-1].date())
        fold_metrics["days"] = len(fold_df)
        folds.append(fold_metrics)
        sharpes.append(fold_metrics["sharpe"])

    if not sharpes:
        return {"n_folds": 0, "folds": [], "error": "no folds met min_fold_days"}

    arr = np.array(sharpes, dtype=float)
    return {
        "n_folds": len(sharpes),
        "folds": folds,
        "sharpe_median": float(np.median(arr)),
        "sharpe_std": float(np.std(arr)),
        "sharpe_min": float(np.min(arr)),
        "sharpe_max": float(np.max(arr)),
        "consistency_rate": float(np.mean(arr > 0)),
    }


def _run_on_df(strategy: str, df: pd.DataFrame, params: dict) -> dict:
    """Backtest a strategy on a pre-sliced dataframe (no further period-slicing).

    Used by walk_forward; mirrors the logic in dual_ma/rsi_meanrev/etc but
    skips the _load/_slice steps so we can pass an arbitrary window.
    """
    close = df["close"].astype(float)
    if strategy == "dual_ma":
        short, long = int(params["short"]), int(params["long"])
        if short >= long:
            raise ValueError("short must be < long")
        fast = close.rolling(short).mean()
        slow = close.rolling(long).mean()
        entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    elif strategy == "rsi_meanrev":
        period = int(params["period"])
        oversold = float(params["oversold"])
        overbought = float(params["overbought"])
        rsi = vbt.RSI.run(close, window=period).rsi
        entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
        exits = (rsi > overbought) & (rsi.shift(1) <= overbought)
    elif strategy == "bb_breakout":
        period = int(params["period"])
        std = float(params["std"])
        bb = vbt.BBANDS.run(close, window=period, alpha=std)
        entries = close > bb.upper
        exits = close < bb.middle
    elif strategy == "macd_cross":
        fast_w, slow_w = int(params["fast"]), int(params["slow"])
        sig_w = int(params["signal"])
        if fast_w >= slow_w:
            raise ValueError("fast must be < slow")
        macd = vbt.MACD.run(close, fast_window=fast_w, slow_window=slow_w, signal_window=sig_w)
        entries = (macd.macd > macd.signal) & (macd.macd.shift(1) <= macd.signal.shift(1))
        exits = (macd.macd < macd.signal) & (macd.macd.shift(1) >= macd.signal.shift(1))
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10_000,
                                     fees=DEFAULT_COSTS.fees, slippage=DEFAULT_COSTS.slippage, freq="1D")
    return _metrics(pf)
