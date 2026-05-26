"""Fetch OHLCV history into Parquet files; one file per symbol.

Routes by symbol suffix:
  - `.SS` (Shanghai) / `.SZ` (Shenzhen) → akshare (CN A-shares)
  - everything else → yfinance (US / HK with `.HK` works too)
"""
from __future__ import annotations
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import PRICES_DIR

log = logging.getLogger(__name__)

DEFAULT_LOOKBACK_YEARS = 20


def _path(symbol: str) -> Path:
    return PRICES_DIR / f"{symbol}.parquet"


def is_a_share(symbol: str) -> bool:
    return symbol.upper().endswith((".SS", ".SZ"))


def _akshare_fetch_em(symbol: str, start: date):
    """East-money endpoint — fast + full columns but flaky on large windows."""
    import akshare as ak
    code = symbol.split(".")[0]
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=date.today().strftime("%Y%m%d"),
        adjust="qfq",
    )
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "turnover",
        "涨跌幅": "chg_pct", "换手率": "turnover_rate",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume", "turnover", "turnover_rate"] if c in df.columns]
    return df[keep].astype(float)


def _akshare_fetch_sina(symbol: str, start: date):
    """Sina endpoint — fallback when east-money is blocked / disconnects."""
    import akshare as ak
    code = symbol.split(".")[0]
    suffix = "sz" if symbol.upper().endswith(".SZ") else "sh"
    df = ak.stock_zh_a_daily(
        symbol=f"{suffix}{code}",
        adjust="qfq",
        start_date=start.strftime("%Y%m%d"),
        end_date=date.today().strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # Sina columns: date, open, high, low, close, volume, amount, outstanding_share, turnover
    # Map to east-money schema: amount → turnover (成交额); sina's "turnover" → turnover_rate (换手率)
    df = df.rename(columns={"amount": "turnover", "turnover": "turnover_rate"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume", "turnover", "turnover_rate"] if c in df.columns]
    return df[keep].astype(float)


def _akshare_fetch(symbol: str, start: date) -> pd.DataFrame:
    """Fetch A-share data via akshare. symbol like '002624.SZ' or '600519.SS'.

    Tries east-money first (more columns, faster); falls back to Sina if EM
    drops the connection or returns empty (a recurring issue with EM's free
    endpoint on multi-year windows).
    """
    try:
        df = _akshare_fetch_em(symbol, start)
        if not df.empty:
            return df
        log.warning("akshare EM returned empty for %s; trying Sina", symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("akshare EM failed for %s (%s); trying Sina", symbol, e)
    return _akshare_fetch_sina(symbol, start)


def _yfinance_fetch(symbol: str, start: date) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start.isoformat(),
        progress=False,
        auto_adjust=True,
        actions=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


def fetch_symbol(symbol: str, *, full_refresh: bool = False) -> pd.DataFrame:
    """Fetch OHLCV for one symbol, merging with existing local Parquet."""
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(symbol)

    existing: pd.DataFrame | None = None
    start = date.today() - timedelta(days=365 * DEFAULT_LOOKBACK_YEARS)

    if p.exists() and not full_refresh:
        existing = pd.read_parquet(p)
        if not existing.empty:
            last = pd.to_datetime(existing.index.max()).date()
            start = last - timedelta(days=5)  # small overlap to repair late updates

    if is_a_share(symbol):
        df = _akshare_fetch(symbol, start)
    else:
        df = _yfinance_fetch(symbol, start)

    if df.empty:
        log.warning("no data for %s", symbol)
        return existing if existing is not None else pd.DataFrame()

    if existing is not None and not existing.empty:
        df = pd.concat([existing[~existing.index.isin(df.index)], df]).sort_index()

    df.to_parquet(p)
    return df


def fetch_all(symbols: list[str], *, full_refresh: bool = False) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            out[s] = fetch_symbol(s, full_refresh=full_refresh)
            log.info("fetched %s rows=%d last=%s",
                     s, len(out[s]),
                     out[s].index.max() if not out[s].empty else "?")
        except Exception as e:  # noqa: BLE001
            log.exception("fetch failed for %s: %s", s, e)
            out[s] = pd.DataFrame()
    return out


def load_local(symbol: str) -> pd.DataFrame:
    p = _path(symbol)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def latest_spot(symbol: str, *, include_post_market: bool = True) -> dict:
    """Get the freshest available price + timestamp.

    Returns: {price, as_of_utc, session ('regular'|'pre'|'post'|'closed'), source}
    Falls back to last close from local Parquet if live fetch fails.
    """
    from datetime import datetime, timezone
    if is_a_share(symbol):
        # akshare doesn't have free realtime in our setup; use load_local
        df = load_local(symbol)
        if df.empty:
            return {"price": None, "as_of_utc": None, "session": "closed", "source": "none"}
        last_idx = df.index.max()
        return {
            "price": float(df["close"].iloc[-1]),
            "as_of_utc": (last_idx + pd.Timedelta(hours=8)).isoformat() if hasattr(last_idx, "isoformat") else str(last_idx),
            "session": "closed",
            "source": "akshare-daily",
        }

    # US stocks/ETFs: use yfinance with prepost=True for after-hours
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        # 1-minute bars including pre/post for accurate latest price
        df = t.history(period="1d", interval="1m", prepost=include_post_market)
        if df is None or df.empty:
            raise ValueError("empty intraday")
        last_idx = df.index.max()
        last_row = df.iloc[-1]
        # Determine session from time
        last_utc = last_idx.astimezone(timezone.utc) if last_idx.tzinfo else last_idx
        # US regular session 13:30-20:00 UTC; pre 09:00-13:30; post 20:00-00:00
        hh = last_utc.hour + last_utc.minute / 60
        if 13.5 <= hh < 20:
            session = "regular"
        elif 9 <= hh < 13.5:
            session = "pre"
        elif 20 <= hh or hh < 1:
            session = "post"
        else:
            session = "closed"
        return {
            "price": float(last_row["Close"]),
            "as_of_utc": last_utc.isoformat(),
            "session": session,
            "source": "yfinance-1m",
        }
    except Exception:
        # Fallback to last daily close
        df = load_local(symbol)
        if df.empty:
            return {"price": None, "as_of_utc": None, "session": "closed", "source": "none"}
        return {
            "price": float(df["close"].iloc[-1]),
            "as_of_utc": pd.Timestamp(df.index.max()).tz_localize("UTC").isoformat()
                         if hasattr(df.index.max(), "isoformat") else str(df.index.max()),
            "session": "closed",
            "source": "yfinance-daily-fallback",
        }


def staleness_seconds(as_of_utc: str | None) -> int | None:
    """How many seconds since `as_of_utc` (ISO string). None if unparseable."""
    if not as_of_utc:
        return None
    try:
        ts = pd.Timestamp(as_of_utc).tz_convert("UTC") if "+" in as_of_utc or "Z" in as_of_utc else pd.Timestamp(as_of_utc).tz_localize("UTC")
        now = pd.Timestamp.now(tz="UTC")
        return int((now - ts).total_seconds())
    except Exception:
        return None
