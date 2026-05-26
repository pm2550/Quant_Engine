"""Ad-hoc analysis + mutation API for the quant engine.

Endpoints:
  POST   /api/analyze                  signal/recommendation for a symbol
  GET    /api/portfolio/snapshot       full per-currency portfolio + signals
  POST   /api/portfolio/position       upsert a held position (creates if new)
  DELETE /api/portfolio/position/{sym} remove a held position
  POST   /api/portfolio/watchlist      add a symbol to watchlist

Writes are auto-backed up to /data2/quant/backups/portfolio.yaml.bak-<ts>.
"""
from __future__ import annotations
import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config as cfg_mod
from . import db, fetcher, recommender, signals, multi_factor, fundamentals

PORTFOLIO_FILE = cfg_mod.CONFIG_DIR / "portfolio.yaml"
BACKUP_DIR = cfg_mod.ROOT / "backups"

log = logging.getLogger(__name__)

app = FastAPI(title="Quant Ad-hoc API", version="1.0")


class AnalyzeRequest(BaseModel):
    symbol: str
    intent: str = "general"          # "buy" | "sell" | "general"
    amount_usd: Optional[float] = None  # user's desired trade size


@app.get("/api/health")
def health():
    return {"ok": True}


def _refresh(symbol: str):
    """Try a fresh fetch; on upstream failure (e.g. akshare/yfinance temporary
    connection drops), fall back to local parquet so the analyze pipeline
    can still serve cached data with a staleness marker, instead of 503.

    The 503 we used to throw made callers (阿雷) report 'engine is down' —
    but it was just one A-share data source briefly unavailable.
    """
    try:
        return fetcher.fetch_symbol(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("fetch failed for %s: %s — falling back to local parquet", symbol, e)
        local = fetcher.load_local(symbol)
        if local is None or local.empty:
            raise HTTPException(
                503,
                f"fetch failed for {symbol} and no local data: {e}. "
                "This is an upstream data source issue, not a quant-engine outage."
            )
        # Synthesize a result resembling fetch_symbol output: just the local df
        return local


def _portfolio_buckets(portfolio: dict) -> dict:
    """Return per-currency totals + per-symbol market_values, currencies, weights.

    Returns: {
      'totals': {'USD': 1480.01, 'CNY': 3042.0},
      'market_values': {sym: value},
      'currencies': {sym: 'USD'|'CNY'},
      'weights': {sym: pct_within_its_currency_bucket}
    }
    """
    held = portfolio.get("positions", {})
    default_ccy = portfolio.get("default_currency", "USD")
    market_values: dict[str, float] = {}
    currencies: dict[str, str] = {}
    for s, info in held.items():
        local = fetcher.load_local(s)
        if local.empty:
            continue
        price = float(local["close"].iloc[-1])
        market_values[s] = price * info["shares"]
        currencies[s] = info.get("currency", default_ccy)
    totals: dict[str, float] = {}
    for s, mv in market_values.items():
        c = currencies[s]
        totals[c] = totals.get(c, 0.0) + mv
    weights = {
        s: (mv / totals[currencies[s]] if totals.get(currencies[s]) else 0.0)
        for s, mv in market_values.items()
    }
    return {
        "totals": totals,
        "market_values": market_values,
        "currencies": currencies,
        "weights": weights,
    }


def _backtest_top(symbol: str, limit: int = 5) -> list[dict]:
    """Top-Sharpe completed backtests for this symbol."""
    try:
        with sqlite3.connect(db.DB_PATH) as c:
            c.row_factory = sqlite3.Row
            rows = c.execute(
                """
                SELECT bt.strategy, bt.symbol, bt.params_json, bt.period_years,
                       br.sharpe, br.total_return, br.annual_return,
                       br.max_drawdown, br.n_trades, br.win_rate
                FROM backtest_tasks bt
                JOIN backtest_results br ON bt.id = br.task_id
                WHERE bt.symbol = ? AND br.n_trades >= 5
                ORDER BY br.sharpe DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        out = []
        for r in rows:
            params = json.loads(r["params_json"])
            params.pop("as_of", None)
            out.append({
                "strategy": r["strategy"],
                "params": params,
                "period_years": r["period_years"],
                "sharpe": round(r["sharpe"], 2),
                "total_return_pct": round(r["total_return"] * 100, 1),
                "annual_return_pct": round(r["annual_return"] * 100, 1),
                "max_drawdown_pct": round(r["max_drawdown"] * 100, 1),
                "n_trades": r["n_trades"],
                "win_rate_pct": round(r["win_rate"] * 100, 1),
            })
        return out
    except Exception:
        log.exception("backtest_top query failed")
        return []


def _suggest_amount(*, intent: str, user_amount_usd: Optional[float],
                    rec_action: str, total_value: float, current_weight: float,
                    target_weight: float, price: float) -> dict:
    """Return {usd, shares, rationale} dict for suggested trade size."""
    # If user told us their intended size, anchor on that
    if user_amount_usd is not None:
        amt = user_amount_usd
        rationale = f"按你说的 ${amt:.0f}"
    else:
        if rec_action in ("ADD", "BUY", "WATCH_BUY"):
            target_pct_cap = max(target_weight, 0.07)  # default 7% if rec didn't specify
            tgt_val = target_pct_cap * total_value if total_value else 0
            amt = max(0, tgt_val - current_weight * total_value)
            if amt < 20:
                amt = 30  # minimum useful trade
            rationale = f"目标权重 {target_pct_cap*100:.1f}% 对应 ${tgt_val:.0f}, 差额 ${amt:.0f}"
        elif rec_action in ("REDUCE", "SELL", "STOP_LOSS", "TAKE_PROFIT"):
            tgt_val = target_weight * total_value if total_value else 0
            amt = max(0, current_weight * total_value - tgt_val)
            if amt < 20:
                amt = 30
            rationale = f"目标权重 {target_weight*100:.1f}% 对应 ${tgt_val:.0f}, 减仓 ${amt:.0f}"
        else:  # HOLD
            amt = 0
            rationale = "不建议操作"

    shares = (amt / price) if price else 0
    return {
        "usd": round(amt, 2),
        "shares": round(shares, 4),
        "rationale": rationale,
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    sym = req.symbol.upper().strip()
    # Allow A-share suffixes (.SS / .SZ) and HK (.HK) etc.
    if not sym or not all(ch.isalnum() or ch in ".-" for ch in sym):
        raise HTTPException(400, f"invalid symbol: {sym!r}")

    portfolio = cfg_mod.load("portfolio")
    strategies_cfg = cfg_mod.load("strategies")
    held = portfolio.get("positions", {})
    watch_syms = {w["symbol"] for w in portfolio.get("watchlist", [])}
    watch_meta = next((w for w in portfolio.get("watchlist", []) if w.get("symbol") == sym), {})
    default_ccy = portfolio.get("default_currency", "USD")

    # Refresh price data
    df = _refresh(sym)
    if df is None or df.empty:
        raise HTTPException(404, f"no price data found for {sym}")

    # Compute signals
    sig = signals.compute(sym, df, strategies_cfg)
    if sig is None:
        raise HTTPException(422, f"signal computation failed for {sym}")

    # Portfolio context — currency-aware
    buckets = _portfolio_buckets(portfolio)
    sym_ccy = (held.get(sym) or watch_meta).get(
        "currency",
        default_ccy if not fetcher.is_a_share(sym) else "CNY",
    )
    bucket_total = buckets["totals"].get(sym_ccy, 0.0)
    cur_weight = buckets["weights"].get(sym, 0.0)
    cur_value = buckets["market_values"].get(sym, 0.0)
    cur_shares = held[sym]["shares"] if sym in held else 0.0

    # Live spot price (after-hours aware for US stocks)
    try:
        spot = fetcher.latest_spot(sym, include_post_market=True)
    except Exception:
        spot = {"price": None, "as_of_utc": None, "session": "closed", "source": "none"}
    spot["staleness_seconds"] = fetcher.staleness_seconds(spot.get("as_of_utc"))

    # Multi-factor scoring (technical + catalyst + sentiment + fundamental + analyst + momentum + macro)
    try:
        fund_data = fundamentals.latest(sym)
        signals_dict = sig.as_dict() if hasattr(sig, "as_dict") else {}
        cur_price_for_score = spot.get("price") or sig.price
        multi = multi_factor.score(sym, signals_dict, fund_data, cur_price_for_score)
    except Exception as e:  # noqa: BLE001
        log.exception("multi_factor failed: %s", e)
        multi = {
            "composite_score": 0.0,
            "action": "HOLD",
            "rationale": "multi_factor failed; default HOLD",
            "catalyst_imminent": False,
            "factor_breakdown": {},
        }

    # Recommendation (sized against its own currency bucket only)
    if sym in held:
        rec = recommender.for_held_multi_factor(
            sig,
            multi,
            current_weight=cur_weight,
            total_value=bucket_total,
        )
    else:
        rec = recommender.for_watch_multi_factor(sig, multi)

    # Suggested amount (in symbol's own currency)
    suggested = _suggest_amount(
        intent=req.intent,
        user_amount_usd=req.amount_usd,
        rec_action=rec.action,
        total_value=bucket_total,
        current_weight=cur_weight,
        target_weight=rec.target_weight,
        price=sig.price,
    )
    # Round share suggestion appropriately for A-share (整百)
    if sym_ccy == "CNY":
        suggested["shares"] = int(round(suggested["shares"] / 100.0)) * 100

    # Risk overlap check (only same-currency members)
    overlap_notes = []
    sector_overlap = portfolio.get("sector_overlap", {})
    for sector, members in sector_overlap.items():
        if sym in members:
            sw = sum(buckets["weights"].get(m, 0.0) for m in members
                     if buckets["currencies"].get(m) == sym_ccy)
            overlap_notes.append({
                "sector": sector,
                "current_pct": round(sw * 100, 1),
                "members": members,
            })

    # Friendly display name from portfolio config
    info = held.get(sym) or watch_meta
    display_name = info.get("name") or sym
    # User-facing ticker (strip exchange suffix for A-shares so LLM doesn't show '002624.SZ')
    user_ticker = sym.split(".")[0] if fetcher.is_a_share(sym) else sym

    return {
        "symbol": user_ticker,
        "display_name": display_name,
        "currency": sym_ccy,
        "intent": req.intent,
        "is_held": sym in held,
        "is_watchlist": sym in watch_syms,
        "spot": spot,
        "as_of_data": sig.last_date,
        "multi_factor": multi,
        "current": {
            "shares": cur_shares,
            "value": round(cur_value, 2),
            "weight_pct": round(cur_weight * 100, 2),
        },
        "portfolio_bucket_total": round(bucket_total, 2),
        "portfolio_totals_by_currency": {c: round(t, 2) for c, t in buckets["totals"].items()},
        "price": {
            "last": round(sig.price, 2),
            "chg_1d_pct": round(sig.chg_1d_pct, 2),
            "chg_5d_pct": round(sig.chg_5d_pct, 2),
            "chg_20d_pct": round(sig.chg_20d_pct, 2),
        },
        "indicators": {
            "rsi": round(sig.rsi, 1) if sig.rsi == sig.rsi else None,
            "ma20": round(sig.ma20, 2) if sig.ma20 == sig.ma20 else None,
            "ma50": round(sig.ma50, 2) if sig.ma50 == sig.ma50 else None,
            "ma200": round(sig.ma200, 2) if sig.ma200 == sig.ma200 else None,
            "above_ma50": sig.above_ma50,
            "above_ma200": sig.above_ma200,
            "atr_pct": round(sig.atr_pct, 2),
        },
        "signal_codes": sig.signal_codes,
        "recommendation": {
            "action": rec.action,
            "confidence": rec.confidence,
            "reason_codes": rec.reason_codes,
            "target_weight_pct": round(rec.target_weight * 100, 2),
            "current_weight_pct": round(rec.current_weight * 100, 2),
            "suggested_trade": suggested,
            "notes": rec.notes,
        },
        "sector_overlap": overlap_notes,
        "backtest_top5": _backtest_top(sym),
        "data_date": sig.last_date,
    }


# ---- Portfolio mutation API ----

class UpsertPositionRequest(BaseModel):
    symbol: str            # raw input; we'll normalise via _resolve_symbol
    shares: float          # positive; 0 not allowed (use DELETE)
    name: Optional[str] = None
    currency: Optional[str] = None  # auto-detect if missing
    theme: Optional[str] = None


class WatchlistAddRequest(BaseModel):
    symbol: str
    theme: Optional[str] = None
    reason: Optional[str] = None


def _backup_portfolio() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKUP_DIR / f"portfolio.yaml.bak-{int(datetime.utcnow().timestamp())}"
    shutil.copy2(PORTFOLIO_FILE, dst)
    return dst


def _load_portfolio_raw() -> dict:
    with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_portfolio_raw(data: dict) -> None:
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def _resolve_symbol(raw: str) -> tuple[str, str]:
    """Normalise user input to (canonical_symbol, currency).

    e.g. '002624' / '完美世界' / '002624.SZ' → ('002624.SZ', 'CNY')
         'amd' / 'AMD' → ('AMD', 'USD')
    """
    s = raw.strip()
    # Hard-coded aliases (keep in sync with stock_query.py)
    aliases = {
        "PWRD": "002624.SZ",
        "PERFECTWORLD": "002624.SZ",
        "完美世界": "002624.SZ",
        "002624": "002624.SZ",
    }
    upper = s.upper()
    canonical = aliases.get(upper) or aliases.get(s) or upper
    if canonical.endswith((".SS", ".SZ")):
        return canonical, "CNY"
    return canonical, "USD"


def _ensure_data(symbol: str) -> int:
    """Make sure we have local Parquet for this symbol; return rows fetched."""
    df = fetcher.fetch_symbol(symbol)
    return len(df) if df is not None else 0


@app.post("/api/portfolio/position")
def upsert_position(req: UpsertPositionRequest) -> dict:
    """Add or update a held position. shares=0 not allowed (use DELETE)."""
    if req.shares <= 0:
        raise HTTPException(400, "shares must be > 0; use DELETE to remove")
    canonical, ccy_default = _resolve_symbol(req.symbol)
    currency = req.currency or ccy_default

    portfolio = _load_portfolio_raw()
    positions = portfolio.setdefault("positions", {})
    before = positions.get(canonical, {}).copy()

    # Auto-fetch if it's a new symbol
    rows = 0
    if canonical not in positions:
        rows = _ensure_data(canonical)
        if rows == 0:
            raise HTTPException(422, f"could not fetch data for {canonical}")

    backup = _backup_portfolio()
    entry = {
        "shares": req.shares,
        "currency": currency,
    }
    if req.name:
        entry["name"] = req.name
    elif before.get("name"):
        entry["name"] = before["name"]
    if req.theme:
        entry["theme"] = req.theme
    elif before.get("theme"):
        entry["theme"] = before["theme"]
    if currency == "CNY":
        entry["market"] = "a_share"
    positions[canonical] = entry
    _save_portfolio_raw(portfolio)

    return {
        "ok": True,
        "symbol": canonical.split(".")[0] if currency == "CNY" else canonical,
        "currency": currency,
        "before": before,
        "after": entry,
        "data_rows_fetched": rows,
        "backup": backup.name,
    }


@app.delete("/api/portfolio/position/{symbol}")
def remove_position(symbol: str) -> dict:
    canonical, _ = _resolve_symbol(symbol)
    portfolio = _load_portfolio_raw()
    positions = portfolio.get("positions", {})
    if canonical not in positions:
        raise HTTPException(404, f"position not found: {canonical}")
    backup = _backup_portfolio()
    removed = positions.pop(canonical)
    _save_portfolio_raw(portfolio)
    return {"ok": True, "removed_symbol": canonical, "removed": removed, "backup": backup.name}


@app.delete("/api/portfolio/watchlist/{symbol}")
def remove_watchlist(symbol: str) -> dict:
    canonical, _ = _resolve_symbol(symbol)
    portfolio = _load_portfolio_raw()
    watch = portfolio.get("watchlist", [])
    new_watch = [w for w in watch if w.get("symbol") != canonical]
    if len(new_watch) == len(watch):
        raise HTTPException(404, f"not in watchlist: {canonical}")
    backup = _backup_portfolio()
    portfolio["watchlist"] = new_watch
    _save_portfolio_raw(portfolio)
    return {"ok": True, "removed_symbol": canonical, "backup": backup.name}


@app.post("/api/portfolio/watchlist")
def add_watchlist(req: WatchlistAddRequest) -> dict:
    canonical, ccy = _resolve_symbol(req.symbol)
    portfolio = _load_portfolio_raw()
    watch = portfolio.setdefault("watchlist", [])
    if any(w.get("symbol") == canonical for w in watch):
        return {"ok": False, "message": f"{canonical} already in watchlist"}
    rows = _ensure_data(canonical)
    if rows == 0:
        raise HTTPException(422, f"could not fetch data for {canonical}")
    backup = _backup_portfolio()
    entry = {"symbol": canonical, "currency": ccy}
    if req.theme:
        entry["theme"] = req.theme
    if req.reason:
        entry["reason"] = req.reason
    watch.append(entry)
    _save_portfolio_raw(portfolio)
    return {"ok": True, "added": entry, "backup": backup.name, "data_rows": rows}


# ---- Audio transcription (auto-routes to cloud or local whisper) ----

class TranscribeRequest(BaseModel):
    audio: str               # local path or http(s) URL
    language: str = "auto"   # auto | zh | en
    prefer: str = "auto"     # auto | cloud | local


@app.post("/api/transcribe")
def transcribe_endpoint(req: TranscribeRequest) -> dict:
    """Transcribe audio. Cloud (Aliyun ASR) if ALIYUN_ASR_KEY set, else local whisper."""
    from . import transcribe as transcribe_mod
    try:
        return transcribe_mod.transcribe(
            req.audio, language=req.language, prefer=req.prefer
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"transcription failed: {e}")


@app.get("/api/portfolio/snapshot")
def portfolio_snapshot() -> dict:
    """Return portfolio composition split by currency + per-stock latest signals."""
    portfolio = cfg_mod.load("portfolio")
    strategies_cfg = cfg_mod.load("strategies")
    held = portfolio.get("positions", {})
    buckets = _portfolio_buckets(portfolio)
    out_pos = []
    for s, info in held.items():
        local = fetcher.load_local(s)
        if local.empty:
            continue
        sig = signals.compute(s, local, strategies_cfg)
        if sig is None:
            continue
        ccy = buckets["currencies"].get(s, "USD")
        user_ticker = s.split(".")[0] if fetcher.is_a_share(s) else s
        out_pos.append({
            "symbol": user_ticker,
            "display_name": info.get("name") or user_ticker,
            "currency": ccy,
            "shares": info["shares"],
            "value": round(buckets["market_values"].get(s, 0), 2),
            "weight_pct": round(buckets["weights"].get(s, 0) * 100, 2),
            "price": round(sig.price, 2),
            "chg_1d_pct": round(sig.chg_1d_pct, 2),
            "rsi": round(sig.rsi, 1) if sig.rsi == sig.rsi else None,
            "signal_codes": sig.signal_codes,
            "data_date": sig.last_date,
        })
    return {
        "totals_by_currency": {c: round(t, 2) for c, t in buckets["totals"].items()},
        "positions": out_pos,
        "data_date": max((p["data_date"] for p in out_pos), default=""),
    }


# ---- P0: Intraday OHLC with key event markers ----


@app.get("/api/intraday")
def intraday_bars(symbol: str, interval: str = "5m", date: str = "today",
                   prepost: bool = True) -> dict:
    """Return 1m/5m OHLC bars + key event time markers for a single trading day.

    Answers questions like:
      "AMD 多久涨 15%" — `time_to_high_minutes`
      "今天哪段涨最猛" — bar with max abs `chg_pct`
      "盘后跳了多少" — `post_market` summary section

    Params:
      symbol:   ticker (e.g. AMD or 002624.SZ)
      interval: 1m | 2m | 5m | 15m | 30m | 60m | 90m | 1h
      date:     "today" or YYYY-MM-DD (only most recent ~30 days for 1m)
      prepost:  include pre/post-market bars (US only)
    """
    import pandas as pd
    if interval not in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
        raise HTTPException(400, f"invalid interval {interval}")

    try:
        import yfinance as yf
    except ImportError as e:
        raise HTTPException(500, f"yfinance unavailable: {e}")

    is_cn = fetcher.is_a_share(symbol)
    if is_cn and interval != "5m":
        raise HTTPException(400, "A-share intraday only supports interval=5m via akshare")

    # Resolve date
    if date == "today":
        target_date = pd.Timestamp.utcnow().date()
    else:
        try:
            target_date = pd.Timestamp(date).date()
        except Exception:
            raise HTTPException(400, f"invalid date {date!r}")

    # ---- Fetch bars ----
    if is_cn:
        try:
            import akshare as ak
            code = symbol.split(".")[0]
            mkt = "sz" if symbol.upper().endswith(".SZ") else "sh"
            df = ak.stock_zh_a_minute(symbol=mkt + code, period="5", adjust="qfq")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"akshare fetch failed: {e}")
        if df is None or df.empty:
            raise HTTPException(404, f"no intraday data for {symbol}")
        df["dt"] = pd.to_datetime(df["day"])
        target_local = (pd.Timestamp(target_date) + pd.Timedelta(hours=8)).date()
        df = df[df["dt"].dt.date == target_local]
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                  "close": "Close", "volume": "Volume"})
        df = df.set_index("dt")
        currency = "CNY"
    else:
        t = yf.Ticker(symbol)
        # 1m needs period<=8d, 5m <=60d. Use 5d for safety.
        period = "5d" if interval in {"1m", "2m"} else "30d"
        try:
            df = t.history(period=period, interval=interval, prepost=prepost)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"yfinance fetch failed: {e}")
        if df is None or df.empty:
            raise HTTPException(404, f"no intraday data for {symbol}")
        df.index = pd.to_datetime(df.index)
        df_utc_dates = df.index.tz_convert("UTC").date if df.index.tz else df.index.date
        df = df[df_utc_dates == target_date]
        currency = "USD"

    if df.empty:
        raise HTTPException(404, f"no bars on {target_date} for {symbol}")

    # ---- Build bar series ----
    bars = []
    open_p = float(df["Open"].iloc[0])
    for ts, row in df.iterrows():
        close_p = float(row["Close"])
        bars.append({
            "ts": ts.isoformat(),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(close_p, 4),
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
            "chg_from_session_open_pct": round((close_p / open_p - 1) * 100, 3) if open_p else 0,
        })

    # ---- Key time markers ----
    high_idx = df["High"].idxmax()
    low_idx = df["Low"].idxmin()
    open_idx = df.index[0]
    close_idx = df.index[-1]
    high_p = float(df["High"].max())
    low_p = float(df["Low"].min())
    last_p = float(df["Close"].iloc[-1])

    # Time to high/low from session open (in minutes)
    minutes_to_high = (high_idx - open_idx).total_seconds() / 60
    minutes_to_low = (low_idx - open_idx).total_seconds() / 60

    # Largest single-bar move
    df_chg = df["Close"].pct_change().abs() * 100
    max_bar_idx = df_chg.idxmax() if df_chg.notna().any() else None
    max_bar_chg = float(df_chg.max()) if max_bar_idx is not None else 0.0

    # Earnings time marker (US only) — check earnings_calendar for today
    earnings_marker = None
    if not is_cn:
        try:
            with db.conn() as c:
                row = c.execute(
                    "SELECT report_date, eps_estimate, eps_actual FROM earnings_calendar "
                    "WHERE symbol=? AND report_date=?",
                    (symbol, str(target_date)),
                ).fetchone()
                if row:
                    earnings_marker = dict(row)
        except Exception:  # noqa: BLE001
            pass

    # Pre/post split for US
    pre_post = None
    if not is_cn and prepost:
        # yfinance flags pre-market roughly 09:00-13:30 UTC, post 20:00-00:00 UTC
        pre_mask = df.index.tz_convert("UTC").hour.isin([9, 10, 11, 12, 13]) if df.index.tz else False
        post_mask = df.index.tz_convert("UTC").hour.isin([20, 21, 22, 23]) if df.index.tz else False
        try:
            pre_df = df[pre_mask] if pre_mask is not False else df.iloc[0:0]
            post_df = df[post_mask] if post_mask is not False else df.iloc[0:0]
            pre_post = {
                "pre_market": _summarize_session(pre_df, open_p) if not pre_df.empty else None,
                "post_market": _summarize_session(post_df, open_p) if not post_df.empty else None,
            }
        except Exception:  # noqa: BLE001
            pre_post = None

    return {
        "symbol": symbol,
        "currency": currency,
        "date": str(target_date),
        "interval": interval,
        "n_bars": len(bars),
        "summary": {
            "open": round(open_p, 4),
            "high": round(high_p, 4),
            "low": round(low_p, 4),
            "last": round(last_p, 4),
            "range_pct": round((high_p - low_p) / open_p * 100, 2) if open_p else 0,
            "chg_from_open_pct": round((last_p / open_p - 1) * 100, 2) if open_p else 0,
            "high_at": high_idx.isoformat(),
            "low_at": low_idx.isoformat(),
            "minutes_open_to_high": round(minutes_to_high, 1),
            "minutes_open_to_low": round(minutes_to_low, 1),
            "max_bar_move_pct": round(max_bar_chg, 2),
            "max_bar_at": max_bar_idx.isoformat() if max_bar_idx is not None else None,
        },
        "earnings_today": earnings_marker,
        "session_split": pre_post,
        "bars": bars,
    }


def _summarize_session(df, day_open: float) -> dict:
    """Summarize a session slice (pre/regular/post) of intraday bars."""
    import pandas as pd  # noqa: F401
    if df.empty:
        return {}
    open_p = float(df["Open"].iloc[0])
    last_p = float(df["Close"].iloc[-1])
    return {
        "first_at": df.index[0].isoformat(),
        "last_at": df.index[-1].isoformat(),
        "n_bars": len(df),
        "open": round(open_p, 4),
        "last": round(last_p, 4),
        "high": round(float(df["High"].max()), 4),
        "low": round(float(df["Low"].min()), 4),
        "chg_within_session_pct": round((last_p / open_p - 1) * 100, 2) if open_p else 0,
        "chg_from_day_open_pct": round((last_p / day_open - 1) * 100, 2) if day_open else 0,
    }


@app.get("/api/intraday/time_to_move")
def time_to_move(
    symbol: str,
    threshold_pct: float = 10.0,    # signed: +10 = 涨 10%, -10 = 跌 10%
    date: str = "today",
    prepost: bool = True,
    reference: str = "session_open",  # session_open | prev_close
) -> dict:
    """从基准价开始, 价格首次达到 ± threshold_pct 用了多少分钟.

    Answers questions like:
      "AMD 今天暴涨 10% 花了多久, 是半个小时吗?"
      "完美世界跌停花了多长时间?"

    Params:
      threshold_pct:  signed (+ 上涨 / - 下跌). 例: +10 = 找首次涨到 open*1.10
      reference:      session_open (默认, 跟当日开盘比) | prev_close (跟昨收比)
    """
    import pandas as pd
    try:
        import yfinance as yf
    except ImportError as e:
        raise HTTPException(500, f"yfinance unavailable: {e}")

    is_cn = fetcher.is_a_share(symbol)
    if is_cn:
        raise HTTPException(400, "A 股 1min 数据不支持; 用 /api/intraday?interval=5m")
    if reference not in {"session_open", "prev_close"}:
        raise HTTPException(400, f"invalid reference {reference!r}")

    if date == "today":
        target_date = pd.Timestamp.utcnow().date()
    else:
        try:
            target_date = pd.Timestamp(date).date()
        except Exception:
            raise HTTPException(400, f"invalid date {date!r}")

    t = yf.Ticker(symbol)
    try:
        df = t.history(period="5d", interval="1m", prepost=prepost)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"yfinance fetch failed: {e}")
    if df is None or df.empty:
        raise HTTPException(404, f"no 1m data for {symbol}")
    df.index = pd.to_datetime(df.index)
    df_utc_dates = df.index.tz_convert("UTC").date if df.index.tz else df.index.date
    df_today = df[df_utc_dates == target_date]
    if df_today.empty:
        raise HTTPException(404, f"no 1m bars on {target_date} for {symbol}")

    open_p = float(df_today["Open"].iloc[0])
    open_ts = df_today.index[0]

    if reference == "prev_close":
        # find prior trading day's close from the 5d window
        df_before = df[df_utc_dates < target_date]
        if df_before.empty:
            raise HTTPException(404, "no prev_close found in 5d window")
        ref_price = float(df_before["Close"].iloc[-1])
        ref_label = "昨收"
    else:
        ref_price = open_p
        ref_label = "今日开盘"

    direction = 1 if threshold_pct >= 0 else -1
    target_price = ref_price * (1 + threshold_pct / 100)

    if direction > 0:
        hit_mask = df_today["High"] >= target_price
    else:
        hit_mask = df_today["Low"] <= target_price

    last_p = float(df_today["Close"].iloc[-1])
    last_chg_from_ref = (last_p / ref_price - 1) * 100

    if not hit_mask.any():
        # 还未触及
        if direction > 0:
            extreme_p = float(df_today["High"].max())
            extreme_idx = df_today["High"].idxmax()
        else:
            extreme_p = float(df_today["Low"].min())
            extreme_idx = df_today["Low"].idxmin()
        extreme_chg = (extreme_p / ref_price - 1) * 100
        extreme_minutes = (extreme_idx - open_ts).total_seconds() / 60
        return {
            "symbol": symbol,
            "date": str(target_date),
            "threshold_pct": threshold_pct,
            "reference": reference,
            "reference_price": round(ref_price, 4),
            "open_ts": open_ts.isoformat(),
            "hit": False,
            "current_price": round(last_p, 4),
            "current_chg_pct": round(last_chg_from_ref, 3),
            "extreme_price_today": round(extreme_p, 4),
            "extreme_chg_pct_today": round(extreme_chg, 3),
            "extreme_at_minutes": round(extreme_minutes, 1),
            "explanation": (
                f"{symbol} {ref_label} ${ref_price:.2f}, "
                f"今日极{('高' if direction>0 else '低')} ${extreme_p:.2f} ({extreme_chg:+.2f}%), "
                f"还未触及 {threshold_pct:+.1f}%"
            ),
        }

    hit_idx = hit_mask.idxmax()
    hit_bar = df_today.loc[hit_idx]
    hit_price = float(hit_bar["High"]) if direction > 0 else float(hit_bar["Low"])
    hit_actual_pct = (hit_price / ref_price - 1) * 100
    minutes_elapsed = (hit_idx - open_ts).total_seconds() / 60

    df_path = df_today.loc[:hit_idx]
    if direction > 0:
        worst_along = float(df_path["Low"].min())
        worst_along_pct = (worst_along / ref_price - 1) * 100
    else:
        worst_along = float(df_path["High"].max())
        worst_along_pct = (worst_along / ref_price - 1) * 100

    bars_n = len(df_path)
    return {
        "symbol": symbol,
        "date": str(target_date),
        "threshold_pct": threshold_pct,
        "reference": reference,
        "reference_price": round(ref_price, 4),
        "open_ts": open_ts.isoformat(),
        "hit": True,
        "hit_ts": hit_idx.isoformat(),
        "hit_price": round(hit_price, 4),
        "hit_actual_pct": round(hit_actual_pct, 3),
        "minutes_elapsed": round(minutes_elapsed, 1),
        "bars_elapsed": bars_n,
        "worst_along_path_price": round(worst_along, 4),
        "worst_along_path_pct": round(worst_along_pct, 3),
        "current_price": round(last_p, 4),
        "current_chg_pct": round(last_chg_from_ref, 3),
        "explanation": (
            f"{symbol} {ref_label} ${ref_price:.2f} → "
            f"{hit_idx.strftime('%H:%M UTC')} 触及 ${hit_price:.2f} "
            f"({hit_actual_pct:+.2f}%), 用时 {minutes_elapsed:.0f} 分钟"
        ),
    }


# ---- P0: What-if simulation ----


class WhatIfTrade(BaseModel):
    symbol: str
    action: str           # "buy" | "sell"
    shares: float

class WhatIfRequest(BaseModel):
    trades: list[WhatIfTrade]


@app.post("/api/whatif")
def whatif(req: WhatIfRequest) -> dict:
    """Simulate proposed trades against current portfolio. Show before/after.

    Returns:
      before:  current totals + per-symbol weights + theme exposure
      after:   same metrics post-trades
      delta:   weight shifts (which themes increased/decreased)
      cash_impact: dict per currency (negative = cost, positive = proceeds)
    """
    import copy
    portfolio = cfg_mod.load("portfolio")
    if not req.trades:
        raise HTTPException(400, "trades list is empty")

    before = _whatif_metrics(portfolio)

    # Build mutated portfolio
    sim = copy.deepcopy(portfolio)
    sim_positions = sim.setdefault("positions", {})
    cash_impact: dict[str, float] = {}

    for t in req.trades:
        sym = t.symbol.strip().upper()
        # Resolve A-share alias
        if not (sym.endswith(".SS") or sym.endswith(".SZ")) and sym in sim_positions:
            pass
        # Look up current price
        df = fetcher.load_local(sym)
        if df.empty:
            raise HTTPException(404, f"no price data for {sym}")
        price = float(df["close"].iloc[-1])
        ccy = sim_positions.get(sym, {}).get("currency",
                                              "CNY" if fetcher.is_a_share(sym) else "USD")
        delta_value = price * t.shares
        if t.action == "buy":
            cash_impact[ccy] = cash_impact.get(ccy, 0) - delta_value
            if sym in sim_positions:
                sim_positions[sym]["shares"] = float(sim_positions[sym]["shares"]) + t.shares
            else:
                sim_positions[sym] = {"shares": t.shares, "currency": ccy,
                                       "name": sym, "theme": "uncategorized"}
        elif t.action == "sell":
            cash_impact[ccy] = cash_impact.get(ccy, 0) + delta_value
            if sym not in sim_positions:
                raise HTTPException(400, f"can't sell {sym} — not in portfolio")
            new_shares = float(sim_positions[sym]["shares"]) - t.shares
            if new_shares < -1e-6:
                raise HTTPException(400, f"selling more {sym} than held "
                                          f"({sim_positions[sym]['shares']} → {new_shares})")
            if abs(new_shares) < 1e-6:
                del sim_positions[sym]
            else:
                sim_positions[sym]["shares"] = new_shares
        else:
            raise HTTPException(400, f"action must be buy/sell, got {t.action!r}")

    after = _whatif_metrics(sim)

    # Compute weight delta per symbol
    weight_delta = {}
    all_syms = set(before["weights"]) | set(after["weights"])
    for s in all_syms:
        b = before["weights"].get(s, 0)
        a = after["weights"].get(s, 0)
        if abs(a - b) > 0.001:
            weight_delta[s] = round(a - b, 4)

    # Theme exposure delta
    theme_delta = {}
    all_themes = set(before["theme_exposure"]) | set(after["theme_exposure"])
    for th in all_themes:
        b = before["theme_exposure"].get(th, 0)
        a = after["theme_exposure"].get(th, 0)
        if abs(a - b) > 0.001:
            theme_delta[th] = round(a - b, 4)

    return {
        "before": before,
        "after": after,
        "delta": {
            "weights": weight_delta,
            "theme_exposure": theme_delta,
            "n_positions": after["n_positions"] - before["n_positions"],
        },
        "cash_impact": {c: round(v, 2) for c, v in cash_impact.items()},
        "trades_applied": [t.dict() for t in req.trades],
    }


def _whatif_metrics(portfolio: dict) -> dict:
    """Compact metrics block used for whatif before/after comparison."""
    buckets = _portfolio_buckets(portfolio)
    held = portfolio.get("positions", {})
    weights = buckets.get("weights", {})

    # Theme exposure: sum weight by `theme` field within each currency bucket
    theme_exposure: dict[str, float] = {}
    for s, info in held.items():
        theme = info.get("theme", "uncategorized")
        w = weights.get(s, 0)
        theme_exposure[theme] = theme_exposure.get(theme, 0) + w

    return {
        "totals_by_currency": {c: round(v, 2) for c, v in buckets.get("totals", {}).items()},
        "n_positions": len(held),
        "weights": {s: round(w, 4) for s, w in weights.items()},
        "theme_exposure": {k: round(v, 4) for k, v in theme_exposure.items()},
    }


# ---- P1: /api/risk — VaR/CVaR/correlation matrix ----


@app.get("/api/risk")
def risk_endpoint(currency: str = "USD", lookback_days: int = 252,
                   include_corr: bool = True) -> dict:
    """Portfolio risk: VaR/CVaR/max drawdown/sigma/correlation matrix.

    Params:
      currency:      USD or CNY (separate buckets)
      lookback_days: window for sigma + VaR estimation
      include_corr:  add per-pair correlation matrix
    """
    from . import risk as risk_mod
    portfolio = cfg_mod.load("portfolio")
    rets = risk_mod._portfolio_returns(portfolio, currency=currency,
                                         lookback_days=lookback_days)
    if rets.empty:
        raise HTTPException(404, f"no return data for {currency} bucket")

    out = {
        "currency": currency,
        "lookback_days": len(rets),
        "annualized_volatility_pct": round(float(rets.std() * (252 ** 0.5) * 100), 2),
        "annualized_return_pct": round(float(rets.mean() * 252 * 100), 2),
        "sharpe_naive": (round(float(rets.mean() * 252 / (rets.std() * (252 ** 0.5))), 3)
                          if rets.std() > 0 else None),
        "var_95": risk_mod.parametric_var(rets, conf=0.95),
        "var_99": risk_mod.parametric_var(rets, conf=0.99),
        "historical_var_95": risk_mod.historical_var(rets, conf=0.95),
        "max_drawdown": risk_mod.max_drawdown(rets),
        "stress_tests": risk_mod.stress_test(portfolio, currency=currency),
    }

    # Translate VaR % to absolute currency amount
    buckets = _portfolio_buckets(portfolio)
    bucket_total = buckets.get("totals", {}).get(currency, 0)
    if bucket_total and out.get("var_95"):
        out["var_95"]["loss_amount"] = round(
            bucket_total * out["var_95"]["var_pct_1d"] / 100, 2)
        out["var_99"]["loss_amount"] = round(
            bucket_total * out["var_99"]["var_pct_1d"] / 100, 2)

    if include_corr:
        out["correlation_matrix"] = _correlation_matrix(portfolio, currency, lookback_days)

    return out


def _correlation_matrix(portfolio: dict, currency: str, lookback_days: int) -> dict:
    """Per-pair correlation of daily returns within the same currency bucket.

    Why: if every holding correlates 0.85+, your N=7 portfolio is effectively
    a 1-position bet — the diversification benefit you think you have is gone.
    """
    import pandas as pd
    held = portfolio.get("positions", {})
    syms = [s for s, info in held.items() if info.get("currency", "USD") == currency]
    if len(syms) < 2:
        return {}
    cols = {}
    for s in syms:
        df = fetcher.load_local(s)
        if df.empty:
            continue
        c = df["close"].astype(float)
        c.index = pd.to_datetime(c.index)
        cols[s] = c.pct_change()
    if len(cols) < 2:
        return {}
    rets = pd.concat(cols, axis=1).dropna(how="any").tail(lookback_days)
    if rets.empty:
        return {}
    corr = rets.corr().round(2)
    # Average pairwise correlation (excluding diagonal)
    n = corr.shape[0]
    avg = (corr.values.sum() - n) / (n * n - n) if n > 1 else None
    return {
        "lookback_days": len(rets),
        "matrix": corr.to_dict(),
        "avg_pairwise": round(float(avg), 3) if avg is not None else None,
    }


# ---- P1: /api/scenario — custom shock stress test ----


class ShockSpec(BaseModel):
    symbol: Optional[str] = None     # apply to a specific holding
    theme: Optional[str] = None      # apply to all holdings sharing this theme
    shock_pct: float                 # e.g. -0.30 for -30%

class ScenarioRequest(BaseModel):
    shocks: list[ShockSpec]
    name: Optional[str] = None       # optional label for the scenario


@app.post("/api/scenario")
def scenario(req: ScenarioRequest) -> dict:
    """Custom stress: apply a shock vector and report P&L per holding + total.

    Examples:
      [{symbol: SOXX, shock_pct: -0.30}]              — SOXX drops 30%
      [{theme: ai_compute, shock_pct: -0.20}]         — all ai_compute holdings -20%
      [{theme: broad_market, shock_pct: 0.05},
       {theme: ai_compute, shock_pct: -0.15}]         — multi-leg
    """
    if not req.shocks:
        raise HTTPException(400, "shocks list is empty")
    portfolio = cfg_mod.load("portfolio")
    held = portfolio.get("positions", {})

    # Build per-symbol shock (last write wins for overlap)
    sym_shock: dict[str, float] = {}
    for sh in req.shocks:
        if sh.symbol:
            sym_shock[sh.symbol] = sh.shock_pct
        elif sh.theme:
            for s, info in held.items():
                if info.get("theme") == sh.theme:
                    sym_shock[s] = sh.shock_pct
        else:
            raise HTTPException(400, "each shock needs symbol or theme")

    # Compute P&L per holding under shock
    rows = []
    pnl_by_ccy: dict[str, float] = {}
    base_by_ccy: dict[str, float] = {}
    for s, info in held.items():
        df = fetcher.load_local(s)
        if df.empty:
            continue
        price = float(df["close"].iloc[-1])
        ccy = info.get("currency", "USD")
        value = price * info["shares"]
        shock = sym_shock.get(s, 0.0)
        pnl = value * shock
        pnl_by_ccy[ccy] = pnl_by_ccy.get(ccy, 0) + pnl
        base_by_ccy[ccy] = base_by_ccy.get(ccy, 0) + value
        if shock != 0:
            rows.append({
                "symbol": s,
                "theme": info.get("theme", "uncategorized"),
                "currency": ccy,
                "value_before": round(value, 2),
                "shock_pct": round(shock * 100, 2),
                "pnl": round(pnl, 2),
                "value_after": round(value * (1 + shock), 2),
            })

    summary = {}
    for ccy, base in base_by_ccy.items():
        pnl = pnl_by_ccy.get(ccy, 0)
        summary[ccy] = {
            "base_value": round(base, 2),
            "total_pnl": round(pnl, 2),
            "pct_impact": round(pnl / base * 100, 2) if base else 0,
            "value_after": round(base + pnl, 2),
        }

    return {
        "scenario_name": req.name or "custom",
        "summary_by_currency": summary,
        "affected_positions": sorted(rows, key=lambda r: r["pnl"]),
        "shocks_applied": [s.dict() for s in req.shocks],
    }


# ---- P1: /api/backtest — query existing results + trigger new run ----


@app.get("/api/backtest")
def backtest_query(symbol: Optional[str] = None, strategy: Optional[str] = None,
                    min_sharpe: Optional[float] = None, limit: int = 20) -> dict:
    """Browse backtest results from SQLite.

    Returns top-N results sorted by Sharpe; filterable by symbol/strategy.
    Backtest worker writes here continuously (24/7 background service).
    """
    where = []
    params: list = []
    if symbol:
        where.append("t.symbol = ?")
        params.append(symbol.upper())
    if strategy:
        where.append("t.strategy = ?")
        params.append(strategy)
    if min_sharpe is not None:
        where.append("r.sharpe IS NOT NULL AND r.sharpe >= ?")
        params.append(min_sharpe)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    with db.conn() as c:
        rows = c.execute(
            "SELECT t.id, t.strategy, t.symbol, t.params_json, t.period_years, "
            "       t.status, t.finished_at, "
            "       r.total_return, r.annual_return, r.sharpe, r.sortino, "
            "       r.max_drawdown, r.win_rate, r.n_trades, r.profit_factor "
            "FROM backtest_tasks t LEFT JOIN backtest_results r ON r.task_id = t.id "
            f"{where_sql} ORDER BY r.sharpe DESC NULLS LAST LIMIT ?",
            (*params, limit),
        ).fetchall()
        results = [dict(r) for r in rows]

    # Decode params_json
    for r in results:
        try:
            r["params"] = json.loads(r.pop("params_json")) if r.get("params_json") else {}
        except Exception:
            r["params"] = {}

    return {
        "n_results": len(results),
        "filters": {"symbol": symbol, "strategy": strategy, "min_sharpe": min_sharpe},
        "results": results,
    }


class BacktestRunRequest(BaseModel):
    strategy: str
    symbol: str
    params: dict
    period_years: int = 5
    walk_forward: bool = False


@app.post("/api/backtest/run")
def backtest_run(req: BacktestRunRequest) -> dict:
    """Run a backtest synchronously (returns metrics directly).

    Use walk_forward=true for K-fold OOS validation — required to trust
    a high in-sample Sharpe before committing real money to the strategy.
    """
    from . import backtest as bt
    try:
        if req.walk_forward:
            return bt.walk_forward(req.strategy, req.symbol, req.params,
                                     period_years=req.period_years)
        else:
            return bt.run(req.strategy, req.symbol, req.params,
                           period_years=req.period_years)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"backtest failed: {e}")


# ---- P2: /api/events — browse newswatch-recorded events ----


@app.get("/api/events")
def events_endpoint(symbol: Optional[str] = None,
                     min_severity: int = 4,
                     since_days: int = 7,
                     limit: int = 50) -> dict:
    """Browse events recorded by newswatch (severity ≥ min_severity, last N days).

    Each event has LLM-graded severity (0-10), category, and per-holding
    impact_json. Use to ask "AMD 最近什么大新闻" or "最近一周 sev>=7 的全市场事件".
    """
    where = ["severity >= ?", "fired_at >= datetime('now', ?)"]
    params: list = [min_severity, f"-{int(since_days)} days"]
    if symbol:
        where.append("(affected_symbols LIKE ? OR affected_symbols LIKE ? OR affected_symbols LIKE ?)")
        params.extend([f"{symbol},%", f"%,{symbol},%", f"%,{symbol}"])

    with db.conn() as c:
        rows = c.execute(
            "SELECT id, severity, category, summary, affected_symbols, "
            "       fired_at, pushed_at, impact_json "
            "FROM events WHERE " + " AND ".join(where) +
            " ORDER BY fired_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

    out = []
    for r in rows:
        ev = dict(r)
        try:
            ev["impact"] = json.loads(ev.pop("impact_json")) if ev.get("impact_json") else {}
        except Exception:
            ev["impact"] = {}
        ev["affected"] = [s for s in (ev.pop("affected_symbols") or "").split(",") if s]
        out.append(ev)

    return {
        "n_events": len(out),
        "filters": {"symbol": symbol, "min_severity": min_severity,
                     "since_days": since_days},
        "events": out,
    }


# ---- P2: /api/audit — LLM call usage / cost / latency ----


@app.get("/api/audit")
def audit_endpoint(days: int = 7) -> dict:
    """LLM usage summary across all backends — call count, tokens, cost, latency.

    Wraps db.llm_audit_summary so 阿雷 can answer "今天 LLM 调用慢吗" and
    "用了多少钱". Cost USD comes from config/llm_routes.yaml `costs:` table.
    """
    summary = db.llm_audit_summary(days=days)

    # Latest 10 errors so 阿雷 can investigate spikes
    with db.conn() as c:
        errors = [dict(r) for r in c.execute(
            "SELECT ts, backend, task, caller, error FROM llm_audit "
            "WHERE success = 0 AND ts >= datetime('now', ?) "
            "ORDER BY ts DESC LIMIT 10",
            (f"-{int(days)} days",),
        ).fetchall()]
    summary["recent_errors"] = errors
    return summary


# ---- P2: /api/history — historical-pattern lookup ----


@app.get("/api/history/move")
def history_move(symbol: str, threshold_pct: float = 15.0,
                  window_days: int = 5, lookback_years: int = 5,
                  followup_days: int = 20) -> dict:
    """Find historical instances where `symbol` moved >= threshold within window.

    Answers: "AMD 历史上涨这么猛后, 隔多久跌多少". For each match, computes
    the subsequent followup_days returns to surface the empirical base rate
    of "what comes after a +15% rally".

    Params:
      threshold_pct:  minimum cumulative move (% in `window_days`)
      window_days:    rolling window to detect the move
      lookback_years: how far back to scan
      followup_days:  per-match forward window for "what happened next"
    """
    df = fetcher.load_local(symbol)
    if df.empty:
        raise HTTPException(404, f"no price data for {symbol}")
    import pandas as pd

    df = df.tail(int(252 * lookback_years))
    if len(df) < window_days + followup_days + 5:
        raise HTTPException(400, f"insufficient history for {symbol}")

    closes = df["close"].astype(float)
    rolling_chg = (closes / closes.shift(window_days) - 1) * 100

    matches = []
    last_match_idx: int | None = None
    threshold = abs(threshold_pct)
    direction = 1 if threshold_pct >= 0 else -1

    for i, (date, chg) in enumerate(rolling_chg.items()):
        if pd.isna(chg):
            continue
        # Direction-aware match
        is_match = (direction == 1 and chg >= threshold) or (direction == -1 and chg <= -threshold)
        if not is_match:
            continue
        # Cool-down: don't double-count overlapping windows
        if last_match_idx is not None and i - last_match_idx < window_days:
            continue
        last_match_idx = i

        # Forward returns
        end_price = closes.iloc[i]
        forward = {}
        for d in (1, 5, 10, 20):
            if d > followup_days:
                break
            j = i + d
            if j >= len(closes):
                continue
            fwd_chg = (closes.iloc[j] / end_price - 1) * 100
            forward[f"forward_{d}d_pct"] = round(float(fwd_chg), 2)

        # Max drawdown over followup window
        end_idx = min(i + followup_days, len(closes) - 1)
        if end_idx > i:
            future = closes.iloc[i:end_idx + 1]
            future_peak = future.cummax()
            future_dd = ((future - future_peak) / future_peak * 100).min()
        else:
            future_dd = 0.0

        matches.append({
            "date": str(date)[:10],
            "rally_window_chg_pct": round(float(chg), 2),
            "price_at_event": round(float(end_price), 2),
            **forward,
            "max_drawdown_within_followup_pct": round(float(future_dd), 2),
        })

    # Aggregate stats
    if matches:
        import statistics
        agg = {}
        for k in ("forward_1d_pct", "forward_5d_pct", "forward_10d_pct",
                   "forward_20d_pct", "max_drawdown_within_followup_pct"):
            vals = [m[k] for m in matches if k in m]
            if vals:
                agg[k] = {
                    "median": round(statistics.median(vals), 2),
                    "mean": round(statistics.mean(vals), 2),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "n": len(vals),
                }
    else:
        agg = {}

    return {
        "symbol": symbol,
        "params": {"threshold_pct": threshold_pct, "window_days": window_days,
                    "lookback_years": lookback_years, "followup_days": followup_days},
        "n_matches": len(matches),
        "aggregates": agg,
        "matches": matches,
    }


# ---- P3: /api/pnl — arbitrary-window P&L attribution ----


@app.get("/api/pnl")
def pnl_endpoint(start: str, end: Optional[str] = None,
                  groupby: str = "symbol") -> dict:
    """Per-holding P&L over [start, end], groupable by symbol or theme.

    NOTE: Computes paper P&L assuming current shares held throughout the
    window — i.e. ignores when each lot was actually bought. For true
    cost-basis P&L use tax_lots (when wired).

    Params:
      start:    YYYY-MM-DD (inclusive)
      end:      YYYY-MM-DD (defaults to most recent close)
      groupby:  "symbol" | "theme"
    """
    import pandas as pd
    if groupby not in ("symbol", "theme"):
        raise HTTPException(400, "groupby must be 'symbol' or 'theme'")
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else None
    except Exception:
        raise HTTPException(400, "invalid date format (expected YYYY-MM-DD)")

    portfolio = cfg_mod.load("portfolio")
    held = portfolio.get("positions", {})
    rows = []
    by_group: dict[str, dict] = {}
    by_ccy: dict[str, dict] = {}

    for sym, info in held.items():
        df = fetcher.load_local(sym)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index)
        # Resolve start_price: closest trading day at or after start
        in_range = df[df.index >= start_ts]
        if in_range.empty:
            continue
        start_price = float(in_range["close"].iloc[0])
        actual_start_date = in_range.index[0]
        # End price
        if end_ts is None:
            end_price = float(df["close"].iloc[-1])
            actual_end_date = df.index[-1]
        else:
            up_to = df[df.index <= end_ts]
            if up_to.empty:
                continue
            end_price = float(up_to["close"].iloc[-1])
            actual_end_date = up_to.index[-1]

        shares = info["shares"]
        ccy = info.get("currency", "USD")
        theme = info.get("theme", "uncategorized")
        pnl = (end_price - start_price) * shares
        ret_pct = (end_price / start_price - 1) * 100 if start_price else 0
        start_value = start_price * shares
        end_value = end_price * shares

        rows.append({
            "symbol": sym,
            "theme": theme,
            "currency": ccy,
            "shares": shares,
            "start_date": str(actual_start_date)[:10],
            "end_date": str(actual_end_date)[:10],
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "start_value": round(start_value, 2),
            "end_value": round(end_value, 2),
            "pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2),
        })

        # Group aggregation
        key = sym if groupby == "symbol" else theme
        g = by_group.setdefault(key, {"start_value": 0, "end_value": 0,
                                       "pnl": 0, "currency": ccy})
        g["start_value"] += start_value
        g["end_value"] += end_value
        g["pnl"] += pnl
        # Currency totals
        cb = by_ccy.setdefault(ccy, {"start": 0, "end": 0, "pnl": 0})
        cb["start"] += start_value
        cb["end"] += end_value
        cb["pnl"] += pnl

    # Round + add return_pct to group aggregates
    grouped = []
    for key, g in by_group.items():
        ret = (g["end_value"] / g["start_value"] - 1) * 100 if g["start_value"] else 0
        grouped.append({
            "key": key,
            "currency": g["currency"],
            "start_value": round(g["start_value"], 2),
            "end_value": round(g["end_value"], 2),
            "pnl": round(g["pnl"], 2),
            "return_pct": round(ret, 2),
        })
    grouped.sort(key=lambda x: -x["pnl"])

    summary = {}
    for ccy, b in by_ccy.items():
        ret = (b["end"] / b["start"] - 1) * 100 if b["start"] else 0
        summary[ccy] = {
            "start_value": round(b["start"], 2),
            "end_value": round(b["end"], 2),
            "pnl": round(b["pnl"], 2),
            "return_pct": round(ret, 2),
        }

    return {
        "start": start,
        "end": end or "latest",
        "groupby": groupby,
        "summary_by_currency": summary,
        "groups": grouped,
        "details": rows,
    }


# ---- P2: /api/alerts — user-defined price/condition alerts ----


_VALID_OPS = {"<", "<=", ">", ">=", "cross_below", "cross_above"}
_VALID_BASIS = {"last", "rsi", "ma20", "ma50", "ma200", "chg_1d_pct", "chg_20d_pct"}


class AlertCreateRequest(BaseModel):
    symbol: str
    op: str                # < | <= | > | >= | cross_below | cross_above
    value: Optional[float] = None
    basis: str = "last"
    note: Optional[str] = None
    cooldown_minutes: int = 60


@app.get("/api/alerts")
def alerts_list(symbol: Optional[str] = None,
                 enabled_only: bool = False) -> dict:
    """List all user-defined alerts."""
    where = []
    params: list = []
    if symbol:
        where.append("symbol = ?")
        params.append(symbol.upper())
    if enabled_only:
        where.append("enabled = 1")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    with db.conn() as c:
        rows = [dict(r) for r in c.execute(
            f"SELECT * FROM user_alerts {where_sql} ORDER BY id DESC", params
        ).fetchall()]
    return {"n_alerts": len(rows), "alerts": rows}


@app.post("/api/alerts")
def alerts_create(req: AlertCreateRequest) -> dict:
    """Create a new alert.

    Examples:
      "AMD 跌到 380":   {symbol:AMD, op:'<=',          value:380,  basis:'last'}
      "GRID RSI > 80":  {symbol:GRID, op:'>',           value:80,   basis:'rsi'}
      "SOXX 跌破 MA50": {symbol:SOXX, op:'cross_below', value:null, basis:'ma50'}
    """
    if req.op not in _VALID_OPS:
        raise HTTPException(400, f"op must be one of {sorted(_VALID_OPS)}")
    if req.basis not in _VALID_BASIS:
        raise HTTPException(400, f"basis must be one of {sorted(_VALID_BASIS)}")
    sym = req.symbol.strip().upper()

    # Validate value requirement: ops other than cross_* need a numeric threshold
    if req.op in ("<", "<=", ">", ">=") and req.value is None:
        raise HTTPException(400, f"op {req.op} requires a value")

    with db.conn() as c:
        cur = c.execute(
            "INSERT INTO user_alerts(symbol, op, value, basis, note, "
            "                        cooldown_minutes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sym, req.op, req.value, req.basis, req.note,
             req.cooldown_minutes, datetime.utcnow().isoformat()),
        )
        new_id = cur.lastrowid
        row = dict(c.execute("SELECT * FROM user_alerts WHERE id=?",
                                (new_id,)).fetchone())
    return {"ok": True, "alert": row}


@app.delete("/api/alerts/{alert_id}")
def alerts_delete(alert_id: int) -> dict:
    with db.conn() as c:
        cur = c.execute("DELETE FROM user_alerts WHERE id=?", (alert_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, f"alert {alert_id} not found")
    return {"ok": True, "deleted_id": alert_id}


@app.patch("/api/alerts/{alert_id}")
def alerts_toggle(alert_id: int, enabled: bool) -> dict:
    """Toggle enabled. Use ?enabled=true|false."""
    with db.conn() as c:
        cur = c.execute("UPDATE user_alerts SET enabled=? WHERE id=?",
                          (1 if enabled else 0, alert_id))
        if cur.rowcount == 0:
            raise HTTPException(404, f"alert {alert_id} not found")
        row = dict(c.execute("SELECT * FROM user_alerts WHERE id=?",
                                (alert_id,)).fetchone())
    return {"ok": True, "alert": row}


@app.post("/api/alerts/scan")
def alerts_scan(dry_run: bool = False) -> dict:
    """Trigger an immediate scan of all enabled alerts (mostly for testing).

    Production scanning happens via systemd timer (quant-alerts.timer)
    every 5 minutes during market hours.
    """
    from . import alert_scanner
    return alert_scanner.scan_once(dry_run=dry_run)


# ---- Tax lots: cost-basis tracking, FIFO sell, harvest, wash sale ----


class TaxLotOpen(BaseModel):
    symbol: str
    shares: float
    price: float
    acquired_at: str           # YYYY-MM-DD
    currency: str = "USD"
    notes: Optional[str] = None


class TaxLotSell(BaseModel):
    symbol: str
    shares: float
    price: float
    sold_at: str               # YYYY-MM-DD
    method: str = "FIFO"       # FIFO | LIFO | HIFO


@app.post("/api/tax/lot")
def tax_lot_open(req: TaxLotOpen) -> dict:
    """Record a buy lot. Idempotent on (symbol, acquired_at, shares, price)
    only by ID — submitting duplicates creates separate lots."""
    from . import tax
    try:
        return {"ok": True, "lot": tax.open_lot(
            symbol=req.symbol, shares=req.shares, price=req.price,
            acquired_at=req.acquired_at, currency=req.currency, notes=req.notes,
        )}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/tax/sell")
def tax_lot_sell(req: TaxLotSell) -> dict:
    """Sell shares; FIFO/LIFO/HIFO match against open lots, return realized P&L.

    Wash sale detection runs automatically — disallowed losses are flagged
    in `wash_sale_warnings` so 阿雷 can warn before submission.
    """
    from . import tax
    try:
        return tax.sell(symbol=req.symbol, shares=req.shares, price=req.price,
                          sold_at=req.sold_at, method=req.method)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/tax/lots")
def tax_lots_list(symbol: Optional[str] = None, open_only: bool = False) -> dict:
    from . import tax
    rows = tax.list_lots(symbol=symbol, open_only=open_only)
    return {"n_lots": len(rows), "lots": rows}


@app.get("/api/expectations")
def expectations_endpoint(symbol: str, horizon_days: int = 5,
                            include_history: bool = False) -> dict:
    """Latest expected forward-return distribution for a symbol/horizon.

    Returns the bootstrap empirical distribution (mean/sigma/percentiles) +
    optionally the recent snapshot history. Use to evaluate whether the
    current realized move is statistically surprising vs the prior.
    """
    from . import expectations as exp
    latest = exp.get_latest(symbol, horizon_days=horizon_days)
    if latest is None:
        raise HTTPException(404, f"no expectation for {symbol} h={horizon_days}")

    out: dict = {"symbol": symbol, "horizon_days": horizon_days, "latest": latest}

    # Compute realized return so far if we have a fresh-enough close
    df = fetcher.load_local(symbol)
    if not df.empty:
        import pandas as pd
        df.index = pd.to_datetime(df.index)
        anchor_date = pd.Timestamp(latest["snapshot_date"])
        future = df[df.index > anchor_date]
        if not future.empty:
            current_close = float(future["close"].iloc[-1])
            realized_pct = (current_close / latest["anchor_close"] - 1) * 100
            sigma = latest["sigma_pct"]
            z_score = (realized_pct - latest["mean_pct"]) / sigma if sigma else 0
            tail_flag = (
                "outside_p95" if realized_pct > latest["p95_pct"]
                else "outside_p5" if realized_pct < latest["p5_pct"]
                else "in_band"
            )
            out["realized_so_far"] = {
                "days_elapsed": int((future.index[-1] - anchor_date).days),
                "current_close": round(current_close, 4),
                "realized_pct": round(realized_pct, 3),
                "z_score_vs_mean": round(z_score, 2),
                "tail_flag": tail_flag,
            }

    if include_history:
        out["history"] = exp.history(symbol, horizon_days=horizon_days, limit=60)
    return out


@app.get("/api/altdata/bilibili")
def altdata_bilibili(keyword: str, trend: bool = False) -> dict:
    """B 站搜索量 / 二创视频领先指标. 用 trend=true 看 7d/30d 变化."""
    from .alt_data import bilibili
    if trend:
        return bilibili.trend(keyword)
    try:
        m = bilibili.snapshot_keyword(keyword)
        bilibili.store_snapshot(keyword, m)
        return {"keyword": keyword, "metrics": m}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"bilibili fetch failed: {e}")


@app.get("/api/tax/harvest")
def tax_harvest() -> dict:
    """Tax-loss harvesting candidates + lots approaching long-term + LT gains.

    Returns three categorized lists:
      harvest_candidates_loss  — open lots underwater (sell to realize loss)
      approaching_long_term    — within 30d of LT threshold (wait, then sell)
      long_term_at_gain        — held > 1y + green (flexible to realize)
    """
    from . import tax
    portfolio = cfg_mod.load("portfolio")
    held = portfolio.get("positions", {})
    # Build current price dict from local parquet
    prices: dict[str, float] = {}
    for sym in held:
        df = fetcher.load_local(sym)
        if not df.empty:
            prices[sym] = float(df["close"].iloc[-1])
    # Also include any symbol that has open lots but isn't in portfolio
    for lot in tax.list_lots(open_only=True):
        if lot["symbol"] not in prices:
            df = fetcher.load_local(lot["symbol"])
            if not df.empty:
                prices[lot["symbol"]] = float(df["close"].iloc[-1])
    return tax.harvest_candidates(prices)
