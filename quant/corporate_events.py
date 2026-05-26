"""公司行为事件: 除权除息日 / 分红 / 股票分拆.

数据源:
  - 美股: yfinance.calendar (Ex-Dividend Date) + yfinance.dividends (历史)
  - A 股: akshare stock_dividend_cninfo (分红配股记录)
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config as cfg_mod, db, fetcher

log = logging.getLogger(__name__)


def _store(symbol: str, event_type: str, event_date: str,
           amount: float | None = None, notes: str = "") -> bool:
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO corporate_events
                (symbol, event_type, event_date, amount, notes, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (symbol, event_type, event_date, amount, notes,
                 datetime.utcnow().isoformat() + "Z"),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def _us_corporate_events(symbol: str) -> int:
    """yfinance: ex-dividend (next), historical dividends, splits."""
    import yfinance as yf
    n = 0
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar or {}

        # Next ex-dividend date
        ex_div = cal.get("Ex-Dividend Date")
        if ex_div:
            d_str = ex_div.isoformat() if hasattr(ex_div, "isoformat") else str(ex_div)[:10]
            # Get last dividend amount (proxy for upcoming if not stated)
            divs = t.dividends
            last_amt = float(divs.iloc[-1]) if divs is not None and not divs.empty else None
            if _store(symbol, "ex_dividend", d_str, last_amt,
                      "next ex-div (estimated amount = last dividend)"):
                n += 1

        # Recent dividend history (last 4)
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                for idx, amt in divs.tail(4).items():
                    d_str = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                    if _store(symbol, "ex_dividend", d_str, float(amt), "historical"):
                        n += 1
        except Exception:
            pass

        # Splits
        try:
            spl = t.splits
            if spl is not None and not spl.empty:
                for idx, ratio in spl.tail(2).items():
                    d_str = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                    if _store(symbol, "split", d_str, float(ratio), f"{ratio}-for-1 split"):
                        n += 1
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        log.warning("US corp events %s: %s", symbol, e)
    return n


def _cn_corporate_events(symbol: str) -> int:
    """akshare A-share dividend/split records via stock_fhps_em (分红派息)."""
    import akshare as ak
    code = symbol.split(".")[0]
    n = 0
    try:
        df = ak.stock_fhps_em(symbol=code)
        if df is None or df.empty:
            return 0
        for _, row in df.head(8).iterrows():
            ex_d = row.get("除权除息日") or row.get("分红派息日")
            if not ex_d or pd.isna(ex_d):
                continue
            d_str = str(ex_d)[:10]
            try:
                amt_str = row.get("现金分红-现金分红比例") or row.get("现金分红")
                amt = float(str(amt_str).replace(",", "")) if amt_str else None
            except (ValueError, TypeError):
                amt = None
            notes = str(row.get("分配方案", ""))[:200]
            if _store(symbol, "ex_dividend", d_str, amt, notes):
                n += 1
    except Exception as e:  # noqa: BLE001
        log.debug("CN corp events %s: %s", symbol, e)
    return n


def fetch_one(symbol: str) -> int:
    return (_cn_corporate_events(symbol) if fetcher.is_a_share(symbol)
            else _us_corporate_events(symbol))


def upcoming(*, days: int = 14, types: list[str] | None = None) -> list[dict]:
    types = types or ["ex_dividend", "split"]
    today = date.today()
    end = today + timedelta(days=days)
    placeholders = ",".join("?" * len(types))
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""SELECT * FROM corporate_events
            WHERE event_date BETWEEN ? AND ? AND event_type IN ({placeholders})
            ORDER BY event_date""",
            [today.isoformat(), end.isoformat()] + types,
        ).fetchall()
    return [dict(r) for r in rows]


def refresh_all(symbols: list[str] | None = None) -> dict:
    db.init()
    if symbols is None:
        portfolio = cfg_mod.load("portfolio")
        symbols = cfg_mod.all_symbols(portfolio)
    out = {"success": [], "skipped": [], "failed": []}
    for sym in symbols:
        try:
            n = fetch_one(sym)
            if n > 0:
                out["success"].append(f"{sym}: {n} events")
            else:
                out["skipped"].append(sym)
        except Exception as e:  # noqa: BLE001
            log.exception("corp events %s: %s", sym, e)
            out["failed"].append(f"{sym}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--upcoming", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.upcoming:
        print(json.dumps(upcoming(days=args.days), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(refresh_all(args.symbols), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
