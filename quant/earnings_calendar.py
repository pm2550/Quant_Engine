"""未来 30 天财报日历 + 历史业绩超预期/不及预期记录。"""
from __future__ import annotations
import logging
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd

from . import config as cfg_mod
from . import db, fetcher

log = logging.getLogger(__name__)


def _us_earnings(symbol: str) -> list[dict]:
    import yfinance as yf
    out: list[dict] = []
    try:
        t = yf.Ticker(symbol)

        # 1. Historical reported earnings
        try:
            df = t.get_earnings_dates(limit=8) if hasattr(t, "get_earnings_dates") else None
            if df is not None and not df.empty:
                for idx, row in df.iterrows():
                    report_date = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
                    out.append({
                        "report_date": report_date,
                        "fiscal_period": None,
                        "eps_estimate": float(row["EPS Estimate"]) if pd.notna(row.get("EPS Estimate")) else None,
                        "eps_actual": float(row["Reported EPS"]) if pd.notna(row.get("Reported EPS")) else None,
                        "surprise_pct": float(row["Surprise(%)"]) if pd.notna(row.get("Surprise(%)")) else None,
                    })
        except Exception as e:  # noqa: BLE001
            log.debug("get_earnings_dates %s: %s", symbol, e)

        # 2. UPCOMING earnings via calendar (this is the critical missing piece)
        try:
            cal = t.calendar
            if cal:
                dates = cal.get("Earnings Date") or []
                if not isinstance(dates, list):
                    dates = [dates]
                for d in dates:
                    if not d:
                        continue
                    rd = d.isoformat() if hasattr(d, "isoformat") else str(d)[:10]
                    if any(o["report_date"] == rd for o in out):
                        continue   # already in historical
                    out.append({
                        "report_date": rd,
                        "fiscal_period": "upcoming",
                        "eps_estimate": cal.get("Earnings Average"),
                        "eps_actual": None,
                        "revenue_estimate": cal.get("Revenue Average"),
                        "revenue_actual": None,
                        "eps_high": cal.get("Earnings High"),
                        "eps_low": cal.get("Earnings Low"),
                    })
        except Exception as e:  # noqa: BLE001
            log.debug("calendar %s: %s", symbol, e)
    except Exception as e:  # noqa: BLE001
        log.warning("yf %s failed: %s", symbol, e)
    return out


def _cn_earnings(symbol: str) -> list[dict]:
    """A-share 业绩预告/披露 via akshare."""
    import akshare as ak
    code = symbol.split(".")[0]
    out: list[dict] = []
    # 业绩预告 (stock_yjyg_em) - quarterly hints
    try:
        df = ak.stock_yjyg_em(date="20260331")  # latest reporting period
        if df is not None and not df.empty:
            row = df[df.get("股票代码", "") == code]
            if not row.empty:
                r = row.iloc[0]
                out.append({
                    "report_date": str(r.get("公告日期", date.today()))[:10],
                    "fiscal_period": "业绩预告",
                    "eps_estimate": None,
                    "eps_actual": None,
                    "surprise_pct": float(r.get("预测净利润-同比变动", 0)) if pd.notna(r.get("预测净利润-同比变动")) else None,
                })
    except Exception as e:  # noqa: BLE001
        log.debug("CN earnings %s: %s", symbol, e)
    return out


def fetch_one(symbol: str) -> list[dict]:
    return _cn_earnings(symbol) if fetcher.is_a_share(symbol) else _us_earnings(symbol)


def store(symbol: str, items: list[dict]) -> int:
    n = 0
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        for it in items:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO earnings_calendar
                    (symbol, fiscal_period, report_date, eps_estimate, eps_actual,
                     revenue_estimate, revenue_actual, surprise_pct, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (symbol, it.get("fiscal_period"), it["report_date"],
                     it.get("eps_estimate"), it.get("eps_actual"),
                     it.get("revenue_estimate"), it.get("revenue_actual"),
                     it.get("surprise_pct"),
                     datetime.utcnow().isoformat() + "Z"),
                )
                n += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
    return n


def upcoming(days_ahead: int = 30) -> list[dict]:
    """Next N days of earnings for held + watchlist symbols."""
    today = date.today().isoformat()
    end = (date.today() + timedelta(days=days_ahead)).isoformat()
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM earnings_calendar WHERE report_date BETWEEN ? AND ? ORDER BY report_date",
            (today, end),
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
            items = fetch_one(sym)
            if not items:
                out["skipped"].append(sym)
                continue
            n = store(sym, items)
            out["success"].append(f"{sym}: {n} dates")
        except Exception as e:  # noqa: BLE001
            log.exception("earnings %s: %s", sym, e)
            out["failed"].append(f"{sym}: {e}")
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--upcoming", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.upcoming:
        print(json.dumps(upcoming(args.days), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(refresh_all(args.symbols), indent=2, ensure_ascii=False))
