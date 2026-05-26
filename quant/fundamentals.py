"""基本面快照: PE/PB/ROE/营收增速/市值. 美股 yfinance + A 股 akshare."""
from __future__ import annotations
import json
import logging
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import config as cfg_mod
from . import db, fetcher

log = logging.getLogger(__name__)


def _us_fundamentals(symbol: str) -> dict | None:
    """yfinance.info — US stocks/ETFs. ETFs have less data than individual stocks."""
    import yfinance as yf
    try:
        info = yf.Ticker(symbol).info
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance info failed for %s: %s", symbol, e)
        return None
    if not info or "symbol" not in info:
        return None
    return {
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "revenue_yoy": info.get("revenueGrowth"),  # decimal, e.g. 0.12 = +12%
        "eps_yoy": info.get("earningsGrowth"),
        "market_cap": info.get("marketCap"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "extra": {
            "trailing_eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "type": info.get("quoteType"),
        },
    }


def _cn_fundamentals(symbol: str) -> dict | None:
    """akshare A-share fundamentals. symbol like '002624.SZ'."""
    import akshare as ak
    code = symbol.split(".")[0]
    out: dict = {"extra": {}}

    # 1. 个股基本信息 (市值/股本/行业)
    try:
        info = ak.stock_individual_info_em(symbol=code)
        if info is not None and not info.empty:
            kv = dict(zip(info["item"].astype(str), info["value"].astype(str)))
            try:
                if kv.get("总市值"):
                    out["market_cap"] = float(str(kv["总市值"]).replace(",", ""))
            except (ValueError, AttributeError):
                pass
            try:
                if kv.get("总股本"):
                    out["shares_outstanding"] = float(str(kv["总股本"]).replace(",", ""))
            except (ValueError, AttributeError):
                pass
            out["extra"]["industry"] = kv.get("行业")
            out["extra"]["name_cn"] = kv.get("股票简称")
    except Exception as e:  # noqa: BLE001
        log.warning("stock_individual_info_em failed for %s: %s", code, e)

    # 2. PE/PB/PS 历史 + 当前分位 (Baidu valuation)
    for indicator, key, pct_key in [
        ("市盈率(TTM)", "pe", "pe_pct_5y"),
        ("市净率",      "pb", "pb_pct_5y"),
        ("市销率",      "ps", "ps_pct_5y"),
    ]:
        try:
            df = ak.stock_zh_valuation_baidu(symbol=code, indicator=indicator, period="近五年")
            if df is not None and not df.empty:
                df = df.dropna(subset=["value"])
                if df.empty:
                    continue
                last = float(df["value"].iloc[-1])
                out[key] = last
                pct = float((df["value"] <= last).mean())
                out["extra"][pct_key] = round(pct, 3)
        except Exception as e:  # noqa: BLE001
            log.debug("baidu valuation %s/%s: %s", code, indicator, e)

    # 3. ROE / 营收同比 / 利润同比 (THS 财务摘要)
    try:
        df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if df is not None and not df.empty:
            # latest column = newest period
            latest_col = df.columns[-1]
            row = df.set_index(df.columns[0])[latest_col]

            def _pct(s):
                if isinstance(s, str) and s.endswith("%"):
                    try:
                        return float(s.rstrip("%")) / 100
                    except ValueError:
                        return None
                return None

            # ROE (净资产收益率)
            roe_str = row.get("净资产收益率")
            out["roe"] = _pct(roe_str) if roe_str else None
            # 营收同比 / 净利润同比
            out["revenue_yoy"] = _pct(row.get("营业总收入同比增长率"))
            out["eps_yoy"] = _pct(row.get("净利润同比增长率"))
            out["extra"]["latest_period"] = str(latest_col)
            out["extra"]["gross_margin"] = _pct(row.get("销售毛利率"))
            out["extra"]["net_margin"] = _pct(row.get("销售净利率"))
            out["extra"]["debt_ratio"] = _pct(row.get("资产负债率"))
    except Exception as e:  # noqa: BLE001
        log.debug("ths abstract %s: %s", code, e)

    return out if (out.get("pe") or out.get("market_cap")) else None


def fetch_one(symbol: str) -> dict | None:
    if fetcher.is_a_share(symbol):
        return _cn_fundamentals(symbol)
    return _us_fundamentals(symbol)


def store(symbol: str, data: dict) -> None:
    today = date.today().isoformat()
    extra = data.pop("extra", {}) if "extra" in data else {}
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO fundamentals
            (symbol, as_of, pe, pb, ps, roe, roa, revenue_yoy, eps_yoy,
             market_cap, shares_outstanding, extra_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                symbol, today,
                data.get("pe"), data.get("pb"), data.get("ps"),
                data.get("roe"), data.get("roa"),
                data.get("revenue_yoy"), data.get("eps_yoy"),
                data.get("market_cap"), data.get("shares_outstanding"),
                json.dumps(extra, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()


def latest(symbol: str) -> dict | None:
    """Most recent fundamentals row for symbol, with extra unpacked."""
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM fundamentals WHERE symbol=? ORDER BY as_of DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    if not row:
        return None
    out = dict(row)
    if out.get("extra_json"):
        try:
            out["extra"] = json.loads(out["extra_json"])
        except json.JSONDecodeError:
            out["extra"] = {}
    out.pop("extra_json", None)
    return out


def refresh_all(*, symbols: list[str] | None = None) -> dict:
    """Refresh fundamentals for portfolio + watchlist (or given list).

    Returns: {success: [...], skipped: [...], failed: [...]}
    """
    db.init()
    if symbols is None:
        portfolio = cfg_mod.load("portfolio")
        symbols = cfg_mod.all_symbols(portfolio)
    out = {"success": [], "skipped": [], "failed": []}
    for sym in symbols:
        try:
            data = fetch_one(sym)
            if not data:
                out["skipped"].append(sym)
                continue
            # ETFs from yfinance often have only some fields; that's fine, store partial
            if all(data.get(k) is None for k in ("pe", "pb", "market_cap")):
                out["skipped"].append(sym)
                continue
            store(sym, data)
            out["success"].append(sym)
            log.info("fundamentals %s: PE=%s PB=%s mkt=%s",
                     sym, data.get("pe"), data.get("pb"), data.get("market_cap"))
            time.sleep(0.5)  # gentle rate limit
        except Exception as e:  # noqa: BLE001
            log.exception("fundamentals failed for %s: %s", sym, e)
            out["failed"].append(f"{sym}: {e}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", help="specific symbols (default: portfolio + watchlist)")
    ap.add_argument("--show", help="show latest fundamentals for one symbol")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.show:
        out = latest(args.show)
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    else:
        out = refresh_all(symbols=args.symbols)
        print(json.dumps(out, indent=2, ensure_ascii=False))
