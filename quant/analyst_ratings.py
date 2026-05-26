"""卖方分析师一致预期 + 评级 + 目标价.

支持三种市场:
  - 美股: yfinance info (targetMeanPrice / recommendationMean)
  - A 股: akshare stock_research_report_em (评级分布 → 数值映射 1=买入 .. 5=卖出)
  - ETF:  yfinance funds_data.top_holdings 加权 top-10 成分股的 rec_mean 和 upside

统一输出字段 (各 market 子集填充):
  market, recommendation_mean (1=strong buy, 5=strong sell), recommendation_key,
  target_mean_price, target_high_price, target_low_price, target_median_price,
  number_of_analyst_opinions, current_price, upside_pct,
  recent_research (CN), recent_changes (US), holdings (ETF),
  weighted_target_upside_pct (ETF only), rating_breakdown (CN only).
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, date

import pandas as pd

from . import config as cfg_mod
from . import db, fetcher

log = logging.getLogger(__name__)

# A-share 评级 → 数值 (与 yfinance recommendationMean 同向: 1=最看好, 5=最看空)
CN_RATING_TO_NUMERIC: dict[str, float] = {
    "买入": 1.0,
    "强烈推荐": 1.0,
    "推荐": 1.5,
    "增持": 2.0,
    "谨慎增持": 2.5,
    "中性": 3.0,
    "持有": 3.0,
    "观望": 3.0,
    "减持": 4.0,
    "谨慎减持": 4.0,
    "卖出": 5.0,
}


def _us_ratings(symbol: str) -> dict | None:
    """yfinance recommendations + price targets (适用美股个股)."""
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance %s failed: %s", symbol, e)
        return None

    out: dict = {
        "market": "us",
        "target_mean_price": info.get("targetMeanPrice"),
        "target_high_price": info.get("targetHighPrice"),
        "target_low_price": info.get("targetLowPrice"),
        "target_median_price": info.get("targetMedianPrice"),
        "recommendation_mean": info.get("recommendationMean"),     # 1 strong buy, 5 strong sell
        "recommendation_key": info.get("recommendationKey"),       # 'buy'/'hold'/etc
        "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
    }
    cur_price = info.get("regularMarketPrice") or info.get("currentPrice")
    if cur_price and out.get("target_mean_price"):
        out["upside_pct"] = round((out["target_mean_price"] / cur_price - 1) * 100, 2)
    out["current_price"] = cur_price

    # Recent analyst rating changes
    try:
        rec = t.recommendations
        if rec is not None and not rec.empty:
            tail = rec.tail(5)
            out["recent_changes"] = [
                {"date": str(idx)[:10], "firm": r.get("Firm"),
                 "to_grade": r.get("To Grade"), "from_grade": r.get("From Grade"),
                 "action": r.get("Action")}
                for idx, r in tail.iterrows() if pd.notna(r.get("To Grade"))
            ][:5]
    except Exception:
        out["recent_changes"] = []

    return out


def _cn_ratings(symbol: str) -> dict | None:
    """A 股研报评级分布 via akshare stock_research_report_em.

    注意 akshare 已不返回机构目标价列, 因此 A 股没有 target_mean_price.
    `recommendation_mean` 由 `东财评级` 字段经 CN_RATING_TO_NUMERIC 映射后均值得出.
    """
    import akshare as ak
    code = symbol.split(".")[0]
    out: dict = {"market": "cn"}
    try:
        df = ak.stock_research_report_em(symbol=code)
    except Exception as e:  # noqa: BLE001
        log.warning("akshare research %s: %s", code, e)
        return None
    if df is None or df.empty:
        return None

    df = df.head(30)
    rating_col = "东财评级" if "东财评级" in df.columns else None
    firm_col = "机构" if "机构" in df.columns else None
    if rating_col is None or firm_col is None:
        log.warning("akshare %s columns unexpected: %s", code, list(df.columns))
        return None

    ratings_series = df[rating_col].dropna().astype(str)
    if not ratings_series.empty:
        numeric = [CN_RATING_TO_NUMERIC[r] for r in ratings_series if r in CN_RATING_TO_NUMERIC]
        if numeric:
            out["recommendation_mean"] = round(sum(numeric) / len(numeric), 4)
        out["recommendation_key"] = Counter(ratings_series).most_common(1)[0][0]
        out["rating_breakdown"] = {str(k): int(v) for k, v in ratings_series.value_counts().to_dict().items()}

    firms = df[firm_col].dropna().astype(str).unique().tolist()
    out["number_of_analyst_opinions"] = len(firms)

    out["recent_research"] = []
    for _, r in df.head(10).iterrows():
        out["recent_research"].append({
            "date": str(r.get("日期", ""))[:10],
            "firm": str(r.get(firm_col, "")) or None,
            "rating": str(r.get(rating_col, "")) or None,
            "title": str(r.get("报告名称", ""))[:80],
        })

    df_local = fetcher.load_local(symbol)
    if not df_local.empty:
        out["current_price"] = float(df_local["close"].iloc[-1])

    return out if (out.get("recommendation_mean") or out.get("rating_breakdown")) else None


def _etf_weighted_ratings(symbol: str) -> dict | None:
    """ETF 加权评级: yfinance funds_data.top_holdings 的成分股一致预期按权重加权.

    覆盖率 = top-10 权重之和 (通常 0.5~0.7); 我们对 covered 部分内部 renormalize.
    """
    import yfinance as yf
    try:
        t = yf.Ticker(symbol)
        fd = getattr(t, "funds_data", None)
        top = fd.top_holdings if fd is not None else None
    except Exception as e:  # noqa: BLE001
        log.debug("yfinance funds_data %s: %s", symbol, e)
        return None
    if top is None or getattr(top, "empty", True):
        return None

    out: dict = {"market": "etf", "holdings": []}
    total_w = 0.0
    weighted_rec = 0.0
    weighted_upside = 0.0
    weighted_upside_w = 0.0
    n_with_rec = 0

    for sub_sym, row in top.iterrows():
        try:
            weight = float(row.get("Holding Percent") or 0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight <= 0:
            continue
        name = str(row.get("Name") or "")
        constituent = _us_ratings(str(sub_sym))
        rec = constituent.get("recommendation_mean") if constituent else None
        upside = constituent.get("upside_pct") if constituent else None
        holding_entry = {
            "symbol": str(sub_sym),
            "name": name,
            "weight": round(weight, 4),
            "rec_mean": rec,
            "upside_pct": upside,
        }
        out["holdings"].append(holding_entry)
        if rec:
            weighted_rec += rec * weight
            total_w += weight
            n_with_rec += 1
        if upside is not None:
            weighted_upside += upside * weight
            weighted_upside_w += weight

    if total_w <= 0:
        return None

    out["recommendation_mean"] = round(weighted_rec / total_w, 4)
    out["coverage_weight"] = round(total_w, 4)
    out["n_constituents_with_rating"] = n_with_rec
    if weighted_upside_w > 0:
        out["weighted_target_upside_pct"] = round(weighted_upside / weighted_upside_w, 2)
    return out


def fetch_one(symbol: str) -> dict | None:
    """统一入口: A 股 → CN; 否则先试 US 个股, 拿不到一致预期则 fallback 到 ETF 加权."""
    if fetcher.is_a_share(symbol):
        return _cn_ratings(symbol)
    us = _us_ratings(symbol)
    has_us_data = bool(us and (us.get("recommendation_mean") or us.get("target_mean_price")))
    if has_us_data:
        return us
    etf = _etf_weighted_ratings(symbol)
    return etf or us


def store(symbol: str, data: dict) -> None:
    """Store as JSON in fundamentals.extra (no separate table needed)."""
    today = date.today().isoformat()
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        # Get existing extra, merge
        row = conn.execute(
            "SELECT extra_json FROM fundamentals WHERE symbol=? AND as_of=?",
            (symbol, today),
        ).fetchone()
        existing = {}
        if row and row[0]:
            try:
                existing = json.loads(row[0])
            except json.JSONDecodeError:
                pass
        existing["analyst_ratings"] = data
        conn.execute(
            """INSERT OR REPLACE INTO fundamentals
            (symbol, as_of, extra_json) VALUES (?,?,?)
            ON CONFLICT(symbol, as_of) DO UPDATE SET extra_json=excluded.extra_json""",
            (symbol, today, json.dumps(existing, ensure_ascii=False, default=str)),
        )
        conn.commit()


def refresh_all(symbols: list[str] | None = None) -> dict:
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
            store(sym, data)
            mean = data.get("target_mean_price", "?")
            ups = data.get("upside_pct", "?")
            out["success"].append(f"{sym}: mean={mean}, upside={ups}%")
        except Exception as e:  # noqa: BLE001
            log.exception("ratings %s: %s", sym, e)
            out["failed"].append(f"{sym}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--show", help="show one symbol's full ratings")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.show:
        out = fetch_one(args.show)
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    else:
        print(json.dumps(refresh_all(args.symbols), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
