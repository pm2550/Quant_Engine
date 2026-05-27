"""SEC EDGAR fundamentals fetcher — point-in-time clean, no rate-limit pain.

Uses SEC's free XBRL Company Facts API:
  https://data.sec.gov/api/xbrl/companyfacts/CIK<10-digit>.json

Each observation includes `filed` (announcement date) → perfect PIT alignment.
ETFs are not in SEC's company-tickers map; they get skipped (no fundamentals
for index products, by design).

Cache: /data2/quant/data/edgar/<CIK>.json (~4MB/company; 98 syms ≈ 400MB).
Refresh once per quarter is enough — but the fetcher does incremental
updates if cached file is < 7 days old (skips re-download).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


log = logging.getLogger(__name__)

CACHE_DIR = Path("/data2/quant/data/edgar")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TICKER_MAP_PATH = CACHE_DIR / "company_tickers.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# Per SEC fair-access: identify yourself in User-Agent (email or contact info)
USER_AGENT = "gaohaopm@gmail.com Quant_Engine"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# Throttle: SEC allows 10 req/sec but we go gentle to keep them happy
SLEEP_BETWEEN_REQUESTS = 0.15

# Map our preferred fundamentals to SEC's us-gaap line item names.
# Multiple alternatives because companies switched between concepts over time
# (e.g. "Revenues" pre-2018 → "RevenueFromContractWithCustomerExcludingAssessedTax" post-ASC-606).
LINE_ITEMS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "operating_cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}


def _refresh_ticker_map(*, max_age_days: int = 7) -> dict:
    """Download SEC's ticker→CIK map (cached, refreshed weekly)."""
    if TICKER_MAP_PATH.exists():
        age_days = (datetime.utcnow().timestamp() - TICKER_MAP_PATH.stat().st_mtime) / 86400
        if age_days < max_age_days:
            with open(TICKER_MAP_PATH) as f:
                return json.load(f)
    log.info("fetching SEC ticker map from %s", TICKER_MAP_URL)
    r = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    with open(TICKER_MAP_PATH, "w") as f:
        json.dump(data, f)
    return data


def ticker_to_cik(symbol: str) -> str | None:
    """Return 10-digit CIK string for ticker, or None if not in SEC list (ETFs)."""
    sym = symbol.upper().split(".")[0]  # strip .SS/.SZ if present
    data = _refresh_ticker_map()
    for v in data.values():
        if v.get("ticker", "").upper() == sym:
            return str(v["cik_str"]).zfill(10)
    return None


def fetch_company_facts(cik: str, *, max_age_days: int = 90) -> dict | None:
    """Download companyfacts JSON for one CIK; cache to disk."""
    cache_path = CACHE_DIR / f"CIK{cik}.json"
    if cache_path.exists():
        age_days = (datetime.utcnow().timestamp() - cache_path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            with open(cache_path) as f:
                return json.load(f)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60)
    except Exception as e:  # noqa: BLE001
        log.warning("EDGAR fetch failed for CIK %s: %s", cik, e)
        return None
    if r.status_code == 404:
        log.info("EDGAR has no facts for CIK %s (foreign filer? sub-reporting?)", cik)
        return None
    if r.status_code != 200:
        log.warning("EDGAR %s for CIK %s", r.status_code, cik)
        return None
    data = r.json()
    with open(cache_path, "w") as f:
        json.dump(data, f)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return data


def _pick_line_item(facts: dict, candidates: list[str]) -> list[dict]:
    """Merge ALL candidate line items into one observation list.

    Needed because companies switch concepts over time (e.g. AAPL Revenues
    pre-2018 → RevenueFromContractWithCustomerExcludingAssessedTax post-ASC-606
    in 2018). If we picked only the most-populated single concept, we'd miss
    everything from the other side of the transition. Dedup happens later in
    _observations_to_quarterly_df by (fy, fp) — last-filed wins.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    merged: list[dict] = []
    for name in candidates:
        if name not in us_gaap:
            continue
        units = us_gaap[name].get("units", {})
        for unit_key in ("USD", "USD/shares", "shares"):
            if unit_key in units:
                merged.extend(units[unit_key])
                break
    return merged


def _observations_to_quarterly_df(obs: list[dict]) -> pd.DataFrame:
    """Turn SEC observations into a clean DataFrame indexed by `filed`.

    Keep only single-quarter (3-month) observations:
      - period duration (end - start) within 60-100 days
      - `fp` in (Q1, Q2, Q3, Q4)  — drops FY annual + cumulative H1/9M frames

    Annual/cumulative observations break TTM arithmetic when mixed with Qs.
    Most SEC observations include `frame` like "CY2024Q1" but for companies
    on non-standard fiscal years (AAPL, MSFT, NVDA), `frame` is often empty —
    so we use duration + fp instead.
    """
    if not obs:
        return pd.DataFrame()
    rows = []
    for o in obs:
        fp = o.get("fp")
        if fp not in ("Q1", "Q2", "Q3", "Q4"):
            continue
        start = o.get("start")
        end = o.get("end")
        if start and end:
            try:
                dur = (pd.Timestamp(end) - pd.Timestamp(start)).days
            except Exception:  # noqa: BLE001
                dur = None
            if dur is not None and not (60 <= dur <= 100):
                continue  # H1 / 9M cumulative; skip
        rows.append({
            "filed": o.get("filed"),
            "period_end": o.get("end"),
            "val": o.get("val"),
            "fy": o.get("fy"),
            "fp": fp,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["filed"] = pd.to_datetime(df["filed"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    # Same (fy, fp) may appear multiple times (10-Q amendments, 10-K restatement); keep latest
    df = df.sort_values("filed").drop_duplicates(subset=["fy", "fp"], keep="last")
    df = df.sort_values("filed").reset_index(drop=True)
    return df


def extract_fundamentals(facts: dict) -> dict[str, pd.DataFrame]:
    """Return dict of line_item_name → quarterly DataFrame (indexed by filed date)."""
    out = {}
    for key, candidates in LINE_ITEMS.items():
        obs = _pick_line_item(facts, candidates)
        df = _observations_to_quarterly_df(obs)
        if not df.empty:
            out[key] = df
    return out


def _build_pit_timeseries(fundamentals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge per-line-item quarterly frames into one daily DataFrame, forward-filled.

    For flow items (revenue, net_income, gross_profit, operating_income, OCF, EPS):
        also computes a rolling 4-quarter TTM (trailing twelve months) value
        indexed by filed-date of the 4th quarter.
    For stock items (cash, long_term_debt, equity):
        forward-fills the latest balance sheet value.

    Output: daily DataFrame, columns = raw quarterly value + TTM where applicable.
    """
    if not fundamentals:
        return pd.DataFrame()

    flow_items = {"revenue", "net_income", "gross_profit", "operating_income",
                  "operating_cashflow", "eps_diluted", "eps_basic"}
    stock_items = {"cash", "long_term_debt", "equity"}

    daily_cols: dict[str, pd.Series] = {}
    earliest = None

    for key, df in fundamentals.items():
        if df.empty:
            continue
        s_raw = df.set_index("filed")["val"]
        s_raw = s_raw[~s_raw.index.duplicated(keep="last")].sort_index()
        if earliest is None or s_raw.index.min() < earliest:
            earliest = s_raw.index.min()
        daily_cols[key + "_Q"] = s_raw

        if key in flow_items:
            # TTM: at each filed date, sum of 4 most recent values in df
            # (quarterly values are non-overlapping single quarters per our filter)
            ttm = s_raw.rolling(4, min_periods=4).sum()
            daily_cols[key + "_TTM"] = ttm

    if not daily_cols:
        return pd.DataFrame()

    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    idx = pd.date_range(earliest.normalize(), end, freq="D")
    daily = pd.DataFrame(index=idx)
    for key, s in daily_cols.items():
        daily[key] = s.reindex(idx, method="ffill")
    return daily


def fetch_and_align(symbol: str) -> pd.DataFrame:
    """End-to-end: ticker → CIK → facts JSON → PIT-aligned daily fundamentals DF."""
    cik = ticker_to_cik(symbol)
    if cik is None:
        return pd.DataFrame()
    facts = fetch_company_facts(cik)
    if facts is None:
        return pd.DataFrame()
    items = extract_fundamentals(facts)
    return _build_pit_timeseries(items)


def fundamental_features_for(symbol: str, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute fundamental features aligned to a given price index.

    Output columns (all PIT-correct; only uses filings dated <= row's date):
      EPS_DIL_TTM    Latest TTM diluted EPS
      REV_YOY        TTM revenue growth YoY (uses TTM/TTM_365d to avoid Q-vs-FY)
      EPS_YOY        TTM EPS growth YoY
      NI_MARGIN      TTM net margin
      GROSS_MARGIN   TTM gross margin
      OP_MARGIN      TTM operating margin
      DEBT_EQUITY    Long-term debt / equity (book leverage)
      OCF_TO_REV     TTM operating cashflow / revenue (cash quality)
      DAYS_SINCE_FILING  Calendar days since most recent 10-Q
    """
    fund_daily = fetch_and_align(symbol)
    if fund_daily.empty:
        return pd.DataFrame(index=price_index)

    px_index = pd.DatetimeIndex(price_index).tz_localize(None) if price_index.tz else pd.DatetimeIndex(price_index)

    # YoY MUST be computed on the daily calendar series (shift 365 days) BEFORE
    # reindexing to the (potentially business-day) price index — otherwise
    # shift(365) on business-day index = ~1.5 calendar years.
    with np.errstate(divide="ignore", invalid="ignore"):
        rev_yoy_daily = (fund_daily["revenue_TTM"] - fund_daily["revenue_TTM"].shift(365)) / fund_daily["revenue_TTM"].shift(365).abs() \
            if "revenue_TTM" in fund_daily.columns else None
        eps_yoy_daily = (fund_daily["eps_diluted_TTM"] - fund_daily["eps_diluted_TTM"].shift(365)) / fund_daily["eps_diluted_TTM"].shift(365).abs() \
            if "eps_diluted_TTM" in fund_daily.columns else None
    fund = fund_daily.reindex(px_index, method="ffill")

    feat = pd.DataFrame(index=px_index)

    rev_ttm = fund.get("revenue_TTM")
    if rev_ttm is not None:
        feat["REV_TTM"] = rev_ttm
        if rev_yoy_daily is not None:
            feat["REV_YOY"] = rev_yoy_daily.reindex(px_index, method="ffill")
    eps_ttm = fund.get("eps_diluted_TTM")
    if eps_ttm is not None:
        feat["EPS_DIL_TTM"] = eps_ttm
        if eps_yoy_daily is not None:
            feat["EPS_YOY"] = eps_yoy_daily.reindex(px_index, method="ffill")
    ni_ttm = fund.get("net_income_TTM")
    if ni_ttm is not None and rev_ttm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            feat["NI_MARGIN"] = ni_ttm / rev_ttm
    gp_ttm = fund.get("gross_profit_TTM")
    if gp_ttm is not None and rev_ttm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            feat["GROSS_MARGIN"] = gp_ttm / rev_ttm
    op_ttm = fund.get("operating_income_TTM")
    if op_ttm is not None and rev_ttm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            feat["OP_MARGIN"] = op_ttm / rev_ttm
    if "long_term_debt_Q" in fund.columns and "equity_Q" in fund.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            feat["DEBT_EQUITY"] = fund["long_term_debt_Q"] / fund["equity_Q"].abs()
    ocf_ttm = fund.get("operating_cashflow_TTM")
    if ocf_ttm is not None and rev_ttm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            feat["OCF_TO_REV"] = ocf_ttm / rev_ttm

    # Days since most recent filing
    last_filing_per_row = fund_daily.dropna(how="all").index.to_series()
    if not last_filing_per_row.empty:
        days_since = pd.Series(index=px_index, dtype=float)
        for d in px_index:
            eligible = last_filing_per_row[last_filing_per_row <= d]
            days_since.loc[d] = (d - eligible.iloc[-1]).days if not eligible.empty else np.nan
        feat["DAYS_SINCE_FILING"] = days_since

    return feat.replace([np.inf, -np.inf], np.nan)


def backfill_all(symbols: Iterable[str], *, dry_run: bool = False) -> dict[str, str]:
    """Fetch + cache EDGAR data for many symbols. Returns per-sym status."""
    out: dict[str, str] = {}
    for sym in symbols:
        cik = ticker_to_cik(sym)
        if cik is None:
            out[sym] = "no-CIK (ETF or non-SEC filer)"
            continue
        if dry_run:
            out[sym] = f"would fetch CIK={cik}"
            continue
        facts = fetch_company_facts(cik)
        if facts is None:
            out[sym] = "fetch-failed"
            continue
        items = extract_fundamentals(facts)
        out[sym] = f"ok ({len(items)} line items)"
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", help="comma-sep symbols; default = all cached parquet")
    ap.add_argument("--probe", help="single symbol — show extracted features tail")
    args = ap.parse_args()

    if args.probe:
        sym = args.probe
        cik = ticker_to_cik(sym)
        print(f"{sym} → CIK={cik}")
        if cik:
            facts = fetch_company_facts(cik)
            if facts:
                items = extract_fundamentals(facts)
                print(f"line items extracted: {list(items.keys())}")
                # Build features against a daily index of the last ~30 days
                idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=30, freq="B")
                feat = fundamental_features_for(sym, idx)
                print(feat.tail(5))
    else:
        if args.syms:
            syms = args.syms.split(",")
        else:
            syms = sorted(p.stem for p in Path("/data2/quant/data/prices").glob("*.parquet"))
        print(f"backfilling {len(syms)} symbols...")
        results = backfill_all(syms)
        ok = sum(1 for v in results.values() if v.startswith("ok"))
        skipped = sum(1 for v in results.values() if "no-CIK" in v)
        failed = sum(1 for v in results.values() if "failed" in v)
        print(f"\nok={ok}  skipped(no-CIK)={skipped}  failed={failed}")
        print("\nfirst 20 results:")
        for s, st in list(results.items())[:20]:
            print(f"  {s:12} {st}")
        # If any skipped, print them in full so we know which ETFs got dropped
        skipped_syms = [s for s, st in results.items() if "no-CIK" in st]
        if skipped_syms:
            print(f"\nskipped (ETFs / non-SEC): {', '.join(skipped_syms)}")
