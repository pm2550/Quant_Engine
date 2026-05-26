"""Auto-discover candidate universe from events + theme ETFs (D1) + sector ETFs + news (D2).

Phase D (2026-05-26):
  - 每天 13:00 UTC 跑 (opportunity_scanner 14:00 前 1h)
  - 4 个独立发现源, 各自产出候选 (symbol, source, reason)
  - 合并去重, 排除已在 portfolio 的, 限制 ≤ 80
  - 写 config/dynamic_universe.yaml (覆写每天刷新)

发现源:
  1. events: events 表 sev>=7 七日内 affected_symbols
  2. theme_etfs: AIQ/BOTZ/ICLN/ARKK/KWEB/IBB/SMCX/ROBO 等 top 10 holdings
  3. sector_etfs: XLK/XLE/XLF/... 跑赢 SPY >3% 的板块, top 5 holdings (D2)
  4. news: news_archive 24h 内 ≥3 次提及的 symbol (D2)
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from . import config as cfg_mod, db

log = logging.getLogger(__name__)

DYNAMIC_UNIVERSE_FILE = cfg_mod.CONFIG_DIR / "dynamic_universe.yaml"
MAX_UNIVERSE_SIZE = 80

# US ticker allowlist: 1-5 uppercase letters, no exchange suffix.
# Allow dual-class (BRK.A, BRK.B, BF.A, BF.B) but exclude .HK, .TW, .T, .L, .SS, .SZ, .CO, .SW etc.
_US_DOTTED_ALLOWLIST = {"BRK.A", "BRK.B", "BF.A", "BF.B"}


def _is_us_tradable(symbol: str) -> bool:
    """True if symbol looks like a US-tradable ticker (no foreign exchange suffix, no cash MM fund)."""
    if not symbol:
        return False
    s = symbol.strip().upper()
    if s in _US_DOTTED_ALLOWLIST:
        return True
    if "." in s:
        return False
    if not (s.isalpha() and 1 <= len(s) <= 5):
        return False
    # Federated/Fidelity money-market style cash placeholders (FGXXX, FBOXX, etc.)
    if s.endswith("XX") and len(s) == 5:
        return False
    return True

# Default theme ETFs to drill — picked for thematic relevance to user's existing positions
# (AI / semicon / clean energy / biotech / china internet / robotics).
DEFAULT_THEME_ETFS: list[dict] = [
    {"etf": "AIQ", "theme": "ai_software", "label": "AI 全球应用"},
    {"etf": "BOTZ", "theme": "ai_compute", "label": "机器人 + AI"},
    {"etf": "ROBO", "theme": "ai_compute", "label": "机器人产业"},
    {"etf": "ICLN", "theme": "clean_energy", "label": "清洁能源"},
    {"etf": "ARKK", "theme": "disruptive_innovation", "label": "ARK 颠覆性创新"},
    {"etf": "KWEB", "theme": "cn_internet", "label": "中概互联"},
    {"etf": "IBB",  "theme": "biotech", "label": "生物科技"},
    {"etf": "SMCX", "theme": "ai_compute", "label": "AI 半导体 + 加密"},
    {"etf": "XBI",  "theme": "biotech", "label": "生物科技 (小盘)"},
    {"etf": "CLOU", "theme": "cloud", "label": "云计算"},
]

# Sector ETFs for D2 rotation drill — 11 SPDR sectors
DEFAULT_SECTOR_ETFS: list[dict] = [
    {"etf": "XLK",  "sector": "technology"},
    {"etf": "XLF",  "sector": "financials"},
    {"etf": "XLE",  "sector": "energy"},
    {"etf": "XLV",  "sector": "healthcare"},
    {"etf": "XLY",  "sector": "consumer_discretionary"},
    {"etf": "XLP",  "sector": "consumer_staples"},
    {"etf": "XLI",  "sector": "industrials"},
    {"etf": "XLU",  "sector": "utilities"},
    {"etf": "XLB",  "sector": "materials"},
    {"etf": "XLRE", "sector": "real_estate"},
    {"etf": "XLC",  "sector": "communication"},
]


# ============================================================================
# Source 1: events 表 sev>=7 (D1)
# ============================================================================

def discover_from_events(*, days: int = 7, min_sev: int = 7) -> list[dict]:
    """events 表里 sev>=N 的 affected_symbols, 去重后返回 candidate list."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    seen: dict[str, dict] = {}
    try:
        with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
            rows = conn.execute(
                "SELECT severity, category, affected_symbols, summary, fired_at "
                "FROM events WHERE fired_at >= ? AND severity >= ? "
                "ORDER BY severity DESC, fired_at DESC",
                (cutoff, min_sev),
            ).fetchall()
    except sqlite3.Error:
        log.exception("events query failed")
        return []

    for sev, category, affected, summary, fired_at in rows:
        if not affected:
            continue
        for sym in [s.strip() for s in affected.split(",") if s.strip()]:
            if sym in seen:
                # Keep highest severity record
                if sev > seen[sym]["severity"]:
                    seen[sym]["severity"] = sev
                    seen[sym]["reason"] = f"{category} sev {sev}: {(summary or '')[:60]}"
                continue
            seen[sym] = {
                "symbol": sym,
                "source": "events",
                "severity": sev,
                "category": category,
                "reason": f"{category} sev {sev}: {(summary or '')[:60]}",
                "fired_at": fired_at,
            }
    return list(seen.values())


# ============================================================================
# Source 2: theme ETF holdings (D1)
# ============================================================================

def discover_from_theme_etfs(*, etfs: list[dict] | None = None, top_n: int = 10) -> list[dict]:
    """每个 theme ETF 抽 top N 持仓作为候选."""
    import yfinance as yf
    etfs = etfs or DEFAULT_THEME_ETFS
    found: dict[str, dict] = {}
    for entry in etfs:
        etf_sym = entry["etf"]
        try:
            t = yf.Ticker(etf_sym)
            fd = getattr(t, "funds_data", None)
            top = fd.top_holdings if fd is not None else None
        except Exception as e:  # noqa: BLE001
            log.warning("theme_etf %s funds_data failed: %s", etf_sym, e)
            continue
        if top is None or getattr(top, "empty", True):
            log.info("theme_etf %s: no holdings data", etf_sym)
            continue
        rows = list(top.iterrows())[:top_n]
        for sub_sym, row in rows:
            sub_sym = str(sub_sym).strip().upper()
            if not _is_us_tradable(sub_sym):
                continue
            try:
                weight = float(row.get("Holding Percent") or 0)
            except (TypeError, ValueError):
                weight = 0.0
            name = str(row.get("Name") or "")
            existing = found.get(sub_sym)
            entry_data = {
                "symbol": sub_sym,
                "source": "theme_etf",
                "etf": etf_sym,
                "etf_theme": entry["theme"],
                "etf_label": entry["label"],
                "holding_pct": round(weight * 100, 2),
                "name": name,
                "reason": f"{etf_sym} ({entry['label']}) top holding {weight * 100:.1f}%",
            }
            # 保留权重最大的 ETF
            if existing is None or weight > (existing.get("holding_pct") or 0) / 100:
                found[sub_sym] = entry_data
    return list(found.values())


# ============================================================================
# Source 3: Sector ETF rotation drill (D2)
# ============================================================================

def discover_from_sector_etfs(*, sectors: list[dict] | None = None,
                               lookback_days: int = 5,
                               benchmark: str = "SPY",
                               outperform_pct: float = 0.03,
                               top_n_holdings: int = 5) -> list[dict]:
    """跑赢 SPY ≥3% 的板块 ETF, 抽其 top N holdings 作为候选."""
    from . import fetcher
    import yfinance as yf

    sectors = sectors or DEFAULT_SECTOR_ETFS

    def _pct_change(sym: str, n: int) -> float | None:
        df = fetcher.load_local(sym)
        if df.empty or len(df) < n + 1:
            try:
                df = fetcher.fetch_symbol(sym)
            except Exception:
                return None
        if df is None or df.empty or len(df) < n + 1:
            return None
        return float(df["close"].iloc[-1] / df["close"].iloc[-(n + 1)] - 1)

    bench_ret = _pct_change(benchmark, lookback_days)
    if bench_ret is None:
        log.warning("benchmark %s return unavailable; skipping sector_etf source", benchmark)
        return []

    found: dict[str, dict] = {}
    for entry in sectors:
        etf_sym = entry["etf"]
        sec_ret = _pct_change(etf_sym, lookback_days)
        if sec_ret is None:
            continue
        rel = sec_ret - bench_ret
        if rel < outperform_pct:
            continue
        # Hot sector → drill holdings
        try:
            t = yf.Ticker(etf_sym)
            fd = getattr(t, "funds_data", None)
            top = fd.top_holdings if fd is not None else None
        except Exception as e:  # noqa: BLE001
            log.warning("sector_etf %s holdings failed: %s", etf_sym, e)
            continue
        if top is None or getattr(top, "empty", True):
            continue
        rows = list(top.iterrows())[:top_n_holdings]
        for sub_sym, row in rows:
            sub_sym = str(sub_sym).strip().upper()
            if not _is_us_tradable(sub_sym):
                continue
            try:
                weight = float(row.get("Holding Percent") or 0)
            except (TypeError, ValueError):
                weight = 0.0
            existing = found.get(sub_sym)
            data = {
                "symbol": sub_sym,
                "source": "sector_rotation",
                "etf": etf_sym,
                "sector": entry["sector"],
                "sector_relative_return_pct": round(rel * 100, 2),
                "holding_pct": round(weight * 100, 2),
                "reason": f"{etf_sym} ({entry['sector']}) {lookback_days}d 相对 {benchmark} "
                          f"{rel * 100:+.1f}% — top holding {weight * 100:.1f}%",
            }
            # 保留 sector rel return 最大的
            if existing is None or rel > (existing.get("sector_relative_return_pct") or 0) / 100:
                found[sub_sym] = data
    return list(found.values())


# ============================================================================
# Source 4: News mention frequency (D2)
# ============================================================================

# Light-weight ticker extraction: simple S&P 100 + popular tickers for matching news titles.
# Avoids false positives like "US" matching "United States".
_TICKER_WATCH_LIST: set[str] = {
    "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "TSLA", "BRK.A", "BRK.B",
    "AVGO", "JPM", "LLY", "V", "UNH", "WMT", "XOM", "MA", "HD", "JNJ", "PG", "ORCL",
    "COST", "ABBV", "BAC", "MRK", "CVX", "KO", "ADBE", "AMD", "PEP", "WFC", "CSCO",
    "TMO", "ACN", "MCD", "CRM", "ABT", "LIN", "DHR", "TXN", "DIS", "NFLX", "VZ",
    "INTC", "AMGN", "INTU", "QCOM", "PFE", "UNP", "PM", "CAT", "RTX", "T", "MS",
    "GS", "NEE", "LOW", "BMY", "SPGI", "ISRG", "HON", "BLK", "PLD", "AMAT", "GE",
    "BKNG", "PLTR", "ARM", "SMCI", "DELL", "ANET", "MU", "SNOW", "NET", "CRWD",
    "COIN", "SHOP", "ABNB", "UBER", "ORCL", "TSM", "ASML", "AVAV", "RKLB", "RBLX",
    # ETFs of interest
    "QQQ", "SPY", "VOO", "VTI", "SOXX", "SMH", "XLK", "XLE", "XLF", "XLY", "XLV",
    "ARKK", "AIQ", "BOTZ", "ICLN", "KWEB", "IBB", "XBI", "ROBO",
}


def discover_from_news_mentions(*, days: int = 1, min_count: int = 3) -> list[dict]:
    """news_archive 内 24h 提到 >=min_count 次的 ticker."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "titles": []})
    try:
        with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
            # news_archive 的时间列是 published_at (主), 兜底用 fetched_at
            rows = conn.execute(
                "SELECT title, source, COALESCE(published_at, fetched_at) AS ts "
                "FROM news_archive "
                "WHERE COALESCE(published_at, fetched_at) >= ? "
                "ORDER BY COALESCE(published_at, fetched_at) DESC LIMIT 5000",
                (cutoff,),
            ).fetchall()
    except sqlite3.Error:
        log.exception("news_archive query failed")
        return []

    import re
    # Match $TICK or TICK in title — must be 2-5 uppercase letters, word boundary
    pat = re.compile(r"(?:\$)?\b([A-Z]{2,5})\b")
    for title, source, ts in rows:
        if not title:
            continue
        for match in pat.findall(title):
            if match in _TICKER_WATCH_LIST:
                counts[match]["count"] += 1
                if len(counts[match]["titles"]) < 3:
                    counts[match]["titles"].append(title[:80])

    out: list[dict] = []
    for sym, info in counts.items():
        if info["count"] < min_count:
            continue
        out.append({
            "symbol": sym,
            "source": "news_mentions",
            "count_24h": info["count"],
            "reason": f"24h 新闻提及 {info['count']} 次: " + "; ".join(info["titles"][:2])[:100],
        })
    out.sort(key=lambda d: d["count_24h"], reverse=True)
    return out


# ============================================================================
# Aggregation + dedup
# ============================================================================

def _portfolio_symbols() -> set[str]:
    try:
        p = cfg_mod.load("portfolio")
    except Exception:
        return set()
    out = set(p.get("positions", {}).keys())
    for w in p.get("watchlist") or []:
        if w.get("symbol"):
            out.add(w["symbol"])
    return out


def _static_universe_symbols() -> set[str]:
    static_file = cfg_mod.CONFIG_DIR / "opportunity_universe.yaml"
    if not static_file.exists():
        return set()
    try:
        data = yaml.safe_load(static_file.read_text()) or {}
    except Exception:
        return set()
    return {e["symbol"] for e in (data.get("universe") or []) if e.get("symbol")}


def aggregate_candidates(
    *,
    enable_events: bool = True,
    enable_theme_etfs: bool = True,
    enable_sector_etfs: bool = True,
    enable_news: bool = True,
    max_size: int = MAX_UNIVERSE_SIZE,
) -> list[dict]:
    """合并 4 个发现源, 去重 + 排除 portfolio + 排除 static_universe (避免重复)."""
    portfolio_syms = _portfolio_symbols()
    static_syms = _static_universe_symbols()
    excluded = portfolio_syms | static_syms

    by_symbol: dict[str, dict] = {}

    def _merge(cands: list[dict]) -> None:
        for c in cands:
            sym = c.get("symbol")
            if not sym or sym in excluded:
                continue
            if sym in by_symbol:
                # accumulate sources for transparency
                by_symbol[sym].setdefault("sources", set()).add(c["source"])
                by_symbol[sym].setdefault("reasons", []).append(c["reason"])
            else:
                by_symbol[sym] = {
                    **c,
                    "sources": {c["source"]},
                    "reasons": [c["reason"]],
                }

    if enable_events:
        _merge(discover_from_events())
    if enable_theme_etfs:
        _merge(discover_from_theme_etfs())
    if enable_sector_etfs:
        _merge(discover_from_sector_etfs())
    if enable_news:
        _merge(discover_from_news_mentions())

    # Sort: multi-source candidates first (more independent confirmations), then alphabetical
    out: list[dict] = []
    for sym, data in by_symbol.items():
        data["sources"] = sorted(data["sources"])
        data["n_sources"] = len(data["sources"])
        out.append(data)
    out.sort(key=lambda d: (-d["n_sources"], d["symbol"]))
    return out[:max_size]


# ============================================================================
# Persistence + CLI
# ============================================================================

def write_dynamic_universe(candidates: list[dict], path: Path | None = None) -> Path:
    path = path or DYNAMIC_UNIVERSE_FILE
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_candidates": len(candidates),
        "universe": [
            {
                "symbol": c["symbol"],
                "sources": c.get("sources", []),
                "n_sources": c.get("n_sources", 1),
                "reason": "; ".join(c.get("reasons", [])[:3])[:200],
            }
            for c in candidates
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
    return path


def run_discovery(*, dry_run: bool = False) -> dict:
    cands = aggregate_candidates()
    summary = {
        "n_candidates": len(cands),
        "by_source": defaultdict(int),
        "by_n_sources": defaultdict(int),
    }
    for c in cands:
        for src in c.get("sources") or []:
            summary["by_source"][src] += 1
        summary["by_n_sources"][c.get("n_sources", 1)] += 1
    summary["by_source"] = dict(summary["by_source"])
    summary["by_n_sources"] = dict(summary["by_n_sources"])

    if not dry_run:
        path = write_dynamic_universe(cands)
        summary["written_to"] = str(path)
    else:
        print(json.dumps([{
            "symbol": c["symbol"],
            "sources": c["sources"],
            "n_sources": c["n_sources"],
            "reasons": c["reasons"][:2],
        } for c in cands], indent=2, ensure_ascii=False))
    return summary


def load_dynamic_universe() -> list[dict]:
    """Return list of {symbol, sources, n_sources, reason} from dynamic_universe.yaml."""
    if not DYNAMIC_UNIVERSE_FILE.exists():
        return []
    try:
        data = yaml.safe_load(DYNAMIC_UNIVERSE_FILE.read_text()) or {}
    except Exception:
        return []
    return data.get("universe") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印候选, 不写 dynamic_universe.yaml")
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--no-theme-etfs", action="store_true")
    ap.add_argument("--no-sector-etfs", action="store_true")
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cands = aggregate_candidates(
        enable_events=not args.no_events,
        enable_theme_etfs=not args.no_theme_etfs,
        enable_sector_etfs=not args.no_sector_etfs,
        enable_news=not args.no_news,
    )
    out = {
        "n_candidates": len(cands),
        "by_source": defaultdict(int),
        "candidates": [{
            "symbol": c["symbol"],
            "sources": c["sources"],
            "reasons": c["reasons"][:2],
        } for c in cands],
    }
    for c in cands:
        for s in c.get("sources") or []:
            out["by_source"][s] += 1
    out["by_source"] = dict(out["by_source"])

    if not args.dry_run:
        path = write_dynamic_universe(cands)
        out["written_to"] = str(path)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
