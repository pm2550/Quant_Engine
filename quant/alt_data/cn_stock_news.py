"""A 股个股新闻抓取 (东方财富 via akshare).

为什么 (2026-05-08 incident): 原 newswatch RSS 24 个全是国际/美国宏观,
A 股个股 0 覆盖. 主人 67% 仓位 002624 完美世界涨停 8h 系统静默 + events
表里全是"霍尔木兹冲突 affected_symbols=[002624.SZ]"这种荒谬关联.

设计:
  - 对持仓 + watchlist 中的所有 A 股 (.SZ/.SS/.BJ), 每 30min 调
    `akshare.stock_news_em(symbol=code)` 拉东方财富个股新闻 (前 10 条)
  - 新条目入 news_archive (raw_hash 去重), source 设成 "em_cn_<code>",
    title 加 "[<name> <code>]" 前缀让 newswatch LLM severity 评级时一眼识别
  - 不评级也不推 — 由 newswatch 主循环自动接管 (它每 5min 扫
    news_archive 里 sev_state 未评的)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone

from .. import db, config as cfg_mod, fetcher

log = logging.getLogger(__name__)

SOURCE_PREFIX = "em_cn"


def _is_a_share(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith((".SZ", ".SS", ".BJ"))


def fetch_for_symbol(symbol: str, *, display_name: str | None = None) -> dict:
    """Fetch latest east-money news for one A-share. Insert into news_archive.

    Returns: {symbol, fetched, inserted, error?}
    """
    code = symbol.split(".")[0]
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=code)
    except Exception as e:  # noqa: BLE001
        log.warning("stock_news_em(%s) failed: %s", code, e)
        return {"symbol": symbol, "fetched": 0, "inserted": 0, "error": str(e)[:200]}

    if df is None or df.empty:
        return {"symbol": symbol, "fetched": 0, "inserted": 0}

    name = display_name or code
    inserted = 0
    fetched = len(df)
    full_source = f"{SOURCE_PREFIX}_{code}"

    with db.conn() as c:
        for _, row in df.iterrows():
            title = str(row.get("新闻标题", "") or "").strip()
            content = str(row.get("新闻内容", "") or "").strip()[:1500]
            url = str(row.get("新闻链接", "") or "").strip()
            pub = str(row.get("发布时间", "") or "").strip()
            if not title or not url:
                continue
            # Tag title with display name + code so LLM/阿雷 see it's about this stock
            tagged_title = f"[{name} {code}] {title}"
            raw_hash = hashlib.sha256((tagged_title + url).encode()).hexdigest()[:16]
            try:
                c.execute(
                    "INSERT INTO news_archive(url, title, source, published_at, "
                    "                         content, raw_hash, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (url, tagged_title, full_source, pub, content, raw_hash,
                     datetime.utcnow().isoformat() + "Z"),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass
        c.commit()

    return {"symbol": symbol, "fetched": fetched, "inserted": inserted}


def run_all(*, dry_run: bool = False) -> dict:
    portfolio = cfg_mod.load("portfolio")
    held = portfolio.get("positions", {})
    watch = portfolio.get("watchlist", [])

    cn_holds = [(s, info.get("name", s)) for s, info in held.items() if _is_a_share(s)]
    cn_watch = [(w["symbol"], w.get("name", w["symbol"]))
                for w in watch if _is_a_share(w["symbol"])]
    targets = list(dict.fromkeys(cn_holds + cn_watch))

    by_symbol = {}
    total_fetched = total_inserted = 0
    for sym, nm in targets:
        if dry_run:
            by_symbol[sym] = {"dry_run": True, "name": nm}
            continue
        r = fetch_for_symbol(sym, display_name=nm)
        by_symbol[sym] = r
        total_fetched += r.get("fetched", 0)
        total_inserted += r.get("inserted", 0)
        time.sleep(1.0)  # gentle pacing

    return {
        "n_targets": len(targets),
        "total_fetched": total_fetched,
        "total_inserted": total_inserted,
        "by_symbol": by_symbol,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", help="single A-share symbol like 002624.SZ; else all CN holdings/watchlist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    db.init()

    if args.symbol:
        if not _is_a_share(args.symbol):
            print(json.dumps({"error": "symbol must be A-share (.SZ/.SS/.BJ)"}))
            return 1
        portfolio = cfg_mod.load("portfolio")
        info = portfolio.get("positions", {}).get(args.symbol, {})
        nm = info.get("name") or args.symbol.split(".")[0]
        r = fetch_for_symbol(args.symbol, display_name=nm)
    else:
        r = run_all(dry_run=args.dry_run)

    print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
