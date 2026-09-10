"""A 股个股新闻抓取 (东方财富 via akshare).

为什么: 原 newswatch RSS 24 个全是国际/美国宏观, A 股个股 0 覆盖. 持仓里
的 A 股标的出现涨停级行情时系统会长时间静默, 且 events 表里 LLM 会把
宏观新闻错误关联到不相关的 A 股代码.

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
NOTICE_SOURCE_PREFIX = "em_notice"

# A6 (2026-09-10): 只有 stock_news_em 时, 002624 近 60 天只拿到 57 条 —— 相对
# 美股持仓 (SEC form4/8-K + 5 个半导体专业源) 严重不足。公司公告是 A 股信息密度
# 最高的来源 (业绩预告 / 重大合同 / 股权变动 / 停复牌), 且是法定披露, 噪音低。
NOTICE_TYPES_HIGH_VALUE = {
    "业绩预告", "业绩报告", "重大事项", "资产重组", "股份回购",
    "增持减持", "股权变动", "融资公告", "风险提示", "停牌复牌",
}


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
                    (url, tagged_title, full_source,
                     db.normalize_timestamp(pub), content, raw_hash,   # A2
                     datetime.utcnow().isoformat() + "Z"),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass
        c.commit()

    return {"symbol": symbol, "fetched": fetched, "inserted": inserted}


def fetch_notices_for_symbols(symbols: dict[str, str], *, days_back: int = 3) -> dict:
    """拉取 A 股公司公告 (东财) 并入 news_archive。

    akshare 的 stock_notice_report 是按"日期 + 全市场"取的 (一天约 1,000 条),
    所以这里一次拉全市场再按持仓代码过滤 —— 比逐个标的调用省得多。
    """
    if not symbols:
        return {"fetched": 0, "inserted": 0, "matched": 0}
    import datetime as _dt
    codes = {s.split(".")[0]: nm for s, nm in symbols.items()}
    fetched = inserted = matched = 0
    try:
        import akshare as ak
    except Exception as e:  # noqa: BLE001
        return {"fetched": 0, "inserted": 0, "matched": 0, "error": repr(e)[:200]}

    for back in range(days_back):
        day = (_dt.date.today() - _dt.timedelta(days=back)).strftime("%Y%m%d")
        try:
            df = ak.stock_notice_report(symbol="全部", date=day)
        except Exception as e:  # noqa: BLE001
            log.warning("stock_notice_report(%s) failed: %s", day, e)
            continue
        if df is None or df.empty:
            continue
        fetched += len(df)
        with db.conn() as c:
            for _, row in df.iterrows():
                code = str(row.get("代码", "") or "").strip()
                if code not in codes:
                    continue
                matched += 1
                title = str(row.get("公告标题", "") or "").strip()
                ntype = str(row.get("公告类型", "") or "").strip()
                url = str(row.get("网址", "") or "").strip()
                pub = str(row.get("公告日期", "") or "").strip()
                if not title or not url:
                    continue
                tagged = f"[{codes[code]} {code} 公告/{ntype}] {title}"
                raw_hash = hashlib.sha256((tagged + url).encode()).hexdigest()[:16]
                try:
                    c.execute(
                        "INSERT INTO news_archive(url, title, source, published_at, "
                        "                         content, raw_hash, fetched_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (url, tagged, f"{NOTICE_SOURCE_PREFIX}_{code}",
                         db.normalize_timestamp(pub),          # A2
                         f"{ntype}: {title}", raw_hash,
                         datetime.utcnow().isoformat() + "Z"),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
            c.commit()
        time.sleep(1.0)
    return {"fetched": fetched, "inserted": inserted, "matched": matched}


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

    # A6: 个股新闻之外再拉公司公告 (法定披露, 信息密度高、噪音低)
    notices = {"skipped": True} if dry_run else fetch_notices_for_symbols(dict(targets))

    return {
        "n_targets": len(targets),
        "total_fetched": total_fetched,
        "total_inserted": total_inserted,
        "notices": notices,
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
