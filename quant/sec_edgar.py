"""SEC EDGAR scanner - 持仓/关注股的 Form 4 (insider) / 8-K (material events) / 13F.

- Form 4: 高管/大股东 买卖, 重要信号 (尤其大额异常)
- 8-K:    重大事件公告 (高管离职/收购/诉讼/战略调整)
- 13F:    机构季度持仓 (Buffett/桥水 等大基金)

Output: 入 news_archive 表 (走 newswatch pipeline 自动评级 + 推送)
        Form 4 大额 (>$1M) 或 8-K 顶级 item 直接 push
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from urllib.parse import urlencode

import requests

from . import config as cfg_mod, db

log = logging.getLogger(__name__)

UA = "claude-quant pm2550@gmail.com"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip"}

# Map portfolio symbol → SEC CIK (zero-padded)
CIK_MAP = {
    "VRT": "0001674101",
    "RKLB": "0001819994",
    "AMD": "0000002488",
    "ARM": "0001973239",
    "NVDA": "0001045810",
    "AVGO": "0001730168",
    "PLTR": "0001321655",
    "TSLA": "0001318605",
    "QCOM": "0000804328",
    "QQQ":  "0001067839",
}

# Big institutional CIKs to track for 13F changes
BIG_HOLDERS = {
    "Berkshire Hathaway":    "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors":     "0001423053",
    "Soros Fund Management": "0001029160",
    "Scion Asset Mgmt (Burry)": "0001649339",
}


def _edgar_filings(cik: str, form_type: str, limit: int = 10) -> list[dict]:
    """Fetch recent filings of a given form type for one CIK."""
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "CIK": cik,
        "type": form_type,
        "dateb": "",
        "owner": "include",
        "count": limit,
        "output": "atom",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    text = r.text
    items = []
    # Atom feed entries - simple regex parse
    for entry in re.finditer(
        r"<entry>(.*?)</entry>",
        text,
        re.DOTALL,
    ):
        block = entry.group(1)
        title_m = re.search(r"<title>([^<]+)</title>", block)
        link_m = re.search(r'<link[^>]*href="([^"]+)"', block)
        updated_m = re.search(r"<updated>([^<]+)</updated>", block)
        if not (title_m and link_m):
            continue
        items.append({
            "title": title_m.group(1).strip(),
            "url": link_m.group(1),
            "filed_at": updated_m.group(1) if updated_m else None,
        })
    return items


def _store_as_news(symbol: str, items: list[dict], form_type: str, source_label: str) -> int:
    """Insert SEC filings into news_archive table so newswatch pipeline picks them up."""
    n = 0
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        for it in items:
            try:
                title = f"[SEC {form_type}] {symbol}: {it['title']}"
                conn.execute(
                    """INSERT INTO news_archive(url, title, source, published_at, content, raw_hash, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (it["url"], title, source_label,
                     it.get("filed_at"),
                     f"{form_type} filing for {symbol}", "",
                     datetime.utcnow().isoformat() + "Z"),
                )
                n += 1
            except sqlite3.IntegrityError:
                pass  # already seen
        conn.commit()
    return n


def scan_form4(*, lookback_hours: int = 24) -> dict:
    """Scan recent Form 4 filings for portfolio + watchlist."""
    portfolio = cfg_mod.load("portfolio")
    syms = list(portfolio.get("positions", {}).keys()) + [w["symbol"] for w in portfolio.get("watchlist", [])]
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    out = {"scanned": 0, "stored": 0, "by_symbol": {}}
    for sym in syms:
        cik = CIK_MAP.get(sym)
        if not cik:
            continue
        try:
            items = _edgar_filings(cik, "4", limit=20)
            recent = []
            for it in items:
                try:
                    fa = datetime.fromisoformat(it["filed_at"].replace("Z", "+00:00"))
                    if fa.replace(tzinfo=None) >= cutoff:
                        recent.append(it)
                except Exception:
                    pass
            n = _store_as_news(sym, recent, "Form 4", f"sec-form4-{sym.lower()}")
            out["scanned"] += len(items)
            out["stored"] += n
            if n > 0:
                out["by_symbol"][sym] = n
            time.sleep(0.2)  # SEC rate limit
        except Exception as e:  # noqa: BLE001
            log.warning("Form 4 %s: %s", sym, e)
    return out


def scan_8k(*, lookback_hours: int = 48) -> dict:
    """Scan recent 8-K (material event) filings."""
    portfolio = cfg_mod.load("portfolio")
    syms = list(portfolio.get("positions", {}).keys()) + [w["symbol"] for w in portfolio.get("watchlist", [])]
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    out = {"scanned": 0, "stored": 0, "by_symbol": {}}
    for sym in syms:
        cik = CIK_MAP.get(sym)
        if not cik:
            continue
        try:
            items = _edgar_filings(cik, "8-K", limit=10)
            recent = []
            for it in items:
                try:
                    fa = datetime.fromisoformat(it["filed_at"].replace("Z", "+00:00"))
                    if fa.replace(tzinfo=None) >= cutoff:
                        recent.append(it)
                except Exception:
                    pass
            n = _store_as_news(sym, recent, "8-K", f"sec-8k-{sym.lower()}")
            out["scanned"] += len(items)
            out["stored"] += n
            if n > 0:
                out["by_symbol"][sym] = n
            time.sleep(0.2)
        except Exception as e:  # noqa: BLE001
            log.warning("8-K %s: %s", sym, e)
    return out


def scan_13f_changes() -> dict:
    """Detect 13F filings — quarterly. Just records new filings.
    Detailed position diff is a separate step (compare last 2 13Fs)."""
    out = {"scanned": 0, "stored": 0, "by_holder": {}}
    for name, cik in BIG_HOLDERS.items():
        try:
            items = _edgar_filings(cik, "13F-HR", limit=4)
            n = _store_as_news(name, items, "13F-HR", f"sec-13f-{cik}")
            out["scanned"] += len(items)
            out["stored"] += n
            if n > 0:
                out["by_holder"][name] = n
            time.sleep(0.3)
        except Exception as e:  # noqa: BLE001
            log.warning("13F %s: %s", name, e)
    return out


def run_all() -> dict:
    db.init()
    return {
        "form4": scan_form4(),
        "8k": scan_8k(),
        "13f": scan_13f_changes(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["form4", "8k", "13f", "all"], default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.mode == "all":
        out = run_all()
    elif args.mode == "form4":
        out = scan_form4()
    elif args.mode == "8k":
        out = scan_8k()
    else:
        out = scan_13f_changes()
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
