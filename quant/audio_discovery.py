"""音频源自动发现: 扫描 podcast RSS / Fed / 巨潮 / IR 页面 → 入 audio_queue."""
from __future__ import annotations
import argparse
import hashlib
import logging
import re
import sqlite3
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin

import feedparser
import requests

from . import config as cfg_mod
from . import db

log = logging.getLogger(__name__)


# ===== Discovery handlers per source type =====

def discover_podcast_rss(src: dict) -> list[dict]:
    """Standard podcast RSS feed - each item has enclosure with audio URL."""
    out: list[dict] = []
    try:
        d = feedparser.parse(src["url"])
    except Exception as e:  # noqa: BLE001
        log.warning("podcast %s parse failed: %s", src["name"], e)
        return out
    for entry in d.entries[:20]:
        # Find audio enclosure
        audio_url = None
        for enc in getattr(entry, "enclosures", []) or []:
            mime = enc.get("type", "")
            url = enc.get("href") or enc.get("url")
            if url and ("audio" in mime or url.endswith((".mp3", ".m4a", ".wav"))):
                audio_url = url
                break
        if not audio_url:
            # try entry.link if it's a direct media URL
            link = getattr(entry, "link", None)
            if link and link.endswith((".mp3", ".m4a", ".wav")):
                audio_url = link
        if not audio_url:
            continue
        out.append({
            "source": src["name"],
            "title": getattr(entry, "title", "(no title)"),
            "audio_url": audio_url,
            "priority": src.get("priority", 5),
        })
    return out


_FED_FOMC_PAGE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"


def discover_fed_fomc(src: dict) -> list[dict]:
    """Fed FOMC calendar page - parse for press conference media links."""
    out: list[dict] = []
    try:
        r = requests.get(_FED_FOMC_PAGE, timeout=20,
                        headers={"User-Agent": "claude-quant/1.0"})
        r.raise_for_status()
        html = r.text
    except Exception as e:  # noqa: BLE001
        log.warning("fed fomc fetch failed: %s", e)
        return out

    # Find links to press conferences (live streaming / replay)
    # Pattern: href="/monetarypolicy/fomcpresconf20260318.htm"
    for m in re.finditer(r'href="(/monetarypolicy/fomcpresconf\d{8}\.htm)"', html):
        page = "https://www.federalreserve.gov" + m.group(1)
        # date encoded in URL
        date_m = re.search(r'(\d{8})', m.group(1))
        title = f"FOMC Press Conference {date_m.group(1) if date_m else ''}"
        out.append({
            "source": src["name"],
            "title": title,
            "audio_url": page,  # We'll resolve to actual MP4 in worker
            "priority": src.get("priority", 9),
        })

    # Also look for direct .mp4 links
    for m in re.finditer(r'href="([^"]+\.mp4)"', html):
        url = m.group(1)
        if not url.startswith("http"):
            url = urljoin(_FED_FOMC_PAGE, url)
        out.append({
            "source": src["name"],
            "title": f"FOMC Media {urlparse(url).path.split('/')[-1]}",
            "audio_url": url,
            "priority": src.get("priority", 9),
        })
    return out


def discover_html_scraper(src: dict) -> list[dict]:
    """Generic HTML scraper - find audio/video links matching CSS-like selectors hint."""
    out: list[dict] = []
    try:
        r = requests.get(src["url"], timeout=20,
                        headers={"User-Agent": "claude-quant/1.0"})
        r.raise_for_status()
        html = r.text
    except Exception as e:  # noqa: BLE001
        log.warning("scraper %s failed: %s", src["name"], e)
        return out

    # Pattern: any href ending in audio extensions or containing 'webcast'
    patterns = [
        r'href="([^"]+\.mp3)"',
        r'href="([^"]+\.m4a)"',
        r'href="([^"]+\.wav)"',
        r'href="([^"]+webcast[^"]*)"',
    ]
    found: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, html, re.IGNORECASE):
            url = m.group(1)
            if not url.startswith("http"):
                url = urljoin(src["url"], url)
            found.add(url)
    for url in list(found)[:10]:
        out.append({
            "source": src["name"],
            "title": f"{src['name']}: {urlparse(url).path.split('/')[-1]}",
            "audio_url": url,
            "priority": src.get("priority", 5),
        })
    return out


HANDLERS = {
    "podcast_rss": discover_podcast_rss,
    "fed_fomc": discover_fed_fomc,
    "html_scraper": discover_html_scraper,
    "api": lambda src: [],   # not implemented yet
}


# ===== Queue & dedup =====

def enqueue(items: list[dict]) -> int:
    """Insert audio items into audio_queue, dedup by audio_url."""
    n = 0
    db.init()
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        for it in items:
            try:
                conn.execute(
                    """INSERT INTO audio_queue
                    (source, title, audio_url, discovered_at, priority, status)
                    VALUES (?,?,?,?,?,?)""",
                    (it["source"], it.get("title", "")[:200], it["audio_url"],
                     datetime.utcnow().isoformat() + "Z",
                     it.get("priority", 5), "pending"),
                )
                n += 1
            except sqlite3.IntegrityError:
                pass  # already in queue
        conn.commit()
    return n


def run_once() -> dict:
    sources = cfg_mod.load("sources").get("audio_sources", [])
    by_type: dict[str, list[dict]] = {}
    total_new = 0
    for src in sources:
        # Map specific source name 'fed_fomc' to handler
        src_type = src.get("type", "html_scraper")
        if src.get("name") == "fed_fomc":
            src_type = "fed_fomc"
        handler = HANDLERS.get(src_type)
        if not handler:
            log.warning("no handler for type=%s", src_type)
            continue
        try:
            items = handler(src)
            log.info("source=%s found %d items", src["name"], len(items))
            n = enqueue(items)
            total_new += n
            by_type.setdefault(src_type, []).extend(items)
        except Exception as e:  # noqa: BLE001
            log.exception("discovery %s failed: %s", src.get("name"), e)
    return {"new_enqueued": total_new, "by_type": {k: len(v) for k, v in by_type.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    res = run_once()
    print(res)


if __name__ == "__main__":
    main()
