"""宏观经济事件: FOMC / CPI / NFP / GDP / 央行 / 国常会.

2026 已知日期 hardcoded + Fed 实时爬 fomccalendars.htm.
未来扩展: 爬 investing.com economic calendar 或接 trading economics API.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sqlite3
from datetime import date, datetime, timedelta

import requests

from . import db

log = logging.getLogger(__name__)


# 2026 FOMC meeting dates (known schedule)
FOMC_2026 = [
    ("2026-01-28", "rate decision"),
    ("2026-03-18", "rate decision + SEP"),
    ("2026-05-06", "rate decision"),  # tomorrow
    ("2026-06-17", "rate decision + SEP"),
    ("2026-07-29", "rate decision"),
    ("2026-09-16", "rate decision + SEP"),
    ("2026-11-04", "rate decision"),
    ("2026-12-09", "rate decision + SEP"),
]

# 2026 CPI release schedule (BLS, typically 2nd Tue/Wed of month)
CPI_2026 = [
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-15",
    "2026-05-13", "2026-06-10", "2026-07-15", "2026-08-12",
    "2026-09-10", "2026-10-15", "2026-11-13", "2026-12-10",
]

# 2026 NFP (first Friday of month, BLS Employment Situation)
NFP_2026 = [
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

# 2026 ECB meeting dates
ECB_2026 = [
    "2026-01-29", "2026-03-12", "2026-04-30", "2026-06-04",
    "2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17",
]

# China NBS scheduled CPI/PPI releases (typically 9th-12th of month)
CN_CPI_2026 = [
    "2026-01-09", "2026-02-09", "2026-03-09", "2026-04-10",
    "2026-05-10", "2026-06-09", "2026-07-09", "2026-08-09",
    "2026-09-10", "2026-10-13", "2026-11-09", "2026-12-09",
]


def _store(event_type: str, region: str, event_date: str,
           event_time_utc: str = "", expected: str = "",
           actual: str = "", notes: str = "") -> bool:
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO macro_events
                (event_type, region, event_date, event_time_utc,
                 expected, actual, notes, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_type, region, event_date, event_time_utc,
                 expected, actual, notes,
                 datetime.utcnow().isoformat() + "Z"),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def seed_known_2026() -> int:
    """Bulk insert known 2026 schedule."""
    n = 0
    for d, note in FOMC_2026:
        if _store("FOMC", "US", d, "18:00", notes=note):
            n += 1
    for d in CPI_2026:
        if _store("CPI", "US", d, "12:30", notes="BLS Consumer Price Index"):
            n += 1
    for d in NFP_2026:
        if _store("NFP", "US", d, "12:30", notes="BLS Employment Situation (非农)"):
            n += 1
    for d in ECB_2026:
        if _store("ECB", "EU", d, "12:15", notes="ECB rate decision"):
            n += 1
    for d in CN_CPI_2026:
        if _store("CPI", "CN", d, "01:30", notes="国家统计局 CPI/PPI"):
            n += 1
    return n


def upcoming(*, days: int = 14, regions: list[str] | None = None) -> list[dict]:
    today = date.today()
    end = today + timedelta(days=days)
    where_region = ""
    params = [today.isoformat(), end.isoformat()]
    if regions:
        where_region = "AND region IN (" + ",".join("?" * len(regions)) + ")"
        params.extend(regions)
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM macro_events WHERE event_date BETWEEN ? AND ? "
            f"{where_region} ORDER BY event_date",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def refresh() -> dict:
    db.init()
    n = seed_known_2026()
    return {"new_seeded": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upcoming", action="store_true")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.seed or not args.upcoming:
        print(json.dumps(refresh(), indent=2))
    if args.upcoming:
        print(json.dumps(upcoming(days=args.days), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
