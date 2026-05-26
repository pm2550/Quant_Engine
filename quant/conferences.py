"""行业活动 / Investor Day - 从 conferences.yaml 读取, 入 macro_events 表."""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta

from . import config as cfg_mod, db

log = logging.getLogger(__name__)


def seed_from_yaml() -> int:
    db.init()
    n = 0
    try:
        cfg = cfg_mod.load("conferences")
    except FileNotFoundError:
        log.warning("conferences.yaml missing")
        return 0
    events = cfg.get("events", [])
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        for e in events:
            sym = e["symbol"]
            ev_type = "conference" if e.get("type") in ("industry_conference", "investor_day") else e.get("type", "conference")
            event_id = sym if sym != "__MACRO__" else "MACRO"
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO macro_events
                    (event_type, region, event_date, event_time_utc,
                     expected, actual, notes, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (f"{ev_type}-{event_id}", "global", e["date"], "",
                     e.get("impact", "medium"), "",
                     f"{e['name']}" + (f" - {e.get('notes','')}" if e.get("notes") else ""),
                     datetime.utcnow().isoformat() + "Z"),
                )
                n += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
    log.info("seeded %d conference events", n)
    return n


def upcoming(*, days: int = 30) -> list[dict]:
    today = date.today()
    end = today + timedelta(days=days)
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM macro_events WHERE event_date BETWEEN ? AND ? "
            "AND event_type LIKE '%-%' AND event_type NOT IN ('FOMC','CPI','NFP','PPI','GDP','PMI','ECB','BOJ') "
            "ORDER BY event_date",
            (today.isoformat(), end.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--upcoming", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.seed:
        n = seed_from_yaml()
        print(f"seeded {n} conferences")
    if args.upcoming:
        print(json.dumps(upcoming(days=args.days), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
