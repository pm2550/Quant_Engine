"""历史相似事件检索 - 把当前事件 embed, 在 events 表里找最近邻 + LLM 回顾."""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
import struct
from datetime import datetime
from typing import Iterable

import numpy as np

from . import db, llm_router

log = logging.getLogger(__name__)


def _vec_to_bytes(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _bytes_to_vec(b: bytes) -> np.ndarray:
    n = len(b) // 4
    return np.array(struct.unpack(f"{n}f", b), dtype=np.float32)


def _store_embedding(event_id: int, text: str, embedding: list[float], model: str = "gemini-embedding-001") -> None:
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO event_embeddings
            (event_id, text, embedding, model, created_at) VALUES (?,?,?,?,?)""",
            (event_id, text[:1500], _vec_to_bytes(embedding), model,
             datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()


def index_event(event_id: int) -> bool:
    """Embed an event's title+summary; store. Skip if already done."""
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute(
            "SELECT 1 FROM event_embeddings WHERE event_id=?", (event_id,)
        ).fetchone()
        if existing:
            return False
        ev = conn.execute(
            """SELECT e.id, e.summary, e.affected_symbols, e.severity, e.fired_at,
                      n.title, n.source
            FROM events e LEFT JOIN news_archive n ON e.news_id=n.id
            WHERE e.id=?""",
            (event_id,),
        ).fetchone()
    if not ev:
        return False
    text = f"[{ev['source'] or 'event'}] {ev['title'] or ''}\n{ev['summary'] or ''}"
    try:
        vec = llm_router.embed([text])[0]
        _store_embedding(event_id, text, vec)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("embed event %d failed: %s", event_id, e)
        return False


def index_all_unembedded(*, limit: int = 100) -> int:
    """Backfill embeddings for events that don't have one yet."""
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        rows = conn.execute(
            """SELECT e.id FROM events e
            LEFT JOIN event_embeddings ee ON e.id = ee.event_id
            WHERE ee.event_id IS NULL
            ORDER BY e.fired_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    n = 0
    for (eid,) in rows:
        if index_event(eid):
            n += 1
    return n


def find_similar(query_text: str, *, top_k: int = 5,
                 min_severity: int = 4, exclude_event_id: int | None = None) -> list[dict]:
    """Embed query_text, find top-k most similar past events."""
    try:
        q_vec = np.array(llm_router.embed([query_text])[0], dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        log.warning("embed query failed: %s", e)
        return []
    if q_vec.size == 0:
        return []
    q_vec /= np.linalg.norm(q_vec) + 1e-10

    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT ee.event_id, ee.text, ee.embedding, e.severity, e.fired_at,
                      e.summary, e.impact_json, e.category, e.affected_symbols
            FROM event_embeddings ee JOIN events e ON ee.event_id = e.id
            WHERE e.severity >= ? AND e.id != ?""",
            (min_severity, exclude_event_id or -1),
        ).fetchall()

    if not rows:
        return []

    candidates = []
    for r in rows:
        v = _bytes_to_vec(r["embedding"])
        if v.size != q_vec.size:
            continue
        v_norm = v / (np.linalg.norm(v) + 1e-10)
        sim = float(q_vec @ v_norm)
        candidates.append((sim, r))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out = []
    for sim, r in candidates[:top_k]:
        out.append({
            "event_id": r["event_id"],
            "similarity": round(sim, 3),
            "fired_at": r["fired_at"],
            "severity": r["severity"],
            "category": r["category"],
            "summary": r["summary"],
            "affected": r["affected_symbols"],
            "text": r["text"][:300],
        })
    return out


def lookup_for_alert(news_title: str, summary: str, *, top_k: int = 3) -> list[dict]:
    """Used by newswatch — find similar past events to enrich an alert."""
    query = f"{news_title}\n{summary}"
    return find_similar(query, top_k=top_k, min_severity=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-all", action="store_true", help="backfill embeddings for unembedded events")
    ap.add_argument("--query", help="find similar events to this text")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.index_all:
        n = index_all_unembedded()
        print(f"indexed {n} new events")
    if args.query:
        results = find_similar(args.query, top_k=args.top_k)
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
