"""音频队列处理 worker - 凌晨低优先级运行。

每个任务:
  1. pending → running, 调 transcribe API
  2. 文字稿 → LLM 提要点 + 持仓影响推演
  3. 高优先级 (priority≥7) 立即推 TG
  4. 中低优先级写入早报附录 (events 表 with category=audio_summary)
  5. status → done; transcript + summary + impact 入库
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import signal
import sqlite3
import time
from datetime import datetime

import requests

from . import config as cfg_mod
from . import db, llm_router, telegram as tg

log = logging.getLogger(__name__)

API_BASE = "http://172.17.0.1:7900"

ANALYZE_PROMPT = """你是金融研究员。给定音频文字稿 + 主人持仓, 输出 JSON:

{
  "summary": "150 字内中文要点 (核心信息, 不寒暄)",
  "tone": "鹰派|鸽派|中性",
  "key_quotes": ["1-3 条最具新闻性的原话"],
  "impacts": [{"symbol":"VRT","direction":"bullish|bearish|neutral","magnitude_pct":-10..10,"reasoning":"..."}],
  "secondary_assets": [{"asset":"GLD","direction":"bullish","reasoning":"..."}],
  "action_suggestion": "针对持仓短期建议, 一句话",
  "importance": 0-10
}

主人持仓: {portfolio}
音频来源: {source}
音频标题: {title}

注意:
- impacts 中 symbol 严格限于上方持仓列表
- 没有显著影响时 impacts 可为空数组
- importance 评分: 0-3 一般, 4-6 板块级, 7-8 跨市场, 9-10 黑天鹅"""


def _portfolio_str() -> str:
    pf = cfg_mod.load("portfolio")
    return ", ".join(
        f"{sym}({info.get('name', sym)})"
        for sym, info in pf.get("positions", {}).items()
    )


def _claim_next() -> dict | None:
    """Atomically claim highest-priority pending task."""
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM audio_queue WHERE status='pending' "
            "ORDER BY priority DESC, id ASC LIMIT 1"
        ).fetchone()
        if not row:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE audio_queue SET status='running', started_at=? WHERE id=?",
            (datetime.utcnow().isoformat() + "Z", row["id"]),
        )
        conn.execute("COMMIT")
        return dict(row)


def _finish(task_id: int, *, transcript: str = "", summary: str = "",
            impact: dict | None = None, error: str | None = None) -> None:
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        if error:
            conn.execute(
                "UPDATE audio_queue SET status='failed', finished_at=?, error=? WHERE id=?",
                (datetime.utcnow().isoformat() + "Z", error[:1000], task_id),
            )
        else:
            conn.execute(
                """UPDATE audio_queue SET status='done', finished_at=?,
                transcript=?, summary=?, impact_json=? WHERE id=?""",
                (datetime.utcnow().isoformat() + "Z",
                 transcript[:50000], summary[:5000],
                 json.dumps(impact, ensure_ascii=False) if impact else None,
                 task_id),
            )
        conn.commit()


def _transcribe_via_api(audio_url: str) -> dict:
    r = requests.post(
        f"{API_BASE}/api/transcribe",
        json={"audio": audio_url, "language": "auto"},
        timeout=900,
    )
    r.raise_for_status()
    return r.json()


def _analyze(transcript: str, source: str, title: str) -> dict:
    prompt = ANALYZE_PROMPT.format(portfolio=_portfolio_str(), source=source, title=title)
    user_msg = transcript[:30000]  # cap to fit context
    return llm_router.chat_json(
        user_msg, task="reasoning", system=prompt,
        max_tokens=3000, timeout=300,
    )


def _push_alert(task: dict, summary_obj: dict) -> None:
    portfolio = cfg_mod.load("portfolio")
    impacts = summary_obj.get("impacts", [])
    lines = [
        f"🎙️ *音频要点 (重要级 {summary_obj.get('importance',0)}/10)*",
        f"📌 {task['title'][:120]}",
        f"_{task['source']}_",
        "",
        f"💡 {summary_obj.get('summary','')[:300]}",
    ]
    if summary_obj.get("tone"):
        lines.append(f"🗣️ 语气: {summary_obj['tone']}")
    if summary_obj.get("key_quotes"):
        lines.append("")
        lines.append("📝 *关键引语:*")
        for q in summary_obj["key_quotes"][:3]:
            lines.append(f"  • {q[:200]}")
    if impacts:
        lines.append("")
        lines.append("*持仓影响:*")
        for i in impacts:
            arrow = {"bullish": "📈", "bearish": "📉", "neutral": "➖"}.get(i.get("direction"), "•")
            lines.append(f"  {arrow} `{i.get('symbol')}` {i.get('magnitude_pct',0):+.1f}% — {i.get('reasoning','')[:100]}")
    if summary_obj.get("action_suggestion"):
        lines.append("")
        lines.append(f"🎯 *建议:* {summary_obj['action_suggestion']}")

    text = "\n".join(lines)
    try:
        tg.send(text, chat_id=portfolio["telegram_target"])
        log.info("pushed audio alert task=%d", task["id"])
    except Exception as e:  # noqa: BLE001
        log.exception("audio alert push failed: %s", e)


def process_one(task: dict, *, push_threshold: int = 7) -> None:
    log.info("processing task #%d source=%s priority=%d title=%s",
             task["id"], task["source"], task["priority"], task["title"][:80])
    try:
        tr = _transcribe_via_api(task["audio_url"])
        transcript = tr.get("text", "")
        if not transcript:
            _finish(task["id"], error="empty transcript")
            return
        log.info("transcript ready: %d chars (backend=%s)", len(transcript), tr.get("backend"))

        analysis = _analyze(transcript, task["source"], task["title"])
        summary = analysis.get("summary", "")
        importance = analysis.get("importance", 5)
        log.info("analysis ready: importance=%d, %d impacts", importance, len(analysis.get("impacts", [])))

        _finish(task["id"], transcript=transcript, summary=summary, impact=analysis)

        # Push if high importance OR high source priority
        effective = max(int(importance or 0), int(task["priority"] or 0))
        if effective >= push_threshold:
            _push_alert(task, analysis)
    except Exception as e:  # noqa: BLE001
        log.exception("task #%d failed: %s", task["id"], e)
        _finish(task["id"], error=str(e))


def run_loop(*, push_threshold: int = 7, idle_sleep: int = 300):
    stop = {"flag": False}

    def on_signal(*_):
        log.info("stop signal received")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    log.info("audio_queue worker started, push_threshold=%d", push_threshold)
    while not stop["flag"]:
        task = _claim_next()
        if not task:
            log.debug("queue empty, sleeping %ds", idle_sleep)
            for _ in range(idle_sleep):
                if stop["flag"]:
                    break
                time.sleep(1)
            continue
        process_one(task, push_threshold=push_threshold)
        time.sleep(2)  # gentle pacing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process one task and exit")
    ap.add_argument("--threshold", type=int, default=7)
    ap.add_argument("--idle-sleep", type=int, default=300)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.once:
        task = _claim_next()
        if task:
            process_one(task, push_threshold=args.threshold)
        else:
            print("queue empty")
    else:
        run_loop(push_threshold=args.threshold, idle_sleep=args.idle_sleep)


if __name__ == "__main__":
    main()
