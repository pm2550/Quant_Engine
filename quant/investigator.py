"""主动调查异动原因 — events.category=price_action 触发, 自动找答案.

为什么 (2026-05-08 incident): anomaly_watcher 已让 002624 涨停进 events 表,
但事件只说 "002624.SZ 单日涨 +10.03%" — 不说**为什么**. 主人想知道的是 "为什么".
investigator 监听 price_action 事件 → SearXNG 搜 → LLM 总结 → 写回事件 +
推送 TG 更新.

设计:
  - 每 60s 扫 events 表 category=price_action AND impact_json LIKE %"investigation_status": "pending"%
  - 限速: 单轮最多 5 个事件 (LLM 调用避免突发)
  - SearXNG (172.17.0.1:8888): query = f"<display_name> <symbol> <direction> <date>"
  - LLM (task=deep_reasoning, kimi-k2-thinking 链首): 读 top 5 results 总结成
    `{cause: "...", confidence: "high|medium|low", sources: [url...]}`
  - 写回 events.impact_json.investigation = {status: done, ...}
  - 推 TG 一条 "事件 #N 已查到原因: ..."
  - 失败 (无搜索结果 / LLM 报错): impact_json.investigation_status = "failed", 写 reason

调用关系:
  anomaly_watcher 写 events.category=price_action (status=pending)
                ↓
  investigator 60s 后 pickup → SearXNG → LLM → 写回 done + TG push
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from . import config as cfg_mod, db, llm_router, telegram

log = logging.getLogger(__name__)

SEARXNG_URL = "http://172.17.0.1:8888/search"
INVESTIGATE_PROMPT = """你是一位金融事件研究员. 给定一只股票的异动 + 5 条 web 搜索结果,
判断"为什么"涨/跌. 输出严格 JSON:

{
  "cause": "(40 字以内, 中文, 一句话原因)",
  "confidence": "high" | "medium" | "low",
  "key_evidence": ["(引用具体一条搜索结果的标题或事实, 不超 30 字)", ...],
  "supporting_sources": [{"title": "...", "url": "..."}, ...]
}

判断原则:
1. 优先用**最近 1-2 天**的搜索结果, 越近越权威越好
2. 如果搜索结果都是过往财报/股东减持等结构性事件, confidence=low (因为不能解释当日异动)
3. 如果搜索结果直接命中 (如 "<symbol> 财报超预期 +X%" 或 "<symbol> 涨停"), confidence=high
4. 找不到合理原因 → cause="搜索结果未直接解释当日异动", confidence="low"
5. 不要编造 — 只能从搜索结果里 distill"""


def _searxng(query: str, n: int = 5, time_range: str = "week") -> list[dict]:
    """Call self-hosted SearXNG. Returns list of {title, url, content}."""
    try:
        r = requests.get(
            SEARXNG_URL,
            params={
                "q": query,
                "format": "json",
                "category_general": 1,
                "time_range": time_range,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for it in data.get("results", [])[:n]:
            results.append({
                "title": it.get("title", "")[:160],
                "url": it.get("url", ""),
                "content": (it.get("content") or "")[:400],
            })
        return results
    except Exception as e:  # noqa: BLE001
        log.warning("searxng fetch failed for %r: %s", query, e)
        return []


def _build_query(symbol: str, anomaly: dict, display_name: Optional[str]) -> str:
    """Build a focused web search query for the price action."""
    code = symbol.split(".")[0]
    a_type = anomaly.get("anomaly_type", "")
    mag = anomaly.get("magnitude_pct", 0)
    if a_type == "price_1d":
        direction = "涨停" if mag >= 9.5 else ("暴涨" if mag > 0 else "暴跌")
    elif a_type == "price_5d":
        direction = "连涨" if mag > 0 else "连跌"
    elif a_type.startswith("ma200"):
        direction = "突破年线" if a_type == "ma200_breakout" else "跌破年线"
    elif a_type.startswith("rsi"):
        direction = "技术反弹" if a_type == "rsi_flip_up" else "技术杀跌"
    elif a_type == "volume_spike":
        direction = "巨量"
    else:
        direction = "异动"
    parts = []
    if display_name and display_name != code:
        parts.append(display_name)
    parts.append(code)
    parts.append(direction)
    return " ".join(parts) + " 原因"


def _portfolio_display_name(symbol: str) -> Optional[str]:
    portfolio = cfg_mod.load("portfolio")
    info = portfolio.get("positions", {}).get(symbol)
    if info:
        return info.get("name")
    for w in portfolio.get("watchlist", []):
        if w.get("symbol") == symbol:
            return w.get("name")
    return None


def _llm_summarize(symbol: str, name: Optional[str], anomaly: dict,
                   results: list[dict]) -> dict:
    """Call LLM to produce {cause, confidence, key_evidence, supporting_sources}."""
    if not results:
        return {
            "cause": "搜索结果为空",
            "confidence": "low",
            "key_evidence": [],
            "supporting_sources": [],
            "_no_results": True,
        }
    name_str = f"{name} ({symbol})" if name else symbol
    summary = anomaly.get("summary", "")
    results_str = "\n\n".join(
        f"[{i+1}] {r['title']}\n  {r['content']}"
        for i, r in enumerate(results)
    )
    user_msg = (
        f"标的: {name_str}\n"
        f"异动: {summary}\n"
        f"\n搜索结果 ({len(results)} 条):\n{results_str}"
    )
    try:
        # task="format" routes to dashscope qwen3.6-plus with json_object mode.
        # Avoid thinking-mode (kimi-k2-thinking) which prepends prose breaking JSON parse.
        out = llm_router.chat_json(
            user_msg,
            task="format",
            system=INVESTIGATE_PROMPT,
            max_tokens=800,
            timeout=120,
        )
        # ensure fields
        out.setdefault("cause", "(LLM 未返回 cause)")
        out.setdefault("confidence", "low")
        out.setdefault("key_evidence", [])
        # backfill sources from results if LLM didn't
        if not out.get("supporting_sources"):
            out["supporting_sources"] = [
                {"title": r["title"], "url": r["url"]} for r in results[:3]
            ]
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("LLM summarize failed: %s", e)
        return {
            "cause": f"(LLM 调查失败: {str(e)[:80]})",
            "confidence": "low",
            "key_evidence": [],
            "supporting_sources": [
                {"title": r["title"], "url": r["url"]} for r in results[:3]
            ],
            "_llm_error": str(e)[:200],
        }


def _claim_pending(limit: int = 5) -> list[dict]:
    """Find events with category=price_action awaiting investigation.

    Atomically marks claimed by setting impact_json.investigation_status='running'
    so concurrent runs don't double-process.
    """
    out: list[dict] = []
    with db.conn() as c:
        rows = c.execute(
            "SELECT id, severity, summary, affected_symbols, impact_json, fired_at "
            "FROM events "
            "WHERE category='price_action' "
            "  AND impact_json LIKE '%\"investigation_status\": \"pending\"%' "
            "ORDER BY fired_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        for r in rows:
            try:
                impact = json.loads(r["impact_json"])
            except Exception:
                continue
            impact["investigation_status"] = "running"
            impact["investigation_started_at"] = datetime.utcnow().isoformat() + "Z"
            c.execute(
                "UPDATE events SET impact_json=? WHERE id=?",
                (json.dumps(impact, ensure_ascii=False), r["id"]),
            )
            out.append({
                "id": r["id"],
                "severity": r["severity"],
                "summary": r["summary"],
                "symbol": r["affected_symbols"],
                "impact": impact,
                "fired_at": r["fired_at"],
            })
        c.commit()
    return out


def _write_finding(event_id: int, finding: dict) -> None:
    with db.conn() as c:
        row = c.execute("SELECT impact_json FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            return
        try:
            impact = json.loads(row["impact_json"])
        except Exception:
            impact = {}
        impact["investigation_status"] = "done" if finding.get("cause") else "failed"
        impact["investigation_finished_at"] = datetime.utcnow().isoformat() + "Z"
        impact["investigation"] = finding
        c.execute(
            "UPDATE events SET impact_json=? WHERE id=?",
            (json.dumps(impact, ensure_ascii=False), event_id),
        )
        c.commit()


def _push_finding(event_id: int, summary: str, finding: dict, chat_id: str,
                   severity: int = 6) -> bool:
    """Send ONE human-readable Telegram message combining the anomaly + cause.

    Format goal: 主人 (non-technical) reads in 5 seconds and knows
      (1) 哪只股动了多少
      (2) 为什么动 — 一句中文
      (3) 多确定 (置信度)
    No "event #N" / "category=price_action" / "sev 7" jargon.
    """
    cause = (finding.get("cause") or "(暂未查到原因)").strip()
    conf = finding.get("confidence", "low")
    conf_zh = {"high": "比较确定", "medium": "可能", "low": "推测"}.get(conf, conf)
    emoji = "🔥" if severity >= 7 else "📈" if "涨" in summary else "📉"
    # Strip ¥/$ and "(现 ...)" tail from summary to keep it clean
    short = summary.split(" (现")[0]
    msg = f"{emoji} {short}\n原因 ({conf_zh}): {cause}"
    try:
        telegram.send(msg, chat_id=chat_id, parse_mode="")
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("telegram push failed for event %d finding: %s", event_id, e)
        return False


def run_once(*, limit: int = 5, dry_run: bool = False, push_threshold: int = 7) -> dict:
    portfolio = cfg_mod.load("portfolio")
    chat_id = str(portfolio.get("telegram_target", "6213084357"))
    pending = _claim_pending(limit=limit)
    out_findings = []
    for ev in pending:
        symbol = ev["symbol"]
        anomaly = ev["impact"]
        name = _portfolio_display_name(symbol)
        query = _build_query(symbol, anomaly, name)
        log.info("investigating event %d: %s | query=%r", ev["id"], symbol, query)
        results = _searxng(query, n=5, time_range="week")
        finding = _llm_summarize(symbol, name, anomaly, results)
        if dry_run:
            out_findings.append({
                "event_id": ev["id"], "query": query,
                "n_results": len(results), "finding": finding,
            })
            continue
        _write_finding(ev["id"], finding)
        # Push only when ALL hold:
        #   1) severity >= push_threshold (default 7 — only material moves; sev 5/6
        #      stay quiet because volume_spike / 5% moves spam-fest)
        #   2) anomaly_watcher hasn't pre-suppressed it (pushed_at is NULL — meaning
        #      sev was high enough to want push)
        #   3) we got a real cause (not "no results" / LLM error / empty cause)
        sev = ev["severity"]
        if sev < push_threshold:
            continue
        # check anomaly_watcher's suppression flag
        with db.conn() as c:
            already = c.execute(
                "SELECT pushed_at FROM events WHERE id=?", (ev["id"],),
            ).fetchone()
        if already and already["pushed_at"] is not None:
            continue   # anomaly_watcher already marked as "do not push"
        if not finding.get("cause") or finding.get("_no_results") \
           or finding.get("_llm_error"):
            continue
        if _push_finding(ev["id"], ev["summary"], finding, chat_id, severity=sev):
            with db.conn() as c:
                c.execute("UPDATE events SET pushed_at=? WHERE id=?",
                          (datetime.utcnow().isoformat() + "Z", ev["id"]))
                c.commit()
        out_findings.append({"event_id": ev["id"], "symbol": symbol,
                              "cause": finding.get("cause", "")[:80],
                              "confidence": finding.get("confidence", "")})
    return {
        "claimed": len(pending),
        "findings": out_findings,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--push-threshold", type=int, default=7,
                    help="severity ≥ this → push TG. Default 7 (only material 1d ≥10%, 5d ≥20%, MA200 break)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    db.init()

    if args.once:
        r = run_once(limit=args.limit, dry_run=args.dry_run,
                      push_threshold=args.push_threshold)
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
        return 0

    log.info("investigator loop started, interval=%ds limit=%d push_threshold=%d",
             args.interval, args.limit, args.push_threshold)
    while True:
        try:
            r = run_once(limit=args.limit, push_threshold=args.push_threshold)
            log.info("claimed=%d findings=%d", r["claimed"], len(r["findings"]))
        except Exception as e:  # noqa: BLE001
            log.exception("investigator iteration failed: %s", e)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
