"""Format alt-data snapshots into Markdown for daily report append.

Static rendering on purpose — we don't want the LLM repackaging numerical
trend data; risk of hallucination outweighs prose quality benefit.
"""
from __future__ import annotations
import json
import logging
import sqlite3

from .. import db
from . import bilibili

log = logging.getLogger(__name__)


_PHASE_EMOJI = {
    "early_excitement": "🚀",
    "sustained": "✅",
    "declining": "📉",
    "controversy": "⚠️",
    "unknown": "•",
}


def _format_pct(p: float | None) -> str:
    if p is None:
        return "—"
    sign = "+" if p > 0 else ""
    return f"{sign}{p:.1f}%"


def _latest_snapshot(source: str, key: str) -> dict | None:
    with db.conn() as c:
        row = c.execute(
            "SELECT metric_date, metrics_json FROM alt_data_metrics "
            "WHERE source=? AND key=? ORDER BY metric_date DESC LIMIT 1",
            (source, key),
        ).fetchone()
    if not row:
        return None
    try:
        m = json.loads(row["metrics_json"])
        m["_metric_date"] = row["metric_date"]
        return m
    except Exception:
        return None


def render_for_keyword(keyword: str, symbol: str | None = None) -> str:
    """Render one keyword's snapshot + 7d/30d trend as a Markdown block."""
    snap = _latest_snapshot("bilibili_search", keyword)
    if not snap:
        return ""
    trend = bilibili.trend(keyword)

    # Header
    sent = snap.get("sentiment") or {}
    phase = sent.get("buzz_phase", "unknown")
    emoji = _PHASE_EMOJI.get(phase, "•")
    lines = []
    label = f"`{symbol}` " if symbol else ""
    lines.append(f"### {emoji} {label}{keyword} (B 站 alt-data)")
    lines.append(f"_{snap['_metric_date']}_")
    lines.append("")

    # Volume + trend
    total = snap.get("total_results", "—")
    avg_play = snap.get("top_avg_plays", 0)
    avg_play_str = f"{avg_play/10000:.1f}万" if avg_play else "—"
    lines.append(f"- 视频总数: **{total}**, top20 均播 **{avg_play_str}**, "
                  f"最近 7 日入 top30: **{snap.get('recent_7d_in_top30', '—')}**")
    if "vs_7d_ago" in trend:
        d7 = trend["vs_7d_ago"]
        d30 = trend["vs_30d_ago"]
        lines.append(
            f"- 趋势: 总数 7d {_format_pct(d7.get('total_results_pct'))} / "
            f"30d {_format_pct(d30.get('total_results_pct'))} | "
            f"均播 7d {_format_pct(d7.get('top_avg_plays_pct'))} / "
            f"30d {_format_pct(d30.get('top_avg_plays_pct'))}"
        )

    # Sentiment + themes (if LLM ran successfully)
    if sent and "error" not in sent:
        score = sent.get("overall_sentiment")
        bd = sent.get("breakdown") or {}
        score_str = f"{score:+.2f}" if isinstance(score, (int, float)) else "—"
        lines.append(f"- 玩家社区情绪: **{score_str}** "
                      f"(正 {bd.get('positive', 0)} / 中 {bd.get('neutral', 0)} / "
                      f"负 {bd.get('negative', 0)}) — 阶段: **{phase}**")
        themes = sent.get("key_themes") or []
        if themes:
            lines.append("- 主题词: " + " · ".join(t for t in themes[:5]))
        concerns = sent.get("concern_signals") or []
        if concerns:
            lines.append("- ⚠️ 担忧: " + " · ".join(c for c in concerns[:3]))
        positives = sent.get("positive_signals") or []
        if positives:
            lines.append("- 👍 好评: " + " · ".join(p for p in positives[:3]))
        if sent.get("reasoning"):
            lines.append(f"- 一句话: _{sent['reasoning']}_")

    return "\n".join(lines)


def render_one_line(keyword: str, symbol: str | None = None) -> str:
    """一行摘要: 情绪 + 阶段 + 活指标的 7d 变化。给日报常态使用。"""
    snap = _latest_snapshot("bilibili_search", keyword)
    if not snap:
        return ""
    sent = snap.get("sentiment") or {}
    phase = sent.get("buzz_phase", "unknown")
    emoji = _PHASE_EMOJI.get(phase, "•")
    score = sent.get("overall_sentiment")
    score_str = f"{score:+.2f}" if isinstance(score, (int, float)) else "—"
    trend = bilibili.trend(keyword)
    d7 = trend.get("vs_7d_ago") or {}
    plays = _format_pct(d7.get("top_avg_plays_pct"))
    label = f"`{symbol}` " if symbol else ""
    return (f"{emoji} {label}{keyword}: 情绪 {score_str} · 阶段 {phase} · "
            f"top20 均播 7d {plays}")


def render_section(*, only_if_anomaly: bool = False, one_line: bool = False) -> str:
    """Render the alt-data section.

    E2 (2026-09-10): 这段原来无条件输出 46 行 —— 整份日报的 36%, 而内容是同一只股票
    的 4 个关键词给出 4 个互相矛盾的情绪值, 且主指标 total_results 恒为 1000 (撞 API
    分页上限, 趋势永远 0.0%)。2026-09 整月 anomaly 只触发过 1 次, 说明其余 29 天那
    46 行是纯冗余。
      only_if_anomaly=True → 近 24h 没有 alt_data_anomaly 事件时返回空串
      one_line=True        → 每个关键词一行摘要
    """
    if only_if_anomaly and not _recent_anomaly():
        log.info("alt-data: 近 24h 无异动, 跳过整段")
        return ""
    blocks = []
    for symbol, keywords in bilibili.DEFAULT_KEYWORDS.items():
        for kw in keywords:
            block = render_one_line(kw, symbol) if one_line else render_for_keyword(kw, symbol)
            if block:
                blocks.append(block)
    if not blocks:
        return ""
    if one_line:
        return "🎮 *Alt-data (B 站)*\n" + "\n".join("  " + b for b in blocks)
    header = "## 🎮 Alt-data 领先指标 (B 站社区)"
    return header + "\n\n" + "\n\n".join(blocks)


def _recent_anomaly(*, hours: int = 24) -> bool:
    """近 hours 小时内有没有**当前仍在追踪的关键词**的 alt_data_anomaly 事件。

    必须按关键词过滤: 关键词列表收敛后 (A5), 已下架关键词留下的历史异动不该再撑开
    整段。实测踩到过 —— "完美新作" 被移出追踪列表后, 它 16 小时前的 plays_drop
    仍然让整段展开成 12 行。
    """
    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    tracked = {kw for kws in bilibili.DEFAULT_KEYWORDS.values() for kw in kws}
    if not tracked:
        return False
    try:
        with db.conn() as c:
            rows = c.execute(
                "SELECT summary FROM events WHERE category='alt_data_anomaly' "
                "AND fired_at >= ?", (cutoff,)).fetchall()
    except Exception as e:  # noqa: BLE001
        log.warning("alt-data anomaly 查询失败, 按有异动处理: %s", e)
        return True
    for r in rows:
        summary = (r["summary"] if isinstance(r, sqlite3.Row) else r[0]) or ""
        if any(kw in summary for kw in tracked):
            return True
    return False
