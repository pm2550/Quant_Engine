"""Format alt-data snapshots into Markdown for daily report append.

Static rendering on purpose — we don't want the LLM repackaging numerical
trend data; risk of hallucination outweighs prose quality benefit.
"""
from __future__ import annotations
import json
import logging

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


def render_section() -> str:
    """Render the full alt-data section. Pulls all known (symbol, keyword) pairs.

    Returns "" if no alt-data exists yet — daily.run skips the section.
    """
    blocks = []
    for symbol, keywords in bilibili.DEFAULT_KEYWORDS.items():
        for kw in keywords:
            block = render_for_keyword(kw, symbol)
            if block:
                blocks.append(block)
    if not blocks:
        return ""
    header = "## 🎮 Alt-data 领先指标 (B 站社区)"
    return header + "\n\n" + "\n\n".join(blocks)
