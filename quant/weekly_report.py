"""周报 - 周日晚发, 综合一周事件 + P&L + 优化建议 + 风险预算."""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config as cfg_mod
from . import db, fetcher, optimizer, risk, llm_router, telegram, factor_attribution

log = logging.getLogger(__name__)


def _week_events(*, days: int = 7, min_severity: int = 5) -> list[dict]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    out: list[dict] = []
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT e.fired_at, e.severity, e.category, e.summary, e.affected_symbols,
                      n.title, n.source
            FROM events e LEFT JOIN news_archive n ON e.news_id = n.id
            WHERE e.fired_at >= ? AND e.severity >= ?
            ORDER BY e.fired_at DESC LIMIT 30""",
            (cutoff, min_severity),
        ).fetchall()
    return [dict(r) for r in rows]


def _portfolio_pnl(portfolio: dict, *, days: int = 7) -> dict:
    """Per-symbol return over last N trading days, weighted contribution."""
    held = portfolio.get("positions", {})
    out_per: dict[str, dict] = {}
    by_ccy: dict[str, dict] = {}
    for sym, info in held.items():
        df = fetcher.load_local(sym)
        if df.empty:
            continue
        c = df["close"].astype(float).tail(days + 1)
        if len(c) < 2:
            continue
        ret_pct = float((c.iloc[-1] / c.iloc[0] - 1) * 100)
        ccy = info.get("currency", "USD")
        weight = info["shares"] * float(c.iloc[-1])
        out_per[sym] = {
            "ret_pct": round(ret_pct, 2),
            "weight": weight,
            "currency": ccy,
        }
        by_ccy.setdefault(ccy, {"total": 0.0, "weighted_ret": 0.0})
        by_ccy[ccy]["total"] += weight
        by_ccy[ccy]["weighted_ret"] += weight * ret_pct
    summary = {}
    for ccy, b in by_ccy.items():
        if b["total"]:
            summary[ccy] = round(b["weighted_ret"] / b["total"], 2)
    return {"per_symbol": out_per, "by_currency_pct": summary}


def render(report_data: dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"📊 *周报 — {today}*", ""]

    # P&L
    pnl = report_data.get("pnl", {})
    if pnl.get("by_currency_pct"):
        lines.append("*📈 一周组合表现:*")
        for ccy, ret in pnl["by_currency_pct"].items():
            sign = "+" if ret >= 0 else ""
            sym = "$" if ccy == "USD" else "¥"
            lines.append(f"  {sym} {ccy}: {sign}{ret}%")
        lines.append("")

        # Top + bottom 3 contributors
        by_sym = pnl.get("per_symbol", {})
        sorted_syms = sorted(by_sym.items(), key=lambda x: x[1]["ret_pct"], reverse=True)
        if sorted_syms:
            lines.append("*🏆 表现最好:*")
            for s, d in sorted_syms[:3]:
                lines.append(f"  • `{s}` {d['ret_pct']:+.2f}%")
            lines.append("")
            if len(sorted_syms) > 3:
                lines.append("*💸 表现最差:*")
                for s, d in sorted_syms[-3:]:
                    lines.append(f"  • `{s}` {d['ret_pct']:+.2f}%")
                lines.append("")

    # Events recap
    events = report_data.get("events", [])
    if events:
        lines.append(f"*📰 本周重大事件 ({len(events)} 条):*")
        for e in events[:6]:
            d = e["fired_at"][:10]
            cat = e.get("category", "?")
            title = (e.get("title") or e.get("summary") or "")[:80]
            lines.append(f"  • {d} [sev {e['severity']}/{cat}] {title}")
        lines.append("")

    # Today's factor attribution
    attr = report_data.get("attribution_today", {})
    if attr and not attr.get("error"):
        lines.append(f"*🔬 今日归因 ({attr['portfolio_ret_today_pct']:+.2f}%):*")
        if attr.get("market_beta"):
            b = attr["market_beta"]
            lines.append(f"  β={b['beta_252d']} → 市场 {b['market_part_pct']:+.2f}% / α {b['alpha_part_pct']:+.2f}%")
        for t in attr.get("by_theme", [])[:4]:
            lines.append(f"  • {t['theme']}: {t['contribution_pct']:+.2f}% ({', '.join(t['members'][:3])})")
        lines.append("")

    # Optimizer drift
    opt = report_data.get("optimization", {})
    if opt.get("drift_vs_current", {}).get("drifts"):
        lines.append("*⚖️ 仓位优化建议 (USD bucket, max-Sharpe):*")
        for d in opt["drift_vs_current"]["drifts"][:5]:
            lines.append(
                f"  {d['action']} `{d['symbol']}` {d['current_pct']}% → {d['optimal_pct']}% "
                f"(drift {d['drift_pct']:+.1f}pp)"
            )
        lines.append("")

    # Risk budget
    rsk = report_data.get("risk", {})
    if rsk and rsk.get("var_95"):
        var95 = rsk["var_95"]["var_pct_1d"]
        var99 = rsk["var_99"]["var_pct_1d"]
        mdd = rsk.get("max_drawdown", {}).get("max_drawdown_pct", 0)
        lines.append("*🎯 风险预算 (USD):*")
        lines.append(f"  VaR 95% (1日): {var95}% | VaR 99%: {var99}%")
        lines.append(f"  最大回撤 (近期): {mdd}%")
        if rsk.get("stress_tests"):
            lines.append("  压力测试:")
            for s in rsk["stress_tests"][:3]:
                lines.append(f"    {s['scenario']}: ${s['value_change']:+.0f}")
        lines.append("")

    # LLM summary
    if report_data.get("llm_summary"):
        lines.append("*💡 一周总结:*")
        lines.append(report_data["llm_summary"][:500])
        lines.append("")

    return "\n".join(lines)


def llm_summarize(report_data: dict) -> str:
    """Have LLM write a 100-word summary of the week's performance + recommendations."""
    try:
        prompt = f"""根据以下数据, 用中文写 80-150 字总结主人这周组合状况, 给出 1-2 条最重要的建议.
不要罗列数据本身, 提炼洞察。直接输出, 不要 markdown 标题。

数据:
{json.dumps(report_data, ensure_ascii=False, default=str)[:4000]}"""
        # Use simple_chat (qwen, non-thinking) so we get clean output, not thinking traces
        out = llm_router.chat(prompt, task="simple_chat", max_tokens=400, timeout=120)
        return out["text"].strip()
    except Exception as e:  # noqa: BLE001
        log.warning("LLM summary failed: %s", e)
        return ""


def run(*, push: bool = False) -> dict:
    portfolio = cfg_mod.load("portfolio")
    pnl = _portfolio_pnl(portfolio, days=7)
    events = _week_events(days=7, min_severity=5)
    opt = optimizer.run_for_currency("USD", target="max_sharpe")
    risk_usd = risk.report("USD")
    attr = factor_attribution.attribute_today(portfolio, currency="USD")

    report_data = {
        "pnl": pnl,
        "events": events,
        "optimization": opt,
        "risk": risk_usd,
        "attribution_today": attr,
    }
    report_data["llm_summary"] = llm_summarize(report_data)

    text = render(report_data)
    out_path = cfg_mod.ROOT / "reports" / f"weekly-{datetime.utcnow().strftime('%Y%m%d')}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    log.info("wrote %s (%d chars)", out_path, len(text))

    if push:
        try:
            telegram.send(text, chat_id=portfolio["telegram_target"])
            log.info("pushed weekly report to Telegram")
        except Exception as e:  # noqa: BLE001
            log.exception("push failed: %s", e)
    return report_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="actually push to Telegram")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run(push=args.push)


if __name__ == "__main__":
    main()
