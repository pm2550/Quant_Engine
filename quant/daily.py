"""End-to-end daily run: signals → recommendations → LLM format → Telegram."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import config as cfg_mod
from . import orchestrator, llm_packager, telegram, decision_log as decision_log_mod

log = logging.getLogger(__name__)


def run(*, dry_run: bool = False, full_refresh: bool = False) -> None:
    raw = orchestrator.run(full_refresh=full_refresh)
    log.info("orchestrator produced %d recommendations", len(raw.get("recommendations", [])))

    # Persist non-HOLD decisions for 30-day review (Phase B-2, 2026-05-26).
    if not dry_run:
        try:
            counts = decision_log_mod.log_from_raw(raw)
            log.info("decision_log: %d logged, %d skipped", counts["logged"], counts["skipped"])
        except Exception:
            log.exception("decision_log write failed (non-fatal)")

    try:
        text = llm_packager.package(raw)
        log.info("LLM produced %d chars", len(text))
    except Exception as e:  # noqa: BLE001
        log.warning("LLM packager failed (%s); falling back to structured-only digest", e)
        # Build a minimal markdown so downstream sections (macro_regime,
        # events_digest, challenger) still get appended.
        recs = raw.get("recommendations", [])
        port = raw.get("portfolio", {})
        lines = [
            "⚠️ *LLM 包装失败 — 显示原始决策 (检查 OLLAMA_CLOUD_KEY / 限流状态)*",
            "",
            f"组合: {port.get('total_value_usd', 0):,.0f} USD, 今日 {port.get('chg_1d_pct', 0):+.2f}%",
            "",
            "*建议:*",
        ]
        for r in recs:
            sym = r.get("symbol", "?")
            act = r.get("action", "?")
            cur_w = r.get("current_weight", 0) * 100
            tgt_w = r.get("target_weight", 0) * 100
            lines.append(f"  {sym} {act}: {cur_w:.1f}% → {tgt_w:.1f}%")
        text = "\n".join(lines)

    # ---- 以下各段改为条件触发 (E4, 2026-09-10) ----------------------------
    # 之前四段无条件 append, 结果 129 行的日报里 105 行是静态段, 而真正的"操作"只有
    # 2 行 ("本日无新信号" + "持有不动")。alt-data 46 行 + challenger 28 行 = 58%
    # 的篇幅是已被实测证伪的数据。现在默认只出摘要, 详情留给周报和 API。
    sections: list[str] = []

    # 宏观: 只在 regime 分档变化 (或分数挪动 ≥10) 时出完整面板, 否则一行
    try:
        from . import macro_regime
        mr = macro_regime.render_section(only_on_change=True)
        if not mr:
            mr = macro_regime.render_section(one_line=True)
        if mr:
            sections.append(mr)
            log.info("appended macro_regime (%d 行)", len(mr.splitlines()))
    except Exception as e:  # noqa: BLE001
        log.warning("macro_regime render failed (skipping): %s", e)

    # 今日重大事件: top 2, 且 base rate 需 n>=30 才展示 (见 events_digest E6 注释)
    try:
        from . import events_digest
        ev = events_digest.render_section(top_k=2)
        if ev:
            sections.append(ev)
            log.info("appended events_digest (%d 行)", len(ev.splitlines()))
    except Exception as e:  # noqa: BLE001
        log.warning("events_digest render failed (skipping): %s", e)

    # Alt-data: 常态一行; 近 24h 真有 anomaly 才出完整块
    try:
        from .alt_data import formatter as alt_fmt
        alt = alt_fmt.render_section(only_if_anomaly=True)
        if not alt:
            alt = alt_fmt.render_section(one_line=True)
        if alt:
            sections.append(alt)
            log.info("appended alt-data (%d 行)", len(alt.splitlines()))
    except Exception as e:  # noqa: BLE001
        log.warning("alt-data render failed (skipping): %s", e)

    # LightGBM challenger: 已从日报撤出 (C5, 2026-09-10)。
    # 从 76 份日报重建 1,479 条预测 / 573 条可验证, 实测逐日截面 rank IC = −0.288,
    # IC 为正的交易日只占 9%; 训练时报的 OOS IC 是 +0.049 —— 符号相反。
    # 榜单还混着不同日期的预测 (159 只里 144 只特征陈旧, 榜首 CELT 的特征停在
    # 2026-03-20), 等于在拿半年前的预测排今天的名次。
    # 现在改为: 照常推理并落库 (model_predictions), 但不进日报。等价格刷新 (A1) 和
    # 重训之后, 用 quant.calibration 看实际 IC 转正了再决定是否恢复展示。
    try:
        from .ml import serve as challenger_serve
        preds, freshness = challenger_serve.get_predictions(refresh=True)
        if preds:
            n = challenger_serve.persist_predictions(preds)
            fresh, stale = challenger_serve.split_by_freshness(preds)
            log.info("challenger: 落库 %d 条 (新鲜 %d / 陈旧 %d), freshness=%s; 不进日报",
                      n, len(fresh), len(stale), freshness)
    except Exception as e:  # noqa: BLE001
        log.warning("challenger persist failed (non-fatal): %s", e)

    if sections:
        text = text + "\n\n" + "\n\n".join(sections)

    # 详情指引 —— 被压缩掉的内容去哪看
    text = text + "\n\n_详情: 宏观/alt-data 完整面板见周报; 全部事件 /api/events; "\
                  "模型校准 /api/calibration_"

    # Save the rendered report
    rpt_dir = cfg_mod.ROOT / "reports"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    rpt_path = rpt_dir / f"report-{datetime.utcnow().strftime('%Y%m%d')}.md"
    rpt_path.write_text(text, encoding="utf-8")
    log.info("wrote %s", rpt_path)

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

    if dry_run:
        log.info("dry-run: skip Telegram push")
        return

    portfolio = cfg_mod.load("portfolio")
    chat_id = portfolio["telegram_target"]
    try:
        res = telegram.send(text, chat_id=chat_id)
        log.info("telegram ok: message_id=%s", res.get("result", {}).get("message_id"))
    except Exception as e:  # noqa: BLE001 — report already saved; a send failure must not fail the unit
        log.error("telegram send failed (report saved at %s, not pushed): %s", rpt_path, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily quant run")
    parser.add_argument("--dry-run", action="store_true", help="don't push to Telegram")
    parser.add_argument("--refresh", action="store_true", help="full refresh of price history")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        run(dry_run=args.dry_run, full_refresh=args.refresh)
    except Exception:
        log.exception("daily run failed")
        sys.exit(1)
