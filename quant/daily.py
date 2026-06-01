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

    text = llm_packager.package(raw)
    log.info("LLM produced %d chars", len(text))

    # Consolidated 24h events digest (replaces per-event TG spam — newswatch /
    # anomaly_watcher / investigator push thresholds were bumped to 9 on
    # 2026-06-01 after the LLM-direction-hit-rate audit returned ≈50% noise).
    try:
        from . import events_digest
        ev_section = events_digest.render_section()
        if ev_section:
            text = text + "\n\n" + ev_section
            log.info("appended events_digest section (%d chars)", len(ev_section))
    except Exception as e:  # noqa: BLE001
        log.warning("events_digest render failed (skipping): %s", e)

    # Append alt-data leading-indicator section (static format, no LLM repack)
    try:
        from .alt_data import formatter as alt_fmt
        alt_section = alt_fmt.render_section()
        if alt_section:
            text = text + "\n\n" + alt_section
            log.info("appended alt-data section (%d chars)", len(alt_section))
    except Exception as e:  # noqa: BLE001
        log.warning("alt-data render failed (skipping): %s", e)

    # Append LightGBM challenger section (Phase 2 of validate-the-engine plan).
    # Runs the challenger in a separate qlib_env subprocess so prod venv doesn't
    # need lightgbm. Falls back to cached predictions if subprocess fails.
    try:
        from .ml import serve as challenger_serve
        portfolio_pos = (raw or {}).get("portfolio", {}).get("weights", {}) or {}
        held_syms = list(portfolio_pos.keys())
        rec_actions = {r.get("symbol"): r.get("action") for r in (raw or {}).get("recommendations", [])
                       if r.get("symbol")}
        preds, freshness = challenger_serve.get_predictions(refresh=True)
        if preds:
            ch_section = challenger_serve.render_section(
                preds, composite_actions=rec_actions,
                held_symbols=held_syms, freshness=freshness,
            )
            if ch_section:
                text = text + "\n\n" + ch_section
                log.info("appended challenger section (%d chars, freshness=%s)",
                         len(ch_section), freshness)
        else:
            log.info("challenger predictions unavailable; skipping section")
    except Exception as e:  # noqa: BLE001
        log.warning("challenger render failed (skipping): %s", e)

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
    res = telegram.send(text, chat_id=chat_id)
    log.info("telegram ok: message_id=%s", res.get("result", {}).get("message_id"))


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
