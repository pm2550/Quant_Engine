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

    # Append alt-data leading-indicator section (static format, no LLM repack)
    try:
        from .alt_data import formatter as alt_fmt
        alt_section = alt_fmt.render_section()
        if alt_section:
            text = text + "\n\n" + alt_section
            log.info("appended alt-data section (%d chars)", len(alt_section))
    except Exception as e:  # noqa: BLE001
        log.warning("alt-data render failed (skipping): %s", e)

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
