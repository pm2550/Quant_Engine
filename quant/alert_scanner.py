"""User-defined alert scanner.

Reads `user_alerts` table, evaluates each enabled rule against the latest
spot/signal data, fires Telegram pings on transitions, respecting
cooldown_minutes per alert.

Run via systemd timer (every 5min during market hours), or `--once` for
manual runs.

Supported rules:
  op:    '<' | '<=' | '>' | '>=' | 'cross_below' | 'cross_above'
  basis: 'last' | 'rsi' | 'ma20' | 'ma50' | 'ma200' | 'chg_1d_pct' | 'chg_20d_pct'

cross_* rules need last_seen_value to detect the actual crossing — won't
re-fire just because the price stays past the threshold.
"""
from __future__ import annotations
import argparse
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import config as cfg_mod
from . import db, fetcher, signals, telegram

log = logging.getLogger(__name__)

VALID_OPS = {"<", "<=", ">", ">=", "cross_below", "cross_above"}
VALID_BASIS = {"last", "rsi", "ma20", "ma50", "ma200", "chg_1d_pct", "chg_20d_pct"}


def _evaluate(basis: str, sig, spot_price: float | None) -> float | None:
    """Extract the basis value from a SymbolSignals object + optional spot price."""
    if basis == "last":
        return float(spot_price) if spot_price is not None else float(sig.price)
    if basis == "rsi":
        return float(sig.rsi) if sig.rsi == sig.rsi else None  # NaN check
    if basis == "ma20":
        return float(sig.ma20) if sig.ma20 == sig.ma20 else None
    if basis == "ma50":
        return float(sig.ma50) if sig.ma50 == sig.ma50 else None
    if basis == "ma200":
        return float(sig.ma200) if sig.ma200 == sig.ma200 else None
    if basis == "chg_1d_pct":
        return float(sig.chg_1d_pct)
    if basis == "chg_20d_pct":
        return float(sig.chg_20d_pct)
    return None


def _check_op(op: str, current: float, threshold: float | None,
                last_seen: float | None) -> tuple[bool, str]:
    """Return (fires, description). cross_* needs last_seen for transition detection."""
    if op == "<" and threshold is not None:
        return current < threshold, f"{current:.2f} < {threshold:.2f}"
    if op == "<=" and threshold is not None:
        return current <= threshold, f"{current:.2f} <= {threshold:.2f}"
    if op == ">" and threshold is not None:
        return current > threshold, f"{current:.2f} > {threshold:.2f}"
    if op == ">=" and threshold is not None:
        return current >= threshold, f"{current:.2f} >= {threshold:.2f}"
    if op == "cross_below" and last_seen is not None and threshold is not None:
        return last_seen >= threshold and current < threshold, \
                f"crossed below {threshold:.2f} ({last_seen:.2f} → {current:.2f})"
    if op == "cross_above" and last_seen is not None and threshold is not None:
        return last_seen <= threshold and current > threshold, \
                f"crossed above {threshold:.2f} ({last_seen:.2f} → {current:.2f})"
    return False, ""


def _cooldown_ok(fired_at: str | None, cooldown_minutes: int) -> bool:
    if not fired_at:
        return True
    try:
        last = datetime.fromisoformat(fired_at.replace("Z", "+00:00").replace("+00:00", ""))
    except Exception:
        return True
    elapsed = (datetime.utcnow() - last).total_seconds() / 60
    return elapsed >= cooldown_minutes


def _spot_for(symbol: str) -> float | None:
    """Best-effort latest spot, including post-market for US."""
    try:
        spot = fetcher.latest_spot(symbol, include_post_market=True)
        return float(spot.get("price")) if spot and spot.get("price") else None
    except Exception:
        return None


def scan_once(*, dry_run: bool = False) -> dict:
    """Scan all enabled alerts once. Returns summary {checked, fired, skipped}."""
    strategies_cfg = cfg_mod.load("strategies")
    fired: list[dict] = []
    checked = 0
    skipped_cooldown = 0

    with db.conn() as c:
        rows = c.execute(
            "SELECT * FROM user_alerts WHERE enabled = 1 ORDER BY symbol"
        ).fetchall()

    # Group by symbol to avoid re-loading the same DataFrame N times
    by_symbol: dict[str, list[dict]] = {}
    for r in rows:
        by_symbol.setdefault(r["symbol"], []).append(dict(r))

    for sym, alerts in by_symbol.items():
        df = fetcher.load_local(sym)
        if df.empty:
            log.warning("alert: no local data for %s, skipping %d alerts", sym, len(alerts))
            continue
        sig = signals.compute(sym, df, strategies_cfg)
        if sig is None:
            continue
        spot = _spot_for(sym)

        for a in alerts:
            checked += 1
            current = _evaluate(a["basis"], sig, spot)
            if current is None:
                continue
            fires, desc = _check_op(a["op"], current, a["value"], a.get("last_seen_value"))
            # Always update last_seen_value for cross_* tracking
            with db.conn() as c:
                c.execute("UPDATE user_alerts SET last_seen_value=? WHERE id=?",
                          (current, a["id"]))

            if not fires:
                continue
            if not _cooldown_ok(a.get("fired_at"), a["cooldown_minutes"]):
                skipped_cooldown += 1
                continue

            fired.append({"id": a["id"], "symbol": sym, "basis": a["basis"],
                          "op": a["op"], "current": current, "desc": desc,
                          "note": a.get("note")})

            # Send TG
            if not dry_run:
                msg = (
                    f"🔔 *自定义告警 #{a['id']}* `{sym}`\n"
                    f"  {a['basis']}: {desc}\n"
                )
                if a.get("note"):
                    msg += f"  备注: {a['note']}\n"
                msg += f"  现价 ${spot or sig.price:.2f}, RSI {sig.rsi:.0f}"
                try:
                    telegram.send(msg)
                except Exception as e:  # noqa: BLE001
                    log.warning("telegram push failed: %s", e)

                with db.conn() as c:
                    c.execute(
                        "UPDATE user_alerts SET fired_at=?, fired_count=fired_count+1 WHERE id=?",
                        (datetime.utcnow().isoformat(), a["id"]),
                    )

    return {
        "checked": checked,
        "fired": len(fired),
        "skipped_cooldown": skipped_cooldown,
        "alerts": fired,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", default=True,
                     help="Single scan (default; this script is timer-driven, not a daemon)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Evaluate rules but don't push to TG / update fired_at")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = scan_once(dry_run=args.dry_run)
    log.info("alert scan: checked=%d fired=%d skipped_cooldown=%d",
              out["checked"], out["fired"], out["skipped_cooldown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
