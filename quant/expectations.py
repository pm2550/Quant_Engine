"""Expected forward-return distributions — bootstrap from history.

Why this exists: the engine needs a *prior* about what each holding should do
in the next N days, so future events / price moves can be measured as
"deviation from expectation" instead of triggering on absolute thresholds.

Model v1 (bootstrap_v1): take the past `lookback_days` of daily closes,
compute every rolling N-day forward return, treat those as the empirical
distribution. No normality assumption, no GARCH — robust at small sample
sizes (we only have 13 holdings).

Calibration tracking (predicted vs realized) is intentionally NOT in this
module yet. Goal for now is: write rows daily, accumulate history. When
3+ months of data exist, we can run calibration analysis from that history.

Run:
    python -m quant.expectations           # all portfolio symbols
    python -m quant.expectations --symbol AMD
"""
from __future__ import annotations
import argparse
import logging
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as cfg_mod
from . import db, fetcher

log = logging.getLogger(__name__)

MODEL_VERSION = "bootstrap_v1"
DEFAULT_HORIZONS = (1, 5, 20)
DEFAULT_LOOKBACK = 252       # ~1 trading year
MIN_SAMPLES = 30             # below this we don't write — distribution unreliable


def bootstrap_distribution(closes: pd.Series, *, horizon_days: int,
                            lookback_days: int = DEFAULT_LOOKBACK) -> dict | None:
    """Empirical distribution of N-day forward returns.

    Take the trailing lookback_days+horizon prices, compute every rolling
    horizon-day forward return, return summary stats. Returns None if we
    can't get enough samples.
    """
    if closes is None or closes.empty:
        return None
    window = closes.tail(lookback_days + horizon_days).astype(float)
    if len(window) < horizon_days + MIN_SAMPLES:
        return None
    fwd = (window.shift(-horizon_days) / window - 1) * 100
    fwd = fwd.dropna()
    if len(fwd) < MIN_SAMPLES:
        return None
    arr = fwd.values
    return {
        "n_samples": int(len(arr)),
        "mean_pct": round(float(np.mean(arr)), 4),
        "median_pct": round(float(np.median(arr)), 4),
        "sigma_pct": round(float(np.std(arr, ddof=1)), 4),
        "p5_pct": round(float(np.percentile(arr, 5)), 4),
        "p25_pct": round(float(np.percentile(arr, 25)), 4),
        "p75_pct": round(float(np.percentile(arr, 75)), 4),
        "p95_pct": round(float(np.percentile(arr, 95)), 4),
        "min_pct": round(float(np.min(arr)), 4),
        "max_pct": round(float(np.max(arr)), 4),
    }


def snapshot_symbol(symbol: str, *, horizons: Iterable[int] = DEFAULT_HORIZONS,
                     lookback_days: int = DEFAULT_LOOKBACK,
                     snapshot_date: str | None = None) -> dict:
    """Generate + store expectation rows for one symbol across horizons.

    Returns {horizon_days: dist_dict | None} for inspection / logging.
    """
    df = fetcher.load_local(symbol)
    if df is None or df.empty:
        return {h: None for h in horizons}
    df.index = pd.to_datetime(df.index)
    closes = df["close"].astype(float)
    snap_date = snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    anchor = float(closes.iloc[-1])

    out: dict = {}
    for h in horizons:
        dist = bootstrap_distribution(closes, horizon_days=h,
                                        lookback_days=lookback_days)
        out[h] = dist
        if dist is None:
            log.info("skip %s h=%d: insufficient samples", symbol, h)
            continue
        with db.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO expectations "
                "(snapshot_date, symbol, horizon_days, model_version, lookback_days, "
                " n_samples, mean_pct, median_pct, sigma_pct, "
                " p5_pct, p25_pct, p75_pct, p95_pct, min_pct, max_pct, anchor_close) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (snap_date, symbol, int(h), MODEL_VERSION, lookback_days,
                 dist["n_samples"], dist["mean_pct"], dist["median_pct"],
                 dist["sigma_pct"], dist["p5_pct"], dist["p25_pct"],
                 dist["p75_pct"], dist["p95_pct"], dist["min_pct"],
                 dist["max_pct"], anchor),
            )
    return out


def snapshot_portfolio(*, snapshot_date: str | None = None) -> dict:
    """Snapshot every holding + watchlist symbol. Returns summary."""
    portfolio = cfg_mod.load("portfolio")
    symbols = cfg_mod.all_symbols(portfolio)
    summary = {"date": snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "n_symbols": len(symbols), "ok": [], "skipped": []}
    for sym in symbols:
        try:
            res = snapshot_symbol(sym, snapshot_date=snapshot_date)
            if any(v is not None for v in res.values()):
                summary["ok"].append({
                    "symbol": sym,
                    "horizons_with_data": [h for h, v in res.items() if v is not None],
                })
            else:
                summary["skipped"].append({"symbol": sym, "reason": "no data or insufficient samples"})
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot %s failed: %s", sym, e)
            summary["skipped"].append({"symbol": sym, "reason": repr(e)[:100]})
    return summary


def get_latest(symbol: str, *, horizon_days: int = 5,
                model_version: str = MODEL_VERSION) -> dict | None:
    """Get the most recent expectation row for a (symbol, horizon)."""
    with db.conn() as c:
        row = c.execute(
            "SELECT * FROM expectations WHERE symbol=? AND horizon_days=? "
            "  AND model_version=? ORDER BY snapshot_date DESC LIMIT 1",
            (symbol, int(horizon_days), model_version),
        ).fetchone()
    return dict(row) if row else None


def history(symbol: str, *, horizon_days: int = 5,
             model_version: str = MODEL_VERSION, limit: int = 90) -> list[dict]:
    """Time series of expectation snapshots — used for future calibration work."""
    with db.conn() as c:
        rows = c.execute(
            "SELECT * FROM expectations WHERE symbol=? AND horizon_days=? "
            "  AND model_version=? ORDER BY snapshot_date DESC LIMIT ?",
            (symbol, int(horizon_days), model_version, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", help="One symbol; default = full portfolio + watchlist")
    ap.add_argument("--horizons", default="1,5,20",
                     help="Comma-separated horizons in trading days")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s %(message)s")
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if args.symbol:
        out = snapshot_symbol(args.symbol, horizons=horizons,
                                lookback_days=args.lookback)
        import json
        print(json.dumps(out, indent=2, default=str))
    else:
        summary = snapshot_portfolio()
        log.info("snapshot done: %d ok, %d skipped",
                  len(summary["ok"]), len(summary["skipped"]))
        import json
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
