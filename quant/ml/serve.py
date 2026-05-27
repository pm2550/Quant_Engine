"""Daily-report integration shim — call out to qlib_env to refresh predictions,
then read the JSON. Keeps lightgbm out of the prod venv.

Two safety nets:
  1. subprocess call has a hard timeout (60s — full inference is ~5s)
  2. If subprocess fails, fall back to last cached JSON (with a staleness warning)
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


log = logging.getLogger(__name__)

QLIB_PYTHON = Path("/data2/quant/qlib_env/bin/python")
PREDICTIONS_JSON = Path("/data2/quant/results/challenger_today.json")
MODEL_PATH = Path("/data2/quant/models/challenger_lgbm.txt")
MAX_STALE_HOURS = 36  # accept yesterday's predictions on failure, not older


def _refresh(symbols: list[str] | None = None, *, timeout: int = 60) -> bool:
    """Spawn qlib_env to refresh predictions JSON. Returns success bool."""
    if not QLIB_PYTHON.exists():
        log.warning("qlib_env python not found at %s; skipping challenger refresh", QLIB_PYTHON)
        return False
    if not MODEL_PATH.exists():
        log.warning("challenger model missing at %s; train it via "
                    "qlib_env/bin/python -m quant.ml.challenger --train-full", MODEL_PATH)
        return False
    cmd = [str(QLIB_PYTHON), "-m", "quant.ml.predict",
           "--out", str(PREDICTIONS_JSON), "--model", str(MODEL_PATH)]
    if symbols:
        cmd.extend(["--syms", ",".join(symbols)])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout, cwd="/data2/quant")
    except subprocess.TimeoutExpired:
        log.warning("challenger refresh timed out after %ds", timeout)
        return False
    except Exception as e:  # noqa: BLE001
        log.warning("challenger subprocess failed: %s", e)
        return False
    if r.returncode != 0:
        log.warning("challenger refresh nonzero exit: stderr=%s", r.stderr[:500])
        return False
    return True


def _load_cached_predictions() -> tuple[dict | None, str]:
    """Return (preds, status) where status is 'fresh' / 'stale' / 'missing'."""
    if not PREDICTIONS_JSON.exists():
        return None, "missing"
    age_hours = (datetime.utcnow().timestamp() - PREDICTIONS_JSON.stat().st_mtime) / 3600
    if age_hours > MAX_STALE_HOURS:
        return None, "stale"
    try:
        with open(PREDICTIONS_JSON) as f:
            preds = json.load(f)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to read predictions json: %s", e)
        return None, "missing"
    status = "fresh" if age_hours < 6 else f"cached ({age_hours:.0f}h old)"
    return preds, status


def get_predictions(symbols: list[str] | None = None,
                     *, refresh: bool = True) -> tuple[dict, str]:
    """Main entry: returns (preds_dict, freshness_label).

    If refresh=True, tries to spawn qlib_env to compute new predictions first.
    Falls back to cached JSON if subprocess fails.
    """
    if refresh:
        ok = _refresh(symbols)
        if not ok:
            log.info("challenger refresh failed; using cached predictions if any")
    preds, status = _load_cached_predictions()
    return preds or {}, status


def render_section(preds: dict, *,
                    composite_actions: dict[str, str] | None = None,
                    held_symbols: list[str] | None = None,
                    top_k: int = 5,
                    freshness: str = "fresh") -> str:
    """Render TG-friendly markdown. Highlights disagreements with composite."""
    if not preds:
        return ""

    items = sorted(preds.items(), key=lambda kv: kv[1]["pred_forward_return"], reverse=True)
    horizon = next(iter(preds.values())).get("horizon_days", 20)
    as_of = next(iter(preds.values())).get("as_of", "?")
    held_set = set(held_symbols or [])

    lines = [
        "📊 *LightGBM Challenger* (Alpha158+macro+EDGAR, 161 特征)",
        f"as_of: {as_of}, 预测 {horizon}d 收益; freshness: {freshness}",
        "OOS: IC +0.049 / RankIC +0.040 / TopDecile Spread +3.5%/20d",
    ]

    def _disagree_marker(sym: str, pred: float) -> str:
        if not composite_actions:
            return ""
        a = composite_actions.get(sym)
        if a in {"REDUCE", "WATCH_SKIP"} and pred > 0.02:
            return f" ⚠️ 分歧 (composite={a})"
        if a in {"ADD", "WATCH_BUY"} and pred < 0:
            return f" ⚠️ 分歧 (composite={a})"
        return ""

    lines.append("\n*Top 5 看多:*")
    for sym, info in items[:top_k]:
        p = info["pred_forward_return"]
        marker = " ⭐持仓" if sym in held_set else ""
        lines.append(f"  {sym}{marker}: {p:+.2%}{_disagree_marker(sym, p)}")

    lines.append("\n*Bottom 5 (看空 / 跑输概率高):*")
    for sym, info in items[-top_k:][::-1]:
        p = info["pred_forward_return"]
        marker = " ⭐持仓" if sym in held_set else ""
        lines.append(f"  {sym}{marker}: {p:+.2%}{_disagree_marker(sym, p)}")

    # Held-only summary: just our positions, ordered
    if held_set:
        held_items = [(s, v) for s, v in items if s in held_set]
        if held_items:
            lines.append("\n*我们持仓的 challenger 预测:*")
            for sym, info in held_items:
                p = info["pred_forward_return"]
                lines.append(f"  {sym}: {p:+.2%}{_disagree_marker(sym, p)}")

    return "\n".join(lines)
