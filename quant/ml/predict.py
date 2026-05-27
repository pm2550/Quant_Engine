"""Daily inference for the LGBM challenger.

Loads a saved booster + feature schema, builds today's features for each
requested symbol, returns the predicted forward 20-day return.

Two consumption paths:
  - `predict_for_symbols(syms)` → dict[sym, pred] for use inside orchestrator
  - CLI: `python -m quant.ml.predict --syms AMD,NVDA` → print + write JSON
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant.ml import features as ml_features
from quant.ml import macro as ml_macro


log = logging.getLogger(__name__)

DEFAULT_MODEL = Path("/data2/quant/models/challenger_lgbm.txt")
DEFAULT_SCHEMA = Path("/data2/quant/models/challenger_lgbm.features.json")
PRICES_DIR = Path("/data2/quant/data/prices")


_MODEL_CACHE: dict = {}


def _load_model(model_path: Path = DEFAULT_MODEL,
                schema_path: Path | None = None) -> tuple:
    """Cache booster + schema in process memory (cheap, file is ~5MB)."""
    key = str(model_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import lightgbm as lgb
    schema_path = schema_path or model_path.with_suffix(".features.json")
    booster = lgb.Booster(model_file=str(model_path))
    with open(schema_path) as f:
        schema = json.load(f)
    _MODEL_CACHE[key] = (booster, schema)
    return booster, schema


def _build_features_for(symbol: str, macro_df: pd.DataFrame | None) -> pd.Series | None:
    """Compute today's feature row for one symbol. Returns None if no data."""
    p = PRICES_DIR / f"{symbol}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty or len(df) < 80:
        return None
    df.index = pd.to_datetime(df.index)
    feats = ml_features.build_features(df)
    if feats.empty:
        return None

    # Take the most recent fully-formed row (drop tail NaNs from warmup-trailing)
    feat_row = feats.dropna(how="all").iloc[-1].copy()
    asof = feats.dropna(how="all").index[-1]

    if macro_df is not None and not macro_df.empty:
        macro_df = macro_df.sort_index()
        macro_idx = pd.to_datetime(macro_df.index).tz_localize(None) if macro_df.index.tz else pd.to_datetime(macro_df.index)
        macro_df = macro_df.copy()
        macro_df.index = macro_idx
        ts = pd.Timestamp(asof).tz_localize(None) if getattr(asof, "tz", None) else pd.Timestamp(asof)
        eligible = macro_df.loc[:ts]
        if not eligible.empty:
            macro_row = eligible.iloc[-1]
            for col, val in macro_row.items():
                feat_row[col] = val

    feat_row.name = asof
    return feat_row


def predict_for_symbols(symbols: list[str],
                         model_path: Path | None = None) -> dict[str, dict]:
    """For each symbol return {pred, as_of, missing_features_count}."""
    booster, schema = _load_model(model_path or DEFAULT_MODEL)
    feat_cols: list[str] = schema["features"]
    horizon = schema.get("horizon_days", 20)

    macro_df = ml_macro.load_macro_features()

    rows = []
    meta = []
    keep_syms = []
    for sym in symbols:
        row = _build_features_for(sym, macro_df)
        if row is None:
            continue
        row_vec = np.array([row.get(c, np.nan) for c in feat_cols], dtype=np.float64)
        n_missing = int(np.isnan(row_vec).sum())
        if n_missing == len(feat_cols):
            continue
        rows.append(row_vec)
        meta.append({"as_of": str(row.name)[:10], "missing": n_missing})
        keep_syms.append(sym)
    if not rows:
        return {}

    X = np.vstack(rows)
    preds = booster.predict(X)

    out: dict[str, dict] = {}
    for sym, p, m in zip(keep_syms, preds, meta):
        out[sym] = {
            "pred_forward_return": float(p),
            "horizon_days": horizon,
            "as_of": m["as_of"],
            "missing_features": m["missing"],
            "n_features": len(feat_cols),
        }
    return out


def render_tg_section(preds: dict[str, dict], *,
                       composite_actions: dict[str, str] | None = None,
                       top_k: int = 5) -> str:
    """Format predictions into a TG-friendly markdown section.

    Highlights:
      - top_k symbols by predicted forward return (bullish picks)
      - bottom_k by predicted return (bearish flags)
      - disagreement signals (challenger says BUY but composite says REDUCE)
    """
    if not preds:
        return ""

    items = sorted(preds.items(), key=lambda kv: kv[1]["pred_forward_return"], reverse=True)
    horizon = next(iter(preds.values()))["horizon_days"]
    as_of = next(iter(preds.values()))["as_of"]

    lines = [f"📊 *LightGBM Challenger* (Alpha158 + macro, 144+7 特征, OOS IC ≈ 0.058)",
             f"as_of: {as_of}, 预测 {horizon}d 收益率"]

    lines.append("\n*Top 看多:*")
    for sym, info in items[:top_k]:
        p = info["pred_forward_return"]
        marker = ""
        if composite_actions:
            comp = composite_actions.get(sym, "")
            if comp in {"REDUCE", "WATCH_SKIP"} and p > 0.02:
                marker = " ⚠️ 分歧 (composite=" + comp + ")"
        lines.append(f"  {sym}: {p:+.2%}{marker}")

    lines.append("\n*Top 看空:*")
    for sym, info in items[-top_k:][::-1]:
        p = info["pred_forward_return"]
        marker = ""
        if composite_actions:
            comp = composite_actions.get(sym, "")
            if comp in {"ADD", "WATCH_BUY"} and p < -0.02:
                marker = " ⚠️ 分歧 (composite=" + comp + ")"
        lines.append(f"  {sym}: {p:+.2%}{marker}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", help="comma-sep, default: all cached parquet")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="/data2/quant/results/challenger_today.json")
    args = ap.parse_args()

    if args.syms:
        syms = args.syms.split(",")
    else:
        syms = sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))

    preds = predict_for_symbols(syms, model_path=Path(args.model))
    if not preds:
        print("no predictions (missing model or features)", file=sys.stderr)
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(preds, f, indent=2)

    print(render_tg_section(preds, top_k=args.top_k))
    print(f"\nfull predictions written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
