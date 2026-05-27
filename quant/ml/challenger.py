"""LightGBM challenger: train on Alpha158-style features across our universe,
walk-forward CV, report IC + RankIC + top-decile spread.

Run:
    python -m quant.ml.challenger          # full pipeline
    python -m quant.ml.challenger --syms AMD,NVDA,VOO  # subset for debug
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant.ml import features as ml_features
from quant.ml import macro as ml_macro
from quant.ml import edgar as ml_edgar

PRICES_DIR = Path("/data2/quant/data/prices")


def _load_parquet(symbol: str) -> pd.DataFrame:
    p = PRICES_DIR / f"{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


HORIZON_DAYS = 20


def _load_all(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        df = _load_parquet(sym)
        if df is None or df.empty or len(df) < 300:
            continue
        out[sym] = df
    return out


def _build_dataset(price_dfs: dict[str, pd.DataFrame],
                    *, horizon_days: int = HORIZON_DAYS,
                    include_macro: bool = True,
                    include_fundamentals: bool = True) -> pd.DataFrame:
    """Returns a long-form df with (date, symbol) index + feature cols + label.

    Macro features (VIX, yields, DXY, commodities) are broadcast: same value
    for every symbol on a given date. Joins via merge_asof to handle CN
    trading-day misalignment (A-share trades when US is closed and vice versa).

    Fundamentals (SEC EDGAR XBRL) are PIT-aligned per symbol via edgar module.
    ETFs and foreign filers without CIK get NaN — LightGBM handles missing.
    """
    rows = []
    for sym, df in price_dfs.items():
        feats = ml_features.build_features(df)
        if feats.empty:
            continue
        if include_fundamentals:
            try:
                fund = ml_edgar.fundamental_features_for(sym, feats.index)
                if not fund.empty:
                    # Prefix to avoid name collisions with technical features
                    fund = fund.add_prefix("FUND_")
                    feats = feats.join(fund)
            except Exception:  # noqa: BLE001
                pass  # silently skip; LGBM handles NaN
        feats["symbol"] = sym
        feats["label"] = ml_features.forward_return_label(df, horizon_days=horizon_days)
        rows.append(feats)
    if not rows:
        return pd.DataFrame()
    big = pd.concat(rows)
    big.index.name = "date"
    big = big.set_index("symbol", append=True).reorder_levels(["date", "symbol"]).sort_index()
    big = big.dropna(subset=["label"])

    if include_macro:
        macro_df = ml_macro.load_macro_features()
        if not macro_df.empty:
            macro_df = macro_df.sort_index()
            macro_df.index = macro_df.index.tz_localize(None) if macro_df.index.tz else macro_df.index
            macro_df.index.name = "date"
            macro_reset = macro_df.reset_index()
            # merge_asof handles CN/US trading-day mismatch (forward-fill last known macro)
            tmp = big.reset_index().sort_values("date")
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.tz_localize(None)
            macro_reset["date"] = pd.to_datetime(macro_reset["date"]).dt.tz_localize(None)
            merged = pd.merge_asof(tmp, macro_reset, on="date", direction="backward")
            big = merged.set_index(["date", "symbol"]).sort_index()

    # Drop rows where most features are NaN (early warm-up)
    feat_cols = [c for c in big.columns if c != "label"]
    big = big.dropna(thresh=int(0.6 * len(feat_cols)), subset=feat_cols)
    return big


def _ic(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 5:
        return float("nan")
    return float(pd.Series(pred).corr(pd.Series(true)))


def _rank_ic(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) < 5:
        return float("nan")
    return float(pd.Series(pred).corr(pd.Series(true), method="spearman"))


def _top_decile_spread(pred: pd.Series, true: pd.Series, *,
                        quantile: float = 0.1) -> float:
    """Mean(top decile true) - Mean(bottom decile true). Higher = better signal."""
    n = len(pred)
    if n < 20:
        return float("nan")
    df = pd.DataFrame({"p": pred, "t": true})
    top = df.nlargest(int(n * quantile), "p")["t"].mean()
    bot = df.nsmallest(int(n * quantile), "p")["t"].mean()
    return float(top - bot)


def walk_forward_train(big: pd.DataFrame, *, n_folds: int = 4,
                        val_years: float = 1.0) -> dict:
    """Walk-forward CV: expanding train, fixed val_years per fold.

    For each fold we report per-day IC (correlation between predicted and
    realized forward return, computed across symbols on each day) then
    average those daily ICs. This is the same metric Qlib reports.
    """
    import lightgbm as lgb  # imported lazily so import-only doesn't need it

    big = big.sort_index()
    dates = big.index.get_level_values("date").unique().sort_values()
    if len(dates) < 252 * 3:
        return {"error": f"need >= 3y of dates, have {len(dates)} days"}

    # Reserve last n_folds * val_years for testing
    val_days = int(252 * val_years)
    total_val = val_days * n_folds
    if total_val >= len(dates):
        return {"error": f"not enough history: val_days*folds={total_val} >= total={len(dates)}"}

    fold_results = []
    cumulative_train_end = len(dates) - total_val

    params = {
        # Qlib's published LGBM Alpha158 baseline hyperparams (csi300)
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 7,
        "colsample_bytree": 0.85,
        "subsample": 0.85,
        "lambda_l1": 20.0,
        "lambda_l2": 50.0,
        "verbose": -1,
        "num_threads": 4,  # don't hog the box
    }

    feat_cols = [c for c in big.columns if c != "label"]
    print(f"dataset: {len(big):,} rows, {len(feat_cols)} features, {len(dates)} unique days")

    for fold in range(n_folds):
        train_end_idx = cumulative_train_end + fold * val_days
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_days
        if val_end_idx > len(dates):
            break

        train_end_date = dates[train_end_idx - 1]
        val_start_date = dates[val_start_idx]
        val_end_date = dates[val_end_idx - 1]

        # Crucial: skip horizon_days to prevent label leakage between train and val
        train_mask = big.index.get_level_values("date") <= (train_end_date - pd.Timedelta(days=HORIZON_DAYS + 2))
        val_mask = ((big.index.get_level_values("date") >= val_start_date)
                    & (big.index.get_level_values("date") <= val_end_date))

        train = big[train_mask]
        val = big[val_mask]
        if train.empty or val.empty:
            continue

        dtrain = lgb.Dataset(train[feat_cols].values, label=train["label"].values)
        dval = lgb.Dataset(val[feat_cols].values, label=val["label"].values, reference=dtrain)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=400,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        pred = booster.predict(val[feat_cols].values)
        val_with_pred = val.assign(pred=pred)

        # Per-day IC across symbols (Qlib-style)
        daily = val_with_pred.groupby(level="date").apply(
            lambda g: pd.Series({
                "ic": _ic(g["pred"].values, g["label"].values),
                "rank_ic": _rank_ic(g["pred"].values, g["label"].values),
                "spread": _top_decile_spread(g["pred"], g["label"]),
                "n_syms": len(g),
            })
        )

        fold_results.append({
            "fold": fold,
            "train_rows": len(train),
            "val_rows": len(val),
            "train_end": str(train_end_date.date()),
            "val_window": f"{val_start_date.date()} → {val_end_date.date()}",
            "mean_daily_ic": float(daily["ic"].mean()),
            "ic_ir": float(daily["ic"].mean() / (daily["ic"].std() + 1e-12)),
            "mean_rank_ic": float(daily["rank_ic"].mean()),
            "mean_top_decile_spread_pct": float(daily["spread"].mean() * 100),
            "best_iter": int(booster.best_iteration or 0),
        })
        print(f"  fold {fold}: train→{train_end_date.date()}, val={val_start_date.date()}→{val_end_date.date()}, "
              f"IC={fold_results[-1]['mean_daily_ic']:+.4f}  RankIC={fold_results[-1]['mean_rank_ic']:+.4f}  "
              f"Spread={fold_results[-1]['mean_top_decile_spread_pct']:+.2f}%")

    if not fold_results:
        return {"error": "no fold completed"}

    summary = {
        "n_folds": len(fold_results),
        "horizon_days": HORIZON_DAYS,
        "median_ic": float(np.median([f["mean_daily_ic"] for f in fold_results])),
        "median_rank_ic": float(np.median([f["mean_rank_ic"] for f in fold_results])),
        "median_spread_pct": float(np.median([f["mean_top_decile_spread_pct"] for f in fold_results])),
        "folds": fold_results,
        "params": params,
        "feature_count": len(feat_cols),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    }
    return summary


def train_full(big: pd.DataFrame, *, save_path: str | Path,
               num_boost_round: int = 400) -> dict:
    """Train one LGBM on ALL available data, save the booster + feature schema.

    Use this for the daily-serve model (not for evaluation — evaluation goes
    through `walk_forward_train` which is honest about OOS behavior).
    """
    import lightgbm as lgb

    feat_cols = [c for c in big.columns if c != "label"]
    big = big.sort_index().dropna(subset=["label"])

    dates = big.index.get_level_values("date").unique().sort_values()
    if len(dates) < 100:
        raise ValueError(f"need >= 100 dates, have {len(dates)}")
    val_cutoff = dates[-60]
    train_mask = big.index.get_level_values("date") < val_cutoff - pd.Timedelta(days=HORIZON_DAYS + 2)
    val_mask = big.index.get_level_values("date") >= val_cutoff
    train = big[train_mask]
    val = big[val_mask]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 7,
        "colsample_bytree": 0.85,
        "subsample": 0.85,
        "lambda_l1": 20.0,
        "lambda_l2": 50.0,
        "verbose": -1,
        "num_threads": 4,
    }
    dtrain = lgb.Dataset(train[feat_cols].values, label=train["label"].values)
    dval = lgb.Dataset(val[feat_cols].values, label=val["label"].values, reference=dtrain)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(30, verbose=False),
                                   lgb.log_evaluation(0)])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(save_path))
    schema_path = save_path.with_suffix(".features.json")
    with open(schema_path, "w") as f:
        json.dump({"features": feat_cols, "horizon_days": HORIZON_DAYS,
                   "best_iteration": int(booster.best_iteration or 0),
                   "trained_at": datetime.utcnow().isoformat() + "Z",
                   "train_rows": len(train), "val_rows": len(val)}, f, indent=2)
    return {
        "model_path": str(save_path),
        "schema_path": str(schema_path),
        "best_iteration": int(booster.best_iteration or 0),
        "train_rows": len(train),
        "val_rows": len(val),
        "n_features": len(feat_cols),
    }


def _discover_symbols() -> list[str]:
    """Union of currently cached parquet files."""
    return sorted(p.stem for p in Path("/data2/quant/data/prices").glob("*.parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", help="comma-sep symbol list (default: all cached parquet)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--val-years", type=float, default=1.0)
    ap.add_argument("--out", default="/data2/quant/results/challenger_baseline.json")
    ap.add_argument("--train-full", action="store_true",
                    help="train one model on ALL data and save (for daily serve)")
    ap.add_argument("--model-path", default="/data2/quant/models/challenger_lgbm.txt",
                    help="output booster path when --train-full is set")
    args = ap.parse_args()

    syms = args.syms.split(",") if args.syms else _discover_symbols()
    print(f"loading {len(syms)} symbols...")
    price_dfs = _load_all(syms)
    print(f"loaded {len(price_dfs)} non-empty (skipped those with < 300 bars)")

    print("building features...")
    big = _build_dataset(price_dfs)
    if big.empty:
        print("empty dataset", file=sys.stderr)
        return 1

    if args.train_full:
        print(f"train-full: saving booster to {args.model_path}")
        info = train_full(big, save_path=args.model_path)
        print(json.dumps(info, indent=2))
        return 0

    print(f"training walk-forward (n_folds={args.folds}, val_years={args.val_years})...")
    summary = walk_forward_train(big, n_folds=args.folds, val_years=args.val_years)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== SUMMARY ===")
    print(f"median IC: {summary.get('median_ic', float('nan')):+.4f}")
    print(f"median RankIC: {summary.get('median_rank_ic', float('nan')):+.4f}")
    print(f"median top-decile spread: {summary.get('median_spread_pct', float('nan')):+.2f}%")
    print(f"saved to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
