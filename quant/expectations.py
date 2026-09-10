"""Expected forward-return distributions — bootstrap from history.

Why this exists: the engine needs a *prior* about what each holding should do
in the next N days, so future events / price moves can be measured as
"deviation from expectation" instead of triggering on absolute thresholds.

Model v1 (bootstrap_v1): take the past `lookback_days` of daily closes,
compute every rolling N-day forward return, treat those as the empirical
distribution. No normality assumption, no GARCH — robust at small sample
sizes (we only have 13 holdings).

Model v2 (bootstrap_v2, 2026-09-10) — v1 校准失败后的修正版:
    v1 实测 (5,141 样本): 20d 平均高估 9.3pp, 90% 区间实际只覆盖 71.9%,
    50% 区间覆盖 37.8%, 预测 sigma 只有实际的 0.73–0.92, 20d rank IC −0.24。
    两个独立的毛病:
      (a) 中心错 —— 252 天窗口正好罩住一段 AI 半导体单边上涨, 把趋势当成了稳态均值。
          DRAM 预测 +33.0% 实际 −7.3% (偏差 −40.3pp), AMKR 偏差 −28.7pp。
          修法: 中心固定为 0, 不再外推历史漂移 (实测把 20d 偏差从 −9.3pp 收到 −2.7pp)。
      (b) 宽度错 —— 经验分布低估波动聚集。
          修法: 先用 trailing 20d 实测波动把历史 forward return 标准化, 再按当前波动
          还原, 最后乘一个按 horizon 拟合的系数 k (见 SIGMA_K)。

    k 是在 2026-05-06~07-15 训练段上拟合 (令 90% 覆盖=90%), 在 07-15 之后的样本外
    验证: 1d 91.6% / 5d 91.1% / 20d 90.3% —— 样本外保持住了, 不是过拟合。

    校准追踪现在由 quant.calibration 模块自动跑 (quant-calibration.timer),
    结果写 model_calibration 表; 偏离容差会在周报里报警。

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

MODEL_VERSION = "bootstrap_v2"
LEGACY_MODEL_VERSION = "bootstrap_v1"     # 历史行保留, 供 calibration 对比
DEFAULT_HORIZONS = (1, 5, 20)
DEFAULT_LOOKBACK = 252       # ~1 trading year (v1 口径, v2 用 VOL_LOOKBACK)
VOL_LOOKBACK = 756           # v2: 波动标准化用 3 年窗口 (更多波动 regime)
VOL_WINDOW = 20              # trailing 实测波动的窗口 (交易日)
MIN_SAMPLES = 30             # below this we don't write — distribution unreliable

# 每个 horizon 的 sigma 系数 —— 用本模块**落地版**函数在 2026-05-06~07-15 训练段
# 拟合 (令 90% 区间覆盖=90%), 07-15 之后样本外验证: 1d 90.8% / 5d 90.4% / 20d 91.8%。
# 已知局限: 20d 的 50% 区间偏宽 (测试段 65%), 即分布中部形状还没对齐 —— 20d 只有
# 404 个样本且窗口重叠, 不宜再调。优先保证 90% 区间 (风险相关的那个) 准确。
# 改这里之前请先跑 `python -m quant.calibration` 看当前覆盖率。
SIGMA_K = {1: 0.98, 5: 0.88, 20: 0.73}
DEFAULT_SIGMA_K = 0.88

# 正态分位数 (v2 用参数化分位, 因为标准化后的分布已经接近对称)
_Z = {5: -1.6449, 25: -0.6745, 75: 0.6745, 95: 1.6449}


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


def vol_scaled_distribution(closes: pd.Series, *, horizon_days: int,
                             lookback_days: int = VOL_LOOKBACK,
                             vol_window: int = VOL_WINDOW,
                             sigma_k: float | None = None) -> dict | None:
    """bootstrap_v2: 零均值 + 波动标准化的 forward-return 分布。

    步骤:
      1. 取 lookback_days 的收盘, 算每日收益率
      2. trailing vol_window 实测日波动 × sqrt(horizon) → 每个时点的 h 日波动尺度
      3. 历史 h 日 forward return 除以各自时点的波动尺度 → 无量纲 z 分布
      4. 取 z 的标准差, 乘以"当前"波动尺度, 再乘 sigma_k → 本次的 sigma
      5. 中心固定 0 (不外推历史漂移), 分位数用正态 z 值

    中心为 0 不是"预测不涨"; 它表示 "我们没有可靠的 drift 估计, 别假装有"。
    v1 那个 drift 估计实测 20d 高估 9.3pp 且 rank IC 为负 —— 用它比不用它更糟。
    """
    if closes is None or closes.empty:
        return None
    k = sigma_k if sigma_k is not None else SIGMA_K.get(horizon_days, DEFAULT_SIGMA_K)
    w = closes.tail(lookback_days + horizon_days).astype(float)
    if len(w) < horizon_days + MIN_SAMPLES + vol_window:
        return None

    ret1 = w.pct_change()
    vol_h = ret1.rolling(vol_window).std() * np.sqrt(horizon_days) * 100     # %
    fwd = (w.shift(-horizon_days) / w - 1) * 100
    z = (fwd / vol_h).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < MIN_SAMPLES:
        return None

    cur_vol = vol_h.iloc[-1]
    if not np.isfinite(cur_vol) or cur_vol <= 0:
        return None

    z_sd = float(np.std(z.values, ddof=1))
    sigma = float(z_sd * cur_vol * k)
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    # 实际分布的尾部比正态厚, 所以 p5/p95 用经验 z 分位而非理论 ±1.645,
    # 但中心仍强制为 0。
    q = {p: float(np.percentile(z.values, p)) for p in (5, 25, 75, 95)}
    z_med = float(np.median(z.values))
    def at(p: int) -> float:
        return float((q[p] - z_med) * cur_vol * k)

    arr_pct = (z.values - z_med) * cur_vol * k
    return {
        "n_samples": int(len(z)),
        "mean_pct": 0.0,            # 刻意为 0 —— 见 docstring
        "median_pct": 0.0,
        "sigma_pct": round(sigma, 4),
        "p5_pct": round(at(5), 4),
        "p25_pct": round(at(25), 4),
        "p75_pct": round(at(75), 4),
        "p95_pct": round(at(95), 4),
        "min_pct": round(float(np.min(arr_pct)), 4),
        "max_pct": round(float(np.max(arr_pct)), 4),
        "sigma_k": k,
        "current_vol_pct": round(float(cur_vol), 4),
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
        # v2 为主用模型; v1 同时继续写入, 这样 quant.calibration 能持续对比两者,
        # 万一 v2 在别的 regime 下更差, 有据可依地回退。
        variants = [
            (MODEL_VERSION, vol_scaled_distribution(closes, horizon_days=h), VOL_LOOKBACK),
            (LEGACY_MODEL_VERSION,
             bootstrap_distribution(closes, horizon_days=h, lookback_days=lookback_days),
             lookback_days),
        ]
        out[h] = variants[0][1]
        for version, dist, lb in variants:
            if dist is None:
                log.info("skip %s h=%d version=%s: insufficient samples", symbol, h, version)
                continue
            with db.conn() as c:
                c.execute(
                    "INSERT OR REPLACE INTO expectations "
                    "(snapshot_date, symbol, horizon_days, model_version, lookback_days, "
                    " n_samples, mean_pct, median_pct, sigma_pct, "
                    " p5_pct, p25_pct, p75_pct, p95_pct, min_pct, max_pct, anchor_close) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (snap_date, symbol, int(h), version, lb,
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
