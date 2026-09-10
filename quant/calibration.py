"""模型校准闭环 —— 预测 vs 实际, 定期回填并打分。

为什么这个模块在 2026-09-10 才出现:
    expectations.py 的 docstring 自 2026-05-06 起就写着 "Calibration tracking is
    intentionally NOT in this module yet ... when 3+ months of data exist, we can
    run calibration analysis from that history"。数据攒了 4 个月, 分析一直没人写,
    于是两个模型带着系统性偏差跑了整整一个季度没被发现:
      · expectations bootstrap_v1 —— 20d 平均高估 9.3pp, 90% 区间实际只覆盖 71.9%
      · challenger_lgbm          —— 训练报 OOS IC +0.049, 实际逐日截面 IC −0.288

覆盖两类模型:
    分布类 (expectations): 覆盖率 / 偏差 / sigma 比
    排序类 (challenger, multi_factor composite): rank IC / 逐日截面 IC / IC 为正日占比

用法:
    python -m quant.calibration              # 打分 + 写 model_calibration + 打印
    python -m quant.calibration --no-write   # 只看不写
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import db, fetcher

log = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 120
MIN_SAMPLES = 30
MIN_SYMBOLS_PER_DAY = 4     # 逐日截面 IC 至少要这么多标的才有意义


# ---------------------------------------------------------------- 价格工具
_PX_CACHE: dict[str, pd.Series] = {}


def _closes(symbol: str) -> pd.Series | None:
    if symbol in _PX_CACHE:
        return _PX_CACHE[symbol]
    df = fetcher.load_local(symbol)
    if df is None or df.empty or "close" not in df.columns:
        _PX_CACHE[symbol] = None      # type: ignore[assignment]
        return None
    s = df["close"].astype(float)
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s.sort_index()
    _PX_CACHE[symbol] = s
    return s


def forward_return_pct(symbol: str, from_date: str, horizon_days: int) -> float | None:
    """从 from_date 收盘往后 horizon_days 个交易日的收益 (%)。

    价格不够新就返回 None —— 绝不拿"最后一根可用 K 线"冒充 horizon 后的价格,
    那正是陈旧 parquet 会悄悄产生假收益的地方 (A1)。
    """
    ser = _closes(symbol)
    if ser is None or ser.empty:
        return None
    i = ser.index.searchsorted(pd.Timestamp(from_date))
    if i >= len(ser) or i + horizon_days >= len(ser):
        return None
    return float((ser.iloc[i + horizon_days] / ser.iloc[i] - 1) * 100)


# ---------------------------------------------------------------- 打分
def score_pending(*, model: str | None = None, limit: int | None = None) -> dict:
    """给 model_predictions 里到期未打分的行回填 realized_pct。"""
    db.init()
    q = ("SELECT id, snapshot_date, model, symbol, horizon_days FROM model_predictions "
         "WHERE realized_pct IS NULL")
    params: list = []
    if model:
        q += " AND model = ?"
        params.append(model)
    q += " ORDER BY snapshot_date"
    if limit:
        q += f" LIMIT {int(limit)}"
    with db.conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(q, params).fetchall()

    scored = skipped = 0
    now = datetime.utcnow().isoformat()
    with db.conn() as c:
        for r in rows:
            rp = forward_return_pct(r["symbol"], r["snapshot_date"], int(r["horizon_days"]))
            if rp is None:
                skipped += 1
                continue
            c.execute("UPDATE model_predictions SET realized_pct = ?, scored_at = ? WHERE id = ?",
                      (rp, now, r["id"]))
            scored += 1
    log.info("score_pending: scored=%d skipped=%d (未到期或价格不足)", scored, skipped)
    return {"scored": scored, "skipped": skipped, "candidates": len(rows)}


# ---------------------------------------------------------------- 排序类模型
def _rank_ic_stats(df: pd.DataFrame, pred_col: str, real_col: str) -> dict:
    """混合 rank IC + 逐日截面 IC。

    逐日截面才是这类模型真正的用途 (今天这批股票里谁更强); 混合 IC 把时序和截面
    搅在一起, 只作参考。注意 20d 重叠窗口 → 有效独立样本远少于行数。
    """
    out = {"n_samples": len(df), "rank_ic": None,
           "daily_rank_ic": None, "ic_positive_day_pct": None, "n_days": 0}
    if len(df) >= MIN_SAMPLES:
        v = df[pred_col].corr(df[real_col], method="spearman")
        out["rank_ic"] = None if pd.isna(v) else float(v)
    ics = []
    for _, g in df.groupby("snapshot_date"):
        if g["symbol"].nunique() < MIN_SYMBOLS_PER_DAY:
            continue
        v = g[pred_col].corr(g[real_col], method="spearman")
        if pd.notna(v):
            ics.append(float(v))
    if ics:
        out["daily_rank_ic"] = float(np.mean(ics))
        out["ic_positive_day_pct"] = float(100.0 * np.mean([i > 0 for i in ics]))
        out["n_days"] = len(ics)
    return out


def calibrate_ranking(model: str, *, horizon_days: int = 20,
                       window_days: int = DEFAULT_WINDOW_DAYS) -> dict | None:
    """对排序类模型 (challenger / composite) 算实际预测力。"""
    since = (date.today() - timedelta(days=window_days)).isoformat()
    with db.conn() as c:
        df = pd.read_sql(
            """SELECT snapshot_date, symbol, pred_value, realized_pct, stale_days
               FROM model_predictions
               WHERE model = ? AND horizon_days = ? AND realized_pct IS NOT NULL
                 AND snapshot_date >= ?""",
            c, params=(model, horizon_days, since))
    if df.empty:
        return None
    res = _rank_ic_stats(df, "pred_value", "realized_pct")
    res.update(model=model, horizon_days=horizon_days, window_days=window_days,
               pred_mean_pct=float(df.pred_value.mean() * 100),
               realized_mean_pct=float(df.realized_pct.mean()),
               n_symbols=int(df.symbol.nunique()))
    # 陈旧 vs 新鲜 分开看 (若有 stale_days 信息)
    fresh = df[df.stale_days.notna() & (df.stale_days <= 3)]
    if len(fresh) >= MIN_SAMPLES:
        res["fresh_only"] = _rank_ic_stats(fresh, "pred_value", "realized_pct")
    return res


# ---------------------------------------------------------------- 分布类模型
def calibrate_expectations(*, horizon_days: int = 20,
                            window_days: int = DEFAULT_WINDOW_DAYS,
                            model_version: str = "bootstrap_v1") -> dict | None:
    """对 expectations 的区间预测算覆盖率/偏差/sigma 比。

    理想: coverage_90 = 90, coverage_50 = 50, bias = 0, sigma_ratio = 1.0
    """
    since = (date.today() - timedelta(days=window_days)).isoformat()
    with db.conn() as c:
        df = pd.read_sql(
            """SELECT snapshot_date, symbol, mean_pct, sigma_pct,
                      p5_pct, p25_pct, p75_pct, p95_pct, anchor_close
               FROM expectations
               WHERE horizon_days = ? AND model_version = ? AND snapshot_date >= ?""",
            c, params=(horizon_days, model_version, since))
    if df.empty:
        return None
    df["realized"] = [forward_return_pct(r.symbol, r.snapshot_date, horizon_days)
                      for r in df.itertuples()]
    df = df.dropna(subset=["realized"])
    if len(df) < MIN_SAMPLES:
        return None
    cov90 = float(((df.realized >= df.p5_pct) & (df.realized <= df.p95_pct)).mean() * 100)
    cov50 = float(((df.realized >= df.p25_pct) & (df.realized <= df.p75_pct)).mean() * 100)
    bias = float((df.realized - df.mean_pct).mean())
    realized_sd = float(df.realized.std())
    ratio = float(df.sigma_pct.mean() / realized_sd) if realized_sd else None
    ic = _rank_ic_stats(df.rename(columns={"mean_pct": "pred_value",
                                            "realized": "realized_pct"}),
                         "pred_value", "realized_pct")
    return {"model": model_version, "horizon_days": horizon_days,
            "window_days": window_days, "n_samples": len(df),
            "coverage_90": cov90, "coverage_50": cov50, "bias_pp": bias,
            "sigma_ratio": ratio, "n_symbols": int(df.symbol.nunique()),
            **{k: ic[k] for k in ("rank_ic", "daily_rank_ic", "ic_positive_day_pct")}}


# ---------------------------------------------------------------- 汇总
def _persist(rec: dict, *, notes: str = "") -> None:
    with db.conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO model_calibration
               (computed_at, model, horizon_days, window_days, n_samples,
                coverage_90, coverage_50, bias_pp, sigma_ratio,
                rank_ic, daily_rank_ic, ic_positive_day_pct, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (date.today().isoformat(), rec["model"], rec["horizon_days"],
             rec["window_days"], rec["n_samples"],
             rec.get("coverage_90"), rec.get("coverage_50"), rec.get("bias_pp"),
             rec.get("sigma_ratio"), rec.get("rank_ic"), rec.get("daily_rank_ic"),
             rec.get("ic_positive_day_pct"), notes or None))


def run_all(*, write: bool = True, window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
    """打分 + 全模型校准 + 落库。给 quant-calibration.timer 调。"""
    db.init()
    out: dict = {"scored": score_pending()}
    results = []
    for h in (1, 5, 20):
        r = calibrate_expectations(horizon_days=h, window_days=window_days)
        if r:
            results.append(r)
            if write:
                _persist(r, notes="bootstrap_v1 区间校准")
    for model in ("challenger_lgbm", "multi_factor_composite"):
        r = calibrate_ranking(model, window_days=window_days)
        if r:
            results.append(r)
            if write:
                _persist(r, notes="排序类预测力")
    out["results"] = results
    return out


def render_section(*, window_days: int = DEFAULT_WINDOW_DAYS) -> str:
    """周报用的紧凑段落。没有任何模型出问题时返回空串。"""
    res = run_all(write=False, window_days=window_days)["results"]
    if not res:
        return ""
    bad: list[str] = []
    for r in res:
        m, h = r["model"], r["horizon_days"]
        if r.get("coverage_90") is not None:
            c90, bias = r["coverage_90"], r["bias_pp"]
            if abs(c90 - 90) > 8 or abs(bias) > 3:
                bad.append(f"  • `{m}` {h}d: 90% 区间实际覆盖 {c90:.0f}% (应 90), "
                           f"偏差 {bias:+.1f}pp, n={r['n_samples']}")
        dic = r.get("daily_rank_ic")
        if dic is not None and dic < 0:
            bad.append(f"  • `{m}` {h}d: 逐日截面 IC {dic:+.3f} (方向为负), "
                       f"IC 为正日占比 {r.get('ic_positive_day_pct', 0):.0f}%, n={r['n_samples']}")
    if not bad:
        return "🎯 *模型校准*: 全部在容差内"
    return "🎯 *模型校准告警*\n" + "\n".join(bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_all(write=not a.no_write, window_days=a.window_days)
    print(f"打分: {out['scored']}")
    for r in out["results"]:
        print(json.dumps(r, ensure_ascii=False, default=str))
    print("\n" + render_section(window_days=a.window_days))


if __name__ == "__main__":
    main()
