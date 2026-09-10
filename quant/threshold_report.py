"""multi_factor 阈值扫描 —— 用已记录的决策复盘数据回答"阈值该设多少"。

为什么不是传统回测:
    composite 的因子里有 events / sentiment / analyst_ratings / alt_data, 这些都是
    point-in-time 数据, 2026-05 系统上线前不存在。硬要重建历史 composite 必然带前视
    偏差 (用今天的新闻库去算三年前的情绪分)。所以这里不做历史回测, 只做**前向验证**:
    拿系统真实产出过、并且已经到期复盘的决策来扫阈值。

    这也是为什么 backtest_tasks 里 120,153 条全是 TA 策略而 composite 一条没有 ——
    不是遗漏, 是不可做。composite 的质量证据来自 model_predictions + calibration。

口径说明 (重要):
    · 绝对收益在不同 regime 下天差地别 (2026-05~06 触发后均值 −4.8%, 07 月 +17.8%),
      不能当预期收益。可信的是"触发组 − 未触发组"的价差。
    · 30 天窗口重叠 → 有效独立样本远少于行数。目前 6 只标的 / 约 3~4 个独立期。

用法:
    python -m quant.threshold_report                 # 全样本
    python -m quant.threshold_report --split 2026-06-30   # 按日期切训练/测试
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from . import db

log = logging.getLogger(__name__)

DEFAULT_GRID = [round(x, 2) for x in np.arange(0.02, 0.32, 0.02)]
MIN_TRIGGERED = 5


def load_reviewed() -> pd.DataFrame:
    """已复盘、有 composite 和实际收益的决策。"""
    with db.conn() as c:
        df = pd.read_sql(
            """SELECT decided_at, symbol, action, composite_score, conviction,
                      actual_return_pct
               FROM decision_log
               WHERE reviewed_at IS NOT NULL
                 AND actual_return_pct IS NOT NULL
                 AND composite_score IS NOT NULL""", c)
    if not df.empty:
        df["date"] = df.decided_at.str[:10]
    return df


def sweep(df: pd.DataFrame, *, grid: list[float] | None = None) -> pd.DataFrame:
    """对每个候选阈值算触发率 / 触发后收益 / 与未触发的价差。"""
    rows = []
    for th in (grid or DEFAULT_GRID):
        sel = df[df.composite_score >= th]
        rest = df[df.composite_score < th]
        if len(sel) < MIN_TRIGGERED:
            continue
        rows.append({
            "阈值": th,
            "触发数": len(sel),
            "触发率%": round(100 * len(sel) / len(df), 1),
            "触发后均值%": round(float(sel.actual_return_pct.mean()), 2),
            "触发后中位%": round(float(sel.actual_return_pct.median()), 2),
            "未触发均值%": round(float(rest.actual_return_pct.mean()), 2) if len(rest) else None,
            "价差pp": round(float(sel.actual_return_pct.mean() - rest.actual_return_pct.mean()), 2)
                      if len(rest) else None,
            "胜率%": round(100 * float((sel.actual_return_pct > 0).mean()), 1),
        })
    return pd.DataFrame(rows)


def quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """composite 五分位 vs 后续收益 —— 看信号本身有没有区分度。"""
    if len(df) < 25:
        return pd.DataFrame()
    d = df.copy()
    try:
        d["q"] = pd.qcut(d.composite_score, 5,
                          labels=["Q1最低", "Q2", "Q3", "Q4", "Q5最高"])
    except ValueError:
        return pd.DataFrame()
    g = d.groupby("q", observed=True).agg(
        n=("actual_return_pct", "size"),
        composite均值=("composite_score", "mean"),
        收益均值=("actual_return_pct", "mean"),
        收益中位=("actual_return_pct", "median"))
    return g.round(2).reset_index()


def recommend(df: pd.DataFrame, *, split: str | None = None) -> dict:
    """挑一个两段都稳的阈值: 价差为正、且触发率在 10%~40% 之间。"""
    if split:
        tr, te = df[df.date <= split], df[df.date > split]
        segs = {"训练": sweep(tr), "测试": sweep(te)}
    else:
        segs = {"全样本": sweep(df)}
    scores: dict[float, list[float]] = {}
    for s in segs.values():
        if s.empty:
            continue
        for r in s.itertuples():
            if r.价差pp is None or not (10 <= r._3 <= 40):   # _3 = 触发率%
                continue
            scores.setdefault(r.阈值, []).append(float(r.价差pp))
    viable = {k: v for k, v in scores.items() if len(v) == len(segs) and all(x > 0 for x in v)}
    best = max(viable, key=lambda k: min(viable[k])) if viable else None
    return {"segments": segs, "viable": viable, "recommended": best}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", help="YYYY-MM-DD, 按此日期切训练/测试段")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_reviewed()
    if df.empty:
        print("没有已复盘的决策 —— 先跑 python -m quant.decision_review")
        return
    print(f"已复盘样本 {len(df)} 条, {df.symbol.nunique()} 只标的, "
          f"{df.date.min()} ~ {df.date.max()}")
    print("\n=== composite 五分位 vs 后续收益 ===")
    q = quintiles(df)
    if not q.empty:
        print(q.to_string(index=False))

    res = recommend(df, split=a.split)
    for name, s in res["segments"].items():
        print(f"\n=== 阈值扫描 · {name}段 ===")
        print(s.to_string(index=False) if not s.empty else "(样本不足)")

    from . import multi_factor
    cur = multi_factor.action_thresholds()
    print(f"\n当前配置: ADD>={cur['add']} / WATCH_BUY>={cur['watch_buy']} "
          f"/ 单日 ADD 上限 {cur['max_adds']}")
    if res["recommended"] is not None:
        print(f"扫描建议 ADD 阈值: {res['recommended']} "
              f"(各段价差 {[round(x,2) for x in res['viable'][res['recommended']]]})")
    else:
        print("扫描未找到各段都稳的阈值 —— 样本还不够, 维持现值")
    print("\n⚠️ 30 天窗口重叠, 有效独立样本远少于行数; 绝对收益受 regime 主导, 只看价差。")


if __name__ == "__main__":
    main()
