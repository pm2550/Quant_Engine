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
from datetime import date, datetime, timedelta
from pathlib import Path


log = logging.getLogger(__name__)

QLIB_PYTHON = Path("/data2/quant/qlib_env/bin/python")
PREDICTIONS_JSON = Path("/data2/quant/results/challenger_today.json")
MODEL_PATH = Path("/data2/quant/models/challenger_lgbm.txt")
MAX_STALE_HOURS = 36   # accept yesterday's predictions on failure, not older
MAX_STALE_DAYS = 3     # C6: 单个 symbol 的 as_of 落后超过这么多天就不参与排序


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


def stale_days(info: dict, *, today: date | None = None) -> int | None:
    """单条预测的陈旧天数 = today - as_of。取不到 as_of 返回 None。"""
    a = info.get("as_of")
    if not a:
        return None
    try:
        d = datetime.strptime(str(a)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    return ((today or date.today()) - d).days


def split_by_freshness(preds: dict, *, max_stale_days: int = MAX_STALE_DAYS,
                        today: date | None = None) -> tuple[dict, dict]:
    """把预测分成 (可用, 陈旧)。

    C6 (2026-09-10): 之前 freshness 只看 JSON 文件 mtime —— 那个文件每天都被重写,
    所以永远是 "fresh"; 而 as_of 取的是 dict 里第一个 symbol 的值。实际上 159 只里
    有 74 只的特征停在 2026-05-26 (价格 parquet 没再刷新, 见 A1), 于是半年前的预测
    和当天的预测被混在一起排序, "Top 5 看多"榜首 CELT 的特征日期是 2026-03-20。
    """
    fresh, stale = {}, {}
    for sym, info in (preds or {}).items():
        sd = stale_days(info, today=today)
        if sd is None or sd > max_stale_days:
            stale[sym] = info
        else:
            fresh[sym] = info
    return fresh, stale


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
    fresh, stale = split_by_freshness(preds)
    status = "fresh" if age_hours < 6 else f"cached ({age_hours:.0f}h old)"
    status = f"{status}; {len(fresh)}/{len(preds)} 只特征在 {MAX_STALE_DAYS} 天内"
    return preds, status


def persist_predictions(preds: dict, *, snapshot_date: str | None = None,
                         model: str = "challenger_lgbm") -> int:
    """把当天的预测写进 model_predictions, 供日后事后验证 (C7)。

    以前预测只存在一个每天被覆盖的 JSON 里, 所以无法算真实 IC —— 我们只有训练时的
    OOS 数字 (+0.049)。落库后 daily 复盘可以回填 realized_pct 算出实际表现。
    """
    if not preds:
        return 0
    from quant import db
    snapshot_date = snapshot_date or date.today().isoformat()
    db.init()
    n = 0
    with db.conn() as c:
        for sym, info in preds.items():
            try:
                pv = float(info["pred_forward_return"])
            except (KeyError, TypeError, ValueError):
                continue
            sd = stale_days(info, today=date.fromisoformat(snapshot_date))
            c.execute(
                """INSERT OR REPLACE INTO model_predictions
                   (snapshot_date, model, symbol, horizon_days, pred_value, as_of,
                    stale_days, extra_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (snapshot_date, model, sym, int(info.get("horizon_days", 20)), pv,
                 info.get("as_of"), sd,
                 json.dumps({k: v for k, v in info.items()
                             if k not in ("pred_forward_return", "as_of", "horizon_days")},
                            ensure_ascii=False)),
            )
            n += 1
    log.info("persisted %d challenger predictions for %s", n, snapshot_date)
    return n


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


def live_performance_line(*, model: str = "challenger_lgbm",
                           horizon_days: int = 20) -> str:
    """从 model_calibration 读**实测**表现, 绝不写死数字。

    踩过的坑 (2026-09-10): 这里原本是一行字符串字面量
        "OOS: IC +0.049 / RankIC +0.040 / TopDecile Spread +3.5%/20d"
    那组数字出自 results/challenger_v4_funds.json —— 2026-05-27 06:34 的一次离线实验。
    模型每周由 quant-challenger-retrain.timer 重训, 特征、宇宙、数据全换过几轮,
    这行数字 3.5 个月从没重算, 却一直印在每天的日报上。而同期实测逐日截面 rank IC
    是 −0.216。写死的性能指标 = 一个永远不会报错的谎。
    """
    try:
        from quant import db
        with db.conn() as c:
            row = c.execute(
                """SELECT computed_at, n_samples, daily_rank_ic, ic_positive_day_pct
                   FROM model_calibration
                   WHERE model = ? AND horizon_days = ?
                   ORDER BY computed_at DESC LIMIT 1""",
                (model, horizon_days)).fetchone()
    except Exception as e:  # noqa: BLE001
        log.warning("读取 model_calibration 失败: %s", e)
        return "实测表现: 读取失败 — 见 /api/calibration"
    if not row or row["daily_rank_ic"] is None:
        return "实测表现: 尚无校准记录 (跑 `python -m quant.calibration`)"
    ic = float(row["daily_rank_ic"])
    pos = row["ic_positive_day_pct"]
    flag = "⚠️ 方向为负, 不要据此下单" if ic < 0 else "✅"
    pos_s = f", IC 为正日占比 {float(pos):.0f}%" if pos is not None else ""
    return (f"实测逐日截面 rank IC {ic:+.3f}{pos_s} "
            f"(n={row['n_samples']}, 截至 {row['computed_at']}) {flag}")


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

    # C8 (2026-09-10): 模型输出是一个**恒为正**的水平值 (实测 156 只全部 +1.5%~+5.3%,
    # 零负值), 所以旧标题里的 "Bottom 5 (看空)" 是错的 —— 那 5 只的预测也是看涨,
    # 只是相对最弱。这个模型只能用来排序, 不能当收益预测读。
    # 另外旧标题印的 "OOS: IC +0.049" 是训练时交叉验证值; 实测逐日截面 rank IC 是
    # −0.216 (964 条已打分预测 / 69 只标的 / 45 个交易日)。印训练指标会误导。
    preds_list = [v.get("pred_forward_return", 0.0) for v in preds.values()]
    n_neg = sum(1 for p in preds_list if p < 0)
    lines = [
        "📊 *LightGBM Challenger* (Alpha158+macro+EDGAR) — **仅供排序, 非收益预测**",
        f"as_of: {as_of}, horizon {horizon}d; freshness: {freshness}",
        live_performance_line(),
        f"_本批 {len(preds_list)} 只中 {n_neg} 只为负 —— 模型存在正向水平偏移_",
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

    lines.append("\n*相对最强 5 只 (排序, 非看多):*")
    for sym, info in items[:top_k]:
        p = info["pred_forward_return"]
        marker = " ⭐持仓" if sym in held_set else ""
        lines.append(f"  {sym}{marker}: {p:+.2%}{_disagree_marker(sym, p)}")

    lines.append("\n*相对最弱 5 只 (注意: 预测值可能仍为正):*")
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
