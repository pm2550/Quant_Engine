"""多因子综合打分 - 替代纯技术 recommender 的死板单维度.

11 个因子各自打 -1.0 ~ +1.0, 加权汇总 -> 综合 score.
其中 'catalyst_window' 是关键: 财报临近时, 技术超买/超卖信号置信度大降.

输出:
  composite_score: -1..+1 (-1=强烈卖出, +1=强烈买入)
  conviction: 0..5 (round(|composite|*5))
  factor_breakdown: 每因子 score/weight/contribution/factors 详情
  top_factors: 按 |contribution| 排序前 3 个与 composite 同向的因子
  counter_factors: 与 composite 反向且 |contribution|>=0.02 的因子 (最多 3 条)
  rationale: 简短决策依据
"""
from __future__ import annotations
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta

from . import config as cfg_mod, db, fetcher

log = logging.getLogger(__name__)

# Factor weights (11 factors, sums to 1.0).
# 改造 (2026-05-26): 加入 alt_data / rating_change / event_intensity, 既有 8 因子权重等比缩小腾出 14 个百分点.
WEIGHTS = {
    "technical":       0.14,   # RSI/MACD/MA
    "events":          0.18,   # 完整事件图 (财报/产品/政策/SEC) 含方向+量级
    "trade_signals":   0.13,   # 成交量异动 + 期权 unusual + 做空比 (订单流)
    "sentiment":       0.10,   # 新闻 events.impact_json LLM 评级 (近 72h)
    "fundamental":     0.08,   # PE 历史分位 / 营收同比
    "analyst":         0.10,   # mean target upside + recommendation_mean 现状
    "momentum":        0.08,   # 20 日累涨 (透支警告)
    "macro_regime":    0.05,   # FOMC 临近 / 系统性 vol
    "alt_data":        0.06,   # 领先指标 (B 站舆情 sentiment + buzz_phase)
    "rating_change":   0.04,   # 评级 / 目标价 / 覆盖机构数变化 (7 日)
    "event_intensity": 0.04,   # 7 日 sev>=6 事件方向加权
}


def _technical_score(signals_dict: dict) -> tuple[float, list[str]]:
    """Score from existing signal codes. -1 (sell) ~ +1 (buy)."""
    score = 0.0
    factors = []
    codes = set(signals_dict.get("signal_codes", []))
    rsi = signals_dict.get("rsi", 50) or 50
    above_50 = signals_dict.get("above_ma50", False)
    above_200 = signals_dict.get("above_ma200", False)

    if "MACD_GOLDEN_CROSS_ABOVE_ZERO" in codes:
        score += 0.4; factors.append("MACD 零轴上金叉")
    if "MACD_DEATH_CROSS_ABOVE_ZERO" in codes:
        score -= 0.4; factors.append("MACD 零轴上死叉")
    if "RSI_EXTREME_OVERSOLD" in codes:
        score += 0.5; factors.append("RSI 极度超卖")
    elif "RSI_OVERSOLD" in codes:
        score += 0.3; factors.append("RSI 超卖")
    if "RSI_EXTREME_OVERBOUGHT" in codes:
        score -= 0.4; factors.append("RSI 极度超买")
    elif "RSI_OVERBOUGHT" in codes:
        score -= 0.2; factors.append("RSI 超买")  # 注意减弱, 不再是 -0.5
    if "BB_BREAK_LOWER" in codes:
        score += 0.2; factors.append("跌破布林下轨 (反弹候选)")
    if "BB_BREAK_UPPER" in codes:
        score -= 0.1; factors.append("突破布林上轨 (动量信号, 也警示过热)")
    if "CROSS_ABOVE_MA200" in codes:
        score += 0.4; factors.append("站上 200 日线 (大趋势转好)")
    if "CROSS_BELOW_MA200" in codes:
        score -= 0.5; factors.append("跌破 200 日线 (大趋势转弱)")

    # baseline trend
    if above_200 and above_50:
        score += 0.1; factors.append("均线多头排列")
    elif not above_200:
        score -= 0.1; factors.append("位于 200 日线下方")
    return max(-1, min(1, score)), factors


def _events_score(symbol: str) -> tuple[float, list[str], bool]:
    """Use event_aggregator to compute directional event tilt + factors + imminent flag."""
    from . import event_aggregator
    try:
        agg = event_aggregator.aggregate(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("event aggregate %s: %s", symbol, e)
        return 0.0, [], False
    factors = agg.get("summary_top3", [])
    score = max(-1, min(1, agg.get("direction_tilt", 0)))
    imminent = agg.get("has_imminent_high_mag", False)
    return score, factors, imminent


def _trade_signals_score(symbol: str) -> tuple[float, list[str]]:
    """Use trade_signals to score order flow."""
    from . import trade_signals as ts
    try:
        out = ts.aggregate(symbol)
    except Exception as e:  # noqa: BLE001
        log.warning("trade signals %s: %s", symbol, e)
        return 0.0, []
    return out.get("composite_direction_score", 0), out.get("all_factors", [])


def _catalyst_score(symbol: str, *, window_days: int = 7) -> tuple[float, list[str]]:
    """Imminent catalyst (earnings/macro). Returns volatility-premium adjusted score.
    Note: catalyst score is **NOT directional** by itself; it widens the uncertainty band.
    Output: positive if upside catalyst likely, negative if downside risk dominant.
    """
    factors = []
    score = 0.0
    today = date.today()

    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        # earnings within window
        upcoming = conn.execute(
            """SELECT * FROM earnings_calendar
            WHERE symbol=? AND eps_actual IS NULL
            AND report_date BETWEEN ? AND ?""",
            (symbol, today.isoformat(), (today + timedelta(days=window_days)).isoformat()),
        ).fetchone()
        # historical reactions (last 4)
        hist = conn.execute(
            """SELECT eps_estimate, eps_actual, surprise_pct
            FROM earnings_calendar WHERE symbol=? AND eps_actual IS NOT NULL
            ORDER BY report_date DESC LIMIT 4""",
            (symbol,),
        ).fetchall()

        # FOMC / macro within window
        macro = conn.execute(
            """SELECT * FROM macro_events
            WHERE event_date BETWEEN ? AND ? AND event_type IN ('FOMC','CPI','NFP')""",
            (today.isoformat(), (today + timedelta(days=3)).isoformat()),
        ).fetchall()

    if upcoming:
        d_until = (datetime.fromisoformat(upcoming["report_date"]).date() - today).days
        if d_until == 0:
            factors.append(f"⚡ 今日财报 (catalyst window 顶峰, 技术信号置信度大降)")
            score += 0.05  # mostly variance-widening, slight up bias from option premium
        elif d_until <= 1:
            factors.append(f"明日财报 (高 catalyst 期, 隔夜风险大)")
        else:
            factors.append(f"{d_until} 天后财报 (临近期)")

        # Historical pattern: average post-earnings reaction
        if hist:
            avg_surprise = sum(h["surprise_pct"] or 0 for h in hist) / len(hist)
            beats = sum(1 for h in hist if (h["surprise_pct"] or 0) > 0)
            if beats >= 3 and avg_surprise > 5:
                factors.append(f"过去 {len(hist)}/4 财报超预期 ({avg_surprise:+.1f}% avg surprise) — 增加上行概率")
                score += 0.15
            elif beats <= 1:
                factors.append(f"过去频繁不及预期 (beats {beats}/{len(hist)}) — 警示")
                score -= 0.15

    if macro:
        for m in macro:
            d_until = (datetime.fromisoformat(m["event_date"]).date() - today).days
            tag = "今日" if d_until == 0 else f"{d_until} 天后"
            factors.append(f"{tag} {m['region']} {m['event_type']} (跨市场 catalyst)")

    return score, factors


def _sentiment_score(symbol: str, *, hours: int = 72) -> tuple[float, list[str]]:
    """Recent news sentiment from events table - LLM-scored impacts on this symbol."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    score = 0.0
    n = 0
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        rows = conn.execute(
            "SELECT impact_json FROM events WHERE fired_at >= ? AND affected_symbols LIKE ?",
            (cutoff, f"%{symbol}%"),
        ).fetchall()
    for (impact_json,) in rows:
        try:
            imp = json.loads(impact_json) if impact_json else {}
        except Exception:
            continue
        for i in imp.get("impacts", []):
            if i.get("symbol") != symbol:
                continue
            mag = i.get("magnitude_pct", 0) or 0
            conf = i.get("confidence", 0.5) or 0.5
            score += (mag / 5.0) * conf  # normalize: 5% = 1.0 score weight
            n += 1
    factors = []
    if n > 0:
        avg = score / n
        score = max(-1, min(1, score / max(n, 3)))
        if avg > 0.3:
            factors.append(f"近期 {n} 条相关新闻情绪偏正 ({avg:+.2f})")
        elif avg < -0.3:
            factors.append(f"近期 {n} 条相关新闻情绪偏负 ({avg:+.2f})")
        else:
            factors.append(f"近期 {n} 条相关新闻情绪中性")
    return score, factors


def _fundamental_score(fundamentals_data: dict | None) -> tuple[float, list[str]]:
    """PE 历史分位 / 营收增速 → 偏多/偏空."""
    if not fundamentals_data:
        return 0.0, []
    extra = fundamentals_data.get("extra") or {}
    factors = []
    score = 0.0

    pe_pct = extra.get("pe_pct_5y")
    if pe_pct is not None:
        if pe_pct > 0.85:
            score -= 0.4; factors.append(f"PE 历史 {pe_pct*100:.0f} 分位 (估值很高)")
        elif pe_pct > 0.65:
            score -= 0.15; factors.append(f"PE 历史 {pe_pct*100:.0f} 分位 (偏高)")
        elif pe_pct < 0.20:
            score += 0.4; factors.append(f"PE 历史 {pe_pct*100:.0f} 分位 (估值很低)")
        elif pe_pct < 0.35:
            score += 0.15; factors.append(f"PE 历史 {pe_pct*100:.0f} 分位 (偏低)")

    rev_yoy = fundamentals_data.get("revenue_yoy")
    if rev_yoy is not None:
        if rev_yoy > 0.20:
            score += 0.2; factors.append(f"营收同比 +{rev_yoy*100:.0f}% (高增长)")
        elif rev_yoy < 0:
            score -= 0.15; factors.append(f"营收同比 {rev_yoy*100:.0f}% (萎缩)")

    return max(-1, min(1, score)), factors


def _analyst_score(fundamentals_data: dict | None, current_price: float | None) -> tuple[float, list[str]]:
    if not fundamentals_data:
        return 0.0, []
    extra = fundamentals_data.get("extra") or {}
    ratings = extra.get("analyst_ratings") or {}
    factors = []
    score = 0.0

    upside = ratings.get("upside_pct")
    if upside is not None and current_price:
        if upside > 20:
            score += 0.4; factors.append(f"卖方目标 upside +{upside:.0f}%")
        elif upside > 5:
            score += 0.15; factors.append(f"卖方目标 upside +{upside:.0f}%")
        elif upside < -10:
            score -= 0.3; factors.append(f"卖方目标 upside {upside:.0f}% (现价高于目标)")
        elif upside < 0:
            score -= 0.1; factors.append(f"卖方目标 upside {upside:.0f}%")

    rec_mean = ratings.get("recommendation_mean")
    if rec_mean is not None:
        if rec_mean < 1.7:
            score += 0.2; factors.append(f"分析师均值评级 {rec_mean:.2f} (强买)")
        elif rec_mean < 2.3:
            score += 0.05; factors.append(f"分析师均值评级 {rec_mean:.2f} (买入)")
        elif rec_mean > 3.0:
            score -= 0.2; factors.append(f"分析师均值评级 {rec_mean:.2f} (中性偏卖)")

    return max(-1, min(1, score)), factors


def _momentum_score(signals_dict: dict) -> tuple[float, list[str]]:
    """20/60 日涨幅 - 极端位置反向."""
    chg_20 = signals_dict.get("chg_20d_pct", 0) or 0
    factors = []
    score = 0.0
    if chg_20 > 50:
        score -= 0.3; factors.append(f"20 日 +{chg_20:.0f}% (透支严重, 回调风险高)")
    elif chg_20 > 20:
        score += 0.05; factors.append(f"20 日 +{chg_20:.0f}% (强势)")
    elif chg_20 < -20:
        score += 0.2; factors.append(f"20 日 {chg_20:.0f}% (深跌, 反弹候选)")
    elif chg_20 < -10:
        score += 0.05; factors.append(f"20 日 {chg_20:.0f}%")
    return score, factors


def _macro_regime_score() -> tuple[float, list[str]]:
    """FOMC 前 + 利率方向 + 系统性 risk-off 标记."""
    factors = []
    score = 0.0
    today = date.today()
    with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
        nearest_fomc = conn.execute(
            "SELECT event_date FROM macro_events WHERE event_type='FOMC' "
            "AND event_date >= ? ORDER BY event_date LIMIT 1",
            (today.isoformat(),),
        ).fetchone()
    if nearest_fomc:
        d = (datetime.fromisoformat(nearest_fomc[0]).date() - today).days
        if d <= 1:
            factors.append(f"FOMC {d} 天后 (跨股 vol 提升, 减仓更稳)")
            score -= 0.05
        elif d <= 3:
            factors.append(f"FOMC {d} 天后 (临近)")
    return score, factors


def _alt_data_score(symbol: str) -> tuple[float, list[str]]:
    """B 站舆情领先指标. 需要 portfolio.yaml 里 positions/watchlist 标的有 alt_data_keyword 字段.

    例: 002624.SZ -> "异环".
    无 keyword 或无数据时返回 (0, []) — score() 会把该因子权重置 0.
    """
    try:
        portfolio = cfg_mod.load("portfolio")
    except Exception:  # noqa: BLE001
        return 0.0, []
    keyword = None
    pos = (portfolio.get("positions") or {}).get(symbol) or {}
    keyword = pos.get("alt_data_keyword")
    if not keyword:
        for w in portfolio.get("watchlist") or []:
            if w.get("symbol") == symbol:
                keyword = w.get("alt_data_keyword")
                break
    if not keyword:
        return 0.0, []

    try:
        with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
            rows = conn.execute(
                "SELECT metric_date, metrics_json FROM alt_data_metrics "
                "WHERE source='bilibili_search' AND key=? ORDER BY metric_date DESC LIMIT 8",
                (keyword,),
            ).fetchall()
    except sqlite3.Error as e:
        log.warning("alt_data %s: %s", keyword, e)
        return 0.0, []
    if not rows:
        return 0.0, []

    try:
        latest = json.loads(rows[0][1])
    except (json.JSONDecodeError, TypeError):
        return 0.0, []

    sentiment_obj = latest.get("sentiment") or {}
    sent = sentiment_obj.get("overall_sentiment")
    phase = sentiment_obj.get("buzz_phase")

    factors: list[str] = []
    score = 0.0

    if isinstance(sent, (int, float)):
        if sent >= 0.5:
            score += 0.5
            factors.append(f"`{keyword}` 社区情绪 +{sent:.2f} (强正)")
        elif sent >= 0.2:
            score += 0.2
            factors.append(f"`{keyword}` 社区情绪 +{sent:.2f}")
        elif sent <= -0.3:
            score -= 0.5
            factors.append(f"`{keyword}` 社区情绪 {sent:+.2f} (负面)")

    if phase == "early_excitement":
        score += 0.3
        factors.append(f"`{keyword}` buzz_phase=early_excitement")
    elif phase == "sustained":
        score += 0.15
        factors.append(f"`{keyword}` buzz_phase=sustained")
    elif phase == "controversy":
        score -= 0.4
        factors.append(f"`{keyword}` buzz_phase=controversy")
    elif phase == "declining":
        score -= 0.3
        factors.append(f"`{keyword}` buzz_phase=declining")

    # 7-day sentiment trend (取 row[6] 或最早的)
    if len(rows) >= 5 and isinstance(sent, (int, float)):
        try:
            older = json.loads(rows[-1][1])
            old_sent = (older.get("sentiment") or {}).get("overall_sentiment")
            if isinstance(old_sent, (int, float)):
                delta = sent - old_sent
                if delta >= 0.3:
                    score += 0.2
                    factors.append(f"情绪 {len(rows)}d 上扬 {delta:+.2f}")
                elif delta <= -0.3:
                    score -= 0.2
                    factors.append(f"情绪 {len(rows)}d 下滑 {delta:+.2f}")
        except (json.JSONDecodeError, TypeError):
            pass

    return max(-1.0, min(1.0, score)), factors


def _rating_change_score(symbol: str) -> tuple[float, list[str]]:
    """跟 ~7 天前的 analyst_ratings 比, 评分给"变化"而非"现状" (后者由 _analyst_score 负责)."""
    try:
        with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
            rows = conn.execute(
                "SELECT as_of, extra_json FROM fundamentals "
                "WHERE symbol=? AND extra_json LIKE '%analyst_ratings%' "
                "ORDER BY as_of DESC LIMIT 8",
                (symbol,),
            ).fetchall()
    except sqlite3.Error:
        return 0.0, []
    if len(rows) < 2:
        return 0.0, []

    try:
        today_data = (json.loads(rows[0][1]) or {}).get("analyst_ratings") or {}
        old_data = (json.loads(rows[-1][1]) or {}).get("analyst_ratings") or {}
    except (json.JSONDecodeError, TypeError):
        return 0.0, []

    factors: list[str] = []
    score = 0.0

    t_r = today_data.get("recommendation_mean")
    o_r = old_data.get("recommendation_mean")
    if isinstance(t_r, (int, float)) and isinstance(o_r, (int, float)):
        delta = o_r - t_r  # 越小越好 (1=买入 5=卖出)
        if delta >= 0.3:
            score += 0.5
            factors.append(f"评级转好 {o_r:.2f}→{t_r:.2f}")
        elif delta <= -0.3:
            score -= 0.5
            factors.append(f"评级转弱 {o_r:.2f}→{t_r:.2f}")

    t_t = today_data.get("target_mean_price")
    o_t = old_data.get("target_mean_price")
    if isinstance(t_t, (int, float)) and isinstance(o_t, (int, float)) and o_t > 0:
        pct = (t_t - o_t) / o_t
        if pct >= 0.05:
            score += 0.3
            factors.append(f"目标价 {pct * 100:+.1f}%")
        elif pct <= -0.05:
            score -= 0.3
            factors.append(f"目标价 {pct * 100:+.1f}%")

    t_n = today_data.get("number_of_analyst_opinions") or 0
    o_n = old_data.get("number_of_analyst_opinions") or 0
    if isinstance(t_n, (int, float)) and isinstance(o_n, (int, float)):
        if t_n - o_n >= 2:
            score += 0.2
            factors.append(f"分析师覆盖 {int(o_n)}→{int(t_n)}")
        elif o_n - t_n >= 3:
            score -= 0.15
            factors.append(f"分析师流失 {int(o_n)}→{int(t_n)}")

    return max(-1.0, min(1.0, score)), factors


def _event_intensity_score(symbol: str, *, days: int = 7) -> tuple[float, list[str]]:
    """7 天内 events 表 sev>=6 事件方向加权.

    跟 _events_score 区别: 后者用 event_aggregator (综合 earnings/macro/news), 这里只看 events 表的高 sev 信号.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    bull = 0
    bear = 0
    max_sev = 0
    try:
        with sqlite3.connect(db.DB_PATH, timeout=10) as conn:
            rows = conn.execute(
                "SELECT severity, impact_json FROM events "
                "WHERE fired_at >= ? AND severity >= 6 AND affected_symbols LIKE ?",
                (cutoff, f"%{symbol}%"),
            ).fetchall()
    except sqlite3.Error:
        return 0.0, []

    for sev, impact_json in rows:
        if sev is not None:
            max_sev = max(max_sev, sev)
        try:
            imp = json.loads(impact_json) if impact_json else {}
        except (json.JSONDecodeError, TypeError):
            continue
        impacts = imp.get("impacts") or []
        for entry in impacts:
            if not isinstance(entry, dict) or entry.get("symbol") != symbol:
                continue
            direction = (entry.get("direction") or entry.get("impact_direction") or "").lower()
            if direction == "bull":
                bull += 1
            elif direction == "bear":
                bear += 1

    if bull == 0 and bear == 0:
        return 0.0, []

    net = bull - bear
    factors: list[str] = []
    score = 0.0
    if net > 0:
        if max_sev >= 8:
            score = 0.6
        elif max_sev >= 7:
            score = 0.4
        else:
            score = 0.2
        factors.append(f"7d 高烈度事件 看多 {bull} / 看空 {bear} (max sev {max_sev})")
    elif net < 0:
        if max_sev >= 7:
            score = -0.4
        else:
            score = -0.2
        factors.append(f"7d 高烈度事件 看多 {bull} / 看空 {bear} (max sev {max_sev})")

    return max(-1.0, min(1.0, score)), factors


def score(symbol: str, signals_dict: dict, fundamentals_data: dict | None = None,
          current_price: float | None = None) -> dict:
    """Compute composite multi-factor score for one symbol.

    数据缺失处理: 一个因子 factors==[] AND score==0 视为无数据, 权重置 0 并把剩余因子等比归一化.
    """
    tech, tech_f = _technical_score(signals_dict)
    events_score, events_f, catalyst_imminent = _events_score(symbol)
    trade_score, trade_f = _trade_signals_score(symbol)
    sent, sent_f = _sentiment_score(symbol)
    fund, fund_f = _fundamental_score(fundamentals_data)
    ana, ana_f = _analyst_score(fundamentals_data, current_price)
    mom, mom_f = _momentum_score(signals_dict)
    macro, macro_f = _macro_regime_score()
    alt, alt_f = _alt_data_score(symbol)
    rch, rch_f = _rating_change_score(symbol)
    eint, eint_f = _event_intensity_score(symbol)

    # Catalyst 临近: technical 权重砍半, events 提权
    weights = dict(WEIGHTS)
    if catalyst_imminent:
        weights["technical"] *= 0.5
        weights["events"] *= 1.4

    factor_data: dict[str, tuple[float, list[str]]] = {
        "technical":       (tech, tech_f),
        "events":          (events_score, events_f),
        "trade_signals":   (trade_score, trade_f),
        "sentiment":       (sent, sent_f),
        "fundamental":     (fund, fund_f),
        "analyst":         (ana, ana_f),
        "momentum":        (mom, mom_f),
        "macro_regime":    (macro, macro_f),
        "alt_data":        (alt, alt_f),
        "rating_change":   (rch, rch_f),
        "event_intensity": (eint, eint_f),
    }

    # Effective weights: 缺数据因子置 0, 其他归一化
    effective_weights = {
        k: (weights[k] if (fs or s != 0) else 0.0)
        for k, (s, fs) in factor_data.items()
    }
    total_w = sum(effective_weights.values())
    if total_w == 0:
        return {
            "composite_score": 0.0,
            "conviction": 0,
            "action": "HOLD",
            "rationale": "数据不足",
            "catalyst_imminent": False,
            "factor_breakdown": {},
            "top_factors": [],
            "counter_factors": [],
        }
    effective_weights = {k: v / total_w for k, v in effective_weights.items()}

    composite = sum(factor_data[k][0] * effective_weights[k] for k in factor_data)

    factor_breakdown: dict[str, dict] = {}
    for k, (s, fs) in factor_data.items():
        w = effective_weights[k]
        factor_breakdown[k] = {
            "score": round(s, 2),
            "factors": fs,
            "weight": round(w, 3),
            "contribution": round(s * w, 3),
        }

    # Translate composite to action — 阈值降 (从 0.40/0.15 → 0.30/0.10)
    if catalyst_imminent:
        action = "DEFER_TO_LLM"
        rationale = "临近 catalyst (财报/FOMC), 单维度信号不可靠, 交 LLM 综合判断"
    elif composite >= 0.30:
        action = "ADD"
        rationale = "多因子综合看多"
    elif composite >= 0.10:
        action = "WATCH_BUY"
        rationale = "多因子综合偏多 (置信度中等)"
    elif composite <= -0.30:
        action = "REDUCE"
        rationale = "多因子综合看空"
    elif composite <= -0.10:
        action = "WATCH_SKIP"
        rationale = "多因子综合偏空"
    else:
        action = "HOLD"
        rationale = "多因子综合中性"

    conviction = min(5, max(0, round(abs(composite) * 5)))

    # Top / counter factors by absolute contribution.
    aligned_sign = 1 if composite >= 0 else -1
    sorted_factors = sorted(
        ((k, fb) for k, fb in factor_breakdown.items() if fb["factors"]),
        key=lambda kv: abs(kv[1]["contribution"]),
        reverse=True,
    )
    top_factors: list[dict] = []
    counter_factors: list[dict] = []
    for k, fb in sorted_factors:
        evidence = "; ".join(fb["factors"][:2]) if fb["factors"] else ""
        entry = {
            "name": k,
            "score": fb["score"],
            "weight": fb["weight"],
            "contribution": fb["contribution"],
            "evidence": evidence,
        }
        signed = fb["contribution"] * aligned_sign
        if signed > 0 and len(top_factors) < 3:
            top_factors.append(entry)
        elif signed < -0.02 and len(counter_factors) < 3:
            counter_factors.append(entry)

    return {
        "composite_score": round(composite, 3),
        "conviction": conviction,
        "action": action,
        "rationale": rationale,
        "catalyst_imminent": catalyst_imminent,
        "factor_breakdown": factor_breakdown,
        "top_factors": top_factors,
        "counter_factors": counter_factors,
    }
