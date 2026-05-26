"""Unit tests for multi_factor.py — the 11-factor composite scoring (post 2026-05-26 update)."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

from quant import multi_factor


# --------- shared helpers for DB-touching factors ---------

@contextmanager
def _temp_db(monkeypatch, tmp_path):
    """Point multi_factor.db.DB_PATH at a fresh sqlite file and create needed tables."""
    db_path = tmp_path / "test_quant.sqlite"
    monkeypatch.setattr(multi_factor.db, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE alt_data_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                key TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                metric_date TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                UNIQUE(source, key, metric_date)
            );
            CREATE TABLE fundamentals (
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                extra_json TEXT,
                PRIMARY KEY (symbol, as_of)
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                severity INTEGER NOT NULL,
                category TEXT,
                summary TEXT NOT NULL,
                impact_json TEXT,
                affected_symbols TEXT,
                fired_at TEXT NOT NULL
            );
            CREATE TABLE macro_events (
                event_date TEXT NOT NULL,
                event_type TEXT NOT NULL,
                region TEXT
            );
            CREATE TABLE earnings_calendar (
                symbol TEXT NOT NULL,
                report_date TEXT NOT NULL,
                eps_estimate REAL,
                eps_actual REAL,
                surprise_pct REAL
            );
            """
        )
    yield db_path


def test_technical_score_extreme_oversold_positive():
    s = {"signal_codes": ["RSI_EXTREME_OVERSOLD", "BB_BREAK_LOWER"], "rsi": 18,
         "above_ma50": True, "above_ma200": True}
    score, factors = multi_factor._technical_score(s)
    assert score > 0.5, f"oversold should produce strong buy signal, got {score}"


def test_technical_score_overbought_negative_but_not_too_extreme():
    """RSI overbought used to score -0.5, now should be milder (-0.2)."""
    s = {"signal_codes": ["RSI_OVERBOUGHT"], "rsi": 73,
         "above_ma50": True, "above_ma200": True}
    score, factors = multi_factor._technical_score(s)
    # New design: RSI_OVERBOUGHT alone shouldn't tank score below -0.2
    assert -0.3 < score < 0.2, f"RSI overbought + multi-trend bullish should be near zero, got {score}"


def test_technical_cross_below_ma200_strongly_negative():
    s = {"signal_codes": ["CROSS_BELOW_MA200"], "rsi": 50,
         "above_ma50": False, "above_ma200": False}
    score, factors = multi_factor._technical_score(s)
    assert score <= -0.5


def test_momentum_overshooting_warning():
    s = {"chg_20d_pct": 70}  # +70% in 20 days
    score, factors = multi_factor._momentum_score(s)
    assert score < -0.2
    assert any("透支" in f for f in factors)


def test_momentum_dropped_creates_buy_opportunity():
    s = {"chg_20d_pct": -25}
    score, factors = multi_factor._momentum_score(s)
    assert score > 0
    assert any("反弹候选" in f for f in factors)


def test_fundamental_high_pe_pct_negative():
    fdata = {
        "pe": 50, "pb": 5,
        "extra": {"pe_pct_5y": 0.92},
        "revenue_yoy": 0.05,
    }
    score, factors = multi_factor._fundamental_score(fdata)
    assert score < 0


def test_fundamental_low_pe_high_growth_positive():
    fdata = {
        "pe": 12, "pb": 1.5,
        "extra": {"pe_pct_5y": 0.10},
        "revenue_yoy": 0.30,
    }
    score, factors = multi_factor._fundamental_score(fdata)
    assert score > 0


def test_analyst_upside_positive_increases_score():
    fdata = {"extra": {"analyst_ratings": {
        "upside_pct": 30,
        "recommendation_mean": 1.5,
    }}}
    score, factors = multi_factor._analyst_score(fdata, current_price=100)
    assert score > 0.3


def test_analyst_target_below_current_price_negative():
    fdata = {"extra": {"analyst_ratings": {
        "upside_pct": -20,
        "recommendation_mean": 2.5,
    }}}
    score, factors = multi_factor._analyst_score(fdata, current_price=100)
    assert score < 0


def test_score_returns_required_fields():
    """End-to-end smoke: composite score has all expected keys (11 factors)."""
    s = {"signal_codes": ["RSI_OVERBOUGHT"], "rsi": 73,
         "above_ma50": True, "above_ma200": True, "chg_20d_pct": 5}
    out = multi_factor.score("AMD", s, fundamentals_data={}, current_price=100)
    assert "composite_score" in out
    assert "action" in out
    assert "factor_breakdown" in out
    assert "conviction" in out
    assert "top_factors" in out
    assert "counter_factors" in out
    assert -1 <= out["composite_score"] <= 1
    assert 0 <= out["conviction"] <= 5
    # 11 factors expected post 2026-05-26 (added alt_data, rating_change, event_intensity)
    assert len(out["factor_breakdown"]) == 11
    assert set(out["factor_breakdown"].keys()) == {
        "technical", "events", "trade_signals", "sentiment", "fundamental",
        "analyst", "momentum", "macro_regime",
        "alt_data", "rating_change", "event_intensity",
    }


# ============================================================================
# Phase A new factors (2026-05-26)
# ============================================================================


def test_alt_data_no_keyword_returns_zero(monkeypatch, tmp_path):
    """无 alt_data_keyword 配置 → 因子=0, factors=[] (将被 score() 当作缺数据)."""
    with _temp_db(monkeypatch, tmp_path):
        monkeypatch.setattr(multi_factor.cfg_mod, "load",
                            lambda name: {"positions": {"AMD": {}}, "watchlist": []})
        s, f = multi_factor._alt_data_score("AMD")
    assert s == 0.0
    assert f == []


def test_alt_data_strong_positive_sentiment_early_excitement(monkeypatch, tmp_path):
    """002624 异环 sentiment=+0.8 + early_excitement + 7d 上扬 → 正分."""
    with _temp_db(monkeypatch, tmp_path) as db_path:
        # Insert 7 days of bilibili snapshots, latest 高 sent, 7d ago 低 sent
        with sqlite3.connect(db_path) as conn:
            for i, sent in enumerate([0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3]):
                d = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
                conn.execute(
                    "INSERT INTO alt_data_metrics (source, key, captured_at, metric_date, metrics_json) "
                    "VALUES ('bilibili_search', '异环', ?, ?, ?)",
                    (datetime.utcnow().isoformat(), d, json.dumps({
                        "sentiment": {"overall_sentiment": sent, "buzz_phase": "early_excitement"}
                    })),
                )

        monkeypatch.setattr(multi_factor.cfg_mod, "load",
                            lambda name: {"positions": {"002624.SZ": {"alt_data_keyword": "异环"}},
                                          "watchlist": []})
        score, factors = multi_factor._alt_data_score("002624.SZ")

    assert score >= 0.5, f"strong positive + early_excitement + uptrend should >=0.5, got {score}"
    assert any("社区情绪" in f and "+0.80" in f for f in factors)
    assert any("early_excitement" in f for f in factors)
    assert any("上扬" in f for f in factors)


def test_alt_data_controversy_phase_negative(monkeypatch, tmp_path):
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO alt_data_metrics (source, key, captured_at, metric_date, metrics_json) "
                "VALUES ('bilibili_search', '某游戏', ?, ?, ?)",
                (datetime.utcnow().isoformat(), datetime.utcnow().date().isoformat(),
                 json.dumps({"sentiment": {"overall_sentiment": -0.4, "buzz_phase": "controversy"}})),
            )
        monkeypatch.setattr(multi_factor.cfg_mod, "load",
                            lambda name: {"positions": {"X.SZ": {"alt_data_keyword": "某游戏"}},
                                          "watchlist": []})
        score, factors = multi_factor._alt_data_score("X.SZ")
    assert score < -0.5
    assert any("controversy" in f for f in factors)


def test_alt_data_keyword_from_watchlist(monkeypatch, tmp_path):
    """alt_data_keyword 在 watchlist entry 上也要识别."""
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO alt_data_metrics (source, key, captured_at, metric_date, metrics_json) "
                "VALUES ('bilibili_search', '异环', ?, ?, ?)",
                (datetime.utcnow().isoformat(), datetime.utcnow().date().isoformat(),
                 json.dumps({"sentiment": {"overall_sentiment": 0.6, "buzz_phase": "sustained"}})),
            )
        monkeypatch.setattr(multi_factor.cfg_mod, "load",
                            lambda name: {"positions": {}, "watchlist": [
                                {"symbol": "WATCHED", "alt_data_keyword": "异环"}]})
        score, factors = multi_factor._alt_data_score("WATCHED")
    assert score > 0
    assert factors


# ---------- _rating_change_score ----------

def test_rating_change_no_history_returns_zero(monkeypatch, tmp_path):
    """没有 fundamentals 历史 → 0."""
    with _temp_db(monkeypatch, tmp_path):
        s, f = multi_factor._rating_change_score("AMD")
    assert s == 0.0
    assert f == []


def test_rating_change_upgrade_in_rec_mean(monkeypatch, tmp_path):
    """7 日前 rec_mean=2.3, 今天 1.8 → 评级转好, +0.5."""
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            for i, rec in enumerate([1.8, 2.0, 2.1, 2.2, 2.3, 2.3, 2.3]):
                d = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
                conn.execute(
                    "INSERT INTO fundamentals (symbol, as_of, extra_json) VALUES (?, ?, ?)",
                    ("AMD", d, json.dumps({
                        "analyst_ratings": {"recommendation_mean": rec,
                                            "target_mean_price": 200,
                                            "number_of_analyst_opinions": 30}
                    })),
                )
        score, factors = multi_factor._rating_change_score("AMD")
    assert score >= 0.5
    assert any("评级转好" in f for f in factors)


def test_rating_change_target_price_jump(monkeypatch, tmp_path):
    """目标价从 200 → 230 (+15%) 触发 +0.3."""
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            for i, tgt in enumerate([230, 228, 220, 215, 210, 205, 200]):
                d = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
                conn.execute(
                    "INSERT INTO fundamentals (symbol, as_of, extra_json) VALUES (?, ?, ?)",
                    ("AMD", d, json.dumps({
                        "analyst_ratings": {"recommendation_mean": 2.0,
                                            "target_mean_price": tgt,
                                            "number_of_analyst_opinions": 30}
                    })),
                )
        score, factors = multi_factor._rating_change_score("AMD")
    assert score >= 0.3
    assert any("目标价" in f and "+15" in f for f in factors)


def test_rating_change_coverage_growth(monkeypatch, tmp_path):
    """分析师覆盖 28 → 32, +0.2."""
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            for i, n in enumerate([32, 31, 30, 29, 28, 28, 28]):
                d = (datetime.utcnow() - timedelta(days=i)).date().isoformat()
                conn.execute(
                    "INSERT INTO fundamentals (symbol, as_of, extra_json) VALUES (?, ?, ?)",
                    ("AMD", d, json.dumps({
                        "analyst_ratings": {"recommendation_mean": 2.0,
                                            "target_mean_price": 200,
                                            "number_of_analyst_opinions": n}
                    })),
                )
        score, factors = multi_factor._rating_change_score("AMD")
    assert score >= 0.2
    assert any("覆盖" in f for f in factors)


# ---------- _event_intensity_score ----------

def test_event_intensity_no_events_returns_zero(monkeypatch, tmp_path):
    with _temp_db(monkeypatch, tmp_path):
        s, f = multi_factor._event_intensity_score("AMD")
    assert s == 0.0
    assert f == []


def test_event_intensity_bull_high_severity(monkeypatch, tmp_path):
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            now = datetime.utcnow().isoformat() + "Z"
            conn.execute(
                "INSERT INTO events (severity, category, summary, impact_json, affected_symbols, fired_at) "
                "VALUES (?,?,?,?,?,?)",
                (8, "price_action", "AMD 大涨", json.dumps({
                    "impacts": [{"symbol": "AMD", "direction": "bull"}]
                }), "AMD", now),
            )
            conn.execute(
                "INSERT INTO events (severity, category, summary, impact_json, affected_symbols, fired_at) "
                "VALUES (?,?,?,?,?,?)",
                (7, "single-stock", "AMD 上调指引", json.dumps({
                    "impacts": [{"symbol": "AMD", "direction": "bull"}]
                }), "AMD", now),
            )
        score, factors = multi_factor._event_intensity_score("AMD")
    assert score >= 0.6
    assert any("看多 2" in f for f in factors)


def test_event_intensity_mixed_directions(monkeypatch, tmp_path):
    """看空 ≥ 看多, 高 sev → 负分."""
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            now = datetime.utcnow().isoformat() + "Z"
            conn.execute(
                "INSERT INTO events (severity, category, summary, impact_json, affected_symbols, fired_at) "
                "VALUES (?,?,?,?,?,?)",
                (7, "macro", "供应链危机", json.dumps({
                    "impacts": [{"symbol": "AMD", "direction": "bear"}]
                }), "AMD", now),
            )
            conn.execute(
                "INSERT INTO events (severity, category, summary, impact_json, affected_symbols, fired_at) "
                "VALUES (?,?,?,?,?,?)",
                (6, "macro", "市场波动", json.dumps({
                    "impacts": [{"symbol": "AMD", "direction": "bear"}]
                }), "AMD", now),
            )
        score, factors = multi_factor._event_intensity_score("AMD")
    assert score <= -0.2
    assert any("看空 2" in f for f in factors)


def test_event_intensity_low_severity_ignored(monkeypatch, tmp_path):
    """sev<6 不计入."""
    with _temp_db(monkeypatch, tmp_path) as db_path:
        with sqlite3.connect(db_path) as conn:
            now = datetime.utcnow().isoformat() + "Z"
            conn.execute(
                "INSERT INTO events (severity, category, summary, impact_json, affected_symbols, fired_at) "
                "VALUES (?,?,?,?,?,?)",
                (5, "macro", "小波动", json.dumps({
                    "impacts": [{"symbol": "AMD", "direction": "bear"}]
                }), "AMD", now),
            )
        score, factors = multi_factor._event_intensity_score("AMD")
    assert score == 0.0


# ---------- score() composition & threshold changes ----------

def test_score_threshold_dropped_to_030_triggers_add(monkeypatch, tmp_path):
    """11 因子合成 0.31 应触发 ADD (新阈值 0.30, 老 0.40 会是 WATCH_BUY)."""
    # 通过 mock 每个 sub-factor 控制结果
    def fake_tech(s): return 0.5, ["MACD 金叉"]
    def fake_events(sym): return 0.5, ["财报超预期"], False
    def fake_trade(sym): return 0.3, ["放量上涨"]
    def fake_sent(sym): return 0.2, ["新闻正面"]
    def fake_fund(d): return 0.2, ["估值合理"]
    def fake_ana(d, p): return 0.3, ["卖方目标 +20%"]
    def fake_mom(s): return 0.0, []
    def fake_macro(): return 0.0, []
    def fake_alt(sym): return 0.5, ["B 站情绪 +0.6"]
    def fake_rch(sym): return 0.0, []
    def fake_eint(sym): return 0.0, []

    monkeypatch.setattr(multi_factor, "_technical_score", fake_tech)
    monkeypatch.setattr(multi_factor, "_events_score", fake_events)
    monkeypatch.setattr(multi_factor, "_trade_signals_score", fake_trade)
    monkeypatch.setattr(multi_factor, "_sentiment_score", fake_sent)
    monkeypatch.setattr(multi_factor, "_fundamental_score", fake_fund)
    monkeypatch.setattr(multi_factor, "_analyst_score", fake_ana)
    monkeypatch.setattr(multi_factor, "_momentum_score", fake_mom)
    monkeypatch.setattr(multi_factor, "_macro_regime_score", fake_macro)
    monkeypatch.setattr(multi_factor, "_alt_data_score", fake_alt)
    monkeypatch.setattr(multi_factor, "_rating_change_score", fake_rch)
    monkeypatch.setattr(multi_factor, "_event_intensity_score", fake_eint)

    out = multi_factor.score("AMD", {"signal_codes": [], "rsi": 50,
                                      "above_ma50": True, "above_ma200": True,
                                      "chg_20d_pct": 5})
    assert out["composite_score"] >= 0.30
    assert out["action"] == "ADD"
    assert out["conviction"] >= 1


def test_score_top_factors_aligned_with_composite(monkeypatch, tmp_path):
    """top_factors 必须跟 composite 同向; counter_factors 反向."""
    def fake_tech(s): return 0.8, ["MACD 强金叉"]    # +
    def fake_events(sym): return 0.7, ["利好催化"], False  # +
    def fake_trade(sym): return 0.0, []
    def fake_sent(sym): return 0.0, []
    def fake_fund(d): return 0.0, []
    def fake_ana(d, p): return 0.0, []
    def fake_mom(s): return -0.5, ["20 日 +60% 透支"]   # -
    def fake_macro(): return 0.0, []
    def fake_alt(sym): return 0.0, []
    def fake_rch(sym): return 0.0, []
    def fake_eint(sym): return 0.0, []

    monkeypatch.setattr(multi_factor, "_technical_score", fake_tech)
    monkeypatch.setattr(multi_factor, "_events_score", fake_events)
    monkeypatch.setattr(multi_factor, "_trade_signals_score", fake_trade)
    monkeypatch.setattr(multi_factor, "_sentiment_score", fake_sent)
    monkeypatch.setattr(multi_factor, "_fundamental_score", fake_fund)
    monkeypatch.setattr(multi_factor, "_analyst_score", fake_ana)
    monkeypatch.setattr(multi_factor, "_momentum_score", fake_mom)
    monkeypatch.setattr(multi_factor, "_macro_regime_score", fake_macro)
    monkeypatch.setattr(multi_factor, "_alt_data_score", fake_alt)
    monkeypatch.setattr(multi_factor, "_rating_change_score", fake_rch)
    monkeypatch.setattr(multi_factor, "_event_intensity_score", fake_eint)

    out = multi_factor.score("AMD", {})
    assert out["composite_score"] > 0
    # technical 和 events 是正贡献 → top_factors
    top_names = [tf["name"] for tf in out["top_factors"]]
    assert "technical" in top_names and "events" in top_names
    # momentum 反向 → counter_factors
    counter_names = [cf["name"] for cf in out["counter_factors"]]
    assert "momentum" in counter_names


def test_score_handles_all_missing_data(monkeypatch):
    """所有因子无数据 → HOLD + conviction=0 + 不爆."""
    for fn in ("_technical_score", "_trade_signals_score", "_sentiment_score",
               "_fundamental_score", "_analyst_score", "_momentum_score",
               "_macro_regime_score", "_alt_data_score", "_rating_change_score",
               "_event_intensity_score"):
        monkeypatch.setattr(multi_factor, fn, lambda *a, **kw: (0.0, []))
    monkeypatch.setattr(multi_factor, "_events_score", lambda sym: (0.0, [], False))

    out = multi_factor.score("UNKNOWN", {})
    assert out["composite_score"] == 0.0
    assert out["action"] == "HOLD"
    assert out["conviction"] == 0


def test_score_renormalizes_when_some_factors_missing(monkeypatch):
    """只有 technical + alt_data 有数据 → 这俩的有效权重之和=1.0, composite 不被无数据稀释."""
    def fake_tech(s): return 1.0, ["RSI 极度超卖"]    # 最强买信号
    def fake_alt(sym): return 1.0, ["sentiment +0.8"]
    monkeypatch.setattr(multi_factor, "_technical_score", fake_tech)
    monkeypatch.setattr(multi_factor, "_alt_data_score", fake_alt)
    # 其他全无数据
    monkeypatch.setattr(multi_factor, "_events_score", lambda sym: (0.0, [], False))
    for fn in ("_trade_signals_score", "_sentiment_score", "_fundamental_score",
               "_analyst_score", "_momentum_score", "_macro_regime_score",
               "_rating_change_score", "_event_intensity_score"):
        monkeypatch.setattr(multi_factor, fn, lambda *a, **kw: (0.0, []))

    out = multi_factor.score("X", {})
    # technical 与 alt_data 都给满分 → composite 接近 1.0 (而非被无数据拖到 ~0.20)
    assert out["composite_score"] >= 0.90
    assert out["action"] == "ADD"
    assert out["conviction"] == 5


def test_score_catalyst_imminent_defers_to_llm(monkeypatch):
    monkeypatch.setattr(multi_factor, "_events_score", lambda sym: (0.3, ["明日财报"], True))
    for fn in ("_technical_score", "_trade_signals_score", "_sentiment_score",
               "_fundamental_score", "_analyst_score", "_momentum_score",
               "_macro_regime_score", "_alt_data_score", "_rating_change_score",
               "_event_intensity_score"):
        monkeypatch.setattr(multi_factor, fn, lambda *a, **kw: (0.0, []))

    out = multi_factor.score("AMD", {})
    assert out["action"] == "DEFER_TO_LLM"
    assert out["catalyst_imminent"] is True
