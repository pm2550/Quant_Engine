"""2026-09-10 全量审计修复的回归测试。

每个 test 对应 docs/AUDIT-20260910.md 里的一个编号。这批问题全是"静默失效"型 ——
不抛异常、不告警, 只是数字悄悄变成假的, 所以必须有测试钉住。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------- A2
class TestTimestampNormalization:
    """A2: news_archive.published_at 曾有 97% 的行是 RFC2822 原样透传。"""

    def test_rfc2822(self):
        from quant import db
        assert db.normalize_timestamp("Fri, 03 Jul 2026 08:12:00 GMT") == "2026-07-03T08:12:00Z"

    def test_rfc2822_with_offset(self):
        from quant import db
        # +0800 必须被换算成 UTC, 不是截断
        assert db.normalize_timestamp("Tue, 4 Aug 2026 01:02:03 +0800") == "2026-08-03T17:02:03Z"

    def test_iso_passthrough(self):
        from quant import db
        assert db.normalize_timestamp("2026-09-09T15:30:00Z") == "2026-09-09T15:30:00Z"

    def test_eastmoney_naive(self):
        from quant import db
        assert db.normalize_timestamp("2026-09-09 15:30:00") == "2026-09-09T15:30:00Z"

    def test_garbage_returns_none(self):
        from quant import db
        # 宁可 NULL 让 COALESCE(published_at, fetched_at) 接手, 也不要留会算错的字符串
        assert db.normalize_timestamp("not a date") is None
        assert db.normalize_timestamp("") is None
        assert db.normalize_timestamp(None) is None

    def test_output_always_sorts_chronologically(self):
        """归一化的核心价值: 字符串排序 == 时间排序。"""
        from quant import db
        raw = ["Fri, 03 Jul 2026 08:12:00 GMT",
               "Mon, 01 Jun 2026 00:00:00 GMT",
               "2026-08-15 12:00:00"]
        norm = sorted(db.normalize_timestamp(r) for r in raw)
        assert norm == ["2026-06-01T00:00:00Z", "2026-07-03T08:12:00Z", "2026-08-15T12:00:00Z"]


# ---------------------------------------------------------------- A3
class TestFeedWeighting:
    """A3: sources.yaml 的 weight 以前被读进 item dict 但从未用于限流。"""

    def test_weight_scales_intake(self):
        from quant import newswatch
        assert newswatch._items_cap({"weight": 0.5}) < newswatch._items_cap({"weight": 1.5})

    def test_explicit_cap_wins(self):
        from quant import newswatch
        assert newswatch._items_cap({"weight": 1.5, "max_items_per_poll": 8}) == 8

    def test_floor_enforced(self):
        from quant import newswatch
        assert newswatch._items_cap({"weight": 0.01}) >= newswatch.MIN_ITEMS_PER_POLL

    def test_noisy_feeds_capped_in_config(self):
        """aljazeera / zerohedge / bbc 近 60 天刷了 11,140 条, 必须有显式上限。"""
        from quant import config as cfg_mod, newswatch
        feeds = {f["name"]: f for f in (cfg_mod.load("sources").get("rss_feeds") or [])}
        for name in ("aljazeera", "zerohedge", "bbc_world"):
            if name in feeds:
                assert newswatch._items_cap(feeds[name]) <= 15, f"{name} 上限过高"


# ---------------------------------------------------------------- A4
class TestBilibiliCappedMetric:
    """A4: total_results 恒为 1000 (API 分页上限), 趋势永远 0.0%。"""

    def test_single_keyword_only(self):
        from quant.alt_data import bilibili
        for sym, kws in bilibili.DEFAULT_KEYWORDS.items():
            assert len(kws) == 1, f"{sym} 有 {len(kws)} 个关键词 — 会产出互相矛盾的情绪值"

    def test_cap_constant(self):
        from quant.alt_data import bilibili
        assert bilibili.TOTAL_RESULTS_CAP == 1000


# ---------------------------------------------------------------- C1/C2/C3
class TestExpectationsV2:
    """C1/C2/C3: bootstrap_v1 20d 高估 9.3pp, 90% 区间只覆盖 71.9%。"""

    @staticmethod
    def _closes(n=900, seed=7, drift=0.0012, vol=0.02):
        rng = np.random.default_rng(seed)
        r = rng.normal(drift, vol, n)
        return pd.Series(100 * np.exp(np.cumsum(r)),
                         index=pd.date_range("2023-01-02", periods=n, freq="B"))

    def test_v2_is_default(self):
        from quant import expectations as E
        assert E.MODEL_VERSION == "bootstrap_v2"
        assert E.LEGACY_MODEL_VERSION == "bootstrap_v1"

    def test_v2_center_is_zero(self):
        """C1 的修法: 不再外推历史漂移。"""
        from quant import expectations as E
        d = E.vol_scaled_distribution(self._closes(), horizon_days=20)
        assert d is not None
        assert d["mean_pct"] == 0.0
        assert d["median_pct"] == 0.0

    def test_v1_extrapolates_drift_v2_does_not(self):
        """同一段上涨数据: v1 的 mean 显著为正, v2 为 0。"""
        from quant import expectations as E
        c = self._closes(drift=0.003)          # 强上涨
        v1 = E.bootstrap_distribution(c, horizon_days=20)
        v2 = E.vol_scaled_distribution(c, horizon_days=20)
        assert v1["mean_pct"] > 3.0, "构造的上涨样本应让 v1 外推出正漂移"
        assert v2["mean_pct"] == 0.0

    def test_sigma_k_per_horizon(self):
        from quant import expectations as E
        assert set(E.SIGMA_K) == {1, 5, 20}
        # 拟合值: horizon 越长 k 越小 (长周期经验分位本身已经够宽)
        assert E.SIGMA_K[1] > E.SIGMA_K[20]

    def test_quantiles_ordered(self):
        from quant import expectations as E
        d = E.vol_scaled_distribution(self._closes(), horizon_days=5)
        assert d["p5_pct"] < d["p25_pct"] < 0 < d["p75_pct"] < d["p95_pct"]

    def test_sigma_scales_with_volatility(self):
        """波动标准化的核心: 当前波动高 → 区间宽。"""
        from quant import expectations as E
        calm = E.vol_scaled_distribution(self._closes(vol=0.008), horizon_days=20)
        wild = E.vol_scaled_distribution(self._closes(vol=0.035), horizon_days=20)
        assert wild["sigma_pct"] > calm["sigma_pct"] * 1.5

    def test_insufficient_data_returns_none(self):
        from quant import expectations as E
        assert E.vol_scaled_distribution(self._closes(n=40), horizon_days=20) is None


# ---------------------------------------------------------------- C9/C10
class TestThresholdsAndConviction:
    """C9/C10: ADD 阈值 0.30 在 4 个月里只触发 3 次; conviction 双重闸门。"""

    def test_add_threshold_reachable(self):
        from quant import multi_factor as M
        t = M.action_thresholds()
        # composite 的 Q5 五分位均值实测 0.17 — 阈值必须低于它才可能触发
        assert t["add"] <= 0.17, f"ADD 阈值 {t['add']} 高于实测 Q5 均值 0.17"

    def test_conviction_reaches_two_at_watch_buy(self):
        """日报按 conviction<2 过滤, 所以能触发 WATCH_BUY 的必须 >=2 星。"""
        from quant import multi_factor as M
        t = M.action_thresholds()
        assert M.conviction_from_composite(t["watch_buy"]) >= 2
        assert M.conviction_from_composite(t["add"]) >= 3

    def test_conviction_monotonic(self):
        from quant import multi_factor as M
        vals = [M.conviction_from_composite(c) for c in (0.0, 0.03, 0.07, 0.13, 0.22, 0.35)]
        assert vals == sorted(vals)

    def test_conviction_symmetric(self):
        from quant import multi_factor as M
        assert M.conviction_from_composite(0.2) == M.conviction_from_composite(-0.2)

    def test_cross_sectional_gate_caps_adds(self):
        """板块同涨时 11 只一起亮 ADD 不是 11 个独立信号。"""
        from quant import multi_factor as M
        recs = [{"symbol": f"S{i}", "composite_score": 0.30 - 0.01 * i,
                 "action": "ADD", "conviction": 5, "rationale": ""} for i in range(8)]
        out = M.apply_cross_sectional_gate(recs, max_adds=3)
        assert sum(1 for r in out if r["action"] == "ADD") == 3
        # 保留的必须是 composite 最高的三个
        kept = {r["symbol"] for r in out if r["action"] == "ADD"}
        assert kept == {"S0", "S1", "S2"}

    def test_gate_noop_when_under_cap(self):
        from quant import multi_factor as M
        recs = [{"symbol": "A", "composite_score": 0.5, "action": "ADD",
                 "conviction": 5, "rationale": ""}]
        assert M.apply_cross_sectional_gate(recs, max_adds=3)[0]["action"] == "ADD"

    def test_demoted_become_watch_buy_not_dropped(self):
        from quant import multi_factor as M
        recs = [{"symbol": f"S{i}", "composite_score": 0.3 - 0.01 * i,
                 "action": "ADD", "conviction": 5, "rationale": ""} for i in range(5)]
        out = M.apply_cross_sectional_gate(recs, max_adds=2)
        assert len(out) == 5
        assert {r["action"] for r in out} == {"ADD", "WATCH_BUY"}


# ---------------------------------------------------------------- C6
class TestChallengerFreshness:
    """C6: freshness 只看 JSON 文件 mtime, as_of 取字典第一个 symbol。"""

    def test_split_by_freshness(self):
        from quant.ml import serve
        today = date(2026, 9, 10)
        preds = {
            "FRESH": {"pred_forward_return": 0.03, "as_of": "2026-09-09"},
            "STALE": {"pred_forward_return": 0.07, "as_of": "2026-03-20"},
            "NOASOF": {"pred_forward_return": 0.02},
        }
        fresh, stale = serve.split_by_freshness(preds, today=today)
        assert set(fresh) == {"FRESH"}
        assert set(stale) == {"STALE", "NOASOF"}, "缺 as_of 必须按陈旧处理"

    def test_stale_days(self):
        from quant.ml import serve
        assert serve.stale_days({"as_of": "2026-09-01"}, today=date(2026, 9, 10)) == 9
        assert serve.stale_days({}, today=date(2026, 9, 10)) is None


# ---------------------------------------------------------------- D2
class TestBacktestFiniteGuards:
    """D2: 0 笔交易 → 方差 0 → sharpe=Inf, 11,251 条假记录顶满榜首。"""

    def test_finite_filters_inf(self):
        from quant import backtest
        assert backtest._finite(float("inf")) == 0.0
        assert backtest._finite(float("-inf")) == 0.0
        assert backtest._finite(float("nan")) == 0.0

    def test_finite_passes_real_values(self):
        from quant import backtest
        assert backtest._finite(2.65) == 2.65
        assert backtest._finite(-0.5) == -0.5

    def test_finite_handles_non_numeric(self):
        from quant import backtest
        assert backtest._finite(None) == 0.0
        assert backtest._finite("abc") == 0.0

    def test_no_inf_in_db(self):
        """数据层面的回归哨兵 —— 历史 Inf 已回填。"""
        from quant import db
        with sqlite3.connect(db.DB_PATH) as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM backtest_results "
                "WHERE sharpe > 1e6 OR sharpe < -1e6 OR profit_factor > 1e6").fetchone()[0]
        assert bad == 0


# ---------------------------------------------------------------- E3
class TestCrossCurrencyWeights:
    """E3: 权重按币种桶算 → 唯一的 CNY 持仓永远 100% → 每日假警告。"""

    def test_usd_is_identity(self):
        from quant import orchestrator
        assert orchestrator.fx_to_usd("USD") == 1.0

    def test_cny_rate_plausible(self):
        from quant import orchestrator
        r = orchestrator.fx_to_usd("CNY")
        assert 0.08 < r < 0.25, f"CNY→USD {r} 不合理"

    def test_unknown_currency_falls_back(self):
        from quant import orchestrator
        assert orchestrator.fx_to_usd("XYZ") == 1.0

    def test_single_cny_position_not_100pct_globally(self):
        """核心回归: 单一 CNY 持仓 + 多个 USD 持仓 → 它的全局权重远低于 100%。"""
        from quant import orchestrator
        mv = {"002624.SZ": ("CNY", 2170.0), "VOO": ("USD", 1000.0), "AMD": ("USD", 742.0)}
        usd = {s: v * orchestrator.fx_to_usd(c) for s, (c, v) in mv.items()}
        total = sum(usd.values())
        w = usd["002624.SZ"] / total
        assert w < 0.30, f"002624 全局权重 {w:.1%} 仍超 30% 上限 — 会继续报假警告"


# ---------------------------------------------------------------- C4
class TestCalibration:
    """C4: expectations.py 自承校准追踪未实现, 4 个月没人补。"""

    def test_rank_ic_detects_positive_signal(self):
        from quant import calibration
        df = pd.DataFrame({
            "snapshot_date": ["2026-01-01"] * 6 + ["2026-01-02"] * 6,
            "symbol": list("ABCDEF") * 2,
            "pred": [5, 4, 3, 2, 1, 0] * 2,
            "real": [5, 4, 3, 2, 1, 0] * 2,
        })
        r = calibration._rank_ic_stats(df, "pred", "real")
        assert r["daily_rank_ic"] == pytest.approx(1.0)
        assert r["ic_positive_day_pct"] == pytest.approx(100.0)

    def test_rank_ic_detects_inverted_signal(self):
        """这正是 challenger 的实际情况 (daily IC −0.216)。"""
        from quant import calibration
        df = pd.DataFrame({
            "snapshot_date": ["2026-01-01"] * 6 + ["2026-01-02"] * 6,
            "symbol": list("ABCDEF") * 2,
            "pred": [5, 4, 3, 2, 1, 0] * 2,
            "real": [0, 1, 2, 3, 4, 5] * 2,
        })
        r = calibration._rank_ic_stats(df, "pred", "real")
        assert r["daily_rank_ic"] == pytest.approx(-1.0)
        assert r["ic_positive_day_pct"] == pytest.approx(0.0)

    def test_daily_ic_skips_thin_cross_sections(self):
        from quant import calibration
        df = pd.DataFrame({
            "snapshot_date": ["2026-01-01"] * 2,
            "symbol": ["A", "B"],
            "pred": [1, 2], "real": [1, 2],
        })
        r = calibration._rank_ic_stats(df, "pred", "real")
        assert r["n_days"] == 0, "2 只标的的截面不该算 IC"

    def test_forward_return_refuses_stale_prices(self):
        """绝不拿"最后一根可用 K 线"冒充 horizon 后的价格。"""
        from quant import calibration
        # 未来日期 → 必然没有 horizon 后的价格
        future = (date.today() + timedelta(days=5)).isoformat()
        assert calibration.forward_return_pct("AMD", future, 20) is None

    def test_tables_exist(self):
        from quant import db
        db.init()
        with sqlite3.connect(db.DB_PATH) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"model_predictions", "model_calibration"} <= names


# ---------------------------------------------------------------- D5/D6
class TestDecisionReview:
    """D5: WATCH_SKIP "<5% 算对" 在牛市里恒为真。D6: HOLD 不入库。"""

    def test_hold_is_loggable(self):
        from quant import decision_log
        assert "HOLD" in decision_log.LOGGABLE_ACTIONS

    def test_watch_skip_not_scored(self):
        from quant import decision_review as D
        assert D._was_correct(-1, 2.0, "WATCH_SKIP") is None
        assert D._was_correct(-1, 50.0, "WATCH_SKIP") is None

    def test_hold_not_scored(self):
        from quant import decision_review as D
        assert D._was_correct(0, 5.0, "HOLD") is None

    def test_directional_actions_still_scored(self):
        from quant import decision_review as D
        assert D._was_correct(1, 3.0, "ADD") == 1
        assert D._was_correct(1, -3.0, "ADD") == 0
        assert D._was_correct(-1, -3.0, "REDUCE") == 1
        assert D._was_correct(-1, 3.0, "REDUCE") == 0

    def test_no_five_pct_grace_band(self):
        """旧逻辑里 REDUCE 后涨 4% 也算"对" —— 必须已废除。"""
        from quant import decision_review as D
        assert D._was_correct(-1, 4.0, "REDUCE") == 0


# ---------------------------------------------------------------- A1
class TestPriceRefresh:
    """A1: 159 只里 143 只冻结数月, 无人发现。"""

    def test_staleness_report_shape(self):
        from quant import price_refresh
        rep = price_refresh.staleness_report()
        assert "total" in rep and "buckets" in rep
        assert isinstance(rep["stale_over_30d"], list)

    def test_universe_includes_cached(self):
        from quant import price_refresh
        syms = price_refresh.universe_symbols()
        assert len(syms) > 50, "全宇宙应该有上百只"

    def test_delisted_dir_separate_from_active(self):
        from quant import price_refresh
        assert price_refresh.DELISTED_DIR.name.startswith("_"), \
            "隔离目录要以 _ 开头, 否则会被 glob('*.parquet') 扫进活跃宇宙"
        # 隔离目录不能被 cached_symbols 扫到
        assert not any(s.startswith("_") for s in price_refresh.cached_symbols())


# ---------------------------------------------------------------- B2
class TestRateLimitHandling:
    """B2: 429 时立刻跳下一个 backend, 而下一个也是 ollama → 整条链一起失败。"""

    def test_detects_rate_limit_variants(self):
        from quant import llm_router as R
        for msg in ("429 Client Error: Too Many Requests",
                    "rate limit exceeded", "quota exhausted", "Too Many Requests"):
            assert R._is_rate_limited(Exception(msg)), msg

    def test_ignores_other_errors(self):
        from quant import llm_router as R
        for msg in ("connection reset", "401 Unauthorized", "timeout"):
            assert not R._is_rate_limited(Exception(msg)), msg

    def test_cooldown_roundtrip(self):
        from quant import llm_router as R
        R._COOLDOWN_UNTIL.clear()
        assert R._cooling("ollama:test") == 0
        R._enter_cooldown("ollama:test", seconds=30)
        assert 0 < R._cooling("ollama:test") <= 30
        assert "ollama:test" in R.rate_limit_state()
        R._COOLDOWN_UNTIL.clear()

    def test_backoff_schedule_increases(self):
        from quant import llm_router as R
        assert list(R.RATE_LIMIT_BACKOFF_S) == sorted(R.RATE_LIMIT_BACKOFF_S)


# ---------------------------------------------------------------- B1
class TestDeadProviderCleanup:
    """B1: dashscope 2026-06-01 下线, 但配置/校验/smoke 仍要求它。"""

    def test_no_routes_reference_dashscope(self):
        from quant import config as cfg_mod
        cfg = cfg_mod.load("llm_routes")
        for task, chain in (cfg.get("routes") or {}).items():
            for entry in chain:
                assert not entry.startswith("dashscope:"), \
                    f"route {task} 仍指向已下线的 dashscope ({entry})"

    def test_dashscope_provider_not_enabled(self):
        from quant import config as cfg_mod
        assert "dashscope" not in (cfg_mod.load("llm_routes").get("providers") or {})


# ---------------------------------------------------------------- E 报告
class TestReportConditionalSections:
    """E2/E4/E7: 四个静态段无条件 append, 129 行说 2 行内容。"""

    def test_macro_has_conditional_modes(self):
        from quant import macro_regime
        import inspect
        sig = inspect.signature(macro_regime.render_section)
        assert "only_on_change" in sig.parameters
        assert "one_line" in sig.parameters

    def test_alt_data_has_conditional_modes(self):
        from quant.alt_data import formatter
        import inspect
        sig = inspect.signature(formatter.render_section)
        assert "only_if_anomaly" in sig.parameters
        assert "one_line" in sig.parameters

    def test_events_digest_defaults_to_two(self):
        from quant import events_digest
        import inspect
        sig = inspect.signature(events_digest.render_section)
        assert sig.parameters["top_k"].default == 2

    def test_base_rate_needs_real_sample(self):
        """n=7~15 的历史中位数没有决策价值。"""
        from quant import events_digest
        assert events_digest.MIN_BASE_RATE_N >= 30

    def test_packager_requires_plain_summary(self):
        from quant import llm_packager
        assert "一句人话总结" in llm_packager.SYSTEM_PROMPT

    def test_packager_has_no_dead_model_reference(self):
        from quant import llm_packager
        assert "deepseek" not in llm_packager.SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------- A7
class TestBilibiliClickOrderDegradation:
    """A7 (生产验证时发现): B 站偶发忽略 order=click, 返回按相关度排序的结果。

    不报错, 只是 top_avg_plays 悄悄变成另一个数量级 —— 实测同一天同一关键词
    5,893,843 vs 433,885 (差 13 倍), 随后触发一次假的 "7d −92.6%" 异动告警。
    """

    def test_trend_suppresses_plays_pct_when_unreliable(self, tmp_path, monkeypatch):
        from quant import db
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
        db.init()
        from quant.alt_data import bilibili
        old = (date.today() - timedelta(days=8)).isoformat()
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 5_000_000,
                                       "top_plays_reliable": True}, metric_date=old)
        # 今天这条是降级抓取 → 不可靠
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 400_000,
                                       "top_plays_reliable": False},
                                metric_date=date.today().isoformat())
        t = bilibili.trend("K")
        assert t["top_plays_reliable"] is False
        assert t["vs_7d_ago"]["top_avg_plays_pct"] is None, \
            "不可靠的快照不能报出 -92% 这种假趋势"

    def test_trend_reports_pct_when_both_reliable(self, tmp_path, monkeypatch):
        from quant import db
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
        db.init()
        from quant.alt_data import bilibili
        old = (date.today() - timedelta(days=8)).isoformat()
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 1000,
                                       "top_plays_reliable": True}, metric_date=old)
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 500,
                                       "top_plays_reliable": True},
                                metric_date=date.today().isoformat())
        t = bilibili.trend("K")
        assert t["vs_7d_ago"]["top_avg_plays_pct"] == -50.0

    def test_legacy_rows_without_flag_treated_as_reliable(self, tmp_path, monkeypatch):
        """A7 之前的历史行没有这个字段, 不能因此全部失效。"""
        from quant import db
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
        db.init()
        from quant.alt_data import bilibili
        old = (date.today() - timedelta(days=8)).isoformat()
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 1000},
                                metric_date=old)
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 800},
                                metric_date=date.today().isoformat())
        t = bilibili.trend("K")
        assert t["vs_7d_ago"]["top_avg_plays_pct"] == -20.0

    def test_anomaly_does_not_fire_on_unreliable_snapshot(self, tmp_path, monkeypatch):
        """最终目的: 降级抓取不得触发告警。"""
        from quant import db
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
        db.init()
        from quant.alt_data import bilibili, anomaly
        old = (date.today() - timedelta(days=8)).isoformat()
        sent = {"overall_sentiment": 0.3, "buzz_phase": "sustained"}
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 5_000_000,
                                       "top_plays_reliable": True, "sentiment": sent},
                                metric_date=old)
        bilibili.store_snapshot("K", {"total_results": 900, "top_avg_plays": 400_000,
                                       "top_plays_reliable": False, "sentiment": sent},
                                metric_date=date.today().isoformat())
        sigs = {f["signal"] for f in anomaly.check_keyword("K", dry_run=True)}
        assert "plays_drop" not in sigs, "降级抓取触发了假告警"


# ---------------------------------------------------------------- C5b / C6b
class TestChallengerTrainingGuards:
    """2026-09-10 追查 challenger 负 IC 时发现的两个训练/展示缺陷。"""

    def test_val_coverage_threshold_exists(self):
        """train_full 的 val_cutoff = dates[-60] 取的是**池化**日期并集, 只要求
        "至少有一只标的还在更新"。143/159 只价格冻结时, 验证集只剩约 20 只
        (1,798 行) 而训练集是 156 只 (525,259 行) —— 早停在错的分布上决定。
        """
        import sys
        sys.path.insert(0, "/data2/quant")
        from quant.ml import challenger
        assert hasattr(challenger, "MIN_VAL_SYMBOL_COVERAGE")
        assert 0.5 <= challenger.MIN_VAL_SYMBOL_COVERAGE <= 0.95

    def test_performance_line_reads_live_data(self):
        """原来是字符串字面量 "OOS: IC +0.049 ..." —— 出自 2026-05-27 的一次离线
        实验, 模型每周重训而这行数字 3.5 个月从没重算, 却天天印在日报上。
        """
        from quant.ml import serve
        # 检查渲染输出, 不是源码 —— 源码的 docstring 里会引用那个旧字符串做说明
        rendered = serve.render_section(
            {"AMD": {"pred_forward_return": 0.03, "horizon_days": 20,
                      "as_of": date.today().isoformat()}},
            held_symbols=["AMD"], freshness="fresh")
        assert "+0.049" not in rendered, "写死的性能指标又回到输出里了"
        line = serve.live_performance_line()
        assert isinstance(line, str) and line
        # 必须带上样本量或明确说没数据, 不能是一个无出处的数字
        assert ("n=" in line) or ("尚无" in line) or ("失败" in line)

    def test_performance_line_survives_missing_table(self, tmp_path, monkeypatch):
        """校准表为空时给出提示而不是崩 / 不是编一个数字。"""
        from quant import db
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "empty.sqlite")
        db.init()
        from quant.ml import serve
        line = serve.live_performance_line()
        assert "尚无校准记录" in line or "失败" in line


# ---------------------------------------------------------------- G1
class TestAudioPromptRendering:
    """G1 (部署重启时发现): audio_queue 310/310 任务全部失败, 子系统从未成功过一次。

    ANALYZE_PROMPT 里有一大段 JSON 示例的裸大括号, 而渲染用的是 str.format() ——
    format 把 `{\\n  "summary": ...` 当成占位符名, 每次必抛 KeyError。异常被
    per-task try/except 吞进 audio_queue.error, 所以对外表现只是"日报里的音频
    要点那段从不出现"。每条还照样先花 6 分钟跑 whisper 转写才在最后一步炸掉。
    """

    def test_prompt_renders_without_error(self):
        from string import Template
        from quant import audio_queue_worker as W
        out = Template(W.ANALYZE_PROMPT).safe_substitute(
            portfolio="AMD(AMD), VOO(Vanguard)", source="wsj_markets", title="T")
        assert "AMD(AMD)" in out
        assert "wsj_markets" in out

    def test_json_example_survives_rendering(self):
        """JSON 示例必须原样保留 —— 它是给 LLM 看的输出格式说明。"""
        from string import Template
        from quant import audio_queue_worker as W
        out = Template(W.ANALYZE_PROMPT).safe_substitute(
            portfolio="X", source="s", title="t")
        for key in ('"summary"', '"impacts"', '"importance"', '"tone"'):
            assert key in out, f"JSON 示例里的 {key} 丢了"

    def test_no_unresolved_placeholders(self):
        from string import Template
        from quant import audio_queue_worker as W
        out = Template(W.ANALYZE_PROMPT).safe_substitute(
            portfolio="X", source="s", title="t")
        for ph in ("$portfolio", "$source", "$title", "{portfolio}", "{source}", "{title}"):
            assert ph not in out, f"占位符 {ph} 没被替换"

    def test_format_would_still_break(self):
        """钉住根因: 换回 str.format 一定炸 —— 防止有人"顺手改回去"。"""
        from quant import audio_queue_worker as W
        with pytest.raises((KeyError, IndexError, ValueError)):
            W.ANALYZE_PROMPT.format(portfolio="X", source="s", title="t")


# ---------------------------------------------------------------- G2
class TestThinkingLeakGuard:
    """G2 (验证周报时发现): "一周总结" 输出的是模型的思考过程而不是总结。

    weekly_report 的注释写着 "simple_chat (qwen, non-thinking)" —— 那个假设在
    2026-06-01 dashscope 下线时就失效了, simple_chat 现在路由到 ollama:kimi-k2.6,
    是个 thinking 模型。max_tokens=400 全烧在思考上 → content 为空 → llm_router
    的抢救逻辑把 thinking 当答案交出来。又是一个配置变更让代码注释的假设静默失效。
    """

    def test_chat_accepts_salvage_opt_out(self):
        from quant import llm_router as R
        import inspect
        assert "allow_thinking_salvage" in inspect.signature(R.chat).parameters

    def test_weekly_disables_thinking_and_salvage(self):
        from quant import weekly_report
        import inspect
        src = inspect.getsource(weekly_report.llm_summarize)
        assert "disable_thinking=True" in src
        assert "allow_thinking_salvage=False" in src

    def test_weekly_discards_leaked_thinking(self, monkeypatch):
        """即使前两道防线都失守, 泄漏特征文本也必须被丢弃而不是展示给主人。"""
        from quant import weekly_report
        leaked = "用户要求根据提供的数据，用中文写80-150字总结...\n分析数据：\n1. 组合收益"
        monkeypatch.setattr(weekly_report.llm_router, "chat",
                            lambda *a, **k: {"text": leaked})
        assert weekly_report.llm_summarize({"pnl": {}}) == ""

    def test_weekly_keeps_real_summary(self, monkeypatch):
        from quant import weekly_report
        good = "本周组合呈现美元资产强势、人民币资产拖累的分化格局。建议关注 VRT 放量暴跌。"
        monkeypatch.setattr(weekly_report.llm_router, "chat",
                            lambda *a, **k: {"text": good})
        assert weekly_report.llm_summarize({"pnl": {}}) == good

    def test_chat_json_still_forces_no_thinking(self):
        """newswatch 等走 chat_json 的调用点靠这个默认值保护, 不能被改掉。"""
        from quant import llm_router as R
        import inspect
        assert 'setdefault("disable_thinking", True)' in inspect.getsource(R.chat_json)


# ---------------------------------------------------------------- D7b
class TestDecisionScoringCadence:
    """D7b: 复盘原来只有月度 timer, 9 月 10 日到期的决策要等 10 月 1 日才打分。"""

    def test_score_only_flag_exists(self):
        from quant import decision_review
        import inspect
        assert "--score-only" in inspect.getsource(decision_review.main)


# ---------------------------------------------------------------- G3
class TestBacktestUniverse:
    """G3: 回测网格在 2026-09-01 对 16 只标的穷举完毕, daemon 此后空转 9 天。

    后果不只是"没有新回测": 这台 OCI 免费实例靠回测服务撑 CPU 占用防回收
    (连续 7 天 <20% 会被收回), 队列空 → CPU 掉到 ~6%。
    A1 修好价格缓存后本地有 157 只标的的完整历史, 没理由只回测 16 只。
    """

    def test_universe_is_larger_than_portfolio(self):
        from quant import task_generator as T
        from quant import config as cfg_mod
        held = set(cfg_mod.all_symbols(cfg_mod.load("portfolio")))
        universe = T.seed_universe()
        assert len(universe) > len(held) * 3, \
            f"播种宇宙只有 {len(universe)} 只, 网格会很快再次穷举"

    def test_portfolio_symbols_come_first(self):
        """持仓/关注要排在前面并拿高优先级 —— 最有用的结果先跑出来。"""
        from quant import task_generator as T
        from quant import config as cfg_mod
        held = set(cfg_mod.all_symbols(cfg_mod.load("portfolio")))
        universe = T.seed_universe()
        head = universe[:len(held)]
        assert set(head) == held, "持仓标的没有排在播种列表最前面"

    def test_priority_tiers_distinct(self):
        from quant import task_generator as T
        assert T.PORTFOLIO_PRIORITY > T.UNIVERSE_PRIORITY

    def test_universe_mode_configurable(self):
        """必须能一行配置退回旧行为 (万一磁盘/CPU 吃紧)。"""
        from quant import config as cfg_mod
        cfg = (cfg_mod.load("strategies") or {}).get("backtest") or {}
        assert cfg.get("universe_mode") in ("full", "portfolio")

    def test_delisted_not_in_universe(self):
        """隔离目录 _delisted/ 里的标的不能被播种 (它们没有新数据了)。"""
        from quant import task_generator as T
        universe = set(T.seed_universe())
        for dead in ("CELT", "APLS", "APGE"):
            assert dead not in universe, f"已退市的 {dead} 又被播种了"


# ---------------------------------------------------------------- G4
class TestQuarantineRegistry:
    """G4 (自己新代码打架): 光把 parquet 移进 _delisted/ 不够。

    universe_symbols() 会从 dynamic_universe.yaml 把标的读回来, refresh() 再从
    yfinance 拉一份**退市前的**历史 (delisted 标的历史仍可下载), parquet 就复活了,
    下次 quarantine 再移走…… 实测: APGE 被隔离后 3 分钟就复活。
    隔离必须是持久化登记, 不能只靠移动文件。
    """

    def test_registry_functions_exist(self):
        from quant import price_refresh as P
        assert callable(P.quarantined_symbols)
        assert callable(P.unquarantine)

    def test_quarantined_excluded_from_refresh_universe(self):
        from quant import price_refresh as P
        banned = set(P.quarantined_symbols())
        if not banned:
            pytest.skip("当前无隔离标的")
        universe = set(P.universe_symbols())
        assert not (banned & universe), \
            f"已隔离标的仍在刷新宇宙里, 会被重新下载: {banned & universe}"

    def test_quarantined_excluded_from_seed_universe(self):
        from quant import price_refresh as P, task_generator as T
        banned = set(P.quarantined_symbols())
        if not banned:
            pytest.skip("当前无隔离标的")
        assert not (banned & set(T.seed_universe()))

    def test_registry_entries_have_reason(self):
        from quant import price_refresh as P
        for sym, meta in P.quarantined_symbols().items():
            assert meta.get("reason"), f"{sym} 的隔离登记没写原因"
            assert meta.get("last_bar"), f"{sym} 的隔离登记没写最后数据日期"

    def test_unquarantine_is_noop_for_unknown(self):
        from quant import price_refresh as P
        assert P.unquarantine("__NOT_A_REAL_SYMBOL__") is False

    def test_walk_forward_uses_same_universe(self):
        """G3b: walk_forward 是任务量的主体 (历史 120,153 条里 111,222 条是它)。
        如果只扩 seed() 而漏了这里, 普通任务跑完 (约 2 小时) 队列就又空了。
        """
        from quant import task_generator as T
        import inspect
        src = inspect.getsource(T.walk_forward)
        assert "seed_universe()" in src, "walk_forward 还在用旧的 portfolio-only 宇宙"
        assert "cfg_mod.all_symbols(portfolio)\n    periods" not in src

    def test_walk_forward_keeps_portfolio_priority(self):
        from quant import task_generator as T
        import inspect
        assert "wf_prio" in inspect.getsource(T.walk_forward)


# ---------------------------------------------------------------- H1 (2026-09-11)
class TestNanoGptFallback:
    """H1: 给每条 route 加 nano-gpt 作为跨 provider fallback。

    为什么必须跨 provider: Ollama Cloud 的所有 route 共享同一份配额, 打满时
    "fallback 到同一家的另一个模型"毫无用处 —— 近 30 天 183 次 429, simple_chat
    失败率曾到 31%, 因为整条链都在同一个配额池里。
    (接入当天实测: Ollama 正在 429, 所有请求直接落到 nano-gpt 才没失败。)
    """

    def test_provider_registered(self):
        from quant import config as cfg_mod
        providers = cfg_mod.load("llm_routes").get("providers") or {}
        assert "nanogpt" in providers
        p = providers["nanogpt"]
        assert p["type"] == "openai_compat"
        # 必须用环境变量引用, 绝不能把 key 写进这个**公开仓库**的 yaml
        assert p.get("api_key_env") == "NANOGPT_API_KEY"
        assert "sk-" not in str(p), "key 泄漏进 llm_routes.yaml 了"

    def test_every_route_has_cross_provider_fallback(self):
        """每条 route 至少要有两个不同 provider —— 否则单家故障就全断。"""
        from quant import config as cfg_mod
        routes = cfg_mod.load("llm_routes").get("routes") or {}
        assert routes
        for task, chain in routes.items():
            providers = {e.split(":", 1)[0] for e in chain}
            assert len(providers) >= 2, f"route {task} 只有一个 provider: {providers}"
            assert "nanogpt" in providers, f"route {task} 没挂 nano-gpt fallback"

    def test_short_output_routes_have_no_thinking_models(self):
        """需要快/短输出的 route 不能挂 thinking 模型。

        实测踩到: fast_reasoning 原挂 z-ai/glm-5.3-flash (thinking 模型),
        max_tokens=60 时思考吃光预算 → content 空 → llm_router 的"抢救 thinking"
        兜底把整段思考当答案返回 ("The user is asking...")。
        """
        from quant import config as cfg_mod
        routes = cfg_mod.load("llm_routes").get("routes") or {}
        SHORT = ("simple_chat", "format", "fast_reasoning")
        KNOWN_THINKING = ("thinking", "glm-5.3-flash")
        for task in SHORT:
            for entry in routes.get(task, []):
                low = entry.lower()
                for bad in KNOWN_THINKING:
                    assert bad not in low, f"route {task} 挂了 thinking 模型 {entry}"

    def test_thinking_models_only_in_deep_routes(self):
        from quant import config as cfg_mod
        routes = cfg_mod.load("llm_routes").get("routes") or {}
        for task in ("deep_reasoning", "review"):
            chain = routes.get(task, [])
            assert any("thinking" in e.lower() for e in chain), \
                f"route {task} 应该用 thinking 模型"

    def test_no_deepseek_anywhere(self):
        """主人明确不用 deepseek (幻觉严重) —— nano-gpt 上有 deepseek 模型, 别误选。"""
        from quant import config as cfg_mod
        cfg = cfg_mod.load("llm_routes")
        for task, chain in (cfg.get("routes") or {}).items():
            for entry in chain:
                assert "deepseek" not in entry.lower(), f"route {task} 选了 deepseek: {entry}"

    def test_provider_surfaces_reasoning_as_thinking(self):
        """nano-gpt 把思考放在 `reasoning` 字段 (不是 reasoning_content)。
        不暴露出来的话 content 为空时只能判"空回复"白扔一次已付费的调用。
        """
        from quant.llm_router import OpenAICompatProvider
        import inspect
        src = inspect.getsource(OpenAICompatProvider.chat)
        assert '"reasoning"' in src
        assert '"thinking"' in src

    def test_provider_surfaces_reported_cost(self):
        """nano-gpt 在 usage 里自报真实 cost —— 付费 provider 上本地价格表估算没意义。"""
        from quant.llm_router import OpenAICompatProvider
        import inspect
        assert "cost_usd_reported" in inspect.getsource(OpenAICompatProvider.chat)

    def test_audit_prefers_reported_cost(self):
        from quant import llm_router as R
        import inspect
        assert "cost_usd_reported" in inspect.signature(R._log_audit).parameters

    def test_nanogpt_not_in_local_price_table(self):
        """不该给它写死价格 —— 294 个模型且会变价, 用实报值。"""
        from quant import config as cfg_mod
        costs = cfg_mod.load("llm_routes").get("costs") or {}
        assert not any(k.startswith("nanogpt:") for k in costs)
