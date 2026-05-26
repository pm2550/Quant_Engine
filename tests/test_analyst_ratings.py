"""Tests for analyst_ratings + rating_changes (US / CN / ETF)."""
from __future__ import annotations

import pandas as pd
import pytest

from quant import analyst_ratings, rating_changes


# ---------- analyst_ratings._cn_ratings ----------

def _stub_research_df():
    return pd.DataFrame([
        {"序号": 1, "股票代码": "002624", "股票简称": "完美世界",
         "报告名称": "异环新品周期可期", "东财评级": "买入", "机构": "东吴证券",
         "2026-盈利预测-收益": 0.83, "2026-盈利预测-市盈率": 18.24,
         "日期": "2026-05-05"},
        {"序号": 2, "股票代码": "002624", "股票简称": "完美世界",
         "报告名称": "Q1点评", "东财评级": "买入", "机构": "国元证券",
         "2026-盈利预测-收益": 0.84, "2026-盈利预测-市盈率": 20.43,
         "日期": "2026-04-30"},
        {"序号": 3, "股票代码": "002624", "股票简称": "完美世界",
         "报告名称": "看好新增长期", "东财评级": "增持", "机构": "开源证券",
         "2026-盈利预测-收益": 0.87, "2026-盈利预测-市盈率": 19.60,
         "日期": "2026-04-29"},
        {"序号": 4, "股票代码": "002624", "股票简称": "完美世界",
         "报告名称": "新一代二游", "东财评级": "中性", "机构": "中信证券",
         "2026-盈利预测-收益": 0.80, "2026-盈利预测-市盈率": 17.00,
         "日期": "2026-04-22"},
    ])


def test_cn_ratings_maps_breakdown_and_rec_mean(monkeypatch):
    import akshare as ak
    monkeypatch.setattr(ak, "stock_research_report_em", lambda symbol: _stub_research_df())
    monkeypatch.setattr(analyst_ratings.fetcher, "load_local", lambda s: pd.DataFrame())

    out = analyst_ratings._cn_ratings("002624.SZ")
    assert out is not None
    assert out["market"] == "cn"
    # 买入=1, 买入=1, 增持=2, 中性=3 → mean = 1.75
    assert out["recommendation_mean"] == pytest.approx(1.75)
    assert out["recommendation_key"] == "买入"  # 最多
    assert out["rating_breakdown"] == {"买入": 2, "增持": 1, "中性": 1}
    assert out["number_of_analyst_opinions"] == 4
    assert len(out["recent_research"]) == 4
    assert out["recent_research"][0]["firm"] == "东吴证券"
    assert out["recent_research"][0]["rating"] == "买入"
    # A 股没有 target_mean_price (akshare 不再返回)
    assert "target_mean_price" not in out


def test_cn_ratings_empty_returns_none(monkeypatch):
    import akshare as ak
    monkeypatch.setattr(ak, "stock_research_report_em", lambda symbol: pd.DataFrame())
    out = analyst_ratings._cn_ratings("000001.SZ")
    assert out is None


def test_cn_ratings_unexpected_columns_returns_none(monkeypatch, caplog):
    import akshare as ak
    df = pd.DataFrame([{"foo": 1, "bar": 2}])
    monkeypatch.setattr(ak, "stock_research_report_em", lambda symbol: df)
    out = analyst_ratings._cn_ratings("002624.SZ")
    assert out is None


# ---------- analyst_ratings._etf_weighted_ratings ----------

class _FakeTopHoldings:
    def __init__(self, rows):
        self._df = pd.DataFrame(rows).set_index("Symbol")

    @property
    def empty(self):
        return self._df.empty

    def iterrows(self):
        return self._df.iterrows()


class _FakeFundsData:
    def __init__(self, rows):
        self.top_holdings = _FakeTopHoldings(rows)


class _FakeTicker:
    def __init__(self, funds_rows=None, info=None):
        self.funds_data = _FakeFundsData(funds_rows) if funds_rows is not None else None
        self.info = info or {}

    @property
    def recommendations(self):
        return None


def test_etf_weighted_ratings_aggregates_constituents(monkeypatch):
    """两个成分股 AMD (rec=1.5 upside=15%) 和 NVDA (rec=1.3 upside=20%) 权重 8% / 7% → 加权均值."""
    rows = [
        {"Symbol": "AMD", "Name": "AMD Inc", "Holding Percent": 0.08},
        {"Symbol": "NVDA", "Name": "NVIDIA Corp", "Holding Percent": 0.07},
    ]
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda s: _FakeTicker(funds_rows=rows))

    def fake_us(sym):
        return {
            "AMD": {"recommendation_mean": 1.5, "upside_pct": 15.0, "target_mean_price": 200, "current_price": 174},
            "NVDA": {"recommendation_mean": 1.3, "upside_pct": 20.0, "target_mean_price": 300, "current_price": 250},
        }[sym]

    monkeypatch.setattr(analyst_ratings, "_us_ratings", fake_us)
    out = analyst_ratings._etf_weighted_ratings("SOXX")
    assert out is not None
    assert out["market"] == "etf"
    # weighted_rec = (1.5*0.08 + 1.3*0.07) / 0.15 = (0.12 + 0.091) / 0.15 = 1.4067
    assert out["recommendation_mean"] == pytest.approx(1.4067, abs=0.001)
    # weighted_upside = (15*0.08 + 20*0.07) / 0.15 = (1.2 + 1.4) / 0.15 = 17.33
    assert out["weighted_target_upside_pct"] == pytest.approx(17.33, abs=0.01)
    assert out["coverage_weight"] == pytest.approx(0.15)
    assert out["n_constituents_with_rating"] == 2
    assert len(out["holdings"]) == 2


def test_etf_weighted_ratings_no_funds_data_returns_none(monkeypatch):
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda s: _FakeTicker(funds_rows=None))
    assert analyst_ratings._etf_weighted_ratings("AMD") is None


def test_etf_skips_constituents_without_rating(monkeypatch):
    """若 top holding 缺数据, 仍正确归一化 covered weight."""
    rows = [
        {"Symbol": "AMD", "Name": "AMD", "Holding Percent": 0.10},
        {"Symbol": "UNKNOWN", "Name": "UNK", "Holding Percent": 0.05},
    ]
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda s: _FakeTicker(funds_rows=rows))
    monkeypatch.setattr(analyst_ratings, "_us_ratings", lambda s: (
        {"recommendation_mean": 1.5, "upside_pct": 15.0} if s == "AMD" else None
    ))
    out = analyst_ratings._etf_weighted_ratings("SOXX")
    assert out["recommendation_mean"] == pytest.approx(1.5)
    assert out["coverage_weight"] == pytest.approx(0.10)


# ---------- analyst_ratings.fetch_one dispatch ----------

def test_fetch_one_dispatches_a_share(monkeypatch):
    called = {}
    monkeypatch.setattr(analyst_ratings, "_cn_ratings", lambda s: (called.setdefault("cn", s), {"market": "cn"})[1])
    monkeypatch.setattr(analyst_ratings, "_us_ratings", lambda s: pytest.fail("US should not be called for A-share"))
    out = analyst_ratings.fetch_one("002624.SZ")
    assert called["cn"] == "002624.SZ"
    assert out == {"market": "cn"}


def test_fetch_one_us_individual_stock(monkeypatch):
    monkeypatch.setattr(analyst_ratings, "_us_ratings",
                        lambda s: {"market": "us", "recommendation_mean": 1.5, "target_mean_price": 200})
    monkeypatch.setattr(analyst_ratings, "_etf_weighted_ratings",
                        lambda s: pytest.fail("ETF fallback should NOT trigger when US has data"))
    out = analyst_ratings.fetch_one("AMD")
    assert out["market"] == "us"


def test_fetch_one_falls_back_to_etf_when_us_empty(monkeypatch):
    """ETF 命中: _us_ratings 返回的 info 没有 rec_mean/target → 走 ETF fallback."""
    monkeypatch.setattr(analyst_ratings, "_us_ratings", lambda s: {"market": "us"})
    monkeypatch.setattr(analyst_ratings, "_etf_weighted_ratings",
                        lambda s: {"market": "etf", "recommendation_mean": 1.4})
    out = analyst_ratings.fetch_one("SOXX")
    assert out["market"] == "etf"


# ---------- rating_changes.diff_snapshot ----------

def test_diff_us_target_price_jump():
    prev = {"market": "us", "target_mean_price": 200.0, "recommendation_mean": 2.0,
            "recent_actions": []}
    cur = {"market": "us", "target_mean_price": 230.0, "recommendation_mean": 2.0,
           "recent_actions": []}
    changes = rating_changes.diff_snapshot(prev, cur, sym="AMD")
    assert any("目标价" in c and "+15.0%" in c for c in changes)


def test_diff_us_rec_mean_upgrade():
    prev = {"market": "us", "recommendation_mean": 2.5, "recent_actions": []}
    cur = {"market": "us", "recommendation_mean": 1.8, "recent_actions": []}
    changes = rating_changes.diff_snapshot(prev, cur, sym="AMD")
    assert any("评级转好" in c for c in changes)


def test_diff_us_no_change_below_threshold():
    prev = {"market": "us", "target_mean_price": 200, "recommendation_mean": 2.0,
            "recent_actions": ["2026-05-01|Goldman|Upgrade|Buy→Buy"]}
    cur = {"market": "us", "target_mean_price": 205, "recommendation_mean": 2.05,
           "recent_actions": ["2026-05-01|Goldman|Upgrade|Buy→Buy"]}  # 2.5% / 0.05
    changes = rating_changes.diff_snapshot(prev, cur, sym="AMD")
    assert changes == []


def test_diff_us_new_firm_action():
    prev = {"market": "us", "recent_actions": []}
    cur = {"market": "us", "recent_actions": [
        "2026-05-25|Morgan Stanley|Upgrade|Hold→Buy",
    ]}
    changes = rating_changes.diff_snapshot(prev, cur, sym="AMD")
    assert any("Morgan Stanley" in c and "Upgrade" in c for c in changes)


def test_diff_cn_uses_cny_unit_and_breakdown_shift():
    prev = {"market": "cn", "rating_breakdown": {"买入": 5, "增持": 2, "中性": 1},
            "recommendation_mean": 1.8, "recent_actions": []}
    cur = {"market": "cn", "rating_breakdown": {"买入": 10, "增持": 2, "中性": 1},
           "recommendation_mean": 1.8, "recent_actions": []}
    changes = rating_changes.diff_snapshot(prev, cur, sym="002624.SZ")
    assert any("买入" in c and "+5" in c for c in changes)
    # 5 + 5 + 2 + 1 不应触发其他 evaluations 误报
    assert all("$" not in c for c in changes)


def test_diff_cn_new_research_report():
    prev = {"market": "cn", "recent_actions": []}
    cur = {"market": "cn", "recent_actions": [
        "2026-05-25|中信证券|买入|异环上线驱动业绩",
    ]}
    changes = rating_changes.diff_snapshot(prev, cur, sym="002624.SZ")
    assert any("中信证券" in c and "买入" in c for c in changes)


def test_diff_etf_weighted_upside_shift():
    prev = {"market": "etf", "weighted_target_upside_pct": 10.0, "recommendation_mean": 1.8,
            "recent_actions": []}
    cur = {"market": "etf", "weighted_target_upside_pct": 18.0, "recommendation_mean": 1.8,
           "recent_actions": []}
    changes = rating_changes.diff_snapshot(prev, cur, sym="SOXX")
    assert any("加权上涨空间" in c for c in changes)


def test_diff_etf_rec_mean_labelled():
    prev = {"market": "etf", "recommendation_mean": 2.2, "recent_actions": []}
    cur = {"market": "etf", "recommendation_mean": 1.7, "recent_actions": []}
    changes = rating_changes.diff_snapshot(prev, cur, sym="SOXX")
    assert any("(ETF 加权)" in c for c in changes)


# ---------- rating_changes._recent_actions extraction ----------

def test_recent_actions_us_format():
    data = {"market": "us", "recent_changes": [
        {"date": "2026-05-01", "firm": "Goldman", "action": "Upgrade",
         "from_grade": "Hold", "to_grade": "Buy"},
    ]}
    acts = rating_changes._recent_actions(data)
    assert acts == ["2026-05-01|Goldman|Upgrade|Hold→Buy"]


def test_recent_actions_cn_format():
    data = {"market": "cn", "recent_research": [
        {"date": "2026-05-05", "firm": "东吴证券", "rating": "买入", "title": "异环点评"},
    ]}
    acts = rating_changes._recent_actions(data)
    assert acts == ["2026-05-05|东吴证券|买入|异环点评"]


def test_recent_actions_etf_returns_empty():
    data = {"market": "etf"}
    assert rating_changes._recent_actions(data) == []
