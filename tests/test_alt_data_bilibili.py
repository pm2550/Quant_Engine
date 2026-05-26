"""Unit tests for alt_data.bilibili — store/trend math (network calls mocked)."""
from __future__ import annotations
from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def temp_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / "altdata.sqlite"
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp)
    db.init()
    yield db
    if tmp.exists():
        tmp.unlink()


def test_store_snapshot_writes_row(temp_db):
    from quant.alt_data import bilibili
    metrics = {"total_results": 1000, "top_avg_plays": 100000,
                "recent_7d_in_top30": 15, "top_videos": []}
    bilibili.store_snapshot("异环", metrics, metric_date="2026-05-06")
    with temp_db.conn() as c:
        row = c.execute("SELECT * FROM alt_data_metrics WHERE key='异环'").fetchone()
    assert row["source"] == "bilibili_search"
    assert row["metric_date"] == "2026-05-06"


def test_store_snapshot_idempotent_on_same_day(temp_db):
    """REPLACE on same (source, key, metric_date) should not duplicate."""
    from quant.alt_data import bilibili
    bilibili.store_snapshot("X", {"total_results": 100}, metric_date="2026-05-06")
    bilibili.store_snapshot("X", {"total_results": 200}, metric_date="2026-05-06")
    with temp_db.conn() as c:
        n = c.execute("SELECT COUNT(*) FROM alt_data_metrics WHERE key='X'").fetchone()[0]
    assert n == 1


def test_trend_needs_at_least_2_snapshots(temp_db):
    from quant.alt_data import bilibili
    bilibili.store_snapshot("X", {"total_results": 100}, metric_date="2026-05-06")
    out = bilibili.trend("X")
    assert "error" in out


def test_trend_computes_pct_changes(temp_db):
    """7d ago = 100 results, today = 200 → +100% trend."""
    from quant.alt_data import bilibili
    # Older snapshot (8 days ago)
    bilibili.store_snapshot("X", {"total_results": 100, "top_avg_plays": 1000,
                                     "recent_7d_in_top30": 5},
                              metric_date="2026-04-28")
    # Recent snapshot (today)
    bilibili.store_snapshot("X", {"total_results": 200, "top_avg_plays": 2500,
                                     "recent_7d_in_top30": 12},
                              metric_date="2026-05-06")
    out = bilibili.trend("X")
    assert out["n_snapshots"] == 2
    assert out["today"]["total_results"] == 200
    # 7d_ago index falls back to oldest if < 7 entries
    assert out["vs_7d_ago"]["total_results_pct"] == 100.0
    assert out["vs_7d_ago"]["top_avg_plays_pct"] == 150.0


def test_snapshot_keyword_handles_api_failure(monkeypatch, temp_db):
    """Network errors should propagate as exceptions, not silent fail."""
    from quant.alt_data import bilibili

    def boom(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(bilibili, "_fetch_search", boom)
    with pytest.raises(RuntimeError):
        bilibili.snapshot_keyword("X")


# ---- analyze_with_llm ----


def test_analyze_with_llm_returns_normalized_dict(monkeypatch, temp_db):
    """LLM gives proper JSON → return cleaned-up dict."""
    from quant.alt_data import bilibili

    def fake_chat_json(prompt, **kw):
        return {
            "overall_sentiment": 0.4,
            "breakdown": {"positive": 12, "neutral": 5, "negative": 3},
            "key_themes": ["技术力突破"],
            "concern_signals": ["商业化激进"],
            "positive_signals": ["美术好"],
            "buzz_phase": "early_excitement",
            "reasoning": "正面主导",
        }
    monkeypatch.setattr(bilibili.llm_router, "chat_json", fake_chat_json)
    monkeypatch.setattr(bilibili.prompts, "load", lambda name: "{keyword} {top_videos_text}")

    out = bilibili.analyze_with_llm("test", [{"title": "x", "play": 100}])
    assert out["overall_sentiment"] == 0.4
    assert out["buzz_phase"] == "early_excitement"
    assert out["breakdown"]["positive"] == 12
    assert "商业化激进" in out["concern_signals"]


def test_analyze_with_llm_returns_error_dict_on_failure(monkeypatch, temp_db):
    from quant.alt_data import bilibili

    def boom(*a, **kw):
        raise RuntimeError("LLM down")
    monkeypatch.setattr(bilibili.llm_router, "chat_json", boom)
    monkeypatch.setattr(bilibili.prompts, "load", lambda name: "x")

    out = bilibili.analyze_with_llm("test", [{"title": "x", "play": 1}])
    assert "error" in out


def test_analyze_with_llm_empty_videos_returns_error(temp_db):
    from quant.alt_data import bilibili
    out = bilibili.analyze_with_llm("test", [])
    assert "error" in out


# ---- formatter ----


def test_formatter_renders_keyword_block(temp_db):
    """Render markdown with sentiment + themes when LLM data present."""
    from quant.alt_data import bilibili, formatter
    metrics = {
        "total_results": 1000, "top_avg_plays": 500000, "recent_7d_in_top30": 18,
        "sentiment": {
            "overall_sentiment": 0.3,
            "breakdown": {"positive": 10, "neutral": 6, "negative": 4},
            "key_themes": ["技术力强"],
            "concern_signals": ["bug 多"],
            "positive_signals": ["美术好"],
            "buzz_phase": "early_excitement",
            "reasoning": "正面主导",
        },
    }
    bilibili.store_snapshot("异环 完美世界", metrics, metric_date="2026-05-06")
    out = formatter.render_for_keyword("异环 完美世界", "002624.SZ")
    assert "002624.SZ" in out
    assert "异环 完美世界" in out
    assert "1000" in out  # total_results
    assert "正 10" in out
    assert "技术力强" in out
    assert "🚀" in out  # early_excitement emoji


def test_formatter_no_snapshot_returns_empty(temp_db):
    from quant.alt_data import formatter
    assert formatter.render_for_keyword("nonexistent_keyword") == ""


# ---- anomaly detection ----


def test_anomaly_volume_drop_fires(monkeypatch, temp_db):
    from quant.alt_data import bilibili, anomaly
    # 8 days ago: 1000 videos, top_avg_plays 500k
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 1000, "top_avg_plays": 500000,
                               "sentiment": {"overall_sentiment": 0.3,
                                              "buzz_phase": "early_excitement"}},
                              metric_date="2026-04-28")
    # Today: 600 videos (-40%), top_avg_plays 250k (-50%) — both should fire
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 600, "top_avg_plays": 250000,
                               "sentiment": {"overall_sentiment": 0.3,
                                              "buzz_phase": "early_excitement"}},
                              metric_date="2026-05-06")
    fires = anomaly.check_keyword("异环 完美世界", dry_run=True)
    sigs = {f["signal"] for f in fires}
    assert "volume_drop" in sigs
    assert "plays_drop" in sigs


def test_anomaly_phase_shift_to_controversy_fires(monkeypatch, temp_db):
    from quant.alt_data import bilibili, anomaly
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 1000, "top_avg_plays": 500000,
                               "sentiment": {"overall_sentiment": 0.3,
                                              "buzz_phase": "sustained"}},
                              metric_date="2026-04-28")
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 1000, "top_avg_plays": 500000,
                               "sentiment": {"overall_sentiment": 0.0,
                                              "buzz_phase": "controversy"}},
                              metric_date="2026-05-06")
    fires = anomaly.check_keyword("异环 完美世界", dry_run=True)
    sigs = {f["signal"] for f in fires}
    assert "phase_shift" in sigs


def test_anomaly_sentiment_drop_fires(monkeypatch, temp_db):
    from quant.alt_data import bilibili, anomaly
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 1000, "top_avg_plays": 500000,
                               "sentiment": {"overall_sentiment": 0.5,
                                              "buzz_phase": "sustained"}},
                              metric_date="2026-04-28")
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 1000, "top_avg_plays": 500000,
                               "sentiment": {"overall_sentiment": -0.1,
                                              "buzz_phase": "sustained"}},
                              metric_date="2026-05-06")
    fires = anomaly.check_keyword("异环 完美世界", dry_run=True)
    sigs = {f["signal"] for f in fires}
    assert "sentiment_drop" in sigs


def test_anomaly_no_fire_when_stable(monkeypatch, temp_db):
    from quant.alt_data import bilibili, anomaly
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 1000, "top_avg_plays": 500000,
                               "sentiment": {"overall_sentiment": 0.3,
                                              "buzz_phase": "sustained"}},
                              metric_date="2026-04-28")
    bilibili.store_snapshot("异环 完美世界",
                              {"total_results": 1010, "top_avg_plays": 510000,
                               "sentiment": {"overall_sentiment": 0.32,
                                              "buzz_phase": "sustained"}},
                              metric_date="2026-05-06")
    assert anomaly.check_keyword("异环 完美世界", dry_run=True) == []
