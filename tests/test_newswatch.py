"""Unit tests for newswatch.py — keyword filter, severity emoji, render formatting.

The LLM-touching paths (`score_severity`, `derive_impact`) and DB-touching
paths (`_dedupe_and_store`, `_record_event`) are integration scope and not
covered here.  Pure functions = pre-filter logic, severity → emoji mapping,
and Telegram message rendering are the high-value targets.
"""
from __future__ import annotations

import pytest

from quant import newswatch


# ---- Keyword pre-filter ----


def test_kw_filter_drops_ignore_terms():
    cfg = {"ignore": ["sports", "celebrity"], "high": [], "medium": []}
    keep, boost = newswatch._kw_filter("Lakers win NBA championship", "sports recap", cfg)
    assert keep is False
    assert boost == 0


def test_kw_filter_high_keyword_boosts_2():
    cfg = {"ignore": [], "high": ["Fed", "rate hike"], "medium": []}
    keep, boost = newswatch._kw_filter("Fed signals rate hike", "details...", cfg)
    assert keep is True
    assert boost == 2


def test_kw_filter_medium_keyword_boosts_1():
    cfg = {"ignore": [], "high": ["Fed"], "medium": ["earnings"]}
    keep, boost = newswatch._kw_filter("Apple Q3 earnings", "results", cfg)
    assert keep is True
    assert boost == 1


def test_kw_filter_high_takes_priority_over_medium():
    cfg = {"ignore": [], "high": ["Fed"], "medium": ["earnings"]}
    # Both "Fed" and "earnings" present — high wins
    keep, boost = newswatch._kw_filter("Fed earnings preview", "", cfg)
    assert keep is True
    assert boost == 2


def test_kw_filter_case_insensitive():
    cfg = {"ignore": ["SPORTS"], "high": ["FED"], "medium": []}
    keep_drop, _ = newswatch._kw_filter("nba sports update", "", cfg)
    keep_high, boost_high = newswatch._kw_filter("fed rate watch", "", cfg)
    assert keep_drop is False
    assert keep_high is True and boost_high == 2


def test_kw_filter_missing_categories_are_safe():
    """Empty/missing config keys shouldn't crash — should default to keep, no boost."""
    keep, boost = newswatch._kw_filter("any title", "any content", {})
    assert keep is True
    assert boost == 0


# ---- Severity → emoji mapping ----


def test_emoji_for_severity_buckets():
    assert newswatch._emoji(10) == "🚨🚨"
    assert newswatch._emoji(9) == "🚨🚨"
    assert newswatch._emoji(8) == "🚨"
    assert newswatch._emoji(7) == "🚨"
    assert newswatch._emoji(6) == "⚡"
    assert newswatch._emoji(5) == "⚡"
    assert newswatch._emoji(4) == "📰"
    assert newswatch._emoji(3) == "📰"
    assert newswatch._emoji(2) == "•"
    assert newswatch._emoji(0) == "•"


def test_direction_emoji_known_and_unknown():
    assert newswatch._direction_emoji("bullish") == "📈"
    assert newswatch._direction_emoji("bearish") == "📉"
    assert newswatch._direction_emoji("neutral") == "➖"
    assert newswatch._direction_emoji("garbage") == "•"


# ---- Alert rendering ----


def _item(title="Fed cuts rates 50bp", source="reuters_world",
           url="https://reuters.com/x", published="2026-05-05T12:00:00Z",
           content="Body."):
    return {
        "title": title, "source": source, "url": url,
        "published_at": published, "content": content,
    }


def _sev(severity=8, category="macro"):
    return {"severity": severity, "category": category, "reasoning": "r"}


def _impact(summary="Risk-on for big tech",
             impacts=None, secondary=None, action=None):
    return {
        "summary": summary,
        "impacts": impacts or [],
        "secondary_assets": secondary or [],
        "action_suggestion": action,
    }


def test_render_alert_includes_severity_and_title(monkeypatch):
    """v4: matches current render_alert, which no longer renders impact['summary']."""
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: [])
    out = newswatch.render_alert(_item(), _sev(8, "macro"), _impact())
    assert "事件等级 8/10" in out
    assert "macro" in out
    assert "Fed cuts rates 50bp" in out
    assert "🚨" in out  # severity 8
    assert "reuters.com" in out


def test_render_alert_per_holding_lines(monkeypatch):
    """v4: render shows base_rate (n=X 中位 ...) only — no LLM direction/confidence
    (accuracy audit found direction hit-rate ~50%, i.e. a coin flip)."""
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: [])
    impacts = [
        {"symbol": "VOO", "direction": "bullish", "confidence": 0.8,
         "reasoning": "lower discount rate",
         "base_rate": {"n_samples": 5,
                        "fwd_5d_pct": {"median": 1.2, "min": -0.5, "max": 3.0, "n": 5},
                        "fwd_20d_pct": {"median": 2.5, "min": -1.0, "max": 6.0, "n": 5},
                        "max_dd_within_max_window_pct": {"median": -2.0, "min": -5.0,
                                                          "max": -0.5, "n": 5}}},
        {"symbol": "GRID", "direction": "neutral", "confidence": 0.4,
         "reasoning": "limited direct exposure", "base_rate": None},
    ]
    out = newswatch.render_alert(_item(), _sev(7), _impact(impacts=impacts))
    assert "相关持仓" in out
    assert "`VOO`" in out
    assert "5d 中位 +1.2%" in out  # base rate, NOT LLM-written magnitude
    assert "`GRID`" in out
    assert "无足够样本" in out  # GRID has no base_rate
    assert "bullish" not in out and "📈" not in out  # direction guess dropped
    assert "0.8" not in out  # confidence dropped


def test_render_alert_filters_low_similarity(monkeypatch):
    """Similar events below 0.6 similarity should not appear."""
    fake_similars = [
        {"fired_at": "2024-01-15", "severity": 7, "similarity": 0.55,
         "summary": "should be filtered"},
        {"fired_at": "2024-03-10", "severity": 8, "similarity": 0.72,
         "summary": "should appear"},
    ]
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: fake_similars)
    out = newswatch.render_alert(_item(), _sev(8), _impact())
    assert "should appear" in out
    assert "should be filtered" not in out
    assert "历史相似事件" in out


def test_render_alert_handles_similar_lookup_failure(monkeypatch):
    """If similar_event lookup raises, render should still succeed."""
    def boom(*a, **k):
        raise RuntimeError("embedding service down")
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", boom)
    out = newswatch.render_alert(_item(), _sev(8), _impact())
    assert "Fed cuts rates 50bp" in out
    assert "历史相似事件" not in out


def test_render_alert_action_hint_no_longer_rendered(monkeypatch):
    """v4: action_hint (an LLM guess) is dropped from alerts along with direction/
    confidence — same accuracy-audit rationale."""
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: [])
    out = newswatch.render_alert(_item(), _sev(8),
                                  {"summary": "x", "impacts": [],
                                   "secondary_assets": [],
                                   "action_hint": "等待回调"})
    assert "🎯" not in out
    assert "等待回调" not in out


# ---- Portfolio context injection (P0 fix) ----


def test_portfolio_lines_includes_positions_and_watchlist():
    p = {
        "positions": {
            "VOO": {"name": "Vanguard S&P", "shares": 1, "currency": "USD"},
            "002624.SZ": {"name": "完美世界", "shares": 200, "currency": "CNY"},
        },
        "watchlist": [{"symbol": "NVDA"}, {"symbol": "ARM"}],
    }
    out = newswatch._portfolio_lines(p)
    assert "VOO (Vanguard S&P, 1股, USD)" in out
    assert "002624.SZ (完美世界, 200股, CNY)" in out
    assert "NVDA (关注池)" in out
    assert "ARM (关注池)" in out


def test_portfolio_lines_empty_safely():
    assert "空" in newswatch._portfolio_lines({"positions": {}, "watchlist": []})


# ---- Snapshot rendering ----


def test_build_snapshots_renders_signals_per_symbol(monkeypatch):
    """Snapshot should include price, RSI, MA state, and 20d chg."""
    import pandas as pd, numpy as np
    fake_df = pd.DataFrame({
        "open": [100]*60, "high": [101]*60, "low": [99]*60,
        "close": np.linspace(100, 110, 60), "volume": [1000]*60,
    }, index=pd.date_range("2025-01-01", periods=60, freq="B"))
    from quant import fetcher, signals
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_df)

    portfolio = {"positions": {"AMD": {"shares": 1, "currency": "USD"}}}
    out = newswatch._build_snapshots(portfolio)
    assert "AMD" in out
    assert "现价" in out
    assert "RSI" in out
    assert "20日" in out
    assert "MA50" in out


def test_build_snapshots_handles_missing_data(monkeypatch):
    """Missing price file should not crash the whole prompt."""
    import pandas as pd
    from quant import fetcher
    monkeypatch.setattr(fetcher, "load_local", lambda s: pd.DataFrame())
    portfolio = {"positions": {"GHOST": {"shares": 1, "currency": "USD"}}}
    out = newswatch._build_snapshots(portfolio)
    assert "GHOST" in out
    assert "无本地行情数据" in out


# ---- Similar event rendering ----


def test_build_similar_history_filters_by_threshold(monkeypatch):
    sims = [
        {"fired_at": "2024-01-15", "severity": 8, "similarity": 0.45,
         "summary": "should drop"},
        {"fired_at": "2024-03-10", "severity": 9, "similarity": 0.72,
         "summary": "should keep"},
    ]
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: sims)
    out = newswatch._build_similar_history({"title": "x"}, {"reasoning": ""})
    assert "should keep" in out
    assert "should drop" not in out


def test_build_similar_history_empty_when_no_match(monkeypatch):
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: [])
    out = newswatch._build_similar_history({"title": "x"}, {"reasoning": ""})
    assert "无显著相似" in out


def test_build_similar_history_swallows_lookup_failure(monkeypatch):
    """Embedding service down must not break impact derivation."""
    def boom(*a, **k):
        raise RuntimeError("vector db offline")
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", boom)
    out = newswatch._build_similar_history({"title": "x"}, {"reasoning": ""})
    assert "不可用" in out


# ---- derive_impact integration (mocked LLM) ----


def test_derive_impact_uses_format_task_for_strict_json(monkeypatch):
    """v3: derive_impact routes to task='format' (dashscope qwen3.6-plus json mode)
    and includes snapshots + similar_history blocks in prompt."""
    captured = {}

    def fake_chat_json(prompt, *, task, **kw):
        captured["task"] = task
        captured["prompt"] = prompt
        return {"summary": "test", "impacts": [], "secondary_assets": []}

    monkeypatch.setattr(newswatch.llm_router, "chat_json", fake_chat_json)
    monkeypatch.setattr(newswatch, "_build_snapshots", lambda p: "  - VOO: snap")
    monkeypatch.setattr(newswatch, "_build_similar_history", lambda *a, **k: "  - hist")
    monkeypatch.setattr(newswatch.similar_event, "find_similar", lambda q, **kw: [])

    item = {"title": "Fed cuts", "source": "reuters", "content": "..."}
    sev = {"severity": 8, "category": "macro", "reasoning": "r"}
    portfolio = {"positions": {"VOO": {"name": "V", "shares": 1, "currency": "USD"}},
                 "watchlist": []}

    newswatch.derive_impact(item, sev, portfolio=portfolio)

    assert captured["task"] == "format"
    assert "snap" in captured["prompt"]
    assert "hist" in captured["prompt"]


def test_derive_impact_returns_safe_default_on_llm_failure(monkeypatch):
    """LLM error must not propagate — newswatch loop relies on this."""
    monkeypatch.setattr(newswatch.llm_router, "chat_json",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("all backends failed")))
    monkeypatch.setattr(newswatch, "_build_snapshots", lambda p: "")
    monkeypatch.setattr(newswatch, "_build_similar_history", lambda *a, **k: "")

    out = newswatch.derive_impact({"title": "x", "source": "y", "content": "z"},
                                    {"severity": 5, "category": "macro"},
                                    portfolio={"positions": {}, "watchlist": []})
    assert out["impacts"] == []
    assert "失败" in out["summary"]


def test_score_severity_injects_portfolio_into_system_prompt(monkeypatch):
    """The whole point of the P0 fix — system prompt must contain holdings list."""
    captured = {}

    def fake_chat_json(prompt, *, task, system, **kw):
        captured["system"] = system
        return {"severity": 6, "category": "macro",
                "portfolio_relevance": "medium",
                "mentioned_holdings": [], "reasoning": "r"}

    monkeypatch.setattr(newswatch.llm_router, "chat_json", fake_chat_json)
    portfolio = {"positions": {"VOO": {"name": "Vanguard S&P", "shares": 1, "currency": "USD"}},
                 "watchlist": []}
    item = {"title": "Fed pivots", "content": "...", "source": "reuters", "region": "US"}
    out = newswatch.score_severity(item, portfolio=portfolio)
    assert "VOO" in captured["system"]
    assert "Vanguard S&P" in captured["system"]
    assert out["portfolio_relevance"] == "medium"


# ---- score_severity_batch: one LLM call scores multiple items ----


def _batch_items(n=3):
    return [
        {"title": f"headline {i}", "content": f"body {i}", "source": "reuters", "region": "US"}
        for i in range(n)
    ]


def test_score_severity_batch_empty_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not call LLM for empty batch")
    monkeypatch.setattr(newswatch.llm_router, "chat_json", boom)
    assert newswatch.score_severity_batch([]) == []


def test_score_severity_batch_aligns_results_by_id(monkeypatch):
    """Response order may differ from input order — must align by id, not position."""
    def fake_chat_json(prompt, *, task, system, **kw):
        return {"results": [
            {"id": 2, "severity": 3, "category": "other",
             "portfolio_relevance": "none", "mentioned_holdings": [], "reasoning": "c"},
            {"id": 0, "severity": 8, "category": "macro",
             "portfolio_relevance": "high", "mentioned_holdings": ["VOO"], "reasoning": "a"},
            {"id": 1, "severity": 5, "category": "industry",
             "portfolio_relevance": "low", "mentioned_holdings": [], "reasoning": "b"},
        ]}
    monkeypatch.setattr(newswatch.llm_router, "chat_json", fake_chat_json)
    out = newswatch.score_severity_batch(_batch_items(3))
    assert [o["severity"] for o in out] == [8, 5, 3]
    assert out[0]["mentioned_holdings"] == ["VOO"]


def test_score_severity_batch_applies_per_item_kw_boost(monkeypatch):
    def fake_chat_json(prompt, *, task, system, **kw):
        return {"results": [
            {"id": 0, "severity": 5, "category": "macro",
             "portfolio_relevance": "medium", "mentioned_holdings": [], "reasoning": "a"},
            {"id": 1, "severity": 5, "category": "macro",
             "portfolio_relevance": "medium", "mentioned_holdings": [], "reasoning": "b"},
        ]}
    monkeypatch.setattr(newswatch.llm_router, "chat_json", fake_chat_json)
    out = newswatch.score_severity_batch(_batch_items(2), kw_boosts=[2, 0])
    assert out[0]["severity"] == 7
    assert out[1]["severity"] == 5


def test_score_severity_batch_missing_id_defaults_to_zero(monkeypatch):
    """If the model drops an item from its response, that slot must not crash —
    it should default to severity 0 while the rest still score normally."""
    def fake_chat_json(prompt, *, task, system, **kw):
        return {"results": [
            {"id": 0, "severity": 7, "category": "macro",
             "portfolio_relevance": "high", "mentioned_holdings": [], "reasoning": "a"},
            # id=1 missing entirely
        ]}
    monkeypatch.setattr(newswatch.llm_router, "chat_json", fake_chat_json)
    out = newswatch.score_severity_batch(_batch_items(2))
    assert out[0]["severity"] == 7
    assert out[1]["severity"] == 0


def test_score_severity_batch_llm_failure_defaults_all_to_zero(monkeypatch):
    """A whole-batch LLM failure must not raise — every item degrades to severity 0,
    matching score_severity's single-item error behavior."""
    def boom(*a, **k):
        raise RuntimeError("all backends failed")
    monkeypatch.setattr(newswatch.llm_router, "chat_json", boom)
    out = newswatch.score_severity_batch(_batch_items(3))
    assert len(out) == 3
    assert all(o["severity"] == 0 for o in out)


def test_score_severity_batch_injects_portfolio_and_ids(monkeypatch):
    captured = {}

    def fake_chat_json(prompt, *, task, system, **kw):
        captured["system"] = system
        captured["prompt"] = prompt
        return {"results": [
            {"id": 0, "severity": 4, "category": "other",
             "portfolio_relevance": "none", "mentioned_holdings": [], "reasoning": "a"},
        ]}
    monkeypatch.setattr(newswatch.llm_router, "chat_json", fake_chat_json)
    portfolio = {"positions": {"VOO": {"name": "Vanguard S&P", "shares": 1, "currency": "USD"}},
                 "watchlist": []}
    newswatch.score_severity_batch(_batch_items(1), portfolio=portfolio)
    assert "VOO" in captured["system"]
    assert "[id=0]" in captured["prompt"]


# ---- v3 architecture: LLM does NOT write magnitude; engine computes base rate ----


def test_compute_base_rate_returns_forward_returns(monkeypatch):
    """Given historical similar events + a symbol, compute real fwd returns."""
    import pandas as pd, numpy as np
    # 100 trading days of synthetic prices
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    closes = np.linspace(100, 130, 100)  # +30% over 100d
    fake_df = pd.DataFrame({"close": closes}, index=dates)
    monkeypatch.setattr(newswatch.fetcher, "load_local", lambda s: fake_df)

    similar = [
        {"fired_at": "2024-02-15", "severity": 8, "similarity": 0.7},
        {"fired_at": "2024-03-15", "severity": 7, "similarity": 0.65},
    ]
    out = newswatch._compute_base_rate("FAKE", similar)
    assert out is not None
    assert out["n_samples"] >= 2
    assert "fwd_5d_pct" in out and "fwd_20d_pct" in out
    # On a +30% linear trend over 100d, fwd_5d should be roughly positive
    assert out["fwd_20d_pct"]["median"] > 0


def test_compute_base_rate_returns_none_for_empty_events(monkeypatch):
    out = newswatch._compute_base_rate("AMD", [])
    assert out is None


def test_compute_base_rate_handles_missing_symbol(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(newswatch.fetcher, "load_local", lambda s: pd.DataFrame())
    out = newswatch._compute_base_rate("NOPE",
                                         [{"fired_at": "2024-01-01", "severity": 8, "similarity": 0.7}])
    assert out is None


def test_derive_impact_replaces_llm_magnitude_with_base_rate(monkeypatch):
    """v3 contract: even if LLM tries to write magnitude_pct, output keeps only base_rate."""
    import pandas as pd, numpy as np
    fake_df = pd.DataFrame({"close": np.linspace(100, 130, 100)},
                            index=pd.date_range("2024-01-01", periods=100, freq="B"))
    monkeypatch.setattr(newswatch.fetcher, "load_local", lambda s: fake_df)
    monkeypatch.setattr(newswatch, "_build_snapshots", lambda p: "  - VOO: snap")
    monkeypatch.setattr(newswatch, "_build_similar_history", lambda *a, **k: "  - hist")
    monkeypatch.setattr(newswatch.similar_event, "find_similar",
                         lambda q, **k: [{"fired_at": "2024-02-15", "severity": 8,
                                           "similarity": 0.7, "summary": "x"}])

    def fake_chat_json(prompt, **kw):
        # Simulate LLM that ignored instructions and still emitted magnitude_pct
        return {
            "summary": "test event",
            "impacts": [
                {"symbol": "VOO", "direction": "bearish", "confidence": 0.7,
                 "magnitude_pct": -2.5,  # ← v3 must drop this
                 "reasoning": "trade tensions"}
            ],
            "secondary_assets": [],
            "action_hint": "观察",
        }
    monkeypatch.setattr(newswatch.llm_router, "chat_json", fake_chat_json)

    portfolio = {"positions": {"VOO": {"name": "V", "shares": 1, "currency": "USD"}},
                 "watchlist": []}
    out = newswatch.derive_impact({"title": "x", "source": "y", "content": "z"},
                                    {"severity": 8, "category": "macro",
                                     "reasoning": "r"},
                                    portfolio=portfolio)
    assert out["impacts"][0]["symbol"] == "VOO"
    # CRITICAL: LLM-written magnitude_pct must be stripped
    assert "magnitude_pct" not in out["impacts"][0]
    # base_rate from real historical computation must be present
    assert out["impacts"][0]["base_rate"] is not None
    assert out["impacts"][0]["base_rate"]["n_samples"] >= 1


def test_render_alert_shows_base_rate_not_llm_magnitude(monkeypatch):
    """Alert TG message must show 'n=X 中位 ...' and never an LLM-written %."""
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: [])
    impact = {
        "summary": "test",
        "impacts": [{
            "symbol": "VOO", "direction": "bearish", "confidence": 0.7,
            "reasoning": "trade tension",
            "base_rate": {
                "n_samples": 8,
                "fwd_5d_pct": {"median": -1.5, "min": -6.0, "max": 2.0, "n": 8},
                "fwd_20d_pct": {"median": -3.2, "min": -12.0, "max": 4.0, "n": 8},
                "max_dd_within_max_window_pct": {"median": -7.0, "min": -15.0,
                                                   "max": -3.0, "n": 8},
            },
        }],
        "secondary_assets": [],
        "action_hint": "观察",
    }
    out = newswatch.render_alert(_item(), _sev(8), impact)
    assert "5d 中位 -1.5%" in out
    assert "20d 中位 -3.2%" in out
    assert "n=8" in out
    # v4: must NOT show LLM-written confidence/direction either, only real base rate
    assert "置信" not in out
    assert "bearish" not in out


def test_render_alert_no_base_rate_shows_disclaimer(monkeypatch):
    """When n=0 historical samples, render must say so (not fake numbers)."""
    monkeypatch.setattr(newswatch.similar_event, "lookup_for_alert", lambda *a, **k: [])
    impact = {
        "summary": "novel event",
        "impacts": [{
            "symbol": "VOO", "direction": "bearish", "confidence": 0.5,
            "reasoning": "first of its kind",
            "base_rate": None,
        }],
        "secondary_assets": [], "action_hint": None,
    }
    out = newswatch.render_alert(_item(), _sev(7), impact)
    assert "无足够样本" in out


# ---- Cluster cooldown ----


def test_cluster_cooldown_finds_recent_pushed_similar(monkeypatch, tmp_path):
    """If a sim>=0.7 event was pushed within 24h, _find_recent_pushed_cluster returns it."""
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "cluster.sqlite")
    db.init()
    from datetime import datetime
    with db.conn() as c:
        c.execute("INSERT INTO events(severity, category, summary, fired_at, pushed_at) "
                   "VALUES (?, ?, ?, ?, ?)",
                   (8, "geopolitical", "Iran tension up",
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat()))

    monkeypatch.setattr(newswatch.similar_event, "find_similar",
                         lambda q, **kw: [{"event_id": 1, "similarity": 0.85,
                                            "summary": "Iran tension up"}])
    cluster = newswatch._find_recent_pushed_cluster(
        {"title": "Iran new escalation"}, {"reasoning": "geo"}
    )
    assert cluster is not None
    assert cluster["event_id"] == 1


def test_cluster_cooldown_ignores_low_similarity(monkeypatch, tmp_path):
    """sim < 0.7 should not trigger cooldown."""
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "cluster2.sqlite")
    db.init()
    from datetime import datetime
    with db.conn() as c:
        c.execute("INSERT INTO events(severity, category, summary, fired_at, pushed_at) "
                   "VALUES (8, 'macro', 'old', ?, ?)",
                   (datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))

    monkeypatch.setattr(newswatch.similar_event, "find_similar",
                         lambda q, **kw: [{"event_id": 1, "similarity": 0.55,
                                            "summary": "different"}])
    assert newswatch._find_recent_pushed_cluster(
        {"title": "Different event"}, {"reasoning": "x"}
    ) is None


def test_cluster_cooldown_ignores_old_pushed(monkeypatch, tmp_path):
    """Similar event but pushed > 24h ago: no cooldown."""
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "cluster3.sqlite")
    db.init()
    from datetime import datetime, timedelta
    old = (datetime.utcnow() - timedelta(hours=30)).isoformat()
    with db.conn() as c:
        c.execute("INSERT INTO events(severity, category, summary, fired_at, pushed_at) "
                   "VALUES (8, 'geopolitical', 'older', ?, ?)",
                   (old, old))

    monkeypatch.setattr(newswatch.similar_event, "find_similar",
                         lambda q, **kw: [{"event_id": 1, "similarity": 0.85,
                                            "summary": "older"}])
    assert newswatch._find_recent_pushed_cluster(
        {"title": "x"}, {"reasoning": "y"}
    ) is None


# ---- Heuristic (embedding-free) cluster dedup ----


def test_heuristic_cluster_catches_high_symbol_overlap(monkeypatch, tmp_path):
    """Same category + 50%+ symbol overlap within 4h window → dedup."""
    from quant import db
    from datetime import datetime
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "heur1.sqlite")
    db.init()
    now = datetime.utcnow().isoformat() + "Z"
    with db.conn() as c:
        cur = c.execute("INSERT INTO events(severity, category, summary, fired_at, "
                         "pushed_at, affected_symbols) VALUES (8, 'policy', 'earlier', ?, ?, ?)",
                         (now, now, "VOO,QQQ,AMD,NVDA"))
        prior_id = cur.lastrowid

    cluster = newswatch._find_recent_pushed_cluster_heuristic(
        severity_info={"category": "policy"},
        impact={"impacts": [{"symbol": "VOO"}, {"symbol": "QQQ"}, {"symbol": "AMD"}]},
        event_id=prior_id + 1,
    )
    assert cluster is not None
    assert cluster["event_id"] == prior_id
    assert cluster["via"] == "heuristic"


def test_heuristic_cluster_ignores_different_category(monkeypatch, tmp_path):
    """Different category should not dedup even with full symbol overlap."""
    from quant import db
    from datetime import datetime
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "heur2.sqlite")
    db.init()
    now = datetime.utcnow().isoformat() + "Z"
    with db.conn() as c:
        c.execute("INSERT INTO events(severity, category, summary, fired_at, "
                   "pushed_at, affected_symbols) VALUES (8, 'policy', 'p', ?, ?, ?)",
                   (now, now, "VOO,QQQ,AMD"))

    assert newswatch._find_recent_pushed_cluster_heuristic(
        severity_info={"category": "geopolitical"},
        impact={"impacts": [{"symbol": "VOO"}, {"symbol": "QQQ"}, {"symbol": "AMD"}]},
        event_id=99,
    ) is None


def test_heuristic_cluster_ignores_low_overlap(monkeypatch, tmp_path):
    """<50% overlap should not dedup."""
    from quant import db
    from datetime import datetime
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "heur3.sqlite")
    db.init()
    now = datetime.utcnow().isoformat() + "Z"
    with db.conn() as c:
        c.execute("INSERT INTO events(severity, category, summary, fired_at, "
                   "pushed_at, affected_symbols) VALUES (8, 'policy', 'p', ?, ?, ?)",
                   (now, now, "VOO,QQQ,AMD,NVDA,SOXX,ARM"))

    # impact has only 1 symbol matching 1 of 6 past → 1/6 = 0.17 overlap
    assert newswatch._find_recent_pushed_cluster_heuristic(
        severity_info={"category": "policy"},
        impact={"impacts": [{"symbol": "VOO"}]},
        event_id=99,
    ) is None


def test_heuristic_cluster_ignores_outside_window(monkeypatch, tmp_path):
    """Past event >4h ago should not trigger heuristic dedup."""
    from quant import db
    from datetime import datetime, timedelta
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "heur4.sqlite")
    db.init()
    old = (datetime.utcnow() - timedelta(hours=5)).isoformat() + "Z"
    with db.conn() as c:
        c.execute("INSERT INTO events(severity, category, summary, fired_at, "
                   "pushed_at, affected_symbols) VALUES (8, 'policy', 'p', ?, ?, ?)",
                   (old, old, "VOO,QQQ,AMD"))

    assert newswatch._find_recent_pushed_cluster_heuristic(
        severity_info={"category": "policy"},
        impact={"impacts": [{"symbol": "VOO"}, {"symbol": "QQQ"}, {"symbol": "AMD"}]},
        event_id=99,
    ) is None
