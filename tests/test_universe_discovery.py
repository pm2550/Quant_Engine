"""Tests for universe_discovery (Phase D, 2026-05-26)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from quant import universe_discovery


@contextmanager
def _temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "discovery_test.sqlite"
    monkeypatch.setattr(universe_discovery.db, "DB_PATH", db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                severity INTEGER NOT NULL,
                category TEXT,
                summary TEXT NOT NULL,
                impact_json TEXT,
                affected_symbols TEXT,
                fired_at TEXT NOT NULL
            );
            CREATE TABLE news_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                published_at TEXT,
                fetched_at TEXT NOT NULL
            );
            """
        )
    yield db_path


# ---- events source ----

def test_discover_from_events_filters_low_sev(monkeypatch, tmp_path):
    with _temp_db(monkeypatch, tmp_path) as p:
        now = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(p) as conn:
            conn.execute(
                "INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                "VALUES (?,?,?,?,?)",
                (8, "price_action", "ARM 大涨", "ARM", now))
            conn.execute(
                "INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                "VALUES (?,?,?,?,?)",
                (5, "macro", "小波动", "AMZN", now))  # sev 5 < min_sev=7 → excluded
        out = universe_discovery.discover_from_events(min_sev=7, days=7)
    syms = {c["symbol"] for c in out}
    assert "ARM" in syms
    assert "AMZN" not in syms


def test_discover_from_events_keeps_highest_sev(monkeypatch, tmp_path):
    """同一 symbol 多条事件, 保留 max severity."""
    with _temp_db(monkeypatch, tmp_path) as p:
        now = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(p) as conn:
            conn.execute("INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                         "VALUES (7, 'price_action', 'ARM 7', 'ARM', ?)", (now,))
            conn.execute("INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                         "VALUES (9, 'price_action', 'ARM 9', 'ARM', ?)", (now,))
        out = universe_discovery.discover_from_events(min_sev=7, days=7)
    arm = [c for c in out if c["symbol"] == "ARM"][0]
    assert arm["severity"] == 9
    assert "ARM 9" in arm["reason"]


def test_discover_from_events_splits_comma_separated(monkeypatch, tmp_path):
    with _temp_db(monkeypatch, tmp_path) as p:
        now = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(p) as conn:
            conn.execute("INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                         "VALUES (8, 'macro', '半导体板块', 'NVDA,AMD,AVGO', ?)", (now,))
        out = universe_discovery.discover_from_events(min_sev=7, days=7)
    assert {c["symbol"] for c in out} == {"NVDA", "AMD", "AVGO"}


def test_discover_from_events_filters_by_age(monkeypatch, tmp_path):
    """超过 days 窗口的 event 不算."""
    with _temp_db(monkeypatch, tmp_path) as p:
        old = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
        recent = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(p) as conn:
            conn.execute("INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                         "VALUES (8, 'x', 'old', 'OLD', ?)", (old,))
            conn.execute("INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                         "VALUES (8, 'x', 'new', 'NEW', ?)", (recent,))
        out = universe_discovery.discover_from_events(min_sev=7, days=7)
    syms = {c["symbol"] for c in out}
    assert "NEW" in syms and "OLD" not in syms


# ---- theme ETF source ----

class _FakeHoldings:
    def __init__(self, rows):
        self._df = pd.DataFrame(rows).set_index("Symbol")

    @property
    def empty(self):
        return self._df.empty

    def iterrows(self):
        return self._df.iterrows()


class _FakeFundsData:
    def __init__(self, rows):
        self.top_holdings = _FakeHoldings(rows)


class _FakeTicker:
    def __init__(self, rows=None):
        self.funds_data = _FakeFundsData(rows) if rows is not None else None


def test_discover_from_theme_etfs_returns_top_holdings(monkeypatch):
    import yfinance as yf
    fake_data = {
        "AIQ": [
            {"Symbol": "NVDA", "Name": "NVIDIA", "Holding Percent": 0.08},
            {"Symbol": "META", "Name": "Meta", "Holding Percent": 0.07},
        ],
    }
    monkeypatch.setattr(yf, "Ticker",
                        lambda s: _FakeTicker(rows=fake_data.get(s)))
    out = universe_discovery.discover_from_theme_etfs(
        etfs=[{"etf": "AIQ", "theme": "ai_software", "label": "AI"}],
        top_n=10,
    )
    syms = {c["symbol"] for c in out}
    assert "NVDA" in syms and "META" in syms
    nvda = next(c for c in out if c["symbol"] == "NVDA")
    assert nvda["holding_pct"] == 8.0
    assert nvda["source"] == "theme_etf"


def test_discover_from_theme_etfs_skips_no_data(monkeypatch):
    """ETF 没 funds_data 时不爆."""
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", lambda s: _FakeTicker(rows=None))
    out = universe_discovery.discover_from_theme_etfs(
        etfs=[{"etf": "AIQ", "theme": "x", "label": "AI"}],
    )
    assert out == []


# ---- sector rotation source ----

def test_discover_from_sector_etfs_filters_underperformers(monkeypatch):
    import yfinance as yf
    # SPY = 0%, XLK = +5% (hot), XLE = +1% (not hot enough vs 3% threshold)
    returns_map = {"SPY": 0.0, "XLK": 0.05, "XLE": 0.01}

    def fake_pct(sym, n):
        return returns_map.get(sym, 0)

    # Patch the inner closure by monkeypatching fetcher
    from quant import fetcher
    df_xlk = pd.DataFrame({"close": [100, 100, 100, 100, 100, 105]})
    df_xle = pd.DataFrame({"close": [100, 100, 100, 100, 100, 101]})
    df_spy = pd.DataFrame({"close": [100, 100, 100, 100, 100, 100]})

    def fake_load(sym):
        return {"XLK": df_xlk, "XLE": df_xle, "SPY": df_spy}.get(sym, pd.DataFrame())

    monkeypatch.setattr(fetcher, "load_local", fake_load)

    fake_holdings = {
        "XLK": [
            {"Symbol": "MSFT", "Name": "Microsoft", "Holding Percent": 0.12},
            {"Symbol": "AAPL", "Name": "Apple", "Holding Percent": 0.10},
        ],
    }
    monkeypatch.setattr(yf, "Ticker",
                        lambda s: _FakeTicker(rows=fake_holdings.get(s)))

    out = universe_discovery.discover_from_sector_etfs(
        sectors=[{"etf": "XLK", "sector": "tech"}, {"etf": "XLE", "sector": "energy"}],
        benchmark="SPY", lookback_days=5, outperform_pct=0.03,
        top_n_holdings=5,
    )
    syms = {c["symbol"] for c in out}
    # XLK hot → drill MSFT/AAPL; XLE not hot → no holdings
    assert "MSFT" in syms and "AAPL" in syms


# ---- news mention source ----

def test_discover_from_news_mentions_counts_tickers(monkeypatch, tmp_path):
    with _temp_db(monkeypatch, tmp_path) as p:
        now = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(p) as conn:
            for title in [
                "NVDA earnings beat estimates strongly",
                "NVDA up 5% after hours",
                "NVDA breaks all-time high",
                "AMD slips slightly",   # only 1 mention
                "Tesla TSLA delivery numbers",
            ]:
                conn.execute("INSERT INTO news_archive (url, title, source, published_at, fetched_at) VALUES (?,?,?,?,?)",
                             (f"http://test/{title[:30]}", title, "test", now, now))
        out = universe_discovery.discover_from_news_mentions(days=1, min_count=3)
    syms = {c["symbol"] for c in out}
    assert "NVDA" in syms  # 3 mentions
    assert "AMD" not in syms  # 1 mention
    assert "TSLA" not in syms  # 1 mention


def test_discover_from_news_mentions_excludes_non_tickers(monkeypatch, tmp_path):
    """随机大写字母不应该被当成 ticker."""
    with _temp_db(monkeypatch, tmp_path) as p:
        now = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(p) as conn:
            for title in [
                "USA NEWS BREAKING TODAY",  # USA / NEWS / BREAKING / TODAY 都不在 watch list
                "USA NEWS UPDATE",
                "USA NEWS REPORT",
            ]:
                conn.execute("INSERT INTO news_archive (url, title, source, published_at, fetched_at) VALUES (?,?,?,?,?)",
                             (f"http://test/{title[:30]}", title, "test", now, now))
        out = universe_discovery.discover_from_news_mentions(days=1, min_count=3)
    assert out == []  # none of these are in _TICKER_WATCH_LIST


# ---- aggregate ----

def test_aggregate_dedupes_across_sources(monkeypatch, tmp_path):
    """同一 symbol 来自 events + theme_etf → n_sources=2."""
    with _temp_db(monkeypatch, tmp_path) as p:
        now = datetime.utcnow().isoformat() + "Z"
        with sqlite3.connect(p) as conn:
            conn.execute("INSERT INTO events (severity, category, summary, affected_symbols, fired_at) "
                         "VALUES (8, 'price_action', 'NVDA hot', 'NVDA', ?)", (now,))

    monkeypatch.setattr(universe_discovery, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(universe_discovery, "_static_universe_symbols", lambda: set())

    # mock theme ETF to also surface NVDA
    monkeypatch.setattr(universe_discovery, "discover_from_theme_etfs",
                        lambda **kw: [{"symbol": "NVDA", "source": "theme_etf",
                                        "reason": "AIQ top holding 8%",
                                        "etf": "AIQ"}])
    monkeypatch.setattr(universe_discovery, "discover_from_sector_etfs",
                        lambda **kw: [])
    monkeypatch.setattr(universe_discovery, "discover_from_news_mentions",
                        lambda **kw: [])

    out = universe_discovery.aggregate_candidates()
    nvda = [c for c in out if c["symbol"] == "NVDA"][0]
    assert nvda["n_sources"] == 2
    assert set(nvda["sources"]) == {"events", "theme_etf"}


def test_aggregate_excludes_portfolio_and_static(monkeypatch, tmp_path):
    """已在 portfolio / static_universe 的标的应该被排除."""
    monkeypatch.setattr(universe_discovery, "_portfolio_symbols", lambda: {"AMD"})
    monkeypatch.setattr(universe_discovery, "_static_universe_symbols", lambda: {"META"})

    monkeypatch.setattr(universe_discovery, "discover_from_events",
                        lambda **kw: [
                            {"symbol": "AMD", "source": "events", "reason": "x"},
                            {"symbol": "META", "source": "events", "reason": "y"},
                            {"symbol": "NEW", "source": "events", "reason": "z"},
                        ])
    monkeypatch.setattr(universe_discovery, "discover_from_theme_etfs",
                        lambda **kw: [])
    monkeypatch.setattr(universe_discovery, "discover_from_sector_etfs",
                        lambda **kw: [])
    monkeypatch.setattr(universe_discovery, "discover_from_news_mentions",
                        lambda **kw: [])

    out = universe_discovery.aggregate_candidates()
    syms = {c["symbol"] for c in out}
    assert syms == {"NEW"}


def test_aggregate_caps_size(monkeypatch):
    monkeypatch.setattr(universe_discovery, "_portfolio_symbols", lambda: set())
    monkeypatch.setattr(universe_discovery, "_static_universe_symbols", lambda: set())
    monkeypatch.setattr(universe_discovery, "discover_from_events",
                        lambda **kw: [{"symbol": f"S{i}", "source": "events",
                                        "reason": "x"} for i in range(100)])
    monkeypatch.setattr(universe_discovery, "discover_from_theme_etfs",
                        lambda **kw: [])
    monkeypatch.setattr(universe_discovery, "discover_from_sector_etfs",
                        lambda **kw: [])
    monkeypatch.setattr(universe_discovery, "discover_from_news_mentions",
                        lambda **kw: [])
    out = universe_discovery.aggregate_candidates(max_size=20)
    assert len(out) == 20


def test_write_dynamic_universe_yaml_format(tmp_path):
    candidates = [
        {"symbol": "META", "sources": ["events", "theme_etf"], "n_sources": 2,
         "reasons": ["events sev 8: meta news", "AIQ top holding 7%"]},
        {"symbol": "GOOG", "sources": ["theme_etf"], "n_sources": 1,
         "reasons": ["AIQ top holding 6%"]},
    ]
    out_path = tmp_path / "dynamic.yaml"
    universe_discovery.write_dynamic_universe(candidates, path=out_path)
    data = yaml.safe_load(out_path.read_text())
    assert data["n_candidates"] == 2
    assert {e["symbol"] for e in data["universe"]} == {"META", "GOOG"}
    meta = next(e for e in data["universe"] if e["symbol"] == "META")
    assert meta["n_sources"] == 2


def test_load_dynamic_universe_handles_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(universe_discovery, "DYNAMIC_UNIVERSE_FILE", tmp_path / "nope.yaml")
    assert universe_discovery.load_dynamic_universe() == []
