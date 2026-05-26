"""Unit tests for api_server endpoints — focus on the new P0/P1 surface.

Uses FastAPI TestClient so no live uvicorn needed. yfinance / akshare are
mocked since these tests must work offline. portfolio.yaml is the real one;
test isolation comes from monkey-patching fetcher.load_local for synthetic
prices when needed.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from quant.api_server import app
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_analyze_recommendation_uses_multi_factor_decision(monkeypatch):
    """A raw bearish technical signal should not reduce when multi-factor is neutral."""
    from quant import api_server as api
    from quant.signals import SymbolSignals

    fake_portfolio = {
        "positions": {"GRID": {"shares": 1.0, "currency": "USD"}},
        "watchlist": [],
        "default_currency": "USD",
        "sector_overlap": {},
    }
    fake_df = pd.DataFrame(
        {"close": [100.0], "open": [100.0], "high": [101.0], "low": [99.0], "volume": [1000]},
        index=pd.date_range("2026-05-15", periods=1),
    )
    fake_sig = SymbolSignals(
        symbol="GRID", last_date="2026-05-15", price=100.0,
        chg_1d_pct=-3.0, chg_5d_pct=0.0, chg_20d_pct=0.0,
        rsi=50.0, rsi_zscore_252d=0.0,
        macd=0.0, macd_signal=0.0, macd_hist=0.0,
        bb_upper=110.0, bb_lower=90.0, bb_pct=0.5,
        ma20=100.0, ma50=100.0, ma200=100.0,
        above_ma50=True, above_ma200=True,
        atr_14=2.0, atr_pct=2.0,
        vol_20d_avg=1000.0, vol_today=1000.0, vol_ratio=1.0,
        signal_codes=["MACD_DEATH_CROSS_ABOVE_ZERO"],
    )
    neutral_multi = {
        "composite_score": 0.0,
        "action": "HOLD",
        "rationale": "neutral test score",
        "catalyst_imminent": False,
        "factor_breakdown": {},
    }

    monkeypatch.setattr(api.cfg_mod, "load", lambda name: fake_portfolio if name == "portfolio" else {"indicators": {}})
    monkeypatch.setattr(api, "_refresh", lambda sym: fake_df)
    monkeypatch.setattr(api.fetcher, "load_local", lambda sym: fake_df)
    monkeypatch.setattr(api.fetcher, "is_a_share", lambda sym: False)
    monkeypatch.setattr(api.fetcher, "latest_spot", lambda sym, include_post_market=True: {
        "price": 100.0, "as_of_utc": "2026-05-15T20:00:00Z", "session": "closed", "source": "test",
    })
    monkeypatch.setattr(api.fetcher, "staleness_seconds", lambda as_of: 0)
    monkeypatch.setattr(api.signals, "compute", lambda sym, df, cfg: fake_sig)
    monkeypatch.setattr(api.fundamentals, "latest", lambda sym: {})
    monkeypatch.setattr(api.multi_factor, "score", lambda sym, signals_dict, fund_data, cur_price: neutral_multi)
    monkeypatch.setattr(api, "_backtest_top", lambda sym: [])

    out = api.analyze(api.AnalyzeRequest(symbol="GRID"))
    assert out["multi_factor"]["composite_score"] == 0.0
    assert out["recommendation"]["action"] == "HOLD"
    assert out["recommendation"]["reason_codes"] == []


def test_refresh_falls_back_to_local_on_upstream_failure(monkeypatch):
    """Upstream fetch failure (akshare/yfinance temporary) must not 503 —
    fall back to local parquet so callers see cached data, not 'engine down'."""
    import pandas as pd
    from quant import api_server as api, fetcher

    def boom(sym):
        raise ConnectionError("Remote end closed connection")
    fake_local = pd.DataFrame({"close": [16.37], "open": [16.0], "high": [16.5],
                                 "low": [15.9], "volume": [1000]},
                                index=pd.date_range("2026-05-06", periods=1))
    monkeypatch.setattr(fetcher, "fetch_symbol", boom)
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_local)

    df = api._refresh("002624.SZ")
    assert not df.empty
    assert float(df["close"].iloc[-1]) == 16.37


def test_refresh_503_only_when_no_local_either(monkeypatch):
    """If both upstream AND local are dead, 503 with helpful message."""
    import pandas as pd
    from quant import api_server as api, fetcher
    from fastapi import HTTPException

    monkeypatch.setattr(fetcher, "fetch_symbol",
                         lambda s: (_ for _ in ()).throw(ConnectionError("upstream down")))
    monkeypatch.setattr(fetcher, "load_local", lambda s: pd.DataFrame())

    with pytest.raises(HTTPException) as ei:
        api._refresh("GHOST")
    assert ei.value.status_code == 503
    assert "no local data" in ei.value.detail or "upstream" in str(ei.value.detail).lower()


# ---- /api/whatif ----


def test_whatif_buy_increases_weight(client, monkeypatch):
    """Buying 1 more share should shift weights toward that symbol."""
    from quant import api_server as api
    # Inject a tiny synthetic portfolio
    fake_portfolio = {
        "positions": {
            "VOO":  {"shares": 1, "currency": "USD", "name": "VOO",  "theme": "broad_market"},
            "SOXX": {"shares": 1, "currency": "USD", "name": "SOXX", "theme": "ai_compute"},
        },
        "watchlist": [],
        "default_currency": "USD",
    }
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: fake_portfolio if name == "portfolio" else {})
    fake_df = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2025-01-01", periods=1))
    monkeypatch.setattr(api.fetcher, "load_local", lambda s: fake_df)
    monkeypatch.setattr(api.fetcher, "is_a_share", lambda s: False)

    r = client.post("/api/whatif", json={"trades": [
        {"symbol": "VOO", "action": "buy", "shares": 1}
    ]})
    assert r.status_code == 200
    out = r.json()
    assert out["before"]["weights"]["VOO"] < out["after"]["weights"]["VOO"]
    assert out["cash_impact"]["USD"] == -100.0


def test_whatif_sell_more_than_held_returns_400(client, monkeypatch):
    from quant import api_server as api
    fake_portfolio = {
        "positions": {"VOO": {"shares": 1, "currency": "USD", "name": "VOO"}},
        "watchlist": [], "default_currency": "USD",
    }
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: fake_portfolio if name == "portfolio" else {})
    monkeypatch.setattr(api.fetcher, "load_local",
                         lambda s: pd.DataFrame({"close": [100.0]},
                                                  index=pd.date_range("2025-01-01", periods=1)))
    monkeypatch.setattr(api.fetcher, "is_a_share", lambda s: False)

    r = client.post("/api/whatif", json={"trades": [
        {"symbol": "VOO", "action": "sell", "shares": 5}
    ]})
    assert r.status_code == 400
    assert "selling more" in r.text


def test_whatif_empty_trades_400(client):
    r = client.post("/api/whatif", json={"trades": []})
    assert r.status_code == 400


def test_whatif_invalid_action_400(client, monkeypatch):
    from quant import api_server as api
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: {"positions": {}, "watchlist": []}
                         if name == "portfolio" else {})
    r = client.post("/api/whatif", json={"trades": [
        {"symbol": "X", "action": "transfer", "shares": 1}
    ]})
    assert r.status_code in (400, 404)


def test_whatif_theme_exposure_shifts(client, monkeypatch):
    """Buying ai_compute symbol should increase that theme's weight."""
    from quant import api_server as api
    fake_portfolio = {
        "positions": {
            "VOO":  {"shares": 1, "currency": "USD", "theme": "broad_market"},
            "SOXX": {"shares": 1, "currency": "USD", "theme": "ai_compute"},
        },
        "watchlist": [], "default_currency": "USD",
    }
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: fake_portfolio if name == "portfolio" else {})
    monkeypatch.setattr(api.fetcher, "load_local",
                         lambda s: pd.DataFrame({"close": [100.0]},
                                                  index=pd.date_range("2025-01-01", periods=1)))
    monkeypatch.setattr(api.fetcher, "is_a_share", lambda s: False)

    r = client.post("/api/whatif", json={"trades": [
        {"symbol": "SOXX", "action": "buy", "shares": 2}
    ]})
    assert r.status_code == 200
    out = r.json()
    assert out["delta"]["theme_exposure"]["ai_compute"] > 0
    assert out["delta"]["theme_exposure"]["broad_market"] < 0


# ---- /api/scenario ----


def test_scenario_theme_shock_applies_to_all_matching(client, monkeypatch):
    from quant import api_server as api
    fake_portfolio = {
        "positions": {
            "SOXX": {"shares": 1, "currency": "USD", "theme": "ai_compute"},
            "AMD":  {"shares": 1, "currency": "USD", "theme": "ai_compute"},
            "VOO":  {"shares": 1, "currency": "USD", "theme": "broad_market"},
        },
        "watchlist": [], "default_currency": "USD",
    }
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: fake_portfolio if name == "portfolio" else {})
    monkeypatch.setattr(api.fetcher, "load_local",
                         lambda s: pd.DataFrame({"close": [100.0]},
                                                  index=pd.date_range("2025-01-01", periods=1)))

    r = client.post("/api/scenario", json={
        "name": "ai_compute -50%",
        "shocks": [{"theme": "ai_compute", "shock_pct": -0.5}],
    })
    assert r.status_code == 200
    out = r.json()
    affected = {row["symbol"] for row in out["affected_positions"]}
    assert "SOXX" in affected and "AMD" in affected
    assert "VOO" not in affected
    # P&L: 2 holdings × $100 × -0.5 = -$100
    assert out["summary_by_currency"]["USD"]["total_pnl"] == -100.0


def test_scenario_symbol_shock_targets_one(client, monkeypatch):
    from quant import api_server as api
    fake_portfolio = {
        "positions": {
            "SOXX": {"shares": 1, "currency": "USD", "theme": "ai_compute"},
            "VOO":  {"shares": 1, "currency": "USD", "theme": "broad_market"},
        },
        "watchlist": [], "default_currency": "USD",
    }
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: fake_portfolio if name == "portfolio" else {})
    monkeypatch.setattr(api.fetcher, "load_local",
                         lambda s: pd.DataFrame({"close": [100.0]},
                                                  index=pd.date_range("2025-01-01", periods=1)))

    r = client.post("/api/scenario", json={
        "shocks": [{"symbol": "SOXX", "shock_pct": -0.30}],
    })
    assert r.status_code == 200
    affected = r.json()["affected_positions"]
    assert len(affected) == 1
    assert affected[0]["symbol"] == "SOXX"


def test_scenario_empty_shocks_400(client):
    r = client.post("/api/scenario", json={"shocks": []})
    assert r.status_code == 400


def test_scenario_shock_without_symbol_or_theme_400(client, monkeypatch):
    from quant import api_server as api
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: {"positions": {}, "watchlist": []}
                         if name == "portfolio" else {})
    monkeypatch.setattr(api.fetcher, "load_local",
                         lambda s: pd.DataFrame({"close": [100.0]},
                                                  index=pd.date_range("2025-01-01", periods=1)))
    r = client.post("/api/scenario", json={"shocks": [{"shock_pct": -0.1}]})
    assert r.status_code == 400


# ---- /api/intraday ----


def test_intraday_rejects_invalid_interval(client):
    r = client.get("/api/intraday?symbol=AMD&interval=2h")
    assert r.status_code == 400


def test_intraday_rejects_invalid_date(client):
    r = client.get("/api/intraday?symbol=AMD&date=not-a-date")
    assert r.status_code == 400


# ---- /api/backtest ----


def test_backtest_query_filter_min_sharpe_excludes_null(client, monkeypatch):
    """min_sharpe filter must not let NULL-sharpe rows through."""
    from quant import api_server as api, db
    # Use a temp DB so we don't touch the real one
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    monkeypatch.setattr(db, "DB_PATH", pd.Path(tmp.name) if hasattr(pd, "Path") else __import__("pathlib").Path(tmp.name))
    db.init()
    with db.conn() as c:
        c.execute("INSERT INTO backtest_tasks(strategy, symbol, params_json, period_years, status, created_at) "
                  "VALUES ('dual_ma','X','{}',5,'done','2026-01-01T00:00:00')")
        tid = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        # row with sharpe=NULL
        c.execute("INSERT INTO backtest_results(task_id, sharpe, finished_at) VALUES (?, NULL, '2026-01-01')",
                  (tid,))
        c.execute("INSERT INTO backtest_tasks(strategy, symbol, params_json, period_years, status, created_at) "
                  "VALUES ('dual_ma','Y','{}',5,'done','2026-01-01T00:00:00')")
        tid2 = c.execute("SELECT last_insert_rowid()").fetchone()[0]
        c.execute("INSERT INTO backtest_results(task_id, sharpe, finished_at) VALUES (?, 1.5, '2026-01-01')",
                  (tid2,))

    r = client.get("/api/backtest?min_sharpe=1.0&limit=10")
    assert r.status_code == 200
    out = r.json()
    sharpes = [row["sharpe"] for row in out["results"]]
    assert None not in sharpes
    assert all(s >= 1.0 for s in sharpes)


def test_backtest_run_unknown_strategy_400(client):
    r = client.post("/api/backtest/run", json={
        "strategy": "no_such_strategy", "symbol": "X",
        "params": {}, "period_years": 1,
    })
    assert r.status_code == 400


# ---- /api/alerts ----


def test_alerts_create_invalid_op_400(client):
    r = client.post("/api/alerts", json={
        "symbol": "AMD", "op": "explodes", "value": 1, "basis": "last"
    })
    assert r.status_code == 400


def test_alerts_create_threshold_op_without_value_400(client):
    r = client.post("/api/alerts", json={
        "symbol": "AMD", "op": "<=", "value": None, "basis": "last"
    })
    assert r.status_code == 400


def test_alerts_create_invalid_basis_400(client):
    r = client.post("/api/alerts", json={
        "symbol": "AMD", "op": "<=", "value": 100, "basis": "imaginary_metric"
    })
    assert r.status_code == 400


def test_alerts_full_lifecycle(client, monkeypatch, tmp_path):
    """Create → list → toggle → delete."""
    from quant import db
    tmp = tmp_path / "alerts.sqlite"
    monkeypatch.setattr(db, "DB_PATH", tmp)
    db.init()

    r = client.post("/api/alerts", json={
        "symbol": "AMD", "op": "<=", "value": 380, "basis": "last", "note": "buy zone"
    })
    assert r.status_code == 200
    alert_id = r.json()["alert"]["id"]

    r = client.get("/api/alerts")
    assert r.status_code == 200
    assert r.json()["n_alerts"] >= 1

    r = client.patch(f"/api/alerts/{alert_id}?enabled=false")
    assert r.status_code == 200
    assert r.json()["alert"]["enabled"] == 0

    r = client.delete(f"/api/alerts/{alert_id}")
    assert r.status_code == 200
    assert r.json()["deleted_id"] == alert_id

    r = client.delete(f"/api/alerts/{alert_id}")  # already gone
    assert r.status_code == 404


# ---- /api/tax ----


def test_tax_lot_open_returns_lot(client, monkeypatch, tmp_path):
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "tax.sqlite")
    db.init()
    r = client.post("/api/tax/lot", json={
        "symbol": "AMD", "shares": 5, "price": 200, "acquired_at": "2024-06-01"
    })
    assert r.status_code == 200
    assert r.json()["lot"]["shares"] == 5


def test_tax_sell_realized_pnl_split_lt_st(client, monkeypatch, tmp_path):
    """Two lots (one LT, one ST) — sell mixes both, response shows split."""
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "tax.sqlite")
    db.init()
    client.post("/api/tax/lot", json={
        "symbol": "X", "shares": 10, "price": 100, "acquired_at": "2024-01-01"
    })
    client.post("/api/tax/lot", json={
        "symbol": "X", "shares": 10, "price": 200, "acquired_at": "2025-04-01"
    })
    r = client.post("/api/tax/sell", json={
        "symbol": "X", "shares": 15, "price": 300, "sold_at": "2025-06-01"
    })
    assert r.status_code == 200
    out = r.json()
    # FIFO: 10 LT @ +$2000, 5 ST @ +$500
    assert out["long_term_pnl"] == 2000
    assert out["short_term_pnl"] == 500


def test_tax_sell_insufficient_shares_400(client, monkeypatch, tmp_path):
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "tax.sqlite")
    db.init()
    r = client.post("/api/tax/sell", json={
        "symbol": "GHOST", "shares": 1, "price": 100, "sold_at": "2026-01-01"
    })
    assert r.status_code == 400


# ---- /api/audit ----


def test_audit_endpoint_returns_summary_shape(client, monkeypatch, tmp_path):
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "audit.sqlite")
    db.init()
    db.log_llm_call(task="t", backend="b:m", success=True, wall_time_s=1.0,
                     tokens_in=100, tokens_out=200, cost_usd=0.0)
    r = client.get("/api/audit?days=1")
    assert r.status_code == 200
    body = r.json()
    assert "total_calls" in body
    assert "by_backend" in body
    assert "recent_errors" in body


# ---- /api/events ----


def test_events_endpoint_filter_by_symbol(client, monkeypatch, tmp_path):
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "events.sqlite")
    db.init()
    with db.conn() as c:
        c.execute("INSERT INTO events(severity, category, summary, "
                   "  affected_symbols, fired_at) VALUES "
                   "(8, 'macro', 'Fed cuts', 'VOO,SOXX', datetime('now'))")
        c.execute("INSERT INTO events(severity, category, summary, "
                   "  affected_symbols, fired_at) VALUES "
                   "(7, 'company', 'Apple news', 'AAPL', datetime('now'))")

    r = client.get("/api/events?symbol=VOO&min_severity=4")
    assert r.status_code == 200
    out = r.json()
    syms = [e["affected"] for e in out["events"]]
    assert any("VOO" in s for s in syms)


# ---- /api/pnl ----


def test_pnl_arbitrary_window(client, monkeypatch):
    """P&L over a window should reflect price movement × shares."""
    import pandas as pd
    from quant import api_server as api
    monkeypatch.setattr(api.cfg_mod, "load",
                         lambda name: ({"positions": {
                             "X": {"shares": 10, "currency": "USD", "theme": "test"}
                         }, "watchlist": []} if name == "portfolio" else {}))
    fake_df = pd.DataFrame(
        {"close": [100.0, 110.0, 120.0]},
        index=pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
    )
    monkeypatch.setattr(api.fetcher, "load_local", lambda s: fake_df)
    monkeypatch.setattr(api.fetcher, "is_a_share", lambda s: False)

    r = client.get("/api/pnl?start=2025-01-01&end=2025-03-01&groupby=symbol")
    assert r.status_code == 200
    out = r.json()
    # 10 shares × ($120 - $100) = $200
    assert out["details"][0]["pnl"] == 200.0
    assert out["details"][0]["return_pct"] == 20.0


# ---- /api/history/move ----


def test_history_move_no_data_returns_404(client, monkeypatch):
    import pandas as pd
    from quant import api_server as api
    monkeypatch.setattr(api.fetcher, "load_local", lambda s: pd.DataFrame())
    r = client.get("/api/history/move?symbol=NOPE")
    assert r.status_code == 404
