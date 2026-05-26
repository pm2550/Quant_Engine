"""Unit tests for expectations.py — bootstrap empirical forward return distributions."""
from __future__ import annotations
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / "exp.sqlite"
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp)
    db.init()
    yield db
    if tmp.exists():
        tmp.unlink()


# ---- bootstrap_distribution: pure function, no DB ----


def test_bootstrap_returns_full_distribution_on_sufficient_data():
    from quant.expectations import bootstrap_distribution
    closes = pd.Series(np.linspace(100, 130, 300))  # +30% smooth uptrend
    out = bootstrap_distribution(closes, horizon_days=5, lookback_days=252)
    assert out is not None
    assert out["n_samples"] >= 30
    # Linear uptrend → all forward returns positive
    assert out["mean_pct"] > 0
    assert out["min_pct"] > 0
    # p5 < median < p95
    assert out["p5_pct"] <= out["median_pct"] <= out["p95_pct"]


def test_bootstrap_returns_none_for_insufficient_data():
    from quant.expectations import bootstrap_distribution
    closes = pd.Series([100, 101, 102, 103, 104])  # only 5 points
    assert bootstrap_distribution(closes, horizon_days=5) is None


def test_bootstrap_handles_volatile_series():
    from quant.expectations import bootstrap_distribution
    np.random.seed(42)
    rets = np.random.normal(0.001, 0.02, 300)
    prices = 100 * np.exp(np.cumsum(rets))
    out = bootstrap_distribution(pd.Series(prices), horizon_days=5)
    assert out is not None
    # Volatile series: sigma should be > 0
    assert out["sigma_pct"] > 0
    # 5-day return roughly sqrt(5) × daily sigma ≈ 4-5%
    assert 2 < out["sigma_pct"] < 10


def test_bootstrap_empty_series_returns_none():
    from quant.expectations import bootstrap_distribution
    assert bootstrap_distribution(pd.Series(dtype=float), horizon_days=5) is None


# ---- snapshot_symbol: writes to DB ----


def test_snapshot_symbol_writes_one_row_per_horizon(temp_db, monkeypatch):
    from quant import expectations as exp, fetcher
    closes = np.linspace(100, 130, 300)
    fake_df = pd.DataFrame({"close": closes},
                            index=pd.date_range("2025-01-01", periods=300, freq="B"))
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_df)

    res = exp.snapshot_symbol("FAKE", horizons=(1, 5, 20),
                                snapshot_date="2026-05-06")
    assert all(res[h] is not None for h in (1, 5, 20))

    with temp_db.conn() as c:
        rows = c.execute("SELECT * FROM expectations WHERE symbol='FAKE'").fetchall()
    assert len(rows) == 3
    horizons = sorted(r["horizon_days"] for r in rows)
    assert horizons == [1, 5, 20]


def test_snapshot_symbol_replace_on_same_day(temp_db, monkeypatch):
    """Same (date, symbol, horizon, model) twice should not duplicate."""
    from quant import expectations as exp, fetcher
    fake_df = pd.DataFrame({"close": np.linspace(100, 110, 300)},
                            index=pd.date_range("2025-01-01", periods=300, freq="B"))
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_df)

    exp.snapshot_symbol("X", horizons=(5,), snapshot_date="2026-05-06")
    exp.snapshot_symbol("X", horizons=(5,), snapshot_date="2026-05-06")
    with temp_db.conn() as c:
        n = c.execute("SELECT COUNT(*) FROM expectations WHERE symbol='X'").fetchone()[0]
    assert n == 1


def test_snapshot_symbol_skips_no_data(temp_db, monkeypatch):
    from quant import expectations as exp, fetcher
    monkeypatch.setattr(fetcher, "load_local", lambda s: pd.DataFrame())
    res = exp.snapshot_symbol("GHOST")
    assert all(v is None for v in res.values())
    with temp_db.conn() as c:
        n = c.execute("SELECT COUNT(*) FROM expectations").fetchone()[0]
    assert n == 0


# ---- get_latest / history ----


def test_get_latest_returns_most_recent(temp_db, monkeypatch):
    from quant import expectations as exp, fetcher
    fake_df = pd.DataFrame({"close": np.linspace(100, 110, 300)},
                            index=pd.date_range("2025-01-01", periods=300, freq="B"))
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_df)

    exp.snapshot_symbol("Y", horizons=(5,), snapshot_date="2026-05-04")
    exp.snapshot_symbol("Y", horizons=(5,), snapshot_date="2026-05-05")
    exp.snapshot_symbol("Y", horizons=(5,), snapshot_date="2026-05-06")
    latest = exp.get_latest("Y", horizon_days=5)
    assert latest is not None
    assert latest["snapshot_date"] == "2026-05-06"


def test_history_orders_descending(temp_db, monkeypatch):
    from quant import expectations as exp, fetcher
    fake_df = pd.DataFrame({"close": np.linspace(100, 110, 300)},
                            index=pd.date_range("2025-01-01", periods=300, freq="B"))
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_df)

    for d in ("2026-05-04", "2026-05-05", "2026-05-06"):
        exp.snapshot_symbol("Z", horizons=(5,), snapshot_date=d)
    hist = exp.history("Z", horizon_days=5)
    assert len(hist) == 3
    assert hist[0]["snapshot_date"] == "2026-05-06"
    assert hist[-1]["snapshot_date"] == "2026-05-04"


# ---- API endpoint smoke test ----


def test_api_expectations_endpoint(monkeypatch, tmp_path):
    """End-to-end via FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from quant import db, fetcher
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "api.sqlite")
    db.init()
    fake_df = pd.DataFrame({"close": np.linspace(100, 130, 300)},
                            index=pd.date_range("2025-01-01", periods=300, freq="B"))
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_df)

    from quant import expectations as exp
    exp.snapshot_symbol("FAKE", horizons=(5,), snapshot_date="2026-05-06")

    from quant.api_server import app
    client = TestClient(app)
    r = client.get("/api/expectations?symbol=FAKE&horizon_days=5")
    assert r.status_code == 200
    out = r.json()
    assert out["latest"]["symbol"] == "FAKE"
    assert out["latest"]["horizon_days"] == 5
    assert "p5_pct" in out["latest"]


def test_api_expectations_404_for_unknown_symbol(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "api2.sqlite")
    db.init()
    from quant.api_server import app
    client = TestClient(app)
    r = client.get("/api/expectations?symbol=GHOST&horizon_days=5")
    assert r.status_code == 404
