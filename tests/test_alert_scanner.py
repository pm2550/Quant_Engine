"""Unit tests for alert_scanner.py — rule evaluation, cooldown, cross_* state."""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def temp_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / "alerts.sqlite"
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp)
    db.init()
    yield db
    if tmp.exists():
        tmp.unlink()


def _insert_alert(db, **kw):
    defaults = dict(symbol="AMD", op="<=", value=380.0, basis="last",
                     enabled=1, cooldown_minutes=60, fired_count=0,
                     created_at=datetime.utcnow().isoformat())
    defaults.update(kw)
    cols = ", ".join(defaults)
    placeholders = ", ".join("?" for _ in defaults)
    with db.conn() as c:
        cur = c.execute(
            f"INSERT INTO user_alerts({cols}) VALUES ({placeholders})",
            tuple(defaults.values()),
        )
        return cur.lastrowid


# ---- Pure functions ----


def test_check_op_simple_threshold():
    from quant.alert_scanner import _check_op
    fires, desc = _check_op("<=", 380.0, 400.0, last_seen=None)
    assert fires is True
    assert "380" in desc and "400" in desc


def test_check_op_does_not_fire_when_above_threshold():
    from quant.alert_scanner import _check_op
    fires, _ = _check_op("<=", 410.0, 400.0, last_seen=None)
    assert fires is False


def test_check_op_cross_below_needs_transition():
    from quant.alert_scanner import _check_op
    # last_seen above, current below → cross
    fires, _ = _check_op("cross_below", 99.0, 100.0, last_seen=101.0)
    assert fires is True
    # last_seen also below → not a transition, no fire
    fires2, _ = _check_op("cross_below", 99.0, 100.0, last_seen=99.5)
    assert fires2 is False


def test_check_op_cross_above_needs_transition():
    from quant.alert_scanner import _check_op
    fires, _ = _check_op("cross_above", 105.0, 100.0, last_seen=98.0)
    assert fires is True


def test_cooldown_blocks_recent_fires():
    from quant.alert_scanner import _cooldown_ok
    recent = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
    old = (datetime.utcnow() - timedelta(minutes=120)).isoformat()
    assert _cooldown_ok(recent, cooldown_minutes=60) is False
    assert _cooldown_ok(old, cooldown_minutes=60) is True
    assert _cooldown_ok(None, cooldown_minutes=60) is True


# ---- _evaluate (basis extraction) ----


def test_evaluate_extracts_each_basis_from_signals():
    from quant.alert_scanner import _evaluate

    class FakeSig:
        price = 100.0
        rsi = 70.0
        ma20 = 95.0
        ma50 = 90.0
        ma200 = 85.0
        chg_1d_pct = 1.5
        chg_20d_pct = 12.5

    sig = FakeSig()
    assert _evaluate("last", sig, None) == 100.0
    assert _evaluate("last", sig, 105.0) == 105.0   # spot overrides close
    assert _evaluate("rsi", sig, None) == 70.0
    assert _evaluate("ma200", sig, None) == 85.0
    assert _evaluate("chg_1d_pct", sig, None) == 1.5
    assert _evaluate("chg_20d_pct", sig, None) == 12.5


def test_evaluate_handles_nan_basis():
    """NaN ma200 (not enough history) should return None instead of crashing."""
    from quant.alert_scanner import _evaluate

    class Sig:
        price = 100.0
        rsi = float("nan")
        ma20 = float("nan")
        ma50 = float("nan")
        ma200 = float("nan")
        chg_1d_pct = 0.0
        chg_20d_pct = 0.0

    assert _evaluate("rsi", Sig(), None) is None
    assert _evaluate("ma200", Sig(), None) is None


# ---- scan_once integration (mocked fetcher + signals) ----


def test_scan_once_fires_alert_when_threshold_met(temp_db, monkeypatch):
    from quant import alert_scanner, fetcher
    import pandas as pd

    aid = _insert_alert(temp_db, symbol="AMD", op="<=", value=380.0, basis="last")

    fake_df = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2025-01-01", periods=1))
    monkeypatch.setattr(fetcher, "load_local", lambda s: fake_df)

    class Sig:
        price = 350.0
        rsi = 50.0
        ma20 = ma50 = ma200 = 350.0
        chg_1d_pct = 0
        chg_20d_pct = 0
        signal_codes = []
    monkeypatch.setattr(alert_scanner.signals, "compute", lambda *a, **k: Sig())
    monkeypatch.setattr(alert_scanner, "_spot_for", lambda s: 350.0)

    sent = []
    monkeypatch.setattr(alert_scanner.telegram, "send", lambda m: sent.append(m))
    monkeypatch.setattr(alert_scanner.cfg_mod, "load",
                         lambda name: {} if name == "strategies" else {})

    out = alert_scanner.scan_once(dry_run=False)
    assert out["fired"] == 1
    assert len(sent) == 1
    assert "AMD" in sent[0]
    # fired_at + fired_count updated
    with temp_db.conn() as c:
        row = c.execute("SELECT * FROM user_alerts WHERE id=?", (aid,)).fetchone()
    assert row["fired_count"] == 1
    assert row["fired_at"] is not None


def test_scan_once_respects_cooldown(temp_db, monkeypatch):
    from quant import alert_scanner, fetcher
    import pandas as pd

    recent = (datetime.utcnow() - timedelta(minutes=10)).isoformat()
    _insert_alert(temp_db, symbol="AMD", op="<=", value=380.0,
                   cooldown_minutes=60, fired_at=recent, fired_count=1)

    monkeypatch.setattr(fetcher, "load_local",
                         lambda s: pd.DataFrame({"close": [100.0]},
                                                  index=pd.date_range("2025-01-01", periods=1)))

    class Sig:
        price = 350.0
        rsi = 50.0; ma20 = ma50 = ma200 = 350.0
        chg_1d_pct = chg_20d_pct = 0
        signal_codes = []
    monkeypatch.setattr(alert_scanner.signals, "compute", lambda *a, **k: Sig())
    monkeypatch.setattr(alert_scanner, "_spot_for", lambda s: 350.0)
    monkeypatch.setattr(alert_scanner.cfg_mod, "load", lambda name: {})

    out = alert_scanner.scan_once(dry_run=False)
    assert out["fired"] == 0
    assert out["skipped_cooldown"] == 1


def test_scan_once_dry_run_does_not_send_or_update(temp_db, monkeypatch):
    from quant import alert_scanner, fetcher
    import pandas as pd

    aid = _insert_alert(temp_db, symbol="AMD", op="<=", value=380.0)

    monkeypatch.setattr(fetcher, "load_local",
                         lambda s: pd.DataFrame({"close": [100.0]},
                                                  index=pd.date_range("2025-01-01", periods=1)))

    class Sig:
        price = 350.0
        rsi = 50.0; ma20 = ma50 = ma200 = 350.0
        chg_1d_pct = chg_20d_pct = 0
        signal_codes = []
    monkeypatch.setattr(alert_scanner.signals, "compute", lambda *a, **k: Sig())
    monkeypatch.setattr(alert_scanner, "_spot_for", lambda s: 350.0)
    monkeypatch.setattr(alert_scanner.cfg_mod, "load", lambda name: {})

    sent = []
    monkeypatch.setattr(alert_scanner.telegram, "send", lambda m: sent.append(m))

    out = alert_scanner.scan_once(dry_run=True)
    assert out["fired"] == 1
    assert sent == []
    # fired_at NOT updated in dry-run
    with temp_db.conn() as c:
        row = c.execute("SELECT * FROM user_alerts WHERE id=?", (aid,)).fetchone()
    assert row["fired_at"] is None
