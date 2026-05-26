"""Unit tests for tax.py — lot tracking, FIFO/LIFO/HIFO matching, wash sale, harvest.

The wash sale, long/short-term boundary, and partial-lot split logic are
the parts most likely to compute the wrong tax bill, so they get focused
tests with synthetic dates.
"""
from __future__ import annotations
from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def temp_db(monkeypatch):
    tmp = Path(tempfile.mkdtemp()) / "tax.sqlite"
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp)
    db.init()
    yield db
    if tmp.exists():
        tmp.unlink()


# ---- Lot creation ----


def test_open_lot_creates_row(temp_db):
    from quant import tax
    lot = tax.open_lot(symbol="AMD", shares=10, price=100, acquired_at="2024-01-01")
    assert lot["symbol"] == "AMD"
    assert lot["shares"] == 10
    assert lot["closed_at"] is None


def test_open_lot_rejects_negative_shares(temp_db):
    from quant import tax
    with pytest.raises(ValueError):
        tax.open_lot(symbol="X", shares=-1, price=100, acquired_at="2024-01-01")


def test_open_lot_rejects_zero_price(temp_db):
    from quant import tax
    with pytest.raises(ValueError):
        tax.open_lot(symbol="X", shares=1, price=0, acquired_at="2024-01-01")


# ---- FIFO matching + long-term boundary ----


def test_fifo_full_close_marks_long_term(temp_db):
    """Held > 365 days → is_long_term = True."""
    from quant import tax
    tax.open_lot(symbol="AMD", shares=10, price=100, acquired_at="2024-01-01")
    out = tax.sell(symbol="AMD", shares=10, price=200, sold_at="2025-06-01")
    assert out["total_realized_pnl"] == 1000.0
    assert out["long_term_pnl"] == 1000.0
    assert out["short_term_pnl"] == 0.0
    assert out["lots_closed"][0]["is_long_term"] is True
    assert out["lots_closed"][0]["holding_days"] > 365


def test_short_term_below_365_days(temp_db):
    from quant import tax
    tax.open_lot(symbol="AMD", shares=5, price=100, acquired_at="2025-01-01")
    out = tax.sell(symbol="AMD", shares=5, price=150, sold_at="2025-06-01")
    assert out["short_term_pnl"] == 250.0
    assert out["long_term_pnl"] == 0.0
    assert out["lots_closed"][0]["is_long_term"] is False


def test_fifo_split_across_long_and_short_term(temp_db):
    """Two lots, one LT one ST — sale should split realized P&L correctly."""
    from quant import tax
    tax.open_lot(symbol="AMD", shares=10, price=100, acquired_at="2024-01-01")  # LT
    tax.open_lot(symbol="AMD", shares=10, price=200, acquired_at="2025-04-01")  # ST
    out = tax.sell(symbol="AMD", shares=15, price=300, sold_at="2025-06-01")
    # FIFO: first close 10 LT @ $100 → +$2000 LT, then 5 ST @ $200 → +$500 ST
    assert out["long_term_pnl"] == 2000.0
    assert out["short_term_pnl"] == 500.0
    assert out["total_realized_pnl"] == 2500.0


def test_partial_close_creates_split_lot(temp_db):
    """Selling 3 of 10 shares should leave the open lot with 7 shares + create
    a closed sister-lot for the 3."""
    from quant import tax
    tax.open_lot(symbol="X", shares=10, price=100, acquired_at="2024-01-01")
    tax.sell(symbol="X", shares=3, price=200, sold_at="2025-06-01")
    open_lots = tax.list_lots(symbol="X", open_only=True)
    closed_lots = [l for l in tax.list_lots(symbol="X") if l["closed_at"]]
    assert len(open_lots) == 1
    assert open_lots[0]["shares"] == 7
    assert len(closed_lots) == 1
    assert closed_lots[0]["shares"] == 3


def test_sell_more_than_available_raises(temp_db):
    from quant import tax
    tax.open_lot(symbol="X", shares=5, price=100, acquired_at="2024-01-01")
    with pytest.raises(ValueError, match="insufficient"):
        tax.sell(symbol="X", shares=10, price=120, sold_at="2025-06-01")


# ---- LIFO / HIFO ----


def test_lifo_uses_most_recent_first(temp_db):
    from quant import tax
    tax.open_lot(symbol="X", shares=5, price=100, acquired_at="2024-01-01")  # older
    tax.open_lot(symbol="X", shares=5, price=200, acquired_at="2025-04-01")  # newer
    out = tax.sell(symbol="X", shares=5, price=300, sold_at="2025-06-01", method="LIFO")
    # LIFO closes the $200 lot first → +$500 ST
    assert out["short_term_pnl"] == 500.0
    assert out["long_term_pnl"] == 0.0


def test_hifo_minimizes_realized_gain(temp_db):
    """HIFO matches highest-cost lot first → smallest gain."""
    from quant import tax
    tax.open_lot(symbol="X", shares=5, price=100, acquired_at="2024-01-01")
    tax.open_lot(symbol="X", shares=5, price=250, acquired_at="2024-02-01")
    out = tax.sell(symbol="X", shares=5, price=300, sold_at="2024-08-01", method="HIFO")
    # $250 lot closed first → +$250 vs FIFO would be +$1000
    assert out["total_realized_pnl"] == 250.0


# ---- Wash sale ----


def test_wash_sale_detected_when_replacement_within_30_days(temp_db):
    """Sell at a loss + buy back within 30 days = wash sale → loss disallowed."""
    from quant import tax
    tax.open_lot(symbol="X", shares=10, price=100, acquired_at="2025-01-01")  # original
    tax.open_lot(symbol="X", shares=10, price=80,  acquired_at="2025-05-15")  # replacement
    out = tax.sell(symbol="X", shares=5, price=50, sold_at="2025-05-20")     # 5 days after
    assert len(out["wash_sale_warnings"]) == 1
    w = out["wash_sale_warnings"][0]
    assert w["disallowed_amount"] > 0
    assert w["loss"] < 0


def test_no_wash_sale_when_replacement_outside_window(temp_db):
    from quant import tax
    tax.open_lot(symbol="X", shares=10, price=100, acquired_at="2025-01-01")
    tax.open_lot(symbol="X", shares=10, price=80,  acquired_at="2025-01-15")  # >30d before sell
    out = tax.sell(symbol="X", shares=5, price=50, sold_at="2025-06-01")
    assert out["wash_sale_warnings"] == []


def test_no_wash_sale_on_gain(temp_db):
    """Wash sale rule applies only to losses — selling at a gain is exempt."""
    from quant import tax
    tax.open_lot(symbol="X", shares=10, price=100, acquired_at="2025-01-01")
    tax.open_lot(symbol="X", shares=10, price=120, acquired_at="2025-05-15")  # within 30d
    out = tax.sell(symbol="X", shares=5, price=200, sold_at="2025-06-01")    # gain
    assert out["wash_sale_warnings"] == []


# ---- Harvest candidates ----


def test_harvest_separates_loss_lt_and_approaching_lt(temp_db):
    from quant import tax
    # Open underwater lot
    tax.open_lot(symbol="LOSER", shares=10, price=100, acquired_at="2025-01-01")
    # Approaching long-term (held ~340 days as of 2026-05-06)
    tax.open_lot(symbol="ALMOST", shares=10, price=50, acquired_at="2025-06-01")
    # Long-term gain
    tax.open_lot(symbol="WINNER", shares=10, price=50, acquired_at="2024-01-01")

    prices = {"LOSER": 70, "ALMOST": 100, "WINNER": 200}
    out = tax.harvest_candidates(prices, today="2026-05-06")

    loss_syms = {x["symbol"] for x in out["harvest_candidates_loss"]}
    approaching_syms = {x["symbol"] for x in out["approaching_long_term"]}
    lt_gain_syms = {x["symbol"] for x in out["long_term_at_gain"]}

    assert "LOSER" in loss_syms
    # ALMOST: 2025-06-01 → 2026-05-06 = 339 days (< 365), days_to_lt = 26 (within 30d window)
    assert "ALMOST" in approaching_syms
    assert "WINNER" in lt_gain_syms


def test_harvest_totals_aggregate_correctly(temp_db):
    from quant import tax
    tax.open_lot(symbol="A", shares=10, price=50,  acquired_at="2024-01-01")
    tax.open_lot(symbol="B", shares=5,  price=100, acquired_at="2024-01-01")
    out = tax.harvest_candidates({"A": 60, "B": 80}, today="2026-05-06")
    # A: +$100, B: -$100 → total 0
    assert out["totals"]["unrealized_pnl"] == 0.0


def test_sell_with_no_open_lots_raises(temp_db):
    from quant import tax
    with pytest.raises(ValueError, match="insufficient"):
        tax.sell(symbol="GHOST", shares=1, price=100, sold_at="2026-01-01")
