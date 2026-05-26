"""Tax lot accounting: cost-basis tracking, FIFO matching, harvest, wash sale.

Why this matters at $1.5k account size: even small accounts can save 5-15%
on net returns through:
  - Long-term vs short-term holding (US: <$40k income → 0% LTCG, vs 24%+ STCG)
  - Tax-loss harvesting in December (offset realized gains)
  - Avoiding wash sales that disallow losses you thought you'd realized

US wash sale rule: if you sell at a loss and buy "substantially identical"
within 30 days (before or after), the loss is disallowed for tax purposes
and added to the cost basis of the replacement lot.

A-share (CN): no individual capital gains tax on listed stock trades, but
we still track lots for P&L attribution — keeps the data model uniform.
"""
from __future__ import annotations
import logging
import sqlite3
from datetime import datetime, timedelta

from . import db

log = logging.getLogger(__name__)

WASH_SALE_WINDOW_DAYS = 30
LONG_TERM_THRESHOLD_DAYS = 365


def open_lot(*, symbol: str, shares: float, price: float, acquired_at: str,
              currency: str = "USD", notes: str | None = None) -> dict:
    """Record a buy. Returns the new lot row."""
    if shares <= 0:
        raise ValueError("shares must be positive")
    if price <= 0:
        raise ValueError("price must be positive")
    with db.conn() as c:
        cur = c.execute(
            "INSERT INTO tax_lots(symbol, currency, shares, cost_basis_per_share, "
            "                     acquired_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (symbol.upper(), currency, shares, price, acquired_at, notes),
        )
        row = c.execute("SELECT * FROM tax_lots WHERE id=?",
                          (cur.lastrowid,)).fetchone()
    return dict(row)


def _open_lots_fifo(c: sqlite3.Connection, symbol: str) -> list[dict]:
    """Return open lots for symbol, oldest first (FIFO)."""
    rows = c.execute(
        "SELECT * FROM tax_lots WHERE symbol=? AND closed_at IS NULL "
        "ORDER BY acquired_at ASC, id ASC",
        (symbol.upper(),),
    ).fetchall()
    return [dict(r) for r in rows]


def _check_wash_sale(c: sqlite3.Connection, symbol: str, sell_date: str,
                       loss_per_share: float) -> tuple[float, list[int]]:
    """Detect wash sale: replacement buy within ±30 days of a loss sale.

    Returns (disallowed_amount_per_share, matched_replacement_ids).
    Caller decides how much of `shares × loss_per_share` to disallow based on
    overlap of replacement-lot shares.
    """
    if loss_per_share >= 0:
        return 0.0, []
    sell_dt = datetime.fromisoformat(sell_date)
    window_start = (sell_dt - timedelta(days=WASH_SALE_WINDOW_DAYS)).strftime("%Y-%m-%d")
    window_end = (sell_dt + timedelta(days=WASH_SALE_WINDOW_DAYS)).strftime("%Y-%m-%d")
    rows = c.execute(
        "SELECT id, acquired_at, shares FROM tax_lots "
        "WHERE symbol=? AND acquired_at >= ? AND acquired_at <= ? "
        "  AND acquired_at != ? "  # exclude the lot being sold itself
        "ORDER BY acquired_at",
        (symbol.upper(), window_start, window_end, sell_date),
    ).fetchall()
    matched = [r["id"] for r in rows]
    return abs(loss_per_share) if matched else 0.0, matched


def sell(*, symbol: str, shares: float, price: float, sold_at: str,
          method: str = "FIFO") -> dict:
    """Match a sale against open lots; record realized P&L per matched lot.

    Returns: {
        total_proceeds, total_cost_basis, total_realized_pnl,
        long_term_pnl, short_term_pnl, lots_closed: [...],
        wash_sale_warnings: [...],
    }

    method: "FIFO" (default) | "LIFO" | "HIFO" (highest-cost-first, minimizes gain)
    """
    if shares <= 0:
        raise ValueError("shares must be positive")
    if method not in ("FIFO", "LIFO", "HIFO"):
        raise ValueError(f"method must be FIFO/LIFO/HIFO, got {method!r}")

    with db.conn() as c:
        # Pre-flight check (no txn yet — pure read, raises before BEGIN)
        open_lots = _open_lots_fifo(c, symbol)
        if method == "LIFO":
            open_lots.reverse()
        elif method == "HIFO":
            open_lots.sort(key=lambda l: -l["cost_basis_per_share"])
        available = sum(l["shares"] for l in open_lots)
        if available + 1e-9 < shares:
            raise ValueError(
                f"insufficient open shares for {symbol}: "
                f"requested {shares}, available {available}"
            )

        c.execute("BEGIN IMMEDIATE")
        try:

            remaining = shares
            closed = []
            total_proceeds = 0.0
            total_cost = 0.0
            total_pnl = 0.0
            long_term_pnl = 0.0
            short_term_pnl = 0.0
            wash_warnings = []

            for lot in open_lots:
                if remaining <= 1e-9:
                    break
                take = min(lot["shares"], remaining)
                proceeds = take * price
                cost = take * lot["cost_basis_per_share"]
                pnl = proceeds - cost
                holding_days = (datetime.fromisoformat(sold_at)
                                  - datetime.fromisoformat(lot["acquired_at"])).days
                is_lt = 1 if holding_days > LONG_TERM_THRESHOLD_DAYS else 0

                # Wash sale check (US only — but record advisory for any currency)
                wash_amt, replacement_ids = _check_wash_sale(
                    c, symbol, sold_at, (price - lot["cost_basis_per_share"])
                )
                wash_disallowed = wash_amt * take if wash_amt > 0 else None
                if wash_disallowed:
                    wash_warnings.append({
                        "lot_id": lot["id"],
                        "shares_sold": take,
                        "loss": round(pnl, 2),
                        "disallowed_amount": round(wash_disallowed, 2),
                        "replacement_lot_ids": replacement_ids,
                        "rule": "US wash sale: replacement buy within ±30 days",
                    })

                if take >= lot["shares"] - 1e-9:
                    # Full close
                    c.execute(
                        "UPDATE tax_lots SET closed_at=?, proceeds_per_share=?, "
                        "  realized_pnl=?, holding_days=?, is_long_term=?, "
                        "  wash_sale_disallowed=?, matched_against=? WHERE id=?",
                        (sold_at, price, round(pnl, 4), holding_days, is_lt,
                         round(wash_disallowed, 4) if wash_disallowed else None,
                         ",".join(str(i) for i in replacement_ids) if replacement_ids else None,
                         lot["id"]),
                    )
                    closed_id = lot["id"]
                else:
                    # Partial: shrink the original lot, create a closed sister-lot
                    new_open_shares = lot["shares"] - take
                    c.execute("UPDATE tax_lots SET shares=? WHERE id=?",
                              (new_open_shares, lot["id"]))
                    cur = c.execute(
                        "INSERT INTO tax_lots(symbol, currency, shares, "
                        "  cost_basis_per_share, acquired_at, closed_at, "
                        "  proceeds_per_share, realized_pnl, holding_days, "
                        "  is_long_term, wash_sale_disallowed, matched_against, notes) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (lot["symbol"], lot["currency"], take,
                         lot["cost_basis_per_share"], lot["acquired_at"],
                         sold_at, price, round(pnl, 4), holding_days, is_lt,
                         round(wash_disallowed, 4) if wash_disallowed else None,
                         ",".join(str(i) for i in replacement_ids) if replacement_ids else None,
                         f"partial of lot {lot['id']}"),
                    )
                    closed_id = cur.lastrowid

                closed.append({
                    "lot_id": closed_id,
                    "shares": round(take, 6),
                    "cost_basis_per_share": round(lot["cost_basis_per_share"], 4),
                    "proceeds_per_share": round(price, 4),
                    "realized_pnl": round(pnl, 2),
                    "holding_days": holding_days,
                    "is_long_term": bool(is_lt),
                    "wash_disallowed": round(wash_disallowed, 2) if wash_disallowed else None,
                })
                total_proceeds += proceeds
                total_cost += cost
                total_pnl += pnl
                if is_lt:
                    long_term_pnl += pnl
                else:
                    short_term_pnl += pnl
                remaining -= take

            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    return {
        "symbol": symbol.upper(),
        "method": method,
        "shares_sold": shares,
        "sold_at": sold_at,
        "price": price,
        "total_proceeds": round(total_proceeds, 2),
        "total_cost_basis": round(total_cost, 2),
        "total_realized_pnl": round(total_pnl, 2),
        "long_term_pnl": round(long_term_pnl, 2),
        "short_term_pnl": round(short_term_pnl, 2),
        "lots_closed": closed,
        "wash_sale_warnings": wash_warnings,
    }


def list_lots(*, symbol: str | None = None, open_only: bool = False) -> list[dict]:
    where = []
    params: list = []
    if symbol:
        where.append("symbol = ?")
        params.append(symbol.upper())
    if open_only:
        where.append("closed_at IS NULL")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    with db.conn() as c:
        rows = c.execute(
            f"SELECT * FROM tax_lots {where_sql} ORDER BY acquired_at, id",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def harvest_candidates(prices: dict[str, float], today: str | None = None) -> dict:
    """For each open lot, compute unrealized P&L using current prices.

    `prices` is a dict {symbol: current_price}. Highlights:
      - lots with unrealized loss (harvest candidates — sell to realize loss)
      - lots about to cross long-term threshold (wait → lower tax)
      - lots already long-term + at gain (safer to realize)

    Args `today` is YYYY-MM-DD; defaults to actual today.
    """
    today_str = today or datetime.utcnow().strftime("%Y-%m-%d")
    today_dt = datetime.fromisoformat(today_str)

    out = {
        "as_of": today_str,
        "harvest_candidates_loss": [],   # currently underwater — sell now to harvest
        "approaching_long_term": [],      # < 30 days from 365-day mark — wait
        "long_term_at_gain": [],          # already LT + green — flexible to realize
        "totals": {"unrealized_pnl": 0.0, "by_currency": {}},
    }

    for lot in list_lots(open_only=True):
        sym = lot["symbol"]
        cur_price = prices.get(sym)
        if cur_price is None:
            continue
        cost = lot["cost_basis_per_share"]
        shares = lot["shares"]
        unrealized = (cur_price - cost) * shares
        held_days = (today_dt - datetime.fromisoformat(lot["acquired_at"])).days
        days_to_lt = LONG_TERM_THRESHOLD_DAYS - held_days
        ccy = lot["currency"]

        item = {
            "lot_id": lot["id"],
            "symbol": sym,
            "shares": shares,
            "cost_basis_per_share": cost,
            "current_price": round(cur_price, 4),
            "unrealized_pnl": round(unrealized, 2),
            "currency": ccy,
            "holding_days": held_days,
            "days_to_long_term": max(0, days_to_lt),
            "is_long_term": held_days > LONG_TERM_THRESHOLD_DAYS,
        }

        out["totals"]["unrealized_pnl"] += unrealized
        out["totals"]["by_currency"].setdefault(ccy, 0.0)
        out["totals"]["by_currency"][ccy] += unrealized

        if unrealized < 0:
            out["harvest_candidates_loss"].append(item)
        elif 0 < days_to_lt <= 30:
            out["approaching_long_term"].append(item)
        elif item["is_long_term"] and unrealized > 0:
            out["long_term_at_gain"].append(item)

    out["totals"]["unrealized_pnl"] = round(out["totals"]["unrealized_pnl"], 2)
    out["totals"]["by_currency"] = {
        k: round(v, 2) for k, v in out["totals"]["by_currency"].items()
    }
    out["harvest_candidates_loss"].sort(key=lambda x: x["unrealized_pnl"])  # most negative first
    out["approaching_long_term"].sort(key=lambda x: x["days_to_long_term"])
    out["long_term_at_gain"].sort(key=lambda x: -x["unrealized_pnl"])  # most positive first
    return out
