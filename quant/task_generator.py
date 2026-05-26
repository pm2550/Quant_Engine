"""Generate backtest tasks (strategies × params × symbols × periods) into the queue."""
from __future__ import annotations
import itertools
import json
import logging
import sqlite3
from datetime import date, timedelta

from . import config as cfg_mod
from . import db

log = logging.getLogger(__name__)


def _params_grid_dual_ma():
    for short, long in itertools.product([5, 10, 15, 20, 30], [20, 50, 100, 150, 200]):
        if short < long:
            yield {"short": short, "long": long}


def _params_grid_rsi():
    for period, oversold, overbought in itertools.product(
        [7, 14, 21, 28],
        [15, 20, 25, 30],
        [65, 70, 75, 80],
    ):
        if oversold < overbought - 30:
            yield {"period": period, "oversold": oversold, "overbought": overbought}


def _params_grid_bb():
    for period, std in itertools.product([10, 20, 30, 50], [1.5, 2.0, 2.5, 3.0]):
        yield {"period": period, "std": std}


def _params_grid_macd():
    for fast, slow, signal in itertools.product([8, 10, 12, 15], [21, 26, 30, 35], [7, 9, 11, 13]):
        if fast < slow:
            yield {"fast": fast, "slow": slow, "signal": signal}


GRIDS = {
    "dual_ma": _params_grid_dual_ma,
    "rsi_meanrev": _params_grid_rsi,
    "bb_breakout": _params_grid_bb,
    "macd_cross": _params_grid_macd,
}


def _viable_symbols_for_period(symbols: list[str], min_rows_per_year: int = 200) -> dict[str, int]:
    """Return {symbol: max_period_years} based on local Parquet history length."""
    from . import fetcher
    out = {}
    for s in symbols:
        df = fetcher.load_local(s)
        if df.empty:
            continue
        years = max(1, len(df) // min_rows_per_year)
        out[s] = years
    return out


def seed(periods: list[int] | None = None) -> int:
    """Push the full strategy×param×symbol grid into the queue.

    Only enqueues (symbol, period) combinations the symbol has enough history for.
    """
    db.init()
    portfolio = cfg_mod.load("portfolio")
    symbols = cfg_mod.all_symbols(portfolio)
    periods = periods or [3, 5, 10]
    viable = _viable_symbols_for_period(symbols)

    n_added, n_skipped = 0, 0
    for strategy, grid_fn in GRIDS.items():
        for params in grid_fn():
            for symbol in symbols:
                max_years = viable.get(symbol, 0)
                for years in periods:
                    if years > max_years:
                        n_skipped += 1
                        continue
                    new_id = db.enqueue(strategy, symbol, params, period_years=years, priority=0)
                    if new_id is not None:
                        n_added += 1
    log.info("seeded %d new tasks (skipped %d for insufficient history)", n_added, n_skipped)
    return n_added


# ---- Walk-forward generator: re-test all strategies at past month-end snapshots ----

def _month_ends_back(n: int) -> list[str]:
    """Return last `n` month-end dates as YYYY-MM-DD, ending last completed month."""
    today = date.today()
    out: list[str] = []
    # Go to end of previous month, then walk back monthly
    y, m = today.year, today.month
    for i in range(n):
        # subtract i+1 months
        mm = m - (i + 1)
        yy = y
        while mm <= 0:
            mm += 12
            yy -= 1
        # last day of that month
        if mm == 12:
            next_month_start = date(yy + 1, 1, 1)
        else:
            next_month_start = date(yy, mm + 1, 1)
        last_day = next_month_start - timedelta(days=1)
        out.append(last_day.isoformat())
    return out


def walk_forward(*, n_months: int = 18, periods: list[int] | None = None) -> int:
    """For each (strategy, params, symbol), re-run with as_of set to past N month-ends."""
    db.init()
    portfolio = cfg_mod.load("portfolio")
    symbols = cfg_mod.all_symbols(portfolio)
    periods = periods or [3, 5]
    viable = _viable_symbols_for_period(symbols)
    month_ends = _month_ends_back(n_months)

    n_added, n_skipped = 0, 0
    for strategy, grid_fn in GRIDS.items():
        for params in grid_fn():
            for symbol in symbols:
                max_years = viable.get(symbol, 0)
                for years in periods:
                    if years > max_years:
                        n_skipped += 1
                        continue
                    for as_of in month_ends:
                        wf_params = dict(params, as_of=as_of)
                        new_id = db.enqueue(strategy, symbol, wf_params,
                                            period_years=years, priority=-1)
                        if new_id is not None:
                            n_added += 1
    log.info("walk-forward: seeded %d tasks (skipped %d)", n_added, n_skipped)
    return n_added


# ---- Refinement generator: take top winners, generate neighbor params ----

NUMERIC_NEIGHBORS = {
    "dual_ma":     {"short": [-2, -1, 1, 2], "long": [-10, -5, 5, 10]},
    "rsi_meanrev": {"period": [-3, -1, 1, 3], "oversold": [-3, -1, 1, 3], "overbought": [-3, -1, 1, 3]},
    "bb_breakout": {"period": [-3, -1, 1, 3], "std": [-0.25, 0.25]},
    "macd_cross":  {"fast": [-1, 1], "slow": [-2, 2], "signal": [-1, 1]},
}


def refine_winners(*, top_n: int = 30, max_neighbors: int = 6) -> int:
    """Pick top-Sharpe completed results (with non-trivial trades), generate neighbor params."""
    db.init()
    with sqlite3.connect(db.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT bt.strategy, bt.symbol, bt.params_json, bt.period_years, br.sharpe
            FROM backtest_tasks bt
            JOIN backtest_results br ON bt.id = br.task_id
            WHERE br.n_trades >= 5 AND br.sharpe > 0.5
            ORDER BY br.sharpe DESC
            LIMIT ?
        """, (top_n,)).fetchall()

    n_added = 0
    for r in rows:
        strategy = r["strategy"]
        base_params = json.loads(r["params_json"])
        # don't refine walk-forward variants
        if "as_of" in base_params:
            continue
        deltas = NUMERIC_NEIGHBORS.get(strategy, {})
        if not deltas:
            continue
        keys = list(deltas.keys())
        # generate cartesian product but cap total
        choices: list[list[float]] = []
        for k in keys:
            choices.append([0] + deltas[k])
        all_combos = list(itertools.product(*choices))[:max_neighbors * 4]
        for combo in all_combos:
            new_params = dict(base_params)
            ok = True
            for k, dv in zip(keys, combo):
                v = base_params.get(k)
                if v is None:
                    ok = False
                    break
                if isinstance(v, int):
                    new_params[k] = max(2, int(v + dv))
                else:
                    new_params[k] = round(float(v) + dv, 2)
            if not ok:
                continue
            if new_params == base_params:
                continue
            # crude validity for known strategies
            if strategy == "dual_ma" and new_params["short"] >= new_params["long"]:
                continue
            if strategy == "macd_cross" and new_params["fast"] >= new_params["slow"]:
                continue
            new_id = db.enqueue(strategy, r["symbol"], new_params,
                                period_years=r["period_years"], priority=1)
            if new_id is not None:
                n_added += 1
    log.info("refine: seeded %d new tasks (from top %d)", n_added, top_n)
    return n_added


def stats() -> dict:
    return db.stats()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "walk_forward", "refine"], default="seed")
    parser.add_argument("--periods", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--n-months", type=int, default=18)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.mode == "seed":
        n = seed(periods=args.periods)
    elif args.mode == "walk_forward":
        n = walk_forward(n_months=args.n_months, periods=args.periods[:2])
    elif args.mode == "refine":
        n = refine_winners(top_n=args.top_n)

    print(f"added {n} tasks; queue stats: {stats()}")
