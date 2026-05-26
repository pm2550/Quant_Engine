"""SQLite schema for backtest task queue and results."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "quant.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    params_json TEXT NOT NULL,
    period_years INTEGER NOT NULL DEFAULT 5,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    priority INTEGER NOT NULL DEFAULT 0,     -- higher = run sooner
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    UNIQUE(strategy, symbol, params_json, period_years)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON backtest_tasks(status, priority DESC, id);

CREATE TABLE IF NOT EXISTS backtest_results (
    task_id INTEGER PRIMARY KEY REFERENCES backtest_tasks(id) ON DELETE CASCADE,
    total_return REAL,
    annual_return REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown REAL,
    win_rate REAL,
    n_trades INTEGER,
    profit_factor REAL,
    extra_json TEXT,
    finished_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_sharpe ON backtest_results(sharpe DESC);

-- ============= Phase 1+ schema additions =============

-- Fundamentals snapshot (per symbol per day)
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT NOT NULL,
    as_of TEXT NOT NULL,           -- YYYY-MM-DD
    pe REAL, pb REAL, ps REAL,
    roe REAL, roa REAL,
    revenue_yoy REAL, eps_yoy REAL,
    market_cap REAL,
    shares_outstanding REAL,
    extra_json TEXT,
    PRIMARY KEY (symbol, as_of)
);

-- Earnings calendar (when companies report)
CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol TEXT NOT NULL,
    fiscal_period TEXT,            -- e.g. Q1 2026
    report_date TEXT NOT NULL,     -- YYYY-MM-DD
    eps_estimate REAL,
    eps_actual REAL,
    revenue_estimate REAL,
    revenue_actual REAL,
    surprise_pct REAL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, report_date)
);

-- Calendar of corporate events: dividends, splits, conferences, etc.
CREATE TABLE IF NOT EXISTS corporate_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,      -- ex_dividend / dividend_pay / split / conference / investor_day
    event_date TEXT NOT NULL,      -- YYYY-MM-DD
    amount REAL,                   -- dividend $ per share, split ratio, etc
    notes TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE (symbol, event_type, event_date)
);
CREATE INDEX IF NOT EXISTS idx_corp_event_date ON corporate_events(event_date);

-- Macro economic events (FOMC, CPI, NFP, etc)
CREATE TABLE IF NOT EXISTS macro_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,      -- FOMC / CPI / NFP / GDP / PMI / ECB / BOJ / 央行 / 国常会
    region TEXT NOT NULL,          -- US / EU / CN / JP
    event_date TEXT NOT NULL,      -- YYYY-MM-DD
    event_time_utc TEXT,           -- HH:MM UTC if known
    expected TEXT,                 -- "consensus" / "estimate" if available
    actual TEXT,                   -- after release
    notes TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE (event_type, region, event_date)
);
CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_events(event_date);

-- News items archive (RSS + searches dedup'd)
CREATE TABLE IF NOT EXISTS news_archive (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    source TEXT NOT NULL,          -- reuters / fed / xinhua / ...
    published_at TEXT,             -- ISO timestamp
    content TEXT,                  -- snippet/summary
    raw_hash TEXT,                 -- for dedup
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_archive(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_source ON news_archive(source, published_at DESC);

-- High-severity events (filtered from news_archive by LLM)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id INTEGER REFERENCES news_archive(id),
    severity INTEGER NOT NULL,     -- 0..10 LLM-scored
    category TEXT,                 -- macro/policy/geopolitical/industry/single-stock
    summary TEXT NOT NULL,
    impact_json TEXT,              -- {symbol: {direction, confidence, reasoning}}
    affected_symbols TEXT,         -- comma-sep for fast filter
    fired_at TEXT NOT NULL,
    pushed_at TEXT,                -- when alerted to user (NULL = not yet)
    archived INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity DESC, fired_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_pushed ON events(pushed_at);

-- Audio transcription queue
CREATE TABLE IF NOT EXISTS audio_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- fed / juchao / ir-VRT / podcast-wsj / ...
    title TEXT,
    audio_url TEXT NOT NULL UNIQUE,
    discovered_at TEXT NOT NULL,
    priority INTEGER DEFAULT 5,    -- 0..10, 10 = highest
    status TEXT DEFAULT 'pending', -- pending / running / done / failed
    transcript TEXT,               -- final transcript
    summary TEXT,                  -- LLM summary
    impact_json TEXT,              -- per-holding impact
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_audio_status ON audio_queue(status, priority DESC);

-- Event embeddings for similarity search
CREATE TABLE IF NOT EXISTS event_embeddings (
    event_id INTEGER PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    text TEXT NOT NULL,            -- embedded text (title + summary)
    embedding BLOB NOT NULL,       -- float32 array bytes
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- LLM call audit log: every chat()/embed() call gets a row.
-- Lets us see which provider/model was used, latency, token spend,
-- and what the call was for — answers "why did the bill jump" or
-- "which task is timing out".
CREATE TABLE IF NOT EXISTS llm_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,              -- ISO UTC
    task TEXT,                     -- ROUTES key (reasoning/simple_chat/...)
    backend TEXT NOT NULL,         -- "ollama:deepseek-v4-pro" / "dashscope:qwen3.6-plus" / "gemini:embedding-001"
    success INTEGER NOT NULL,      -- 1 = OK, 0 = error
    wall_time_s REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,                 -- estimated, NULL if unknown
    caller TEXT,                   -- module/function (best-effort)
    prompt_chars INTEGER,
    response_chars INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_audit_ts ON llm_audit(ts DESC);
CREATE INDEX IF NOT EXISTS idx_llm_audit_backend ON llm_audit(backend, ts DESC);

-- User-defined price/condition alerts.  Scanner timer evaluates `cond` every
-- N minutes during market hours and pushes to TG when it flips True.
-- Examples:
--   AMD 跌到 380:   {symbol: AMD, op: '<=', value: 380, basis: 'last'}
--   GRID RSI > 80:  {symbol: GRID, op: '>',  value: 80, basis: 'rsi'}
--   SOXX 跌破 MA50: {symbol: SOXX, op: 'cross_below', value: null, basis: 'ma50'}
CREATE TABLE IF NOT EXISTS user_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    op TEXT NOT NULL,              -- '<' | '<=' | '>' | '>=' | 'cross_below' | 'cross_above'
    value REAL,                    -- threshold (NULL for cross_* rules using basis line)
    basis TEXT NOT NULL,           -- 'last' | 'rsi' | 'ma20' | 'ma50' | 'ma200' | 'chg_1d_pct' | 'chg_20d_pct'
    note TEXT,                     -- user's reason for the alert
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    fired_at TEXT,                 -- last time it triggered (NULL = never)
    fired_count INTEGER NOT NULL DEFAULT 0,
    last_seen_value REAL,          -- most recent observed value (for cross_* state)
    cooldown_minutes INTEGER NOT NULL DEFAULT 60   -- min interval between re-fires
);
CREATE INDEX IF NOT EXISTS idx_user_alerts_enabled ON user_alerts(enabled, symbol);

-- Tax lots: cost-basis tracking for long/short-term capital gains.
-- Each "buy" creates one lot.  A "sell" is matched against open lots
-- (default FIFO) and writes per-lot realized P&L + holding_days.
--
-- US tax rules baked in: holding_days > 365 → long-term (typically 15-20%
-- federal vs short-term ordinary income 24-37%).
-- A-share has no equivalent capital-gains tax for individuals on T+ trades,
-- but we still track lots for P&L attribution.
CREATE TABLE IF NOT EXISTS tax_lots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    shares REAL NOT NULL,
    cost_basis_per_share REAL NOT NULL,
    acquired_at TEXT NOT NULL,           -- YYYY-MM-DD
    closed_at TEXT,                      -- NULL = open
    proceeds_per_share REAL,             -- price at sale
    realized_pnl REAL,                   -- (proceeds - cost) * closed_shares
    holding_days INTEGER,                -- closed_at - acquired_at
    is_long_term INTEGER,                -- 1 = held > 365 days
    wash_sale_disallowed REAL,           -- $ amount of loss disallowed (NULL = no wash)
    matched_against TEXT,                -- comma-sep IDs of replacement lots that triggered wash
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_tax_lots_symbol ON tax_lots(symbol, acquired_at);
CREATE INDEX IF NOT EXISTS idx_tax_lots_open ON tax_lots(symbol, closed_at);

-- Alternative data tracking: time series of leading indicators.
-- One row per (source, key, date) — e.g. ('bilibili_search', '异环 完美世界',
-- '2026-05-06', {total_results:1000, total_plays_top100:5234567, ...}).
-- newswatch / daily report / alerts can compare today vs 7d/30d trend.
CREATE TABLE IF NOT EXISTS alt_data_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,             -- 'bilibili_search' / 'fred' / 'tsm_revenue' / ...
    key TEXT NOT NULL,                -- search keyword / FRED series id / ticker / etc
    captured_at TEXT NOT NULL,        -- ISO UTC of capture
    metric_date TEXT NOT NULL,        -- YYYY-MM-DD (for daily metrics) or same as captured_at
    metrics_json TEXT NOT NULL,       -- JSON blob of the actual numbers
    notes TEXT,
    UNIQUE(source, key, metric_date)
);
CREATE INDEX IF NOT EXISTS idx_alt_source_key_date ON alt_data_metrics(source, key, metric_date DESC);

-- Expected forward return distributions, one row per (date, symbol, horizon).
-- Generated daily after close by quant.expectations: bootstrap empirical
-- distribution from past N days of rolling forward returns.
--
-- Used to detect "real" anomalies: actual_5d_return outside [p5, p95] = real
-- statistical surprise, not just "RSI > 75" heuristic. Calibration tracking
-- (predicted vs realized) deferred — write data now, query later.
CREATE TABLE IF NOT EXISTS expectations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,        -- YYYY-MM-DD when this row was generated
    symbol TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,      -- 1 / 5 / 20
    model_version TEXT NOT NULL,        -- 'bootstrap_v1' / future 'garch_v1' etc
    lookback_days INTEGER NOT NULL,     -- window used for the bootstrap
    n_samples INTEGER NOT NULL,         -- forward-return data points in distribution
    mean_pct REAL NOT NULL,
    median_pct REAL NOT NULL,
    sigma_pct REAL NOT NULL,
    p5_pct REAL NOT NULL,
    p25_pct REAL NOT NULL,
    p75_pct REAL NOT NULL,
    p95_pct REAL NOT NULL,
    min_pct REAL NOT NULL,
    max_pct REAL NOT NULL,
    anchor_close REAL NOT NULL,         -- price at snapshot date — anchor for actual return
    UNIQUE(snapshot_date, symbol, horizon_days, model_version)
);
CREATE INDEX IF NOT EXISTS idx_expectations_date ON expectations(snapshot_date DESC, symbol);
CREATE INDEX IF NOT EXISTS idx_expectations_symbol ON expectations(symbol, horizon_days, snapshot_date DESC);

-- Decision log: persist each non-HOLD recommendation so we can review predictive accuracy
-- 30 days later. Drives calibration / hit_rate dashboards instead of guessing whether the
-- engine is right. Written by daily.py at end of run.
CREATE TABLE IF NOT EXISTS decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,                -- ADD/WATCH_BUY/REDUCE/WATCH_SKIP/STOP_LOSS/DEFER_TO_LLM
    composite_score REAL,
    conviction INTEGER,                  -- 0-5 stars
    entry_price REAL,                    -- price at decision time
    currency TEXT,
    top_factors_json TEXT,               -- JSON list of structured top_factors
    counter_factors_json TEXT,
    review_due_at TEXT NOT NULL,         -- decided_at + 30 days
    reviewed_at TEXT,                    -- when review_decision ran (NULL = pending)
    actual_return_pct REAL,              -- pct change from entry_price to review price
    was_correct INTEGER                  -- 1 if direction matched action (NULL until reviewed)
);
CREATE INDEX IF NOT EXISTS idx_decision_log_pending ON decision_log(reviewed_at, review_due_at);
CREATE INDEX IF NOT EXISTS idx_decision_log_symbol ON decision_log(symbol, decided_at DESC);
"""


@contextmanager
def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    try:
        yield c
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript(SCHEMA)


def enqueue(strategy: str, symbol: str, params: dict, *, period_years: int = 5, priority: int = 0) -> int | None:
    """Insert task; return new id, or None if it already exists."""
    with conn() as c:
        try:
            cur = c.execute(
                "INSERT INTO backtest_tasks(strategy, symbol, params_json, period_years, priority, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (strategy, symbol, json.dumps(params, sort_keys=True), period_years, priority,
                 datetime.utcnow().isoformat()),
            )
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None


def claim_next() -> dict | None:
    """Atomically claim the next pending task (highest priority first)."""
    with conn() as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute(
            "SELECT * FROM backtest_tasks WHERE status='pending' "
            "ORDER BY priority DESC, id ASC LIMIT 1"
        ).fetchone()
        if not row:
            c.execute("COMMIT")
            return None
        c.execute(
            "UPDATE backtest_tasks SET status='running', started_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), row["id"]),
        )
        c.execute("COMMIT")
        return dict(row) | {"params": json.loads(row["params_json"])}


def finish(task_id: int, *, result: dict | None = None, error: str | None = None) -> None:
    with conn() as c:
        if error:
            c.execute(
                "UPDATE backtest_tasks SET status='failed', finished_at=?, error=? WHERE id=?",
                (datetime.utcnow().isoformat(), error[:1000], task_id),
            )
            return
        c.execute(
            "UPDATE backtest_tasks SET status='done', finished_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), task_id),
        )
        if result:
            extra = {k: v for k, v in result.items()
                     if k not in {"total_return","annual_return","sharpe","sortino","max_drawdown","win_rate","n_trades","profit_factor"}}
            c.execute(
                "INSERT OR REPLACE INTO backtest_results "
                "(task_id, total_return, annual_return, sharpe, sortino, max_drawdown, win_rate, n_trades, profit_factor, extra_json, finished_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    result.get("total_return"),
                    result.get("annual_return"),
                    result.get("sharpe"),
                    result.get("sortino"),
                    result.get("max_drawdown"),
                    result.get("win_rate"),
                    result.get("n_trades"),
                    result.get("profit_factor"),
                    json.dumps(extra),
                    datetime.utcnow().isoformat(),
                ),
            )


def log_llm_call(*, task: str | None, backend: str, success: bool,
                  wall_time_s: float | None = None,
                  tokens_in: int | None = None, tokens_out: int | None = None,
                  cost_usd: float | None = None, caller: str | None = None,
                  prompt_chars: int | None = None, response_chars: int | None = None,
                  error: str | None = None) -> None:
    """Append one row to llm_audit. Best-effort: never propagates exceptions."""
    try:
        with conn() as c:
            c.execute(
                "INSERT INTO llm_audit (ts, task, backend, success, wall_time_s, "
                "tokens_in, tokens_out, cost_usd, caller, prompt_chars, response_chars, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), task, backend, 1 if success else 0,
                 wall_time_s, tokens_in, tokens_out, cost_usd, caller,
                 prompt_chars, response_chars, (error or "")[:500] if error else None),
            )
    except Exception:  # noqa: BLE001
        # Never let audit logging break a real LLM call.
        pass


def llm_audit_summary(days: int = 7) -> dict:
    """Aggregate metrics over the last `days` days for quick health/cost view."""
    with conn() as c:
        rows = c.execute(
            "SELECT backend, "
            "COUNT(*) AS n_calls, "
            "SUM(success) AS n_success, "
            "SUM(tokens_in) AS tokens_in, "
            "SUM(tokens_out) AS tokens_out, "
            "SUM(cost_usd) AS cost_usd, "
            "AVG(wall_time_s) AS avg_latency_s, "
            "MAX(wall_time_s) AS max_latency_s "
            "FROM llm_audit WHERE ts >= datetime('now', ?) "
            "GROUP BY backend ORDER BY n_calls DESC",
            (f"-{int(days)} days",),
        ).fetchall()
        by_backend = [dict(r) for r in rows]
        total = c.execute(
            "SELECT COUNT(*) AS n, SUM(cost_usd) AS cost FROM llm_audit "
            "WHERE ts >= datetime('now', ?)",
            (f"-{int(days)} days",),
        ).fetchone()
        return {"days": days, "total_calls": total["n"] or 0,
                "total_cost_usd": total["cost"] or 0.0,
                "by_backend": by_backend}


def stats() -> dict:
    with conn() as c:
        r = c.execute(
            "SELECT status, COUNT(*) as n FROM backtest_tasks GROUP BY status"
        ).fetchall()
        return {row["status"]: row["n"] for row in r}
