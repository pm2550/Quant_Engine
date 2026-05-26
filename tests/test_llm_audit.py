"""Unit tests for llm_audit table — audit write + summary aggregation."""
from __future__ import annotations
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db(monkeypatch):
    """Point quant.db at a fresh temp SQLite file for the duration of the test."""
    tmp = Path(tempfile.mkdtemp()) / "test.sqlite"
    from quant import db
    monkeypatch.setattr(db, "DB_PATH", tmp)
    db.init()
    yield db
    if tmp.exists():
        tmp.unlink()


def test_log_llm_call_inserts_row(temp_db):
    temp_db.log_llm_call(
        task="reasoning", backend="ollama:deepseek-v4-pro", success=True,
        wall_time_s=2.3, tokens_in=100, tokens_out=500, cost_usd=0.0,
        caller="test.fn", prompt_chars=400, response_chars=2000,
    )
    with temp_db.conn() as c:
        row = c.execute("SELECT * FROM llm_audit").fetchone()
    assert row["task"] == "reasoning"
    assert row["backend"] == "ollama:deepseek-v4-pro"
    assert row["success"] == 1
    assert row["tokens_in"] == 100
    assert row["wall_time_s"] == 2.3


def test_log_llm_call_failure_records_error(temp_db):
    temp_db.log_llm_call(
        task="reasoning", backend="ollama:glm-5.1", success=False,
        wall_time_s=0.5, error="connection timeout",
    )
    with temp_db.conn() as c:
        row = c.execute("SELECT * FROM llm_audit").fetchone()
    assert row["success"] == 0
    assert "timeout" in row["error"]


def test_log_llm_call_swallows_exceptions(monkeypatch, temp_db):
    """Audit logger must NEVER raise — broken DB shouldn't break a real LLM call."""
    def boom(*args, **kw):
        raise RuntimeError("disk full")
    monkeypatch.setattr(temp_db, "conn", boom)
    # Should not raise
    temp_db.log_llm_call(task="x", backend="x:x", success=True)


def test_audit_summary_aggregates_per_backend(temp_db):
    for _ in range(3):
        temp_db.log_llm_call(task="simple_chat", backend="dashscope:qwen3.6-plus",
                              success=True, wall_time_s=1.0, tokens_in=50, tokens_out=100, cost_usd=0.0)
    temp_db.log_llm_call(task="reasoning", backend="ollama:deepseek-v4-pro",
                          success=True, wall_time_s=8.0, tokens_in=200, tokens_out=1500, cost_usd=0.0)
    temp_db.log_llm_call(task="reasoning", backend="ollama:deepseek-v4-pro",
                          success=False, wall_time_s=2.0, error="timeout")

    summary = temp_db.llm_audit_summary(days=7)
    assert summary["total_calls"] == 5
    by = {r["backend"]: r for r in summary["by_backend"]}
    assert by["dashscope:qwen3.6-plus"]["n_calls"] == 3
    assert by["dashscope:qwen3.6-plus"]["n_success"] == 3
    assert by["dashscope:qwen3.6-plus"]["tokens_in"] == 150
    assert by["ollama:deepseek-v4-pro"]["n_calls"] == 2
    assert by["ollama:deepseek-v4-pro"]["n_success"] == 1


def test_estimate_cost_unknown_backend_returns_none():
    from quant.llm_router import _estimate_cost
    assert _estimate_cost("unknown:model", 1000, 1000) is None


def test_estimate_cost_known_backend_returns_zero_for_free_tier():
    """All current providers are at $0 marginal cost (subscription/free).

    Backends are loaded from config/llm_routes.yaml `costs:` table.
    """
    from quant import llm_router as r
    r.reload_config()
    assert r._estimate_cost("dashscope:qwen3.6-plus", 100_000, 50_000) == 0.0
    assert r._estimate_cost("ollama:kimi-k2-thinking", 1_000_000, 1_000_000) == 0.0
    assert r._estimate_cost("ollama:glm-5.1", 1_000_000, 1_000_000) == 0.0
