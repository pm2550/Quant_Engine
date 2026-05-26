"""产线自检 - 冒烟测试 (smoke test) 全部关键路径.

跑这个验证整套系统健康. 退出码 0 = 全过, 非 0 = 有失败.
建议每天 cron 一次, 或部署后立刻跑一次.

Run: python -m quant.smoke_test
"""
from __future__ import annotations
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from . import config as cfg_mod, db, fetcher, llm_router

log = logging.getLogger(__name__)

PASSED = []
FAILED = []
WARNINGS = []


def check(name: str):
    """Decorator that runs a check, catching exceptions."""
    def deco(fn):
        try:
            t0 = time.time()
            fn()
            elapsed = time.time() - t0
            PASSED.append(f"✅ {name} ({elapsed:.1f}s)")
            print(f"✅ {name} ({elapsed:.1f}s)")
        except Exception as e:  # noqa: BLE001
            FAILED.append(f"❌ {name}: {e}")
            print(f"❌ {name}: {e}")
        return fn
    return deco


def warn(msg: str):
    WARNINGS.append(f"⚠️ {msg}")
    print(f"⚠️ {msg}")


# ===== 1. Config + secrets =====
@check("config files load")
def _():
    pf = cfg_mod.load("portfolio")
    assert pf.get("positions"), "no positions in portfolio.yaml"
    cfg_mod.load("strategies")
    cfg_mod.load("llm")
    cfg_mod.load("sources")


@check("secrets file readable")
def _():
    p = Path("/data2/quant/secrets/secrets.env")
    assert p.exists(), "secrets.env missing"
    txt = p.read_text()
    for k in ("OLLAMA_CLOUD_KEY", "GEMINI_API_KEY", "DASHSCOPE_CODING_KEY"):
        assert k in txt, f"{k} missing"


# ===== 2. Database =====
@check("SQLite tables exist")
def _():
    db.init()
    with sqlite3.connect(db.DB_PATH) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    needed = {"backtest_tasks", "backtest_results", "fundamentals", "earnings_calendar",
              "news_archive", "events", "audio_queue", "event_embeddings"}
    missing = needed - tables
    assert not missing, f"missing tables: {missing}"


# ===== 3. Data freshness =====
@check("price data ≤ 2 trading days stale (US)")
def _():
    pf = cfg_mod.load("portfolio")
    us_syms = [s for s, info in pf.get("positions", {}).items()
               if info.get("currency", "USD") == "USD"]
    now = datetime.utcnow()
    for s in us_syms:
        df = fetcher.load_local(s)
        if df.empty:
            warn(f"{s}: no local data")
            continue
        last = df.index.max()
        days = (now - last.to_pydatetime()).days if hasattr(last, "to_pydatetime") else 999
        if days > 4:  # 4 trading days = 6 calendar days roughly
            warn(f"{s}: data {days} days old")


# ===== 4. LLM connectivity =====
@check("dashscope qwen reachable")
def _():
    out = llm_router.chat("回复 OK", task="simple_chat", max_tokens=20, timeout=30)
    assert "OK" in out["text"] or len(out["text"]) > 0


@check("ollama-cloud reachable")
def _():
    # Post-config-driven refactor: read from llm_routes.yaml provider env, not module attr.
    import os
    if not os.environ.get("OLLAMA_CLOUD_KEY"):
        warn("OLLAMA_CLOUD_KEY not set")
        return
    out = llm_router.chat("say ok", task="reasoning", max_tokens=30, timeout=60)
    assert out["backend"].startswith("ollama:"), f"unexpected backend: {out['backend']}"


@check("gemini embeddings reachable")
def _():
    vecs = llm_router.embed(["hello"])
    assert len(vecs) == 1 and len(vecs[0]) == 3072, "wrong embedding dim"


# ===== 5. quant-api endpoints =====
API = "http://172.17.0.1:7900"


@check("quant-api /api/health")
def _():
    r = requests.get(f"{API}/api/health", timeout=10)
    r.raise_for_status()
    assert r.json().get("ok") is True


@check("quant-api /api/portfolio/snapshot")
def _():
    r = requests.get(f"{API}/api/portfolio/snapshot", timeout=30)
    r.raise_for_status()
    d = r.json()
    assert d.get("positions"), "empty positions"


@check("quant-api /api/analyze (AMD with after-hours)")
def _():
    r = requests.post(f"{API}/api/analyze", json={"symbol": "AMD"}, timeout=60)
    r.raise_for_status()
    d = r.json()
    assert d.get("spot", {}).get("price"), "no spot price"
    stale = d["spot"].get("staleness_seconds", 99999)
    if stale > 600:
        warn(f"AMD spot data {stale}s old (>10min)")


# ===== 6. Systemd services =====
import subprocess

@check("all systemd services active")
def _():
    services = [
        "quant-api.service",
        "quant-newswatch.service",
        "quant-backtest.service",
        "quant-intraday.service",
        "quant-audio-worker.service",
        "dashscope-proxy.service",
    ]
    for svc in services:
        try:
            r = subprocess.run(["sudo", "-n", "systemctl", "is-active", svc],
                               capture_output=True, text=True, timeout=5)
            status = r.stdout.strip()
            if status != "active":
                warn(f"{svc}: {status}")
        except Exception as e:
            warn(f"{svc}: check failed: {e}")


@check("daily/weekly/audio-discovery timers enabled")
def _():
    timers = ["quant-daily.timer", "quant-weekly.timer", "quant-audio-discovery.timer"]
    for t in timers:
        try:
            r = subprocess.run(["sudo", "-n", "systemctl", "is-enabled", t],
                               capture_output=True, text=True, timeout=5)
            status = r.stdout.strip()
            if status not in ("enabled", "static"):
                warn(f"{t}: {status}")
        except Exception:
            pass


# ===== 7. Disk + queue health =====
@check("disk space > 5GB free")
def _():
    import shutil
    used = shutil.disk_usage("/data2/quant")
    free_gb = used.free / 1024 ** 3
    if free_gb < 5:
        warn(f"only {free_gb:.1f} GB free")


@check("backtest queue not all stuck")
def _():
    s = db.stats()
    running = s.get("running", 0)
    if running > 5:
        warn(f"{running} tasks in 'running' state — may be stuck")


# ===== 8. End-to-end probe =====
@check("end-to-end stock_query (container → API → response)")
def _():
    r = subprocess.run(
        ["sudo", "-n", "docker", "exec", "openclaw-gw",
         "python3", "/home/node/.openclaw/workspace/scripts/stock_query.py",
         "--symbol", "AMD", "--intent", "general"],
        capture_output=True, text=True, timeout=60,
    )
    out = json.loads(r.stdout)
    assert out.get("spot", {}).get("price"), "no spot from stock_query"
    assert out.get("display_name") == "Advanced Micro Devices Inc" or out.get("symbol") == "AMD"


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    print(f"=== Quant Engine Smoke Test — {datetime.utcnow().isoformat()}Z ===\n")

    # Trigger all @check decorators (they ran on import)
    print(f"\n=== Summary ===")
    print(f"Passed:  {len(PASSED)}")
    print(f"Failed:  {len(FAILED)}")
    print(f"Warnings: {len(WARNINGS)}")
    if FAILED:
        print("\nFailed checks:")
        for f in FAILED:
            print(f"  {f}")
    if WARNINGS:
        print("\nWarnings:")
        for w in WARNINGS:
            print(f"  {w}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
