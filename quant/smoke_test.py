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
    cfg_mod.load("sources")
    # config/llm.yaml retired 2026-06-02 (qwen coding plan expiry, commit 7d9d90a).
    # Routing now lives in config/llm_routes.yaml — check that instead.
    routes = cfg_mod.load("llm_routes")
    assert routes.get("routes"), "no routes in llm_routes.yaml"
    assert routes.get("providers"), "no providers in llm_routes.yaml"


@check("secrets file readable")
def _():
    p = Path("/data2/quant/secrets/secrets.env")
    assert p.exists(), "secrets.env missing"
    txt = p.read_text()
    for k in ("OLLAMA_CLOUD_KEY", "GEMINI_API_KEY"):
        assert k in txt, f"{k} missing"
    # DASHSCOPE_CODING_KEY intentionally not required — coding plan expired 2026-06-01.


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
@check("simple_chat route reachable")
def _():
    out = llm_router.chat("回复 OK", task="simple_chat", max_tokens=20, timeout=30)
    assert "OK" in out["text"] or len(out["text"]) > 0


@check("ollama-cloud reachable")
def _():
    """直接打 ollama provider, 不经过 route 链。

    2026-09-11: 原来这里是 `chat(task="reasoning")` 然后断言 backend 以 "ollama:" 开头。
    加了 nano-gpt 跨 provider fallback 之后, Ollama 限流时链路会正确地落到 nano-gpt ——
    于是这条检查开始失败, 而"fallback 生效"恰恰是我们想要的行为。
    检查单个 provider 的可达性就该绕开 fallback 链, 否则它测的是"谁兜底"而不是"谁可达"。
    限流不算挂 (配额打满是常态, 链路已能接住), 只告警。
    """
    import os
    if not os.environ.get("OLLAMA_CLOUD_KEY"):
        warn("OLLAMA_CLOUD_KEY not set")
        return
    provider = llm_router._providers().get("ollama")
    if provider is None:
        warn("ollama provider 未配置")
        return
    try:
        out = provider.chat("glm-5.1", [{"role": "user", "content": "say ok"}],
                             max_tokens=30, timeout=60)
        assert out.get("text") or out.get("thinking"), "ollama 返回空"
    except Exception as e:  # noqa: BLE001
        if llm_router._is_rate_limited(e):
            warn(f"ollama-cloud 限流中 (配额打满, nano-gpt 会接住): {repr(e)[:80]}")
            return
        raise


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


# ===== 5b. 2026-09-10 全量审计的回归哨兵 =====
# 每一项都对应 docs/AUDIT-20260910.md 里的一个编号。这些问题全部是"静默失效"型 ——
# 没有报错、没有告警, 只是数字悄悄变成假的。所以必须有哨兵。

@check("A1: 价格缓存新鲜度 (>90% 在 2 天内)")
def _():
    from . import price_refresh
    rep = price_refresh.staleness_report()
    total = rep["total"]
    fresh = rep["buckets"].get("≤2天", 0)
    assert total > 0, "没有任何缓存价格"
    pct = 100.0 * fresh / total
    if pct < 90:
        warn(f"只有 {fresh}/{total} ({pct:.0f}%) 只价格在 2 天内 — "
             f"challenger 会用陈旧特征出预测; 跑 python -m quant.price_refresh --stale-only")
    assert pct >= 50, f"价格缓存严重陈旧: 仅 {pct:.0f}% 新鲜"


@check("A2: news_archive.published_at 全为 ISO")
def _():
    with sqlite3.connect(db.DB_PATH) as conn:
        bad = conn.execute(
            "SELECT COUNT(*) FROM news_archive WHERE published_at IS NOT NULL "
            "AND published_at NOT LIKE '____-__-__T%'").fetchone()[0]
    assert bad == 0, (f"{bad} 行 published_at 不是 ISO 格式 — "
                      "时间窗口筛选会静默失效 (db.normalize_timestamp 没接上?)")


@check("A4: B 站 total_results 撞顶时不报假趋势")
def _():
    from .alt_data import bilibili
    for sym, kws in bilibili.DEFAULT_KEYWORDS.items():
        for kw in kws:
            t = bilibili.trend(kw)
            if t.get("total_results_capped"):
                v = (t.get("vs_7d_ago") or {}).get("total_results_pct")
                assert v is None, (f"{kw}: total_results 撞 1000 上限却报出趋势 {v} — "
                                   "这个数是假的")


@check("C6: challenger 预测不混用陈旧特征")
def _():
    from .ml import serve
    preds, _ = serve.get_predictions(refresh=False)
    if not preds:
        warn("challenger 无预测文件")
        return
    fresh, stale = serve.split_by_freshness(preds)
    pct = 100.0 * len(fresh) / max(len(preds), 1)
    if pct < 80:
        warn(f"challenger 只有 {len(fresh)}/{len(preds)} ({pct:.0f}%) 只特征在 3 天内 — "
             "排序会混入旧日期的预测")


@check("C9: multi_factor 阈值可触发 (非永不触发)")
def _():
    from . import multi_factor
    t = multi_factor.action_thresholds()
    assert t["add"] <= 0.20, (f"ADD 阈值 {t['add']} 偏高 — 历史上 0.30 导致 4 个月仅触发 3 次; "
                              "跑 python -m quant.threshold_report 复核")
    # conviction 映射必须让可触发的信号够到 2 星 (日报按 <2 过滤)
    c = multi_factor.conviction_from_composite(t["add"])
    assert c >= 2, f"composite 达到 ADD 阈值时 conviction 只有 {c} 星 — 会被日报过滤掉"


@check("D2: backtest_results 无 Inf/NaN 指标")
def _():
    with sqlite3.connect(db.DB_PATH) as conn:
        bad = conn.execute(
            "SELECT COUNT(*) FROM backtest_results WHERE sharpe > 1e6 OR sharpe < -1e6 "
            "OR sortino > 1e6 OR profit_factor > 1e6").fetchone()[0]
    assert bad == 0, f"{bad} 条回测结果是 Inf — 会顶满 ORDER BY sharpe DESC"


@check("E3: 组合权重用跨币种口径")
def _():
    from . import orchestrator
    # USDCNY 必须取到真实汇率, 否则 A 股持仓权重会算错
    r = orchestrator.fx_to_usd("CNY")
    assert 0.08 < r < 0.25, f"CNY→USD 汇率 {r} 不合理 (检查 data/macro/usdcny.parquet)"


@check("C4: 校准闭环在跑 (model_calibration 有近期记录)")
def _():
    with sqlite3.connect(db.DB_PATH) as conn:
        row = conn.execute(
            "SELECT MAX(computed_at) FROM model_calibration").fetchone()
    last = row[0] if row else None
    if not last:
        warn("model_calibration 表为空 — 跑 python -m quant.calibration")
        return
    age = (datetime.utcnow().date() - datetime.strptime(last, "%Y-%m-%d").date()).days
    if age > 3:
        warn(f"最近一次模型校准是 {last} ({age} 天前) — quant-calibration.timer 是否在跑?")


@check("B2: LLM 限流熔断状态")
def _():
    st = llm_router.rate_limit_state()
    if st:
        warn(f"以下 backend 正在限流冷却中: {st}")


@check("H1: 每条 LLM route 都有跨 provider fallback")
def _():
    """2026-09-11: Ollama Cloud 所有 route 共享同一份配额, 打满时"fallback 到同一家
    的另一个模型"毫无用处 —— 整条链会一起 429。每条 route 必须至少跨两家。
    """
    routes = cfg_mod.load("llm_routes").get("routes") or {}
    assert routes, "没有任何 route 配置"
    single = []
    for task, chain in routes.items():
        if len({e.split(":", 1)[0] for e in chain}) < 2:
            single.append(task)
    assert not single, f"以下 route 只有单一 provider, 该家故障就全断: {single}"


@check("H1b: nano-gpt 可达 (付费 fallback)")
def _():
    import os
    if not os.environ.get("NANOGPT_API_KEY"):
        warn("NANOGPT_API_KEY 未设置 — 所有 route 会退回单 provider")
        return
    out = llm_router.chat("回复 OK", task="simple_chat", max_tokens=60, timeout=90)
    assert out.get("text"), "simple_chat 链全部失败"


@check("H1c: LLM 近 7 天花费")
def _():
    """nano-gpt 是付费的。实报 cost 已写入 llm_audit, 这里盯着别悄悄烧钱。"""
    with sqlite3.connect(db.DB_PATH) as conn:
        row = conn.execute(
            "SELECT ROUND(SUM(COALESCE(cost_usd,0)), 4), COUNT(*) FROM llm_audit "
            "WHERE ts >= date('now','-7 day') AND backend LIKE 'nanogpt:%'").fetchone()
    cost, n = (row[0] or 0.0), (row[1] or 0)
    if n:
        print(f"   nano-gpt 近 7 天: {n} 次调用, ${cost}")
    if cost > 5.0:
        warn(f"nano-gpt 近 7 天花费 ${cost} — 超过 $5, 检查是不是 Ollama 一直在 429")


# ===== 6. Systemd services =====
import subprocess

@check("all systemd services active")
def _():
    """只检查常驻 daemon (Type=simple)。

    2026-09-10: 原来把 quant-newswatch.service 也列进来 —— 那是 timer 驱动的 oneshot,
    平时本来就是 inactive, 于是这条检查每次都告警。长期常亮的告警会训练人忽略告警,
    真出问题时反而看不见 (这次审计里 smoke_test 因为一个已废弃的 llm.yaml 红了 21 小时
    没人管, 就是同一个病)。oneshot 单元改为检查"上一次是否成功"。
    """
    daemons = [
        "quant-api.service",
        "quant-backtest.service",
        "quant-intraday.service",
        "quant-audio-worker.service",
        "quant-anomaly-watcher.service",
        "quant-investigator.service",
        # dashscope-proxy.service deliberately retired 2026-06-02 — do not check.
    ]
    for svc in daemons:
        try:
            r = subprocess.run(["sudo", "-n", "systemctl", "is-active", svc],
                               capture_output=True, text=True, timeout=5)
            status = r.stdout.strip()
            if status != "active":
                warn(f"{svc}: {status} (常驻服务不该是这个状态)")
        except Exception as e:
            warn(f"{svc}: check failed: {e}")


@check("timer-driven oneshot units last run OK")
def _():
    """oneshot 单元看 ExecMainStatus (上次退出码), 不看 Active。"""
    oneshots = [
        "quant-newswatch.service",
        "quant-daily.service",
        "quant-calibration.service",
        "quant-price-refresh.service",
        "quant-expectations.service",
    ]
    for svc in oneshots:
        try:
            r = subprocess.run(
                ["sudo", "-n", "systemctl", "show", "-p", "ExecMainStatus",
                 "-p", "Result", "--value", svc],
                capture_output=True, text=True, timeout=5)
            vals = [v for v in r.stdout.strip().splitlines() if v]
            if not vals:
                continue
            result = vals[-1] if len(vals) > 1 else vals[0]
            if result not in ("success", "0", ""):
                warn(f"{svc}: 上次运行 result={result}")
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
