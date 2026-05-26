"""分析师评级变化 / 目标价大幅调整 告警.

每天对持仓 + watchlist 跑 analyst_ratings.fetch_one (US / CN / ETF 自动分发),
跟前一天对比并 push TG; 同时 store() 写入 fundamentals.extra_json 持久化.

变化触发条件 (按 market):
  - US:  target_mean_price 移动 ≥5%, recommendation_mean 移动 ≥0.3, 新 Upgrade/Downgrade action
  - CN:  recommendation_mean (映射) ≥0.3, rating_breakdown 单评级 ±3 篇, 新机构覆盖
  - ETF: weighted_target_upside_pct ≥ 5pp, recommendation_mean ≥0.3
"""
from __future__ import annotations
import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config as cfg_mod, db, fetcher, telegram, analyst_ratings

log = logging.getLogger(__name__)

STATE_FILE = cfg_mod.RESULTS_DIR / "rating_changes.json"


def _load() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"snapshots": {}, "alerts_sent": []}


def _save(s: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


def _recent_actions(data: dict) -> list[str]:
    """统一抽取 recent_actions 列表 (US firm upgrades / CN 新研报 / ETF 暂空)."""
    market = data.get("market", "us")
    if market == "us":
        return [
            f"{c.get('date')}|{c.get('firm')}|{c.get('action')}|{c.get('from_grade')}→{c.get('to_grade')}"
            for c in data.get("recent_changes") or []
        ]
    if market == "cn":
        return [
            f"{r.get('date')}|{r.get('firm') or ''}|{r.get('rating') or ''}|{r.get('title') or ''}"
            for r in data.get("recent_research") or []
        ]
    return []


def _snapshot_one(symbol: str) -> dict | None:
    """拉一致预期 → 持久化到 fundamentals.extra → 返回比较用的紧凑快照."""
    data = analyst_ratings.fetch_one(symbol)
    if not data:
        return None
    try:
        analyst_ratings.store(symbol, data)
    except Exception:
        log.exception("store fundamentals.extra failed for %s", symbol)

    snap = {
        "market": data.get("market") or ("cn" if fetcher.is_a_share(symbol) else "us"),
        "target_mean_price": data.get("target_mean_price"),
        "recommendation_mean": data.get("recommendation_mean"),
        "recommendation_key": data.get("recommendation_key"),
        "n_analysts": data.get("number_of_analyst_opinions"),
        "current_price": data.get("current_price"),
        "rating_breakdown": data.get("rating_breakdown"),
        "weighted_target_upside_pct": data.get("weighted_target_upside_pct"),
        "coverage_weight": data.get("coverage_weight"),
        "recent_actions": _recent_actions(data),
        "ts": datetime.utcnow().isoformat() + "Z",
    }
    return snap


def diff_snapshot(prev: dict, cur: dict, *, sym: str) -> list[str]:
    """按市场分支生成可读变化描述."""
    out = []
    market = cur.get("market") or "us"
    unit = "¥" if market == "cn" else "$"

    # Target price change (US 才有数值目标价; CN 无, ETF 无聚合 target)
    p_t = prev.get("target_mean_price") or 0
    c_t = cur.get("target_mean_price") or 0
    if p_t and c_t and abs(c_t - p_t) / p_t >= 0.05:
        pct = (c_t - p_t) / p_t * 100
        arrow = "📈" if pct > 0 else "📉"
        out.append(f"{arrow} `{sym}` 目标价 {unit}{p_t:.0f} → {unit}{c_t:.0f} ({pct:+.1f}%)")

    # Recommendation mean shift (US/CN/ETF 都有, 量纲一致 1=买入 .. 5=卖出)
    p_r = prev.get("recommendation_mean")
    c_r = cur.get("recommendation_mean")
    if p_r and c_r and abs(c_r - p_r) >= 0.3:
        suffix = " (ETF 加权)" if market == "etf" else ""
        if c_r < p_r:
            out.append(f"⬆️ `{sym}` 评级转好{suffix} {p_r:.2f} → {c_r:.2f} (1=买入, 5=卖出)")
        else:
            out.append(f"⬇️ `{sym}` 评级转弱{suffix} {p_r:.2f} → {c_r:.2f}")

    # ETF weighted upside shift
    if market == "etf":
        p_u = prev.get("weighted_target_upside_pct")
        c_u = cur.get("weighted_target_upside_pct")
        if p_u is not None and c_u is not None and abs(c_u - p_u) >= 5:
            arrow = "📈" if c_u > p_u else "📉"
            out.append(
                f"{arrow} `{sym}` (ETF) 成分股加权上涨空间 {p_u:+.1f}% → {c_u:+.1f}%"
            )

    # CN rating breakdown shift: 任一评级类别 ±3 篇
    if market == "cn":
        p_b = prev.get("rating_breakdown") or {}
        c_b = cur.get("rating_breakdown") or {}
        for k in set(p_b) | set(c_b):
            delta = c_b.get(k, 0) - p_b.get(k, 0)
            if abs(delta) >= 3:
                sign = "+" if delta > 0 else ""
                if k in ("买入", "增持", "强烈推荐", "推荐") and delta > 0:
                    emoji = "🟢"
                elif k in ("减持", "卖出") and delta > 0:
                    emoji = "🔴"
                else:
                    emoji = "•"
                out.append(
                    f"{emoji} `{sym}` 研报「{k}」{sign}{delta} 篇 ({p_b.get(k, 0)} → {c_b.get(k, 0)})"
                )

    # New rating actions (US: firm upgrades; CN: 新研报; ETF: skip)
    p_set = set(prev.get("recent_actions") or [])
    c_set = set(cur.get("recent_actions") or [])
    new_items = list(c_set - p_set)[:3]
    for act in new_items:
        parts = act.split("|")
        if len(parts) >= 4:
            d, firm, action, info = parts[:4]
            label = "🆕 研报" if market == "cn" else "🆕"
            firm_part = firm or "?"
            info_part = info[:50] if info else ""
            out.append(f"{label} `{sym}` {firm_part} {action} {info_part}".rstrip())
    return out


def run_once(*, dry_run: bool = False) -> int:
    state = _load()
    prev_snaps = state.get("snapshots", {})
    portfolio = cfg_mod.load("portfolio")
    syms = list(portfolio.get("positions", {}).keys()) + [
        w["symbol"] for w in portfolio.get("watchlist", [])
    ]
    new_snaps = {}
    all_changes = []
    for sym in syms:
        try:
            cur = _snapshot_one(sym)
        except Exception:
            log.exception("snapshot failed for %s", sym)
            continue
        if not cur:
            continue
        new_snaps[sym] = cur
        if sym in prev_snaps:
            try:
                changes = diff_snapshot(prev_snaps[sym], cur, sym=sym)
                all_changes.extend(changes)
            except Exception:
                log.exception("diff failed for %s", sym)

    state["snapshots"] = new_snaps
    if not dry_run:
        _save(state)

    if not all_changes:
        log.info("no rating changes")
        return 0

    text = f"📊 *分析师评级变化 — {date.today().isoformat()}*\n\n" + "\n".join(all_changes)
    if dry_run:
        print(text)
        return len(all_changes)

    try:
        telegram.send(text, chat_id=portfolio["telegram_target"])
        log.info("pushed rating changes: %d", len(all_changes))
    except Exception:
        log.exception("push failed")
    return len(all_changes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--init", action="store_true",
                    help="just snapshot current, don't compare (first run)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.init:
        state = _load()
        portfolio = cfg_mod.load("portfolio")
        syms = list(portfolio.get("positions", {}).keys()) + [
            w["symbol"] for w in portfolio.get("watchlist", [])
        ]
        snaps = {}
        for sym in syms:
            try:
                s = _snapshot_one(sym)
            except Exception:
                log.exception("init snapshot failed for %s", sym)
                continue
            if s:
                snaps[sym] = s
        state["snapshots"] = snaps
        _save(state)
        print(f"initialized {len(snaps)} symbol snapshots")
        return

    n = run_once(dry_run=args.dry_run)
    print(f"rating_changes: {n} alerts")


if __name__ == "__main__":
    main()
