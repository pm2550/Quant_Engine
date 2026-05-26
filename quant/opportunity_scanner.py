"""Scan a candidate universe for high-conviction NEW opportunities outside the portfolio.

Phase C (2026-05-26): 静态 universe (opportunity_universe.yaml) scanner.
Phase D (2026-05-26): + 动态发现 (dynamic_universe.yaml 来自 universe_discovery) + 跟踪 + 升级路径.

每日 14:00 UTC:
  1. 合并 static + dynamic + tracked 三方候选
  2. 对全部跑 signals + multi_factor (无论是否过阈值, 都更新 tracked_candidates 历史)
  3. composite>=threshold + 不在 cooldown → 今天推送的新机会
  4. tracked 里连续 PROMOTE_WINDOW 天 conviction>=4 → 推 "升级到 watchlist" 提示
  5. 落 reports/opportunities_<date>.md
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from . import (config as cfg_mod, fetcher, multi_factor, signals as signals_mod,
               telegram, universe_discovery, tracked_candidates)

log = logging.getLogger(__name__)

UNIVERSE_FILE = cfg_mod.CONFIG_DIR / "opportunity_universe.yaml"
STATE_FILE = cfg_mod.RESULTS_DIR / "opportunity_scanner_state.json"
DEFAULT_THRESHOLD = 0.30
DEFAULT_TOP_N = 5
DEFAULT_COOLDOWN_HOURS = 24


def _load_universe() -> dict:
    if not UNIVERSE_FILE.exists():
        return {"universe": [], "ignore": {}}
    with UNIVERSE_FILE.open() as f:
        return yaml.safe_load(f) or {"universe": [], "ignore": {}}


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_pushed": {}}
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {"last_pushed": {}}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _portfolio_symbols() -> set[str]:
    try:
        p = cfg_mod.load("portfolio")
    except Exception:
        return set()
    out = set(p.get("positions", {}).keys())
    for w in p.get("watchlist") or []:
        if w.get("symbol"):
            out.add(w["symbol"])
    return out


def _is_ignored(symbol: str, ignore: dict) -> bool:
    """ignore = {symbol: 'YYYY-MM-DD'}; entry is active if today <= date."""
    end = ignore.get(symbol)
    if not end:
        return False
    try:
        end_date = datetime.fromisoformat(str(end)).date()
    except (ValueError, TypeError):
        return False
    return date.today() <= end_date


def _cooldown_active(symbol: str, state: dict, *, hours: int) -> bool:
    last = state.get("last_pushed", {}).get(symbol)
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return False
    return (datetime.utcnow() - last_dt) < timedelta(hours=hours)


def _scan_one(symbol: str, strategies: dict) -> dict | None:
    """Fetch + signals + multi_factor for a single symbol. Returns score dict or None."""
    try:
        df = fetcher.load_local(symbol)
        if df.empty or len(df) < 50:
            # Try fetching fresh (might be a fresh symbol w/o cached parquet)
            df = fetcher.fetch_symbol(symbol)
        if df is None or df.empty or len(df) < 50:
            log.warning("scan %s: insufficient price history", symbol)
            return None
    except Exception:
        log.exception("scan %s: fetch failed", symbol)
        return None

    try:
        sig = signals_mod.compute(symbol, df, strategies)
        if sig is None:
            return None
        sig_dict = sig.as_dict()
        current_price = float(df["close"].iloc[-1])
        result = multi_factor.score(symbol, sig_dict, fundamentals_data=None,
                                     current_price=current_price)
        result["symbol"] = symbol
        result["current_price"] = current_price
        return result
    except Exception:
        log.exception("scan %s: scoring failed", symbol)
        return None


def _build_combined_universe() -> tuple[list[dict], dict[str, list[str]]]:
    """合并 static + dynamic + tracked symbols 进 unified list.

    Returns (entries, sources_per_symbol) where entries = list[{symbol, theme, reason}]
    and sources_per_symbol maps each symbol → list of which sources it came from
    (static / dynamic / tracked).
    """
    static_cfg = _load_universe()
    static_entries = static_cfg.get("universe") or []
    static_syms = {e["symbol"]: e for e in static_entries if e.get("symbol")}

    dynamic_entries = universe_discovery.load_dynamic_universe()
    dynamic_syms = {e["symbol"]: e for e in dynamic_entries if e.get("symbol")}

    tracked_state = tracked_candidates.load_state()
    tracked_syms = tracked_candidates.tracked_symbols(tracked_state)

    sources_per_symbol: dict[str, list[str]] = {}
    combined: dict[str, dict] = {}

    for sym, e in static_syms.items():
        combined[sym] = dict(e)
        sources_per_symbol.setdefault(sym, []).append("static")

    for sym, e in dynamic_syms.items():
        if sym in combined:
            combined[sym]["dynamic_reason"] = e.get("reason")
            combined[sym]["dynamic_sources"] = e.get("sources") or []
        else:
            combined[sym] = {
                "symbol": sym,
                "theme": None,
                "reason": e.get("reason"),
                "dynamic_sources": e.get("sources") or [],
            }
        sources_per_symbol.setdefault(sym, []).append("dynamic")

    for sym in tracked_syms:
        if sym not in combined:
            st = tracked_state.get(sym) or {}
            combined[sym] = {
                "symbol": sym,
                "theme": None,
                "reason": "tracked",
                "tracked_since": st.get("first_added_at"),
            }
        sources_per_symbol.setdefault(sym, []).append("tracked")

    return list(combined.values()), sources_per_symbol


def run_scan(*, threshold: float = DEFAULT_THRESHOLD, top_n: int = DEFAULT_TOP_N,
             cooldown_hours: int = DEFAULT_COOLDOWN_HOURS, dry_run: bool = False,
             push: bool = True) -> dict:
    static_cfg = _load_universe()
    ignore = static_cfg.get("ignore") or {}
    state = _load_state()
    excluded = _portfolio_symbols()
    strategies = cfg_mod.load("strategies")

    universe, sources_per_symbol = _build_combined_universe()
    tracked_state = tracked_candidates.load_state()

    all_scored: list[dict] = []      # 给 tracked_candidates 更新 (全部 score, 不管阈值)
    push_candidates: list[dict] = [] # 今天要推 TG 的新机会 (>= threshold + 不在 cooldown)

    for entry in universe:
        sym = entry.get("symbol")
        if not sym or sym in excluded:
            continue
        if _is_ignored(sym, ignore):
            log.info("skip %s: in ignore list", sym)
            continue

        result = _scan_one(sym, strategies)
        if not result:
            continue
        result["theme"] = entry.get("theme")
        result["reason_static"] = entry.get("reason")
        result["sources"] = sources_per_symbol.get(sym) or []
        all_scored.append(result)

        composite = result.get("composite_score", 0)
        in_cooldown = _cooldown_active(sym, state, hours=cooldown_hours)
        if composite >= threshold and not in_cooldown:
            push_candidates.append(result)

    push_candidates.sort(key=lambda r: r["composite_score"], reverse=True)
    push_candidates = push_candidates[:top_n]

    # ---- Update tracked_candidates state with all_scored, prune expired ----
    if not dry_run:
        tracked_state = tracked_candidates.update_for_scored(tracked_state, all_scored)
        tracked_state, removed = tracked_candidates.prune_expired(tracked_state)
        if removed:
            log.info("pruned %d expired tracked candidates: %s", len(removed), removed)
    else:
        # For dry-run still compute the would-be promotions on a copy
        import copy
        tracked_state = tracked_candidates.update_for_scored(copy.deepcopy(tracked_state), all_scored)

    promotions = tracked_candidates.find_promotion_candidates(tracked_state)

    summary = {
        "scanned_at": datetime.utcnow().isoformat() + "Z",
        "universe_size": len(universe),
        "scored": len(all_scored),
        "n_candidates": len(push_candidates),
        "n_promotions": len(promotions),
        "n_tracked": len(tracked_state),
        "candidates": push_candidates,
        "promotions": promotions,
    }

    if not push_candidates and not promotions:
        log.info("no new opportunities and no promotions (threshold=%s)", threshold)
        if not dry_run:
            tracked_candidates.save_state(tracked_state)
        return summary

    parts = []
    if push_candidates:
        parts.append(_format_md(push_candidates))
    if promotions:
        parts.append(tracked_candidates.format_promotion_message(promotions))
    text = "\n\n".join(parts)

    rpt_dir = cfg_mod.ROOT / "reports"
    rpt_dir.mkdir(parents=True, exist_ok=True)
    rpt_path = rpt_dir / f"opportunities_{date.today().isoformat()}.md"
    rpt_path.write_text(text, encoding="utf-8")
    log.info("wrote %s", rpt_path)

    if dry_run:
        print(text)
        return summary

    # Cooldown + tracked state persist
    now_iso = datetime.utcnow().isoformat()
    for c in push_candidates:
        state["last_pushed"][c["symbol"]] = now_iso
    _save_state(state)
    for p in promotions:
        tracked_candidates.mark_promoted(tracked_state, p["symbol"])
    tracked_candidates.save_state(tracked_state)

    if push:
        try:
            portfolio = cfg_mod.load("portfolio")
            telegram.send(text, chat_id=portfolio["telegram_target"])
            log.info("pushed %d opportunities + %d promotions",
                     len(push_candidates), len(promotions))
        except Exception:
            log.exception("opportunity push failed")

    return summary


_FACTOR_CHINESE: dict[str, str] = {
    "technical": "技术", "events": "事件", "trade_signals": "订单流",
    "sentiment": "情绪", "fundamental": "基本面", "analyst": "分析师",
    "momentum": "动量", "macro_regime": "宏观", "alt_data": "领先指标",
    "rating_change": "评级变化", "event_intensity": "事件烈度",
}


def _format_md(candidates: list[dict]) -> str:
    parts = [f"🆕 *新机会扫描 — {date.today().isoformat()}*",
             f"持仓外 {len(candidates)} 只高 conviction 候选\n"]

    for c in candidates:
        sym = c["symbol"]
        conv = c.get("conviction", 0)
        stars = "★" * conv + "☆" * (5 - conv) if conv > 0 else "-"
        composite = c.get("composite_score", 0)
        theme = c.get("theme")
        reason_static = c.get("reason_static")

        parts.append(f"\n*{sym}* — 信心 {stars} ({conv}/5) | composite {composite:+.2f}"
                     f"{' | ' + theme if theme else ''}")
        if reason_static:
            parts.append(f"  💡 静态理由: {reason_static}")
        parts.append(f"  💲 现价 ${c.get('current_price', 0):.2f}")

        top = c.get("top_factors") or []
        if top:
            parts.append("  📊 关键因子:")
            for tf in top[:3]:
                name_cn = _FACTOR_CHINESE.get(tf.get("name") or "", tf.get("name") or "?")
                contrib = tf.get("contribution", 0)
                ev = tf.get("evidence", "")
                parts.append(f"    • {name_cn} ({contrib:+.2f}): {ev[:80]}")

        counter = c.get("counter_factors") or []
        if counter:
            parts.append("  ⚠️ 反向因子:")
            for cf in counter[:2]:
                name_cn = _FACTOR_CHINESE.get(cf.get("name") or "", cf.get("name") or "?")
                contrib = cf.get("contribution", 0)
                ev = cf.get("evidence", "")
                parts.append(f"    • {name_cn} ({contrib:+.2f}): {ev[:80]}")

    parts.append("\n_主人回复 `/watch SYMBOL` 加入关注池, 或 `/ignore SYMBOL` 忽略 30 天._")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="composite_score 阈值, 默认 0.30")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--cooldown-hours", type=int, default=DEFAULT_COOLDOWN_HOURS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = run_scan(threshold=args.threshold, top_n=args.top_n,
                   cooldown_hours=args.cooldown_hours,
                   dry_run=args.dry_run, push=not args.no_push)
    print(json.dumps({k: v for k, v in out.items() if k != "candidates"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
