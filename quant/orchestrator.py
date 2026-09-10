"""Daily orchestrator: fetch → compute signals → generate recommendations → write JSON."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path


def _upcoming_earnings_for_report(*, days: int = 7) -> list[dict]:
    """Earnings dates in next N days for held + watchlist."""
    try:
        from . import earnings_alerter
        return earnings_alerter.upcoming_earnings(days=days)
    except Exception:
        return []


def _important_dates_for_report(*, days: int = 7) -> dict:
    """Aggregated upcoming events: earnings + corp + macro."""
    try:
        from . import important_dates
        return important_dates.aggregate(days=days)
    except Exception:
        return {"earnings": [], "corporate": [], "macro": []}


def _recent_audio_highlights(*, hours: int = 24, min_importance: int = 4) -> list[dict]:
    """Pull recent done audio_queue items for daily report appendix."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
    out: list[dict] = []
    try:
        with sqlite3.connect(db_mod.DB_PATH, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT title, source, summary, impact_json, finished_at
                FROM audio_queue
                WHERE status='done' AND finished_at >= ?
                ORDER BY finished_at DESC LIMIT 8""",
                (cutoff,),
            ).fetchall()
        for r in rows:
            try:
                imp = json.loads(r["impact_json"]) if r["impact_json"] else {}
            except json.JSONDecodeError:
                imp = {}
            importance = imp.get("importance", 0) or 0
            if importance < min_importance:
                continue
            out.append({
                "title": r["title"],
                "source": r["source"],
                "summary": (imp.get("summary") or r["summary"] or "")[:300],
                "tone": imp.get("tone"),
                "importance": importance,
                "key_quotes": (imp.get("key_quotes") or [])[:2],
                "impacts": [
                    {"symbol": i.get("symbol"),
                     "direction": i.get("direction"),
                     "magnitude_pct": i.get("magnitude_pct")}
                    for i in (imp.get("impacts") or [])[:5]
                ],
            })
    except Exception:
        pass
    return out

import sqlite3
from . import config as cfg_mod
from . import fetcher, signals, recommender, fundamentals, multi_factor, db as db_mod

log = logging.getLogger(__name__)


# ---- 跨币种换算 (E3, 2026-09-10) -------------------------------------------
# 之前权重是"持仓市值 / 同币种总市值", 而 002624 是唯一 CNY 持仓 → 它的权重恒为
# 100%, 于是风险检查每天都报"单股权重 100.0% 超过 30% 上限"。那是算法产物, 不是
# 真实集中度。同币种权重对"同币种内怎么调仓"仍然正确, 所以两个都保留:
#   weights        — 按币种桶 (用于 target_weight / delta_shares 计算)
#   weights_global — 全部折成 USD 后的真实组合占比 (用于风险/集中度检查)
FX_FILES = {"CNY": "usdcny.parquet", "JPY": "usdjpy.parquet"}
FX_FALLBACK = {"USD": 1.0, "CNY": 1.0 / 7.1, "JPY": 1.0 / 155.0}


def fx_to_usd(currency: str) -> float:
    """1 单位该币种值多少美元。取不到行情时回落到 FX_FALLBACK 并告警。"""
    cur = (currency or "USD").upper()
    if cur == "USD":
        return 1.0
    fname = FX_FILES.get(cur)
    if fname:
        p = Path("/data2/quant/data/macro") / fname
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                col = "close" if "close" in df.columns else df.columns[0]
                rate = float(df[col].dropna().iloc[-1])   # 该行情是 USD→外币 的报价
                if rate > 0:
                    return 1.0 / rate
            except Exception as e:  # noqa: BLE001
                log.warning("fx_to_usd(%s) 读取 %s 失败: %s", cur, fname, e)
    fb = FX_FALLBACK.get(cur)
    if fb is None:
        log.warning("fx_to_usd(%s): 无汇率数据且无回落值, 按 1.0 处理", cur)
        return 1.0
    log.warning("fx_to_usd(%s): 用回落汇率 %.4f", cur, fb)
    return fb


def run(*, full_refresh: bool = False) -> dict:
    portfolio = cfg_mod.load("portfolio")
    strategies = cfg_mod.load("strategies")

    symbols = cfg_mod.all_symbols(portfolio)
    log.info("symbols: %s", symbols)

    raw = fetcher.fetch_all(symbols, full_refresh=full_refresh)

    sigs: dict[str, signals.SymbolSignals] = {}
    for sym in symbols:
        df = raw.get(sym)
        sig = signals.compute(sym, df, strategies)
        if sig:
            sigs[sym] = sig

    fundamentals_data = {sym: fundamentals.latest(sym) for sym in sigs}

    multi_scores: dict[str, dict] = {}
    for sym, sig in sigs.items():
        try:
            multi_scores[sym] = multi_factor.score(
                sym,
                sig.as_dict(),
                fundamentals_data.get(sym),
                sig.price,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("multi_factor failed for %s: %s", sym, e)
            multi_scores[sym] = {
                "composite_score": 0.0,
                "action": "HOLD",
                "rationale": "multi_factor failed; default HOLD",
                "catalyst_imminent": False,
                "factor_breakdown": {},
            }

    # Per-currency totals (we don't FX-convert; show ¥ and $ separately)
    held = portfolio.get("positions", {})
    watch_info = {w.get("symbol"): w for w in portfolio.get("watchlist", [])}
    default_ccy = portfolio.get("default_currency", "USD")

    def ccy_of(sym: str) -> str:
        return (held.get(sym) or watch_info.get(sym) or {}).get("currency", default_ccy)

    market_values = {
        sym: sigs[sym].price * held[sym]["shares"]
        for sym in held if sym in sigs
    }
    # Group market values by currency
    totals_by_ccy: dict[str, float] = {}
    for sym, mv in market_values.items():
        c = ccy_of(sym)
        totals_by_ccy[c] = totals_by_ccy.get(c, 0.0) + mv

    # Weight = position value / its-currency total (同币种内调仓用这个)
    weights = {
        sym: (mv / totals_by_ccy[ccy_of(sym)] if totals_by_ccy.get(ccy_of(sym)) else 0.0)
        for sym, mv in market_values.items()
    }

    # 跨币种真实权重: 全部折成 USD 再算占比 (风险/集中度检查用这个) —— E3
    mv_usd = {sym: mv * fx_to_usd(ccy_of(sym)) for sym, mv in market_values.items()}
    total_usd = sum(mv_usd.values())
    weights_global = {
        sym: (v / total_usd if total_usd else 0.0) for sym, v in mv_usd.items()
    }

    # Daily change per currency bucket
    chg_by_ccy: dict[str, float] = {}
    for sym in weights:
        c = ccy_of(sym)
        chg_by_ccy[c] = chg_by_ccy.get(c, 0.0) + weights[sym] * sigs[sym].chg_1d_pct

    # Data date - might be older than today if weekend/holiday
    data_dates = {sigs[s].last_date for s in sigs}
    latest_data_date = max(data_dates) if data_dates else None

    def enrich(rec_dict: dict, sym: str) -> dict:
        """Add target_value, delta_value, delta_shares (in symbol's own currency)."""
        c = ccy_of(sym)
        ccy_total = totals_by_ccy.get(c, 0.0)
        cur_w = rec_dict["current_weight"]
        tgt_w = rec_dict["target_weight"]
        is_held = sym in held
        cur_val = market_values.get(sym, cur_w * ccy_total) if is_held else 0.0
        tgt_val = tgt_w * ccy_total
        delta_val = tgt_val - cur_val
        price = sigs[sym].price if sym in sigs else None
        delta_shares = (delta_val / price) if price and price > 0 else 0.0
        if not is_held and rec_dict["action"] != "WATCH_BUY":
            cur_val = tgt_val = delta_val = delta_shares = 0.0
        rec_dict["currency"] = c
        # Friendly display name: held positions have name; watchlist may have it too.
        info = held.get(sym) or next(
            (w for w in portfolio.get("watchlist", []) if w.get("symbol") == sym), {}
        )
        rec_dict["display_name"] = info.get("name") or sym
        rec_dict["current_value"] = round(cur_val, 2)
        rec_dict["target_value"] = round(tgt_val, 2)
        rec_dict["delta_value"] = round(delta_val, 2)
        rec_dict["delta_shares"] = round(delta_shares, 4)
        if c == "USD":
            rec_dict["current_value_usd"] = round(cur_val, 2)
            rec_dict["target_value_usd"] = round(tgt_val, 2)
            rec_dict["delta_usd"] = round(delta_val, 2)
        return rec_dict

    recs: list[dict] = []
    for sym in held:
        if sym not in sigs:
            continue
        rec = recommender.for_held_multi_factor(
            sigs[sym],
            multi_scores.get(sym, {}),
            current_weight=weights.get(sym, 0.0),
            total_value=totals_by_ccy.get(ccy_of(sym), 0.0),
        )
        recs.append(enrich(recommender.to_dict(rec), sym))

    watch_syms = [w["symbol"] for w in portfolio.get("watchlist", [])]
    for sym in watch_syms:
        if sym in held or sym not in sigs:
            continue
        rec = recommender.for_watch_multi_factor(sigs[sym], multi_scores.get(sym, {}))
        recs.append(enrich(recommender.to_dict(rec), sym))

    # 横截面闸门 (C9): 同一天只让 composite 最高的 N 个保留 ADD。composite 的逐日
    # 截面 rank IC = +0.436 但时序 IC ≈ 0 —— 板块同涨时 11 只一起亮 ADD 不是 11 个
    # 独立信号。超出上限的降级为 WATCH_BUY。
    recs = multi_factor.apply_cross_sectional_gate(recs)

    # Concentration risk (now measured on cross-currency weights — see E3)
    def _name_of(sym: str) -> str:
        # 美股直接用代号 (主人都认识 VOO/AMD/...). A 股用 "中文名 (六位代号)"
        if fetcher.is_a_share(sym):
            info = held.get(sym, {})
            nm = info.get("name") or sym
            return f"{nm} ({sym.split('.')[0]})"
        return sym

    risk_notes: list[str] = []
    risk_cfg = portfolio.get("risk", {})
    cap = risk_cfg.get("position_concentration_max", 0.30)
    # E3: 用跨币种权重, 否则唯一的 CNY 持仓永远是"100%"
    for sym, w in weights_global.items():
        if w > cap:
            risk_notes.append(
                f"{_name_of(sym)} 单股权重 {w*100:.1f}% 超过 {cap*100:.0f}% 上限"
            )

    overlap = portfolio.get("sector_overlap", {})
    for sector, members in overlap.items():
        sw = sum(weights_global.get(m, 0.0) for m in members)
        if sw > 0.5:
            risk_notes.append(
                f"{sector} 板块合计 {sw*100:.1f}%（{', '.join(_name_of(m) for m in members)}）"
            )

    # Per-currency portfolio summary
    portfolio_buckets = {
        c: {
            "total": round(t, 2),
            "chg_1d_pct": round(chg_by_ccy.get(c, 0.0), 2),
        }
        for c, t in totals_by_ccy.items()
    }

    # Data freshness — flag if any held symbol has stale data
    now = datetime.utcnow()
    staleness_per_symbol = {}
    max_stale_days = 0
    for sym in sigs:
        try:
            last_d = datetime.fromisoformat(sigs[sym].last_date)
            days = (now - last_d).days
            staleness_per_symbol[sym] = days
            max_stale_days = max(max_stale_days, days)
        except (ValueError, TypeError):
            pass
    freshness_status = "fresh" if max_stale_days <= 1 else "stale" if max_stale_days <= 5 else "very_stale"

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_date": latest_data_date,
        "freshness": {
            "status": freshness_status,
            "max_stale_days": max_stale_days,
            "per_symbol_stale_days": staleness_per_symbol,
            "note": "all prices are end-of-day close. For real-time use ad-hoc /api/analyze (includes after-hours)",
        },
        "portfolio": {
            "by_currency": portfolio_buckets,
            "total_value_usd": round(totals_by_ccy.get("USD", 0.0), 2),
            "chg_1d_pct": round(chg_by_ccy.get("USD", 0.0), 2),
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "weights_global": {k: round(v, 4) for k, v in weights_global.items()},
            "market_values": {k: round(v, 2) for k, v in market_values.items()},
            "currencies": {sym: ccy_of(sym) for sym in market_values},
        },
        "signals": {sym: s.as_dict() for sym, s in sigs.items()},
        "multi_factor": multi_scores,
        "fundamentals": {sym: f for sym, f in fundamentals_data.items() if f},
        "recommendations": recs,
        "risk_notes": risk_notes,
        "overnight_audio": _recent_audio_highlights(),
        "earnings_this_week": _upcoming_earnings_for_report(),
        "important_dates": _important_dates_for_report(),
    }

    cfg_mod.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_local = cfg_mod.RESULTS_DIR / "latest.json"
    dated = cfg_mod.RESULTS_DIR / f"signals-{datetime.utcnow().strftime('%Y%m%d')}.json"
    for p in (out_local, dated):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("wrote %s", out_local)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="full refresh of price history")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out = run(full_refresh=args.refresh)
    print(json.dumps(out, indent=2, ensure_ascii=False))
