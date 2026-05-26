"""RSS 新闻监控 - 轮询所有源, 去重, LLM 评级, 高优先级即时推 TG.

工作流:
  1. 按 sources.yaml 各 feed 的 fetch_interval 轮询
  2. 去重 (news_archive 表 url unique constraint)
  3. 关键词预筛 (keywords_priority) - drop 体育/娱乐
  4. 剩下的丢给 LLM (qwen) 评级 severity 0-10 + classify category
  5. severity ≥ 4 → 调 event_impact 推演影响
  6. severity ≥ 7 → 立即推 Telegram
  7. severity 4-6 → 入 events 表, 等日报汇总

Run: python -m quant.newswatch [--once] [--sources reuters_world,fed_press]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import signal
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests

from . import config as cfg_mod
from . import db, llm_router, telegram, similar_event, prompts, fetcher

log = logging.getLogger(__name__)


# Prompts now live in /data2/quant/prompts/*.md — see quant.prompts module.
# Edit the .md file to change wording; no Python edit needed.


# ---- Keyword pre-filter ----
def _kw_filter(title: str, content: str, kw_cfg: dict) -> tuple[bool, int]:
    """Return (keep, priority_boost). priority_boost adds to LLM's severity later."""
    text = (title + " " + content).lower()
    if any(k.lower() in text for k in kw_cfg.get("ignore", [])):
        return False, 0
    boost = 0
    if any(k.lower() in text for k in kw_cfg.get("high", [])):
        boost = 2
    elif any(k.lower() in text for k in kw_cfg.get("medium", [])):
        boost = 1
    return True, boost


# ---- RSS fetching ----
def _fetch_feed(feed_cfg: dict) -> list[dict]:
    """Fetch one RSS feed, return list of {url, title, content, source, published_at}."""
    url = feed_cfg["url"]
    name = feed_cfg["name"]
    log.debug("fetching %s", name)
    try:
        d = feedparser.parse(url)
    except Exception as e:  # noqa: BLE001
        log.warning("feed %s failed: %s", name, e)
        return []
    items = []
    for entry in d.entries[:30]:
        link = getattr(entry, "link", None)
        title = getattr(entry, "title", None)
        if not link or not title:
            continue
        content = getattr(entry, "summary", "")[:1500]
        pub = getattr(entry, "published", None) or getattr(entry, "updated", None)
        items.append({
            "url": link,
            "title": title,
            "content": content,
            "source": name,
            "weight": feed_cfg.get("weight", 1.0),
            "region": feed_cfg.get("region", "global"),
            "published_at": pub,
        })
    return items


def _dedupe_and_store(items: list[dict]) -> list[tuple[int, dict]]:
    """Insert new items into news_archive, return (id, item) for newly inserted."""
    new_items: list[tuple[int, dict]] = []
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        for it in items:
            raw_hash = hashlib.sha256((it["title"] + it["url"]).encode()).hexdigest()[:16]
            try:
                cur = conn.execute(
                    "INSERT INTO news_archive(url, title, source, published_at, content, raw_hash, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (it["url"], it["title"], it["source"], it.get("published_at"),
                     it["content"], raw_hash, datetime.utcnow().isoformat() + "Z"),
                )
                new_items.append((cur.lastrowid, it))
            except sqlite3.IntegrityError:
                pass  # already seen
        conn.commit()
    return new_items


# ---- LLM scoring ----
def _portfolio_lines(portfolio: dict | None = None) -> str:
    """Render portfolio positions as compact list — used in both severity and impact prompts."""
    portfolio = portfolio if portfolio is not None else cfg_mod.load("portfolio")
    held_lines = []
    for sym, info in portfolio.get("positions", {}).items():
        nm = info.get("name", sym)
        ccy = info.get("currency", "USD")
        held_lines.append(f"  - {sym} ({nm}, {info.get('shares')}股, {ccy})")
    for w in portfolio.get("watchlist", []):
        held_lines.append(f"  - {w['symbol']} (关注池)")
    return "\n".join(held_lines) if held_lines else "  (空)"


def score_severity(item: dict, *, kw_boost: int = 0,
                    portfolio: dict | None = None) -> dict:
    """Severity + category + portfolio relevance from LLM.

    Why portfolio context matters: the same 7/10 macro event affects
    the user's holdings very differently from an unrelated SaaS startup
    headline. Without it, every "big" headline got 7+, drowning out the
    handful that actually move the user's PnL.
    """
    portfolio_str = _portfolio_lines(portfolio)
    user_msg = (
        f"来源: {item['source']} ({item.get('region','?')})\n"
        f"标题: {item['title']}\n"
        f"摘要: {item['content'][:600]}"
    )
    try:
        out = llm_router.chat_json(
            user_msg,
            task="simple_chat",
            system=prompts.load("newswatch_severity").format(portfolio=portfolio_str),
            max_tokens=300,
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("severity scoring failed: %s", e)
        return {"severity": 0, "category": "other", "reasoning": f"err:{e}",
                "portfolio_relevance": "none", "mentioned_holdings": []}
    sev = out.get("severity", 0)
    if isinstance(sev, str):
        try:
            sev = int(sev)
        except ValueError:
            sev = 0
    sev = min(10, max(0, int(sev) + kw_boost))
    return {
        "severity": sev,
        "category": out.get("category", "other"),
        "portfolio_relevance": out.get("portfolio_relevance", "none"),
        "mentioned_holdings": out.get("mentioned_holdings", []) or [],
        "reasoning": out.get("reasoning", ""),
    }


# ---- Per-holding snapshot for impact prompt ----
def _build_snapshots(portfolio: dict) -> str:
    """Render per-holding {price, RSI, MA state, 20d chg, signal codes} as text.

    Why: impact reasoning is much better when the LLM knows e.g. AMD is
    already RSI 75 (overbought) — a "bullish" macro headline shouldn't add
    much further upside. Without it the LLM hallucinates uniform reactions.
    """
    from . import fetcher, signals
    try:
        strategies_cfg = cfg_mod.load("strategies")
    except Exception:
        strategies_cfg = {}
    lines = []
    for sym, info in portfolio.get("positions", {}).items():
        try:
            df = fetcher.load_local(sym)
            if df.empty:
                lines.append(f"  - {sym}: (无本地行情数据)")
                continue
            sig = signals.compute(sym, df, strategies_cfg)
            if sig is None:
                continue
            ma_state = ("MA50↑" if sig.above_ma50 else "MA50↓") + " " + \
                        ("MA200↑" if sig.above_ma200 else "MA200↓")
            tags = ",".join(sig.signal_codes[:3]) if sig.signal_codes else "—"
            lines.append(
                f"  - {sym}: 现价 {sig.price:.2f}, RSI {sig.rsi:.0f}, "
                f"20日 {sig.chg_20d_pct:+.1f}%, {ma_state}, 信号: {tags}"
            )
        except Exception as e:  # noqa: BLE001
            log.debug("snapshot for %s failed: %s", sym, e)
            lines.append(f"  - {sym}: (snapshot 失败)")
    return "\n".join(lines) if lines else "  (无持仓)"


def _build_similar_history(item: dict, severity_info: dict, top_k: int = 3) -> str:
    """Render top-k historically similar events for base-rate context (LLM prompt only)."""
    try:
        sims = similar_event.lookup_for_alert(
            item["title"], severity_info.get("reasoning", ""), top_k=top_k
        )
    except Exception as e:  # noqa: BLE001
        log.debug("similar_event lookup failed: %s", e)
        return "  (相似事件检索不可用)"
    sims = [s for s in sims if s.get("similarity", 0) > 0.55]
    if not sims:
        return "  (无显著相似历史事件)"
    out = []
    for s in sims:
        date = (s.get("fired_at") or "")[:10]
        sev = s.get("severity", "?")
        sim_score = s.get("similarity", 0)
        summary = (s.get("summary") or s.get("text", ""))[:120]
        out.append(f"  - {date} sev={sev} sim={sim_score:.2f} — {summary}")
    return "\n".join(out)


# ---- Cluster de-dup: skip events too similar to one already pushed in last N hours ----
CLUSTER_SIM_THRESHOLD = 0.7
CLUSTER_COOLDOWN_HOURS = 24
HEURISTIC_WINDOW_HOURS = 4
HEURISTIC_OVERLAP_MIN = 0.5


def _find_recent_pushed_cluster(item: dict, severity_info: dict) -> dict | None:
    """Return the most-similar event pushed within last N hours, or None.

    Used to suppress 'see-saw' alerts where the same news cluster (e.g. 美伊冲突)
    gets reported by 8 different feeds with slightly different angles.
    """
    try:
        sims = similar_event.find_similar(
            f"{item['title']}\n{severity_info.get('reasoning','')}",
            top_k=5, min_severity=4,
        )
    except Exception:  # noqa: BLE001
        return None

    cutoff = datetime.utcnow() - timedelta(hours=CLUSTER_COOLDOWN_HOURS)
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        for s in sims:
            if s.get("similarity", 0) < CLUSTER_SIM_THRESHOLD:
                break  # similarities are sorted desc
            row = conn.execute(
                "SELECT pushed_at FROM events WHERE id=? AND pushed_at IS NOT NULL",
                (s["event_id"],),
            ).fetchone()
            if not row:
                continue
            try:
                pushed_dt = datetime.fromisoformat(row["pushed_at"].replace("Z", ""))
            except Exception:
                continue
            if pushed_dt >= cutoff:
                return {"event_id": s["event_id"], "similarity": s["similarity"],
                        "pushed_at": row["pushed_at"], "summary": s.get("summary")}
    return None


def _find_recent_pushed_cluster_heuristic(severity_info: dict, impact: dict,
                                           event_id: int) -> dict | None:
    """Embedding-free fallback dedup: same category + recent + symbol overlap.

    Why: when Gemini embed API is unavailable (key blocked / quota) the
    similarity-based dedup silently returns None and lets duplicates through.
    This catches the obvious clusters (same category, 4h window, ≥50% symbol
    overlap) without needing any LLM call.
    """
    cat = severity_info.get("category")
    if not cat:
        return None
    impacts = impact.get("impacts") or []
    affected = {x.get("symbol") for x in impacts if x.get("symbol")}
    if not affected:
        return None
    cutoff = datetime.utcnow() - timedelta(hours=HEURISTIC_WINDOW_HOURS)
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, affected_symbols, pushed_at, summary FROM events "
            "WHERE pushed_at IS NOT NULL AND category=? AND pushed_at>=? "
            "AND id != ? ORDER BY pushed_at DESC LIMIT 20",
            (cat, cutoff.isoformat() + "Z", event_id),
        ).fetchall()
    for r in rows:
        past = {s for s in (r["affected_symbols"] or "").split(",") if s}
        if not past:
            continue
        overlap = len(past & affected) / max(len(past | affected), 1)
        if overlap >= HEURISTIC_OVERLAP_MIN:
            return {"event_id": r["id"], "similarity": round(overlap, 2),
                    "pushed_at": r["pushed_at"], "summary": r["summary"],
                    "via": "heuristic"}
    return None


# ---- Base rate: real magnitude from historical similar events, not LLM imagination ----
def _compute_base_rate(symbol: str, similar_events: list[dict],
                        windows: tuple[int, ...] = (5, 20)) -> dict | None:
    """For each historical similar event, compute symbol's forward N-day return + max DD.

    Aggregates across events into median / range / n. This is the *real* magnitude
    estimate that replaces what LLM used to write as `magnitude_pct`.

    Returns None if insufficient samples (n < 1).
    """
    if not similar_events:
        return None
    df = fetcher.load_local(symbol)
    if df is None or df.empty:
        return None
    import pandas as pd
    df.index = pd.to_datetime(df.index)
    closes = df["close"].astype(float)

    samples = {f"fwd_{w}d_pct": [] for w in windows}
    samples["max_dd_within_max_window_pct"] = []
    max_window = max(windows)

    for ev in similar_events:
        fired = ev.get("fired_at")
        if not fired:
            continue
        try:
            ev_date = pd.Timestamp(fired[:10])
        except Exception:
            continue
        # Locate trading day at or after event date
        in_range = closes[closes.index >= ev_date]
        if in_range.empty or len(in_range) < max_window + 1:
            continue
        anchor_price = float(in_range.iloc[0])
        # Forward returns
        for w in windows:
            if w >= len(in_range):
                continue
            fwd = (float(in_range.iloc[w]) / anchor_price - 1) * 100
            samples[f"fwd_{w}d_pct"].append(fwd)
        # Max drawdown within max window
        future = in_range.iloc[: max_window + 1]
        if len(future) > 1:
            peak = future.cummax()
            dd = ((future - peak) / peak * 100).min()
            samples["max_dd_within_max_window_pct"].append(float(dd))

    import statistics
    out = {"n_samples": 0}
    for key, vals in samples.items():
        if not vals:
            continue
        out["n_samples"] = max(out["n_samples"], len(vals))
        out[key] = {
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "n": len(vals),
        }
    return out if out["n_samples"] > 0 else None


# ---- Impact推演: LLM 只填 direction; 数字由 base rate 算 ----
def derive_impact(item: dict, severity_info: dict,
                   portfolio: dict | None = None) -> dict:
    """Two-stage impact derivation:

    1. LLM identifies which holdings are involved + direction (bull/bear/neutral) ONLY
       — it does NOT write magnitude. Past LLM-written magnitudes were hallucinations.
    2. For each LLM-flagged symbol, compute *real* base rate from historical similar
       events: forward 5/20-day return median + range + sample size + max drawdown.

    The TG alert shows historical numbers, not LLM imagination.
    """
    portfolio = portfolio if portfolio is not None else cfg_mod.load("portfolio")
    portfolio_str = _portfolio_lines(portfolio)
    snapshots_str = _build_snapshots(portfolio)
    similar_str = _build_similar_history(item, severity_info)

    # --- Stage 1: LLM (direction only, no numbers) ---
    prompt = prompts.load("newswatch_impact").format(
        portfolio=portfolio_str,
        snapshots=snapshots_str,
        similar_history=similar_str,
        title=item["title"],
        source=item["source"],
        content=item["content"][:1500],
        severity=severity_info["severity"],
        category=severity_info["category"],
    )
    try:
        llm_out = llm_router.chat_json(
            prompt, task="format", max_tokens=2000, timeout=300,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("impact LLM failed: %s", e)
        return {"summary": "(impact 推演失败)", "impacts": [],
                "secondary_assets": [], "base_rate_source": None}

    # --- Stage 2: enrich each impact with historical base rate ---
    try:
        similar_events = similar_event.find_similar(
            f"{item['title']}\n{severity_info.get('reasoning','')}",
            top_k=15, min_severity=5,
        )
    except Exception as e:  # noqa: BLE001
        log.debug("similar lookup for base rate failed: %s", e)
        similar_events = []
    similar_events = [s for s in similar_events if s.get("similarity", 0) > 0.55]

    enriched_impacts = []
    for imp in llm_out.get("impacts", []):
        sym = imp.get("symbol")
        if not sym:
            continue
        # Drop any leftover magnitude_pct field — LLM might still emit it
        clean = {k: v for k, v in imp.items() if k != "magnitude_pct"}
        clean["base_rate"] = _compute_base_rate(sym, similar_events)
        enriched_impacts.append(clean)

    return {
        "summary": llm_out.get("summary", ""),
        "impacts": enriched_impacts,
        "secondary_assets": llm_out.get("secondary_assets", []),
        "action_hint": llm_out.get("action_hint") or llm_out.get("action_suggestion"),
        "base_rate_source": {
            "n_similar_events": len(similar_events),
            "min_similarity": (round(min(s["similarity"] for s in similar_events), 3)
                                if similar_events else None),
            "max_similarity": (round(max(s["similarity"] for s in similar_events), 3)
                                if similar_events else None),
        },
    }


# ---- Push & store ----
EMOJI_BY_SEV = {
    range(9, 11): "🚨🚨",
    range(7, 9):  "🚨",
    range(5, 7):  "⚡",
    range(3, 5):  "📰",
}


def _emoji(sev: int) -> str:
    for r, e in EMOJI_BY_SEV.items():
        if sev in r:
            return e
    return "•"


def _direction_emoji(d: str) -> str:
    return {"bullish": "📈", "bearish": "📉", "neutral": "➖"}.get(d, "•")


def render_alert(item: dict, severity_info: dict, impact: dict) -> str:
    sev = severity_info["severity"]
    cat = severity_info["category"]
    lines = [
        f"{_emoji(sev)} *事件等级 {sev}/10 ({cat})*",
        "",
        f"📌 {item['title']}",
        f"_{item['source']} • {item.get('published_at','')[:10]}_",
        "",
    ]
    if impact.get("summary"):
        lines.append(f"💡 {impact['summary']}")
        lines.append("")

    impacts = impact.get("impacts", [])
    if impacts:
        lines.append("*持仓影响 (LLM 方向 + 历史 base rate):*")
        for imp in impacts:
            sym = imp.get("symbol", "?")
            d = imp.get("direction", "neutral")
            conf = imp.get("confidence", 0)
            br = imp.get("base_rate")
            # Header line: direction + confidence + reasoning (no LLM magnitude!)
            lines.append(f"  {_direction_emoji(d)} `{sym}` {d} (置信 {conf:.1f}) — "
                          f"{imp.get('reasoning','')[:100]}")
            # Sub-line: real base rate from similar historical events
            if br and br.get("n_samples", 0) > 0:
                fr5 = br.get("fwd_5d_pct")
                fr20 = br.get("fwd_20d_pct")
                dd = br.get("max_dd_within_max_window_pct")
                pieces = []
                if fr5:
                    pieces.append(f"5d 中位 {fr5['median']:+.1f}% [{fr5['min']:+.1f},{fr5['max']:+.1f}]")
                if fr20:
                    pieces.append(f"20d 中位 {fr20['median']:+.1f}% [{fr20['min']:+.1f},{fr20['max']:+.1f}]")
                if dd:
                    pieces.append(f"20d 最大回撤中位 {dd['median']:.1f}%")
                if pieces:
                    n = max((fr5 or {}).get("n", 0), (fr20 or {}).get("n", 0))
                    lines.append(f"    历史 n={n}: " + " | ".join(pieces))
            else:
                lines.append("    历史: 无足够样本 (n<1) — 仅 LLM 定性判断")
        lines.append("")

    sec = impact.get("secondary_assets", [])
    if sec:
        lines.append("*关联资产 (无历史 base rate):*")
        for s in sec[:3]:
            lines.append(f"  • {s.get('asset','?')}: {s.get('direction','?')} — {s.get('reasoning','')[:80]}")
        lines.append("")

    if impact.get("action_hint"):
        lines.append(f"🎯 *提示:* {impact['action_hint']}")

    # 历史相似事件 (向量检索)
    try:
        sims = similar_event.lookup_for_alert(item["title"], impact.get("summary", ""), top_k=3)
    except Exception:
        sims = []
    sims = [s for s in sims if s["similarity"] > 0.6]  # only meaningful matches
    if sims:
        lines.append("")
        lines.append("*📚 历史相似事件:*")
        for s in sims:
            d = s["fired_at"][:10]
            lines.append(f"  • {d} sev={s['severity']} sim={s['similarity']:.2f} — {(s.get('summary') or s.get('text',''))[:80]}")

    lines.append("")
    lines.append(f"🔗 {item['url'][:120]}")
    return "\n".join(lines)


def _record_event(news_id: int, severity_info: dict, impact: dict, item: dict) -> int:
    affected = ",".join(i.get("symbol", "") for i in impact.get("impacts", []) if i.get("symbol"))
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        cur = conn.execute(
            "INSERT INTO events(news_id, severity, category, summary, impact_json, affected_symbols, fired_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (news_id, severity_info["severity"], severity_info["category"],
             impact.get("summary", item["title"][:200]),
             json.dumps(impact, ensure_ascii=False),
             affected,
             datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
        eid = cur.lastrowid
    # Async (best-effort) index for future similarity search
    try:
        similar_event.index_event(eid)
    except Exception as e:  # noqa: BLE001
        log.debug("auto-index event %d failed: %s", eid, e)
    return eid


def _mark_pushed(event_id: int) -> None:
    with sqlite3.connect(db.DB_PATH, timeout=30) as conn:
        conn.execute(
            "UPDATE events SET pushed_at=? WHERE id=?",
            (datetime.utcnow().isoformat() + "Z", event_id),
        )
        conn.commit()


def process_item(news_id: int, item: dict, kw_cfg: dict, push_threshold: int) -> dict | None:
    """Score one item, derive impact if material, push if high severity."""
    keep, kw_boost = _kw_filter(item["title"], item["content"], kw_cfg)
    if not keep:
        return None
    sev_info = score_severity(item, kw_boost=kw_boost)
    if sev_info["severity"] < 4:
        return None  # ignore low-impact noise

    impact = derive_impact(item, sev_info)
    event_id = _record_event(news_id, sev_info, impact, item)
    log.info("event #%d severity=%d cat=%s: %s",
             event_id, sev_info["severity"], sev_info["category"], item["title"][:80])

    if sev_info["severity"] >= push_threshold:
        # Cluster cooldown: don't see-saw on the same narrative within 24h.
        # Skip push if a similar event (sim >= 0.7) was already pushed in the cooldown window.
        cluster = _find_recent_pushed_cluster(item, sev_info)
        if not cluster:
            cluster = _find_recent_pushed_cluster_heuristic(sev_info, impact, event_id)
        if cluster:
            log.info("event #%d suppressed by cluster cooldown — similar to "
                      "event #%d (sim %.2f via %s) pushed at %s",
                      event_id, cluster["event_id"], cluster["similarity"],
                      cluster.get("via", "embedding"), cluster["pushed_at"])
            return {"event_id": event_id, "severity": sev_info["severity"],
                    "cluster_skipped": cluster}
        try:
            text = render_alert(item, sev_info, impact)
            portfolio = cfg_mod.load("portfolio")
            telegram.send(text, chat_id=portfolio["telegram_target"])
            _mark_pushed(event_id)
            log.info("pushed event #%d to Telegram", event_id)
        except Exception as e:  # noqa: BLE001
            log.exception("push failed: %s", e)
    return {"event_id": event_id, "severity": sev_info["severity"]}


# ---- Main loop ----
def run_once(*, sources_filter: set[str] | None = None, push_threshold: int = 7) -> int:
    sources = cfg_mod.load("sources")
    db.init()
    feeds = sources.get("rss_feeds", [])
    kw_cfg = sources.get("keywords_priority", {})

    all_items: list[dict] = []
    for feed in feeds:
        if sources_filter and feed["name"] not in sources_filter:
            continue
        all_items.extend(_fetch_feed(feed))

    log.info("fetched %d items across %d feeds", len(all_items), len(feeds))
    new_items = _dedupe_and_store(all_items)
    log.info("%d new items after dedup", len(new_items))

    n_events = 0
    for nid, it in new_items:
        try:
            r = process_item(nid, it, kw_cfg, push_threshold=push_threshold)
            if r:
                n_events += 1
        except Exception as e:  # noqa: BLE001
            log.exception("process_item failed for %s: %s", it["url"], e)
    return n_events


def loop(*, interval_seconds: int = 300, push_threshold: int = 7):
    stop = {"flag": False}

    def on_signal(*_):
        log.info("stop signal")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    log.info("newswatch loop started, interval=%ds, push_threshold=%d", interval_seconds, push_threshold)
    while not stop["flag"]:
        try:
            n = run_once(push_threshold=push_threshold)
            log.info("cycle done: %d new events", n)
        except Exception:
            log.exception("cycle failed")
        for _ in range(interval_seconds):
            if stop["flag"]:
                break
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single cycle then exit")
    ap.add_argument("--interval", type=int, default=300, help="loop interval seconds")
    ap.add_argument("--threshold", type=int, default=7, help="severity to push immediately")
    ap.add_argument("--sources", help="comma-separated source names to filter")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    src = set(args.sources.split(",")) if args.sources else None
    if args.once:
        n = run_once(sources_filter=src, push_threshold=args.threshold)
        print(f"newswatch: {n} events processed")
    else:
        loop(interval_seconds=args.interval, push_threshold=args.threshold)


if __name__ == "__main__":
    main()
