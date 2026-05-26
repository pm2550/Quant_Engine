"""Alt-data anomaly detector — fires TG ping on big trend shifts.

Runs after each snapshot.  Triggers on:

  - 视频总数 7d 跌幅 >= 25%       (社区热度衰减 / 玩家流失早期信号)
  - top20 均播 7d 跌幅 >= 40%      (头部内容观看动能崩塌)
  - sentiment 7d 跌幅 >= 0.4 (绝对值)  (口碑反转)
  - buzz_phase 转入 controversy / declining (社区共识转负)

Cooldown: 同一 (key, signal_type) 24 小时内不重复推, 避免连日噪音。
"""
from __future__ import annotations
import argparse
import json
import logging
from datetime import datetime, timedelta

from .. import db, telegram
from . import bilibili

log = logging.getLogger(__name__)

# Thresholds — tuned conservatively (rather miss than spam)
VOLUME_DROP_7D_PCT = -25.0
PLAYS_DROP_7D_PCT = -40.0
SENTIMENT_DROP_7D_ABS = 0.4
COOLDOWN_HOURS = 24

NEGATIVE_PHASES = {"controversy", "declining"}


def _format_sigma(label: str, val: float | None, threshold: float, *, fmt: str = "+.1f") -> str:
    if val is None:
        return f"  - {label}: 数据不足"
    return f"  - {label}: {val:{fmt}}% (阈值 {threshold:{fmt}}%)"


def _last_alert_for(key: str, signal_type: str) -> datetime | None:
    """Look up the last time we fired this exact alert.

    Reuses `events` table with category='alt_data_anomaly' so we don't add
    a third alerts table.  Could move to a dedicated table later.
    """
    with db.conn() as c:
        row = c.execute(
            "SELECT fired_at FROM events "
            "WHERE category='alt_data_anomaly' "
            "  AND summary LIKE ? "
            "ORDER BY fired_at DESC LIMIT 1",
            (f"[{signal_type}] {key} %",),
        ).fetchone()
    if not row or not row["fired_at"]:
        return None
    try:
        return datetime.fromisoformat(row["fired_at"].replace("Z", ""))
    except Exception:
        return None


def _record_alert(key: str, signal_type: str, summary: str) -> None:
    with db.conn() as c:
        c.execute(
            "INSERT INTO events(severity, category, summary, "
            "  affected_symbols, fired_at, pushed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (6, "alt_data_anomaly",
             f"[{signal_type}] {key} — {summary}",
             "",  # symbol resolution happens via bilibili.DEFAULT_KEYWORDS
             datetime.utcnow().isoformat(),
             datetime.utcnow().isoformat()),
        )


def _within_cooldown(key: str, signal_type: str) -> bool:
    last = _last_alert_for(key, signal_type)
    if not last:
        return False
    return (datetime.utcnow() - last) < timedelta(hours=COOLDOWN_HOURS)


def _resolve_symbol(keyword: str) -> str | None:
    for sym, kws in bilibili.DEFAULT_KEYWORDS.items():
        if keyword in kws:
            return sym
    return None


def check_keyword(keyword: str, *, dry_run: bool = False) -> list[dict]:
    """Check one keyword for anomalies.  Returns list of fired alerts."""
    trend = bilibili.trend(keyword)
    if "error" in trend:
        return []

    # Latest sentiment
    with db.conn() as c:
        rows = c.execute(
            "SELECT metric_date, metrics_json FROM alt_data_metrics "
            "WHERE source='bilibili_search' AND key=? "
            "ORDER BY metric_date DESC LIMIT 8",
            (keyword,),
        ).fetchall()
    parsed = []
    for r in rows:
        try:
            m = json.loads(r["metrics_json"])
            m["_date"] = r["metric_date"]
            parsed.append(m)
        except Exception:
            continue

    fires = []
    sym = _resolve_symbol(keyword)
    sym_label = f"{sym} ({keyword})" if sym else keyword

    # 1. Volume drop
    vol_pct = trend.get("vs_7d_ago", {}).get("total_results_pct")
    if vol_pct is not None and vol_pct <= VOLUME_DROP_7D_PCT:
        sig = "volume_drop"
        if not _within_cooldown(keyword, sig):
            fires.append({
                "signal": sig,
                "summary": f"视频总数 7d {vol_pct:+.1f}% (≤{VOLUME_DROP_7D_PCT}% 阈值) — 社区热度可能衰减",
                "value": vol_pct,
            })

    # 2. Plays drop
    plays_pct = trend.get("vs_7d_ago", {}).get("top_avg_plays_pct")
    if plays_pct is not None and plays_pct <= PLAYS_DROP_7D_PCT:
        sig = "plays_drop"
        if not _within_cooldown(keyword, sig):
            fires.append({
                "signal": sig,
                "summary": f"top20 均播 7d {plays_pct:+.1f}% (≤{PLAYS_DROP_7D_PCT}% 阈值) — 头部观看动能下降",
                "value": plays_pct,
            })

    # 3. Sentiment drop / phase shift
    if len(parsed) >= 2:
        latest_sent = (parsed[0].get("sentiment") or {})
        # Find a comparison snapshot ~7d back
        cmp_idx = min(7, len(parsed) - 1)
        prev_sent = (parsed[cmp_idx].get("sentiment") or {})

        cur_score = latest_sent.get("overall_sentiment")
        prev_score = prev_sent.get("overall_sentiment")
        if isinstance(cur_score, (int, float)) and isinstance(prev_score, (int, float)):
            drop = cur_score - prev_score
            if drop <= -SENTIMENT_DROP_7D_ABS:
                sig = "sentiment_drop"
                if not _within_cooldown(keyword, sig):
                    fires.append({
                        "signal": sig,
                        "summary": f"sentiment {prev_score:+.2f} → {cur_score:+.2f} (Δ{drop:+.2f}, 阈值 ≤-{SENTIMENT_DROP_7D_ABS}) — 口碑反转",
                        "value": cur_score,
                    })

        cur_phase = latest_sent.get("buzz_phase")
        prev_phase = prev_sent.get("buzz_phase")
        if cur_phase in NEGATIVE_PHASES and prev_phase not in NEGATIVE_PHASES and prev_phase:
            sig = "phase_shift"
            if not _within_cooldown(keyword, sig):
                fires.append({
                    "signal": sig,
                    "summary": f"buzz_phase: {prev_phase} → **{cur_phase}** — 社区共识转负",
                    "value": cur_phase,
                })

    if not fires:
        return []

    # Push to Telegram
    if not dry_run:
        for f in fires:
            msg = (
                f"🔔 *Alt-data 异动告警* — {sym_label}\n"
                f"  信号: {f['signal']}\n"
                f"  {f['summary']}\n"
                f"  数据日期: {parsed[0]['_date'] if parsed else '?'}"
            )
            try:
                telegram.send(msg)
            except Exception as e:  # noqa: BLE001
                log.warning("telegram push failed for %s/%s: %s", keyword, f["signal"], e)
            _record_alert(keyword, f["signal"], f["summary"])

    return fires


def check_all(*, dry_run: bool = False) -> dict:
    out = {"ts": datetime.utcnow().isoformat(), "checked": 0, "fired": 0, "alerts": []}
    for sym, kws in bilibili.DEFAULT_KEYWORDS.items():
        for kw in kws:
            out["checked"] += 1
            try:
                fires = check_keyword(kw, dry_run=dry_run)
                for f in fires:
                    out["fired"] += 1
                    out["alerts"].append({"symbol": sym, "keyword": kw, **f})
            except Exception as e:  # noqa: BLE001
                log.warning("check %s failed: %s", kw, e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = check_all(dry_run=args.dry_run)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
