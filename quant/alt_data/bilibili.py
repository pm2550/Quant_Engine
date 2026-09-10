"""B 站搜索量 / 二创视频 / 播放量监控 — 中国游戏 / IP 类标的的领先指标.

Why this matters: 完美世界这类 game publisher 的真实流水信号比 iOS 付费榜更早
出现在 B 站. 二创视频数 + 头部播放量是玩家社区粘性的代理. 主流 patterns:

    - 二创量持续上升 + 头部播放量增长 → 玩家粘性强, 流水可期
    - 二创量见顶 + 头部播放量跌 → 热度透支, 流水可能随后掉

We snapshot daily, store in alt_data_metrics table, and let the daily
report / alerts compare 7d/30d trends.
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from datetime import datetime, timezone

import requests

from .. import db, llm_router, prompts

log = logging.getLogger(__name__)

API_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Referer": "https://search.bilibili.com/",
}

# Default keywords to track for known A-share holdings.
# Owner can extend via CLI or by editing this dict.
#
# A5 (2026-09-10): 原来是 4 个词 ["异环","完美世界 异环","诛仙世界","完美新作"],
# 全指向同一只股票, 结果同一天产出 4 个互相矛盾的情绪值 (+0.70/+0.15/−0.20/+0.60),
# 日报里占掉 46 行 (整份报告的 36%)。而且宽松的搜索词会把无关视频卷进来 ——
# "完美新作" 命中过黑神话悟空和"城市大富翁"。
# 收敛到 1 个精确词: 异环 (该公司当前唯一有量的在研产品, 也是股价的主要叙事)。
# 要加词请确认: (a) 对应一个独立的营收/叙事来源, (b) 搜索结果不被无关内容污染。
DEFAULT_KEYWORDS = {
    "002624.SZ": ["异环"],
}

# A4: total_results 是 B 站搜索接口的分页上限 (恒为 1000), 不是真实视频数。
# 以它算出来的 7d/30d 趋势永远是 0.0% —— 这个指标从上线第一天就是死的。
# 活的指标: top_avg_plays (播放量) 和 recent_7d_in_top30 (发布速度)。
TOTAL_RESULTS_CAP = 1000


def _fetch_search(keyword: str, *, page: int = 1, order: str = "pubdate") -> dict:
    """Single call to B 站 search API.  Returns raw JSON's `data` block."""
    r = requests.get(
        API_URL,
        params={"search_type": "video", "keyword": keyword,
                 "order": order, "duration": 0, "page": page},
        headers=HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"bilibili API error: code={body.get('code')} "
                            f"msg={body.get('message')}")
    return body.get("data", {}) or {}


def snapshot_keyword(keyword: str) -> dict:
    """One snapshot of a keyword: total result count + top-N play stats.

    Returns dict written to alt_data_metrics.metrics_json:
      {
        total_results, top_n, top_plays, top_views, top_videos: [{title, play, ...}, ...]
      }
    """
    by_pubdate = _fetch_search(keyword, order="pubdate", page=1)
    time.sleep(0.5)
    by_click = _fetch_search(keyword, order="click", page=1)

    total = by_click.get("numResults") or by_pubdate.get("numResults") or 0
    pubdate_results = by_pubdate.get("result") or []
    click_results = by_click.get("result") or []

    # A7 (2026-09-10): B 站偶发忽略 order=click, 返回按相关度排序的结果 —— 不报错,
    # 只是 top_avg_plays 悄悄变成完全不同的数量级。实测同一天同一关键词:
    #   正常 (按播放降序): top20 均播 5,893,843
    #   降级 (未排序):     top20 均播   433,885   ← 差 13 倍
    # 这个垃圾值随后会让 anomaly.py 触发一次假的 "top20 均播 7d −92.6%" 告警。
    # 所以这里显式校验"是否真的按播放降序", 不可靠就标记出来, 由 trend/anomaly 跳过。
    _plays = [int(v.get("play") or 0) for v in click_results[:20]]
    top_plays_reliable = bool(
        len(_plays) >= 2
        and all(_plays[i] >= _plays[i + 1] for i in range(len(_plays) - 1))
    )
    if not top_plays_reliable and _plays:
        log.warning("bilibili order=click 降级 (%s): 结果未按播放降序, top_avg_plays 不可信",
                     keyword)

    # Top-N most-played (proxy for "is anyone watching this content?")
    top_n = min(20, len(click_results))
    top_videos = []
    top_plays_sum = 0
    for v in click_results[:top_n]:
        play = int(v.get("play") or 0)
        top_plays_sum += play
        title = (v.get("title") or "").replace('<em class="keyword">', "").replace("</em>", "")
        top_videos.append({
            "title": title[:120],
            "author": v.get("author"),
            "play": play,
            "video_review": int(v.get("video_review") or 0),  # 弹幕数
            "favorites": int(v.get("favorites") or 0),
            "pubdate": v.get("pubdate"),
            "duration": v.get("duration"),
            "bvid": v.get("bvid"),
        })

    # Recent activity (UGC velocity): videos in pubdate top-30 with pubdate within 7d
    cutoff_ts = time.time() - 7 * 86400
    recent_7d = sum(1 for v in pubdate_results[:30]
                     if (v.get("pubdate") or 0) >= cutoff_ts)

    total_i = int(total)
    return {
        "total_results": total_i,
        # A4: 撞上分页上限时这个数没有信息量, 下游 (trend / anomaly / formatter)
        # 必须据此跳过基于 total_results 的趋势判断。
        "total_results_capped": total_i >= TOTAL_RESULTS_CAP,
        "top_n": top_n,
        "top_plays_sum": top_plays_sum,
        "top_avg_plays": top_plays_sum // max(top_n, 1),
        "top_plays_reliable": top_plays_reliable,    # A7
        "recent_7d_in_top30": recent_7d,
        "top_videos": top_videos,
        "sentiment": analyze_with_llm(keyword, top_videos),
    }


def analyze_with_llm(keyword: str, top_videos: list[dict]) -> dict:
    """Send top-N video titles to LLM for structured sentiment analysis.

    Returns the structured dict described in prompts/altdata_sentiment.md
    (overall_sentiment / breakdown / key_themes / concern_signals /
    positive_signals / buzz_phase / reasoning).

    On failure returns a sentinel dict so the snapshot still completes —
    sentiment is a bonus signal, not a blocker.
    """
    if not top_videos:
        return {"error": "no videos to analyze"}
    lines = []
    for v in top_videos[:20]:
        play = v.get("play", 0)
        review = v.get("video_review", 0)
        title = (v.get("title") or "").replace("\n", " ")[:120]
        lines.append(f"  [{play:>9,} 播放, {review:>5} 弹幕] {title}")
    prompt = prompts.load("altdata_sentiment").format(
        keyword=keyword,
        top_videos_text="\n".join(lines),
    )
    try:
        # task="format" → ollama:glm-5.1 (json mode), 见 config/llm_routes.yaml。
        # 历史上这里走 dashscope qwen3.6-plus, 该 provider 2026-06-01 随 coding plan
        # 到期下线。thinking-mode 模型会占用输出预算, 所以 max_tokens 留 1500 余量。
        out = llm_router.chat_json(
            prompt, task="format", max_tokens=1500, timeout=300,
        )
        # Normalize fields so downstream consumers don't hit KeyError
        return {
            "overall_sentiment": float(out.get("overall_sentiment", 0)),
            "breakdown": out.get("breakdown") or {},
            "key_themes": out.get("key_themes") or [],
            "concern_signals": out.get("concern_signals") or [],
            "positive_signals": out.get("positive_signals") or [],
            "buzz_phase": out.get("buzz_phase", "unknown"),
            "reasoning": out.get("reasoning", ""),
        }
    except Exception as e:  # noqa: BLE001
        log.warning("LLM sentiment analysis for %s failed: %s", keyword, e)
        return {"error": repr(e)[:200]}


def store_snapshot(keyword: str, metrics: dict, *,
                    source: str = "bilibili_search",
                    metric_date: str | None = None) -> int:
    """Upsert one snapshot row. metric_date defaults to today (UTC)."""
    ts = datetime.now(timezone.utc).isoformat()
    if metric_date is None:
        metric_date = ts[:10]
    with db.conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO alt_data_metrics "
            "(source, key, captured_at, metric_date, metrics_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (source, keyword, ts, metric_date,
             json.dumps(metrics, ensure_ascii=False)),
        )
        new_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]
    return new_id


def trend(keyword: str, *, source: str = "bilibili_search",
            lookback_days: int = 30) -> dict:
    """Compute 7d vs 30d trend on key metrics. Returns deltas in pct."""
    with db.conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT metric_date, metrics_json FROM alt_data_metrics "
            "WHERE source=? AND key=? AND metric_date >= date('now', ?) "
            "ORDER BY metric_date DESC",
            (source, keyword, f"-{int(lookback_days)} days"),
        ).fetchall()]
    if len(rows) < 2:
        return {"keyword": keyword, "n_snapshots": len(rows),
                "error": "need at least 2 snapshots for trend"}
    parsed = []
    for r in rows:
        try:
            m = json.loads(r["metrics_json"])
            parsed.append({"date": r["metric_date"], **m})
        except Exception:
            continue
    if len(parsed) < 2:
        return {"keyword": keyword, "n_snapshots": 0, "error": "no parseable rows"}

    last = parsed[0]
    week_ago = parsed[min(7, len(parsed) - 1)]
    month_ago = parsed[-1]

    def _pct(a, b):
        if not b:
            return None
        return round((a / b - 1) * 100, 2)

    def _capped(rec: dict) -> bool:
        """A4: 历史行没有 total_results_capped 字段, 按数值回推。"""
        v = rec.get("total_results_capped")
        if v is not None:
            return bool(v)
        return int(rec.get("total_results") or 0) >= TOTAL_RESULTS_CAP

    # A4: 任一端撞上分页上限, total_results 的变化率就是假的 —— 返回 None 而不是 0.0%,
    # 让下游能区分"没变化"和"测不出来"。
    cap_now = _capped(last)
    def _total_pct(ref: dict):
        if cap_now or _capped(ref):
            return None
        return _pct(last.get("total_results", 0), ref.get("total_results", 0))

    def _reliable(rec: dict) -> bool:
        """A7: 历史行没有这个字段, 当作可靠 (那时还没有降级检测)。"""
        v = rec.get("top_plays_reliable")
        return True if v is None else bool(v)

    plays_ok_now = _reliable(last)
    def _plays_pct(ref: dict):
        if not plays_ok_now or not _reliable(ref):
            return None
        return _pct(last.get("top_avg_plays", 0), ref.get("top_avg_plays", 0))

    return {
        "keyword": keyword,
        "n_snapshots": len(parsed),
        "total_results_capped": cap_now,
        "top_plays_reliable": plays_ok_now,
        "today": {"date": last["date"], "total_results": last.get("total_results"),
                   "total_results_capped": cap_now,
                   "top_avg_plays": last.get("top_avg_plays"),
                   "recent_7d_in_top30": last.get("recent_7d_in_top30")},
        "vs_7d_ago": {
            "date": week_ago["date"],
            "total_results_pct": _total_pct(week_ago),
            "top_avg_plays_pct": _plays_pct(week_ago),
        },
        "vs_30d_ago": {
            "date": month_ago["date"],
            "total_results_pct": _total_pct(month_ago),
            "top_avg_plays_pct": _plays_pct(month_ago),
        },
    }


def run_all_keywords(keywords: dict[str, list[str]] | None = None) -> dict:
    """Snapshot every (symbol, keyword) pair. Returns summary dict."""
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    results = {}
    for sym, kw_list in keywords.items():
        for kw in kw_list:
            try:
                m = snapshot_keyword(kw)
                store_snapshot(kw, m)
                results[kw] = {
                    "ok": True,
                    "total_results": m["total_results"],
                    "top_avg_plays": m["top_avg_plays"],
                    "recent_7d_in_top30": m["recent_7d_in_top30"],
                }
                log.info("bilibili snapshot %s: total=%d, top_avg_plays=%d",
                          kw, m["total_results"], m["top_avg_plays"])
                time.sleep(1.0)  # be nice to the API
            except Exception as e:  # noqa: BLE001
                log.warning("bilibili snapshot %s failed: %s", kw, e)
                results[kw] = {"ok": False, "error": repr(e)}
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", help="Snapshot a single keyword (overrides defaults)")
    ap.add_argument("--trend", action="store_true",
                     help="Print trend for keyword instead of snapshot")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.trend and args.keyword:
        print(json.dumps(trend(args.keyword), indent=2, ensure_ascii=False))
        return 0
    if args.keyword:
        m = snapshot_keyword(args.keyword)
        store_snapshot(args.keyword, m)
        print(json.dumps({k: v for k, v in m.items() if k != "top_videos"},
                          indent=2, ensure_ascii=False))
        return 0

    out = run_all_keywords()
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # After snapshot, check for anomalies and push TG alerts
    from . import anomaly
    anomaly_out = anomaly.check_all(dry_run=False)
    log.info("anomaly check: %d keywords, %d alerts fired",
              anomaly_out["checked"], anomaly_out["fired"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
