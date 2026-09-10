"""全宇宙价格刷新 —— 让 challenger 的 159 只标的不再停在 2026-05-26。

A1 (2026-09-10) 的根因:
    orchestrator.run() 只 fetch portfolio + watchlist (17 只)。challenger 宇宙里
    其余 140 多只是 opportunity_scanner / universe_discovery 首次发现时抓一次,
    之后再没人刷。结果 159 个 parquet 里:
        2026-09-09 收盘:  16 只
        2026-05-26 冻结:  74 只   ← 近一半
        其余散落 6–8 月:  69 只
    而 quant.ml.predict 对每个 symbol 取 `feats.iloc[-1]` —— 陈旧 parquet 照样能
    产出预测, 只是 as_of 是几个月前。于是"Top 5 看多"榜把不同日期的预测混在一起排序,
    榜首 CELT 的特征日期是 2026-03-20。这是 challenger 实测 IC 为负的主要工程原因。

设计取舍:
    不是"把宇宙缩到能刷新的范围", 而是真的全刷 —— 159 只增量抓取约 3~5 分钟,
    一天一次完全可接受, 没必要牺牲截面宽度 (截面越宽 rank IC 越可信)。
    A 股走 akshare, 美股走 yfinance, 都是增量 (只补最后一根之后的)。

用法:
    python -m quant.price_refresh                # 刷新所有已缓存的 parquet
    python -m quant.price_refresh --stale-only   # 只刷落后 >2 天的 (日常用这个)
    python -m quant.price_refresh --max 50       # 限量, 给首次补齐分批用
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config as cfg_mod
from . import fetcher

log = logging.getLogger(__name__)

PRICES_DIR = Path("/data2/quant/data/prices")
DELISTED_DIR = PRICES_DIR / "_delisted"
QUARANTINE_REGISTRY = DELISTED_DIR / "QUARANTINE.json"
QUARANTINE_AFTER_DAYS = 45   # 连续这么久都拉不到新 bar, 视为退市/改名
SLEEP_BETWEEN = 0.35        # yfinance 礼貌间隔
STALE_AFTER_DAYS = 2        # 周末/假日不算陈旧, 所以给 2 天余量


def cached_symbols() -> list[str]:
    return sorted(p.stem for p in PRICES_DIR.glob("*.parquet"))


def quarantined_symbols() -> dict[str, dict]:
    """已隔离标的登记表 {symbol: {last_bar, quarantined_at, reason}}。

    2026-09-10: 光把 parquet 移进 _delisted/ 不够 —— universe_symbols() 会从
    dynamic_universe.yaml 把它读回来, refresh() 再从 yfinance 拉一份**退市前的**
    历史 (delisted 标的的历史数据仍然可下载), parquet 就复活了, 下次 quarantine
    再移走…… 两个组件互相打架, 无限循环。实测踩到: APGE 被隔离后 3 分钟就复活。
    所以隔离必须是持久化登记, 而不只是移动文件。
    """
    try:
        return json.loads(QUARANTINE_REGISTRY.read_text())
    except Exception:  # noqa: BLE001
        return {}


def unquarantine(symbol: str) -> bool:
    """手动放行 (标的重新上市 / 误判)。返回是否真的移除了登记。"""
    reg = quarantined_symbols()
    if symbol not in reg:
        return False
    reg.pop(symbol)
    QUARANTINE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    QUARANTINE_REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2))
    src = DELISTED_DIR / f"{symbol}.parquet"
    if src.exists():
        src.rename(PRICES_DIR / src.name)
    log.info("解除隔离: %s", symbol)
    return True


def universe_symbols() -> list[str]:
    """已缓存的 + 配置里声明的, 去重。"""
    syms = set(cached_symbols())
    try:
        pf = cfg_mod.load("portfolio") or {}
        syms |= set(cfg_mod.all_symbols(pf))
    except Exception as e:  # noqa: BLE001
        log.warning("portfolio 读取失败: %s", e)
    for name in ("dynamic_universe", "opportunity_universe"):
        try:
            cfg = cfg_mod.load(name) or {}
        except Exception:  # noqa: BLE001
            continue
        for v in cfg.values():
            if isinstance(v, list):
                syms |= {x for x in v if isinstance(x, str)}
            elif isinstance(v, dict):
                syms |= {k for k in v if isinstance(k, str)}
    banned = set(quarantined_symbols())
    if banned:
        dropped = syms & banned
        if dropped:
            log.info("跳过 %d 个已隔离标的: %s", len(dropped), sorted(dropped)[:10])
        syms -= banned
    return sorted(s for s in syms if s and not s.startswith("_"))


def last_bar_date(symbol: str) -> date | None:
    p = PRICES_DIR / f"{symbol}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if df.empty:
            return None
        return pd.to_datetime(df.index).max().date()
    except Exception:  # noqa: BLE001
        return None


def staleness_report() -> dict:
    """当前缓存的新鲜度分布 —— 给 smoke_test / 周报用。"""
    today = date.today()
    buckets: dict[str, int] = {}
    worst: list[tuple[str, str]] = []
    for s in cached_symbols():
        d = last_bar_date(s)
        if d is None:
            buckets["无数据"] = buckets.get("无数据", 0) + 1
            continue
        age = (today - d).days
        key = ("≤2天" if age <= 2 else "3-7天" if age <= 7
               else "8-30天" if age <= 30 else ">30天")
        buckets[key] = buckets.get(key, 0) + 1
        if age > 30:
            worst.append((s, d.isoformat()))
    return {"total": len(cached_symbols()), "buckets": buckets,
            "stale_over_30d": sorted(worst, key=lambda x: x[1])[:20]}


def refresh(*, stale_only: bool = False, max_symbols: int | None = None,
            stale_after_days: int = STALE_AFTER_DAYS) -> dict:
    today = date.today()
    syms = universe_symbols()
    if stale_only:
        syms = [s for s in syms
                if (d := last_bar_date(s)) is None or (today - d).days > stale_after_days]
    if max_symbols:
        syms = syms[:max_symbols]

    ok = failed = unchanged = 0
    advanced: list[str] = []
    t0 = time.time()
    for i, s in enumerate(syms, 1):
        before = last_bar_date(s)
        try:
            df = fetcher.fetch_symbol(s)
            after = pd.to_datetime(df.index).max().date() if df is not None and not df.empty else None
            if after is None:
                failed += 1
            elif before is None or after > before:
                ok += 1
                advanced.append(s)
            else:
                unchanged += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.warning("refresh %s 失败: %s", s, e)
        if i % 25 == 0:
            log.info("进度 %d/%d (前进 %d / 未变 %d / 失败 %d)", i, len(syms), ok, unchanged, failed)
        time.sleep(SLEEP_BETWEEN)

    out = {"requested": len(syms), "advanced": ok, "unchanged": unchanged,
           "failed": failed, "elapsed_s": round(time.time() - t0, 1),
           "advanced_symbols": advanced[:50]}
    log.info("price_refresh 完成: %s", {k: v for k, v in out.items() if k != "advanced_symbols"})
    return out


def quarantine_delisted(*, after_days: int = QUARANTINE_AFTER_DAYS,
                         dry_run: bool = False) -> dict:
    """把长期拉不到新数据的标的移出活跃宇宙。

    为什么必须做: 退市标的的 parquet 会永远停在最后一个交易日, 而 quant.ml.predict
    对每个 symbol 取 `feats.iloc[-1]` —— 于是它照样产出预测并参与排序。实测抓到:
    2026-09-09 日报的 "Top 5 看多" 榜首 CELT (+7.09%) 是一家**已退市**公司,
    特征日期 2026-03-20。APLS / APGE 同理。
    移到 _delisted/ 而不是删除 —— 历史回测和已落库的预测还需要这些价格。
    """
    today = date.today()
    moved, kept = [], []
    for s_ in cached_symbols():
        d = last_bar_date(s_)
        if d is None or (today - d).days <= after_days:
            continue
        src = PRICES_DIR / f"{s_}.parquet"
        if dry_run:
            moved.append((s_, d.isoformat()))
            continue
        try:
            DELISTED_DIR.mkdir(parents=True, exist_ok=True)
            src.rename(DELISTED_DIR / src.name)
            moved.append((s_, d.isoformat()))
            log.warning("隔离疑似退市标的 %s (最后 bar %s)", s_, d)
        except Exception as e:  # noqa: BLE001
            kept.append(s_)
            log.warning("隔离 %s 失败: %s", s_, e)

    if moved and not dry_run:
        # 写持久化登记, 否则下次 refresh 会把它们从配置里读回来重新下载
        reg = quarantined_symbols()
        now = datetime.utcnow().isoformat()
        for sym, last in moved:
            reg[sym] = {"last_bar": last, "quarantined_at": now,
                         "reason": f">{after_days} 天无新数据 (疑似退市/改名)"}
        QUARANTINE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        QUARANTINE_REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2))
        log.warning("隔离登记已更新, 共 %d 个标的; 误判可用 "
                     "`python -m quant.price_refresh --unquarantine SYM` 放行", len(reg))
    return {"quarantined": moved, "failed": kept, "dry_run": dry_run,
            "registry_size": len(quarantined_symbols())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-only", action="store_true", help="只刷落后的")
    ap.add_argument("--max", type=int, help="最多刷几只")
    ap.add_argument("--report", action="store_true", help="只看新鲜度分布, 不刷")
    ap.add_argument("--quarantine", action="store_true",
                     help="把 >45 天没新数据的标的移到 _delisted/")
    ap.add_argument("--dry-run", action="store_true", help="配合 --quarantine 只看不动")
    ap.add_argument("--unquarantine", metavar="SYM", help="解除某标的的隔离 (重新上市/误判)")
    ap.add_argument("--list-quarantined", action="store_true", help="列出已隔离标的")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.report:
        print(json.dumps(staleness_report(), ensure_ascii=False, indent=2))
        return
    if a.list_quarantined:
        print(json.dumps(quarantined_symbols(), ensure_ascii=False, indent=2))
        return
    if a.unquarantine:
        ok = unquarantine(a.unquarantine.upper())
        print(f"{a.unquarantine}: {'已解除隔离' if ok else '不在隔离名单中'}")
        return
    if a.quarantine:
        print(json.dumps(quarantine_delisted(dry_run=a.dry_run),
                         ensure_ascii=False, indent=2))
        return
    print(json.dumps(refresh(stale_only=a.stale_only, max_symbols=a.max),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
