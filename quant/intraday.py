"""Intraday alerter: poll prices during US market hours, push alerts on triggers.

Triggers:
  1. INTRADAY_MOVE: abs(chg from open) > config.risk.intraday_move_alert_pct
  2. STOP_LOSS_NEAR: live price < MA200 (held positions only)
  3. WATCH_BUY: watchlist symbol trips a fresh buy signal vs prior close

Dedup: state file keeps {date: 'YYYY-MM-DD', sent: [<sym>:<rule>:<bucket>, ...]}
       — bucket is "5pct"/"7pct"/etc so escalating moves can re-alert.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import signal
import time
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config as cfg_mod
from . import fetcher
from . import telegram as tg

log = logging.getLogger(__name__)

# Market hours in UTC (loose supersets, real fetcher returns empty outside session):
#   US:      13:00 - 21:30 UTC  (Mon-Fri, EDT/EST shift handled by superset)
#   CN A-share: 01:00 - 07:00 UTC (Mon-Fri, 9:30-15:00 北京 with lunch break)

STATE_FILE = cfg_mod.RESULTS_DIR / "intraday_alerts.json"


def _is_us_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(13, 0) <= t <= dtime(21, 30)


def _is_cn_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(1, 0) <= t <= dtime(7, 30)


def _active_markets(now: datetime | None = None) -> set[str]:
    """Return which markets ('US', 'CN') are currently open (loose superset)."""
    now = now or datetime.now(timezone.utc)
    out: set[str] = set()
    if _is_us_market_hours(now):
        out.add("US")
    if _is_cn_market_hours(now):
        out.add("CN")
    return out


def _is_likely_market_hours() -> bool:
    return bool(_active_markets())


def _fetch_us_intraday(s: str) -> dict | None:
    t = yf.Ticker(s)
    df = t.history(period="2d", interval="5m", prepost=False)
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    today_utc = pd.Timestamp.utcnow().date()
    today_df = df[df.index.tz_convert("UTC").date == today_utc]
    if today_df.empty:
        today_df = df.tail(78)  # last day's session
    open_p = float(today_df["Open"].iloc[0])
    last_p = float(today_df["Close"].iloc[-1])
    high_p = float(today_df["High"].max())
    low_p = float(today_df["Low"].min())
    vol = int(today_df["Volume"].sum())
    local = fetcher.load_local(s)
    prev_close = float(local["close"].iloc[-1]) if not local.empty else open_p
    return {
        "symbol": s, "currency": "USD",
        "open": open_p, "last": last_p, "high": high_p, "low": low_p,
        "volume": vol, "prev_close": prev_close,
        "chg_from_open_pct": (last_p / open_p - 1) * 100 if open_p else 0,
        "chg_from_prev_close_pct": (last_p / prev_close - 1) * 100 if prev_close else 0,
    }


def _fetch_cn_intraday(s: str) -> dict | None:
    """A-share intraday minute bars via akshare. Symbol like '002624.SZ'."""
    import akshare as ak
    code = s.split(".")[0]
    try:
        df = ak.stock_zh_a_minute(symbol=("sz" if s.upper().endswith(".SZ") else "sh") + code,
                                  period="5", adjust="qfq")
    except Exception as exc:  # noqa: BLE001
        log.debug("akshare minute fetch failed for %s: %s", s, exc)
        return None
    if df is None or df.empty:
        return None
    # akshare returns: day(time str), open, high, low, close, volume
    df["day"] = pd.to_datetime(df["day"])
    today_local = (datetime.now(timezone.utc) + pd.Timedelta(hours=8)).date()
    today_df = df[df["day"].dt.date == today_local]
    if today_df.empty:
        today_df = df.tail(48)  # last day's session (4h × 12 5min bars)
    open_p = float(today_df["open"].iloc[0])
    last_p = float(today_df["close"].iloc[-1])
    high_p = float(today_df["high"].max())
    low_p = float(today_df["low"].min())
    vol = int(today_df["volume"].astype(float).sum())
    local = fetcher.load_local(s)
    prev_close = float(local["close"].iloc[-1]) if not local.empty else open_p
    return {
        "symbol": s, "currency": "CNY",
        "open": open_p, "last": last_p, "high": high_p, "low": low_p,
        "volume": vol, "prev_close": prev_close,
        "chg_from_open_pct": (last_p / open_p - 1) * 100 if open_p else 0,
        "chg_from_prev_close_pct": (last_p / prev_close - 1) * 100 if prev_close else 0,
    }


def fetch_intraday(symbols: list[str], active_markets: set[str] | None = None) -> dict[str, dict]:
    """Routes US vs CN by symbol suffix; skips symbols whose market is closed."""
    out: dict[str, dict] = {}
    if active_markets is None:
        active_markets = _active_markets()
    for s in symbols:
        is_cn = fetcher.is_a_share(s)
        market = "CN" if is_cn else "US"
        if active_markets and market not in active_markets:
            continue
        try:
            data = _fetch_cn_intraday(s) if is_cn else _fetch_us_intraday(s)
        except Exception as exc:  # noqa: BLE001
            log.debug("intraday fetch failed for %s: %s", s, exc)
            data = None
        if data is None:
            continue
        data["ts"] = datetime.now(timezone.utc).isoformat()
        out[s] = data
    return out


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"date": None, "sent": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _bucket_for_move(pct: float) -> str:
    a = abs(pct)
    if a >= 15:
        return "15pct"
    if a >= 10:
        return "10pct"
    if a >= 7:
        return "7pct"
    if a >= 5:
        return "5pct"
    return "0pct"


def detect_alerts(intraday: dict[str, dict],
                  portfolio: dict,
                  strategies_cfg: dict) -> list[dict]:
    alerts: list[dict] = []

    held = portfolio.get("positions", {})
    watch_syms = {w["symbol"] for w in portfolio.get("watchlist", [])}
    risk = portfolio.get("risk", {})
    move_threshold = risk.get("intraday_move_alert_pct", 0.05) * 100

    for sym, data in intraday.items():
        chg_open = data["chg_from_open_pct"]
        chg_prev = data["chg_from_prev_close_pct"]

        # Rule 1: large intraday move
        if abs(chg_open) >= move_threshold or abs(chg_prev) >= move_threshold:
            big = chg_open if abs(chg_open) > abs(chg_prev) else chg_prev
            alerts.append({
                "symbol": sym,
                "rule": "INTRADAY_MOVE",
                "bucket": _bucket_for_move(big),
                "value_pct": big,
                "data": data,
                "is_held": sym in held,
                "is_watch": sym in watch_syms,
            })

        # Rule 2: stop-loss approach for held positions
        if sym in held:
            local = fetcher.load_local(sym)
            if not local.empty and len(local) >= 200:
                ma200 = float(local["close"].tail(200).mean())
                if data["last"] < ma200 and data["prev_close"] >= ma200:
                    # crossed below ma200 today
                    alerts.append({
                        "symbol": sym,
                        "rule": "BREAK_MA200",
                        "bucket": "below",
                        "ma200": ma200,
                        "data": data,
                        "is_held": True,
                    })

        # Rule 3: watchlist breakout (above prev close + above 20-day high)
        if sym in watch_syms:
            local = fetcher.load_local(sym)
            if not local.empty and len(local) >= 20:
                hi20 = float(local["high"].tail(20).max())
                if data["last"] > hi20 and chg_open > 0:
                    alerts.append({
                        "symbol": sym,
                        "rule": "WATCH_BREAKOUT",
                        "bucket": "above_20d_high",
                        "high20": hi20,
                        "data": data,
                        "is_watch": True,
                    })

    return alerts


def _ccy_sym(c: str) -> str:
    return "¥" if c == "CNY" else "$"


def render(alerts: list[dict]) -> str:
    """One Telegram message for a batch of new alerts."""
    if not alerts:
        return ""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"⚡ *盘中告警* — {today}", ""]

    for a in alerts:
        sym = a["symbol"]
        d = a["data"]
        ccy = d.get("currency", "USD")
        sign = _ccy_sym(ccy)
        if a["rule"] == "INTRADAY_MOVE":
            arrow = "🚀" if a["value_pct"] > 0 else "📉"
            tag = "持仓" if a.get("is_held") else "关注"
            lines.append(
                f"{arrow} `{sym}` ({tag}) {a['value_pct']:+.1f}% — 现价 {sign}{d['last']:.2f} | 开 {sign}{d['open']:.2f} 昨收 {sign}{d['prev_close']:.2f}"
            )
        elif a["rule"] == "BREAK_MA200":
            lines.append(
                f"⚠️ `{sym}` (持仓) 跌破 MA200 — 现价 {sign}{d['last']:.2f} < MA200 {sign}{a['ma200']:.2f}"
            )
        elif a["rule"] == "WATCH_BREAKOUT":
            lines.append(
                f"📈 `{sym}` (关注) 突破 20 日高 — 现价 {sign}{d['last']:.2f} > 20日高 {sign}{a['high20']:.2f} (开盘 +{d['chg_from_open_pct']:.1f}%)"
            )
    return "\n".join(lines)


def run_once(*, dry_run: bool = False) -> int:
    """One poll cycle. Returns number of alerts pushed."""
    portfolio = cfg_mod.load("portfolio")
    strategies_cfg = cfg_mod.load("strategies")
    symbols = cfg_mod.all_symbols(portfolio)

    intraday = fetch_intraday(symbols)
    if not intraday:
        log.info("no intraday data (likely off-hours)")
        return 0

    found = detect_alerts(intraday, portfolio, strategies_cfg)
    if not found:
        log.debug("no triggers from %d symbols", len(intraday))
        return 0

    today_str = date.today().isoformat()
    state = load_state()
    if state.get("date") != today_str:
        state = {"date": today_str, "sent": []}
    sent = set(state["sent"])

    fresh: list[dict] = []
    for a in found:
        key = f"{a['symbol']}:{a['rule']}:{a['bucket']}"
        if key in sent:
            continue
        fresh.append(a)
        sent.add(key)

    if not fresh:
        log.info("all %d triggers already alerted today", len(found))
        return 0

    msg = render(fresh)
    if dry_run:
        print("[DRY-RUN] would push:\n" + msg)
        return len(fresh)

    res = tg.send(msg, chat_id=portfolio["telegram_target"])
    log.info("pushed %d new alerts: telegram message_id=%s",
             len(fresh), res.get("result", {}).get("message_id"))

    state["sent"] = sorted(sent)
    save_state(state)
    return len(fresh)


def loop(*, interval_seconds: int = 300, dry_run: bool = False):
    log.info("intraday loop started, interval=%ds, dry_run=%s", interval_seconds, dry_run)
    stop = {"flag": False}

    def on_signal(*_):
        log.info("stop signal received")
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    while not stop["flag"]:
        try:
            if _is_likely_market_hours():
                run_once(dry_run=dry_run)
            else:
                log.debug("off-hours, skipping")
        except Exception:
            log.exception("intraday cycle failed")
        for _ in range(interval_seconds):
            if stop["flag"]:
                break
            time.sleep(1)
    log.info("intraday loop stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single poll then exit")
    ap.add_argument("--dry-run", action="store_true", help="don't push to Telegram")
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.once:
        n = run_once(dry_run=args.dry_run)
        log.info("single run pushed %d alerts", n)
    else:
        loop(interval_seconds=args.interval, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
