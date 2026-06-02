"""Macro regime overlay — risk-on/off composite, NOT a per-symbol ML feature.

Why this exists separate from quant.ml.macro:
  The 2026-06-01 ablation proved that broadcasting macro values (VIX, yields,
  Fed funds, unemployment, CPI) as cross-sectional features destroys the
  challenger's IC. Reason: same value across all symbols on a given day
  becomes a calendar-period fingerprint and the model memorizes it.

  The industry-standard solution is to use macro as a TOTAL EXPOSURE overlay
  instead of per-symbol alpha — i.e. "today's regime is risk-off so trim
  overall position to 70%" rather than "today macro says NVDA is +1.2%".

What this module returns:
  - A 0-100 risk score (0 = full risk-on, 100 = full risk-off / defensive)
  - Each underlying signal's current reading + percentile
  - A 1-line summary suitable for TG digest
  - A suggested total-exposure adjustment (e.g. "trim to 80%")

Used by daily.py as the single-line regime overlay in the morning digest.
Does NOT feed challenger or 11-factor composite — those operate at the
per-symbol cross-sectional level where macro broadcast is poison.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant.ml import macro as macro_data


log = logging.getLogger(__name__)
CACHE = Path("/data2/quant/data/macro")


def _load(fname: str) -> pd.Series | None:
    p = CACHE / fname
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.iloc[:, 0].sort_index()


def _percentile(s: pd.Series, value: float, *, lookback_days: int = 252 * 3) -> float:
    """Where does `value` rank in the last N days of series? Returns 0.0-1.0."""
    if s is None or s.empty or value is None or np.isnan(value):
        return float("nan")
    recent = s.iloc[-lookback_days:].dropna()
    if recent.empty:
        return float("nan")
    return float((recent < value).sum() / len(recent))


def compute() -> dict:
    """Return a dict with current readings + risk score + summary."""
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    signals: list[dict] = []
    risk_score = 0.0
    weight_sum = 0.0

    def add(*, name: str, value: float, pct: float, risk_contrib: float,
            weight: float, interpretation: str):
        signals.append({
            "name": name,
            "value": value,
            "percentile": pct,
            "risk_contrib": risk_contrib,
            "weight": weight,
            "interpretation": interpretation,
        })

    # ---- VIX (vol regime) ----
    vix_s = _load("vix.parquet")
    if vix_s is not None and not vix_s.empty:
        vix_now = float(vix_s.iloc[-1])
        vix_pct = _percentile(vix_s, vix_now, lookback_days=252 * 3)  # 3y window
        # VIX high → risk-off (positive contribution)
        risk = vix_pct  # 0-1
        add(name="VIX", value=vix_now, pct=vix_pct,
            risk_contrib=risk, weight=1.5,
            interpretation=f"{vix_now:.1f} ({vix_pct*100:.0f}% 分位 / 3y)")
        risk_score += risk * 1.5
        weight_sum += 1.5

    # ---- T10Y2Y yield curve (recession indicator) ----
    curve_s = _load("fred_t10y2y.parquet")
    if curve_s is not None and not curve_s.empty:
        curve_now = float(curve_s.iloc[-1])
        # < 0 = inverted (recession warning); 0 = flat; > 1.0 = normal/healthy
        # Map: inverted = 1.0 risk, +0.5pp = 0.5 risk, +1.5pp = 0 risk
        risk = max(0.0, min(1.0, (0.5 - curve_now) / 1.0 + 0.5))
        add(name="T10Y2Y 期限利差", value=curve_now, pct=float("nan"),
            risk_contrib=risk, weight=1.2,
            interpretation=f"{curve_now:+.2f}pp " +
                ("⚠️ 倒挂" if curve_now < 0 else "正常" if curve_now > 1.0 else "扁平"))
        risk_score += risk * 1.2
        weight_sum += 1.2

    # ---- Fed funds rate (DFF) 90d change ----
    dff_s = _load("fred_dff.parquet")
    if dff_s is not None and not dff_s.empty:
        dff_now = float(dff_s.iloc[-1])
        if len(dff_s) > 90:
            dff_chg = float(dff_s.iloc[-1] - dff_s.iloc[-90])
        else:
            dff_chg = 0.0
        # Rate hikes = risk-off; 50bp hike in 90d = 0.5 risk; 100bp+ = 0.8+
        risk = max(0.0, min(1.0, abs(dff_chg) * 0.8 if dff_chg > 0 else 0.0))
        add(name="Fed funds 90d", value=dff_chg, pct=float("nan"),
            risk_contrib=risk, weight=1.0,
            interpretation=f"当前 {dff_now:.2f}%, 90d 变化 {dff_chg:+.2f}pp")
        risk_score += risk * 1.0
        weight_sum += 1.0

    # ---- Unemployment trend (UNRATE 6m change in pp) ----
    unrate_s = _load("fred_unrate.parquet")
    if unrate_s is not None and not unrate_s.empty:
        un_now = float(unrate_s.iloc[-1])
        if len(unrate_s) > 6:
            un_chg = float(un_now - unrate_s.iloc[-7])  # 6 months
        else:
            un_chg = 0.0
        # Sahm rule: 6m rise of 0.5pp in 3-month avg = recession trigger
        risk = max(0.0, min(1.0, un_chg / 0.5))
        add(name="失业率 6m", value=un_chg, pct=float("nan"),
            risk_contrib=risk, weight=1.5,
            interpretation=f"当前 {un_now:.1f}%, 6m 变化 {un_chg:+.2f}pp" +
                (" ⚠️ Sahm 触发" if un_chg >= 0.5 else ""))
        risk_score += risk * 1.5
        weight_sum += 1.5

    # ---- USDJPY trend (yen carry / risk-off) ----
    jpy_s = _load("usdjpy.parquet")
    if jpy_s is not None and not jpy_s.empty:
        jpy_now = float(jpy_s.iloc[-1])
        if len(jpy_s) > 60:
            jpy_chg60 = float(jpy_now / jpy_s.iloc[-60] - 1)
        else:
            jpy_chg60 = 0.0
        # Yen weakening rapidly (USDJPY up >5% in 60d) = carry trade stress
        risk = max(0.0, min(1.0, abs(jpy_chg60) * 8))  # 5% move → 0.4 risk
        add(name="USDJPY 60d", value=jpy_chg60 * 100, pct=float("nan"),
            risk_contrib=risk, weight=0.8,
            interpretation=f"当前 {jpy_now:.2f}, 60d {jpy_chg60*100:+.1f}%")
        risk_score += risk * 0.8
        weight_sum += 0.8

    # ---- Curve direction over 60d (steepening vs flattening) ----
    if curve_s is not None and len(curve_s) > 60:
        curve_chg60 = float(curve_now - curve_s.iloc[-60])
        risk = max(0.0, min(1.0, -curve_chg60 / 0.5)) if curve_chg60 < 0 else 0.0
        # Flattening fast = recession warning building
        add(name="期限利差 60d 变化", value=curve_chg60, pct=float("nan"),
            risk_contrib=risk, weight=0.5,
            interpretation=f"{curve_chg60:+.2f}pp 60d" +
                ("（扁平化）" if curve_chg60 < -0.2 else ""))
        risk_score += risk * 0.5
        weight_sum += 0.5

    if weight_sum == 0:
        return {"score_0_100": 50.0, "signals": [], "summary": "无 macro 数据"}

    score_0_100 = (risk_score / weight_sum) * 100

    # Suggested exposure: 100% at score 0, 50% at score 100 (linear)
    suggested_exposure_pct = max(50, 100 - score_0_100 * 0.5)

    if score_0_100 < 25:
        band = "🟢 低风险"
    elif score_0_100 < 50:
        band = "🟡 中性"
    elif score_0_100 < 75:
        band = "🟠 偏防御"
    else:
        band = "🔴 风险"

    return {
        "score_0_100": round(score_0_100, 1),
        "band": band,
        "suggested_exposure_pct": round(suggested_exposure_pct),
        "signals": signals,
        "asof": today.date().isoformat(),
    }


def render_section() -> str:
    """One-block TG-friendly markdown for the daily digest."""
    r = compute()
    score = r.get("score_0_100", 50)
    band = r.get("band", "")
    exp = r.get("suggested_exposure_pct", 100)
    signals = r.get("signals", [])
    if not signals:
        return ""

    lines = [
        f"🌐 *宏观风险面板 {band} {score:.0f}/100*",
        f"_建议持仓上限 ~{exp}% (regime overlay, 不替代每只股票决策)_",
        "",
    ]
    for s in signals:
        rc = s["risk_contrib"]
        marker = "🔴" if rc >= 0.6 else "🟠" if rc >= 0.3 else "🟢"
        lines.append(f"  {marker} {s['name']}: {s['interpretation']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_section())
