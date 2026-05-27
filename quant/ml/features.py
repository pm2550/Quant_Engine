"""Alpha158 feature set ported to pure pandas/numpy.

Based on Microsoft Qlib's `Alpha158DL.get_feature_config()` — same formulas,
just expressed in pandas instead of Qlib's DSL so we don't carry the qlib
dependency (no aarch64 wheel) and can run on any plain Python stack.

Input: per-symbol price DataFrame with columns
    open, high, low, close, volume (index = DatetimeIndex)
Output: feature DataFrame, same index, columns named per Qlib convention
    (KMID, KLEN, ..., MA5, MA10, ..., CORR60, ...).

All features are price-ratio or unit-free; no z-scoring is done here —
LightGBM is tree-based and doesn't need it.

KBAR (9), PRICE/VWAP (4), VOLUME (1), rolling ops (19 × 5 windows = 95)
                                           = ~109 features (close to "158").
We drop a few that need genuine rolling regression (RSQR/RESI) since they
add ~minutes per symbol and are not in the highest-IC subset per Qlib's
own ablation table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


WINDOWS = (5, 10, 20, 30, 60)


def _ref(s: pd.Series, d: int) -> pd.Series:
    return s.shift(d)


def _kbar_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    hl = h - l + 1e-12
    return {
        "KMID":  (c - o) / o,
        "KLEN":  (h - l) / o,
        "KMID2": (c - o) / hl,
        "KUP":   (h - np.maximum(o, c)) / o,
        "KUP2":  (h - np.maximum(o, c)) / hl,
        "KLOW":  (np.minimum(o, c) - l) / o,
        "KLOW2": (np.minimum(o, c) - l) / hl,
        "KSFT":  (2 * c - h - l) / o,
        "KSFT2": (2 * c - h - l) / hl,
    }


def _price_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    # Default Qlib config: windows=[0], feature=[OPEN, HIGH, LOW, VWAP].
    # We don't have VWAP; use close as proxy (Qlib's loader does the same when missing).
    c = df["close"]
    out = {
        "OPEN0": df["open"] / c,
        "HIGH0": df["high"] / c,
        "LOW0":  df["low"]  / c,
        "VWAP0": c / c,  # always 1; kept for naming compatibility
    }
    return out


def _volume_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    # Default Qlib config doesn't request raw volume at extra windows; only window=0.
    v = df["volume"]
    return {"VOLUME0": v / (v + 1e-12)}  # always 1; included for parity


def _rolling_features(df: pd.DataFrame, windows=WINDOWS) -> dict[str, pd.Series]:
    """All Qlib rolling operators that don't need regression-style apply."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    diff_c = c - _ref(c, 1)
    abs_diff_c = diff_c.abs()
    diff_v = v - _ref(v, 1)
    abs_diff_v = diff_v.abs()
    log_v = np.log(v + 1.0)
    abs_ret_x_v = (c / _ref(c, 1) - 1).abs() * v

    out: dict[str, pd.Series] = {}
    for d in windows:
        rc = c.rolling(d, min_periods=max(2, d // 4))
        rh = h.rolling(d, min_periods=max(2, d // 4))
        rl = l.rolling(d, min_periods=max(2, d // 4))
        rv = v.rolling(d, min_periods=max(2, d // 4))
        r_diff_c = diff_c.rolling(d, min_periods=max(2, d // 4))
        r_abs_diff_c = abs_diff_c.rolling(d, min_periods=max(2, d // 4))
        r_diff_v = diff_v.rolling(d, min_periods=max(2, d // 4))
        r_abs_diff_v = abs_diff_v.rolling(d, min_periods=max(2, d // 4))
        r_abs_ret_x_v = abs_ret_x_v.rolling(d, min_periods=max(2, d // 4))

        out[f"ROC{d}"]  = _ref(c, d) / c
        out[f"MA{d}"]   = rc.mean() / c
        out[f"STD{d}"]  = rc.std() / c
        out[f"MAX{d}"]  = rh.max() / c
        out[f"MIN{d}"]  = rl.min() / c
        out[f"QTLU{d}"] = rc.quantile(0.8) / c
        out[f"QTLD{d}"] = rc.quantile(0.2) / c
        out[f"RANK{d}"] = rc.rank(pct=True)
        out[f"RSV{d}"]  = (c - rl.min()) / (rh.max() - rl.min() + 1e-12)
        out[f"IMAX{d}"] = rh.apply(lambda s: float(np.argmax(s)) if len(s) else np.nan, raw=True) / d
        out[f"IMIN{d}"] = rl.apply(lambda s: float(np.argmin(s)) if len(s) else np.nan, raw=True) / d
        out[f"IMXD{d}"] = (out[f"IMAX{d}"] * d - out[f"IMIN{d}"] * d) / d
        out[f"CORR{d}"] = c.rolling(d).corr(log_v)
        out[f"CORD{d}"] = (c / _ref(c, 1)).rolling(d).corr(np.log(v / _ref(v, 1) + 1.0))

        up = (c > _ref(c, 1)).astype(float)
        dn = (c < _ref(c, 1)).astype(float)
        out[f"CNTP{d}"] = up.rolling(d, min_periods=max(2, d // 4)).mean()
        out[f"CNTN{d}"] = dn.rolling(d, min_periods=max(2, d // 4)).mean()
        out[f"CNTD{d}"] = out[f"CNTP{d}"] - out[f"CNTN{d}"]

        gain_sum = diff_c.clip(lower=0).rolling(d).sum()
        loss_sum = (-diff_c).clip(lower=0).rolling(d).sum()
        abs_sum = r_abs_diff_c.sum() + 1e-12
        out[f"SUMP{d}"] = gain_sum / abs_sum
        out[f"SUMN{d}"] = loss_sum / abs_sum
        out[f"SUMD{d}"] = (gain_sum - loss_sum) / abs_sum

        out[f"VMA{d}"]   = rv.mean() / (v + 1e-12)
        out[f"VSTD{d}"]  = rv.std()  / (v + 1e-12)
        out[f"WVMA{d}"]  = r_abs_ret_x_v.std() / (r_abs_ret_x_v.mean() + 1e-12)
        v_gain_sum = diff_v.clip(lower=0).rolling(d).sum()
        v_loss_sum = (-diff_v).clip(lower=0).rolling(d).sum()
        v_abs_sum = r_abs_diff_v.sum() + 1e-12
        out[f"VSUMP{d}"] = v_gain_sum / v_abs_sum
        out[f"VSUMN{d}"] = v_loss_sum / v_abs_sum
        out[f"VSUMD{d}"] = (v_gain_sum - v_loss_sum) / v_abs_sum

    return out


def build_features(df: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """Compute the full Alpha158-style feature set for one symbol.

    Returns a DataFrame indexed by date, columns = feature names.
    Caller is responsible for handling NaN (early rows) and dropping
    rows where the label can't be computed.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"price df missing columns: {missing}")
    feats: dict[str, pd.Series] = {}
    feats.update(_kbar_features(df))
    feats.update(_price_features(df))
    feats.update(_volume_features(df))
    feats.update(_rolling_features(df, windows=windows))
    out = pd.DataFrame(feats, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def forward_return_label(df: pd.DataFrame, *, horizon_days: int = 20) -> pd.Series:
    """Label = (close[t+h] - close[t+1]) / close[t+1].

    Open-to-open semantics: at day t we make a decision based on close[t],
    we'd execute next day's open ~= close[t+1], and observe close[t+h].
    Returned series is indexed at day t (the decision day).
    """
    c = df["close"]
    fwd = c.shift(-horizon_days)
    entry = c.shift(-1)
    return ((fwd - entry) / entry).rename(f"FWDRET{horizon_days}")
