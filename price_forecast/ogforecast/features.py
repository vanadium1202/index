"""Feature engineering. Every feature at date t uses only data available at close of t."""

from __future__ import annotations

import numpy as np
import pandas as pd

# which futures curve describes each target market
CURVE_FOR = {"brent": "wti", "wti": "wti", "henry_hub": "ng"}

MACRO = ["usd_index", "sp500"]
LEVELS = ["vix", "ovx", "breakeven_10y", "ust_10y"]
INVENTORIES = ["crude_stocks", "gas_storage", "crude_production", "refinery_util"]


def trading_days(calendar_days: int) -> int:
    """7 calendar days ~ 5 trading days, 15 ~ 11."""
    return max(1, int(round(calendar_days * 5 / 7)))


def _rsi(logp: pd.Series, n: int = 14) -> pd.Series:
    d = logp.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return up / (up + dn + 1e-12) - 0.5


def _seasonal_anomaly(s: pd.Series, years: int = 5) -> pd.Series:
    """Deviation (%) from the average of the same calendar week over the prior `years` years."""
    week = s.index.isocalendar().week.astype(int).clip(upper=52)
    year = s.index.year
    out = pd.Series(np.nan, index=s.index)
    frame = pd.DataFrame({"v": s.values, "w": week.values, "y": year}, index=s.index)
    wk = frame.groupby(["y", "w"])["v"].mean()
    for (y, w), grp in frame.groupby(["y", "w"]):
        hist = [wk.get((y - k, w)) for k in range(1, years + 1)]
        hist = [h for h in hist if h is not None and np.isfinite(h)]
        if len(hist) >= 3:
            out.loc[grp.index] = grp["v"] / np.mean(hist) - 1
    return out


def build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return the feature matrix for one target market (rows = dates)."""
    p = np.log(df[target])
    r = p.diff()
    f = {}
    for k in (1, 5, 10, 21, 63, 126):
        f[f"ret_{k}"] = p - p.shift(k)
    f["vol_10"] = r.rolling(10).std() * np.sqrt(252)
    f["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["vol_63"] = r.rolling(63).std() * np.sqrt(252)
    f["vol_ewm10"] = np.sqrt((r**2).ewm(halflife=10).mean() * 252)  # reacts fastest to shocks
    f["vol_ratio"] = f["vol_10"] / f["vol_63"]
    f["gap_ma50"] = p - p.rolling(50).mean()
    f["gap_ma200"] = p - p.rolling(200).mean()
    f["z_252"] = (p - p.rolling(252).mean()) / p.rolling(252).std()
    f["rsi_14"] = _rsi(p)
    f["drawdown_252"] = p - p.rolling(252).max()

    # other energy markets
    for other in ("brent", "wti", "henry_hub"):
        if other != target and other in df:
            po = np.log(df[other])
            f[f"{other}_ret_5"] = po - po.shift(5)
            f[f"{other}_ret_21"] = po - po.shift(21)
    if {"brent", "wti"} <= set(df.columns):
        spread = df["brent"] - df["wti"]
        f["brent_wti_spread"] = spread
        f["brent_wti_spread_chg21"] = spread - spread.shift(21)

    # futures curve: backwardation (>0) signals tight prompt market
    curve = CURVE_FOR.get(target, target)
    c1, c2, c4 = (f"{curve}_c{i}" for i in (1, 2, 4))
    if {c1, c2} <= set(df.columns):
        f["curve_1_2"] = np.log(df[c1] / df[c2])
        f["curve_1_2_chg5"] = f["curve_1_2"] - f["curve_1_2"].shift(5)
    if {c1, c4} <= set(df.columns):
        f["curve_1_4"] = np.log(df[c1] / df[c4])

    for c in MACRO:
        if c in df:
            lc = np.log(df[c])
            f[f"{c}_ret_5"] = lc - lc.shift(5)
            f[f"{c}_ret_21"] = lc - lc.shift(21)
    for c in LEVELS:
        if c in df:
            f[c] = df[c]
            f[f"{c}_chg5"] = df[c] - df[c].shift(5)

    for c in INVENTORIES:
        if c in df and df[c].notna().sum() > 3 * 252:
            f[f"{c}_anom"] = _seasonal_anomaly(df[c])
            f[f"{c}_chg4w"] = np.log(df[c] / df[c].shift(20))

    doy = df.index.dayofyear
    f["season_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=df.index)
    f["season_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=df.index)
    return pd.DataFrame(f, index=df.index)


def build_target(df: pd.DataFrame, target: str, h: int) -> pd.Series:
    """Forward log return over h trading days (NaN where the future is unknown)."""
    p = np.log(df[target])
    return p.shift(-h) - p
