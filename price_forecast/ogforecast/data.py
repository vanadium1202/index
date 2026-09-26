"""Data loading: FRED (no key), EIA API v2 (free key), user CSV, or a synthetic market.

Every loader returns one business-day DataFrame indexed by date. Column names:

    brent, wti, henry_hub              spot prices (targets)
    wti_c1, wti_c2, wti_c4             NYMEX WTI futures, months 1/2/4   (EIA)
    ng_c1, ng_c2                       NYMEX Henry Hub futures, months 1/2 (EIA)
    usd_index, vix, ovx, breakeven_10y, ust_10y, sp500                (FRED)
    crude_stocks, gas_storage, crude_production, refinery_util        (EIA weekly)

Weekly EIA series are shifted to their *publication* date, so the model never sees
an inventory number before the market did.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import requests

FRED_SERIES = {
    "brent": "DCOILBRENTEU",
    "wti": "DCOILWTICO",
    "henry_hub": "DHHNGSP",
    "usd_index": "DTWEXBGS",
    "vix": "VIXCLS",
    "ovx": "OVXCLS",
    "breakeven_10y": "T10YIE",
    "ust_10y": "DGS10",
    "sp500": "SP500",
}

EIA_DAILY = {
    "wti_c1": "PET.RCLC1.D",
    "wti_c2": "PET.RCLC2.D",
    "wti_c4": "PET.RCLC4.D",
    "ng_c1": "NG.RNGC1.D",
    "ng_c2": "NG.RNGC2.D",
}

# (series id, business days between the week-ending date and public release)
EIA_WEEKLY = {
    "crude_stocks": ("PET.WCESTUS1.W", 5),          # week ends Fri, WPSR out Wed
    "crude_production": ("PET.WCRFPUS2.W", 5),
    "refinery_util": ("PET.WPULEUS3.W", 5),
    "gas_storage": ("NG.NW2_EPG0_SWO_R48_BCF.W", 6),  # week ends Fri, WNGSR out Thu
}

WEEKLY_COLUMNS = list(EIA_WEEKLY)

_TIMEOUT = 30


def _fred(series_id: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=".")
    date_col = df.columns[0]  # "DATE" or "observation_date" depending on FRED version
    s = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df[date_col]), dtype=float)
    return s.dropna()


def _eia(series_id: str, api_key: str, start: str) -> pd.Series:
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    resp = requests.get(url, params={"api_key": api_key, "start": start}, timeout=_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()["response"]["data"]
    s = pd.Series(
        [float(r["value"]) if r["value"] is not None else np.nan for r in rows],
        index=pd.to_datetime([r["period"] for r in rows]),
    )
    return s.sort_index().dropna()


def load_live(start: str = "2005-01-01", eia_key: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Download FRED (always) and EIA (if a key is given or EIA_API_KEY is set)."""
    eia_key = eia_key or os.environ.get("EIA_API_KEY")
    cols: dict[str, pd.Series] = {}
    for name, sid in FRED_SERIES.items():
        try:
            cols[name] = _fred(sid, start)
        except Exception as exc:  # one missing driver should not kill the run
            if verbose:
                print(f"  ! FRED {sid} ({name}) skipped: {exc}")
    if eia_key:
        for name, sid in EIA_DAILY.items():
            try:
                cols[name] = _eia(sid, eia_key, start)
            except Exception as exc:
                if verbose:
                    print(f"  ! EIA {sid} ({name}) skipped: {exc}")
        for name, (sid, lag) in EIA_WEEKLY.items():
            try:
                s = _eia(sid, eia_key, start)
                s.index = s.index + pd.offsets.BDay(lag)
                cols[name] = s
            except Exception as exc:
                if verbose:
                    print(f"  ! EIA {sid} ({name}) skipped: {exc}")
    elif verbose:
        print("  (no EIA key: futures-curve and inventory drivers disabled; "
              "get a free key at https://www.eia.gov/opendata/register.php)")
    if not cols:
        raise RuntimeError("No data could be downloaded - check network access to fred.stlouisfed.org")
    return to_business_days(pd.DataFrame(cols))


GITHUB_MIRRORS = {
    "brent": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv",
    "wti": "https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv",
    "henry_hub": "https://raw.githubusercontent.com/datasets/natural-gas/main/data/daily.csv",
}


def load_github(start: str = "2005-01-01") -> pd.DataFrame:
    """EIA daily spot prices mirrored by github.com/datasets (no key; prices only, no drivers)."""
    cols = {}
    for name, url in GITHUB_MIRRORS.items():
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        s = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")["Price"]
        cols[name] = s.loc[start:]
    return to_business_days(pd.DataFrame(cols))


def load_csv(path: str) -> pd.DataFrame:
    """Load a user CSV: first column = date, other columns named as in the module docstring.

    Extra columns (e.g. ttf, jkm, dubai, indian_apm_gas) are allowed and can be used as targets.
    Weekly columns must already be dated by their release date.
    """
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df.iloc[:, 0])
    df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    return to_business_days(df)


def to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """Align to a business-day calendar and forward-fill (holidays, weekly data).

    Forward-fill is limited so a dead series does not silently look like a flat market.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # log-price models cannot use non-positive prints (e.g. WTI at -37.63 on 2020-04-20)
    df = df.mask(df <= 0)
    idx = pd.bdate_range(df.index.min(), df.index.max())
    out = df.reindex(df.index.union(idx)).sort_index()
    limits = {c: 10 if c in WEEKLY_COLUMNS else 5 for c in out.columns}
    for c, lim in limits.items():
        out[c] = out[c].ffill(limit=lim)
    return out.reindex(idx)


def load_synthetic(start: str = "2008-01-01", end: str | None = None, seed: int = 7) -> pd.DataFrame:
    """A synthetic but structurally realistic market, used for offline testing only.

    Features: regime-switching volatility, mean reversion to a drifting equilibrium,
    inventory feedback on price, curve slope tied to inventories, a Brent-WTI spread,
    seasonal gas with storage feedback, and correlated macro drivers.
    Accuracy numbers produced on this data say nothing about real markets.
    """
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    idx = pd.bdate_range(start, end)
    n = len(idx)
    doy = idx.dayofyear.values

    # volatility regime: 0 calm, 1 stressed
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        p_switch = 0.004 if regime[t - 1] == 0 else 0.02
        regime[t] = 1 - regime[t - 1] if rng.random() < p_switch else regime[t - 1]
    oil_vol = np.where(regime == 1, 0.55, 0.28) / np.sqrt(252)
    gas_vol = np.where(regime == 1, 0.80, 0.50) / np.sqrt(252)

    macro = rng.standard_normal((n, 3))  # usd, equity, rates shocks
    def slow_ar(level, sd):  # equilibrium wanders for years but stays anchored
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.999 * x[t - 1] + rng.normal(0, sd)
        return np.log(level) + x

    eq_oil = slow_ar(75, 0.010)
    eq_gas = slow_ar(3.5, 0.010)

    lb = np.empty(n)
    lh = np.empty(n)
    inv = np.empty(n)
    stor = np.empty(n)
    lb[0], lh[0], inv[0], stor[0] = eq_oil[0], eq_gas[0], 0.0, 0.0
    for t in range(1, n):
        # inventory surplus (in % of normal) mean-reverts and grows when price is high
        inv[t] = 0.995 * inv[t - 1] + 0.02 * (lb[t - 1] - eq_oil[t - 1]) + rng.normal(0, 0.15)
        stor[t] = 0.99 * stor[t - 1] + 0.03 * (lh[t - 1] - eq_gas[t - 1]) + rng.normal(0, 0.4)
        shock_o = 0.6 * rng.standard_normal() - 0.25 * macro[t, 0] + 0.3 * macro[t, 1]
        shock_g = 0.8 * rng.standard_normal() + 0.25 * shock_o
        lb[t] = (lb[t - 1] + 0.004 * (eq_oil[t - 1] - lb[t - 1]) - 0.0004 * inv[t - 1]
                 + oil_vol[t] * shock_o)
        lh[t] = (lh[t - 1] + 0.008 * (eq_gas[t - 1] - lh[t - 1]) - 0.0003 * stor[t - 1]
                 + gas_vol[t] * shock_g)
    season = 0.12 * np.cos(2 * np.pi * (doy - 15) / 365.25)  # winter premium

    spread = np.empty(n)
    spread[0] = 4.0
    for t in range(1, n):
        spread[t] = spread[t - 1] + 0.02 * (4.0 - spread[t - 1]) + rng.normal(0, 0.25)

    brent = np.exp(lb)
    wti = brent - spread
    hh = np.exp(lh + season)
    slope = -0.004 * inv + rng.normal(0, 0.003, n)  # stocks high -> contango
    df = pd.DataFrame(
        {
            "brent": brent,
            "wti": wti,
            "henry_hub": hh,
            "wti_c1": wti * (1 + rng.normal(0, 0.002, n)),
            "wti_c2": wti * (1 - slope),
            "wti_c4": wti * (1 - 3 * slope),
            "ng_c1": hh * (1 + rng.normal(0, 0.004, n)),
            "ng_c2": hh * (1 - 0.005 * stor / 5 + rng.normal(0, 0.004, n)),
            "usd_index": 110 * np.exp(np.cumsum(0.004 * macro[:, 0])),
            "sp500": 2000 * np.exp(np.cumsum(0.0003 + 0.011 * macro[:, 1])),
            "vix": 15 + 12 * regime + np.abs(rng.normal(0, 3, n)),
            "ovx": 100 * np.sqrt(252) * oil_vol + rng.normal(0, 2, n),
            "breakeven_10y": 2.2 + np.cumsum(0.01 * macro[:, 2]) * 0.2,
            "ust_10y": 3.0 + np.cumsum(0.03 * macro[:, 2]) * 0.2,
            "crude_stocks": 440_000 * (1 + inv / 100) * (1 + 0.02 * np.sin(2 * np.pi * doy / 365.25)),
            "gas_storage": 2800 * (1 + stor / 100) + 900 * np.sin(2 * np.pi * (doy - 100) / 365.25),
        },
        index=idx,
    )
    # weekly series: keep Wednesday/Thursday releases only, then forward-fill
    weekly_mask = {"crude_stocks": 2, "gas_storage": 3}
    for col, wd in weekly_mask.items():
        df.loc[df.index.dayofweek != wd, col] = np.nan
    return to_business_days(df)
