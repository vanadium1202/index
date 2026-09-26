"""Models, walk-forward backtest, accuracy metrics and the live forecast.

Target: log return over h trading days. Point models:
    random_walk  - "tomorrow's price = today's price" (the benchmark that is hard to beat)
    ridge        - regularised linear model on standardised drivers
    gbm          - shallow gradient-boosted trees (non-linear, handles missing drivers)
    ensemble     - mean of ridge and gbm

Uncertainty: volatility-scaled conformal intervals. Residuals from the out-of-sample
backtest are divided by the (EWMA) volatility at forecast time; their empirical quantiles are
re-scaled by today's volatility. Coverage is itself measured out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, build_target

MODEL_NAMES = ["random_walk", "ridge", "gbm", "ensemble"]
QUANTILES = {"p2_5": 0.025, "p10": 0.10, "p50": 0.50, "p90": 0.90, "p97_5": 0.975}


def _ridge():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(0, 5, 21)),
    )


def _gbm(seed: int = 0):
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=3,
        learning_rate=0.03,
        max_iter=250,
        min_samples_leaf=80,
        l2_regularization=1.0,
        random_state=seed,
    )


def fit_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> dict[str, np.ndarray]:
    keep = X_tr.columns[X_tr.notna().mean() > 0.7]
    X_tr, X_te = X_tr[keep], X_te[keep]
    ridge = _ridge().fit(X_tr, y_tr)
    gbm = _gbm().fit(X_tr, y_tr)
    pr, pg = ridge.predict(X_te), gbm.predict(X_te)
    return {
        "random_walk": np.zeros(len(X_te)),
        "ridge": pr,
        "gbm": pg,
        "ensemble": 0.5 * (pr + pg),
    }


def vol_scale(X: pd.DataFrame, h: int) -> pd.Series:
    """Expected h-day return std from EWMA volatility (10-day half-life).

    In the 2018-2026 backtest this gave the same interval coverage as a 21/63-day blend with
    ranges 2-5% narrower, because it reacts faster when a shock starts and fades."""
    return X["vol_ewm10"] * np.sqrt(h / 252)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    h: int,
    test_start: pd.Timestamp,
    refit_every: int = 63,
    min_train: int = 750,
) -> pd.DataFrame:
    """Expanding-window backtest. At each refit date t the model only trains on rows whose
    h-day target was already realised by t (row date <= t - h), so there is no look-ahead."""
    valid = X["vol_63"].notna() & X["ret_126"].notna()
    X, y = X[valid], y[valid]
    dates = X.index
    start = max(int(np.searchsorted(dates, test_start)), min_train + h)
    last_known = y.notna().values.nonzero()[0].max()  # final row whose target is known
    rows = []
    for i in range(start, last_known + 1, refit_every):
        tr_end = i - h  # exclusive: rows [0, i-h) have targets realised strictly before t
        X_tr, y_tr = X.iloc[:tr_end], y.iloc[:tr_end]
        m = y_tr.notna()
        j = min(i + refit_every, last_known + 1)
        preds = fit_predict(X_tr[m], y_tr[m], X.iloc[i:j])
        block = pd.DataFrame(preds, index=dates[i:j])
        block["actual"] = y.iloc[i:j].values
        rows.append(block)
    out = pd.concat(rows)
    out["scale"] = vol_scale(X, h).reindex(out.index)
    return out


def _conformal_bands(bt: pd.DataFrame, model: str, h: int, min_cal: int = 250) -> pd.DataFrame:
    """Out-of-sample intervals for each backtest date, calibrated only on residuals whose
    outcomes were known at that date."""
    z = ((bt["actual"] - bt[model]) / bt["scale"]).values
    n = len(bt)
    lo80 = np.full(n, np.nan)
    hi80 = np.full(n, np.nan)
    lo95 = np.full(n, np.nan)
    hi95 = np.full(n, np.nan)
    for i in range(min_cal + h, n):
        cal = z[: i - h]
        cal = cal[np.isfinite(cal)]
        q = np.quantile(cal, [0.025, 0.10, 0.90, 0.975])
        c, s = bt[model].iat[i], bt["scale"].iat[i]
        lo95[i], lo80[i], hi80[i], hi95[i] = c + q * s
    return pd.DataFrame({"lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95}, index=bt.index)


def _dm_test(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> float:
    """Diebold-Mariano p-value (squared loss, Newey-West variance with h-1 lags).
    Small p-value + lower RMSE => model is significantly better than the benchmark."""
    d = e_model**2 - e_bench**2
    d = d[np.isfinite(d)]
    if len(d) < 30 or np.allclose(d, 0):
        return float("nan")
    dc = d - d.mean()
    var = np.mean(dc**2)
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.mean(dc[k:] * dc[:-k])
    t = d.mean() / np.sqrt(var / len(d))
    return float(2 * stats.norm.sf(abs(t)))


def score(bt: pd.DataFrame, h: int, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Accuracy of every model in the backtest, on the price scale.

    `since` restricts the scored period (e.g. the last 2 years). Interval bands are always
    calibrated on the whole backtest history available at each date, exactly as they would
    have been in live use, and only then restricted to the scored period."""
    bt = bt[bt["actual"].notna()]
    bands_all = {m: _conformal_bands(bt, m, h) for m in MODEL_NAMES}
    if since is not None:
        keep = bt.index >= since
        bt = bt[keep]
        bands_all = {m: b[keep] for m, b in bands_all.items()}
    e_rw = bt["actual"].values
    rmse_rw = np.sqrt(np.mean(e_rw**2))
    out = {}
    for m in MODEL_NAMES:
        pred, act = bt[m].values, bt["actual"].values
        err = act - pred
        pct_err = np.exp(pred - act) - 1  # (forecast price / actual price) - 1
        rmse = np.sqrt(np.mean(err**2))
        moved = np.abs(act) > 1e-4
        bands = bands_all[m]
        ok = bands["lo80"].notna().values
        a = act[ok]
        row = {
            "MAPE_%": 100 * np.mean(np.abs(pct_err)),
            "median_APE_%": 100 * np.median(np.abs(pct_err)),
            "RMSE_%": 100 * rmse,
            "within_5%": 100 * np.mean(np.abs(pct_err) <= 0.05),
            "within_10%": 100 * np.mean(np.abs(pct_err) <= 0.10),
            "direction_hit_%": (
                np.nan if m == "random_walk"
                else 100 * np.mean(np.sign(pred[moved]) == np.sign(act[moved]))
            ),
            "theil_U": rmse / rmse_rw,
            "DM_pvalue_vs_RW": np.nan if m == "random_walk" else _dm_test(err, e_rw, h),
            "coverage80_%": 100 * np.mean((a >= bands["lo80"].values[ok]) & (a <= bands["hi80"].values[ok])),
            "coverage95_%": 100 * np.mean((a >= bands["lo95"].values[ok]) & (a <= bands["hi95"].values[ok])),
            "n_forecasts": len(act),
        }
        out[m] = row
    return pd.DataFrame(out).T


@dataclass
class HorizonResult:
    market: str
    calendar_days: int
    h: int
    backtest: pd.DataFrame
    metrics: pd.DataFrame
    metrics_recent: pd.DataFrame
    chosen: str
    forecast: dict = field(default_factory=dict)


def choose_model(metrics: pd.DataFrame) -> str:
    """Pick the lowest-RMSE model; fall back to the random walk unless the gain is real."""
    best = metrics["RMSE_%"].idxmin()
    if best == "random_walk":
        return best
    beats = metrics.loc[best, "theil_U"] < 0.995 and metrics.loc[best, "DM_pvalue_vs_RW"] < 0.20
    return best if beats else "random_walk"


def run_market(
    df: pd.DataFrame,
    market: str,
    calendar_days: int,
    h: int,
    backtest_years: float = 8,
    refit_every: int = 63,
) -> HorizonResult:
    X = build_features(df, market)
    y = build_target(df, market, h)
    first = df[market].first_valid_index()
    last = df[market].last_valid_index()
    X, y = X.loc[first:last], y.loc[first:last]
    test_start = last - pd.DateOffset(years=backtest_years)
    bt = walk_forward(X, y, h, test_start, refit_every=refit_every)
    metrics = score(bt, h)
    metrics_recent = score(bt, h, since=last - pd.DateOffset(years=2))
    chosen = choose_model(metrics)

    # --- live forecast from the most recent date ---
    valid = X["vol_63"].notna() & X["ret_126"].notna()
    Xv, yv = X[valid], y[valid]
    m = yv.notna()
    preds = fit_predict(Xv[m], yv[m], Xv.iloc[[-1]])
    center = float(preds[chosen][0])
    s_now = float(vol_scale(Xv.iloc[[-1]], h).iloc[0])
    z = ((bt["actual"] - bt[chosen]) / bt["scale"]).dropna().values
    price_now = float(df[market].loc[:last].iloc[-1])
    as_of = Xv.index[-1]
    fc = {
        "market": market,
        "as_of": as_of.date().isoformat(),
        "last_price": price_now,
        "horizon_calendar_days": calendar_days,
        "horizon_trading_days": h,
        "target_date": (as_of + pd.offsets.BDay(h)).date().isoformat(),
        "model": chosen,
        "forecast_price": price_now * np.exp(center),
        "expected_change_%": 100 * (np.exp(center) - 1),
        "prob_up_%": 100 * float(np.mean(center + z * s_now > 0)),
        "annualised_vol_%": 100 * s_now / np.sqrt(h / 252),
        "backtest_MAPE_%": float(metrics.loc[chosen, "MAPE_%"]),
        "backtest_within_5%": float(metrics.loc[chosen, "within_5%"]),
        "backtest_direction_hit_%": float(metrics.loc[chosen, "direction_hit_%"]),
        "backtest_coverage80_%": float(metrics.loc[chosen, "coverage80_%"]),
        "last2y_MAPE_%": float(metrics_recent.loc[chosen, "MAPE_%"]),
        "last2y_coverage80_%": float(metrics_recent.loc[chosen, "coverage80_%"]),
        "all_models_change_%": {k: 100 * (np.exp(float(v[0])) - 1) for k, v in preds.items()},
    }
    for name, q in QUANTILES.items():
        fc[name] = price_now * np.exp(center + np.quantile(z, q) * s_now)
    return HorizonResult(market, calendar_days, h, bt, metrics, metrics_recent, chosen, fc)
