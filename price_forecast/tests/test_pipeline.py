import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogforecast.data import load_synthetic  # noqa: E402
from ogforecast.features import build_features, build_target, trading_days  # noqa: E402
from ogforecast.model import run_market, walk_forward  # noqa: E402

DF = load_synthetic("2012-01-01", "2020-12-31", seed=3)


def test_features_do_not_look_ahead():
    """Changing prices after date t must not change any feature at or before t."""
    cut = DF.index[1500]
    X1 = build_features(DF, "brent").loc[:cut]
    shocked = DF.copy()
    shocked.loc[shocked.index > cut, ["brent", "wti", "henry_hub", "crude_stocks"]] *= 3
    X2 = build_features(shocked, "brent").loc[:cut]
    pd.testing.assert_frame_equal(X1, X2)


def test_target_is_forward_return():
    y = build_target(DF, "wti", 5)
    i = 100
    expected = np.log(DF["wti"].iloc[i + 5] / DF["wti"].iloc[i])
    assert np.isclose(y.iloc[i], expected)
    assert y.iloc[-5:].isna().all()


def test_backtest_training_never_sees_future_targets():
    """Poison every target from the first test date onwards; predictions made at the first
    refit must be unchanged because those targets are not yet realised."""
    h = 5
    X = build_features(DF, "brent")
    y = build_target(DF, "brent", h)
    start = X.index[1400]
    bt1 = walk_forward(X, y, h, start, refit_every=40)
    y2 = y.copy()
    first = bt1.index[0]
    pos = y2.index.get_loc(first)
    y2.iloc[pos - h + 1:] = 1.0  # targets that are unknown at `first`
    bt2 = walk_forward(X, y2, h, start, refit_every=40)
    block = bt1.index[:40]
    pd.testing.assert_frame_equal(bt1.loc[block, ["ridge", "gbm"]], bt2.loc[block, ["ridge", "gbm"]])


def test_run_market_outputs_consistent_forecast():
    res = run_market(DF, "henry_hub", 15, trading_days(15), backtest_years=3, refit_every=126)
    f = res.forecast
    assert f["p2_5"] < f["p10"] < f["p90"] < f["p97_5"]
    assert 0 <= f["prob_up_%"] <= 100
    assert res.chosen in res.metrics.index
    assert 60 < res.metrics.loc["random_walk", "coverage80_%"] < 95


def test_load_auto_extends_prices_and_keeps_drivers(monkeypatch):
    from ogforecast import data

    idx = pd.bdate_range("2024-01-01", periods=10)
    live = pd.DataFrame({"brent": np.arange(10.0) + 80, "vix": 15.0}, index=idx)
    live.iloc[-2:, 0] = np.nan  # FRED lags the mirror by two days
    gh = pd.DataFrame({"brent": np.arange(10.0) + 80, "wti": 70.0}, index=idx)
    monkeypatch.setattr(data, "load_live", lambda start, key=None: live.copy())
    monkeypatch.setattr(data, "load_github", lambda start: gh.copy())
    out = data.load_auto("2024-01-01")
    assert out["brent"].iloc[-1] == 89.0 and out["vix"].notna().all() and "wti" in out
    assert out.attrs["source"] == "auto"

    def fail(*a, **k):
        raise RuntimeError("blocked")

    monkeypatch.setattr(data, "load_live", fail)
    assert data.load_auto("2024-01-01").attrs["source"] == "github"


def test_prices_not_extended_past_last_observation():
    """A driver series that runs longer must not drag the price series (and as-of date) forward."""
    from ogforecast.data import to_business_days

    idx = pd.bdate_range("2026-09-14", "2026-09-25")
    df = pd.DataFrame({"brent": 100.0, "vix": 20.0, "crude_stocks": np.nan}, index=idx)
    df.loc["2026-09-23":, "brent"] = np.nan   # prices published to the 22nd only
    df.loc["2026-09-16", "brent"] = np.nan    # a holiday inside the series
    df.loc["2026-09-16", "crude_stocks"] = 4.0e5
    out = to_business_days(df)
    assert out["brent"].last_valid_index() == pd.Timestamp("2026-09-22")
    assert out.loc["2026-09-16", "brent"] == 100.0          # holiday gap still filled
    assert out["crude_stocks"].last_valid_index() == pd.Timestamp("2026-09-25")  # weekly carries
