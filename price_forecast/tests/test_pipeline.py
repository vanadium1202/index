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
