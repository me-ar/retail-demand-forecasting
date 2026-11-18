import numpy as np
import pandas as pd
import pytest

from forecast import features as F
from forecast.baselines import multiplicative_profile, seasonal_naive_weekly, seasonal_naive_yearly
from forecast.metrics import bias, coverage, smape, wape


def make_panel(days=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    rows = []
    for store in (1, 2):
        for item in (1, 2, 3):
            level = 20 * store + 10 * item
            dow = 1 + 0.3 * (dates.dayofweek >= 5)
            mu = level * dow * (1 + 0.1 * np.sin(2 * np.pi * dates.dayofyear / 365))
            rows.append(pd.DataFrame({"date": dates, "store": store, "item": item,
                                      "sales": rng.poisson(mu)}))
    return pd.concat(rows, ignore_index=True)


def test_smape_known_values():
    assert smape([100], [100]) == 0
    assert smape([0], [0]) == 0
    assert smape([100], [50]) == pytest.approx(100 * 2 * 50 / 150)


def test_wape_bias_coverage():
    y, p = np.array([10, 20, 30]), np.array([12, 18, 30])
    assert wape(y, p) == pytest.approx(100 * 4 / 60)
    assert bias(y, p) == 0
    assert coverage(y, y - 1, y + 1) == 1.0


def test_features_do_not_leak_within_horizon():
    """Changing sales after a cutoff must not change any feature in the next 90 days."""
    df = make_panel()
    cutoff = pd.Timestamp("2016-06-01")
    a = F.build(df.copy())
    tampered = df.copy()
    tampered.loc[tampered["date"] >= cutoff, "sales"] *= 10
    b = F.build(tampered)
    window = (a["date"] >= cutoff) & (a["date"] < cutoff + pd.Timedelta(days=F.HORIZON))
    cols = [c for c in F.FEATURES_BASE if not c.startswith("te_")]
    pd.testing.assert_frame_equal(a.loc[window, cols], b.loc[window, cols])


def test_lags_are_aligned_per_series():
    df = F.build(make_panel())
    s = df[(df.store == 1) & (df.item == 2)].reset_index(drop=True)
    assert s.loc[400, "lag_364"] == s.loc[400 - 364, "sales"]
    assert df.groupby(["store", "item"]).head(91)["lag_91"].isna().all()  # no bleed across series


def test_seasonal_naive_weekday_alignment():
    df = make_panel()
    cutoff = pd.Timestamp("2017-01-01")
    train, test = df[df.date < cutoff], df[(df.date >= cutoff) & (df.date < cutoff + pd.Timedelta(days=14))]
    p = seasonal_naive_weekly(train, test)
    first = test.index[0]
    same_dow = train[(train.store == test.loc[first, "store"]) & (train.item == test.loc[first, "item"])]
    assert p.loc[first] == same_dow.iloc[-7]["sales"]
    assert seasonal_naive_yearly(train, test).notna().all()


def test_profile_baseline_beats_naive_on_structured_data():
    df = make_panel()
    cutoff = pd.Timestamp("2017-03-01")
    train = df[df.date < cutoff]
    test = df[(df.date >= cutoff) & (df.date < cutoff + pd.Timedelta(days=90))]
    assert smape(test.sales, multiplicative_profile(train, test)) < smape(
        test.sales, seasonal_naive_weekly(train, test))
