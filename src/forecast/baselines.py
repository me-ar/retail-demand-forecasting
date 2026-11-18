"""Statistical baselines. A model is only worth deploying if it beats these."""
from __future__ import annotations

import pandas as pd


def seasonal_naive_weekly(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Repeat the last observed week (same weekday) across the horizon."""
    last = train[train["date"] > train["date"].max() - pd.Timedelta(days=7)]
    key = last.assign(dow=last["date"].dt.dayofweek).set_index(["store", "item", "dow"])["sales"]
    idx = pd.MultiIndex.from_arrays([test["store"], test["item"], test["date"].dt.dayofweek])
    return pd.Series(key.reindex(idx).to_numpy(), index=test.index)


def seasonal_naive_yearly(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Same weekday 52 weeks earlier (364-day lag keeps the weekday aligned)."""
    hist = train.set_index(["store", "item", "date"])["sales"]
    idx = pd.MultiIndex.from_arrays([test["store"], test["item"], test["date"] - pd.Timedelta(days=364)])
    return pd.Series(hist.reindex(idx).to_numpy(), index=test.index)


def multiplicative_profile(train: pd.DataFrame, test: pd.DataFrame, recent_days: int = 365) -> pd.Series:
    """Classical decomposition: recent level × day-of-week factor × month factor × trend.

    A strong baseline for retail: level from the last year, seasonal profiles from all history,
    and a year-over-year growth factor.
    """
    tr = train.assign(dow=train["date"].dt.dayofweek, month=train["date"].dt.month,
                      year=train["date"].dt.year)
    end = tr["date"].max()
    recent = tr[tr["date"] > end - pd.Timedelta(days=recent_days)]
    prior = tr[(tr["date"] <= end - pd.Timedelta(days=recent_days))
               & (tr["date"] > end - pd.Timedelta(days=2 * recent_days))]
    level = recent.groupby(["store", "item"])["sales"].mean()
    growth = recent["sales"].mean() / prior["sales"].mean() if len(prior) else 1.0
    item_mean = tr.groupby("item")["sales"].mean()
    dow_f = tr.groupby(["item", "dow"])["sales"].mean() / item_mean.reindex(
        tr.groupby(["item", "dow"])["sales"].mean().index.get_level_values(0)).to_numpy()
    mon_f = tr.groupby(["item", "month"])["sales"].mean() / item_mean.reindex(
        tr.groupby(["item", "month"])["sales"].mean().index.get_level_values(0)).to_numpy()
    # level is a 12-month average -> already averages out monthly seasonality
    te = test.assign(dow=test["date"].dt.dayofweek, month=test["date"].dt.month)
    years_ahead = ((te["date"] - end).dt.days / 365.25 + 0.5).to_numpy()
    lv = level.reindex(pd.MultiIndex.from_arrays([te["store"], te["item"]])).to_numpy()
    df_ = dow_f.reindex(pd.MultiIndex.from_arrays([te["item"], te["dow"]])).to_numpy()
    mf = mon_f.reindex(pd.MultiIndex.from_arrays([te["item"], te["month"]])).to_numpy()
    return pd.Series(lv * df_ * mf * growth ** years_ahead, index=test.index)


BASELINES = {
    "Seasonal naive (last week)": seasonal_naive_weekly,
    "Seasonal naive (last year)": seasonal_naive_yearly,
    "Multiplicative profile": multiplicative_profile,
}
