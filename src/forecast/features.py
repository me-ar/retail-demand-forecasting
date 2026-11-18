"""Feature engineering for a *direct* 90-day-ahead global model.

Every lag / rolling feature uses information at least HORIZON days old, so the model can
forecast the full 90-day window in one shot without recursive error accumulation and
without any leakage from the period it is predicting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 90
LAGS = [91, 98, 105, 112, 119, 182, 364, 728]
ROLL_WINDOWS = [7, 28, 91, 182, 364]


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["year"] = d.dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def lag_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Assumes df is sorted by (store, item, date) with a contiguous daily index per series."""
    g = df.groupby(["store", "item"], sort=False)[target]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(HORIZON + 1)
    gs = shifted.groupby([df["store"], df["item"]], sort=False)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = gs.transform(lambda s, w=w: s.rolling(w, min_periods=max(1, w // 2)).mean())
    df["roll_std_28"] = gs.transform(lambda s: s.rolling(28, min_periods=14).std())
    for alpha in (0.1, 0.02):
        df[f"ewm_{alpha}"] = gs.transform(lambda s, a=alpha: s.ewm(alpha=a).mean())
    # same weekday, averaged over the last 4 available weeks (lags 91..112 share the weekday)
    df["dow_mean_4w"] = df[["lag_91", "lag_98", "lag_105", "lag_112"]].mean(axis=1)
    return df


def target_encodings(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Store-item level and day-of-week/month profiles, computed on the training window only."""
    si = train.groupby(["store", "item"])["sales"].mean().rename("te_store_item")
    dow = (train.groupby(["item", "dow"])["sales"].mean()
           / train.groupby("item")["sales"].mean()).rename("te_item_dow")
    mon = (train.groupby(["item", "month"])["sales"].mean()
           / train.groupby("item")["sales"].mean()).rename("te_item_month")
    df = df.drop(columns=[c for c in ["te_store_item", "te_item_dow", "te_item_month"] if c in df])
    df = df.join(si, on=["store", "item"]).join(dow, on=["item", "dow"]).join(mon, on=["item", "month"])
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return lag_features(calendar_features(df))


FEATURES_BASE = (["store", "item", "dow", "month", "dayofyear", "weekofyear", "year", "is_weekend",
                  "is_month_start", "is_month_end", "doy_sin", "doy_cos"]
                 + [f"lag_{x}" for x in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
                 + ["roll_std_28", "ewm_0.1", "ewm_0.02", "dow_mean_4w",
                    "te_store_item", "te_item_dow", "te_item_month"])
