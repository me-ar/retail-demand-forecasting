"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def smape(y, yhat) -> float:
    """Symmetric MAPE in % (0/0 counted as 0), as used in the Kaggle competition."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    out = np.where(denom == 0, 0.0, 2 * np.abs(yhat - y) / np.where(denom == 0, 1, denom))
    return float(100 * out.mean())


def wape(y, yhat) -> float:
    """Weighted APE: total absolute error / total demand — robust to low-volume series."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * np.abs(y - yhat).sum() / np.abs(y).sum())


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def bias(y, yhat) -> float:
    """Positive = over-forecasting (excess stock); negative = under-forecasting (stock-outs)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(100 * (yhat.sum() - y.sum()) / y.sum())


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))
