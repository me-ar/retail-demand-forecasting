"""Rolling-origin backtest: baselines vs a global LightGBM model, with quantile intervals.

Usage: python -m forecast.backtest
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import features as F
from .baselines import BASELINES
from .metrics import bias, coverage, mae, smape, wape

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "train.csv"
URL = ("https://raw.githubusercontent.com/ugursaricam/store-item-demand-forecasting/master/"
       "datasets/train.csv")
REPORTS, FIGS = ROOT / "reports", ROOT / "reports" / "figures"
CUTOFFS = ["2017-01-01", "2017-04-01", "2017-07-01", "2017-10-02"]  # forecast the 90 days after each
LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=42, num_threads=0)
N_ROUNDS = 800


def load() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    return pd.read_csv(DATA, parse_dates=["date"])


def fit_predict_lgb(feat: pd.DataFrame, cutoff: pd.Timestamp):
    train_mask = (feat["date"] < cutoff) & feat["lag_364"].notna()
    test_mask = (feat["date"] >= cutoff) & (feat["date"] < cutoff + pd.Timedelta(days=F.HORIZON))
    train_raw = feat.loc[feat["date"] < cutoff]
    data = F.target_encodings(train_raw, feat.loc[train_mask | test_mask].copy())
    tr, te = data.loc[data["date"] < cutoff], data.loc[data["date"] >= cutoff]
    cats = ["store", "item"]
    dtrain = lgb.Dataset(tr[F.FEATURES_BASE], tr["sales"], categorical_feature=cats, free_raw_data=False)
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=N_ROUNDS)
    pred = model.predict(te[F.FEATURES_BASE])
    # 80% prediction interval from two quantile models (fewer rounds — intervals need less precision)
    q = {}
    for a in (0.1, 0.9):
        qp = {**LGB_PARAMS, "objective": "quantile", "alpha": a}
        qp.pop("tweedie_variance_power")
        q[a] = lgb.train(qp, dtrain, num_boost_round=300).predict(te[F.FEATURES_BASE])
    return te, pred, q[0.1], q[0.9], model


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    raw = load()
    t0 = time.time()
    feat = F.build(raw.copy())
    print(f"features built in {time.time() - t0:.0f}s: {feat.shape}")

    rows, last = [], None
    for c in CUTOFFS:
        cutoff = pd.Timestamp(c)
        train = raw[raw["date"] < cutoff]
        test = raw[(raw["date"] >= cutoff) & (raw["date"] < cutoff + pd.Timedelta(days=F.HORIZON))]
        test = test.sort_values(["store", "item", "date"]).reset_index(drop=True)
        preds = {}
        for name, fn in BASELINES.items():
            p = fn(train, test).to_numpy()
            preds[name] = p
            rows.append({"cutoff": c, "model": name, "smape": smape(test.sales, p),
                         "wape": wape(test.sales, p), "mae": mae(test.sales, p),
                         "bias_pct": bias(test.sales, p)})
        t = time.time()
        te, p, lo, hi, model = fit_predict_lgb(feat, cutoff)
        rows.append({"cutoff": c, "model": "LightGBM (global, direct)", "smape": smape(te.sales, p),
                     "wape": wape(te.sales, p), "mae": mae(te.sales, p), "bias_pct": bias(te.sales, p),
                     "coverage_80": coverage(te.sales, lo, hi)})
        print(f"cutoff {c}: LightGBM sMAPE={rows[-1]['smape']:.2f} ({time.time() - t:.0f}s), "
              + ", ".join(f"{r['model']}={r['smape']:.2f}" for r in rows[-4:-1]))
        # test and te are both sorted by (store, item, date) -> rows align
        assert np.array_equal(test["sales"].to_numpy(), te["sales"].to_numpy())
        ens = 0.5 * p + 0.5 * preds["Multiplicative profile"]
        rows.append({"cutoff": c, "model": "Ensemble (LightGBM + profile)", "smape": smape(te.sales, ens),
                     "wape": wape(te.sales, ens), "mae": mae(te.sales, ens),
                     "bias_pct": bias(te.sales, ens)})
        # Noise floor: if sales ~ Poisson(true mean), even a perfect mean forecast has this sMAPE.
        sim = np.random.default_rng(0).poisson(np.clip(ens, 0, None))
        rows.append({"cutoff": c, "model": "Poisson noise floor (reference)", "smape": smape(sim, ens),
                     "wape": wape(sim, ens), "mae": mae(sim, ens), "bias_pct": 0.0})
        print(f"   ensemble sMAPE={rows[-2]['smape']:.2f}, noise floor~{rows[-1]['smape']:.2f}")
        last = (te.assign(pred=p, lo=lo, hi=hi), model)

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "backtest_by_fold.csv", index=False)
    summary = (res.groupby("model")[["smape", "wape", "mae", "bias_pct", "coverage_80"]].mean()
               .sort_values("smape").round(3))
    summary.to_csv(REPORTS / "backtest_summary.csv")
    (REPORTS / "backtest_summary.md").write_text(summary.to_markdown())
    print(summary.to_string())

    # ---- figures ----
    te, model = last
    imp = pd.Series(model.feature_importance("gain"), index=F.FEATURES_BASE).sort_values()
    imp = imp / imp.sum()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.tail(15).plot.barh(ax=ax, color="#2563eb")
    ax.set(title="LightGBM feature importance (gain share, last fold)", xlabel="share of total gain")
    fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (s, i) in zip(axes, [(1, 15), (7, 5)], strict=True):
        hist = raw[(raw.store == s) & (raw.item == i) & (raw.date >= "2017-07-01")]
        f = te[(te.store == s) & (te.item == i)]
        ax.plot(hist.date, hist.sales, color="#334155", lw=1, label="actual")
        ax.plot(f.date, f.pred, color="#2563eb", lw=1.6, label="LightGBM forecast")
        ax.fill_between(f.date, f.lo, f.hi, color="#2563eb", alpha=0.15, label="80% interval")
        ax.axvline(pd.Timestamp(CUTOFFS[-1]), color="grey", ls="--", lw=1)
        ax.set(title=f"Store {s} · Item {i}", ylabel="units sold")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "forecast_examples.png")

    plot_folds(res)
    with open(REPORTS / "summary.json", "w") as fh:
        json.dump({"cutoffs": CUTOFFS, "horizon_days": F.HORIZON,
                   "n_series": int(raw.groupby(["store", "item"]).ngroups),
                   "summary": summary.reset_index().to_dict(orient="records")}, fh, indent=2, default=float)


def plot_folds(res: pd.DataFrame) -> None:
    pivot = res.pivot(index="cutoff", columns="model", values="smape")
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot(ax=ax, marker="o")
    ax.set(title="sMAPE by backtest fold (lower is better)", ylabel="sMAPE (%)", xlabel="forecast origin")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(FIGS / "smape_by_fold.png")


if __name__ == "__main__":
    main()
