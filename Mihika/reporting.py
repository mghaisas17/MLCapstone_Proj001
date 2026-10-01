"""Tables and figures for the volatility-forecasting experiments.

Everything reads the files written by ``experiment.py`` -- nothing here fits a
model -- so plots can be regenerated without re-running the experiments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment import HORIZON_LABELS
from models_and_metrics import diebold_mariano

HORIZON_ORDER = list(HORIZON_LABELS.values())
KEY_MODELS = [
    "HAR-RV-L", "GARCH(1,1)", "Ridge | stats", "Ridge | signature",
    "XGBoost | signature", "MLP | stats+signature", "Signature LSTM",
]


@dataclass
class Results:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    ablations: pd.DataFrame
    meta: dict  # horizon key -> meta.json contents


def load_results(output_dir: str | Path) -> Results:
    root = Path(output_dir)
    metrics, preds, abl, meta = [], [], [], {}
    for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        meta[d.name] = json.loads((d / "meta.json").read_text())
        metrics.append(pd.read_csv(d / "metrics.csv"))
        preds.append(pd.read_parquet(d / "predictions.parquet"))
        if (d / "ablations.csv").exists():
            abl.append(pd.read_csv(d / "ablations.csv"))
    if not metrics:
        raise FileNotFoundError(f"No completed horizons found under {root}")
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return Results(cat(metrics), cat(preds), cat(abl), meta)


def _horizons(df: pd.DataFrame) -> list[str]:
    return [h for h in HORIZON_ORDER if h in set(df["Horizon"])]


# ---------------------------------------------------------------- tables


def summary_table(res: Results) -> pd.DataFrame:
    """Per horizon/model: fold-mean RMSE, fold SD, rank, and % vs Mean and HAR."""
    g = res.metrics.groupby(["Horizon", "Model"])["RMSE"].agg(Mean_RMSE="mean", Fold_SD="std").reset_index()
    for ref, col in (("Mean", "Pct_vs_Mean"), ("HAR-RV-L", "Pct_vs_HAR")):
        base = g[g.Model == ref][["Horizon", "Mean_RMSE"]].rename(columns={"Mean_RMSE": "ref"})
        g = g.merge(base, on="Horizon", how="left")
        g[col] = 100 * (1 - g["Mean_RMSE"] / g["ref"])
        g = g.drop(columns="ref")
    g["Rank"] = g.groupby("Horizon")["Mean_RMSE"].rank(method="dense").astype(int)
    g["Horizon"] = pd.Categorical(g["Horizon"], HORIZON_ORDER, ordered=True)
    return g.sort_values(["Horizon", "Rank"]).reset_index(drop=True)


def dm_table(res: Results, horizon_key: str) -> pd.DataFrame:
    """Pooled out-of-sample RMSE with Diebold-Mariano tests against the best
    model and against HAR-RV-L. The lag length is the horizon measured in
    forecast rows (``horizon_rows``), because overlapping targets are what
    autocorrelate the loss differential."""
    label = HORIZON_LABELS[horizon_key]
    p = res.predictions[res.predictions.Horizon == label]
    h = res.meta[horizon_key]["horizon_rows"]
    pred = p.pivot(index="Time", columns="Model", values="Prediction").sort_index()
    actual = p.drop_duplicates("Time").set_index("Time")["Actual"].sort_index().loc[pred.index]
    err = pred.rsub(actual, axis=0)
    pooled = np.sqrt((err**2).mean())
    best = pooled.idxmin()
    rows = []
    for m in pooled.sort_values().index:
        row = {"Model": m, "Pooled_RMSE": pooled[m], "Best": m == best}
        for ref, col in ((best, "best"), ("HAR-RV-L", "HAR")):
            if m == ref or ref not in err:
                row[f"DM_vs_{col}"] = row[f"p_vs_{col}"] = np.nan
            else:
                row[f"DM_vs_{col}"], row[f"p_vs_{col}"] = diebold_mariano(
                    err[m].to_numpy(), err[ref].to_numpy(), h
                )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def plot_relative_heatmap(res: Results):
    mean_rmse = res.metrics.groupby(["Horizon", "Model"])["RMSE"].mean().unstack().reindex(_horizons(res.metrics))
    rel = 100 * (1 - mean_rmse.div(mean_rmse["Mean"], axis=0)).drop(columns="Mean")
    fig, ax = plt.subplots(figsize=(1.2 * rel.shape[1] + 3, 4.5))
    sns.heatmap(rel, annot=True, fmt=".1f", center=0, cmap="RdYlGn", ax=ax,
                cbar_kws={"label": "% RMSE improvement vs mean forecast"})
    ax.set(title="Out-of-sample improvement over the no-information forecast", xlabel="", ylabel="")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def plot_fold_stability(res: Results, models=KEY_MODELS):
    m = res.metrics
    base = m[m.Model == "Mean"][["Horizon", "Fold", "RMSE"]].rename(columns={"RMSE": "MeanRMSE"})
    z = m.merge(base, on=["Horizon", "Fold"])
    z["Relative RMSE"] = z["RMSE"] / z["MeanRMSE"]
    z = z[z.Model.isin(models)]
    g = sns.catplot(data=z, x="Horizon", y="Relative RMSE", hue="Model", col="Fold", kind="point",
                    order=_horizons(m), height=4, aspect=1.25)
    for ax in g.axes.flat:
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.tick_params(axis="x", rotation=25)
    g.fig.subplots_adjust(top=0.82)
    g.fig.suptitle("Does each model's advantage persist across time folds? (<1 beats the mean)")
    return g.fig


def _best_model(res: Results) -> dict[str, str]:
    summ = summary_table(res)
    return {str(h): d.sort_values("Rank").iloc[0]["Model"] for h, d in summ.groupby("Horizon", observed=True)}


def plot_predictions(res: Results):
    best, hz = _best_model(res), _horizons(res.predictions)
    fig, axes = plt.subplots(len(hz), 1, figsize=(14, 2.8 * len(hz)), constrained_layout=True, squeeze=False)
    for ax, h in zip(axes[:, 0], hz):
        sub = res.predictions[res.predictions.Horizon == h]
        a = sub.drop_duplicates("Time").sort_values("Time")
        ax.plot(pd.to_datetime(a.Time), a.Actual, color="black", lw=1.1, label="Actual")
        for model, ls in (("HAR-RV-L", "--"), (best[h], "-")):
            s = sub[sub.Model == model].sort_values("Time")
            ax.plot(pd.to_datetime(s.Time), s.Prediction, ls=ls, lw=1.0, alpha=0.9, label=model)
        ax.set(title=h, ylabel="Volatility")
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    return fig


def plot_cumulative_advantage(res: Results):
    best, hz = _best_model(res), _horizons(res.predictions)
    fig, axes = plt.subplots(len(hz), 1, figsize=(14, 2.6 * len(hz)), constrained_layout=True, squeeze=False)
    for ax, h in zip(axes[:, 0], hz):
        sub = res.predictions[res.predictions.Horizon == h]
        har = sub[sub.Model == "HAR-RV-L"].set_index("Time")
        win = sub[sub.Model == best[h]].set_index("Time")
        z = (((har.Actual - har.Prediction) ** 2) - ((win.Actual - win.Prediction) ** 2)).sort_index().cumsum()
        ax.plot(pd.to_datetime(z.index), z.to_numpy(), color="#276FBF")
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=f"{h}: {best[h]} versus HAR-RV-L", ylabel="Cumulative\nSE advantage")
    return fig


def plot_regimes(res: Results, models=KEY_MODELS):
    r = res.predictions.copy()
    r["Regime"] = r.groupby("Horizon")["Actual"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["Low", "Middle", "High"])
    )
    r["SE"] = (r.Actual - r.Prediction) ** 2
    rr = np.sqrt(r.groupby(["Horizon", "Model", "Regime"], observed=True).SE.mean()).reset_index(name="RMSE")
    har = rr[rr.Model == "HAR-RV-L"][["Horizon", "Regime", "RMSE"]].rename(columns={"RMSE": "HAR"})
    rr = rr.merge(har, on=["Horizon", "Regime"])
    rr["% improvement vs HAR"] = 100 * (1 - rr.RMSE / rr.HAR)
    rr = rr[rr.Model.isin(models) & (rr.Model != "HAR-RV-L")]
    g = sns.catplot(data=rr, x="Regime", y="% improvement vs HAR", hue="Model", col="Horizon",
                    col_order=_horizons(r), col_wrap=3, kind="bar", height=3.4, aspect=1.1)
    for ax in g.axes.flat:
        ax.axhline(0, color="black", lw=0.8)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Which models help in calm and stressed periods?")
    return g.fig


def plot_ablations(res: Results):
    if res.ablations.empty:
        raise ValueError("No ablation results; run with run_ablations=True")
    controls = {"Signature level": "L1-1", "Augmentation": "Raw path", "Dimensions": "Price only"}
    diag = res.ablations.groupby(["Horizon", "Experiment", "Setting"])["RMSE"].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, (exp, control) in zip(axes, controls.items()):
        z = diag[diag.Experiment == exp]
        base = z[z.Setting == control][["Horizon", "RMSE"]].rename(columns={"RMSE": "ctl"})
        z = z.merge(base, on="Horizon")
        z["Improvement"] = 100 * (1 - z.RMSE / z.ctl)
        mat = z.pivot(index="Horizon", columns="Setting", values="Improvement").reindex(_horizons(diag))
        sns.heatmap(mat, annot=True, fmt=".1f", center=0, cmap="RdYlGn", cbar=False, ax=ax)
        ax.set(title=f"{exp} (% vs {control})", xlabel="", ylabel="")
    return fig
