"""Tables and figures for the volatility-forecasting experiments.

Everything reads the files written by ``experiment.py`` -- nothing here fits a
model -- so plots can be regenerated without re-running the experiments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiment import HORIZON_LABELS
from models_and_metrics import diebold_mariano

HORIZON_ORDER = list(HORIZON_LABELS.values())
KEY_MODELS = [
    "HAR-RV-L", "GARCH(1,1) | rolling refit", "Ridge | stats", "Ridge | signature",
    "Ridge | lag bank (matched)", "Ridge | multiscale stats",
    "XGBoost | signature", "MLP | stats+signature", "Signature LSTM",
    "HAR-RV-L + flag", "Ridge | signature + flag",
]
# (without the HMM flag, with it): what the flag adds to an otherwise identical model
FLAG_PAIRS = [
    ("HAR-RV-L", "HAR-RV-L + flag"),
    ("Ridge | signature", "Ridge | signature + flag"),
    ("Ridge | signature", "Ridge | signature + stress dim"),
    ("XGBoost | signature", "XGBoost | signature + flag"),
]


@dataclass
class Results:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    ablations: pd.DataFrame
    meta: dict  # horizon key -> meta.json contents
    flags: pd.DataFrame = field(default_factory=pd.DataFrame)  # out-of-sample flag per test row
    flag_diag: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-fold flag diagnostics


def load_results(output_dir: str | Path) -> Results:
    root = Path(output_dir)
    metrics, preds, abl, meta, flags, fdiag = [], [], [], {}, [], []
    for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        meta[d.name] = json.loads((d / "meta.json").read_text())
        metrics.append(pd.read_csv(d / "metrics.csv"))
        preds.append(pd.read_parquet(d / "predictions.parquet"))
        if (d / "ablations.csv").exists():
            abl.append(pd.read_csv(d / "ablations.csv"))
        if (d / "flags.parquet").exists():
            flags.append(pd.read_parquet(d / "flags.parquet"))
            fdiag.append(pd.read_csv(d / "flag_diagnostics.csv"))
    if not metrics:
        raise FileNotFoundError(f"No completed horizons found under {root}")
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return Results(cat(metrics), cat(preds), cat(abl), meta, cat(flags), cat(fdiag))


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


# ---------------------------------------------------------------- anomaly flag


def flag_diagnostics_table(res: Results) -> pd.DataFrame:
    """How the flag behaves per fold: how often it is on, how long episodes last,
    and whether volatility is actually higher while it is on."""
    if res.flag_diag.empty:
        raise ValueError("No flag results; run with use_flag=True")
    d = res.flag_diag.copy()
    d["Vol_ratio_flagged_vs_not"] = d["Mean_vol_flagged"] / d["Mean_vol_unflagged"]
    d["Horizon"] = pd.Categorical(d["Horizon"], HORIZON_ORDER, ordered=True)
    return d.sort_values(["Horizon", "Fold"]).drop(columns="HorizonKey").reset_index(drop=True)


def flag_effect_table(res: Results) -> pd.DataFrame:
    """For each (model, model + flag) pair: pooled out-of-sample RMSE overall and
    split by flagged / unflagged rows, with a Diebold-Mariano test of the flag
    version against the original. Positive Pct_* means the flag lowered the error."""
    if res.flags.empty:
        raise ValueError("No flag results; run with use_flag=True")
    rows = []
    for hk, meta in res.meta.items():
        label = HORIZON_LABELS[hk]
        p = res.predictions[res.predictions.Horizon == label]
        pred = p.pivot(index="Time", columns="Model", values="Prediction").sort_index()
        actual = p.drop_duplicates("Time").set_index("Time")["Actual"].sort_index().loc[pred.index]
        err = pred.rsub(actual, axis=0)
        flag = res.flags[res.flags.Horizon == label].set_index("Time")["flag"].reindex(pred.index).fillna(0).to_numpy() == 1
        rmse = lambda e, m: float(np.sqrt((e[m] ** 2).mean())) if m.any() else np.nan
        everything = np.ones(len(err), bool)
        for base, new in FLAG_PAIRS:
            if base not in err or new not in err:
                continue
            row = {"Horizon": label, "Base": base, "With_flag": new, "N_flagged": int(flag.sum()), "N": len(err)}
            for tag, mask in (("all", everything), ("flagged", flag), ("unflagged", ~flag)):
                b, n = rmse(err[base], mask), rmse(err[new], mask)
                row[f"RMSE_base_{tag}"], row[f"RMSE_flag_{tag}"] = b, n
                row[f"Pct_{tag}"] = 100 * (1 - n / b) if b else np.nan
            row["DM_stat"], row["DM_p"] = diebold_mariano(err[new].to_numpy(), err[base].to_numpy(), meta["horizon_rows"])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_flag_timeline(res: Results):
    """Realized volatility through the out-of-sample period, flagged stretches shaded."""
    if res.flags.empty:
        raise ValueError("No flag results; run with use_flag=True")
    hz = _horizons(res.flags)
    fig, axes = plt.subplots(len(hz), 1, figsize=(14, 2.6 * len(hz)), constrained_layout=True, squeeze=False)
    for ax, h in zip(axes[:, 0], hz):
        f = res.flags[res.flags.Horizon == h].sort_values("Time")
        a = res.predictions[res.predictions.Horizon == h].drop_duplicates("Time").sort_values("Time")
        t = pd.to_datetime(a.Time)
        ax.plot(t, a.Actual, color="black", lw=1.0, label="Realized volatility")
        ft = pd.to_datetime(f.Time)
        ax.fill_between(ft, 0, float(a.Actual.max()), where=(f.flag.to_numpy() == 1), step="mid",
                        color="#C8553D", alpha=0.3, label="Anomaly flag", linewidth=0)
        ax.set(title=f"{h}  (flagged {100 * f.flag.mean():.0f}% of out-of-sample rows)", ylabel="Volatility")
        ax.legend(loc="upper right", fontsize=8)
    return fig


# ---------------------------------------------------------------- fairness checks

CONTROLS = ["Ridge | lag bank (matched)", "Ridge | multiscale stats", "Ridge | stats"]


def control_table(res: Results) -> pd.DataFrame:
    """Is the signature gain specific to signatures? Ridge on signatures against the
    same Ridge on signature-free feature sets: a lag bank with exactly as many
    columns as the signature, a hand-built multi-scale volatility set, and the plain
    summary stats. ``Pct_sig_better`` > 0 means signatures have the lower RMSE;
    DM_stat < 0 (small DM_p) means signatures are significantly better."""
    rows = []
    for hk, meta in res.meta.items():
        label = HORIZON_LABELS[hk]
        p = res.predictions[res.predictions.Horizon == label]
        pred = p.pivot(index="Time", columns="Model", values="Prediction").sort_index()
        actual = p.drop_duplicates("Time").set_index("Time")["Actual"].sort_index().loc[pred.index]
        err = pred.rsub(actual, axis=0)
        if "Ridge | signature" not in err:
            continue
        rm = lambda e: float(np.sqrt((e**2).mean()))
        for ctrl in CONTROLS:
            if ctrl not in err:
                continue
            dm, pv = diebold_mariano(err["Ridge | signature"].to_numpy(), err[ctrl].to_numpy(), meta["horizon_rows"])
            rows.append({
                "Horizon": label, "Control": ctrl, "RMSE_signature": rm(err["Ridge | signature"]),
                "RMSE_control": rm(err[ctrl]),
                "Pct_sig_better": 100 * (1 - rm(err["Ridge | signature"]) / rm(err[ctrl])),
                "DM_stat": dm, "DM_p": pv,
            })
    if not rows:
        raise ValueError(
            "No control-model results found. This run predates the controls; re-run with the current "
            "code into a fresh output_dir (or delete the horizon folders in the old one)."
        )
    return pd.DataFrame(rows)


def bias_table(res: Results, models=None) -> pd.DataFrame:
    """Mean forecast / mean realized volatility per fold (1.0 = unbiased). A model
    whose forecasts sit far above 1 in some folds is reverting to the wrong level,
    which is how a stale long-run variance would show up in GARCH."""
    p = res.predictions
    if models is not None:
        p = p[p.Model.isin(models)]
    g = p.groupby(["Horizon", "Model", "Fold"]).agg(pred=("Prediction", "mean"), actual=("Actual", "mean")).reset_index()
    g["Bias"] = g.pred / g.actual
    t = g.pivot_table(index=["Horizon", "Model"], columns="Fold", values="Bias")
    t.columns = [f"Fold {c}" for c in t.columns]
    pooled = p.groupby(["Horizon", "Model"]).agg(pred=("Prediction", "mean"), actual=("Actual", "mean"))
    t["Pooled"] = pooled.pred / pooled.actual
    t = t.reset_index()
    t["Horizon"] = pd.Categorical(t["Horizon"], HORIZON_ORDER, ordered=True)
    return t.sort_values(["Horizon", "Model"]).reset_index(drop=True)


# ---------------------------------------------------------------- comparing runs


def compare_runs(runs: dict, model: str = "Ridge | signature", folds=None, metric: str = "Pct_vs_HAR") -> pd.DataFrame:
    """One row per run, one column per horizon, for a single model: use it to compare
    signature settings that were each run into their own output directory.

    ``folds`` restricts the comparison to some folds (e.g. choose a setting on folds
    (1, 2) and confirm on fold (3,)); picking the best of many settings on the same
    folds you then report is optimistic. ``metric`` is a column of ``summary_table``:
    Pct_vs_HAR / Pct_vs_Mean (positive = better) or Mean_RMSE (lower = better)."""
    from dataclasses import replace as _replace

    out = {}
    for name, path in runs.items():
        res = load_results(path)
        if folds is not None:
            res = _replace(res, metrics=res.metrics[res.metrics.Fold.isin(list(folds))])
        s = summary_table(res)
        s = s[s.Model == model]
        out[name] = s.set_index("Horizon")[metric]
    table = pd.DataFrame(out).T
    return table[[h for h in HORIZON_ORDER if h in table.columns]]


def flag_accuracy_table(res: Results, quantile: float = 0.8) -> pd.DataFrame:
    """How often does the flag fire when it should? "Should" is defined as the realized
    volatility over the forecast horizon landing in the top ``1 - quantile`` of that fold's
    out-of-sample rows, so this measures the flag as a warning of high upcoming volatility.

    Precision = P(high | flagged); Recall = P(flagged | high); Lift = precision / base rate
    (1.0 = no better than chance); False_alarm_rate = P(flagged | not high). AUC_flag ranks
    rows by the stress probability; AUC_HAR ranks them by the HAR-RV-L forecast, a simple
    benchmark the flag must beat to be adding something."""
    from sklearn.metrics import roc_auc_score

    if res.flags.empty:
        raise ValueError("No flag results; run with use_flag=True")
    rows = []
    for hk in res.meta:
        label = HORIZON_LABELS[hk]
        f = res.flags[res.flags.Horizon == label]
        har = res.predictions[(res.predictions.Horizon == label) & (res.predictions.Model == "HAR-RV-L")]
        d = f.merge(har[["Time", "Fold", "Actual", "Prediction"]], on=["Time", "Fold"])
        d["high"] = d.groupby("Fold")["Actual"].transform(lambda s: s >= s.quantile(quantile))
        groups = [(str(k), g) for k, g in d.groupby("Fold")] + [("All", d)]
        for fold, g in groups:
            high, flag = g["high"].to_numpy(bool), g["flag"].to_numpy(bool)
            two_classes = 0 < high.sum() < len(high)
            prec = high[flag].mean() if flag.any() else np.nan
            rows.append({
                "Horizon": label, "Fold": fold, "N": len(g), "Base_rate": high.mean(),
                "Share_flagged": flag.mean(), "Precision": prec,
                "Recall": flag[high].mean() if high.any() else np.nan,
                "Lift": prec / high.mean() if flag.any() and high.any() else np.nan,
                "False_alarm_rate": flag[~high].mean() if (~high).any() else np.nan,
                "AUC_flag": roc_auc_score(high, g["p_stress"]) if two_classes else np.nan,
                "AUC_HAR": roc_auc_score(high, g["Prediction"]) if two_classes else np.nan,
            })
    out = pd.DataFrame(rows)
    out["Horizon"] = pd.Categorical(out["Horizon"], HORIZON_ORDER, ordered=True)
    return out.sort_values(["Horizon", "Fold"]).reset_index(drop=True)
