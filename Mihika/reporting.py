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
    anomaly_flags: pd.DataFrame = field(default_factory=pd.DataFrame)  # out-of-sample anomaly score/flag per row


def load_results(output_dir: str | Path) -> Results:
    root = Path(output_dir)
    metrics, preds, abl, meta, flags, fdiag, aflags = [], [], [], {}, [], [], []
    for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        meta[d.name] = json.loads((d / "meta.json").read_text())
        metrics.append(pd.read_csv(d / "metrics.csv"))
        preds.append(pd.read_parquet(d / "predictions.parquet"))
        if (d / "ablations.csv").exists():
            abl.append(pd.read_csv(d / "ablations.csv"))
        if (d / "flags.parquet").exists():
            flags.append(pd.read_parquet(d / "flags.parquet"))
            fdiag.append(pd.read_csv(d / "flag_diagnostics.csv"))
        if (d / "anomaly_flags.parquet").exists():
            aflags.append(pd.read_parquet(d / "anomaly_flags.parquet"))
    if not metrics:
        raise FileNotFoundError(f"No completed horizons found under {root}")
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return Results(cat(metrics), cat(preds), cat(abl), meta, cat(flags), cat(fdiag), cat(aflags))


def _horizons(df: pd.DataFrame) -> list[str]:
    return [h for h in HORIZON_ORDER if h in set(df["Horizon"])]


TARGET_LABELS = {"vol": "Volatility", "drawdown": "Max drawdown", "drawdown_ratio": "Drawdown / realized vol"}


def _target_label(res: Results) -> str:
    targets = {m.get("target", "vol") for m in res.meta.values()}
    return TARGET_LABELS.get(targets.pop(), "Target") if len(targets) == 1 else "Target"


def _pooled(res: Results, horizon_key: str):
    """(predictions, actual, errors) on one horizon's out-of-sample rows, one column per model."""
    p = res.predictions[res.predictions.Horizon == HORIZON_LABELS[horizon_key]]
    pred = p.pivot(index="Time", columns="Model", values="Prediction").sort_index()
    actual = p.drop_duplicates("Time").set_index("Time")["Actual"].sort_index().loc[pred.index]
    return pred, actual, pred.rsub(actual, axis=0)


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
        ax.set(title=h, ylabel=_target_label(res))
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
        if f.empty:
            continue  # horizons without a flag (the intraday-path variants)
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


# ---------------------------------------------------------------- intraday paths for daily/weekly horizons

INTRADAY_COMPARISONS = [
    # (model, [what it is compared against])
    ("Ridge | intraday signature", [
        "Ridge | intraday lag bank (matched)", "Ridge | intraday multiscale stats", "HAR-RV-L (intraday RV)",
        "Ridge | signature", "HAR-RV-L"]),
    ("Ridge | intraday signature + HAR-RV", ["HAR-RV-L (intraday RV)"]),
    ("HAR-RV-L (intraday RV)", ["HAR-RV-L"]),
]


def pair_table(res: Results, comparisons, intraday_only: bool = False) -> pd.DataFrame:
    """Pooled out-of-sample RMSE of each ``(model, [versus, ...])`` comparison present in the results.
    ``Pct_better`` > 0 means ``Model`` has the lower RMSE; ``DM_stat`` < 0 with a small ``DM_p`` means
    it is significantly better."""
    rows = []
    for hk, meta in res.meta.items():
        if intraday_only and not meta.get("intraday_paths"):
            continue
        _, _, err = _pooled(res, hk)
        rm = lambda e: float(np.sqrt((e**2).mean()))
        for model, versus in comparisons:
            if model not in err:
                continue
            for other in versus:
                if other not in err:
                    continue
                dm, pv = diebold_mariano(err[model].to_numpy(), err[other].to_numpy(), meta["horizon_rows"])
                rows.append({
                    "Horizon": HORIZON_LABELS[hk], "Model": model, "Versus": other,
                    "RMSE_model": rm(err[model]), "RMSE_versus": rm(err[other]),
                    "Pct_better": 100 * (1 - rm(err[model]) / rm(err[other])), "DM_stat": dm, "DM_p": pv,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Horizon"] = pd.Categorical(out["Horizon"], HORIZON_ORDER, ordered=True)
    return out.sort_values(["Horizon"], kind="stable").reset_index(drop=True)


def intraday_table(res: Results) -> pd.DataFrame:
    """The questions the intraday-path horizons (``*_i``) exist to answer, on identical rows and targets:

    * signature vs lag bank / multi-scale stats built from the same intraday window (is it the
      signature, or just having the intraday path?);
    * vs HAR-RV-L on intraday realized variance (the standard strong baseline) and vs the
      daily-path signature (does the intraday path add anything?);
    * signature + HAR-RV vs HAR-RV alone (does the signature add on top of the baseline?);
    * intraday-RV HAR vs the daily-estimator HAR.

    ``Pct_better`` > 0 means ``Model`` has the lower pooled RMSE; ``DM_stat`` < 0 with a small
    ``DM_p`` means it is significantly better."""
    out = pair_table(res, INTRADAY_COMPARISONS, intraday_only=True)
    if out.empty:
        raise ValueError("No intraday-path results; run the *_i horizons (see the notebook section).")
    return out


# ---------------------------------------------------------------- anomaly score in the forecasts

ANOMALY_USES = [
    # (use, base model, model with the anomaly score; "{w}" = sig / free)
    ("feature", "HAR-RV-L", "HAR-RV-L + anomaly [{w}]"),
    ("feature", "Ridge | signature", "Ridge | signature + anomaly [{w}]"),
    ("residual", "HAR-RV-L", "HAR-RV-L + anomaly residual [{w}]"),
    # the decisive test: does the score add anything beyond fixing HAR's average bias?
    ("residual vs calibrated", "HAR-RV-L calibrated", "HAR-RV-L + anomaly residual [{w}]"),
    ("gating", "HAR-RV-L", "HAR-RV-L | anomaly-gated [{w}]"),
]


def anomaly_effect_table(res: Results) -> pd.DataFrame:
    """Does the anomaly score improve the forecasts, and is the signature detector better than the
    best signature-free one?

    For each use (feature / residual / gating) and detector (sig / free): pooled RMSE of the base model
    and of the same model with the anomaly score, % change (positive = better), RMSE on rows the detector
    flags vs the rest, and a Diebold-Mariano test against the base. "Sig vs free" compares the two
    anomaly versions directly (positive = the signature detector's version is better)."""
    if res.anomaly_flags.empty:
        raise ValueError("No anomaly results; run the *_i horizons with use_anomaly=True")
    rows = []
    for hk, meta in res.meta.items():
        label = HORIZON_LABELS[hk]
        p = res.predictions[res.predictions.Horizon == label]
        if p.empty:
            continue
        pred = p.pivot(index="Time", columns="Model", values="Prediction").sort_index()
        actual = p.drop_duplicates("Time").set_index("Time")["Actual"].sort_index().loc[pred.index]
        err = pred.rsub(actual, axis=0)
        af = res.anomaly_flags[res.anomaly_flags.Horizon == label]
        rm = lambda e, m: float(np.sqrt((e[m] ** 2).mean())) if m.any() else np.nan
        for use, base, tmpl in ANOMALY_USES:
            new = {w: tmpl.format(w=w) for w in ("sig", "free")}
            if base not in err or any(n not in err for n in new.values()):
                continue
            for w, name in new.items():
                flag = af[af.Which == w].set_index("Time")["flag"].reindex(pred.index).fillna(0).to_numpy() == 1
                allr = np.ones(len(err), bool)
                dm, pv = diebold_mariano(err[name].to_numpy(), err[base].to_numpy(), meta["horizon_rows"])
                row = {"Horizon": label, "Use": use, "Base": base, "Detector": ANOMALY_LABEL[w], "N_flagged": int(flag.sum())}
                for tag, m in (("all", allr), ("flagged", flag), ("unflagged", ~flag)):
                    b, n = rm(err[base], m), rm(err[name], m)
                    row[f"Pct_{tag}"] = 100 * (1 - n / b) if b else np.nan
                row["RMSE_base"], row["RMSE_with"] = rm(err[base], allr), rm(err[name], allr)
                row["DM_p_vs_base"] = pv
                if w == "sig":
                    dm2, pv2 = diebold_mariano(err[new["sig"]].to_numpy(), err[new["free"]].to_numpy(), meta["horizon_rows"])
                    row["Pct_sig_vs_free"] = 100 * (1 - rm(err[new["sig"]], allr) / rm(err[new["free"]], allr))
                    row["DM_p_sig_vs_free"] = pv2
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No anomaly model pairs found in these results")
    out["Horizon"] = pd.Categorical(out["Horizon"], HORIZON_ORDER, ordered=True)
    return out.sort_values(["Horizon", "Use", "Base", "Detector"]).reset_index(drop=True)


ANOMALY_LABEL = {"sig": "signature", "free": "signature-free"}


# ---------------------------------------------------------------- drawdown

# (model, [what it is compared against]): is it the signature, or the learner, or just the window?
SHAPE_COMPARISONS = [
    ("Ridge | signature", ["Ridge | lag bank (matched)", "Ridge | stats", "HAR-RV-L calibrated"]),
    ("Ridge | intraday signature", [
        "Ridge | intraday lag bank (matched)", "Ridge | intraday multiscale stats", "HAR-RV-L calibrated"]),
    ("XGBoost | signature", ["XGBoost | lag bank (matched)", "Ridge | signature"]),
    ("MLP | signature", ["MLP | lag bank (matched)", "Ridge | signature"]),
    ("XGBoost | intraday signature", ["XGBoost | intraday lag bank (matched)", "Ridge | intraday signature"]),
    ("MLP | intraday signature", ["MLP | intraday lag bank (matched)", "Ridge | intraday signature"]),
]


def shape_table(res: Results) -> pd.DataFrame:
    """Signature models against signature-free ones with the same learner, and the nonlinear learners
    against Ridge on the same signature. On the ``drawdown_ratio`` target (scale removed) a win over
    the matched lag bank means the signature predicts the *shape* of the coming path."""
    out = pair_table(res, SHAPE_COMPARISONS)
    if out.empty:
        raise ValueError("None of the shape comparisons are in these results (see DRAWDOWN_MODELS).")
    return out


def anomaly_auc_table(res: Results, quantile: float = 0.8, base: str = "HAR-RV-L calibrated") -> pd.DataFrame:
    """Does a high anomaly score come before a bad outcome? "Bad" = the target (e.g. the drawdown) in
    the top ``1 - quantile`` of its fold's test rows. ``AUC_score`` ranks rows by the detector's score
    alone; ``AUC_base`` by the ``base`` model's forecast; ``AUC_score_within_base`` is the score's AUC
    within quintiles of that forecast (0.5 = nothing beyond what the forecast already knows)."""
    from sklearn.metrics import roc_auc_score

    if res.anomaly_flags.empty:
        raise ValueError("No anomaly results; run the *_i horizons with use_anomaly=True")

    def auc(y, s):
        return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

    rows = []
    for hk in res.meta:
        label = HORIZON_LABELS[hk]
        p = res.predictions[(res.predictions.Horizon == label) & (res.predictions.Model == base)]
        if p.empty:
            continue
        for which, af in res.anomaly_flags[res.anomaly_flags.Horizon == label].groupby("Which"):
            d = af.merge(p[["Time", "Fold", "Actual", "Prediction"]], on=["Time", "Fold"])
            d["bad"] = d.groupby("Fold")["Actual"].transform(lambda s: s >= s.quantile(quantile))
            d["stratum"] = d.groupby("Fold")["Prediction"].transform(
                lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
            for fold, g in [(str(k), g) for k, g in d.groupby("Fold")] + [("All", d)]:
                bad = g["bad"].to_numpy(bool)
                within = [auc(bad[m], g["score"].to_numpy()[m])
                          for m in (g["stratum"].to_numpy() == k for k in range(5))]
                within = [v for v in within if np.isfinite(v)]
                rows.append({
                    "Horizon": label, "Detector": ANOMALY_LABEL.get(which, which), "Fold": fold, "N": len(g),
                    "AUC_score": auc(bad, g["score"].to_numpy()),
                    "AUC_base": auc(bad, g["Prediction"].to_numpy()),
                    "AUC_score_within_base": float(np.mean(within)) if within else np.nan,
                    "Bad_rate_flagged": float(bad[g["flag"].to_numpy() == 1].mean()) if (g["flag"] == 1).any() else np.nan,
                    "Bad_rate": float(bad.mean()),
                })
    out = pd.DataFrame(rows)
    out["Horizon"] = pd.Categorical(out["Horizon"], HORIZON_ORDER, ordered=True)
    return out.sort_values(["Horizon", "Detector", "Fold"]).reset_index(drop=True)


# ---------------------------------------------------------------- interpretation figures
#
# These work on any run (vol or drawdown targets). Colours: green = significantly better than the
# comparison (Diebold-Mariano p < 0.05), red = significantly worse, grey = not distinguishable.

_GOOD, _BAD, _NEUTRAL = "#2E8B57", "#C8553D", "#8C8C8C"


def as_pairs(comparisons) -> list[tuple[str, str]]:
    """[(model, [versus, ...]), ...] -> [(model, versus), ...] for plot_pair_skill."""
    return [(m, v) for m, versus in comparisons for v in versus]


def anomaly_pairs() -> list[tuple[str, str]]:
    """Each anomaly model against its base (both detectors), then signature vs signature-free directly."""
    pairs = [(tmpl.format(w=w), base) for _, base, tmpl in ANOMALY_USES for w in ("sig", "free")]
    pairs += [(tmpl.format(w="sig"), tmpl.format(w="free")) for tmpl in dict.fromkeys(t for _, _, t in ANOMALY_USES)]
    return pairs


def _bootstrap_pct(e_model, e_ref, block: int, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """90% circular block-bootstrap interval for the % RMSE improvement of model over ref. Blocks keep
    the autocorrelation that overlapping targets create."""
    e2m, e2r = np.asarray(e_model, float) ** 2, np.asarray(e_ref, float) ** 2
    n = len(e2m)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]
    pct = 100 * (1 - np.sqrt(e2m[idx].mean(1)) / np.sqrt(e2r[idx].mean(1)))
    return float(np.quantile(pct, 0.05)), float(np.quantile(pct, 0.95))


def skill_frame(res: Results, pairs, n_boot: int = 1000) -> pd.DataFrame:
    """% RMSE improvement of each (model, versus) pair with a block-bootstrap 90% interval and the DM p."""
    rows = []
    for hk, meta in res.meta.items():
        _, _, err = _pooled(res, hk)
        h = meta["horizon_rows"]
        for model, versus in pairs:
            if model not in err or versus not in err:
                continue
            em, ev = err[model].to_numpy(), err[versus].to_numpy()
            lo, hi = _bootstrap_pct(em, ev, block=max(2 * h, 20), n_boot=n_boot)
            dm, pv = diebold_mariano(em, ev, h)
            rows.append({
                "Horizon": HORIZON_LABELS[hk], "Model": model, "Versus": versus,
                "Pct_better": 100 * (1 - np.sqrt((em**2).mean()) / np.sqrt((ev**2).mean())),
                "CI_low": lo, "CI_high": hi, "DM_stat": dm, "DM_p": pv,
            })
    return pd.DataFrame(rows)


def plot_pair_skill(res: Results, pairs, title: str = "", n_boot: int = 1000, label_versus: bool = True):
    """Forest plot: per horizon, each model's % RMSE improvement over its comparison with a 90%
    interval. A bar crossing 0 means the data can't tell the two apart. The axis is clipped so one
    blown-up model doesn't flatten the rest; anything beyond it is drawn as an arrow at the edge
    with its value written next to it."""
    f = skill_frame(res, pairs, n_boot)
    if f.empty:
        raise ValueError("None of these model pairs are in these results")
    hz = _horizons(f)
    n_rows = f.groupby("Horizon").size().max()
    width, per_row = (7.0, 0.62) if label_versus else (5.2, 0.42)
    fig, axes = plt.subplots(1, len(hz), figsize=(width * len(hz), per_row * n_rows + 1.6), squeeze=False)
    for ax, h in zip(axes[0], hz):
        g = f[f.Horizon == h].sort_values("Pct_better").reset_index(drop=True)
        lim = max(5.0, 1.25 * float(np.nanpercentile(np.abs(g[["CI_low", "CI_high"]].to_numpy()), 75)))
        colors = [_GOOD if (p < 0.05 and v > 0) else _BAD if (p < 0.05 and v < 0) else _NEUTRAL
                  for p, v in zip(g.DM_p, g.Pct_better)]
        y = np.arange(len(g))
        ax.hlines(y, g.CI_low.clip(-lim, lim), g.CI_high.clip(-lim, lim), colors=colors, lw=2.2)
        inside = g.Pct_better.abs() <= lim
        ax.scatter(g.Pct_better[inside], y[inside], c=np.asarray(colors)[inside], s=36, zorder=3)
        for k in np.where(~inside)[0]:
            v = g.Pct_better[k]
            ax.scatter(np.sign(v) * lim, k, marker=">" if v > 0 else "<", c=colors[k], s=60, zorder=3)
            ax.annotate(f"{v:.0f}%", (np.sign(v) * lim, k), xytext=(-4 if v > 0 else 4, 6),
                        textcoords="offset points", ha="right" if v > 0 else "left", fontsize=7)
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.axvline(0, color="black", lw=0.8)
        labels = [f"{m}\nvs {v}" if label_versus else m for m, v in zip(g.Model, g.Versus)]
        ax.set_yticks(y, labels, fontsize=7 if label_versus else 8)
        ax.set(title=h, xlabel="% RMSE better (90% CI)")
    fig.suptitle(title or "Improvement over the comparison model (green / red = significant, DM p < 0.05)")
    fig.tight_layout()
    return fig


def plot_skill(res: Results, models=None, ref: str = "HAR-RV-L", n_boot: int = 1000):
    """Every model against one reference (HAR-RV-L by default) with uncertainty: the ranking table
    with error bars. Overlapping intervals between two models mean their order is not reliable."""
    models = models or [m for m in res.predictions.Model.unique() if m not in (ref, "Mean")]
    return plot_pair_skill(res, [(m, ref) for m in models], n_boot=n_boot, label_versus=False,
                           title=f"% RMSE improvement over {ref} (90% block-bootstrap CI; green = DM p < 0.05)")


LEARNERS = ("Ridge", "XGBoost", "MLP")
FEATURE_SETS = ("signature", "lag bank (matched)", "intraday signature", "intraday lag bank (matched)")


def plot_learner_grid(res: Results, ref: str = "HAR-RV-L"):
    """Learner x feature set, % RMSE improvement over ``ref``. Reading across a row: does the signature
    beat its matched lag bank for this learner? Down a column: does a nonlinear learner get more out of
    the same features? Blank cells were not run."""
    summ = summary_table(res)
    hz = [h for h in _horizons(res.metrics)]
    fig, axes = plt.subplots(1, len(hz), figsize=(5.5 * len(hz), 3.2), squeeze=False)
    for ax, h in zip(axes[0], hz):
        s = summ[summ.Horizon == h].set_index("Model")
        base = s.loc[ref, "Mean_RMSE"] if ref in s.index else np.nan
        grid = pd.DataFrame(
            [[100 * (1 - s.loc[f"{l} | {f}", "Mean_RMSE"] / base) if f"{l} | {f}" in s.index else np.nan
              for f in FEATURE_SETS] for l in LEARNERS],
            index=LEARNERS, columns=[f.replace(" (matched)", "") for f in FEATURE_SETS])
        vals = np.abs(grid.to_numpy()[np.isfinite(grid.to_numpy())])
        lim = max(5.0, float(np.percentile(vals, 75))) if vals.size else 5.0  # colours clipped, numbers exact
        sns.heatmap(grid, annot=True, fmt=".1f", center=0, vmin=-lim, vmax=lim, cmap="RdYlGn", cbar=False, ax=ax)
        ax.set(title=f"{h}: % better than {ref}", xlabel="", ylabel="")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_calibration(res: Results, models=("HAR-RV-L", "HAR-RV-L calibrated", "Ridge | signature"), bins: int = 10):
    """Binned forecast vs outcome (deciles of each model's forecast, log scale). Points above the
    diagonal = the model under-forecasts in that range, below = over-forecasts. A model that is
    systematically off can be beaten by a simple rescaling, which is what 'HAR-RV-L calibrated' does."""
    hz = _horizons(res.predictions)
    fig, axes = plt.subplots(1, len(hz), figsize=(4.6 * len(hz), 4.2), squeeze=False)
    for ax, h in zip(axes[0], hz):
        sub = res.predictions[res.predictions.Horizon == h]
        lo, hi = np.inf, -np.inf
        for m in models:
            s = sub[sub.Model == m]
            if s.empty:
                continue
            q = pd.qcut(s.Prediction.rank(method="first"), bins, labels=False)
            g = s.groupby(q).agg(pred=("Prediction", "mean"), act=("Actual", "mean"))
            ax.plot(g.pred, g.act, marker="o", ms=4, lw=1.2, label=m)
            lo, hi = min(lo, g.min().min()), max(hi, g.max().max())
        if np.isfinite(lo):
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=0.8)
            ax.set(xscale="log", yscale="log")
        ax.set(title=h, xlabel="Mean forecast (decile)", ylabel=f"Mean realized {_target_label(res).lower()}")
        ax.legend(fontsize=7)
    fig.suptitle("Calibration: are forecasts the right size? (dashed = perfect)")
    fig.tight_layout()
    return fig


def plot_error_over_time(res: Results, models, ref: str = "HAR-RV-L", window: int = 250):
    """Rolling RMSE ratio model / ref over the out-of-sample period (below 1 = model better). Shows
    whether a pooled gain is steady or comes from a few episodes."""
    hz = _horizons(res.predictions)
    fig, axes = plt.subplots(len(hz), 1, figsize=(14, 2.7 * len(hz)), constrained_layout=True, squeeze=False)
    for ax, h in zip(axes[:, 0], hz):
        hk = next(k for k in res.meta if HORIZON_LABELS[k] == h)
        _, actual, err = _pooled(res, hk)
        if ref not in err:
            continue
        r = (err[ref] ** 2).rolling(window, min_periods=window // 2).mean()
        for m in models:
            if m in err:
                ratio = np.sqrt((err[m] ** 2).rolling(window, min_periods=window // 2).mean() / r)
                ax.plot(pd.to_datetime(ratio.index), ratio.to_numpy(), lw=1.1, label=m)
        ax.axhline(1, color="black", lw=0.8)
        ax2 = ax.twinx()
        ax2.fill_between(pd.to_datetime(actual.index), 0, actual.to_numpy(), color="grey", alpha=0.15, lw=0)
        ax2.set_yticks([])
        ax.set(title=f"{h}: rolling {window}-row RMSE relative to {ref} (grey = realized {_target_label(res).lower()})",
               ylabel="RMSE ratio")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    return fig


def plot_anomaly_response(res: Results, base: str = "HAR-RV-L calibrated", bins: int = 10):
    """The most direct picture of whether the anomaly score carries information. Top row: the base
    model's log error, log(actual / forecast), by decile of the anomaly score. A rising line means
    rows the detector finds unusual are followed by outcomes larger than the base model expects,
    which is information the forecast is missing. Flat = nothing to add. Bottom row: share of rows
    whose outcome lands in its fold's top 20%, by score decile (dashed = the 20% base rate)."""
    if res.anomaly_flags.empty:
        raise ValueError("No anomaly results; run the *_i horizons with use_anomaly=True")
    hz = _horizons(res.anomaly_flags)
    fig, axes = plt.subplots(2, len(hz), figsize=(5.5 * len(hz), 7), squeeze=False)
    for j, h in enumerate(hz):
        p = res.predictions[(res.predictions.Horizon == h) & (res.predictions.Model == base)]
        for which, af in res.anomaly_flags[res.anomaly_flags.Horizon == h].groupby("Which"):
            d = af.merge(p[["Time", "Fold", "Actual", "Prediction"]], on=["Time", "Fold"])
            if d.empty:
                continue
            d["log_err"] = np.log(d.Actual / d.Prediction)
            d["bad"] = d.groupby("Fold")["Actual"].transform(lambda s: s >= s.quantile(0.8))
            d["decile"] = pd.qcut(d.score.rank(method="first"), bins, labels=False) + 1
            g = d.groupby("decile").agg(m=("log_err", "mean"), se=("log_err", "sem"), bad=("bad", "mean"))
            lab = ANOMALY_LABEL.get(which, which)
            axes[0, j].errorbar(g.index, g.m, yerr=1.96 * g.se, marker="o", ms=4, capsize=2, label=lab)
            axes[1, j].plot(g.index, g.bad, marker="o", ms=4, label=lab)
        axes[0, j].axhline(0, color="black", lw=0.8)
        axes[0, j].set(title=h, xlabel="Anomaly score decile (10 = most unusual)", ylabel=f"log(actual / {base})")
        axes[1, j].axhline(0.2, color="black", ls="--", lw=0.8)
        axes[1, j].set(xlabel="Anomaly score decile", ylabel="Share in top 20% of outcomes")
        axes[0, j].legend(fontsize=8)
    fig.suptitle("Does a high anomaly score precede outcomes the forecast misses?")
    fig.tight_layout()
    return fig


def plot_anomaly_timeline(res: Results):
    """Realized outcome through the out-of-sample period with each detector's flagged rows marked. A
    useful flag fires in a minority of coherent episodes, around the spikes, and a similar share of
    the time in every fold (a flag on most of the time means the score drifted out of its training
    range)."""
    if res.anomaly_flags.empty:
        raise ValueError("No anomaly results; run the *_i horizons with use_anomaly=True")
    hz = _horizons(res.anomaly_flags)
    fig, axes = plt.subplots(len(hz), 1, figsize=(14, 2.8 * len(hz)), constrained_layout=True, squeeze=False)
    colors = {"sig": "#C8553D", "free": "#276FBF"}
    for ax, h in zip(axes[:, 0], hz):
        a = res.predictions[res.predictions.Horizon == h].drop_duplicates("Time").sort_values("Time")
        ax.plot(pd.to_datetime(a.Time), a.Actual, color="black", lw=0.9)
        top = float(a.Actual.max())
        for k, (which, af) in enumerate(res.anomaly_flags[res.anomaly_flags.Horizon == h].groupby("Which")):
            fl = af.sort_values("Time")
            on = fl[fl.flag == 1]
            ax.scatter(pd.to_datetime(on.Time), np.full(len(on), top * (1.04 + 0.06 * k)), marker="|", s=60,
                       color=colors.get(which, None),
                       label=f"{ANOMALY_LABEL.get(which, which)} flag ({100 * fl.flag.mean():.0f}% of rows)")
        ax.set(title=h, ylabel=_target_label(res))
        ax.legend(loc="upper left", fontsize=8)
    return fig
