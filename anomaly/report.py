"""Tables and figures for the anomaly study (reads the files written by run.py)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import EVENTS, HORIZONS

KIND_ORDER = ["signature", "lag bank", "stats", "raw", "baseline"]


@dataclass
class Results:
    metrics: pd.DataFrame
    scores: pd.DataFrame
    selection: dict  # horizon -> selection.json
    meta: dict


def load_results(output_dir) -> Results:
    root = Path(output_dir)
    metrics, scores, selection, meta = [], [], {}, {}
    for hk in HORIZONS:
        d = root / hk
        if not (d / "meta.json").exists():
            continue
        meta[hk] = json.loads((d / "meta.json").read_text())
        selection[hk] = json.loads((d / "selection.json").read_text())
        m = pd.read_csv(d / "metrics.csv")
        m.insert(0, "Horizon", hk)
        metrics.append(m)
        s = pd.read_parquet(d / "scores.parquet")
        s.insert(0, "Horizon", hk)
        scores.append(s)
    if not metrics:
        raise FileNotFoundError(f"No completed horizons under {root}")
    return Results(pd.concat(metrics, ignore_index=True), pd.concat(scores, ignore_index=True), selection, meta)


def _summary(res: Results) -> pd.DataFrame:
    rows = []
    for hk, m in res.metrics.groupby("Horizon"):
        last = m.Fold.max()
        g = m.groupby(["Config", "Kind", "Detector"])
        sel = m[m.Fold < last].groupby(["Config", "Kind", "Detector"]).mean(numeric_only=True)
        conf = m[m.Fold == last].set_index(["Config", "Kind", "Detector"])
        t = pd.DataFrame({
            "Select (folds 1-2)": sel["AUC_surprise_strat"],
            "Confirm (last fold)": conf["AUC_surprise_strat"],
            "AUC_surprise": g["AUC_surprise"].mean(), "Lift": g["Lift"].mean(),
            "AUC_level": g["AUC_level"].mean(), "Fwd_vol_ratio": g["Fwd_vol_ratio"].mean(),
            "n_features": g["n_features"].first(),
        }).reset_index()
        t.insert(0, "Horizon", hk)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def leaderboard(res: Results, top: int = 10) -> pd.DataFrame:
    """Best configs per horizon by stratified surprise AUC on the selection folds."""
    s = _summary(res).sort_values(["Horizon", "Select (folds 1-2)"], ascending=[True, False])
    return s.groupby("Horizon").head(top).reset_index(drop=True)


def kind_table(res: Results) -> pd.DataFrame:
    """Best selection-fold score per representation kind x detector (rows: horizon/kind)."""
    s = _summary(res)
    t = s.groupby(["Horizon", "Kind", "Detector"])["Select (folds 1-2)"].max().unstack("Detector")
    return t.reindex([(h, k) for h in res.meta for k in KIND_ORDER if (h, k) in t.index])


def gate_table(res: Results) -> pd.DataFrame:
    rows = []
    for hk, sel in res.selection.items():
        b, n, g = sel["best_signature"], sel["best_signature_free"], sel["gate"]
        boot = g["bootstrap_last_fold"]
        rows.append({
            "Horizon": hk, "Best signature": f"{b['Config']} [{b['Detector']}]",
            "Sig select": b["Select"], "Sig confirm": b["Confirm"],
            "Best signature-free": f"{n['Config']} [{n['Detector']}]",
            "Free select": n["Select"], "Free confirm": n["Confirm"],
            "Last-fold diff": boot["diff"], "CI low": boot["ci_low"], "CI high": boot["ci_high"],
            "p (sig no better)": boot["p_value"], "Gate passed": g["passed"],
        })
    return pd.DataFrame(rows)


def synthetic_table(res: Results) -> pd.DataFrame:
    rows = []
    for hk, sel in res.selection.items():
        for which, d in sel["synthetic_order_only"].items():
            rows.append({"Horizon": hk, "Which": which, "Config": d["Config"], "Detector": d["Detector"],
                         "AUC_injected": d.get("AUC_injected")})
    return pd.DataFrame(rows)


def plot_kind_heatmap(res: Results):
    t = kind_table(res)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.5))
    sns.heatmap(t, annot=True, fmt=".3f", center=0.5, cmap="RdYlGn", ax=ax,
                cbar_kws={"label": "stratified surprise AUC (0.5 = nothing beyond HAR)"})
    ax.set(title="Best config per representation and detector (selection folds)", xlabel="", ylabel="")
    fig.tight_layout()
    return fig


def plot_timeline(res: Results, horizon: str = "1d"):
    """Out-of-sample scores of the selected signature and signature-free detectors (percentile
    within each fold), with the labeled crises shaded."""
    sel = res.selection[horizon]
    s = res.scores[res.scores.Horizon == horizon]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    for which, color in (("best_signature", "#276FBF"), ("best_signature_free", "#C8553D")):
        c = sel[which]
        z = s[(s.Config == c["Config"]) & (s.Detector == c["Detector"])].sort_values("Time").copy()
        z["pct"] = z.groupby("Fold").Score.rank(pct=True)
        ax.plot(pd.to_datetime(z.Time), z.pct.rolling(5, min_periods=1).mean(), lw=0.8, color=color,
                label=f"{which.replace('_', ' ')}: {c['Config']} [{c['Detector']}]")
    for name, (a, b) in EVENTS.items():
        ax.axvspan(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), color="grey", alpha=0.15)
    ax.set(ylabel="score percentile (5-row mean)", title=f"{HORIZONS[horizon].label}: anomaly scores, crises shaded")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig
