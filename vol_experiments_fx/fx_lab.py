"""Shared setup for the two FX notebooks in this folder, plus the volatility-direction target.

Both notebooks (``vol_fx.ipynb``, ``anomaly_fx.ipynb``) import their settings from here, so the
data, the end date and the channels are the same everywhere:

* EUR/USD hourly bars from Dukascopy, with the bid-ask spread (``spread=True``), from 2012 to the
  end of the last finished month. Stopping at a finished month means every month is a cached
  candle file: nothing falls back to the slow, rate-limited tick files.
* New channels (``forecast_data.fx_hourly_channels``): time-of-day-adjusted returns and ranges,
  and the abnormal spread. All use past data only.
* Horizons: 1 and 5 trading days, signatures from the hourly path.

The direction target asks whether volatility over the next h days will be higher than over the
last h days. It is scored two ways: from the volatility forecasts (forecast above the past level
= "up"), and with logistic classifiers on the same features.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "Mihika"):  # the anomaly package and Mihika's modules
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import experiment as ex  # noqa: E402
import reporting as rp  # noqa: E402
from anomaly.config import AnomalyConfig  # noqa: E402
from forecast_data import duration_steps, fetch_dukascopy_hourly, fx_hourly_channels  # noqa: E402

LOGGER = logging.getLogger(__name__)

RESULTS = HERE / "results"
DATA_DIR = ROOT / "Mihika" / "data"  # the Dukascopy download cache shared with the other notebooks
CACHE_DIR = HERE / "cache"  # feature datasets built by this folder's runs
SYMBOL = "EURUSD"
START = "2012-01-01"


def last_full_month_end(today: pd.Timestamp | None = None) -> str:
    """Last day of the previous month (UTC): the end date of every run here."""
    today = pd.Timestamp.now(tz="UTC") if today is None else pd.Timestamp(today)
    return str((today.normalize().replace(day=1) - pd.Timedelta(days=1)).date())


END = last_full_month_end()
HORIZONS = ("1d_i", "5d_i")  # vol horizons: daily-bar targets and baselines, hourly-path signatures

# Hourly path channels: time-of-day-adjusted price and range, Dukascopy volume, abnormal spread
MAIN_DIMS = ("price_sa", "activity", "range_sa", "spread")
CHANNEL_ABLATIONS = {
    "Original (price, activity, range)": ("price", "activity", "range"),
    "Adjusted price only": ("price_sa",),
    "Adjusted, no spread": ("price_sa", "activity", "range_sa"),
    "Adjusted + spread (main)": MAIN_DIMS,
}
CHANNEL_REFERENCE = "Original (price, activity, range)"

BASELINES = ("Mean", "HAR-RV-L", "HAR-RV-L calibrated", "HAR-RV-L (intraday RV)", "Ridge | stats")
SIGNATURE_MODELS = (
    "Ridge | intraday signature", "Ridge | intraday lag bank (matched)", "Ridge | intraday multiscale stats",
    "Ridge | intraday signature + HAR-RV", "MLP | intraday signature", "MLP | intraday lag bank (matched)",
)
# (model, [compared against]): is it the signature, the learner, or just the hourly window?
VOL_COMPARISONS = [
    ("Ridge | intraday signature", [
        "Ridge | intraday lag bank (matched)", "Ridge | intraday multiscale stats", "HAR-RV-L (intraday RV)",
        "HAR-RV-L calibrated"]),
    ("Ridge | intraday signature + HAR-RV", ["HAR-RV-L (intraday RV)"]),
    ("MLP | intraday signature", ["MLP | intraday lag bank (matched)", "Ridge | intraday signature"]),
    ("HAR-RV-L (intraday RV)", ["HAR-RV-L"]),
]
# Anomaly score in the forecasts: as a HAR feature and as a residual correction (gating dropped:
# it fits separate coefficients on a few dozen flagged rows and was unstable).
ANOMALY_MODELS = tuple(f"{m} [{w}]" for m in ("HAR-RV-L + anomaly", "HAR-RV-L + anomaly residual")
                       for w in ex.ANOMALY_VARIANTS)

# Anomaly study grid: channels are the only axis; the rest is fixed in advance
ANOMALY_CHANNELS = ("price", "price+volume+range+flow", "price_sa", "price_sa+spread",
                    "price_sa+volume+range_sa+spread")


# ---------------------------------------------------------------- configs


def vol_config(quick: bool = False, **changes) -> ex.ExperimentConfig:
    cfg = ex.ExperimentConfig(
        asset="fx", fx_symbol=SYMBOL, fx_start=START, intraday_end=END, fx_channels=True,
        horizons=HORIZONS, n_folds=3, target_rows={"1d": None, "5d": None},
        models=BASELINES + SIGNATURE_MODELS,
        spec_overrides={h: {"intraday_dims": MAIN_DIMS} for h in HORIZONS},
        run_ablations=True, intraday_ablation_dims=CHANNEL_ABLATIONS,
        use_flag=False, use_anomaly=False,
        data_dir=str(DATA_DIR), cache_dir=str(CACHE_DIR), output_dir=str(RESULTS / "vol"),
    )
    if quick:  # a short history and fewer epochs, to check that everything runs
        cfg = replace(cfg, fx_start="2021-01-01", torch_epochs=25, output_dir=cfg.output_dir + "_quick")
    return replace(cfg, **changes)


def anomaly_config(quick: bool = False, **changes) -> AnomalyConfig:
    cfg = AnomalyConfig(
        asset="fx", symbol=SYMBOL, fx_channels=True, horizons=("1d", "5d"), start=START, end=END,
        target_rows={"1d": None, "5d": None},
        channel_sets=ANOMALY_CHANNELS, normalizations=("trailing_vol",), window_mults=(1.0,),
        lead_lag=(True,), depths=(2, 3), stage_b_detectors=("hmm",), stage_b_top_k=3,
        data_dir=str(DATA_DIR), output_dir=str(RESULTS / "anomaly"),
    )
    if quick:
        cfg = replace(cfg, start="2021-01-01", n_bootstrap=100, output_dir=cfg.output_dir + "_quick")
    return replace(cfg, **changes)


def load_hourly(start: str = START, end: str = END) -> pd.DataFrame:
    """The hourly bars every run here uses, with the spread and the adjusted channels."""
    bars = fetch_dukascopy_hourly(start, end, symbol=SYMBOL, data_dir=DATA_DIR / "dukascopy", spread=True)
    return fx_hourly_channels(bars)


# ---------------------------------------------------------------- tables for the anomaly study


def channel_table(res) -> pd.DataFrame:
    """Anomaly study: the best signature config of each channel set (any depth), per horizon and
    detector. ``Select`` is the stratified surprise AUC on the selection folds, ``Confirm`` on the
    last fold (0.5 = no information beyond the volatility forecast)."""
    m = res.metrics[res.metrics.Kind == "signature"].copy()
    m["Channels"] = m.Config.str.split(" | ", regex=False).str[1].str.split("|", regex=False).str[0]
    rows = []
    for (hk, ch, det), g in m.groupby(["Horizon", "Channels", "Detector"]):
        last = res.metrics[res.metrics.Horizon == hk].Fold.max()
        sel = g[g.Fold < last].groupby("Config").AUC_surprise_strat.mean()
        best = sel.idxmax()
        rows.append({"Horizon": hk, "Channels": ch, "Detector": det, "Best config": best, "Select": sel.max(),
                     "Confirm": float(g[(g.Fold == last) & (g.Config == best)].AUC_surprise_strat.mean())})
    order = {c: i for i, c in enumerate(ANOMALY_CHANNELS)}
    out = pd.DataFrame(rows)
    return out.sort_values(["Horizon", "Detector", "Channels"], key=lambda c: c.map(order) if c.name == "Channels" else c) \
        .reset_index(drop=True)


# ---------------------------------------------------------------- data checks


def plot_seasonality(hourly: pd.DataFrame):
    """Mean |return|, range and log spread by New York hour, before and after the adjustment. After
    it, the return and range lines should be roughly flat; the spread line is centred on zero."""
    import matplotlib.pyplot as plt

    idx = pd.DatetimeIndex(hourly.index).tz_convert("America/New_York")
    hour = pd.Series(idx.hour, index=hourly.index)
    r = np.log(hourly["close"]).diff().abs()
    hl = np.log(hourly["high"] / hourly["low"])
    panels = [
        ("|return|", r, hourly["ret_sa"].abs()),
        ("range", hl, hourly["range_sa"]),
        ("log spread", np.log(hourly["spread"].clip(lower=1e-7)), hourly["spread_sa"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, (name, raw, adj) in zip(axes, panels):
        raw_g, adj_g = raw.groupby(hour).mean(), adj.groupby(hour).mean()
        if name == "log spread":
            ax.plot(raw_g.index, raw_g - raw_g.mean(), marker="o", ms=3, label="raw (centred)")
            ax.plot(adj_g.index, adj_g, marker="o", ms=3, label="abnormal spread (spread_sa)")
        else:
            ax.plot(raw_g.index, raw_g / raw_g.mean(), marker="o", ms=3, label="raw")
            ax.plot(adj_g.index, adj_g / adj_g.mean(), marker="o", ms=3, label="adjusted")
            ax.axhline(1, color="black", lw=0.8)
        ax.set(title=name, xlabel="New York hour", xticks=range(0, 24, 3))
        ax.legend(fontsize=8)
    fig.suptitle("Time-of-day pattern before and after the adjustment (relative to the daily mean)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- tables for the vol run


def ablation_table(res: rp.Results) -> pd.DataFrame:
    """Fold-mean RMSE of Ridge on the intraday signature under each ablation setting, as % better
    than HAR-RV-L and than the reference setting of its group (level 1 only / raw path / the
    original channels). Same rows and folds throughout."""
    a = res.ablations
    if a.empty:
        raise ValueError("No ablations in these results (run_ablations / intraday_ablation_dims).")
    g = a.groupby(["Horizon", "Experiment", "Setting"], sort=False)["RMSE"].mean().rename("Mean_RMSE").reset_index()
    har = res.metrics[res.metrics.Model == "HAR-RV-L"].groupby("Horizon")["RMSE"].mean()
    g["Pct_vs_HAR"] = 100 * (1 - g.Mean_RMSE / g.Horizon.map(har))
    refs = {"Signature level": "L1-1", "Augmentation": "Raw path", "Channels": CHANNEL_REFERENCE}
    ref = g[g.apply(lambda r: r.Setting == refs.get(r.Experiment), axis=1)].set_index(["Horizon", "Experiment"]).Mean_RMSE
    g["Pct_vs_group_ref"] = 100 * (1 - g.Mean_RMSE / pd.Series(list(zip(g.Horizon, g.Experiment))).map(ref).to_numpy())
    return g


def compare_with_previous(res: rp.Results, previous_dir: str | Path = ROOT / "Mihika" / "results" / "fx" / "intraday"):
    """The previous FX intraday run (original channels, end = the day it ran) next to this one, for the
    models both have: % better than HAR-RV-L. The rows differ (different end dates), so compare the
    levels, not row by row; the like-for-like channel comparison is the ablation table."""
    previous_dir = Path(previous_dir)
    if not previous_dir.exists() or not any((d / "meta.json").exists() for d in previous_dir.iterdir()):
        LOGGER.info("No previous FX run at %s", previous_dir)
        return None
    old = rp.summary_table(rp.load_results(previous_dir))
    new = rp.summary_table(res)
    both = old.merge(new, on=["Horizon", "Model"], suffixes=("_previous", "_new"))
    return both[["Horizon", "Model", "Pct_vs_HAR_previous", "Pct_vs_HAR_new", "Mean_RMSE_previous", "Mean_RMSE_new"]]


# ---------------------------------------------------------------- volatility direction


def horizon_dataset(cfg: ex.ExperimentConfig, horizon: str):
    """(dataset, spec, folds, horizon_steps) exactly as the vol run built them (read from its cache)."""
    bars = ex.load_bars(horizon, cfg)
    spec = ex.base_spec_for(horizon, bars, cfg)
    intraday = ex.load_intraday(cfg)
    ds = ex.get_dataset(bars, spec, cfg, include_seq=False,
                        sub_window=ex.per_horizon(cfg.lstm_sub_window, horizon), intraday=intraday)
    folds = ex.purged_walk_forward(ds.times, ds.target_end, cfg.n_folds, cfg.min_train_rows)
    return ds, spec, folds, duration_steps(spec.horizon, spec.bar_freq)


def past_vol(ds, spec, horizon_steps: int) -> np.ndarray:
    """Realized volatility over the h bars up to each row, on the target's scale: the HAR feature for
    the look-back equal to the horizon is log(mean variance per bar), so sqrt(exp(.) * h)."""
    k = list(spec.har_windows).index(spec.horizon)
    return np.sqrt(np.exp(ds.X_har[:, k].astype(float)) * horizon_steps)


def _logit(X_tr, y_tr, times, target_end):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=3000))])
    inner = ex.purged_walk_forward(times, target_end, 5, min_train_rows=50, strict=False)
    inner = [(f.train, f.test) for f in inner if 0 < y_tr[f.train].mean() < 1 and 0 < y_tr[f.test].mean() < 1]
    if not inner:
        return pipe.set_params(logit__C=1.0).fit(X_tr, y_tr)
    search = GridSearchCV(pipe, {"logit__C": np.logspace(-4, 1, 6)}, scoring="neg_log_loss", cv=inner, n_jobs=1)
    return search.fit(X_tr, y_tr).best_estimator_


DIRECTION_FEATURES = {  # logistic classifiers: HAR features plus one hourly-window representation
    "Logit | HAR": lambda d: d.X_har,
    "Logit | HAR + intraday HAR-RV": lambda d: np.hstack([d.X_har, d.X_harrv]),
    "Logit | HAR + intraday multiscale stats": lambda d: np.hstack([d.X_har, d.X_imulti]),
    "Logit | HAR + intraday lag bank (matched)": lambda d: np.hstack([d.X_har, d.X_ilag]),
    "Logit | HAR + intraday signature": lambda d: np.hstack([d.X_har, d.X_isig]),
}


def run_direction(cfg: ex.ExperimentConfig, output_dir: str | Path | None = None) -> Path:
    """Direction of volatility, per horizon: up = realized vol over the next h days above the last h days.

    Writes ``direction.parquet`` per horizon with, for every test row and model, ``score`` (higher =
    more likely up) and ``p_up`` where the model gives a probability:
    * the vol forecasts of the vol run, scored as log(forecast / past vol) (no refit);
    * logistic classifiers (DIRECTION_FEATURES), refit per fold on the same purged folds, with C
      chosen by an inner purged walk-forward CV on log loss.
    Finished horizons are skipped."""
    out_root = Path(output_dir or Path(cfg.output_dir).parent / "direction")
    vol = rp.load_results(cfg.output_dir)
    for horizon in cfg.horizons:
        out = out_root / horizon
        if (out / "meta.json").exists():
            LOGGER.info("[%s] direction already complete, skipping", horizon)
            continue
        out.mkdir(parents=True, exist_ok=True)
        ds, spec, folds, h = horizon_dataset(cfg, horizon)
        past = past_vol(ds, spec, h)
        up = (ds.y > past).astype(int)
        frames = []
        # 1) from the vol forecasts already made
        label = ex.HORIZON_LABELS[horizon]
        p = vol.predictions[vol.predictions.Horizon == label]
        pos = pd.Series(np.arange(len(ds.times)), index=pd.DatetimeIndex(ds.times))
        i = pos.reindex(pd.DatetimeIndex(pd.to_datetime(p.Time, utc=True))).to_numpy()
        if np.isnan(i).any():
            raise RuntimeError(f"[{horizon}] vol predictions do not match the dataset rows")
        i = i.astype(int)
        frames.append(pd.DataFrame({
            "Horizon": label, "Fold": p.Fold.to_numpy(), "Time": ds.times[i], "Model": p.Model.to_numpy(),
            "Source": "vol forecast", "score": np.log(p.Prediction.to_numpy() / past[i]), "p_up": np.nan, "up": up[i],
        }))
        # 2) classifiers
        for fold in folds:
            tr, te = fold.train, fold.test
            for name, feats in DIRECTION_FEATURES.items():
                X = feats(ds).astype(float)
                model = _logit(X[tr], up[tr], ds.times[tr], ds.target_end[tr])
                prob = model.predict_proba(X[te])[:, 1]
                frames.append(pd.DataFrame({
                    "Horizon": label, "Fold": fold.number, "Time": ds.times[te], "Model": name,
                    "Source": "classifier", "score": prob, "p_up": prob, "up": up[te],
                }))
            LOGGER.info("[%s] direction fold %d/%d done", horizon, fold.number, len(folds))
        pd.concat(frames, ignore_index=True).to_parquet(out / "direction.parquet", index=False)
        meta = {"horizon": horizon, "horizon_rows": vol.meta[horizon]["horizon_rows"],
                "up_rate": float(up.mean()), "n_rows": int(len(up))}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_root


def load_direction(root: str | Path) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    dirs = sorted(d for d in root.iterdir() if (d / "meta.json").exists())
    if not dirs:
        raise FileNotFoundError(f"No direction results under {root}")
    return (pd.concat([pd.read_parquet(d / "direction.parquet") for d in dirs], ignore_index=True),
            {d.name: json.loads((d / "meta.json").read_text()) for d in dirs})


def _block_idx(n: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]


def direction_table(df: pd.DataFrame, meta: dict, ref: str = "Logit | HAR", n_boot: int = 500) -> pd.DataFrame:
    """Per horizon and model, pooled over the out-of-sample rows: AUC, hit rate (sign of the score or
    p_up > 0.5 against the outcome), log loss and Brier score (classifiers only), the fold-mean AUC,
    and the AUC difference against ``ref`` with a 90% circular block-bootstrap interval (blocks of
    2 x the horizon in rows, at least 10) and a one-sided p-value (share of draws <= 0)."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    rows = []
    for hk, m in meta.items():
        label = ex.HORIZON_LABELS[hk]
        d = df[df.Horizon == label]
        wide = d.pivot_table(index="Time", columns="Model", values="score").sort_index()
        y = d.drop_duplicates("Time").set_index("Time")["up"].reindex(wide.index).to_numpy()
        block = max(10, 2 * int(m["horizon_rows"]))
        idx = _block_idx(len(y), block, n_boot, seed=0)
        ok = [k for k in range(n_boot) if 0 < y[idx[k]].mean() < 1]
        idx = idx[ok]
        ref_s = wide[ref].to_numpy() if ref in wide else None
        ref_boot = np.array([roc_auc_score(y[k], ref_s[k]) for k in idx]) if ref_s is not None else None
        for model in wide.columns:
            s = wide[model].to_numpy()
            src = d[d.Model == model].Source.iloc[0]
            g = d[d.Model == model]
            fold_auc = g.groupby("Fold").apply(lambda x: roc_auc_score(x.up, x.score) if 0 < x.up.mean() < 1 else np.nan)
            pred_up = (s > 0.5) if src == "classifier" else (s > 0)
            row = {"Horizon": label, "Model": model, "Source": src, "N": len(y), "Up_rate": float(y.mean()),
                   "AUC": roc_auc_score(y, s), "AUC_fold_mean": float(fold_auc.mean()),
                   "Hit_rate": float((pred_up == y).mean())}
            if src == "classifier":
                pr = np.clip(s, 1e-6, 1 - 1e-6)
                row["Log_loss"], row["Brier"] = log_loss(y, pr), brier_score_loss(y, pr)
            if ref_boot is not None and model != ref:
                boot = np.array([roc_auc_score(y[k], s[k]) for k in idx]) - ref_boot
                row["AUC_vs_ref"] = row["AUC"] - roc_auc_score(y, ref_s)
                row["CI_low"], row["CI_high"] = np.quantile(boot, [0.05, 0.95])
                row["p_one_sided"] = float((boot <= 0).mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    out["Horizon"] = pd.Categorical(out["Horizon"], rp.HORIZON_ORDER, ordered=True)
    return out.sort_values(["Horizon", "Source", "AUC"], ascending=[True, True, False]).reset_index(drop=True)


def plot_direction(table: pd.DataFrame, ref: str = "Logit | HAR"):
    """AUC difference against ``ref`` with its 90% interval, per horizon (green = interval above 0)."""
    import matplotlib.pyplot as plt

    t = table[table.Model != ref].dropna(subset=["AUC_vs_ref"])
    hz = [h for h in rp.HORIZON_ORDER if h in set(t.Horizon.astype(str))]
    fig, axes = plt.subplots(1, len(hz), figsize=(7 * len(hz), 0.45 * t.groupby("Horizon").size().max() + 1.8),
                             squeeze=False)
    for ax, h in zip(axes[0], hz):
        g = t[t.Horizon.astype(str) == h].sort_values("AUC_vs_ref").reset_index(drop=True)
        colors = ["#2E8B57" if lo > 0 else "#C8553D" if hi < 0 else "#8C8C8C" for lo, hi in zip(g.CI_low, g.CI_high)]
        y = np.arange(len(g))
        ax.hlines(y, g.CI_low, g.CI_high, colors=colors, lw=2.2)
        ax.scatter(g.AUC_vs_ref, y, c=colors, s=36, zorder=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y, [f"{m} ({s})" for m, s in zip(g.Model, g.Source)], fontsize=8)
        ax.set(title=h, xlabel=f"AUC minus {ref} (90% block-bootstrap CI)")
    fig.suptitle("Volatility direction: does the model rank up-moves better than the HAR classifier?")
    fig.tight_layout()
    return fig
