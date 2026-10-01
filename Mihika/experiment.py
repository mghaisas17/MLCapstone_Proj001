"""Volatility-forecasting experiment runner.

One pipeline for every horizon: load bars (memory-safe), build/cache features,
split with a purged expanding-window walk-forward CV, fit every registered
model on identical folds, and save tidy results that ``reporting.py`` turns
into tables and figures.

Why purged: the target at time t is realized volatility over (t, t + h], so a
training row whose label window reaches into the test block has seen test-period
returns. Any training row with ``target_end >= first test time`` is dropped.

Run from the shell (keeps heavy work out of a notebook kernel):

    python experiment.py --horizons 5m 1h --output results/main
    python experiment.py --quick --horizons 1d
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forecast_data import (
    SPECS,
    ForecastDataset,
    ForecastSpec,
    build_dataset,
    duration_steps,
    fetch_binance_bars,
    load_or_download_yahoo,
)
from models_and_metrics import (
    EPS,
    evaluate_garch,
    evaluate_har_log,
    fit_mlp,
    fit_signature_lstm,
    fit_xgb,
)

LOGGER = logging.getLogger(__name__)

FEATURE_VERSION = 1  # bump to invalidate cached feature files
HORIZON_LABELS = {"5m": "5 min", "1h": "1 hour", "1d": "1 day", "7d": "7 day", "30d": "30 day"}
BINANCE_HORIZONS = ("5m", "1h")
DAILY_HORIZONS = ("1d", "7d", "30d")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    horizons: tuple = ("5m", "1h", "1d", "7d", "30d")
    n_folds: int = 3
    min_train_rows: int = 50

    # data
    symbol: str = "BTCUSDT"
    binance_start: str = "2026-07-01"
    binance_end: str = "2026-09-15"
    ticker: str = "BTC-USD"
    yahoo_start: str = "2015-01-01"
    yahoo_end: str | None = None  # None -> today
    # Target number of forecast rows per horizon; the sampling stride is chosen
    # to hit it (None = every bar). Purging handles the label overlap.
    target_rows: dict = field(
        default_factory=lambda: {"5m": 4000, "1h": 4000, "1d": None, "7d": None, "30d": None}
    )

    # models
    models: tuple | None = None  # None -> every registered model valid for the horizon
    torch_epochs: int = 200
    weight_decay: dict = field(
        default_factory=lambda: {"5m": 1e-4, "1h": 1e-4, "1d": 1e-4, "7d": 5e-4, "30d": 1e-3}
    )
    lstm_sub_window: dict = field(
        default_factory=lambda: {"5m": "30min", "1h": "6h", "1d": "5D", "7d": "5D", "30d": "5D"}
    )
    xgb_estimators: int = 500
    seed: int = 42

    # ablations (signature level / augmentation / dimensions), same folds
    run_ablations: bool = True
    ablation_models: tuple = ("Ridge | signature",)

    # io
    data_dir: str = "data"
    cache_dir: str = "cache"
    output_dir: str = "results/main"
    force_download: bool = False
    resume: bool = True

    def quick(self) -> "ExperimentConfig":
        """Small, fast variant for checking that everything runs."""
        end = pd.Timestamp(self.binance_end)
        return replace(
            self,
            binance_start=str((end - pd.Timedelta(days=9)).date()),
            yahoo_start="2021-01-01",
            target_rows={h: 800 for h in HORIZON_LABELS},
            torch_epochs=25,
            xgb_estimators=100,
            output_dir=self.output_dir + "_quick",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        d = json.loads(Path(path).read_text())
        for k in ("horizons", "models", "ablation_models"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# --------------------------------------------------------------------------
# Purged expanding-window walk-forward CV
# --------------------------------------------------------------------------


@dataclass
class Fold:
    number: int
    train: np.ndarray
    test: np.ndarray


def purged_walk_forward(
    times,
    target_end,
    n_folds: int,
    min_train_rows: int = 50,
    strict: bool = True,
) -> list[Fold]:
    """Expanding-window folds over n_folds + 1 consecutive blocks (block 0 is
    train-only). For each fold the train set is everything before the test block
    minus rows whose label window (``target_end``) reaches the first test time."""
    n = len(times)
    edges = np.linspace(0, n, n_folds + 2).astype(int)
    folds = []
    for k in range(n_folds):
        test = np.arange(edges[k + 1], edges[k + 2])
        if len(test) == 0:
            continue
        before = np.arange(0, edges[k + 1])
        train = before[np.asarray(target_end[before] < times[test[0]])]
        if len(train) < min_train_rows:
            if strict:
                raise ValueError(
                    f"Fold {k + 1} has only {len(train)} training rows after purging "
                    f"(need {min_train_rows}); use fewer folds or more data."
                )
            continue
        folds.append(Fold(k + 1, train, test))
    return folds


def fit_ridge(X, y, times, target_end, alphas=None, inner_splits: int = 5) -> Pipeline:
    """Ridge with alpha chosen by an inner purged walk-forward CV on the
    training rows only (the inner CV has the same label overlap as the outer)."""
    alphas = np.logspace(-4, 4, 9) if alphas is None else alphas
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    inner = purged_walk_forward(times, target_end, inner_splits, min_train_rows=10, strict=False)
    if not inner:
        return pipe.set_params(ridge__alpha=1.0).fit(X, y)
    search = GridSearchCV(
        pipe,
        {"ridge__alpha": alphas},
        scoring="neg_mean_squared_error",
        cv=[(f.train, f.test) for f in inner],
        n_jobs=1,
    )
    return search.fit(X, y).best_estimator_


# --------------------------------------------------------------------------
# Model registry: adding a model is adding one ModelSpec
# --------------------------------------------------------------------------


@dataclass
class FoldContext:
    ds: ForecastDataset
    fold: Fold
    horizon_key: str
    horizon_steps: int
    cfg: ExperimentConfig
    returns: pd.Series | None = None

    @property
    def y_train(self):
        return self.ds.y[self.fold.train]

    @property
    def y_test(self):
        return self.ds.y[self.fold.test]

    def features(self, kind: str) -> np.ndarray:
        d = self.ds
        return {
            "stats": d.X_stats,
            "sig": d.X_sig,
            "stats+sig": np.hstack([d.X_stats, d.X_sig]),
            "har": d.X_har,
            "seq": d.X_seq,
        }[kind]

    def xy(self, kind: str):
        X = self.features(kind)
        return X[self.fold.train], X[self.fold.test]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fit_predict: Callable[[FoldContext], np.ndarray]
    horizons: tuple | None = None  # None -> every horizon
    needs_seq: bool = False


def _mean(c):
    return np.full(len(c.fold.test), c.y_train.mean())


def _har(c):
    d, tr, te = c.ds, c.fold.train, c.fold.test
    return evaluate_har_log(
        d.X_har[tr], d.y_har_log[tr], d.X_har[te], c.y_test, c.horizon_steps
    ).predictions


def _garch(c):
    times = c.ds.times
    returns_train = c.returns.loc[: times[c.fold.train[-1]]]
    anchors = pd.DatetimeIndex(times[c.fold.test])
    return evaluate_garch(
        returns_train, c.returns, anchors, c.y_test, c.horizon_steps, vol="GARCH"
    ).predictions


def _ridge(kind):
    def run(c):
        Xtr, Xte = c.xy(kind)
        tr = c.fold.train
        model = fit_ridge(Xtr, c.y_train, c.ds.times[tr], c.ds.target_end[tr])
        return model.predict(Xte)

    return run


def _xgb(kind):
    def run(c):
        Xtr, Xte = c.xy(kind)
        model = fit_xgb(Xtr, c.y_train, random_state=c.cfg.seed, n_estimators=c.cfg.xgb_estimators)
        return model.predict(Xte)

    return run


def _mlp(kind):
    def run(c):
        Xtr, Xte = c.xy(kind)
        model = fit_mlp(
            Xtr, c.y_train, weight_decay=c.cfg.weight_decay[c.horizon_key],
            epochs=c.cfg.torch_epochs, random_state=c.cfg.seed + c.fold.number,
        )
        return model.predict(Xte)

    return run


def _lstm(c):
    Xtr, Xte = c.xy("seq")
    model = fit_signature_lstm(
        Xtr, c.y_train, weight_decay=c.cfg.weight_decay[c.horizon_key],
        epochs=c.cfg.torch_epochs, random_state=c.cfg.seed + c.fold.number,
    )
    return model.predict(Xte)


MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec("Mean", _mean),
        ModelSpec("HAR-RV-L", _har),
        ModelSpec("GARCH(1,1)", _garch, horizons=DAILY_HORIZONS),
        ModelSpec("Ridge | stats", _ridge("stats")),
        ModelSpec("Ridge | signature", _ridge("sig")),
        ModelSpec("Ridge | stats+signature", _ridge("stats+sig")),
        ModelSpec("XGBoost | stats", _xgb("stats")),
        ModelSpec("XGBoost | signature", _xgb("sig")),
        ModelSpec("MLP | stats", _mlp("stats")),
        ModelSpec("MLP | stats+signature", _mlp("stats+sig")),
        ModelSpec("Signature LSTM", _lstm, needs_seq=True),
    ]
}


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def evaluate_models(
    ds: ForecastDataset,
    folds: list[Fold],
    horizon_key: str,
    horizon_steps: int,
    cfg: ExperimentConfig,
    model_names,
    *,
    returns: pd.Series | None = None,
    experiment: str = "Main",
    setting: str = "Main spec",
    keep_predictions: bool = True,
):
    rows, preds = [], []
    for fold in folds:
        ctx = FoldContext(ds, fold, horizon_key, horizon_steps, cfg, returns)
        for name in model_names:
            spec = MODELS[name]
            if spec.horizons is not None and horizon_key not in spec.horizons:
                continue
            t0 = time.time()
            pred = np.maximum(np.asarray(spec.fit_predict(ctx), dtype=float), EPS)
            rows.append(
                {
                    "HorizonKey": horizon_key, "Horizon": HORIZON_LABELS[horizon_key],
                    "Experiment": experiment, "Setting": setting, "Model": name,
                    "Fold": fold.number, "RMSE": rmse(ctx.y_test, pred),
                    "N_train": len(fold.train), "N_test": len(fold.test),
                    "Seconds": round(time.time() - t0, 2),
                }
            )
            if keep_predictions:
                preds.append(
                    pd.DataFrame(
                        {
                            "HorizonKey": horizon_key, "Horizon": HORIZON_LABELS[horizon_key],
                            "Fold": fold.number, "Time": ds.times[fold.test], "Model": name,
                            "Actual": ctx.y_test, "Prediction": pred,
                        }
                    )
                )
        LOGGER.info("[%s] fold %d/%d done", horizon_key, fold.number, len(folds))
    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


# --------------------------------------------------------------------------
# Data and cached feature datasets
# --------------------------------------------------------------------------

_BINANCE_MEMO: dict = {}


def load_bars(horizon: str, cfg: ExperimentConfig) -> pd.DataFrame:
    if horizon in BINANCE_HORIZONS:
        freqs = tuple(sorted({SPECS[h].bar_freq for h in BINANCE_HORIZONS}))
        key = (cfg.symbol, cfg.binance_start, cfg.binance_end, freqs)
        if key not in _BINANCE_MEMO:
            _BINANCE_MEMO.clear()
            _BINANCE_MEMO[key] = fetch_binance_bars(
                cfg.binance_start, cfg.binance_end, freqs=freqs, symbol=cfg.symbol,
                data_dir=Path(cfg.data_dir) / "bars", force_download=cfg.force_download,
            )
        return _BINANCE_MEMO[key][SPECS[horizon].bar_freq]
    end = cfg.yahoo_end or str(pd.Timestamp.now(tz="UTC").date())
    return load_or_download_yahoo(
        cfg.yahoo_start, end, ticker=cfg.ticker, data_dir=Path(cfg.data_dir) / "yahoo",
        force_download=cfg.force_download,
    )


def choose_stride(n_bars: int, spec: ForecastSpec, target_rows: int | None) -> int:
    if not target_rows:
        return 1
    h = duration_steps(spec.horizon, spec.bar_freq)
    warm = max(
        duration_steps(spec.lookback, spec.bar_freq),
        max(duration_steps(w, spec.bar_freq) for w in spec.har_windows),
    )
    candidates = max(1, n_bars - h - (warm - 1))
    return max(1, math.ceil(candidates / target_rows))


def base_spec_for(horizon: str, bars: pd.DataFrame, cfg: ExperimentConfig) -> ForecastSpec:
    spec = replace(SPECS[horizon], depth=3, lead_lag=True, time_aug=True)
    return replace(spec, sample_stride=choose_stride(len(bars), spec, cfg.target_rows.get(horizon)))


def _utc_index(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def get_dataset(
    bars: pd.DataFrame, spec: ForecastSpec, cfg: ExperimentConfig, *, include_seq: bool, sub_window: str
) -> ForecastDataset:
    """Build features once and cache to disk, keyed by spec + data fingerprint."""
    fingerprint = {
        "v": FEATURE_VERSION, "spec": asdict(spec), "seq": include_seq, "sub": sub_window,
        "n": len(bars), "first": str(bars.index[0]), "last": str(bars.index[-1]),
        "close_sum": round(float(bars["close"].sum()), 6),
    }
    key = hashlib.md5(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = Path(cfg.cache_dir) / f"{spec.name}_{key}.npz"

    if path.exists():
        z = np.load(path, allow_pickle=False)
        return ForecastDataset(
            times=_utc_index(z["times"]), target_end=_utc_index(z["target_end"]), y=z["y"],
            X_har=z["X_har"], X_stats=z["X_stats"], X_sig=z["X_sig"],
            har_names=z["har_names"].tolist(), stat_names=z["stat_names"].tolist(),
            y_har_log=z["y_har_log"], X_seq=z["X_seq"] if "X_seq" in z.files else None,
        )

    ds = build_dataset(bars, spec, include_lstm_seq=include_seq, sub_window=sub_window)
    ds.times, ds.target_end = _utc_index(ds.times), _utc_index(ds.target_end)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"X_seq": ds.X_seq} if ds.X_seq is not None else {}
    np.savez_compressed(
        path, times=ds.times.asi8, target_end=ds.target_end.asi8, y=ds.y, X_har=ds.X_har,
        X_stats=ds.X_stats, X_sig=ds.X_sig, y_har_log=ds.y_har_log,
        har_names=np.asarray(ds.har_names, dtype=str), stat_names=np.asarray(ds.stat_names, dtype=str),
        **extra,
    )
    return ds


# --------------------------------------------------------------------------
# Ablations: which signature construction choices matter (same folds)
# --------------------------------------------------------------------------


def signature_level_slices(spec: ForecastSpec) -> dict[str, slice]:
    dim = len(spec.dims) * (2 if spec.lead_lag else 1) + int(spec.time_aug)
    ends = np.cumsum([dim**k for k in range(1, spec.depth + 1)])
    return {f"L1-{k}": slice(0, int(end)) for k, end in enumerate(ends, 1)}


def dimension_variants(horizon: str) -> dict[str, tuple]:
    third = "flow" if horizon in BINANCE_HORIZONS else "range"
    return {
        "Price only": ("price",),
        "Price + activity": ("price", "activity"),
        "Full path": ("price", "activity", third),
    }


def run_ablations(bars, base_spec, base_ds, folds, horizon_key, horizon_steps, cfg, returns):
    frames = []

    def run(ds, experiment, setting):
        m, _ = evaluate_models(
            ds, folds, horizon_key, horizon_steps, cfg, cfg.ablation_models,
            returns=returns, experiment=experiment, setting=setting, keep_predictions=False,
        )
        frames.append(m)

    for setting, sl in signature_level_slices(base_spec).items():
        run(replace(base_ds, X_sig=base_ds.X_sig[:, sl]), "Signature level", setting)

    variants = {
        "Augmentation": {
            "Raw path": dict(lead_lag=False, time_aug=False),
            "Time": dict(lead_lag=False, time_aug=True),
            "Lead-lag + time": {},
        },
        "Dimensions": {
            name: ({} if name == "Full path" else dict(dims=dims))
            for name, dims in dimension_variants(horizon_key).items()
        },
    }
    for experiment, settings in variants.items():
        for setting, changes in settings.items():
            if not changes:
                run(base_ds, experiment, setting)
                continue
            ds = get_dataset(
                bars, replace(base_spec, **changes), cfg, include_seq=False,
                sub_window=cfg.lstm_sub_window[horizon_key],
            )
            if not (ds.times == base_ds.times).all():
                raise RuntimeError("Ablation rows differ from the base dataset; folds would not match.")
            run(ds, experiment, setting)
            del ds
            gc.collect()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_horizon(horizon: str, cfg: ExperimentConfig) -> Path:
    out = Path(cfg.output_dir) / horizon
    if cfg.resume and (out / "meta.json").exists():
        LOGGER.info("[%s] already complete, skipping (resume=True)", horizon)
        return out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bars = load_bars(horizon, cfg)
    spec = base_spec_for(horizon, bars, cfg)
    horizon_steps = duration_steps(spec.horizon, spec.bar_freq)
    names = list(cfg.models) if cfg.models else list(MODELS)
    need_seq = any(MODELS[n].needs_seq for n in names)

    ds = get_dataset(bars, spec, cfg, include_seq=need_seq, sub_window=cfg.lstm_sub_window[horizon])
    folds = purged_walk_forward(ds.times, ds.target_end, cfg.n_folds, cfg.min_train_rows)
    returns = np.log(bars["close"]).diff().dropna() if horizon in DAILY_HORIZONS else None
    LOGGER.info(
        "[%s] rows=%d stride=%d folds=%s", horizon, len(ds.y), spec.sample_stride,
        [(len(f.train), len(f.test)) for f in folds],
    )

    metrics, preds = evaluate_models(ds, folds, horizon, horizon_steps, cfg, names, returns=returns)
    metrics.to_csv(out / "metrics.csv", index=False)
    preds.to_parquet(out / "predictions.parquet", index=False)

    if cfg.run_ablations:
        ablations = run_ablations(bars, spec, ds, folds, horizon, horizon_steps, cfg, returns)
        ablations.to_csv(out / "ablations.csv", index=False)

    meta = {
        "horizon": horizon, "horizon_steps": horizon_steps, "stride": spec.sample_stride,
        # horizon length measured in *forecast rows*, the lag unit the DM test needs
        "horizon_rows": max(1, math.ceil(horizon_steps / spec.sample_stride)),
        "n_rows": int(len(ds.y)), "n_folds": len(folds),
        "fold_sizes": [[len(f.train), len(f.test)] for f in folds],
        "first_time": str(ds.times[0]), "last_time": str(ds.times[-1]),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))  # written last = "complete"
    del ds, bars, metrics, preds
    gc.collect()
    return out


def run_all(cfg: ExperimentConfig, *, subprocess_per_horizon: bool = False) -> Path:
    """Run every horizon. With ``subprocess_per_horizon`` each horizon runs in a
    fresh Python process, so memory is released between horizons and a crash
    cannot take down a notebook kernel; finished horizons are skipped on re-run."""
    root = Path(cfg.output_dir)
    cfg.to_json(root / "config.json")
    for horizon in cfg.horizons:
        if not subprocess_per_horizon:
            run_horizon(horizon, cfg)
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(root / "config.json"),
               "--horizons", horizon]
        code = subprocess.run(cmd).returncode
        if code != 0:
            LOGGER.error("[%s] subprocess exited with code %s; re-run to resume", horizon, code)
    return root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config written by ExperimentConfig.to_json")
    p.add_argument("--horizons", nargs="+", choices=list(HORIZON_LABELS))
    p.add_argument("--output")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
    if args.quick:
        cfg = cfg.quick()
    if args.horizons:
        cfg = replace(cfg, horizons=tuple(args.horizons))
    if args.output:
        cfg = replace(cfg, output_dir=args.output)
    if args.no_resume:
        cfg = replace(cfg, resume=False)
    if args.config:  # invoked per horizon by run_all(subprocess_per_horizon=True)
        for horizon in cfg.horizons:
            run_horizon(horizon, cfg)
    else:
        run_all(cfg)


if __name__ == "__main__":
    main()
