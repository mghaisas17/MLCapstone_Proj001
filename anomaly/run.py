"""Run the anomaly study.

Per horizon:
  Stage A  Mahalanobis + Isolation Forest on every path recipe: signatures at each depth, the
           matched lag bank, stats and raw, plus two volatility baselines (current vol level,
           HAR-RV forecast).
  Stage B  the autoencoder on the top Stage A signature configs and the best of each control.
  Select   best signature config and best signature-free config by mean stratified surprise AUC
           on the first folds; confirm on the last fold.
  Gate     passes if the signature config beats the signature-free one on the last fold
           (block bootstrap, one-sided p < 0.05) as well as on the selection folds.
  Synthetic  order-only injected anomalies (same returns, different shape): can each selected
           detector find them?

Results: <output_dir>/<horizon>/{metrics.csv, scores.parquet, selection.json, meta.json}; finished
path recipes are saved under parts/ so an interrupted run resumes where it stopped.

    python -m anomaly.run --horizons 1d          # from the repo root
    python -m anomaly.run --quick
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import detectors as det
from .config import CHANNEL_SETS, HORIZONS, AnomalyConfig, RepConfig
from .evaluate import block_bootstrap_diff, fold_labels, reorder_injection, score_metrics
from .features import build_rows, extract_windows, representations
from experiment import purged_walk_forward
from forecast_data import fetch_binance_klines_hourly

LOGGER = logging.getLogger(__name__)
SIGNATURE = "signature"


def load_bars(spec, cfg: AnomalyConfig) -> pd.DataFrame:
    end = cfg.end or str(pd.Timestamp.now(tz="UTC").date())
    return fetch_binance_klines_hourly(cfg.start, end, symbol=cfg.symbol, data_dir=Path(cfg.data_dir) / "klines",
                                       freq=spec.bar_freq)


def _config_key(rep_name: str, rep: RepConfig) -> str:
    if rep_name == "stats":  # depends only on normalization and window
        return f"stats | {rep.normalize}|w{rep.window_mult:g}"
    if rep_name == "raw":  # does not depend on the lead-lag augmentation
        return f"raw | {rep.channels}|{rep.normalize}|w{rep.window_mult:g}"
    return f"{rep_name} | {rep.name}"


def _kind_of_key(key: str) -> str:
    return key.split(" | ")[0].split(" d")[0] if key.startswith(SIGNATURE) else key.split(" | ")[0]


def _evaluate(X, name_key, detector, rows, folds, labels, cfg, n_features):
    metrics, scores = [], []
    for fold in folds:
        model = det.make(detector, cfg).fit(X[fold.train])
        s_te, s_tr = model.score(X[fold.test]), model.score(X[fold.train])
        m = score_metrics(s_te, s_tr, labels[fold.number], cfg.flag_rate)
        metrics.append({"Config": name_key, "Kind": _kind_of_key(name_key), "Detector": detector,
                        "Fold": fold.number, "n_features": n_features, **m})
        scores.append(pd.DataFrame({"Time": rows.times[fold.test], "Fold": fold.number, "Config": name_key,
                                    "Detector": detector, "Score": s_te.astype(np.float32)}))
    return metrics, scores


def _baselines(rows, folds, labels, cfg):
    metrics, scores = [], []
    for name in ("vol level", "HAR forecast"):
        for fold in folds:
            lab = labels[fold.number]
            if name == "vol level":
                s_te, s_tr = rows.har_X[fold.test, 0], rows.har_X[fold.train, 0]
            else:
                s_te, s_tr = lab.har_pred_test, lab.har_pred_train
            m = score_metrics(s_te, s_tr, lab, cfg.flag_rate)
            metrics.append({"Config": name, "Kind": "baseline", "Detector": "none", "Fold": fold.number,
                            "n_features": 1, **m})
            scores.append(pd.DataFrame({"Time": rows.times[fold.test], "Fold": fold.number, "Config": name,
                                        "Detector": "none", "Score": np.asarray(s_te, np.float32)}))
    return metrics, scores


def _part_path(out: Path, tag: str) -> Path:
    return out / "parts" / (hashlib.md5(tag.encode()).hexdigest()[:12] + ".parquet")


def _run_rep(rep, detectors, bars, rows, folds, labels, spec, cfg, out, done: set, only=None):
    """Evaluate every representation of one path recipe with the given detectors (resumable)."""
    tag = f"{rep.name}|{','.join(detectors)}|{only}"
    part_m, part_s = _part_path(out, "m" + tag), _part_path(out, "s" + tag)
    if cfg.resume and part_m.exists() and part_s.exists():
        m_df = pd.read_parquet(part_m)
        if len(m_df):
            done.update(zip(m_df.Config, m_df.Detector))  # so later recipes don't redo shared controls
        return m_df, pd.read_parquet(part_s)
    W = int(round(spec.window * rep.window_mult))
    win = extract_windows(bars, rows.pos, W, rep.normalize)
    reps = representations(win, rep, spec, cfg.depths, cfg.max_sig_features)
    metrics, scores = [], []
    for rep_name, X in reps.items():
        key = _config_key(rep_name, rep)
        if only is not None and key not in only:
            continue
        for d in detectors:
            if (key, d) in done:
                continue
            done.add((key, d))
            m, s = _evaluate(X, key, d, rows, folds, labels, cfg, X.shape[1])
            metrics += m
            scores += s
    del win, reps
    gc.collect()
    m_df = pd.DataFrame(metrics)
    s_df = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    part_m.parent.mkdir(parents=True, exist_ok=True)
    m_df.to_parquet(part_m)
    s_df.to_parquet(part_s)
    return m_df, s_df


def _selection_score(metrics: pd.DataFrame, last_fold: int) -> pd.DataFrame:
    sel = metrics[metrics.Fold < last_fold].groupby(["Config", "Kind", "Detector"])["AUC_surprise_strat"].mean()
    conf = metrics[metrics.Fold == last_fold].set_index(["Config", "Kind", "Detector"])["AUC_surprise_strat"]
    return pd.DataFrame({"Select": sel, "Confirm": conf}).reset_index().sort_values("Select", ascending=False)


def _rep_of_key(key: str, cfg: AnomalyConfig):
    """The RepConfig that produces a config key (any matching one for keys that ignore some fields)."""
    for rep in cfg.rep_configs():
        for name in ("lag bank", "stats", "raw"):
            if _config_key(name, rep) == key:
                return rep
        if key.startswith(SIGNATURE) and key.split(" | ", 1)[1] == rep.name:
            return rep
    return None


def synthetic_check(choice: dict, bars, rows, folds, spec, cfg) -> dict:
    """AUC for telling order-only injected windows from the originals, for one selected config."""
    if choice["Kind"] == "baseline":
        return {"AUC_injected": 0.5, "note": "vol baselines are unchanged by re-ordering returns"}
    rep = _rep_of_key(choice["Config"], cfg)
    fold = folds[-1]
    W = int(round(spec.window * rep.window_mult))
    win = extract_windows(bars, rows.pos, W, rep.normalize)
    rep_name = choice["Config"].split(" | ")[0]
    X = representations(win, rep, spec, cfg.depths, cfg.max_sig_features)[rep_name]
    model = det.make(choice["Detector"], cfg).fit(X[fold.train])
    rng = np.random.default_rng(cfg.seed)
    sample = np.sort(rng.choice(fold.test, size=min(2000, len(fold.test)), replace=False))
    sub, injected = reorder_injection(win, sample, cfg.seed)
    Xs = representations(sub, rep, spec, cfg.depths, cfg.max_sig_features)[rep_name]
    from sklearn.metrics import roc_auc_score

    return {"AUC_injected": float(roc_auc_score(injected, model.score(Xs)))}


def run_horizon(hk: str, cfg: AnomalyConfig) -> Path:
    spec = HORIZONS[hk]
    out = Path(cfg.output_dir) / hk
    if cfg.resume and (out / "meta.json").exists():
        LOGGER.info("[%s] already complete, skipping", hk)
        return out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bars = load_bars(spec, cfg)
    max_w = int(round(spec.window * max(cfg.window_mults)))
    rows = build_rows(bars, spec, max_w, cfg.target_rows.get(hk))
    folds = purged_walk_forward(rows.times, rows.target_end, cfg.n_folds, cfg.min_train_rows)
    labels = {f.number: fold_labels(rows, f, cfg.surprise_quantile) for f in folds}
    LOGGER.info("[%s] %d rows, folds %s", hk, len(rows.y), [(len(f.train), len(f.test)) for f in folds])

    m_all, s_all = _baselines(rows, folds, labels, cfg)
    metrics, scores = [pd.DataFrame(m_all)], [pd.concat(s_all, ignore_index=True)]
    done: set = set()
    for i, rep in enumerate(cfg.rep_configs(), 1):  # Stage A
        LOGGER.info("[%s] stage A %d/%d: %s", hk, i, len(cfg.rep_configs()), rep.name)
        m, s = _run_rep(rep, cfg.stage_a_detectors, bars, rows, folds, labels, spec, cfg, out, done)
        metrics.append(m)
        scores.append(s)
    stage_a = pd.concat(metrics, ignore_index=True)

    # Stage B: autoencoder on the top signature configs and the best config of each control kind
    ranked = _selection_score(stage_a, folds[-1].number)
    picks = list(ranked[ranked.Kind == SIGNATURE].Config.drop_duplicates().head(cfg.stage_b_top_k))
    for kind in ("lag bank", "stats", "raw"):
        best = ranked[ranked.Kind == kind].Config.head(1).tolist()
        picks += best
    for key in dict.fromkeys(picks):
        rep = _rep_of_key(key, cfg)
        LOGGER.info("[%s] stage B autoencoder: %s", hk, key)
        m, s = _run_rep(rep, ("autoencoder",), bars, rows, folds, labels, spec, cfg, out, set(), only={key})
        metrics.append(m)
        scores.append(s)
    metrics_df = pd.concat(metrics, ignore_index=True)
    scores_df = pd.concat(scores, ignore_index=True)

    # Selection, gate, synthetic check
    last = folds[-1].number
    ranked = _selection_score(metrics_df, last)
    best_sig = ranked[ranked.Kind == SIGNATURE].iloc[0].to_dict()
    best_non = ranked[ranked.Kind != SIGNATURE].iloc[0].to_dict()
    lab = labels[last]

    def last_scores(choice):
        s = scores_df[(scores_df.Config == choice["Config"]) & (scores_df.Detector == choice["Detector"])
                      & (scores_df.Fold == last)]
        return s.Score.to_numpy(float)

    spacing = (
        pd.Series(rows.times).diff().median())
    block = max(1, int(round(pd.Timedelta(days=10 if hk == "1d" else 1) / spacing)))  # ~1 day (1h), ~10 days (1d)
    boot = block_bootstrap_diff(last_scores(best_sig), last_scores(best_non), lab, block, cfg.n_bootstrap, cfg.seed)
    passed = bool(best_sig["Select"] > best_non["Select"] and boot["diff"] > 0 and boot["p_value"] < 0.05)
    synthetic = {}
    for label_, choice in (("best signature", best_sig), ("best signature-free", best_non)):
        synthetic[label_] = {**choice, **synthetic_check(choice, bars, rows, folds, spec, cfg)}
    selection = {
        "horizon": hk, "best_signature": best_sig, "best_signature_free": best_non,
        "gate": {"passed": passed, "bootstrap_last_fold": boot, "block_rows": block,
                 "rule": "signature beats signature-free on selection folds AND on the last fold "
                         "(block bootstrap, one-sided p < 0.05), on stratified surprise AUC"},
        "synthetic_order_only": synthetic,
    }
    metrics_df.to_csv(out / "metrics.csv", index=False)
    scores_df.to_parquet(out / "scores.parquet", index=False)
    (out / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
    meta = {"horizon": hk, "rows": int(len(rows.y)), "folds": [[len(f.train), len(f.test)] for f in folds],
            "first_time": str(rows.times[0]), "last_time": str(rows.times[-1]),
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))  # written last = complete
    return out


def run_all(cfg: AnomalyConfig, subprocess_per_horizon: bool = False) -> Path:
    # absolute paths: per-horizon subprocesses run from the repo root, not the caller's directory
    cfg = replace(cfg, output_dir=str(Path(cfg.output_dir).resolve()), data_dir=str(Path(cfg.data_dir).resolve()))
    root = Path(cfg.output_dir)
    cfg.to_json(root / "config.json")
    for hk in cfg.horizons:
        if subprocess_per_horizon:
            code = subprocess.run([sys.executable, "-m", "anomaly.run", "--config", str(root / "config.json"),
                                   "--horizons", hk], cwd=Path(__file__).resolve().parent.parent).returncode
            if code != 0:
                LOGGER.error("[%s] exited with code %s; re-run to resume", hk, code)
        else:
            run_horizon(hk, cfg)
    best = {}
    for hk in cfg.horizons:
        f = root / hk / "selection.json"
        if f.exists():
            best[hk] = json.loads(f.read_text())
    (root / "best_configs.json").write_text(json.dumps(best, indent=2, default=float))
    return root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--horizons", nargs="+", choices=list(HORIZONS))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    cfg = AnomalyConfig.from_json(args.config) if args.config else AnomalyConfig()
    if args.quick:
        cfg = cfg.quick()
    if args.output:
        cfg = replace(cfg, output_dir=args.output)
    if args.horizons:
        cfg = replace(cfg, horizons=tuple(args.horizons))
    if args.config:  # invoked per horizon by run_all(subprocess_per_horizon=True)
        for hk in cfg.horizons:
            run_horizon(hk, cfg)
    else:
        run_all(cfg)


if __name__ == "__main__":
    main()
