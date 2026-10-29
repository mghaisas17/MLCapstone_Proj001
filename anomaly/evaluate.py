"""How an anomaly score is judged.

There are no ground-truth anomaly labels (5 labeled crises are far too few), so a *useful*
anomaly is defined as one that comes before volatility a standard model did not expect:

  surprise  = log realized vol over the next horizon - HAR-RV forecast (HAR fit on the fold's
              training rows)
  "missed"  = the top 20% of surprises in the fold's test rows

Metrics for a score on a fold's test rows:
  AUC_surprise        P(score before a missed event > score before an ordinary row); 0.5 = no
                      information beyond the HAR forecast
  AUC_surprise_strat  the same, computed within terciles of the HAR forecast and averaged, so a
                      detector can't win just by flagging volatile periods   <- used for selection
  Lift                hit rate of "missed" among flagged rows / base rate (flag = score above the
                      training rows' (1 - flag_rate) quantile)
  AUC_level           AUC for the top 20% of realized vol itself (Aayush-style: rewards
                      volatility clustering, reported for comparison)
  Fwd_vol_ratio       median realized vol after flagged rows / after unflagged rows (Aayush's test)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass
class FoldLabels:
    har_pred_train: np.ndarray
    har_pred_test: np.ndarray
    missed: np.ndarray  # bool, test rows
    strata: np.ndarray  # 0/1/2 terciles of the HAR forecast, test rows
    high_vol: np.ndarray  # bool, top quantile of realized vol, test rows
    y_test: np.ndarray


def fold_labels(rows, fold, q: float) -> FoldLabels:
    tr, te = fold.train, fold.test
    A = np.column_stack([np.ones(len(rows.y)), rows.har_X])
    beta, *_ = np.linalg.lstsq(A[tr], rows.y[tr], rcond=None)
    pred = A @ beta
    surprise = rows.y[te] - pred[te]
    strata = pd.qcut(pd.Series(pred[te]).rank(method="first"), 3, labels=False).to_numpy()
    return FoldLabels(
        har_pred_train=pred[tr], har_pred_test=pred[te], missed=surprise >= np.quantile(surprise, q),
        strata=strata, high_vol=rows.y[te] >= np.quantile(rows.y[te], q), y_test=rows.y[te],
    )


def _auc(label, score) -> float:
    label = np.asarray(label, bool)
    if label.all() or not label.any():
        return np.nan
    return float(roc_auc_score(label, score))


def stratified_auc(label, score, strata) -> float:
    vals = [_auc(label[strata == s], score[strata == s]) for s in np.unique(strata)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def score_metrics(score_test, score_train, lab: FoldLabels, flag_rate: float) -> dict:
    score_test = np.nan_to_num(np.asarray(score_test, float), nan=0.0)
    thr = np.quantile(score_train, 1 - flag_rate)
    flagged = score_test >= thr
    base = lab.missed.mean()
    vol = np.exp(lab.y_test)
    return {
        "AUC_surprise": _auc(lab.missed, score_test),
        "AUC_surprise_strat": stratified_auc(lab.missed, score_test, lab.strata),
        "Lift": float(lab.missed[flagged].mean() / base) if flagged.any() else np.nan,
        "Share_flagged": float(flagged.mean()),
        "AUC_level": _auc(lab.high_vol, score_test),
        "Fwd_vol_ratio": float(np.median(vol[flagged]) / np.median(vol[~flagged])) if flagged.any() and (~flagged).any() else np.nan,
    }


def block_bootstrap_diff(score_a, score_b, lab: FoldLabels, block: int, n_boot: int, seed: int = 0) -> dict:
    """Stratified-AUC(a) - stratified-AUC(b) on one fold's test rows, with a moving-block
    bootstrap (blocks of consecutive rows) so overlapping windows aren't treated as independent."""
    rng = np.random.default_rng(seed)
    n = len(lab.missed)
    block = max(1, min(block, n))
    point = stratified_auc(lab.missed, score_a, lab.strata) - stratified_auc(lab.missed, score_b, lab.strata)
    starts = np.arange(0, n - block + 1)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, size=int(np.ceil(n / block)))])[:n]
        m, st = lab.missed[idx], lab.strata[idx]
        diffs.append(stratified_auc(m, score_a[idx], st) - stratified_auc(m, score_b[idx], st))
    diffs = np.asarray([d for d in diffs if np.isfinite(d)])
    return {
        "diff": float(point),
        "ci_low": float(np.quantile(diffs, 0.025)) if len(diffs) else np.nan,
        "ci_high": float(np.quantile(diffs, 0.975)) if len(diffs) else np.nan,
        "p_value": float((diffs <= 0).mean()) if len(diffs) else np.nan,  # one-sided: a no better than b
    }


def reorder_injection(win: dict, rows_idx: np.ndarray, seed: int = 0) -> tuple[dict, np.ndarray]:
    """Order-only synthetic anomalies: for half of the given rows, the window's returns are
    re-ordered into a V (all falls, then all rises) or an inverted V. The returns themselves are
    unchanged, so realized variance, net move, skew and kurtosis are identical; only the path's
    shape differs. Order-blind summaries cannot see this; path signatures can."""
    rng = np.random.default_rng(seed)
    sub = {k: v[rows_idx].copy() for k, v in win.items()}
    injected = np.zeros(len(rows_idx), bool)
    injected[rng.permutation(len(rows_idx))[: len(rows_idx) // 2]] = True
    for j in np.where(injected)[0]:
        sub["r"][j] = np.sort(sub["r"][j])[:: (1 if rng.random() < 0.5 else -1)]
    return sub, injected
