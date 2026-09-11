"""Standard novelty-detection baselines on signature features, plus a
classical handcrafted-feature baseline (rolling realized-volatility
z-score) that skips signatures entirely -- the pointwise-statistics foil
that motivates using path signatures in the first place, analogous to the
paper's own TAMSD comparison (Section 4.1).

All `score_*` functions return higher = more anomalous, matching the sign
convention used by every other model in this project (sklearn's
decision_function is the opposite: positive = inlier, so it is negated
here).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def fit_sklearn_baselines(X_train: np.ndarray, random_state: int = 0) -> dict:
    """Fit the four standard sklearn novelty/outlier models on a signature
    feature matrix (n_train_windows, feature_dim) built from the reference
    ("normal") period."""
    models = {
        "one_class_svm": OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(X_train),
        "isolation_forest": IsolationForest(random_state=random_state).fit(X_train),
        "local_outlier_factor": LocalOutlierFactor(novelty=True).fit(X_train),
        "elliptic_envelope": EllipticEnvelope(random_state=random_state).fit(X_train),
    }
    return models


def score_sklearn_baselines(models: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    return {name: -model.decision_function(X) for name, model in models.items()}


def realized_vol_zscore(
    raw_paths: list[np.ndarray],
    return_channel_idx: int,
    ref_mean: float,
    ref_std: float,
) -> np.ndarray:
    """Classical pointwise-statistics baseline: realized volatility (std of
    the return channel's increments) within each window, z-scored against
    a reference period's realized-volatility distribution. No signature,
    no path geometry -- just the single number most volatility models use.
    """
    vols = np.array([np.std(np.diff(p[:, return_channel_idx])) for p in raw_paths])
    std_safe = ref_std if ref_std > 1e-12 else 1.0
    return np.abs(vols - ref_mean) / std_safe


def fit_realized_vol_reference(raw_paths: list[np.ndarray], return_channel_idx: int) -> tuple[float, float]:
    vols = np.array([np.std(np.diff(p[:, return_channel_idx])) for p in raw_paths])
    return float(vols.mean()), float(vols.std())
