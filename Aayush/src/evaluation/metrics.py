"""Shared metrics for comparing horizons and models: AUROC (thin wrapper,
kept here so evaluation notebooks import one module), precision@k, and a
summary table combining synthetic AUROC, event detection, and
Benjamini-Hochberg FDR/power -- the three-way comparison the project brief
asks for ("which horizon + which model catches which kind of event").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision among the top-k highest-scored (most anomalous) points."""
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[order]))


def false_discovery_rate(rejections: np.ndarray, y_true: np.ndarray) -> float:
    """Empirical FDR given a boolean rejection mask and true labels
    (1 = genuinely anomalous), for validating Benjamini-Hochberg control on
    the synthetic-injection experiments where ground truth is known."""
    n_rejected = rejections.sum()
    if n_rejected == 0:
        return 0.0
    false_rejections = np.sum(rejections & (y_true == 0))
    return float(false_rejections / n_rejected)


def power(rejections: np.ndarray, y_true: np.ndarray) -> float:
    """Empirical statistical power: fraction of true anomalies rejected."""
    n_true = np.sum(y_true == 1)
    if n_true == 0:
        return float("nan")
    return float(np.sum(rejections & (y_true == 1)) / n_true)


def summarize_horizon_model(
    horizon: str,
    model: str,
    synthetic_auroc: dict[float, float] | None,
    event_results: pd.DataFrame | None,
    regime_hit_rate: float | None,
) -> dict:
    """One row of the final "which horizon + which model" comparison
    table."""
    row = {"horizon": horizon, "model": model}
    if synthetic_auroc:
        row["synthetic_auroc_max"] = max(synthetic_auroc.values())
        row["synthetic_auroc_at_smallest_magnitude"] = list(synthetic_auroc.values())[0]
    if event_results is not None:
        row["events_detected"] = int(event_results["detected"].sum())
        row["events_total"] = len(event_results)
        row["false_positive_rate"] = event_results.attrs.get("false_positive_rate")
    if regime_hit_rate is not None:
        row["regime_transition_hit_rate"] = regime_hit_rate
    return row
