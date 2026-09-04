"""Threshold/p-value calibration for any of the test statistics above, plus
multiple-testing correction -- mirrors the paper's Section 4.1 methodology
of comparing empirical p-values against a parametric Weibull tail fit, then
controlling the false discovery rate across many simultaneous window
tests with Benjamini-Hochberg.

The paper's own Type-I tail bound (Section 3.2) has the form
    P_I(r) <= C2 * exp{-(C1^2/2) * [(r/scale)^(2p) v (r/scale)^(2p/N)]}
which for a fixed signature depth N reduces, in its dominant regime, to a
Weibull-type tail A*exp(-B*r^(2/N)) in r -- that parametric family is what
we fit here directly from a calibration sample, rather than re-deriving
the constants C1, C2, p analytically (those depend on exponential-moment
properties of the reference measure mu that are impractical to estimate
directly from a finite crypto price history).
"""
from __future__ import annotations

import numpy as np


def empirical_pvalue(score: float, calibration_scores: np.ndarray) -> float:
    """P(calibration score >= observed score), with the standard +1
    correction so the p-value is never exactly zero."""
    n = len(calibration_scores)
    exceed = np.sum(calibration_scores >= score)
    return float((1 + exceed) / (n + 1))


def empirical_pvalues(scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    return np.array([empirical_pvalue(s, calibration_scores) for s in scores])


def fit_weibull_tail(calibration_scores: np.ndarray, N: int, tail_fraction: float = 0.3) -> tuple[float, float]:
    """Fit survival function P(score > r) ~= A * exp(-B * r^(2/N)) to the
    upper tail of a calibration sample of "normal" scores via least squares
    on log(survival) vs r^(2/N). Returns (A, B).

    Only the top `tail_fraction` of scores are used for the fit -- this is
    a tail approximation (peaks-over-threshold style), not a fit to the
    bulk of the distribution, matching how the paper describes fitting this
    curve "to the tail of the reference measure" (Section 4.1). Fitting the
    full range biases the intercept A badly, since the bulk of the
    distribution near 0 does not follow the pure Weibull-tail form."""
    sorted_scores = np.sort(calibration_scores)
    n = len(sorted_scores)
    survival = 1.0 - np.arange(1, n + 1) / (n + 1)
    cutoff = int(n * (1 - tail_fraction))
    tail_scores, tail_survival = sorted_scores[cutoff:], survival[cutoff:]
    positive = tail_scores > 0
    x = tail_scores[positive] ** (2.0 / N)
    y = np.log(tail_survival[positive])
    B, log_A = -np.polyfit(x, y, 1)
    return float(np.exp(log_A)), float(B)


def weibull_pvalue(score: float, A: float, B: float, N: int) -> float:
    if score <= 0:
        return 1.0
    return float(min(1.0, A * np.exp(-B * score ** (2.0 / N))))


def weibull_threshold(alpha: float, A: float, B: float, N: int) -> float:
    """r(alpha) solving A*exp(-B*r^(2/N)) = alpha."""
    if alpha <= 0 or alpha >= A:
        raise ValueError(f"alpha must be in (0, {A}) for this fitted tail")
    return float((np.log(A / alpha) / B) ** (N / 2.0))


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR control at level q. Returns
    (boolean rejection mask in original order, the p-value threshold used)."""
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool), 0.0
    k_max = np.max(np.where(below)[0])
    p_threshold = sorted_p[k_max]
    rejections_sorted = sorted_p <= p_threshold
    rejections = np.zeros(n, dtype=bool)
    rejections[order] = rejections_sorted
    return rejections, float(p_threshold)
