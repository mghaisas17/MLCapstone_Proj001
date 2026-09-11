"""Signature-MMD: a two-sample Maximum Mean Discrepancy test in truncated
signature-feature space, per Alden, Horvath & Issa, "Signature Maximum Mean
Discrepancy Two-Sample Statistical Tests" (arXiv:2506.01718, June 2025) --
same research community as this project's base paper (novelty detection on
path space).

Where this differs from `ExpectedSignatureDistance`: that measure is the
isotropic distance from a single window's signature to the reference *mean*
signature -- it only ever compares one point against a mean vector. MMD
compares two whole *samples* (sets of signatures) against each other,
picking up spread/cross-term information a single point-to-mean distance
throws away. Concretely here, a small trailing batch of B consecutive
windows (a short recent history, not just "right now") is compared against
the reference sample as two distributions.

Cheap to add given the existing pipeline: since every model here already
uses a *truncated* signature, the truncated signature kernel is just the
plain inner product of the already-computed feature vectors,
kappa(x, y) = <S_N(x), S_N(y)> -- no new signature machinery, no kernel PDE
solve, just the standard unbiased two-sample MMD^2 U-statistic over a Gram
matrix built from vectors this project already computes.

Important honesty check, found while verifying this against known cases:
using the truncated feature vectors directly as the kernel's feature map
makes this a *linear* kernel in signature-feature space. For a linear
kernel, MMD^2(X, Y) collapses exactly to ||mean(X) - mean(Y)||^2 -- it is
mean-shift-only and provably blind to same-mean, different-spread
distributions (verified directly: a same-mean/different-variance synthetic
pair gave MMD^2 ~ 0, as the algebra predicts). It is *not* the fully general,
potentially nonlinear signature-kernel MMD the paper describes, which would
need a genuine (nonlinear) path kernel via the signature-kernel PDE/Gram
methods -- out of scope here. What this class *does* add over
`ExpectedSignatureDistance`: comparing a trailing *batch* of `batch_size`
windows' mean signature against the reference mean, instead of one window's
own signature against it, requires a sustained shift across several
consecutive windows rather than reacting to a single noisy one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.signatures import signature


def mmd_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased two-sample MMD^2 estimate, using each sample's own rows as
    an explicit finite-dimensional feature map (the truncated signature
    kernel is exactly the inner product of these already-truncated vectors).
    X: (m, d) sample from the test distribution. Y: (n, d) sample from the
    reference distribution. Falls back to a biased (but still valid) estimate
    when a sample has only one point, since the unbiased U-statistic needs
    >= 2 points per side to drop the diagonal."""
    m, n = X.shape[0], Y.shape[0]
    Kxx, Kyy, Kxy = X @ X.T, Y @ Y.T, X @ Y.T

    term_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) if m > 1 else Kxx.mean()
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) if n > 1 else Kyy.mean()
    term_xy = Kxy.mean()
    return float(term_xx + term_yy - 2 * term_xy)


@dataclass
class SignatureMMD:
    depth: int
    batch_size: int = 5  # number of trailing windows compared as one "test sample"

    def fit(self, reference_paths: list[np.ndarray]) -> "SignatureMMD":
        self.ref_sigs_ = np.vstack([signature(p, self.depth) for p in reference_paths])
        return self

    def score_series(self, paths: list[np.ndarray]) -> np.ndarray:
        """One score per path in `paths`, each computed from a trailing
        batch of `batch_size` consecutive paths ending at that index (fewer
        for the first `batch_size - 1` points, down to a single-path batch)."""
        sigs = np.vstack([signature(p, self.depth) for p in paths])
        scores = np.empty(len(paths))
        for i in range(len(paths)):
            start = max(0, i - self.batch_size + 1)
            batch = sigs[start : i + 1]
            scores[i] = mmd_squared(batch, self.ref_sigs_)
        return scores
