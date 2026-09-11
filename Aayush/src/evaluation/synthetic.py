"""Synthetic sanity check, mirroring the paper's own validation order
(Section 4.1): before trusting scores on ambiguous real market events,
confirm that AUROC increases as an injected anomaly gets more pronounced,
using data where we control the ground truth.

The injected anomaly mirrors the paper's Brownian-motion-with-a-spike
construction: a random-onset, ramping perturbation added to one channel of
an otherwise normal reference window.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def inject_spike(path: np.ndarray, channel: int, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    """Add a ramping spike `magnitude * sqrt(max(t - theta, 0))` (capped at
    1) to `channel`, onset theta drawn uniformly within the window -- the
    exact perturbation family used in the paper's anomalous-diffusion
    experiment (Section 4.1), adapted to our windows' own time axis."""
    L = path.shape[0]
    t = np.linspace(0.0, 1.0, L)
    theta = rng.uniform(0.0, 1.0)
    ramp = np.sqrt(np.clip(t - theta, 0.0, None))
    ramp = np.minimum(ramp, 1.0)
    spiked = path.copy()
    spiked[:, channel] = spiked[:, channel] + magnitude * ramp
    return spiked


def bootstrap_normal_paths(reference_paths: list[np.ndarray], n: int, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.integers(0, len(reference_paths), size=n)
    return [reference_paths[i].copy() for i in idx]


def auroc_vs_spike_magnitude(
    reference_paths: list[np.ndarray],
    score_fn,
    channel: int,
    magnitudes: list[float],
    n_per_class: int = 100,
    seed: int = 0,
) -> dict[float, float]:
    """For each magnitude, sample `n_per_class` normal paths and
    `n_per_class` spiked paths (bootstrapped from `reference_paths`), score
    both with `score_fn` (a callable path -> float, higher = more
    anomalous), and report AUROC. Expect AUROC to rise with magnitude.

    `reference_paths` should already be in the exact form each model's
    `score`/`score_many` expects (post normalize + add_basepoint, i.e. the
    same paths used to fit the model), not raw extracted channels.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for magnitude in magnitudes:
        normal = bootstrap_normal_paths(reference_paths, n_per_class, rng)
        spiked = [
            inject_spike(p, channel, magnitude, rng)
            for p in bootstrap_normal_paths(reference_paths, n_per_class, rng)
        ]
        y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
        scores = np.array([score_fn(p) for p in normal] + [score_fn(p) for p in spiked])
        results[magnitude] = float(roc_auc_score(y, scores))
    return results
