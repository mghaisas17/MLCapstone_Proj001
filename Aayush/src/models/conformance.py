"""Conformance score (paper Section 2.3): distance from a point to the
reference set under the *variance norm* of the reference measure,

    ||x||_{nu-cov} = sup_{x*: Cov_nu(x*,x*) <= 1} x*(x)

For a Gaussian (or approximately Gaussian) reference measure, the paper
notes this coincides with the norm of the Cameron-Martin space, which is
exactly the Mahalanobis distance under the reference covariance -- that is
the special case implemented here, using a shrinkage (Ledoit-Wolf)
covariance estimate since signature feature dimension can be comparable to
the number of reference/training windows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

from src.features.signatures import signature


@dataclass
class ConformanceScore:
    depth: int

    def fit(self, train_paths: list[np.ndarray]) -> "ConformanceScore":
        sigs = np.vstack([signature(p, self.depth) for p in train_paths])
        self.mean_ = sigs.mean(axis=0)
        cov_estimator = LedoitWolf().fit(sigs)
        self.precision_ = cov_estimator.get_precision()
        return self

    def score(self, path: np.ndarray) -> float:
        sig = signature(path, self.depth)
        diff = sig - self.mean_
        return float(np.sqrt(diff @ self.precision_ @ diff))

    def score_many(self, paths: list[np.ndarray]) -> np.ndarray:
        return np.array([self.score(p) for p in paths])
