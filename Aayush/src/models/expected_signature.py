"""Distance-to-expected-signature test statistic (paper Eq. 2.7-2.8):

    f(x) = || S_N(x) - E_mu[S_N(X)] ||

Unlike the CVaR-OCSVM's linear functional <w, S_N(x)>, this is an isotropic
norm around the reference mean -- by construction it grows with the size
of *any* deviation from typical behavior, regardless of direction, which
is exactly the property the CVaR-OCSVM's single learned direction `w`
cannot guarantee. The paper uses this as its primary statistic for the
Brownian-motion-with-a-spike experiment (Section 4.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.signatures import signature


@dataclass
class ExpectedSignatureDistance:
    depth: int

    def fit(self, train_paths: list[np.ndarray]) -> "ExpectedSignatureDistance":
        sigs = np.vstack([signature(p, self.depth) for p in train_paths])
        self.mean_ = sigs.mean(axis=0)
        return self

    def score(self, path: np.ndarray) -> float:
        sig = signature(path, self.depth)
        return float(np.linalg.norm(sig - self.mean_))

    def score_many(self, paths: list[np.ndarray]) -> np.ndarray:
        return np.array([self.score(p) for p in paths])
