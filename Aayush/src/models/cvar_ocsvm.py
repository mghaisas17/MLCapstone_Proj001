"""The paper's Theorem 2.6 result: a one-class SVM whose CVaR objective is
replaced by a smooth polynomial surrogate, which the shuffle product turns
into a deterministic function of the *expected signature* alone.

    w* = argmin_w { 0.5*||w||^2 + CVaR^n_alpha(-<w, S_N(X)>) }
    CVaR^n_alpha(<w,S(X)>) = min_{rho in [-K,K]} sum_{m=0}^{n} b_m(w) rho^m
    b_m = 1{m=1} + (1/(1-alpha)) * sum_{i=m}^{n} a_i C(i,m) (-1)^m <w^{shuffle(i-m)}, E_mu[S(X)]>

Cost note: w^{shuffle k} lives in tensor levels 0..N*k, so evaluating the
objective needs the *expected* signature up to level N*n. This is only ever
computed once (from the reference/training set) and is independent of how
many windows are later scored, but it does mean this model is deliberately
run at a smaller signature depth (`ocsvm_depth`, default 2) than the other
models in this project -- otherwise N*n blows up combinatorially. This is a
scoping choice for tractability on a laptop, not a limitation of the
theorem itself.

Known limitation, observed on real BTC data (not on the small synthetic
Brownian-motion tests this was validated against): the optimizer tends to
push ||w|| toward whatever bound is imposed rather than settle at a clean
interior optimum -- i.e. the regularized objective doesn't clearly
penalize growing ||w|| for this (a_coeffs, mean_signature) combination.
The most likely cause is that a single global degree-2 polynomial is a
poor approximation of the max(.,0) hinge over the wide score range real,
fat-tailed, autocorrelated crypto returns produce (unlike the paper's
clean synthetic Brownian-motion-with-a-spike setting). `w_bound` below
keeps this from diverging outright, but the resulting w/rho should be
treated as a best-effort fit, not a fully converged optimum -- a natural
next step for improving it is a higher `poly_degree` (better hinge
approximation, at combinatorial cost) or fitting on a less
overlapping/more i.i.d.-like resample of the reference period.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from src.features.signatures import signature
from src.models.shuffle_algebra import pair, shuffle_power


def fit_max_polynomial(K: float, degree: int, n_points: int = 4000) -> np.ndarray:
    """Least-squares polynomial approximation of max(x, 0) on [-K, K],
    returned as coefficients [a_0, a_1, ..., a_degree] (a_0 first)."""
    xs = np.linspace(-K, K, n_points)
    ys = np.maximum(xs, 0.0)
    coeffs_highest_first = np.polyfit(xs, ys, degree)
    return coeffs_highest_first[::-1]


def _split_levels(flat: np.ndarray, d: int, depth: int) -> list[np.ndarray]:
    """Split a level-0-excluded flat vector (as produced by
    src.features.signatures.signature) into a level0=0-prefixed list of
    per-level arrays, for use with the shuffle algebra."""
    levels = [np.array([0.0])]
    offset = 0
    for k in range(1, depth + 1):
        size = d ** k
        levels.append(flat[offset:offset + size])
        offset += size
    return levels


def mean_signature_levels(paths: list[np.ndarray], total_depth: int, d: int) -> list[np.ndarray]:
    """E_mu[S(X)] estimated from a reference/training set of paths,
    including level 0 (always 1), up to `total_depth` (= ocsvm_depth * n)."""
    sigs = [signature(p, total_depth, include_level_zero=True) for p in paths]
    mean_flat = np.mean(sigs, axis=0)
    levels, offset = [], 0
    for k in range(total_depth + 1):
        size = d ** k
        levels.append(mean_flat[offset:offset + size])
        offset += size
    return levels


def _b_coefficients(w_levels: list[np.ndarray], d: int, alpha: float,
                     a_coeffs: np.ndarray, mean_sig: list[np.ndarray]) -> np.ndarray:
    n = len(a_coeffs) - 1
    w_powers = [shuffle_power(w_levels, k, d) for k in range(n + 1)]  # w^{shuffle 0..n}
    b = np.zeros(n + 1)
    for m in range(n + 1):
        total = 0.0
        for i in range(m, n + 1):
            total += a_coeffs[i] * comb(i, m) * ((-1) ** m) * pair(w_powers[i - m], mean_sig)
        b[m] = (1.0 if m == 1 else 0.0) + total / (1.0 - alpha)
    return b


def _min_poly_over_rho(b: np.ndarray, K: float) -> tuple[float, float]:
    """min_{rho in [-K,K]} sum_m b_m rho^m, returned as (rho*, value)."""
    if len(b) == 3 and b[2] > 0:
        rho_unclipped = -b[1] / (2 * b[2])
        rho_star = float(np.clip(rho_unclipped, -K, K))
        value = float(np.polyval(b[::-1], rho_star))
        return rho_star, value
    res = minimize_scalar(lambda r: np.polyval(b[::-1], r), bounds=(-K, K), method="bounded")
    return float(res.x), float(res.fun)


@dataclass
class CVaROCSVM:
    ocsvm_depth: int          # signature truncation depth N used for w
    d: int                    # path channel count
    alpha: float = 0.9        # CVaR confidence level
    K: float = 10.0           # polynomial approximation / rho search range
    poly_degree: int = 2      # degree n of the smooth max(.,0) surrogate

    def fit(self, train_paths: list[np.ndarray]) -> "CVaROCSVM":
        """Solve w* = argmin_w { CVaR_alpha(-<w,S_N(X)>) + 0.5||w||^2 }.

        Eq. 2.10's CVaR term is of the *negated* pairing -<w,phi(X)>, so we
        apply Theorem 2.6 to (-w) rather than w: (-w)^{shuffle k} =
        (-1)^k w^{shuffle k}, handled here by simply negating the flat
        vector before splitting into levels for the b_m computation.
        """
        self.a_coeffs_ = fit_max_polynomial(self.K, self.poly_degree)
        total_depth = self.ocsvm_depth * self.poly_degree
        self.mean_sig_ = mean_signature_levels(train_paths, total_depth, self.d)

        w_dim = sum(self.d ** k for k in range(1, self.ocsvm_depth + 1))

        def cvar_and_rho(w_flat: np.ndarray) -> tuple[float, float]:
            neg_w_levels = _split_levels(-w_flat, self.d, self.ocsvm_depth)
            b = _b_coefficients(neg_w_levels, self.d, self.alpha, self.a_coeffs_, self.mean_sig_)
            return _min_poly_over_rho(b, self.K)

        def objective(w_flat: np.ndarray) -> float:
            _, cvar_value = cvar_and_rho(w_flat)
            return 0.5 * float(np.dot(w_flat, w_flat)) + cvar_value

        # The degree-n polynomial only approximates max(.,0) accurately
        # inside [-K, K]; if the search is allowed to push ||w|| (and hence
        # the argument <w,S_N(x)>) far outside that range, the surrogate
        # stops being a valid CVaR approximation and the "optimum" just
        # runs off to whatever direction the now-meaningless polynomial
        # tail rewards. Bound ||w|| so a typical training path's pairing
        # stays within [-K, K] (Cauchy-Schwarz: |<w,S(x)>| <= ||w|| ||S(x)||),
        # which is what keeps this well-posed in practice.
        train_sig_norms = [
            np.linalg.norm(signature(p, self.ocsvm_depth)) for p in train_paths
        ]
        typical_norm = max(np.median(train_sig_norms), 1e-6)
        w_bound = self.K / typical_norm
        bounds = [(-w_bound, w_bound)] * w_dim

        w0 = np.zeros(w_dim)
        result = minimize(objective, w0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000})
        self.w_ = result.x
        # Per the OCSVM/CVaR equivalence (Tsyurmasto et al. 2014, cited in
        # eq. 2.10) the CVaR variational auxiliary variable at the optimal w
        # plays the role of the OCSVM decision offset rho in
        # g(x) = <w,phi(x)> - rho (positive = normal).
        self.rho_, _ = cvar_and_rho(self.w_)
        self.optim_result_ = result
        return self

    def score(self, path: np.ndarray) -> float:
        """Anomaly score = rho* - <w*, S_N(x)>, i.e. -g(x) in the OCSVM
        decision-function convention -- higher means more anomalous,
        consistent with every other model in this project."""
        sig = signature(path, self.ocsvm_depth, include_level_zero=False)
        return float(self.rho_ - np.dot(self.w_, sig))

    def score_many(self, paths: list[np.ndarray]) -> np.ndarray:
        return np.array([self.score(p) for p in paths])
