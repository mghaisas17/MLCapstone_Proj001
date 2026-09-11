"""A minimal Gaussian Hidden Markov Model (diagonal covariance), fit via the
Baum-Welch EM algorithm, hand-rolled in pure NumPy.

Not from a library: `hmmlearn` requires compiling a pybind11 C++ extension,
which fails on this machine for the same reason `iisignature` does (broken
Xcode Command Line Tools, missing standard headers) -- see
`src/features/signatures.py` for the same workaround applied to signatures.
A 2-3 state HMM is a well-understood, moderate-complexity algorithm, so
hand-rolling it here is the same "verify against known cases, ship it"
approach used there, not a shortcut.

Used in this project as a non-signature comparator: fit on daily returns
(and optionally a couple of other simple per-day features) from the calm
reference periods, then treat the "high-volatility" state's posterior
probability on real history as a per-day anomaly-like score -- a
regime-switching baseline that never touches path signatures at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp


def _log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log-density of a diagonal-covariance Gaussian, evaluated for every
    row of `x` (T, d) against one (mean, var) pair (each shape (d,)).
    Returns shape (T,)."""
    d = mean.shape[0]
    diff = x - mean
    return -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff**2 / var, axis=1))


@dataclass
class GaussianHMM:
    n_states: int
    n_iter: int = 100
    tol: float = 1e-4
    seed: int = 0
    means_: np.ndarray = field(default=None, repr=False)
    variances_: np.ndarray = field(default=None, repr=False)
    transmat_: np.ndarray = field(default=None, repr=False)
    startprob_: np.ndarray = field(default=None, repr=False)
    log_likelihood_: list = field(default_factory=list, repr=False)

    def _log_emission_matrix(self, X: np.ndarray) -> np.ndarray:
        """(T, n_states) matrix of log P(x_t | state k)."""
        T = X.shape[0]
        log_B = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            log_B[:, k] = _log_gaussian_pdf(X, self.means_[k], self.variances_[k])
        return log_B

    def _forward_backward(self, log_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        T, K = log_B.shape
        log_pi = np.log(self.startprob_ + 1e-300)
        log_A = np.log(self.transmat_ + 1e-300)

        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = log_B[t] + logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

        log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, log_beta, log_likelihood

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """X: (T, d) observation matrix (a single contiguous sequence)."""
        rng = np.random.default_rng(self.seed)
        T, d = X.shape
        K = self.n_states

        # Init: means from random data points, variances from the overall
        # per-feature variance, uniform transition/start probabilities.
        init_idx = rng.choice(T, size=K, replace=False)
        self.means_ = X[init_idx].copy()
        overall_var = X.var(axis=0) + 1e-8
        self.variances_ = np.tile(overall_var, (K, 1))
        self.transmat_ = np.full((K, K), 1.0 / K)
        self.startprob_ = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        self.log_likelihood_ = []
        for _ in range(self.n_iter):
            log_B = self._log_emission_matrix(X)
            log_alpha, log_beta, ll = self._forward_backward(log_B)
            self.log_likelihood_.append(float(ll))

            # E-step: state posteriors (gamma) and pairwise transition posteriors (xi)
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)

            log_A = np.log(self.transmat_ + 1e-300)
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                log_xi_t = (log_alpha[t][:, None] + log_A + log_B[t + 1][None, :]
                            + log_beta[t + 1][None, :] - ll)
                xi_sum += np.exp(log_xi_t)

            # M-step
            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            for k in range(K):
                weight = gamma[:, k]
                w_sum = weight.sum() + 1e-12
                self.means_[k] = (weight[:, None] * X).sum(axis=0) / w_sum
                diff = X - self.means_[k]
                self.variances_[k] = (weight[:, None] * diff**2).sum(axis=0) / w_sum + 1e-8

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def state_posteriors(self, X: np.ndarray) -> np.ndarray:
        """(T, n_states) posterior P(state_t = k | all observations)."""
        log_B = self._log_emission_matrix(X)
        log_alpha, log_beta, ll = self._forward_backward(log_B)
        return np.exp(log_alpha + log_beta - ll)

    def high_vol_state(self) -> int:
        """Index of the state with the largest total emission variance --
        the "crisis"/high-volatility state, under the convention that a
        higher-variance state is the more anomalous one."""
        return int(np.argmax(self.variances_.sum(axis=1)))
