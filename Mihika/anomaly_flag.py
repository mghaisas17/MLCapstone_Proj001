"""HMM anomaly flag: "is the market in its stress regime right now?"

An unsupervised Gaussian HMM is fit on a bundle of per-bar features. Its
*filtered* probability of the stress state (the highest-volatility state) is the
soft flag; thresholding it gives the 0/1 flag used as an extra input to the
volatility forecasts.

No look-ahead:
  * only filtered (forward) probabilities are used, never smoothed/Viterbi ones;
  * the HMM and feature scaler are fit on observations that ended before the test
    block starts;
  * a row/bar only sees state bars that had completed by the time it is known.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12
FEATURE_NAMES = ("ret", "log_range", "rel_volume", "flow_imbalance", "trailing_vol")


# --------------------------------------------------------------------------
# Diagonal-covariance Gaussian HMM (scaled forward-backward, vectorized)
# --------------------------------------------------------------------------


class DiagGaussianHMM:
    def __init__(self, n_states: int = 3, n_iter: int = 60, tol: float = 1e-4, seed: int = 0):
        self.K, self.n_iter, self.tol, self.seed = n_states, n_iter, tol, seed

    def _emission(self, X):
        """Per-row-rescaled emission likelihoods B (T,K) and the log offsets removed."""
        diff = X[:, None, :] - self.means_[None]
        logB = -0.5 * (np.log(2 * np.pi * self.vars_)[None].sum(-1) + (diff**2 / self.vars_[None]).sum(-1))
        m = logB.max(axis=1)
        return np.exp(logB - m[:, None]), m

    def _forward(self, B):
        T = B.shape[0]
        alpha, c = np.empty_like(B), np.empty(T)
        a = self.pi_ * B[0]
        c[0] = a.sum() + EPS
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A_) * B[t]
            c[t] = a.sum() + EPS
            alpha[t] = a / c[t]
        return alpha, c

    def _backward(self, B, c):
        T = B.shape[0]
        beta = np.ones_like(B)
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A_ @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def _fit_once(self, X, seed):
        rng = np.random.default_rng(seed)
        T, d = X.shape
        K = self.K
        self.means_ = X[rng.choice(T, size=K, replace=False)].copy()
        self.vars_ = np.tile(X.var(axis=0) + 1e-6, (K, 1))
        self.A_ = np.full((K, K), 0.1 / max(K - 1, 1)) + np.eye(K) * (0.9 - 0.1 / max(K - 1, 1))
        self.pi_ = np.full(K, 1.0 / K)
        prev = -np.inf
        ll = -np.inf
        for _ in range(self.n_iter):
            B, m = self._emission(X)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            ll = float(np.log(c).sum() + m.sum())
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + EPS
            xi = self.A_ * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
            self.pi_ = gamma[0] / gamma[0].sum()
            self.A_ = xi / (xi.sum(axis=1, keepdims=True) + EPS)
            w = gamma.sum(axis=0) + EPS
            self.means_ = (gamma.T @ X) / w[:, None]
            self.vars_ = np.stack(
                [(gamma[:, k, None] * (X - self.means_[k]) ** 2).sum(0) / w[k] for k in range(K)]
            ) + 1e-6
            if abs(ll - prev) < self.tol * max(1.0, abs(ll)):
                break
            prev = ll
        return ll

    def fit(self, X: np.ndarray, restarts: int = 3) -> "DiagGaussianHMM":
        best_ll, best = -np.inf, None
        for r in range(restarts):
            ll = self._fit_once(X, self.seed + r)
            if ll > best_ll:
                best_ll = ll
                best = (self.means_.copy(), self.vars_.copy(), self.A_.copy(), self.pi_.copy())
        self.means_, self.vars_, self.A_, self.pi_ = best
        self.log_likelihood_ = best_ll
        return self

    def filter(self, X: np.ndarray) -> np.ndarray:
        """P(state_t | x_1..x_t) with the fitted parameters held fixed (causal)."""
        B, _ = self._emission(X)
        alpha, _ = self._forward(B)
        return alpha


def to_ns(values) -> np.ndarray:
    """UTC datetimes as datetime64[ns], independent of pandas' datetime resolution."""
    idx = pd.DatetimeIndex(values)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return np.asarray(idx.tz_localize(None), dtype="datetime64[ns]")


# --------------------------------------------------------------------------
# Features, fitting, and alignment to forecast rows / fine bars
# --------------------------------------------------------------------------


def state_bars(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate fine bars to the HMM's observation cadence."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "dollar_volume": "sum"}
    if "signed_dollar_flow" in bars.columns:
        agg["signed_dollar_flow"] = "sum"
    sb = bars.resample(freq, label="left", closed="left").agg(agg)
    return sb.dropna(subset=["close"])


def state_features(sb: pd.DataFrame, names=FEATURE_NAMES) -> pd.DataFrame:
    """Causal per-state-bar features (each uses only the current and earlier bars)."""
    ret = np.log(sb["close"]).diff().fillna(0.0)
    feats = {
        "ret": ret,
        "log_range": np.log(sb["high"] / sb["low"] + 1e-8),
        "rel_volume": np.log1p(sb["dollar_volume"])
        - np.log1p(sb["dollar_volume"]).rolling(30, min_periods=1).mean(),
        "trailing_vol": ret.rolling(10, min_periods=2).std().fillna(0.0),
    }
    if "signed_dollar_flow" in sb.columns:
        feats["flow_imbalance"] = sb["signed_dollar_flow"] / (sb["dollar_volume"] + EPS)
    chosen = [n for n in names if n in feats]
    if not chosen:
        raise ValueError(f"None of the requested flag features {names} are available.")
    return pd.DataFrame({n: feats[n] for n in chosen}, index=sb.index)


@dataclass
class FlagFit:
    p_stress: pd.Series  # indexed by the *end* time of each state bar
    threshold: float
    n_train_obs: int
    feature_names: tuple
    stress_state: int

    def at(self, known_times) -> np.ndarray:
        """Stress probability using only state bars completed by each known time."""
        ends = to_ns(self.p_stress.index)
        idx = np.searchsorted(ends, to_ns(known_times), side="right") - 1
        out = np.zeros(len(idx))
        ok = idx >= 0
        out[ok] = self.p_stress.to_numpy()[idx[ok]]
        return out

    def flag(self, p: np.ndarray) -> np.ndarray:
        return (p >= self.threshold).astype(float)


def fit_flag(
    bars: pd.DataFrame,
    flag_freq: str,
    cutoff,
    *,
    n_states: int = 3,
    features=FEATURE_NAMES,
    threshold: float = 0.5,
    restarts: int = 3,
    seed: int = 0,
    clip: float = 6.0,
) -> FlagFit:
    """Fit on state bars that ended at or before ``cutoff``; filter the whole series."""
    sb = state_bars(bars, flag_freq)
    F = state_features(sb, features)
    ends = sb.index + pd.Timedelta(flag_freq)
    cutoff = pd.Timestamp(cutoff)
    train = np.asarray(ends <= cutoff)
    if train.sum() < max(30, 5 * n_states):
        raise ValueError(f"Only {int(train.sum())} state bars before the cutoff; cannot fit the flag HMM.")
    mu, sd = F.values[train].mean(0), F.values[train].std(0) + 1e-9
    Z = np.clip((F.values - mu) / sd, -clip, clip)

    hmm = DiagGaussianHMM(n_states, seed=seed).fit(Z[train], restarts=restarts)
    names = tuple(F.columns)
    stress = int(np.argmax(hmm.vars_[:, names.index("ret")])) if "ret" in names else int(np.argmax(hmm.vars_.sum(1)))
    p = hmm.filter(Z)[:, stress]
    return FlagFit(pd.Series(p, index=ends), threshold, int(train.sum()), names, stress)


def episodes(flag: np.ndarray) -> tuple[int, float]:
    """(# of flagged episodes, mean episode length in rows)."""
    f = np.asarray(flag).astype(int)
    if f.sum() == 0:
        return 0, 0.0
    starts = int(((f[1:] == 1) & (f[:-1] == 0)).sum() + f[0])
    return starts, float(f.sum() / max(starts, 1))
