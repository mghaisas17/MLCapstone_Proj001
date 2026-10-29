"""Anomaly detectors with one interface: ``fit(X_train)`` then ``score(X)`` (higher = more unusual).

mahalanobis  distance from the training mean under a Ledoit-Wolf covariance. On signature
             features this is the expected-signature-distance / conformance score family
             from Aayush's work (distance of a path's signature to the expected signature).
iforest      Isolation Forest: unusual points are isolated by fewer random splits.
autoencoder  small MLP that compresses and reconstructs the features; score = reconstruction
             error. Early stopping on the last 20% of the training rows.
hmm          features -> PCA (fit on training rows) -> 3-state Gaussian HMM fit on the training
             rows in time order; score = -log p(x_t | x_1..x_{t-1}) from the causal forward
             filter, i.e. how unexpected a window is given the regime the model believes it is in.
             Rows must be passed in time order (they are: folds are contiguous blocks).

All standardize features with statistics from the training rows only.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class _Scaled:
    def _fit_scaler(self, X):
        self.scaler = StandardScaler().fit(X)
        keep = self.scaler.scale_ > 1e-12  # drop constant columns (e.g. the time increment)
        self.keep = keep if keep.any() else np.ones_like(keep)
        return self._t(X)

    def _t(self, X):
        return np.nan_to_num(self.scaler.transform(X)[:, self.keep], nan=0.0, posinf=0.0, neginf=0.0)


class Mahalanobis(_Scaled):
    def fit(self, X):
        Z = self._fit_scaler(X)
        lw = LedoitWolf().fit(Z)
        self.mean, self.prec = lw.location_, lw.precision_
        return self

    def score(self, X):
        D = self._t(X) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, self.prec, D), 0.0))


class IForest(_Scaled):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X):
        self.model = IsolationForest(n_estimators=200, max_samples=min(256, len(X)), random_state=self.seed,
                                     n_jobs=1).fit(self._fit_scaler(X))
        return self

    def score(self, X):
        return -self.model.score_samples(self._t(X))


class AutoEncoder(_Scaled):
    def __init__(self, epochs: int = 60, patience: int = 8, seed: int = 0, code: int = 16):
        self.epochs, self.patience, self.seed, self.code = epochs, patience, seed, code

    def fit(self, X):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        Z = self._fit_scaler(X).astype(np.float32)
        d = Z.shape[1]
        hidden = int(min(128, max(self.code * 2, d // 2)))
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, self.code), nn.ReLU(),
                                 nn.Linear(self.code, hidden), nn.ReLU(), nn.Linear(hidden, d))
        cut = int(len(Z) * 0.8)  # time-ordered: validate on the latest training rows
        Zf, Zv = torch.tensor(Z[:cut]), torch.tensor(Z[cut:])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        loader = torch.utils.data.DataLoader(Zf, batch_size=256, shuffle=True)
        best, best_state, since = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            for xb in loader:
                opt.zero_grad()
                nn.functional.mse_loss(self.net(xb), xb).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(nn.functional.mse_loss(self.net(Zv), Zv)) if len(Zv) else 0.0
            if val < best - 1e-9:
                best, since = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def score(self, X):
        import torch

        Z = torch.tensor(self._t(X).astype(np.float32))
        with torch.no_grad():
            return ((self.net(Z) - Z) ** 2).mean(dim=1).numpy()


class HMM(_Scaled):
    def __init__(self, n_states: int = 3, n_components: int = 5, seed: int = 0):
        self.n_states, self.n_components, self.seed = n_states, n_components, seed

    def fit(self, X):
        from sklearn.decomposition import PCA
        from anomaly_flag import DiagGaussianHMM

        Z = self._fit_scaler(X)
        self.pca = PCA(n_components=min(self.n_components, Z.shape[1]), random_state=self.seed).fit(Z)
        P = self.pca.transform(Z)
        self.p_scale = P.std(axis=0) + 1e-9
        self.hmm = DiagGaussianHMM(self.n_states, seed=self.seed).fit(P / self.p_scale, restarts=3)
        return self

    def score(self, X):
        P = self.pca.transform(self._t(X)) / self.p_scale
        return -self.hmm.predictive_loglik(np.clip(P, -20, 20))


def make(name: str, cfg) -> object:
    if name == "mahalanobis":
        return Mahalanobis()
    if name == "iforest":
        return IForest(seed=cfg.seed)
    if name == "autoencoder":
        return AutoEncoder(epochs=cfg.ae_epochs, patience=cfg.ae_patience, seed=cfg.seed)
    if name == "hmm":
        return HMM(n_states=getattr(cfg, "hmm_states", 3), n_components=getattr(cfg, "hmm_components", 5),
                   seed=cfg.seed)
    raise ValueError(f"unknown detector {name!r}")
