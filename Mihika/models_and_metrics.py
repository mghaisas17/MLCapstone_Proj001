from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from arch import arch_model
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBRegressor

EPS = 1e-12
LOGGER = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model: Any
    predictions: np.ndarray
    metrics: dict[str, float]


# --------------------------------------------------------------------------
# Metrics -- RMSE only (MAE/QLIKE dropped; DM tests handle significance)
# --------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), EPS)
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def evaluate_model(name: str, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> ModelResult:
    pred = np.maximum(model.predict(X_test), EPS)
    metrics = regression_metrics(y_test, pred)
    LOGGER.info("%s | RMSE=%.6g", name, metrics["RMSE"])
    return ModelResult(name=name, model=model, predictions=pred, metrics=metrics)


def metrics_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    return pd.DataFrame({name: result.metrics for name, result in results.items()}).T.sort_values("RMSE")


# --------------------------------------------------------------------------
# HAR-RV-L: log(RV) at each HAR window + leverage term, direct per-horizon fit
# --------------------------------------------------------------------------


def fit_har_log(X_har_train: np.ndarray, y_har_log_train: np.ndarray, hac_lags: int = 5) -> Any:
    """
    OLS on log(1/h * sum future RV + eps) ~ log_RV(1) + log_RV(7) + log_RV(30) + leverage.
    HAC (Newey-West) standard errors correct for the autocorrelation that
    overlapping h-day targets induce -- see model.summary() for coefficients/SEs.
    """
    bad_X = ~np.isfinite(X_har_train)
    bad_y = ~np.isfinite(y_har_log_train)
    if bad_X.any() or bad_y.any():
        bad_cols = np.where(bad_X.any(axis=0))[0].tolist()
        raise ValueError(
            f"fit_har_log: {bad_X.any(axis=1).sum()} row(s) of X_har (columns {bad_cols}) and "
            f"{bad_y.sum()} row(s) of y_har_log are non-finite (NaN/inf). validate_ohlc() should "
            "have already dropped non-positive OHLC bars -- if this still fires, inspect var_inc "
            "around the offending rows for another source of a zero/negative price ratio."
        )
    LOGGER.info("Fitting log-HAR-RV-L with HAC(%d) standard errors", hac_lags)
    X = sm.add_constant(X_har_train)
    return sm.OLS(y_har_log_train, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def predict_har_log(model: Any, X_har: np.ndarray, train_resid: np.ndarray, horizon_steps: int) -> np.ndarray:
    """
    Back-transform log-space HAR forecasts to the common volatility scale
    (sqrt of summed variance over the horizon), with a Duan smearing
    correction so exp() of a log-mean forecast isn't systematically biased low.
    """
    smear = float(np.mean(np.exp(train_resid)))
    pred_log_mean_rv = model.predict(sm.add_constant(X_har, has_constant="add"))
    mean_rv_hat = smear * np.exp(pred_log_mean_rv)
    return np.sqrt(np.maximum(mean_rv_hat * horizon_steps, EPS))


def evaluate_har_log(
    X_har_train: np.ndarray,
    y_har_log_train: np.ndarray,
    X_har_test: np.ndarray,
    y_test: np.ndarray,
    horizon_steps: int,
    name: str = "HAR-RV-L",
) -> ModelResult:
    model = fit_har_log(X_har_train, y_har_log_train)
    pred = predict_har_log(model, X_har_test, model.resid, horizon_steps)
    return ModelResult(name=name, model=model, predictions=pred, metrics=regression_metrics(y_test, pred))


# --------------------------------------------------------------------------
# GARCH(1,1) / EGARCH(1,1): fixed-parameter rolling forecast of cumulative
# h-day variance, used as the long-horizon econometric baselines.
# --------------------------------------------------------------------------


def _garch_spec(returns: pd.Series, vol: str) -> Any:
    return arch_model(returns * 100, mean="Zero", vol=vol, p=1, o=1 if vol == "EGARCH" else 0, q=1, rescale=False)


def fit_garch(returns_train: pd.Series, vol: str = "GARCH") -> Any:
    LOGGER.info("Fitting %s(1,1)", vol)
    return _garch_spec(returns_train, vol).fit(disp="off")


def garch_cumulative_forecast(
    fitted_res: Any,
    returns_full: pd.Series,
    anchor_times: pd.DatetimeIndex,
    horizon_steps: int,
    vol: str = "GARCH",
) -> np.ndarray:
    """
    h-step cumulative variance forecast at each anchor time, using the params
    fitted on the training window only, but the actual realized returns up to
    each anchor to drive the variance recursion (standard fixed-parameter
    rolling backtest -- no future information is used).
    """
    forecaster = _garch_spec(returns_full, vol)
    fc = forecaster.forecast(params=fitted_res.params, start=anchor_times[0], horizon=horizon_steps, reindex=False)
    variance = fc.variance.reindex(anchor_times).to_numpy()
    cumulative_var = variance.sum(axis=1) / 100**2
    return np.sqrt(np.maximum(cumulative_var, 0))


def evaluate_garch(
    returns_train: pd.Series,
    returns_full: pd.Series,
    anchor_times: pd.DatetimeIndex,
    y_test: np.ndarray,
    horizon_steps: int,
    vol: str = "GARCH",
) -> ModelResult:
    res = fit_garch(returns_train, vol=vol)
    pred = garch_cumulative_forecast(res, returns_full, anchor_times, horizon_steps, vol=vol)
    name = "EGARCH(1,1)" if vol == "EGARCH" else "GARCH(1,1)"
    return ModelResult(name=name, model=res, predictions=pred, metrics=regression_metrics(y_test, pred))


# --------------------------------------------------------------------------
# XGBoost -- one fitter reused for both the "stats" baseline and the
# signature model, so the only thing that differs is the feature set.
# --------------------------------------------------------------------------


def fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
    n_jobs: int = 1,
) -> XGBRegressor:
    LOGGER.info("Fitting XGBoost")
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=5.0,
        tree_method="hist",
        # Avoid a large temporary memory spike in notebooks.  Using every CPU
        # core can make histogram buffers and native worker memory large enough
        # for the OS to kill the kernel, especially after signature generation.
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def fit_signature_ridge(
    X_train: np.ndarray, y_train: np.ndarray, *, alphas: np.ndarray | None = None, n_splits: int = 5, n_jobs: int = 1
) -> Pipeline:
    """n_jobs defaults to 1 -- GridSearchCV's joblib backend pickles a full
    copy of X_train to every worker process, which multiplies memory use by
    the worker count for a large signature matrix. Raise it only if you have
    RAM to spare."""
    LOGGER.info("Fitting Signature + Ridge with time-series CV")
    alphas = np.logspace(-4, 4, 9) if alphas is None else alphas
    max_splits = max(2, min(n_splits, len(X_train) - 1))
    pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": alphas},
        scoring="neg_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=max_splits),
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    LOGGER.info("Best Ridge alpha: %s", search.best_params_["ridge__alpha"])
    return search.best_estimator_


def fit_and_evaluate_standard_models(
    dataset: Any, train_idx: np.ndarray, test_idx: np.ndarray, horizon_steps: int
) -> dict[str, ModelResult]:
    """HAR-RV-L, XGBoost on summary stats, Ridge and XGBoost on signatures."""
    y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
    results: dict[str, ModelResult] = {
        "HAR-RV-L": evaluate_har_log(
            dataset.X_har[train_idx], dataset.y_har_log[train_idx], dataset.X_har[test_idx], y_test, horizon_steps
        ),
    }
    xgb_stats = fit_xgb(dataset.X_stats[train_idx], y_train)
    results["XGBoost (stats)"] = evaluate_model("XGBoost (stats)", xgb_stats, dataset.X_stats[test_idx], y_test)

    ridge = fit_signature_ridge(dataset.X_sig[train_idx], y_train)
    results["Signature Ridge"] = evaluate_model("Signature Ridge", ridge, dataset.X_sig[test_idx], y_test)

    xgb_sig = fit_xgb(dataset.X_sig[train_idx], y_train)
    results["XGBoost (signature)"] = evaluate_model("XGBoost (signature)", xgb_sig, dataset.X_sig[test_idx], y_test)
    return results


# --------------------------------------------------------------------------
# PyTorch models -- one small training loop reused by the MLP (plain and
# +signature variants) and the signature-LSTM; only the architecture and the
# feature shape (2-D vs 3-D sequence) differ.
# --------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 64, 16)):
        super().__init__()
        dims = [in_dim, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SignatureLSTM(nn.Module):
    """Consumes a sequence of per-sub-window signature vectors (one 'channel'
    per 5-day chunk of the lookback) and reads out the final hidden state."""

    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


@dataclass
class TorchRegressor:
    """Bundles a fitted torch model with its feature scaler so it exposes the
    same .predict(X) interface as every sklearn/xgboost model above."""

    model: nn.Module
    scaler: StandardScaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            n, t, f = X.shape
            Xs = self.scaler.transform(X.reshape(-1, f)).reshape(n, t, f)
        else:
            Xs = self.scaler.transform(X)
        with torch.no_grad():
            return self.model(torch.tensor(Xs, dtype=torch.float32)).numpy()


def _fit_scaler(X: np.ndarray) -> StandardScaler:
    flat = X.reshape(-1, X.shape[-1]) if X.ndim == 3 else X
    return StandardScaler().fit(flat)


def _train_torch(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    weight_decay: float,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 64,
    random_state: int = 42,
) -> nn.Module:
    torch.manual_seed(random_state)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=min(batch_size, len(X)), shuffle=True
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    weight_decay: float = 1e-4,
    hidden: tuple[int, ...] = (128, 64, 16),
    **kw: Any,
) -> TorchRegressor:
    """Plain MLP (e.g. on X_stats) or MLP+signature (on X_stats concatenated
    with X_sig) -- same function, only the X passed in differs."""
    LOGGER.info("Fitting MLP %s (weight_decay=%.1e)", hidden, weight_decay)
    scaler = _fit_scaler(X_train)
    model = _train_torch(
        MLP(X_train.shape[1], hidden), scaler.transform(X_train), y_train, weight_decay=weight_decay, **kw
    )
    return TorchRegressor(model, scaler)


def fit_signature_lstm(
    X_seq_train: np.ndarray, y_train: np.ndarray, *, weight_decay: float = 1e-4, hidden: int = 32, **kw: Any
) -> TorchRegressor:
    LOGGER.info("Fitting signature-LSTM (hidden=%d, weight_decay=%.1e)", hidden, weight_decay)
    scaler = _fit_scaler(X_seq_train)
    n, t, f = X_seq_train.shape
    X_scaled = scaler.transform(X_seq_train.reshape(-1, f)).reshape(n, t, f)
    model = _train_torch(SignatureLSTM(f, hidden), X_scaled, y_train, weight_decay=weight_decay, **kw)
    return TorchRegressor(model, scaler)


# --------------------------------------------------------------------------
# Walk-forward evaluation + Diebold-Mariano comparison
# --------------------------------------------------------------------------


def walk_forward_folds(n: int, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds (reuses sklearn's TimeSeriesSplit) shared by
    every model so all comparisons use the same rolling-origin backtest."""
    return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n)))


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, h: int) -> tuple[float, float]:
    """
    Diebold-Mariano test on squared-error loss, with a Newey-West long-run
    variance using h-1 lags to account for the autocorrelation that
    overlapping h-step-ahead forecasts induce. Returns (dm_stat, two-sided p).
    """
    d = errors_a**2 - errors_b**2
    n = len(d)
    var_d = np.var(d, ddof=0)
    for lag in range(1, min(h, n - 1)):
        weight = 1 - lag / h
        var_d += 2 * weight * np.cov(d[lag:], d[:-lag])[0, 1]
    var_d = max(var_d, 1e-12) / n
    dm_stat = float(d.mean() / np.sqrt(var_d))
    p_value = float(2 * (1 - sp_stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_value


@dataclass
class WalkForwardResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return self.y_true - self.y_pred

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.errors**2)))


def compare_models(results: dict[str, WalkForwardResult], horizon_steps: int) -> pd.DataFrame:
    """RMSE-ranked table with a Diebold-Mariano test of every model against
    the current best -- this is the only comparison table the project needs."""
    best_name = min(results, key=lambda k: results[k].rmse)
    best_errors = results[best_name].errors
    rows = []
    for name, res in results.items():
        is_best = name == best_name
        dm_stat, p_value = (np.nan, np.nan) if is_best else diebold_mariano(res.errors, best_errors, horizon_steps)
        rows.append(
            {"Model": name, "RMSE": res.rmse, "Best": is_best, "DM stat vs best": dm_stat, "DM p-value": p_value}
        )
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
