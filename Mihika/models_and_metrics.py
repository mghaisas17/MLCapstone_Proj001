from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

EPS = 1e-12
LOGGER = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model: Any
    predictions: np.ndarray
    metrics: dict[str, float]


def fit_har(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    LOGGER.info("Fitting HAR linear regression")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 400,
    min_samples_leaf: int = 5,
    max_features: str | float = "sqrt",
    random_state: int = 42,
) -> RandomForestRegressor:
    LOGGER.info("Fitting Random Forest")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def fit_signature_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alphas: np.ndarray | None = None,
    n_splits: int = 5,
) -> Pipeline:
    LOGGER.info("Fitting Signature + Ridge with time-series CV")
    if alphas is None:
        alphas = np.logspace(-4, 4, 9)

    max_splits = max(2, min(n_splits, len(X_train) - 1))
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": alphas},
        scoring="neg_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=max_splits),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    LOGGER.info("Best Ridge alpha: %s", search.best_params_["ridge__alpha"])
    return search.best_estimator_


def fit_signature_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> XGBRegressor:
    LOGGER.info("Fitting Signature + XGBoost")
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
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual_var = np.asarray(y_true, dtype=float) ** 2
    forecast_var = np.maximum(np.asarray(y_pred, dtype=float) ** 2, EPS)
    return float(np.mean(np.log(forecast_var) + actual_var / forecast_var))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), EPS)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "QLIKE": qlike(y_true, y_pred),
    }


def evaluate_model(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ModelResult:
    pred = np.maximum(model.predict(X_test), EPS)
    metrics = regression_metrics(y_test, pred)
    LOGGER.info(
        "%s | RMSE=%.6g MAE=%.6g QLIKE=%.6g",
        name,
        metrics["RMSE"],
        metrics["MAE"],
        metrics["QLIKE"],
    )
    return ModelResult(name=name, model=model, predictions=pred, metrics=metrics)


def fit_and_evaluate_standard_models(
    dataset: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    include_xgb: bool = True,
) -> dict[str, ModelResult]:
    """
    Fit the same model families as the original notebook:
      - HAR-RV on X_har
      - Random Forest on summary statistics
      - Ridge on signature features
      - XGBoost on signature features
    """
    y_train = dataset.y[train_idx]
    y_test = dataset.y[test_idx]
    results: dict[str, ModelResult] = {}

    har = fit_har(dataset.X_har[train_idx], y_train)
    results["HAR"] = evaluate_model("HAR", har, dataset.X_har[test_idx], y_test)

    rf = fit_random_forest(dataset.X_stats[train_idx], y_train)
    results["Random Forest"] = evaluate_model(
        "Random Forest", rf, dataset.X_stats[test_idx], y_test
    )

    ridge = fit_signature_ridge(dataset.X_sig[train_idx], y_train)
    results["Signature Ridge"] = evaluate_model(
        "Signature Ridge", ridge, dataset.X_sig[test_idx], y_test
    )

    if include_xgb:
        xgb = fit_signature_xgb(dataset.X_sig[train_idx], y_train)
        results["Signature XGBoost"] = evaluate_model(
            "Signature XGBoost", xgb, dataset.X_sig[test_idx], y_test
        )

    return results


def metrics_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    return (
        pd.DataFrame({name: result.metrics for name, result in results.items()})
        .T
        .sort_values("RMSE")
    )
