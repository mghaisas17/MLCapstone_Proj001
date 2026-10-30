"""Strategy performance metrics, computed on a daily log-return series."""
from __future__ import annotations

import numpy as np


def sharpe(returns: np.ndarray, periods_per_year: int = 365) -> float:
    r = np.asarray(returns)
    if r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.exp(np.cumsum(returns))
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def cumulative_return(returns: np.ndarray) -> float:
    return float(np.exp(np.sum(returns)) - 1.0)


def turnover(actions_or_m) -> float:
    """Mean absolute day-over-day change in exposure -- how much the strategy trades."""
    a = np.asarray(actions_or_m, dtype=float)
    return float(np.mean(np.abs(np.diff(a)))) if len(a) > 1 else 0.0


def summarize(name: str, returns: np.ndarray, exposure: np.ndarray) -> dict:
    return dict(
        strategy=name,
        cumulative_return=cumulative_return(returns),
        sharpe=sharpe(returns),
        max_drawdown=max_drawdown(returns),
        turnover=turnover(exposure),
        mean_exposure=float(np.mean(exposure)),
    )
