"""Simple, non-signature engineered features per rolling window: the "does
the signature representation even matter" comparator. Built directly from
raw OHLCV bars using classical technical/statistical summaries, with no path
construction or signature machinery involved at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.paths.windows import make_windows

FEATURE_NAMES = [
    "realized_vol", "skew", "kurtosis", "total_return",
    "max_drawdown", "mean_volume", "mean_range",
]


def build_engineered_features(df: pd.DataFrame, window: int, step: int) -> pd.DataFrame:
    """One row per window (indexed by window end-time): realized volatility,
    skew, excess kurtosis, and total (net) return of daily log-returns; max
    drawdown within the window; mean volume; mean intraday log-range."""
    windows = make_windows(df, window, step)
    rows, end_times = [], []
    for w in windows:
        d = w.data
        close = d["close"].to_numpy(dtype=float)
        log_ret = np.diff(np.log(close))
        cum_log_ret = np.cumsum(log_ret)

        realized_vol = float(log_ret.std())
        skew = float(scipy_stats.skew(log_ret)) if len(log_ret) > 2 else 0.0
        kurt = float(scipy_stats.kurtosis(log_ret)) if len(log_ret) > 2 else 0.0
        total_return = float(cum_log_ret[-1]) if len(cum_log_ret) else 0.0

        path = np.concatenate(([0.0], cum_log_ret))
        running_max = np.maximum.accumulate(path)
        max_drawdown = float((path - running_max).min())

        mean_volume = float(d["volume"].to_numpy(dtype=float).mean())
        mean_range = float((np.log(d["high"].to_numpy(dtype=float)) - np.log(d["low"].to_numpy(dtype=float))).mean())

        rows.append([realized_vol, skew, kurt, total_return, max_drawdown, mean_volume, mean_range])
        end_times.append(w.end_time)

    return pd.DataFrame(rows, index=pd.DatetimeIndex(end_times, name="window_end"), columns=FEATURE_NAMES)
