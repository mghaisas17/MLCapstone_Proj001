"""Bull/bear and high/low-volatility regime labeling from daily BTC prices,
used to check whether anomaly-score peaks cluster near regime transitions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bull_bear_labels(daily_close: pd.Series, ma_window: int = 200) -> pd.Series:
    """Simple trend-following regime label: bull when price is above its
    own `ma_window`-day moving average, bear otherwise."""
    ma = daily_close.rolling(ma_window).mean()
    return (daily_close > ma).map({True: "bull", False: "bear"})


def volatility_regime_labels(daily_close: pd.Series, vol_window: int = 20, quantile: float = 0.66) -> pd.Series:
    """High/low volatility regime via a rolling realized-vol quantile
    split: "high" when trailing realized vol is above its own historical
    `quantile`, "low" otherwise."""
    log_ret = np.log(daily_close).diff()
    realized_vol = log_ret.rolling(vol_window).std()
    threshold = realized_vol.quantile(quantile)
    return (realized_vol > threshold).map({True: "high_vol", False: "low_vol"})


def regime_transition_dates(labels: pd.Series) -> pd.DatetimeIndex:
    """Timestamps where the regime label changes from the previous day."""
    changed = labels != labels.shift(1)
    return labels.index[changed.fillna(False)]


def find_calm_stretches(
    daily_close: pd.Series,
    events: dict[str, dict[str, str]],
    min_length_days: int = 45,
    vol_window: int = 20,
    quantile: float = 0.66,
    event_buffer_days: int = 30,
) -> list[tuple[str, str]]:
    """Auto-detect calm ("low_vol") stretches across the full history, long
    enough and far enough from every labeled event to serve as additional
    reference/calibration periods -- reuses `volatility_regime_labels`
    (already computes a per-day low/high-vol split) rather than a separate
    detector. Returns (start, end) ISO date-string pairs, sorted by start.

    A stretch survives if it's a contiguous "low_vol" run of at least
    `min_length_days`, and doesn't come within `event_buffer_days` of any
    labeled event window in `events` (configs/horizons.yaml's `events` dict).
    """
    labels = volatility_regime_labels(daily_close, vol_window=vol_window, quantile=quantile).dropna()

    # Group into contiguous runs of the same label via a change-point cumsum.
    run_id = (labels != labels.shift(1)).cumsum()
    candidates = []
    for _, run in labels.groupby(run_id):
        if run.iloc[0] != "low_vol":
            continue
        start, end = run.index[0], run.index[-1]
        if (end - start).days + 1 < min_length_days:
            continue
        candidates.append((start, end))

    buffer = pd.Timedelta(days=event_buffer_days)
    event_windows = [
        (pd.Timestamp(ev["start"], tz="UTC") - buffer, pd.Timestamp(ev["end"], tz="UTC") + buffer)
        for ev in events.values()
    ]

    def overlaps_any_event(start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return any(start <= ev_end and end >= ev_start for ev_start, ev_end in event_windows)

    calm = [(s, e) for s, e in candidates if not overlaps_any_event(s, e)]
    calm.sort(key=lambda p: p[0])
    return [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in calm]


def score_near_transitions(
    score_series: pd.Series,
    transition_dates: pd.DatetimeIndex,
    tolerance: pd.Timedelta,
    threshold: float,
    higher_is_anomalous: bool = True,
) -> float:
    """Fraction of regime transitions for which the score crosses
    `threshold` at least once within `tolerance` of the transition date --
    i.e. whether the model actually flags an anomaly there, not merely
    whether a window happens to exist nearby (which would be true almost
    everywhere given how densely windows overlap)."""
    if len(transition_dates) == 0:
        return float("nan")
    hits = 0
    for t in transition_dates:
        nearby = score_series[(score_series.index >= t - tolerance) & (score_series.index <= t + tolerance)]
        if len(nearby) == 0:
            continue
        crossed = (nearby >= threshold) if higher_is_anomalous else (nearby <= threshold)
        if crossed.any():
            hits += 1
    return hits / len(transition_dates)
