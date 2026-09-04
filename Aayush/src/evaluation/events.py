"""Check whether an anomaly-score/p-value time series actually flags the
labeled historical events in configs/horizons.yaml (COVID crash,
2021 BTC retail-mania runup/crash, 2022 crypto crashes, 2023 banking
crisis): detection lag, and false-positive rate elsewhere in the series.
"""
from __future__ import annotations

import pandas as pd


def detect_event(
    score_series: pd.Series,
    threshold: float,
    event_start: str,
    event_end: str,
    higher_is_anomalous: bool = True,
) -> dict:
    """`score_series` is indexed by window end-time. Returns whether the
    threshold was crossed inside [event_start, event_end], and if so, the
    lag from event_start to the first crossing."""
    event_start_ts, event_end_ts = pd.Timestamp(event_start, tz="UTC"), pd.Timestamp(event_end, tz="UTC")
    window = score_series.loc[event_start_ts:event_end_ts]
    crossed = (window >= threshold) if higher_is_anomalous else (window <= threshold)

    if not crossed.any():
        return {"detected": False, "lag": None, "first_crossing_time": None}

    first_crossing_time = crossed[crossed].index[0]
    lag = first_crossing_time - event_start_ts
    return {"detected": True, "lag": lag, "first_crossing_time": first_crossing_time}


def false_positive_rate_outside_events(
    score_series: pd.Series,
    threshold: float,
    event_windows: dict[str, dict[str, str]],
    higher_is_anomalous: bool = True,
) -> float:
    """Fraction of windows outside every labeled event window that still
    cross the threshold -- the "false alarm" rate for a given threshold."""
    mask_in_event = pd.Series(False, index=score_series.index)
    for ev in event_windows.values():
        start, end = pd.Timestamp(ev["start"], tz="UTC"), pd.Timestamp(ev["end"], tz="UTC")
        mask_in_event |= (score_series.index >= start) & (score_series.index <= end)

    outside = score_series[~mask_in_event]
    if len(outside) == 0:
        return float("nan")
    crossed = (outside >= threshold) if higher_is_anomalous else (outside <= threshold)
    return float(crossed.mean())


def evaluate_all_events(
    score_series: pd.Series,
    threshold: float,
    event_windows: dict[str, dict[str, str]],
    higher_is_anomalous: bool = True,
) -> pd.DataFrame:
    rows = []
    for name, ev in event_windows.items():
        result = detect_event(score_series, threshold, ev["start"], ev["end"], higher_is_anomalous)
        result["event"] = name
        rows.append(result)
    df = pd.DataFrame(rows).set_index("event")
    df.attrs["false_positive_rate"] = false_positive_rate_outside_events(
        score_series, threshold, event_windows, higher_is_anomalous
    )
    return df
