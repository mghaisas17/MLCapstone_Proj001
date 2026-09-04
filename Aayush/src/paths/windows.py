"""Rolling-window slicing of a time-indexed DataFrame into overlapping paths.

Each window becomes one "path" that gets turned into channels, transformed,
and signature-encoded downstream. Windows are defined in bar/trade counts
(native resolution of the horizon), not wall-clock time, matching
configs/horizons.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Window:
    index: int          # window sequence number
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    data: pd.DataFrame  # the raw rows (bars or ticks) inside this window


def make_windows(df: pd.DataFrame, window: int, step: int) -> list[Window]:
    """Slice `df` (sorted, time-indexed) into overlapping windows of `window`
    rows with stride `step`. The last partial window (if any) is dropped so
    every window has an identical number of rows -- required for
    fixed-shape signature feature matrices downstream."""
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    if len(df) < window:
        return []

    windows = []
    for i, start in enumerate(range(0, len(df) - window + 1, step)):
        chunk = df.iloc[start:start + window]
        windows.append(Window(index=i, start_time=chunk.index[0], end_time=chunk.index[-1], data=chunk))
    return windows
