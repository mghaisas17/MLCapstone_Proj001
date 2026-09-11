"""Glue: raw OHLCV/tick data -> rolling windows -> channels -> transforms
-> signature feature matrix, driven by one horizon's entry in
configs/horizons.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.signatures import signature
from src.paths.transforms import (
    add_basepoint,
    extract_channels,
    fit_normalization_scale,
    invisibility_reset,
    normalize,
)
from src.paths.windows import make_windows


@dataclass
class HorizonConfig:
    name: str
    source: str            # "binance_ticks" | "binance_klines_1m" | "yahoo_daily"
    window: int
    step: int
    depth: int
    channels: list[str]

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "HorizonConfig":
        return cls(name=name, source=d["source"], window=d["window"], step=d["step"],
                   depth=d["depth"], channels=list(d["channels"]))


def raw_paths_for_horizon(df: pd.DataFrame, cfg: HorizonConfig) -> tuple[list[pd.Timestamp], list[np.ndarray]]:
    """Slice `df` into this horizon's windows and extract each one's raw
    (unnormalized) channel matrix. Returns (window_end_times, raw_paths)."""
    windows = make_windows(df, cfg.window, cfg.step)
    end_times = [w.end_time for w in windows]
    raw_paths = [extract_channels(w.data, cfg.channels, cfg.source) for w in windows]
    return end_times, raw_paths


def build_signature_features(
    raw_paths: list[np.ndarray],
    cfg: HorizonConfig,
    scale: np.ndarray,
    use_invisibility_reset: bool = False,
) -> np.ndarray:
    """Apply normalization + basepoint (+ optional invisibility-reset) and
    compute the truncated signature feature matrix, shape
    (n_windows, feature_dim)."""
    feats = []
    for path in raw_paths:
        p = normalize(path, scale)
        p = add_basepoint(p)
        if use_invisibility_reset:
            p = invisibility_reset(p)
        feats.append(signature(p, cfg.depth))
    return np.vstack(feats)


def raw_paths_by_reference_period(
    df: pd.DataFrame, cfg: HorizonConfig, periods: list[dict]
) -> list[list[np.ndarray]]:
    """Like `raw_paths_for_horizon`, but run once per disjoint reference
    period and kept separate (a list of per-period raw-path lists) rather
    than flattened -- so a caller can split train/calibration *within* each
    period (e.g. first ~40% of each period's own windows for training) rather
    than cutting the pooled list once, which would let whichever period
    happens to sort first/last dominate one side of the split. Windows are
    never built across a period boundary: each period's slice is windowed
    independently, so two disconnected calm stretches can never be spliced
    into one fake window."""
    return [raw_paths_for_horizon(df.loc[p["start"]:p["end"]], cfg)[1] for p in periods]


def raw_paths_for_reference_periods(
    df: pd.DataFrame, cfg: HorizonConfig, periods: list[dict]
) -> list[np.ndarray]:
    """Flattened convenience wrapper around `raw_paths_by_reference_period`,
    for callers that just want one pooled list of reference raw paths (e.g.
    to fit a single model or a single normalization scale across every
    period)."""
    per_period = raw_paths_by_reference_period(df, cfg, periods)
    return [path for period_paths in per_period for path in period_paths]


def fit_reference_scale_multi(df: pd.DataFrame, cfg: HorizonConfig, periods: list[dict]) -> np.ndarray:
    """Multi-period analogue of `fit_reference_scale`: pools increment
    statistics across every reference period (different market eras) before
    fitting the per-channel increment-scale normalization, rather than fitting
    from one contiguous stretch."""
    raw_paths = raw_paths_for_reference_periods(df, cfg, periods)
    if not raw_paths:
        raise RuntimeError(
            f"No {cfg.name!r} windows could be built from the given reference periods "
            f"(need >= {cfg.window} rows in at least one period)."
        )
    return fit_normalization_scale(raw_paths)


def fit_reference_scale(df: pd.DataFrame, cfg: HorizonConfig) -> np.ndarray:
    """Fit the increment-scale normalization from a reference ("normal")
    period slice of `df` (already trimmed to that period by the caller)."""
    _, raw_paths = raw_paths_for_horizon(df, cfg)
    if not raw_paths:
        raise RuntimeError(
            f"Reference period too short to build even one {cfg.name!r} window "
            f"(need >= {cfg.window} rows)."
        )
    return fit_normalization_scale(raw_paths)


def build_feature_frame(
    df: pd.DataFrame,
    cfg: HorizonConfig,
    scale: np.ndarray,
    use_invisibility_reset: bool = False,
) -> pd.DataFrame:
    """Full pipeline: df -> windows -> channels -> normalize -> signature,
    returned as a DataFrame indexed by window end-time with one column per
    signature coordinate."""
    end_times, raw_paths = raw_paths_for_horizon(df, cfg)
    if not raw_paths:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="window_end"))
    feats = build_signature_features(raw_paths, cfg, scale, use_invisibility_reset)
    columns = [f"sig_{i}" for i in range(feats.shape[1])]
    return pd.DataFrame(feats, index=pd.DatetimeIndex(end_times, name="window_end"), columns=columns)
