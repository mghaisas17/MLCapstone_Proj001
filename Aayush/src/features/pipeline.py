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
