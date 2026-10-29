"""Rows, windows and the four window representations.

A *row* is a time at which we score the latest window. Its window is the last W bars
that had completed by then; nothing later is used. Everything is vectorized over rows.

Representations of a window (all from the same bars):
  signature  truncated signature of the (time, lead-lag) path of the chosen channels
  lag bank   the same increments as flat lagged terms, with exactly as many columns as the
             signature (control: is it the signature, or just the information in the window?)
  stats      hand-built summaries: realized variance at several scales, skew, kurtosis,
             downside share, range, volume concentration, flow imbalance
  raw        block sums of the increments (input for the autoencoder / other detectors)

``normalize="trailing_vol"`` divides returns by the volatility of the W bars *before* the
window (and ranges by their mean range), so a quiet and a wild window with the same shape
look the same: detectors then see shape, not volatility level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_data import duration_steps

EPS = 1e-12


# ---------------------------------------------------------------- rows, targets, HAR inputs


@dataclass
class Rows:
    pos: np.ndarray  # position of each row's last bar
    times: pd.DatetimeIndex  # time the row is known (end of its last bar)
    target_end: pd.DatetimeIndex  # end of the forecast horizon
    y: np.ndarray  # log realized volatility over the next horizon
    har_X: np.ndarray  # log realized variance per bar over each HAR look-back
    horizon_steps: int


def bar_returns(bars: pd.DataFrame) -> np.ndarray:
    return np.diff(np.log(bars["close"].to_numpy(float)), prepend=np.log(float(bars["close"].iloc[0])))


def build_rows(bars: pd.DataFrame, spec, max_window: int, target_rows: int | None) -> Rows:
    r = bar_returns(bars)
    csum = np.concatenate([[0.0], np.cumsum(r**2)])
    h = duration_steps(spec.horizon, spec.bar_freq)
    every = duration_steps(spec.sample_every, spec.bar_freq)
    har = [duration_steps(w, spec.bar_freq) for w in spec.har_windows]
    warm = max(2 * max_window, max(har))  # 2x window: the trailing normalization needs W earlier bars
    pos = np.arange(warm - 1, len(bars) - h, every)
    if target_rows and len(pos) > target_rows:
        pos = pos[:: int(np.ceil(len(pos) / target_rows))]
    y = 0.5 * np.log(csum[pos + 1 + h] - csum[pos + 1] + EPS)
    har_X = np.column_stack([np.log((csum[pos + 1] - csum[pos + 1 - w]) / w + EPS) for w in har])
    bar = pd.Timedelta(spec.bar_freq)
    times = pd.DatetimeIndex(bars.index[pos]) + bar
    return Rows(pos, times, times + pd.Timedelta(spec.horizon), y, har_X, h)


# ---------------------------------------------------------------- windows


def extract_windows(bars: pd.DataFrame, pos: np.ndarray, W: int, normalize: str) -> dict:
    r = bar_returns(bars)
    hl = np.log(bars["high"].to_numpy(float) / bars["low"].to_numpy(float) + EPS)
    dv = bars["dollar_volume"].to_numpy(float)
    flow = bars["signed_dollar_flow"].to_numpy(float) if "signed_dollar_flow" in bars else np.zeros(len(bars))
    idx = pos[:, None] + np.arange(-W + 1, 1)
    win = {"r": r[idx], "hl": hl[idx], "dv": dv[idx], "flow": flow[idx]}
    if normalize == "trailing_vol":
        prev = pos[:, None] + np.arange(-2 * W + 1, -W + 1)
        sigma = np.sqrt(np.mean(r[prev] ** 2, axis=1, keepdims=True))
        sigma = np.maximum(sigma, 1e-6)
        win["r"] = np.clip(win["r"] / sigma, -50, 50)
        win["hl"] = np.clip(win["hl"] / np.maximum(hl[prev].mean(axis=1, keepdims=True), 1e-6), 0, 50)
    elif normalize != "none":
        raise ValueError(f"unknown normalize={normalize!r}")
    return win


def _share(x: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return x / np.where(denom > 0, denom, 1.0)


def base_paths(win: dict, channels: tuple, path_points: int) -> np.ndarray:
    """(n, L, d) cumulative paths starting at 0, subsampled to <= path_points points."""
    n, W = win["r"].shape
    zero = np.zeros((n, 1))
    total_dv = win["dv"].sum(axis=1, keepdims=True)
    cols = []
    for c in channels:
        if c == "price":
            inc = win["r"]
        elif c == "activity":
            inc = np.where(total_dv > 0, _share(win["dv"], total_dv), 1.0 / W)
        elif c == "range":
            inc = win["hl"]
        elif c == "flow":
            inc = _share(win["flow"], total_dv)
        else:
            raise ValueError(f"unknown channel {c!r}")
        cols.append(np.concatenate([zero, np.cumsum(inc, axis=1)], axis=1))
    path = np.stack(cols, axis=2)
    if path.shape[1] > path_points:
        keep = np.unique(np.round(np.linspace(0, path.shape[1] - 1, path_points)).astype(int))
        path = path[:, keep]
    return path


def augment(path: np.ndarray, lead_lag: bool) -> np.ndarray:
    if lead_lag:
        L = path.shape[1]
        rep = np.repeat(np.arange(L), 2)
        path = np.concatenate([path[:, rep[1:]], path[:, rep[:-1]]], axis=2)
    t = np.broadcast_to(np.linspace(0.0, 1.0, path.shape[1])[None, :, None], (path.shape[0], path.shape[1], 1))
    return np.concatenate([t, path], axis=2)


def sig_length(dim: int, depth: int) -> int:
    return sum(dim**k for k in range(1, depth + 1))


def level_slice(dim: int, depth: int) -> slice:
    return slice(0, sig_length(dim, depth))


def signature(path: np.ndarray, depth: int, batch: int = 2000) -> np.ndarray:
    import iisignature

    out = [np.asarray(iisignature.sig(path[i : i + batch], depth), dtype=np.float32)
           for i in range(0, len(path), batch)]
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------- controls


def lag_bank(base: np.ndarray, n_target: int) -> np.ndarray:
    """Flat lagged terms of the path increments, truncated/padded to n_target columns, in the
    same priority order as forecast_data.lag_bank_features: raw, squared, cross-channel,
    lag-k products (k = 1, 2, ...), absolute."""
    inc = np.diff(base, axis=1)
    n, m, d = inc.shape
    parts = [inc.reshape(n, -1), (inc**2).reshape(n, -1)]
    if d > 1:
        parts.append(np.concatenate([inc[:, :, i] * inc[:, :, j] for i in range(d) for j in range(i + 1, d)], axis=1))
    lag = 1
    while sum(p.shape[1] for p in parts) < n_target and lag < m:
        parts.append((inc[:, lag:] * inc[:, :-lag]).reshape(n, -1))
        lag += 1
    parts.append(np.abs(inc).reshape(n, -1))
    X = np.concatenate(parts, axis=1)
    if X.shape[1] < n_target:
        X = np.pad(X, ((0, 0), (0, n_target - X.shape[1])))
    return X[:, :n_target].astype(np.float32)


def stats(win: dict) -> np.ndarray:
    r, hl, dv, flow = win["r"], win["hl"], win["dv"], win["flow"]
    n, W = r.shape
    r2 = r**2
    var = r2.mean(axis=1) + EPS
    sd = np.sqrt(var)
    total_dv = dv.sum(axis=1)
    share = np.where(total_dv[:, None] > 0, _share(dv, total_dv[:, None]), 1.0 / W)
    feats = [
        np.log(var), np.log(np.abs(r).mean(axis=1) + EPS), r.sum(axis=1), np.log(np.abs(r).max(axis=1) + EPS),
        (r2 * (r < 0)).sum(axis=1) / (r2.sum(axis=1) + EPS),
        ((r - r.mean(axis=1, keepdims=True)) ** 3).mean(axis=1) / sd**3,
        ((r - r.mean(axis=1, keepdims=True)) ** 4).mean(axis=1) / sd**4,
    ]
    for frac in (8, 4, 2):
        k = max(2, W // frac)
        feats.append(np.log(r2[:, -k:].mean(axis=1) + EPS))
    feats += [
        np.log(hl.mean(axis=1) + EPS), share.std(axis=1) * W, share[:, -max(1, W // 4):].sum(axis=1),
        _share(flow.sum(axis=1), total_dv),
    ]
    return np.column_stack(feats).astype(np.float32)


def raw(win: dict, channels: tuple, blocks: int = 32) -> np.ndarray:
    """Block sums of each channel's increments (scale-free shares for volume and flow)."""
    n, W = win["r"].shape
    k = min(blocks, W)
    edges = np.linspace(0, W, k + 1).astype(int)
    total_dv = win["dv"].sum(axis=1, keepdims=True)
    series = {
        "price": win["r"], "range": win["hl"],
        "activity": np.where(total_dv > 0, _share(win["dv"], total_dv), 1.0 / W),
        "flow": _share(win["flow"], total_dv),
    }
    cols = [np.add.reduceat(series[c], edges[:-1], axis=1) for c in channels]
    return np.concatenate(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------- all representations for one RepConfig


def representations(win: dict, rep, spec, depths, max_features: int) -> dict:
    """{name: matrix} for one path recipe: signatures at each allowed depth, the lag bank
    matched to the deepest signature computed, stats, and raw."""
    from .config import CHANNEL_SETS

    channels = CHANNEL_SETS[rep.channels]
    base = base_paths(win, channels, spec.path_points)
    path = augment(base, rep.lead_lag)
    dim = path.shape[2]
    allowed = [d for d in sorted(depths) if sig_length(dim, d) <= max_features]
    out = {}
    if allowed:
        full = signature(path, allowed[-1])
        for d in allowed:
            out[f"signature d{d}"] = full[:, level_slice(dim, d)]
        out["lag bank"] = lag_bank(base, out[f"signature d{allowed[-1]}"].shape[1])
    out["stats"] = stats(win)
    out["raw"] = raw(win, channels)
    return out
