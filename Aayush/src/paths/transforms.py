"""Turn a raw window (bars or ticks) into a numeric multichannel path, and
the standard signature-literature path transforms: time-augmentation,
basepoint addition, invisibility-reset, lead-lag, and normalization.

Design choice: price channels are always expressed as *cumulative log
return from the window's own start* (i.e. log(p_t) - log(p_0)), not raw
log price. BTC ranged from ~$200 to ~$120,000 over 2014-2026, so an
un-centered log-price channel would make windows from different eras
wildly different in scale; zeroing each window removes that and makes the
signature comparable across the whole history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def extract_channels(window_data: pd.DataFrame, channels: list[str], source: str) -> np.ndarray:
    """Build the (L, d) raw channel matrix for one window.

    `source` is one of {"binance_ticks", "binance_klines_1m", "yahoo_daily"}
    and determines which raw columns back each requested channel name.
    """
    L = len(window_data)
    cols = {}

    if "time" in channels:
        t = (window_data.index - window_data.index[0]).total_seconds().to_numpy(dtype=float)
        span = t[-1] if t[-1] > 0 else 1.0
        cols["time"] = t / span

    if source == "binance_ticks":
        price = window_data["price"].to_numpy(dtype=float)
        if "log_price" in channels:
            cols["log_price"] = np.log(price) - np.log(price[0])
        if "signed_volume" in channels:
            sign = np.where(window_data["is_buyer_maker"].to_numpy(dtype=bool), -1.0, 1.0)
            cols["signed_volume"] = np.cumsum(sign * window_data["qty"].to_numpy(dtype=float))
    else:  # bar data: binance_klines_1m (pre-resampled) or yahoo_daily
        close = window_data["close"].to_numpy(dtype=float)
        if "log_return" in channels:
            log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
            cols["log_return"] = np.cumsum(log_ret)
        if "volume" in channels:
            cols["volume"] = window_data["volume"].to_numpy(dtype=float)
        if "log_range" in channels:
            # Intraday/interday dispersion (Parkinson-style realized-range
            # proxy) that close-to-close log_return misses entirely -- a bar
            # that spikes and reverts looks identical to a flat bar in
            # log_return alone. Raw per-bar value, not cumulative, matching
            # how `volume` above is already treated (not every channel needs
            # the cumulative-from-start zeroing that price-level channels
            # need -- only channels representing an unbounded level do).
            high = window_data["high"].to_numpy(dtype=float)
            low = window_data["low"].to_numpy(dtype=float)
            cols["log_range"] = np.log(high) - np.log(low)
        if "overnight_gap" in channels:
            # log(open / previous bar's close): jump/gap risk between bars,
            # distinct from the log_return channel's own close-to-close move.
            # The first bar in a window has no preceding close *within the
            # window*, so its gap is set to 0 -- "no prior information
            # available yet" -- mirroring log_return's own `prepend=log(close[0])`
            # convention for the same edge case.
            opens = window_data["open"].to_numpy(dtype=float)
            prev_close = np.concatenate(([close[0]], close[:-1]))
            gap = np.log(opens) - np.log(prev_close)
            gap[0] = 0.0
            cols["overnight_gap"] = gap

    missing = [c for c in channels if c not in cols]
    if missing:
        raise ValueError(f"Unsupported channel(s) {missing} for source={source!r}")

    return np.column_stack([cols[c] for c in channels]).astype(float)


def add_basepoint(path: np.ndarray) -> np.ndarray:
    """Prepend a zero row so the signature can see the path's own starting
    level (otherwise a pure translation of the path is invisible to the
    signature, since the signature only depends on increments)."""
    zero = np.zeros((1, path.shape[1]))
    return np.vstack([zero, path])


def invisibility_reset(path: np.ndarray) -> np.ndarray:
    """Chevyrev-Kormilitzin's invisibility-reset transform: append an
    "activity" channel that is 1 while the path is live, then append one
    final step where the activity channel drops to 0 and every other
    channel resets to 0. Lets a one-class model trained on fixed-length
    windows generalise to shorter/killed paths without the truncation
    point itself leaking information through the ordinary channels.
    For our fixed-length windows this mainly adds a clean, path-independent
    terminal segment that anchors the signature's highest-order terms.
    """
    L, d = path.shape
    active = np.ones((L, 1))
    live = np.hstack([path, active])
    reset_row = np.zeros((1, d + 1))  # activity -> 0, all channels -> 0
    return np.vstack([live, reset_row])


def lead_lag(series: np.ndarray) -> np.ndarray:
    """Lead-lag transform of a single 1D series into a 2D path (lag, lead),
    doubling the number of points. Standard trick for letting a signature
    see quadratic-variation-like information from a scalar series."""
    lag = np.repeat(series, 2)[:-1]
    lead = np.repeat(series, 2)[1:]
    return np.column_stack([lag, lead])


def apply_recency_weights(path: np.ndarray, halflife: float) -> np.ndarray:
    """Reweight a path's increments by recency (exponential decay, given
    `halflife` in steps) before the signature is computed: the increment
    closest to the window's end counts at ~full weight, and increments
    further back count for exponentially less. Rebuilt by cumulative-summing
    the reweighted increments back onto the path's own starting level, so
    everything downstream (`add_basepoint`, `signature`) is unaffected -- it
    just receives a differently-shaped path.

    Apply this *after* `normalize` (so the increment-scale fit, which assumes
    unweighted increments, isn't itself biased toward the window's tail) and
    *before* `add_basepoint`. Apply identically to both live-scored windows
    and reference/calibration paths used to fit a model -- weighting only one
    side would bias every distance/score computed between them."""
    L = path.shape[0]
    if L < 2:
        return path
    increments = np.diff(path, axis=0)
    steps_from_end = (L - 2) - np.arange(L - 1)  # 0 = most recent increment, larger = older
    weights = np.exp(-np.log(2) / halflife * steps_from_end)
    weighted_increments = increments * weights[:, None]
    return np.vstack([path[:1], path[:1] + np.cumsum(weighted_increments, axis=0)])


def normalize(path: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-channel rescaling by an increment-scale fit on a reference
    ("normal") period only -- never fit per-window, or a test window's own
    volatility would leak into its own features. Channels are divided, not
    centered: every channel here already starts at (or near) zero by
    construction (cumulative log-return, cumulative signed volume, a
    [0,1]-normalized time axis), so centering would shift that zero point
    and break the basepoint convention `add_basepoint` relies on."""
    scale_safe = np.where(scale < 1e-12, 1.0, scale)
    return path / scale_safe


def fit_normalization_scale(paths: list[np.ndarray]) -> np.ndarray:
    """Fit per-channel increment scale (std of increments) across a set of
    reference paths (typically every window in configs/horizons.yaml's
    reference_period). Signature terms are built from iterated integrals of
    increments, so it's the increment scale -- not the raw level -- that
    should be normalized to make windows from different volatility eras
    (e.g. BTC in 2015 vs. 2024) comparable."""
    all_increments = np.vstack([np.diff(p, axis=0) for p in paths if len(p) > 1])
    return all_increments.std(axis=0)
