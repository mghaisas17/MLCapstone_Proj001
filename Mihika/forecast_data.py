from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import iisignature
import numpy as np
import pandas as pd
import requests
import yfinance as yf

EPS = 1e-12
LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure simple progress logging for notebook/script use."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    LOGGER.setLevel(level)


@dataclass(frozen=True)
class ForecastSpec:
    name: str
    horizon: str
    bar_freq: str
    lookback: str
    har_windows: tuple[str, ...]
    dims: tuple[str, ...]
    depth: int = 3
    lead_lag: bool = True
    time_aug: bool = True
    path_points: int | None = None
    sample_stride: int = 1
    variance_method: str = "close"
    # Intraday-path mode: rows, target, HAR and stats stay on the daily bars; the signature (and its
    # controls) are computed from the intraday bars covering the same lookback instead.
    intraday_freq: str | None = None
    intraday_dims: tuple | None = None  # None -> same as `dims`
    intraday_path_points: int | None = None  # None -> native intraday resolution


SPECS: dict[str, ForecastSpec] = {
    "5m": ForecastSpec(
        name="5m",
        horizon="5min",
        bar_freq="5s",
        lookback="2h",
        har_windows=("5min", "30min", "2h"),
        dims=("price", "activity", "flow"),
        path_points=64,
        sample_stride=12,
        variance_method="close",
    ),
    "1h": ForecastSpec(
        name="1h",
        horizon="1h",
        bar_freq="1min",
        lookback="24h",
        har_windows=("1h", "6h", "24h"),
        dims=("price", "activity", "flow"),
        path_points=96,
        sample_stride=15,
        variance_method="close",
    ),
    "1d": ForecastSpec(
        name="1d",
        horizon="1D",
        bar_freq="1D",
        lookback="30D",
        har_windows=("1D", "7D", "30D"),
        dims=("price", "activity", "range"),
        variance_method="gk",
    ),
    "7d": ForecastSpec(
        name="7d",
        horizon="7D",
        bar_freq="1D",
        lookback="30D",
        har_windows=("1D", "7D", "30D"),
        dims=("price", "activity", "range"),
        variance_method="gk",
    ),
    "30d": ForecastSpec(
        name="30d",
        horizon="30D",
        bar_freq="1D",
        lookback="30D",
        har_windows=("1D", "7D", "30D"),
        dims=("price", "activity", "range"),
        variance_method="gk",
    ),
}


for _key in ("1d", "7d", "30d"):  # same rows/target/baselines as the daily horizons, signature from hourly paths
    SPECS[f"{_key}_i"] = replace(SPECS[_key], name=f"{_key}_i", intraday_freq="1h")


@dataclass
class ForecastDataset:
    times: np.ndarray
    target_end: np.ndarray
    y: np.ndarray
    X_har: np.ndarray
    X_stats: np.ndarray
    X_sig: np.ndarray
    har_names: list[str]
    stat_names: list[str]
    y_har_log: np.ndarray  # log(mean future RV over horizon + EPS): the HAR model's own target
    X_seq: np.ndarray | None = None  # (n, n_chunks, sig_dim) sub-window signature sequence, LSTM-only
    # Non-signature controls, built from the same window as X_sig:
    X_lag: np.ndarray | None = None  # flat lag bank with exactly as many columns as X_sig
    X_multi: np.ndarray | None = None  # hand-built multi-scale realized-volatility features
    # Intraday-path features (only for specs with intraday_freq); same rows as everything above:
    X_isig: np.ndarray | None = None  # signature of the intraday path
    X_ilag: np.ndarray | None = None  # lag bank with exactly as many columns as X_isig
    X_imulti: np.ndarray | None = None  # multi-scale realized-variance features from intraday returns
    X_harrv: np.ndarray | None = None  # HAR-RV-L features from intraday realized variance


STAT_NAMES = [
    "return",
    "realized_vol",
    "price_range",
    "max_drawdown",
    "path_efficiency",
    "log_total_dollar_volume",
    "log_volume_std",
    "activity_first_half",
    "flow_imbalance",
    "total_hl_range",
    "hl_range_std",
]

BINANCE_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quote_qty",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


def _normalize_date(value: str | date | datetime | pd.Timestamp) -> date:
    return pd.Timestamp(value).date()


def _inclusive_dates(start: str | date, end: str | date) -> list[date]:
    start_d = _normalize_date(start)
    end_d = _normalize_date(end)
    if end_d < start_d:
        raise ValueError("end must be on or after start")
    return list(pd.date_range(start_d, end_d, freq="D").date)


def _ensure_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_binance_timestamp(series: pd.Series) -> pd.Series:
    """Handle Binance historical timestamps stored in ms or us."""
    numeric = pd.to_numeric(series, errors="coerce")
    med = numeric.dropna().median()
    if pd.isna(med):
        return pd.to_datetime(series, utc=True, errors="coerce")
    unit = "us" if med >= 1e14 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def prepare_binance_trades(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = _parse_binance_timestamp(df["timestamp"])

    for col in ["price", "qty", "quote_qty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
    df["dollar_volume"] = df["quote_qty"]
    df["trade_sign"] = np.where(df["is_buyer_maker"], -1.0, 1.0)
    df["signed_dollar_flow"] = df["trade_sign"] * df["dollar_volume"]

    return (
        df.dropna(subset=["timestamp", "price", "qty"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _download_one_binance_day(
    symbol: str,
    day: date,
    timeout: int = 60,
) -> pd.DataFrame:
    date_str = day.strftime("%Y-%m-%d")
    url = (
        "https://data.binance.vision/data/spot/daily/trades/"
        f"{symbol}/{symbol}-trades-{date_str}.zip"
    )
    LOGGER.info("Downloading Binance %s %s", symbol, date_str)
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise FileNotFoundError(
            f"Binance file unavailable for {symbol} on {date_str} "
            f"(HTTP {response.status_code})"
        )

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_file = zf.namelist()[0]
        return pd.read_csv(zf.open(csv_file), header=None, names=BINANCE_COLUMNS)


def load_or_download_binance(
    start: str | date,
    end: str | date,
    symbol: str = "BTCUSDT",
    data_dir: str | Path = "data/binance",
    force_download: bool = False,
    skip_unavailable: bool = True,
) -> pd.DataFrame:
    """
    Load Binance daily trade files for an inclusive date range.

    Each day is cached separately as parquet. Existing dates are read from disk;
    only missing dates are downloaded unless force_download=True.
    """
    data_dir = _ensure_data_dir(data_dir)
    dates = _inclusive_dates(start, end)
    frames: list[pd.DataFrame] = []

    LOGGER.info(
        "Binance request: %s | %s to %s | %d day(s)",
        symbol,
        dates[0],
        dates[-1],
        len(dates),
    )

    for k, day in enumerate(dates, start=1):
        day_str = day.strftime("%Y-%m-%d")
        path = data_dir / f"{symbol}_trades_{day_str}.parquet"

        if path.exists() and not force_download:
            LOGGER.info("[%d/%d] Reading cached %s", k, len(dates), path.name)
            raw = pd.read_parquet(path)
        else:
            try:
                raw = _download_one_binance_day(symbol, day)
            except FileNotFoundError as exc:
                if skip_unavailable:
                    LOGGER.warning("[%d/%d] %s", k, len(dates), exc)
                    continue
                raise
            raw.to_parquet(path, index=False)
            LOGGER.info("[%d/%d] Cached %s (%s rows)", k, len(dates), path.name, f"{len(raw):,}")

        frames.append(raw)

    if not frames:
        raise ValueError("No Binance data could be loaded for the requested date range.")

    trades = prepare_binance_trades(pd.concat(frames, ignore_index=True))
    LOGGER.info("Loaded Binance trades: %s rows", f"{len(trades):,}")
    return trades


def binance_to_bars(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    LOGGER.info("Resampling Binance trades to %s bars", freq)
    df = trades.set_index("timestamp")
    bars = pd.DataFrame(
        {
            "open": df["price"].resample(freq).first(),
            "high": df["price"].resample(freq).max(),
            "low": df["price"].resample(freq).min(),
            "close": df["price"].resample(freq).last(),
            "volume": df["qty"].resample(freq).sum(),
            "dollar_volume": df["dollar_volume"].resample(freq).sum(),
            "signed_dollar_flow": df["signed_dollar_flow"].resample(freq).sum(),
        }
    )

    bars["close"] = bars["close"].ffill()
    for col in ["open", "high", "low"]:
        bars[col] = bars[col].fillna(bars["close"])
    for col in ["volume", "dollar_volume", "signed_dollar_flow"]:
        bars[col] = bars[col].fillna(0.0)

    bars = bars.dropna(subset=["close"])
    LOGGER.info("Created %s bars", f"{len(bars):,}")
    return bars


def load_or_download_binance_bars(
    start: str | date,
    end: str | date,
    freq: str,
    symbol: str = "BTCUSDT",
    data_dir: str | Path = "data/binance",
    force_download: bool = False,
    skip_unavailable: bool = True,
    delete_raw_cache: bool = True,
) -> pd.DataFrame:
    """Load Binance trades one day at a time and immediately aggregate to bars.

    This is the memory-safe alternative to ``load_or_download_binance`` for
    forecasting experiments. Only one raw trade day is held in memory and new
    raw trade parquet files are not created. Existing raw caches can be removed
    after resampling because they are reproducible and much larger than bars.
    """
    data_dir = _ensure_data_dir(data_dir)
    dates = _inclusive_dates(start, end)
    daily_bars: list[pd.DataFrame] = []

    for k, day in enumerate(dates, start=1):
        day_str = day.strftime("%Y-%m-%d")
        path = data_dir / f"{symbol}_trades_{day_str}.parquet"
        raw = None
        if path.exists() and not force_download:
            LOGGER.info("[%d/%d] Reading cached %s", k, len(dates), path.name)
            try:
                raw = pd.read_parquet(path)
            except Exception:
                # A disk-full failure can leave a truncated parquet file.
                path.unlink(missing_ok=True)
        if raw is None:
            try:
                raw = _download_one_binance_day(symbol, day)
            except FileNotFoundError as exc:
                if skip_unavailable:
                    LOGGER.warning("[%d/%d] %s", k, len(dates), exc)
                    continue
                raise

        trades = prepare_binance_trades(raw)
        daily_bars.append(binance_to_bars(trades, freq))
        if delete_raw_cache and path.exists():
            path.unlink()
        del raw, trades

    if not daily_bars:
        raise ValueError("No Binance data could be loaded for the requested date range.")

    bars = pd.concat(daily_bars).sort_index()
    bars = bars.loc[~bars.index.duplicated(keep="last")]
    LOGGER.info("Loaded Binance %s bars: %s", freq, f"{len(bars):,}")
    return bars


def _yahoo_cache_path(data_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("-", "_")
    return data_dir / f"{safe}_1d.parquet"


def _clean_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df["dollar_volume"] = df["close"] * df["volume"]
    cols = ["open", "high", "low", "close", "volume", "dollar_volume"]
    return df[cols].dropna().sort_index()


def load_or_download_yahoo(
    start: str | date,
    end: str | date,
    ticker: str = "BTC-USD",
    data_dir: str | Path = "data/yahoo",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load Yahoo daily OHLCV for an inclusive date range with a persistent cache.

    The cache is extended only when the requested range falls outside the dates
    already stored. yfinance's end date is exclusive, so one day is added.
    """
    data_dir = _ensure_data_dir(data_dir)
    cache_path = _yahoo_cache_path(data_dir, ticker)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts < start_ts:
        raise ValueError("end must be on or after start")

    cached = None
    if cache_path.exists() and not force_download:
        LOGGER.info("Reading Yahoo cache %s", cache_path)
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index, utc=True)

    needs_download = force_download or cached is None or cached.empty
    dl_start = start_ts
    dl_end = end_ts

    if cached is not None and not cached.empty and not force_download:
        cached_min, cached_max = cached.index.min(), cached.index.max()
        if start_ts < cached_min or end_ts > cached_max:
            dl_start = min(start_ts, cached_min)
            dl_end = max(end_ts, cached_max)
            needs_download = True
        else:
            needs_download = False

    if needs_download:
        LOGGER.info("Downloading Yahoo %s from %s through %s", ticker, dl_start.date(), dl_end.date())
        raw = yf.download(
            ticker,
            start=dl_start.date().isoformat(),
            end=(dl_end.date() + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            raise ValueError(f"Yahoo returned no data for {ticker}.")
        full = _clean_yahoo(raw)
        full.to_parquet(cache_path)
        LOGGER.info("Updated Yahoo cache %s (%s rows)", cache_path.name, f"{len(full):,}")
    else:
        full = cached.sort_index()
        LOGGER.info("Requested Yahoo range already present in cache")

    out = full.loc[(full.index >= start_ts) & (full.index <= end_ts)].copy()
    if out.empty:
        raise ValueError("No Yahoo rows found inside the requested date range.")
    LOGGER.info("Loaded Yahoo rows: %s", f"{len(out):,}")
    return out


def load_or_download_data(
    source: str,
    start: str | date,
    end: str | date,
    *,
    symbol: str = "BTCUSDT",
    ticker: str = "BTC-USD",
    data_root: str | Path = "data",
    force_download: bool = False,
) -> pd.DataFrame:
    """Unified wrapper for raw Yahoo or Binance data."""
    source = source.lower()
    data_root = Path(data_root)
    if source == "binance":
        return load_or_download_binance(
            start=start,
            end=end,
            symbol=symbol,
            data_dir=data_root / "binance",
            force_download=force_download,
        )
    if source == "yahoo":
        return load_or_download_yahoo(
            start=start,
            end=end,
            ticker=ticker,
            data_dir=data_root / "yahoo",
            force_download=force_download,
        )
    raise ValueError("source must be 'binance' or 'yahoo'")


def validate_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Drop bars with a non-positive open/high/low/close. A $0 (or negative)
    price is invalid market data, not a modeling edge case -- left in place
    it turns into inf the moment any log-ratio touches it (Garman-Klass's
    log(high/low), close-to-close log-returns, the HAR leverage term).
    Logs the dropped timestamps so the source data can be inspected.
    """
    cols = [c for c in ("open", "high", "low", "close") if c in bars.columns]
    bad = (bars[cols] <= 0).any(axis=1)
    if bad.any():
        LOGGER.warning(
            "Dropping %d bar(s) with non-positive OHLC (invalid market data), e.g. at %s",
            int(bad.sum()), list(bars.index[bad][:5]),
        )
        bars = bars.loc[~bad]
    return bars


def add_variance_increment(bars: pd.DataFrame, method: str = "close") -> pd.DataFrame:
    bars = validate_ohlc(bars).copy()
    if method == "close":
        r = np.log(bars["close"]).diff()
        # The very first bar has no prior price, so diff() leaves it NaN --
        # left unfixed, any window whose largest HAR span equals the full
        # lookback (true for the 5m/1h specs) includes that NaN in its very
        # first sample, which then poisons log_RV/leverage with NaN/inf.
        bars["var_inc"] = (r**2).fillna(0.0)
    elif method == "gk":
        log_hl = np.log(bars["high"] / bars["low"])
        log_co = np.log(bars["close"] / bars["open"])
        # Garman-Klass is a per-bar *estimator*, not a variance itself, and can
        # be negative on any single bar; only the aggregate (sum/mean over a
        # window) should be floored at zero, so no floor is applied here.
        bars["var_inc"] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    else:
        raise ValueError("method must be 'close' or 'gk'")
    return bars


def resample_path(path: np.ndarray, n_points: int | None) -> np.ndarray:
    if n_points is None or len(path) == n_points:
        return path
    old_t = np.linspace(0, 1, len(path))
    new_t = np.linspace(0, 1, n_points)
    return np.column_stack(
        [np.interp(new_t, old_t, path[:, j]) for j in range(path.shape[1])]
    )


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    n, d = path.shape
    out = np.zeros((2 * n - 1, 2 * d), dtype=float)
    out[0] = np.r_[path[0], path[0]]
    k = 1
    for i in range(1, n):
        out[k] = np.r_[path[i], path[i - 1]]
        out[k + 1] = np.r_[path[i], path[i]]
        k += 2
    return out


def time_augment(path: np.ndarray) -> np.ndarray:
    t = np.linspace(0, 1, len(path))
    return np.column_stack([t, path])


def _from_start(x: pd.Series) -> np.ndarray:
    """A level series relative to its first value; gaps are carried forward (back at the start)."""
    v = x.ffill().bfill().to_numpy(float)
    return np.zeros(len(v)) if np.isnan(v).all() else v - v[0]


def make_base_path(window: pd.DataFrame, dims: Iterable[str]) -> np.ndarray:
    dims = tuple(dims)
    coordinates: list[np.ndarray] = []

    if "price" in dims:
        log_price = np.log(window["close"].to_numpy())
        coordinates.append(log_price - log_price[0])

    if "activity" in dims:
        dv = window["dollar_volume"].to_numpy()
        coordinates.append(np.cumsum(dv) / (dv.sum() + EPS))

    if "flow" in dims:
        if "signed_dollar_flow" not in window.columns:
            raise ValueError("'flow' dimension requires signed_dollar_flow (use Binance data).")
        flow = window["signed_dollar_flow"].to_numpy()
        dv = window["dollar_volume"].to_numpy()
        coordinates.append(np.cumsum(flow) / (np.sum(np.abs(dv)) + EPS))

    if "range" in dims:
        hl = np.log(window["high"].to_numpy() / window["low"].to_numpy())
        coordinates.append(np.cumsum(hl))

    # FX channels from fx_hourly_channels (time-of-day adjusted, past data only)
    if "price_sa" in dims:  # log price rebuilt from adjusted returns (the first bar's return is before the window)
        r = window["ret_sa"].to_numpy(float)
        coordinates.append(np.concatenate([[0.0], np.cumsum(r[1:])]))

    if "range_sa" in dims:
        coordinates.append(np.cumsum(window["range_sa"].to_numpy(float)))

    if "spread" in dims:  # running mean of the abnormal log spread: rises while spreads are unusually wide
        sp = window["spread_sa"].to_numpy(float)
        coordinates.append(np.cumsum(sp) / len(sp))

    # Other FX pairs over the same hours (fx_cross_channels): "x:GBPUSD" is GBP/USD's log price from
    # the window start; "usd" is the dollar factor (average log move of the dollar against the pairs).
    for d in dims:
        if d.startswith("x:"):
            col = f"lclose__{d[2:]}"
            if col not in window.columns:
                raise ValueError(f"'{d}' needs a {col} column (set cross_pairs; see fx_cross_channels).")
            coordinates.append(_from_start(window[col]))
    if "usd" in dims:
        if "usd_log" not in window.columns:
            raise ValueError("'usd' needs a usd_log column (set cross_pairs; see fx_cross_channels).")
        coordinates.append(_from_start(window["usd_log"]))

    if "stress" in dims:
        if "stress_prob" not in window.columns:
            raise ValueError("'stress' dimension requires a stress_prob column (see anomaly_flag.py).")
        coordinates.append(window["stress_prob"].to_numpy(float))

    if not coordinates:
        raise ValueError("No valid path dimensions were requested.")
    return np.column_stack(coordinates)


def make_path(window: pd.DataFrame, spec: ForecastSpec) -> np.ndarray:
    path = make_base_path(window, spec.dims)
    path = resample_path(path, spec.path_points)
    if spec.lead_lag:
        path = lead_lag_transform(path)
    if spec.time_aug:
        path = time_augment(path)
    return path.astype(np.float64)


def signature_features(path: np.ndarray, depth: int) -> np.ndarray:
    return iisignature.sig(path, depth).astype(np.float32)


def signature_info(path: np.ndarray, depth: int) -> dict[str, int]:
    d = path.shape[1]
    return {
        "path_dimension": d,
        "depth": depth,
        "signature_features": int(iisignature.siglength(d, depth)),
    }


def lag_bank_features(window: pd.DataFrame, spec: ForecastSpec, n_target: int) -> np.ndarray:
    """Dimension-matched, non-signature control for the signature features.

    Uses exactly the information the signature sees (increments of the same base
    channels at the same resolution) but as a flat bank of lagged terms instead of
    iterated integrals, truncated to ``n_target`` columns (= the signature length)
    in priority order: raw increments, squared increments, contemporaneous
    cross-channel products, lag-k products (k = 1, 2, ...), absolute increments.
    If Ridge on this does as well as Ridge on signatures, the gain is not
    specific to signatures.
    """
    path = resample_path(make_base_path(window, spec.dims), spec.path_points)
    inc = np.diff(path, axis=0)
    m, d = inc.shape
    parts = [inc.ravel(), (inc**2).ravel()]
    if d > 1:
        parts.append(np.concatenate([inc[:, i] * inc[:, j] for i in range(d) for j in range(i + 1, d)]))
    lag = 1
    while sum(p.size for p in parts) < n_target and lag < m:
        parts.append((inc[lag:] * inc[:-lag]).ravel())
        lag += 1
    parts.append(np.abs(inc).ravel())
    v = np.concatenate(parts)
    if v.size < n_target:
        v = np.pad(v, (0, n_target - v.size))
    return v[:n_target].astype(np.float32)


def multiscale_features(window: pd.DataFrame) -> np.ndarray:
    """Hand-built multi-scale volatility features: at 5 geometrically spaced
    look-back scales, realized variance, mean absolute return, mean log range,
    volume, signed return, downside semivariance, largest move and (Binance)
    signed flow. A strong, signature-free competitor with a few dozen columns."""
    r = np.diff(np.log(window["close"].to_numpy(float)))
    hl = np.log(window["high"].to_numpy(float) / window["low"].to_numpy(float))[1:]
    dv = window["dollar_volume"].to_numpy(float)[1:]
    flow = window["signed_dollar_flow"].to_numpy(float)[1:] if "signed_dollar_flow" in window.columns else None
    scales = sorted({max(2, int(round(s))) for s in np.geomspace(2, len(r), 5)})
    out = []
    for s in scales:
        rr = r[-s:]
        out += [
            np.log(np.mean(rr**2) + EPS), np.log(np.mean(np.abs(rr)) + EPS),
            np.log(np.mean(hl[-s:]) + EPS), np.log1p(np.mean(dv[-s:])), rr.sum(),
            np.log(np.mean(np.minimum(rr, 0.0) ** 2) + EPS), np.log(np.abs(rr).max() + EPS),
        ]
        if flow is not None:
            out.append(flow[-s:].sum() / (dv[-s:].sum() + EPS))
    return np.asarray(out, dtype=np.float32)


def summary_features(window: pd.DataFrame) -> np.ndarray:
    price = window["close"].to_numpy()
    log_price = np.log(price)
    returns = np.diff(log_price)
    total_return = log_price[-1] - log_price[0]
    rv = np.sqrt(np.sum(returns**2))
    price_range = log_price.max() - log_price.min()
    running_max = np.maximum.accumulate(log_price)
    max_drawdown = np.min(log_price - running_max)
    total_movement = np.sum(np.abs(returns)) + EPS
    efficiency = abs(total_return) / total_movement

    dv = window["dollar_volume"].to_numpy()
    total_dv = np.sum(dv)
    log_volume_std = np.std(np.log1p(dv))
    half = max(1, len(dv) // 2)
    first_half_activity = np.sum(dv[:half]) / (total_dv + EPS)

    if "signed_dollar_flow" in window.columns:
        flow_imbalance = window["signed_dollar_flow"].sum() / (total_dv + EPS)
    else:
        flow_imbalance = 0.0

    hl = np.log(window["high"].to_numpy() / window["low"].to_numpy())

    return np.array(
        [
            total_return,
            rv,
            price_range,
            max_drawdown,
            efficiency,
            np.log1p(total_dv),
            log_volume_std,
            first_half_activity,
            flow_imbalance,
            hl.sum(),
            hl.std(),
        ],
        dtype=np.float32,
    )


def future_volatility_target(bars: pd.DataFrame, i: int, horizon_steps: int) -> float:
    """sqrt(sum of var_inc over the horizon) -- the common y every model is scored
    against, regardless of what scale/transform it was fit in."""
    future_var = bars["var_inc"].iloc[i + 1 : i + 1 + horizon_steps].sum()
    return float(np.sqrt(max(future_var, 0)))


def har_log_features(bars: pd.DataFrame, i: int, har_steps: list[int]) -> np.ndarray:
    """
    Crypto-style HAR-RV-L features at index i: log(RV) averaged over each
    HAR window (RV(1), RV(7), RV(30), ...), plus a leverage/asymmetry term
    (Corsi & Reno's HAR-RV-L) using the sign of the most recent daily return.
    """
    var_inc = bars["var_inc"].to_numpy()
    log_rvs = [
        np.log(max(var_inc[i - w + 1 : i + 1].mean(), 0.0) + EPS) for w in har_steps
    ]
    # return over the shortest HAR window (one day), which is one bar for daily bars
    ret = np.log(bars["close"].iloc[i] / bars["close"].iloc[i - har_steps[0]])
    leverage = min(ret, 0.0)
    return np.array([*log_rvs, leverage], dtype=np.float32)


def har_log_target(bars: pd.DataFrame, i: int, horizon_steps: int) -> float:
    """log(mean future RV over the horizon + EPS): z_t = log(1/h * sum RV_{t+j} + eps)."""
    future_mean_var = bars["var_inc"].iloc[i + 1 : i + 1 + horizon_steps].mean()
    return float(np.log(max(future_mean_var, 0.0) + EPS))


def future_drawdown_targets(
    intraday: pd.DataFrame, known_times, horizon: str, freq: str = "1h", min_coverage: float = 0.9,
    ends=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Max drawdown of the log price over (known, known + horizon], from intraday closes, and that
    drawdown divided by the realized volatility of the same window.

    The path starts at the close of the last bar completed at the known time, so a fall straight
    from that price counts. Bars are labelled by their open time. A row whose window is missing
    more than ``1 - min_coverage`` of its bars gets NaN for both. ``ends`` (default: known +
    horizon) closes each window where its target does, which for markets that close at weekends
    (FX) is ``horizon`` trading days later; ``horizon`` then sets how many bars to expect."""
    labels = _ns(intraday.index)
    known = _ns(known_times)
    step = np.timedelta64(int(pd.Timedelta(freq).value), "ns")
    span = np.timedelta64(int(pd.Timedelta(horizon).value), "ns")
    j0 = np.searchsorted(labels, known, side="left")  # first bar after the known time
    stop = known + span if ends is None else _ns(ends)
    j1 = np.searchsorted(labels, stop, side="left")  # first bar after the window
    expected = int(pd.Timedelta(horizon) / pd.Timedelta(freq))
    log_close = np.log(intraday["close"].to_numpy(float))
    dd, ratio = np.full(len(known), np.nan), np.full(len(known), np.nan)
    for k in range(len(known)):
        a, b = int(j0[k]), int(j1[k])
        if a < 1 or labels[a - 1] != known[k] - step or b - a < min_coverage * expected:
            continue
        path = log_close[a - 1 : b]
        dd[k] = float(np.max(np.maximum.accumulate(path) - path))
        rv = float(np.sqrt(np.sum(np.diff(path) ** 2)))
        ratio[k] = dd[k] / rv if rv > 0 else np.nan
    return dd, ratio


def duration_steps(duration: str, bar_freq: str) -> int:
    duration_td = pd.Timedelta(duration)
    bar_td = pd.Timedelta(bar_freq)
    steps = int(duration_td / bar_td)
    if steps < 1:
        raise ValueError(f"{duration_td} smaller than bar size {bar_td}")
    return steps


def chunk_signatures(window: pd.DataFrame, sub_window_steps: int, spec: ForecastSpec) -> np.ndarray:
    """
    Split an *already-sliced* lookback window (the same one build_dataset just
    used for the full-window signature) into consecutive sub-window chunks
    and compute one signature per chunk -- this is what the LSTM consumes.
    Reusing the window slice avoids re-walking `bars` a second time.
    """
    n_chunks = max(1, len(window) // sub_window_steps)
    chunks = [window.iloc[k * sub_window_steps : (k + 1) * sub_window_steps] for k in range(n_chunks)]
    return np.stack(
        [signature_features(make_path(chunk, spec), spec.depth) for chunk in chunks if len(chunk) >= 2]
    )


def _ns(index) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return np.asarray(idx.tz_localize(None), dtype="datetime64[ns]")


class _IntradayWindows:
    """Intraday bars matched to the daily rows. A row labelled day D is known at the end of D, so its
    intraday window is the ``lookback`` of bars ending with the last bar of D and nothing later."""

    def __init__(self, intra: pd.DataFrame, ends: np.ndarray, spec: ForecastSpec):
        self.intra = intra
        self.ends = ends  # position just past the last intraday bar of each kept row
        self.lookback = duration_steps(spec.lookback, spec.intraday_freq)
        self.har_steps = [duration_steps(w, spec.intraday_freq) for w in spec.har_windows]
        self.var_frame = intra

    def window(self, k: int) -> tuple[pd.DataFrame, int]:
        j1 = int(self.ends[k])
        return self.intra.iloc[j1 - self.lookback : j1], j1


def _intraday_alignment(bars, intraday, spec, candidates):
    """Keep only rows with a complete intraday window and no look-ahead; return them with the windows."""
    intra = add_variance_increment(intraday, method="close")
    freq = pd.Timedelta(spec.intraday_freq)
    cand = np.asarray(candidates)
    known = _ns(bars.index[cand] + pd.Timedelta(spec.bar_freq))
    labels = _ns(intra.index)
    j1 = np.searchsorted(labels, known, side="left")  # first bar not yet known
    need = max(
        duration_steps(spec.lookback, spec.intraday_freq),
        max(duration_steps(w, spec.intraday_freq) for w in spec.har_windows),
    )
    ok = (j1 >= need) & (j1 <= len(labels))
    last = np.where(ok, labels[np.clip(j1 - 1, 0, len(labels) - 1)], np.datetime64("NaT"))
    ok &= last == (known - np.timedelta64(int(freq.value), "ns"))  # the window ends exactly at the row's known time
    if not ok.any():
        raise ValueError("No daily rows have a complete intraday window; check the intraday date range.")
    return cand[ok], _IntradayWindows(intra, j1[ok], spec)


def build_dataset(
    bars: pd.DataFrame,
    spec: ForecastSpec,
    progress_every: int | None = None,
    include_lstm_seq: bool = False,
    sub_window: str = "5D",
    intraday: pd.DataFrame | None = None,
) -> ForecastDataset:
    """
    Single pass over `bars` that builds every feature representation at once
    (HAR, summary stats, full-window signature, and -- only if requested --
    the LSTM's per-sub-window signature sequence). Building the LSTM sequence
    here, from the *same* sliced `window`, avoids a second walk over `bars`
    and avoids ever holding two independently-built copies of the dataset in
    memory; `include_lstm_seq=False` (the default) skips that extra work
    entirely for experiments that don't need it.
    """
    LOGGER.info("Building forecast dataset for horizon=%s", spec.name)
    bars = add_variance_increment(bars, method=spec.variance_method)

    horizon_steps = duration_steps(spec.horizon, spec.bar_freq)
    lookback_steps = duration_steps(spec.lookback, spec.bar_freq)
    har_steps = [duration_steps(w, spec.bar_freq) for w in spec.har_windows]
    sub_window_steps = duration_steps(sub_window, spec.bar_freq) if include_lstm_seq else None
    min_history = max(lookback_steps, max(har_steps))

    candidate_indices = range(
        min_history - 1,
        len(bars) - horizon_steps,
        spec.sample_stride,
    )
    intra = None
    if spec.intraday_freq:
        if intraday is None:
            raise ValueError(f"Spec {spec.name} uses intraday paths; pass the intraday bars.")
        candidate_indices, intra = _intraday_alignment(bars, intraday, spec, candidate_indices)
    total = len(candidate_indices)
    if total <= 0:
        raise ValueError(
            "Not enough bars for the requested lookback/horizon. "
            f"Need more than {min_history + horizon_steps} bars; got {len(bars)}."
        )

    if progress_every is None:
        progress_every = max(1, total // 20)

    times, target_ends = [], []
    y_list, har_list, har_log_y_list, stat_list, sig_list = [], [], [], [], []
    lag_list, multi_list = [], []
    seq_list = [] if include_lstm_seq else None
    isig_list, ilag_list, imulti_list, harrv_list = [], [], [], []

    for count, i in enumerate(candidate_indices, start=1):
        if count == 1 or count % progress_every == 0 or count == total:
            LOGGER.info("Feature generation progress: %d/%d (%.1f%%)", count, total, 100 * count / total)

        window = bars.iloc[i - lookback_steps + 1 : i + 1]
        path = make_path(window, spec)
        sig = signature_features(path, spec.depth)

        times.append(bars.index[i])
        target_ends.append(bars.index[i + horizon_steps])
        y_list.append(future_volatility_target(bars, i, horizon_steps))
        har_list.append(har_log_features(bars, i, har_steps))
        har_log_y_list.append(har_log_target(bars, i, horizon_steps))
        stat_list.append(summary_features(window))
        sig_list.append(sig)
        lag_list.append(lag_bank_features(window, spec, sig.shape[0]))
        multi_list.append(multiscale_features(window))
        if include_lstm_seq:
            seq_list.append(chunk_signatures(window, sub_window_steps, spec))
        if intra is not None:
            iw, j1 = intra.window(count - 1)
            ispec = replace(
                spec, dims=tuple(spec.intraday_dims or spec.dims), path_points=spec.intraday_path_points
            )
            isig = signature_features(make_path(iw, ispec), spec.depth)
            isig_list.append(isig)
            ilag_list.append(lag_bank_features(iw, ispec, isig.shape[0]))
            imulti_list.append(multiscale_features(iw))
            harrv_list.append(har_log_features(intra.var_frame, j1 - 1, intra.har_steps))

    dataset = ForecastDataset(
        times=np.asarray(times),
        target_end=np.asarray(target_ends),
        y=np.asarray(y_list, dtype=np.float32),
        X_har=np.asarray(har_list, dtype=np.float32),
        X_stats=np.asarray(stat_list, dtype=np.float32),
        X_sig=np.asarray(sig_list, dtype=np.float32),
        har_names=[f"log_RV_{w}" for w in spec.har_windows] + ["leverage"],
        stat_names=STAT_NAMES.copy(),
        y_har_log=np.asarray(har_log_y_list, dtype=np.float32),
        X_seq=np.stack(seq_list).astype(np.float32) if include_lstm_seq else None,
        X_lag=np.asarray(lag_list, dtype=np.float32),
        X_multi=np.asarray(multi_list, dtype=np.float32),
        X_isig=np.asarray(isig_list, dtype=np.float32) if intra is not None else None,
        X_ilag=np.asarray(ilag_list, dtype=np.float32) if intra is not None else None,
        X_imulti=np.asarray(imulti_list, dtype=np.float32) if intra is not None else None,
        X_harrv=np.asarray(harrv_list, dtype=np.float32) if intra is not None else None,
    )
    LOGGER.info(
        "Finished dataset: n=%s | HAR=%s | stats=%s | signature=%s | seq=%s",
        f"{len(dataset.y):,}",
        dataset.X_har.shape[1],
        dataset.X_stats.shape[1],
        dataset.X_sig.shape[1],
        dataset.X_seq.shape if include_lstm_seq else "skipped",
    )
    return dataset


def generate_forecast_dataset(
    source: str,
    spec_name: str,
    start: str | date,
    end: str | date,
    *,
    symbol: str = "BTCUSDT",
    ticker: str = "BTC-USD",
    data_root: str | Path = "data",
    force_download: bool = False,
    save_dataset_path: str | Path | None = None,
) -> ForecastDataset:
    """
    High-level wrapper: load/download raw data -> bars -> features -> target dataset.
    """
    if spec_name not in SPECS:
        raise KeyError(f"Unknown spec '{spec_name}'. Choose from {list(SPECS)}")
    spec = SPECS[spec_name]
    source = source.lower()

    if source == "binance" and spec_name not in {"5m", "1h"}:
        LOGGER.warning("Using Binance with a daily-horizon spec; this is allowed but may be expensive.")
    if source == "yahoo" and spec_name in {"5m", "1h"}:
        raise ValueError("Yahoo daily data cannot generate 5m/1h forecast datasets. Use Binance.")

    raw = load_or_download_data(
        source=source,
        start=start,
        end=end,
        symbol=symbol,
        ticker=ticker,
        data_root=data_root,
        force_download=force_download,
    )

    if source == "binance":
        bars = binance_to_bars(raw, spec.bar_freq)
    else:
        bars = raw.copy()

    dataset = build_dataset(bars, spec)

    if save_dataset_path is not None:
        save_forecast_dataset(dataset, save_dataset_path)

    return dataset


def chronological_split(
    dataset: ForecastDataset,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    n = len(dataset.y)
    cut = int(n * (1 - test_fraction))
    if cut <= 0 or cut >= n:
        raise ValueError("Dataset is too small for the requested split.")

    test_start_time = dataset.times[cut]
    train_idx = np.where(dataset.target_end < test_start_time)[0]
    test_idx = np.arange(cut, n)
    return train_idx, test_idx


def save_forecast_dataset(dataset: ForecastDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=dataset.times,
        target_end=dataset.target_end,
        y=dataset.y,
        X_har=dataset.X_har,
        X_stats=dataset.X_stats,
        X_sig=dataset.X_sig,
        har_names=np.asarray(dataset.har_names, dtype=object),
        stat_names=np.asarray(dataset.stat_names, dtype=object),
        y_har_log=dataset.y_har_log,
    )
    LOGGER.info("Saved forecast dataset to %s", path)


def load_forecast_dataset(path: str | Path) -> ForecastDataset:
    data = np.load(path, allow_pickle=True)
    return ForecastDataset(
        times=data["times"],
        target_end=data["target_end"],
        y=data["y"],
        X_har=data["X_har"],
        X_stats=data["X_stats"],
        X_sig=data["X_sig"],
        har_names=data["har_names"].tolist(),
        stat_names=data["stat_names"].tolist(),
        y_har_log=data["y_har_log"],
    )


# --------------------------------------------------------------------------
# Memory-safe Binance loader
#
# A single BTCUSDT day is ~1M+ trades, so a few months of raw ticks does not
# fit in memory (this is what kept crashing the notebook kernel). Instead each
# day's zip is streamed to disk, read in chunks, and reduced straight to bar
# files for every requested frequency. Raw trades are never concatenated or
# written out; only the small per-day bar files are kept, and a day that is
# already cached is never downloaded again.
# --------------------------------------------------------------------------

_BAR_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "dollar_volume": "sum",
    "signed_dollar_flow": "sum",
}


def _chunk_to_partial_bars(chunk: pd.DataFrame, freq: str) -> pd.DataFrame:
    ts = _parse_binance_timestamp(chunk["timestamp"])
    price = pd.to_numeric(chunk["price"], errors="coerce")
    qty = pd.to_numeric(chunk["qty"], errors="coerce")
    dollars = pd.to_numeric(chunk["quote_qty"], errors="coerce")
    sign = np.where(chunk["is_buyer_maker"].astype(bool), -1.0, 1.0)
    df = pd.DataFrame(
        {"price": price, "qty": qty, "dollar_volume": dollars, "flow": sign * dollars}
    )
    df.index = ts
    df = df.loc[df.index.notna() & df["price"].notna() & df["qty"].notna()]
    g = df.groupby(df.index.floor(freq))
    return pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["qty"].sum(),
            "dollar_volume": g["dollar_volume"].sum(),
            "signed_dollar_flow": g["flow"].sum(),
        }
    )


def _binance_day_to_bars(
    symbol: str, day: date, freqs: Iterable[str], chunksize: int, timeout: int = 120
) -> dict[str, pd.DataFrame]:
    """Download one day's trade zip to a temp file and reduce it to bars."""
    import tempfile

    date_str = day.strftime("%Y-%m-%d")
    url = (
        "https://data.binance.vision/data/spot/daily/trades/"
        f"{symbol}/{symbol}-trades-{date_str}.zip"
    )
    freqs = list(freqs)
    partials: dict[str, list[pd.DataFrame]] = {f: [] for f in freqs}
    with tempfile.TemporaryFile() as tmp:
        with requests.get(url, timeout=timeout, stream=True) as response:
            if response.status_code != 200:
                raise FileNotFoundError(
                    f"Binance file unavailable for {symbol} on {date_str} "
                    f"(HTTP {response.status_code})"
                )
            for block in response.iter_content(chunk_size=1 << 20):
                tmp.write(block)
        tmp.seek(0)
        with zipfile.ZipFile(tmp) as zf:
            with zf.open(zf.namelist()[0]) as handle:
                reader = pd.read_csv(
                    handle,
                    header=None,
                    names=BINANCE_COLUMNS,
                    usecols=["price", "qty", "quote_qty", "timestamp", "is_buyer_maker"],
                    chunksize=chunksize,
                )
                for chunk in reader:
                    for f in freqs:
                        partials[f].append(_chunk_to_partial_bars(chunk, f))
    out = {}
    for f in freqs:
        # chunks are time-ordered, so first/last across chunk boundaries are correct
        out[f] = pd.concat(partials[f]).groupby(level=0).agg(_BAR_AGG)
    return out


def _fill_bar_gaps(bars: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Regular grid; empty buckets carry the last close with zero volume."""
    bars = bars.sort_index()
    bars = bars.loc[~bars.index.duplicated(keep="last")]
    bars = bars.reindex(pd.date_range(bars.index[0], bars.index[-1], freq=freq))
    bars["close"] = bars["close"].ffill()
    for col in ("open", "high", "low"):
        bars[col] = bars[col].fillna(bars["close"])
    for col in ("volume", "dollar_volume", "signed_dollar_flow"):
        bars[col] = bars[col].fillna(0.0)
    bars.index.name = "timestamp"
    return bars.dropna(subset=["close"])


def fetch_binance_bars(
    start: str | date,
    end: str | date,
    freqs: Iterable[str] = ("5s", "1min"),
    symbol: str = "BTCUSDT",
    data_dir: str | Path = "data/bars",
    force_download: bool = False,
    skip_unavailable: bool = True,
    chunksize: int = 500_000,
) -> dict[str, pd.DataFrame]:
    """Bars for an inclusive date range at each frequency, built one day at a
    time with bounded memory. Returns {freq: DataFrame}."""
    data_dir = _ensure_data_dir(data_dir)
    freqs = list(freqs)
    dates = _inclusive_dates(start, end)
    per_freq: dict[str, list[pd.DataFrame]] = {f: [] for f in freqs}

    for k, day in enumerate(dates, start=1):
        day_str = day.strftime("%Y-%m-%d")
        paths = {f: data_dir / f"{symbol}_{f}_{day_str}.parquet" for f in freqs}
        if not force_download and all(p.exists() for p in paths.values()):
            try:
                cached = {f: pd.read_parquet(p) for f, p in paths.items()}
            except Exception:
                # a truncated cache file (e.g. after a disk-full error): rebuild
                for p in paths.values():
                    p.unlink(missing_ok=True)
            else:
                for f, b in cached.items():
                    per_freq[f].append(b)
                continue
        try:
            LOGGER.info("[%d/%d] Downloading %s %s", k, len(dates), symbol, day_str)
            day_bars = _binance_day_to_bars(symbol, day, freqs, chunksize)
        except FileNotFoundError as exc:
            if skip_unavailable:
                LOGGER.warning("[%d/%d] %s", k, len(dates), exc)
                continue
            raise
        for f, b in day_bars.items():
            b.to_parquet(paths[f])
            per_freq[f].append(b)

    if not per_freq[freqs[0]]:
        raise ValueError("No Binance data could be loaded for the requested date range.")
    return {f: _fill_bar_gaps(pd.concat(parts), f) for f, parts in per_freq.items()}


# --------------------------------------------------------------------------
# Hourly bars from Binance 1-minute klines (long history for intraday paths)
#
# Klines are one row per minute (~0.7M rows a year), so they reach back to 2017-08
# at a fraction of the size of the trade files. Each carries the traded quote volume
# and the taker-buy quote volume, which gives the same dollar volume and signed flow
# columns as the trade-based bars (flow = taker buy - taker sell).
# --------------------------------------------------------------------------

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore",
]
KLINES_START = pd.Timestamp("2017-08-01")


def _klines_to_hourly(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()].copy()  # tolerate a header row
    ts = _parse_binance_timestamp(df["open_time"])
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in
           ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote")}
    d = pd.DataFrame(num)
    d["signed"] = 2.0 * d["taker_buy_quote"] - d["quote_volume"]
    d.index = ts
    d = d.loc[d.index.notna()].sort_index()
    g = d.groupby(d.index.floor(freq))
    return pd.DataFrame(
        {
            "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
            "close": g["close"].last(), "volume": g["volume"].sum(),
            "dollar_volume": g["quote_volume"].sum(), "signed_dollar_flow": g["signed"].sum(),
        }
    )


def _download_klines(url: str, timeout: int = 120, attempts: int = 5) -> pd.DataFrame:
    """Download and read one klines zip. Transient network failures (timeouts, dropped connections,
    truncated archives) are retried with exponential backoff (2, 4, 8, 16 s); a missing file
    (HTTP 404) is not retried and raises FileNotFoundError."""
    import tempfile
    import time

    for attempt in range(1, attempts + 1):
        try:
            with tempfile.TemporaryFile() as tmp:
                with requests.get(url, timeout=timeout, stream=True) as response:
                    if response.status_code == 404:
                        raise FileNotFoundError(f"Binance klines unavailable: {url} (HTTP 404)")
                    if response.status_code >= 400:  # e.g. 5xx from the archive: transient, retried
                        raise requests.exceptions.HTTPError(f"HTTP {response.status_code} for {url}")
                    for block in response.iter_content(chunk_size=1 << 20):
                        tmp.write(block)
                tmp.seek(0)
                with zipfile.ZipFile(tmp) as zf:
                    with zf.open(zf.namelist()[0]) as handle:
                        return pd.read_csv(handle, header=None, names=KLINE_COLUMNS, usecols=range(11), dtype=str)
        except FileNotFoundError:
            raise
        except (requests.exceptions.RequestException, zipfile.BadZipFile, OSError) as exc:
            if attempt == attempts:
                raise
            wait = 2**attempt
            LOGGER.warning("Download failed (%s), retrying in %ds [%d/%d]: %s", type(exc).__name__, wait,
                           attempt, attempts - 1, url)
            time.sleep(wait)


def fetch_binance_klines_hourly(
    start: str | date,
    end: str | date,
    symbol: str = "BTCUSDT",
    data_dir: str | Path = "data/klines",
    force_download: bool = False,
    freq: str = "1h",
) -> pd.DataFrame:
    """Hourly OHLC + dollar volume + signed flow for an inclusive date range, built a month at a
    time (daily files for months without a monthly archive, e.g. the current one) and cached."""
    base = "https://data.binance.vision/data/spot"
    data_dir = _ensure_data_dir(data_dir)
    start_ts = max(pd.Timestamp(start), KLINES_START)
    end_ts = pd.Timestamp(end)
    pieces: list[pd.DataFrame] = []

    def cached(path, build):
        if path.exists() and not force_download:
            try:
                return pd.read_parquet(path)
            except Exception:
                path.unlink(missing_ok=True)
        out = build()
        out.to_parquet(path)
        return out

    for month in pd.period_range(start_ts, end_ts, freq="M"):
        tag = month.strftime("%Y-%m")
        try:
            LOGGER.info("Klines %s %s (monthly)", symbol, tag)
            pieces.append(cached(
                data_dir / f"{symbol}_{freq}_{tag}.parquet",
                lambda: _klines_to_hourly(
                    _download_klines(f"{base}/monthly/klines/{symbol}/1m/{symbol}-1m-{tag}.zip"), freq),
            ))
            continue
        except FileNotFoundError:
            pass  # not published yet (current month): fall back to daily files
        for day in pd.date_range(max(month.start_time, start_ts), min(month.end_time.normalize(), end_ts)):
            dtag = day.strftime("%Y-%m-%d")
            try:
                pieces.append(cached(
                    data_dir / f"{symbol}_{freq}_{dtag}.parquet",
                    lambda: _klines_to_hourly(
                        _download_klines(f"{base}/daily/klines/{symbol}/1m/{symbol}-1m-{dtag}.zip"), freq),
                ))
            except FileNotFoundError as exc:
                LOGGER.warning("%s", exc)
    if not pieces:
        raise ValueError("No Binance klines could be loaded for the requested date range.")
    bars = _fill_bar_gaps(pd.concat(pieces), freq)
    bars = bars.loc[(bars.index >= start_ts.tz_localize("UTC")) & (bars.index < (end_ts + pd.Timedelta(days=1)).tz_localize("UTC"))]
    return bars


# --------------------------------------------------------------------------
# FX (EUR/USD and other Dukascopy pairs): hourly and daily bars for the new pipeline
# --------------------------------------------------------------------------

# FX trades ~5 days a week and every window is counted in bars, so a week is 5 daily bars and a
# month 22 (the classic Corsi HAR 1/5/22), instead of crypto's 7/30. Daily and hourly bars both
# come from Dukascopy, so the hourly path ends exactly where its daily row does.
FX_SPECS: dict[str, ForecastSpec] = {
    key: replace(SPECS["1d"], name=f"fx_{key}", horizon=h, lookback="22D", har_windows=("1D", "5D", "22D"))
    for key, h in {"1d": "1D", "5d": "5D", "22d": "22D"}.items()
}
for _key in ("1d", "5d", "22d"):
    FX_SPECS[f"{_key}_i"] = replace(FX_SPECS[_key], name=f"fx_{_key}_i", intraday_freq="1h")

# The FX day runs 22:00 to 22:00 UTC (the New York 5pm close), so Sunday evening's trading
# belongs to Monday and Friday ends at the weekly close.
FX_DAY_START = pd.Timedelta(hours=-2)
DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed"
# Hourly candle record: seconds from the start of the month, open, close, low, high (integer
# points), volume -- big-endian, like the tick files decoded in forecast_data_fx_local.py.
# One file per month at {symbol}/{year}/{month-1:02d}/{BID|ASK}_candles_hour_1.bi5 with a record for
# every hour of the month (flat, zero-volume ones when closed). Checked against the live server for
# EURUSD Jan 2024: 744 records, OHLC and volume equal to bars built from the hourly tick files.
DUKASCOPY_CANDLE = np.dtype([("sec", ">u4"), ("open", ">u4"), ("close", ">u4"), ("low", ">u4"), ("high", ">u4"),
                             ("vol", ">f4")])
DUKASCOPY_TICK = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_vol", ">f4"), ("bid_vol", ">f4")])


def _dukascopy_point(symbol: str) -> float:
    return 1e3 if "JPY" in symbol.upper() else 1e5


_DUKASCOPY = {"session": None, "last": 0.0}
DUKASCOPY_PAUSE = 0.5  # seconds between requests: Dukascopy rate-limits (HTTP 429) rapid bursts
_sleep = __import__("time").sleep  # module-level so tests can skip the waits


def _dukascopy_get(url: str, timeout: int = 60, attempts: int = 8) -> bytes | None:
    """One Dukascopy file, or None when it does not exist (404 or empty: weekends, the current
    month). Requests reuse one connection and are spaced DUKASCOPY_PAUSE apart. When the server
    rate-limits (429) or is busy (503), wait as long as its Retry-After asks, else 15 s, 30 s, ...
    up to 10 min; other failures are retried with backoff (2, 4, 8, ... s). Finished months are
    cached, so a re-run resumes where it stopped."""
    import time

    if _DUKASCOPY["session"] is None:
        _DUKASCOPY["session"] = requests.Session()
    for attempt in range(1, attempts + 1):
        _sleep(max(0.0, DUKASCOPY_PAUSE - (time.monotonic() - _DUKASCOPY["last"])))
        _DUKASCOPY["last"] = time.monotonic()
        try:
            response = _DUKASCOPY["session"].get(url, timeout=timeout)
            if response.status_code == 404:
                return None
            if response.status_code in (429, 503):
                if attempt == attempts:
                    raise RuntimeError(
                        f"Dukascopy is still rate-limiting (HTTP {response.status_code}) after {attempts} tries "
                        f"for {url}. Wait 10-15 minutes and re-run: finished months are cached and are not "
                        "downloaded again.")
                after = response.headers.get("Retry-After", "")
                wait = float(after) if after.replace(".", "", 1).isdigit() else min(600.0, 15.0 * 2 ** (attempt - 1))
                LOGGER.warning("Dukascopy rate limit (HTTP %d), waiting %.0fs [%d/%d]: %s",
                               response.status_code, wait, attempt, attempts - 1, url)
                _sleep(wait)
                continue
            if response.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code} for {url}")
            return response.content or None
        except requests.exceptions.RequestException as exc:
            if attempt == attempts:
                raise
            wait = 2**attempt
            LOGGER.warning("Dukascopy download failed (%s), retrying in %ds: %s", type(exc).__name__, wait, url)
            _sleep(wait)
    return None


def decode_dukascopy_candles(content: bytes, start: pd.Timestamp, point: float) -> pd.DataFrame:
    """Decode one LZMA-compressed Dukascopy candle file (records timed from ``start``)."""
    import lzma

    rec = np.frombuffer(lzma.decompress(content), dtype=DUKASCOPY_CANDLE)
    return pd.DataFrame(
        {"open": rec["open"] / point, "high": rec["high"] / point, "low": rec["low"] / point,
         "close": rec["close"] / point, "volume": rec["vol"].astype(float)},
        index=pd.DatetimeIndex(start + pd.to_timedelta(rec["sec"].astype(np.int64), unit="s"), name="timestamp"),
    )


def _candles_look_valid(c: pd.DataFrame) -> bool:
    """Guard against a misread file: prices positive, high >= open/close >= low on (nearly) every bar."""
    if c.empty:
        return False
    tol = 1e-9
    ok = (c["low"] > 0) & (c["high"] + tol >= c[["open", "close"]].max(axis=1)) & \
        (c["low"] - tol <= c[["open", "close"]].min(axis=1))
    return bool(ok.mean() > 0.99)


def _mid_hourly_from_candles(symbol: str, month: pd.Period, spread: bool = False) -> pd.DataFrame | None:
    """Mid-price hourly bars for a month from Dukascopy's BID and ASK hourly candle files, or None.
    With ``spread``, also the hour's relative bid-ask spread: (ask - bid) / mid, averaged over the
    hour's open and close quotes."""
    start = pd.Timestamp(month.start_time, tz="UTC")
    point = _dukascopy_point(symbol)
    sides = {}
    for side in ("BID", "ASK"):
        content = _dukascopy_get(f"{DUKASCOPY_URL}/{symbol}/{start.year}/{start.month - 1:02d}/{side}_candles_hour_1.bi5")
        if content is None:
            return None
        sides[side] = decode_dukascopy_candles(content, start, point)
        if not _candles_look_valid(sides[side]):
            LOGGER.warning("Dukascopy %s %s %s candles failed the sanity check; using ticks", symbol, month, side)
            return None
    bid, ask = sides["BID"].align(sides["ASK"], join="inner")
    mid = (bid[["open", "high", "low", "close"]] + ask[["open", "high", "low", "close"]]) / 2
    mid["volume"] = bid["volume"] + ask["volume"]
    if spread:
        mid["spread"] = 0.5 * ((ask["open"] - bid["open"]) / mid["open"] + (ask["close"] - bid["close"]) / mid["close"])
    return mid[mid["volume"] > 0]  # flat zero-volume hours are the closed market


def _mid_hourly_from_ticks(symbol: str, day: pd.Timestamp, spread: bool = False) -> pd.DataFrame:
    """Fallback: one UTC day of mid-price hourly bars from Dukascopy's hourly tick files
    (the decoder from forecast_data_fx_local.py). With ``spread``, also the mean relative spread."""
    import lzma

    point = _dukascopy_point(symbol)
    rows = []
    for hour in pd.date_range(day, periods=24, freq="h"):
        content = _dukascopy_get(
            f"{DUKASCOPY_URL}/{symbol}/{hour.year}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5")
        if content is None:
            continue
        rec = np.frombuffer(lzma.decompress(content), dtype=DUKASCOPY_TICK)
        if not len(rec):
            continue
        mid = (rec["bid"] + rec["ask"]) / (2 * point)
        row = {"timestamp": hour, "open": mid[0], "high": mid.max(), "low": mid.min(), "close": mid[-1],
               "volume": float(rec["bid_vol"].sum() + rec["ask_vol"].sum())}
        if spread:
            row["spread"] = float(np.mean((rec["ask"].astype(float) - rec["bid"]) / (point * mid)))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"] + (["spread"] if spread else []),
                            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"), dtype=float)
    return pd.DataFrame(rows).set_index("timestamp")


def _fx_bar_columns(bars: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the pipeline expects. FX has no trade direction, so signed flow uses the
    tick rule at the bar level: the sign of the bar's move times its (quote) volume."""
    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    bars["dollar_volume"] = bars["volume"] * bars["close"]
    bars["signed_dollar_flow"] = np.sign(bars["close"] - bars["open"]) * bars["dollar_volume"]
    return bars


def fetch_dukascopy_hourly(
    start: str | date,
    end: str | date,
    symbol: str = "EURUSD",
    data_dir: str | Path = "data/dukascopy",
    force_download: bool = False,
    spread: bool = False,
) -> pd.DataFrame:
    """Hourly mid-price OHLC + quote volume for an FX pair, cached per month. Only trading hours
    are kept (no weekend bars), so windows counted in bars are windows of trading time.

    ``spread=True`` adds the relative bid-ask spread of each hour (column ``spread``). Those months
    are cached separately (``*_1h_ba_*``), so the mid-only cache other runs use is left untouched;
    the mid prices are the same in both.

    Each month comes from Dukascopy's hourly candle files (2 downloads). If those are missing (the
    current month) or fail the sanity check, the month is built from the hourly tick files instead
    (24 downloads per day), the source already used in forecast_data_fx_local.py."""
    data_dir = _ensure_data_dir(data_dir)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    this_month = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    pieces = []
    for month in pd.period_range(start_ts, end_ts, freq="M"):
        path = data_dir / f"{symbol}_1h_{'ba_' if spread else ''}{month}.parquet"
        if path.exists() and not force_download:
            pieces.append(pd.read_parquet(path))
            continue
        LOGGER.info("Dukascopy %s %s (hourly)", symbol, month)
        bars = _mid_hourly_from_candles(symbol, month, spread)
        if bars is None:
            days = pd.date_range(pd.Timestamp(month.start_time, tz="UTC"),
                                 min(pd.Timestamp(month.end_time, tz="UTC").normalize(),
                                     pd.Timestamp(end_ts, tz="UTC")), freq="D")
            parts = [_mid_hourly_from_ticks(symbol, d, spread) for d in days if d.weekday() != 5]
            parts = [p for p in parts if len(p)]  # empty days (holidays) would turn every column to object
            bars = pd.concat(parts).astype(float) if parts else None
        if bars is None or bars.empty:
            LOGGER.warning("No Dukascopy data for %s %s", symbol, month)
            continue
        bars = _fx_bar_columns(bars)
        if month < this_month:  # a finished month never changes; the current one is refetched
            bars.to_parquet(path)
        pieces.append(bars)
    if not pieces:
        raise ValueError(f"No Dukascopy data could be loaded for {symbol} {start}..{end}.")
    bars = pd.concat(pieces).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    lo, hi = pd.Timestamp(start_ts, tz="UTC"), pd.Timestamp(end_ts, tz="UTC") + pd.Timedelta(days=1)
    return validate_ohlc(bars.loc[(bars.index >= lo) & (bars.index < hi)])


FX_SESSION_TZ = "America/New_York"  # FX sessions are fixed in New York time, so DST does not move them


def fx_hourly_channels(hourly: pd.DataFrame, weeks: int = 52, min_weeks: int = 8) -> pd.DataFrame:
    """Add the time-of-day-adjusted FX channels to hourly bars, using only past data.

    FX volatility and spreads follow the clock (Asia, London, New York, the 17:00 New York
    rollover), so each hour is compared with the same hour of the week (in New York time) over the
    previous ``weeks`` weeks. Each estimate uses only earlier bars, so the columns can be computed
    once over the whole history with no look-ahead. Until a slot has ``min_weeks`` earlier
    observations there is no adjustment (factor 1).

    * ``ret_sa``: log return / (that slot's mean |return| / the mean |return| of all slots). The
      level of volatility is kept; only its intraday shape is removed.
    * ``range_sa``: log(high / low), adjusted the same way.
    * ``spread_sa``: log(spread) minus that slot's mean log spread: 0 = a normal spread for that
      hour, > 0 = wider than usual (clipped to +-3). Needs a ``spread`` column.
    """
    h = hourly.copy()
    idx = pd.DatetimeIndex(h.index)
    local = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(FX_SESSION_TZ)
    slot = pd.Series(local.dayofweek * 24 + local.hour, index=h.index)
    per_week = max(1, int(round(len(h) / max(1.0, (idx[-1] - idx[0]) / pd.Timedelta(weeks=1)))))

    def past_slot_mean(x: pd.Series) -> pd.Series:
        return x.groupby(slot).transform(lambda v: v.shift(1).rolling(weeks, min_periods=min_weeks).mean())

    def factor(x: pd.Series) -> pd.Series:
        overall = x.shift(1).rolling(weeks * per_week, min_periods=min_weeks * per_week).mean()
        f = past_slot_mean(x) / overall
        return f.where(np.isfinite(f) & (f > 0)).fillna(1.0).clip(0.2, 5.0)

    r = np.log(h["close"]).diff().fillna(0.0)
    hl = np.log(h["high"] / h["low"]).clip(lower=0.0)
    h["ret_sa"] = r / factor(r.abs())
    h["range_sa"] = hl / factor(hl)
    if "spread" in h.columns:
        ls = np.log(h["spread"].clip(lower=1e-7))
        h["spread_sa"] = (ls - past_slot_mean(ls)).fillna(0.0).clip(-3.0, 3.0)
    return h


def usd_side(symbol: str) -> int:
    """+1 if the dollar is the base currency (USDJPY: a rise = a stronger dollar), -1 if it is the
    quote (EURUSD: a rise = a weaker dollar), 0 if the pair has no dollar."""
    s = symbol.upper()
    return 1 if s[:3] == "USD" else -1 if s[3:6] == "USD" else 0


def fx_cross_channels(hourly: pd.DataFrame, symbol: str, others: dict, max_gap_hours: int = 3) -> pd.DataFrame:
    """Add other pairs' prices over the same hours, for paths with several pairs.

    * ``lclose__<PAIR>``: the other pair's log close at each of this pair's hourly bars. A missing
      hour takes the last earlier close (at most ``max_gap_hours`` back), so nothing uses later data.
    * ``usd_log``: the dollar factor, the cumulative mean of the hourly log moves of the dollar
      against every pair available that hour (this one included), signed so that up = stronger dollar.
    """
    h = hourly.copy()
    moves = {symbol: np.log(h["close"]).diff()}
    for sym, bars in others.items():
        lc = np.log(bars["close"]).sort_index()
        lc = lc[~lc.index.duplicated(keep="last")].reindex(h.index).ffill(limit=max_gap_hours)
        h[f"lclose__{sym}"] = lc
        moves[sym] = lc.diff()
    signed = [usd_side(s) * m for s, m in moves.items() if usd_side(s)]
    if signed:
        h["usd_log"] = pd.concat(signed, axis=1).mean(axis=1).fillna(0.0).cumsum()
    return h


def fx_daily_from_hourly(hourly: pd.DataFrame, min_hours: int = 20) -> pd.DataFrame:
    """Daily FX bars from hourly ones, one per FX day (22:00 to 22:00 UTC). Each bar is labelled
    by its START time, so ``label + 1 day`` is when it is complete -- the convention every
    dataset here uses -- and the hourly path for that row ends with the day's last hourly bar.
    Days with fewer than ``min_hours`` hourly bars (holidays, the Sunday stub) are dropped."""
    day = (hourly.index - FX_DAY_START).floor("D")  # 22:00 UTC belongs to the next date
    g = hourly.groupby(day)
    daily = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                  volume=("volume", "sum"), dollar_volume=("dollar_volume", "sum"),
                  signed_dollar_flow=("signed_dollar_flow", "sum"))
    daily = daily[g.size() >= min_hours]
    daily.index = pd.DatetimeIndex(daily.index + FX_DAY_START, name="timestamp")
    return validate_ohlc(daily)
