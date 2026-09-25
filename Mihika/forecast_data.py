from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
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
    ret = np.log(bars["close"].iloc[i] / bars["close"].iloc[i - 1])
    leverage = min(ret, 0.0)
    return np.array([*log_rvs, leverage], dtype=np.float32)


def har_log_target(bars: pd.DataFrame, i: int, horizon_steps: int) -> float:
    """log(mean future RV over the horizon + EPS): z_t = log(1/h * sum RV_{t+j} + eps)."""
    future_mean_var = bars["var_inc"].iloc[i + 1 : i + 1 + horizon_steps].mean()
    return float(np.log(max(future_mean_var, 0.0) + EPS))


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


def build_dataset(
    bars: pd.DataFrame,
    spec: ForecastSpec,
    progress_every: int | None = None,
    include_lstm_seq: bool = False,
    sub_window: str = "5D",
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
    seq_list = [] if include_lstm_seq else None

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
        if include_lstm_seq:
            seq_list.append(chunk_signatures(window, sub_window_steps, spec))

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
