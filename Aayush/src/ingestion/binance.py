"""Real tick-level trades and 1-minute klines from Binance's free public
data archive (https://data.binance.vision) -- no API key required.

Yahoo Finance never exposes individual trades; this is our source for
anything finer than a daily bar. To keep data volume tractable for a group
project we pull klines/trades only for the date ranges actually needed
(reference period + intraday/macro construction + targeted event windows),
not a full multi-year tick archive.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "binance"
BASE_URL = "https://data.binance.vision/data/spot"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]
TRADE_COLUMNS = ["trade_id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]


def _end_of_day_inclusive(ts_str: str) -> pd.Timestamp:
    """A bare date string as an `end` argument should mean "through the end
    of that day", not midnight at its start."""
    ts = pd.Timestamp(ts_str, tz="UTC")
    if ts.time() == pd.Timestamp("00:00:00").time() and ":" not in ts_str:
        ts += pd.Timedelta(hours=23, minutes=59, seconds=59, milliseconds=999)
    return ts


def _download_zip_csv(url: str, columns: list[str]) -> pd.DataFrame | None:
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            first_line = f.readline().decode()
        has_header = not first_line.strip().split(",")[0].lstrip("-").isdigit()
        with zf.open(name) as f:
            df = pd.read_csv(f, header=0 if has_header else None, names=None if has_header else columns)
    return df


def _month_range(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    while cur <= end:
        yield cur.year, cur.month
        cur = (cur + pd.offsets.MonthBegin(1))


def _day_range(start: pd.Timestamp, end: pd.Timestamp):
    for d in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        yield d


def fetch_klines(symbol: str, interval: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch 1-minute (or other interval) klines for [start, end] via monthly
    archives, trimmed to the exact range."""
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), _end_of_day_inclusive(end)
    cache_path = RAW_DIR / f"{symbol}_{interval}_klines_{start}_{end}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    frames = []
    for year, month in _month_range(start_ts, end_ts):
        url = f"{BASE_URL}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
        df = _download_zip_csv(url, KLINE_COLUMNS)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No Binance klines found for {symbol} {interval} in [{start}, {end}]")

    klines = pd.concat(frames, ignore_index=True)
    klines["open_time"] = pd.to_datetime(klines["open_time"].astype("int64"), unit="ms", utc=True)
    klines = klines.set_index("open_time").sort_index()
    klines = klines.loc[(klines.index >= start_ts) & (klines.index <= end_ts)]
    for col in ["open", "high", "low", "close", "volume"]:
        klines[col] = klines[col].astype(float)
    klines = klines[["open", "high", "low", "close", "volume"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    klines.to_parquet(cache_path)
    return klines


def fetch_trades(symbol: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch raw trade-by-trade tick data for [start, end] via daily archives.

    Intended for short, targeted windows (a calm reference stretch, or a
    few days around a specific event) -- daily trade files for a busy pair
    like BTCUSDT can be tens of MB each.
    """
    start_ts, end_ts = pd.Timestamp(start, tz="UTC"), _end_of_day_inclusive(end)
    cache_path = RAW_DIR / f"{symbol}_trades_{start}_{end}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    frames = []
    for day in _day_range(start_ts, end_ts):
        url = f"{BASE_URL}/daily/trades/{symbol}/{symbol}-trades-{day:%Y-%m-%d}.zip"
        df = _download_zip_csv(url, TRADE_COLUMNS)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No Binance trades found for {symbol} in [{start}, {end}]")

    trades = pd.concat(frames, ignore_index=True)
    trades["time"] = pd.to_datetime(trades["time"].astype("int64"), unit="ms", utc=True)
    trades = trades.set_index("time").sort_index()
    trades = trades.loc[(trades.index >= start_ts) & (trades.index <= end_ts)]
    trades["price"] = trades["price"].astype(float)
    trades["qty"] = trades["qty"].astype(float)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(cache_path)
    return trades


if __name__ == "__main__":
    k = fetch_klines("BTCUSDT", "1m", "2023-10-01", "2023-10-02")
    print("klines:", k.shape, k.index.min(), k.index.max())
    t = fetch_trades("BTCUSDT", "2023-10-01", "2023-10-01")
    print("trades:", t.shape, t.index.min(), t.index.max())
