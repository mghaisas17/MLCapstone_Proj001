"""Deribit's DVOL index -- a real, market-quoted BTC implied-volatility
index (the crypto-options equivalent of VIX), via Deribit's public API.
No API key needed, no options-chain scraping required: DVOL already *is*
the market's own aggregated implied-vol number.

Public endpoint docs: https://docs.deribit.com/#public-get_volatility_index_data
Only a few `resolution` values (in seconds) are actually populated for this
endpoint -- 3600 (hourly) is reliable back to DVOL's launch (~2021-04-01);
finer resolutions (e.g. 60) return empty even though the API accepts the
parameter without error.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
BASE_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
DVOL_LAUNCH = "2021-04-01"  # earliest date with any data, found empirically


def fetch_dvol(currency: str = "BTC", start: str = DVOL_LAUNCH, end: str | None = None,
               resolution: int = 3600, use_cache: bool = True) -> pd.DataFrame:
    """Fetch DVOL (annualized implied vol, in percentage points) as an
    hourly (by default) OHLC-style series, indexed by UTC timestamp.

    Returns columns [open, high, low, close] -- all in the same "20.5 means
    20.5% annualized implied vol" units DVOL itself uses.
    """
    end = end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    cache_path = RAW_DIR / f"dvol_{currency}_{start}_{end}_{resolution}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    # Deribit returns at most ~1000 candles per call; page through in chunks.
    chunk_ms = resolution * 1000 * 900
    frames = []
    cur = start_ms
    while cur < end_ms:
        chunk_end = min(cur + chunk_ms, end_ms)
        resp = requests.get(BASE_URL, params=dict(
            currency=currency, start_timestamp=cur, end_timestamp=chunk_end, resolution=resolution,
        ), timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(f"Deribit API error: {payload['error']}")
        data = payload["result"]["data"]
        if data:
            frames.append(pd.DataFrame(data, columns=["ts_ms", "open", "high", "low", "close"]))
        cur = chunk_end
        time.sleep(0.1)  # polite pacing, no documented rate limit but no key either

    if not frames:
        raise RuntimeError(f"No DVOL data returned for {currency} in [{start}, {end}] at resolution={resolution}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ts_ms")
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()[["open", "high", "low", "close"]]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


if __name__ == "__main__":
    dvol = fetch_dvol()
    print(dvol.shape, dvol.index.min(), dvol.index.max())
    print(dvol.tail())
