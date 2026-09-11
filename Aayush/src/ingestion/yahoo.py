"""Daily BTC-USD OHLCV from Yahoo Finance -- the multi-year macro backbone.

Yahoo has no trade-level data at any lookback; it only ever stores
pre-built bars. Daily bars go back to ~2014 for BTC-USD, which is enough
to span every labeled event in configs/horizons.yaml.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_daily(ticker: str = "BTC-USD", start: str = "2014-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch full-history daily OHLCV and cache it to data/raw/.

    Returns a DataFrame indexed by UTC date with columns
    [open, high, low, close, volume].
    """
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_DIR / f"{ticker.replace('-', '_')}_daily.parquet")
    return df


def load_daily(ticker: str = "BTC-USD") -> pd.DataFrame:
    """Load previously cached daily data, fetching it if not present."""
    path = RAW_DIR / f"{ticker.replace('-', '_')}_daily.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return fetch_daily(ticker)


if __name__ == "__main__":
    data = fetch_daily()
    print(data.shape, data.index.min(), data.index.max())
    print(data.tail())
