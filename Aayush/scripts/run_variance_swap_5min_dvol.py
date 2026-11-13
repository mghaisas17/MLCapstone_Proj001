"""5-min BTC variance swap using DVOL as a (forward-filled) REAL strike --
the direct answer to "why do we need a proxy strike after the DVOL pipeline
exists": DVOL only has native hourly resolution (resolution=60 returns empty,
resolution=300 is rejected by Deribit's API), so a true 5-min-native real
strike doesn't exist. This forward-fills each hour's DVOL close across that
hour's 5-min bars (build_variance_series_dvol_ffill) as the closest
approximation, restricted to DVOL's actual coverage window (2021-04-01+).

Also runs the SAME signal/window through the plain proxy strike
(build_variance_series) for a clean, apples-to-apples strike-type comparison
-- isolating what switching from "benchmark against our own trailing
average" to "benchmark against the market's own priced-in forecast" actually
buys, with everything else (signal, window, costs) held fixed.
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd
from scipy import stats

from src.strategy.variance_swap import (
    VarSwapConfig, backtest, build_variance_series, build_variance_series_dvol_ffill,
    realistic_entry_cost,
)

OHLCV_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_five_min_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/five_min_anomaly_score.csv"
DVOL_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_dvol_hourly.csv"
PERIODS_PER_YEAR = 365 * 24 * 12


def ann_sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std() * np.sqrt(PERIODS_PER_YEAR)) if x.std() > 1e-16 else 0.0


def run(df: pd.DataFrame, weights: np.ndarray, cost: float, n_years: float) -> dict:
    bt = backtest(df, weights, VarSwapConfig(entry_cost=cost))
    pnl = bt["pnl"].to_numpy()
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    ann_pnl = cum[-1] / n_years
    calmar = ann_pnl / abs(dd.min()) if dd.min() != 0 else np.nan
    _, p = stats.ttest_1samp(pnl, 0.0)
    return dict(sharpe=ann_sharpe(pnl), calmar=calmar, p=p,
                ann_pnl_10m=ann_pnl * 10_000_000, max_dd_10m=dd.min() * 10_000_000)


dvol_df = build_variance_series_dvol_ffill(OHLCV_CSV, ANOMALY_CSV, DVOL_CSV, PERIODS_PER_YEAR)
n_years = (dvol_df["date"].max() - dvol_df["date"].min()).days / 365.25
fl = dvol_df["flag_lag"].to_numpy()
cost = realistic_entry_cost(dvol_df["realized_var"].mean(), 2.0, periods_per_year=PERIODS_PER_YEAR)

print(f"DVOL-ffill coverage: {len(dvol_df)} bars, {n_years:.2f} yrs "
      f"({dvol_df['date'].min().date()} -> {dvol_df['date'].max().date()}), "
      f"flag rate {fl.mean():.3%} ({int(fl.sum())} flagged)")

rows = []
for label, w in [
    ("always_long", np.ones(len(dvol_df))),
    ("always_short_blind", -np.ones(len(dvol_df))),
    ("long_only_when_flagged", np.where(fl == 1, 1.0, 0.0)),
    ("short_step_aside_when_flagged", np.where(fl == 1, 0.0, -1.0)),
    ("short_flip_to_long_when_flagged", np.where(fl == 1, 1.0, -1.0)),
]:
    rows.append(dict(strike="dvol_ffill", strategy=label, **run(dvol_df, w, cost, n_years)))

# Same signal, same window, proxy (trailing-average) strike instead of DVOL --
# isolates what the real strike buys over the proxy.
proxy_df = build_variance_series(OHLCV_CSV, ANOMALY_CSV, VarSwapConfig())
proxy_df = proxy_df[proxy_df["date"] >= dvol_df["date"].min()].reset_index(drop=True)
fl_p = proxy_df["flag_lag"].to_numpy()
cost_p = realistic_entry_cost(proxy_df["realized_var"].mean(), 2.0, periods_per_year=PERIODS_PER_YEAR)
for label, w in [
    ("always_short_blind", -np.ones(len(proxy_df))),
    ("long_only_when_flagged", np.where(fl_p == 1, 1.0, 0.0)),
]:
    rows.append(dict(strike="proxy_trailing_avg", strategy=label, **run(proxy_df, w, cost_p, n_years)))

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n=== Strike comparison (same window, same signal where applicable) ===")
print(table.to_string(index=False))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/five_min_dvol_ffill_comparison.csv", index=False)

# 5-block robustness, DVOL-ffill flip-to-long only
print("\n=== 5-block robustness, DVOL-ffill, short/flip-to-long when flagged ===")
w_flip = np.where(fl == 1, 1.0, -1.0)
n_blocks = 5
block_size = len(dvol_df) // n_blocks
block_rows = []
for b in range(n_blocks):
    s, e = b * block_size, (b + 1) * block_size if b < n_blocks - 1 else len(dvol_df)
    block = dvol_df.iloc[s:e].reset_index(drop=True)
    wb = w_flip[s:e]
    nb_years = (block["date"].max() - block["date"].min()).days / 365.25
    r = run(block, wb, cost, nb_years)
    block_rows.append(dict(period=f"{block['date'].min().date()} -> {block['date'].max().date()}",
                            n_flagged=int((block['flag_lag'] == 1).sum()),
                            flag_rate=(block['flag_lag'] == 1).mean(), **r))
block_table = pd.DataFrame(block_rows)
print(block_table.to_string(index=False))
block_table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/five_min_dvol_ffill_blocks.csv", index=False)

print("\nSaved reports/five_min_dvol_ffill_comparison.csv and reports/five_min_dvol_ffill_blocks.csv")
