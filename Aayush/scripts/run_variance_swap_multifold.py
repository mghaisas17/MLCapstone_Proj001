"""Robustness check for the hourly variance-swap result: instead of one
65/35 split, cut the full ~3-year scored period (Sept 2023 - Aug 2026)
into 4 sequential blocks and report the flag-only long-vol strategy's
stats on EACH independently. A real, structural edge should show up (even
if noisily) across most blocks; an edge concentrated in just one lucky
block is a very different, much weaker claim than the single-split result
suggested.
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd
from scipy import stats

from src.strategy.variance_swap import VarSwapConfig, build_variance_series, realistic_entry_cost, backtest, backtest_spot_rebalance

OHLCV_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_hourly_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/one_hour_anomaly_score.csv"

cfg = VarSwapConfig(strike_window=24 * 5)
df = build_variance_series(OHLCV_CSV, ANOMALY_CSV, cfg)
df = df.dropna(subset=["strike_var"]).reset_index(drop=True)  # drop the warm-up with no trailing baseline yet
mean_hourly_var = df["realized_var"].mean()
cost_cfg = VarSwapConfig(strike_window=cfg.strike_window, entry_cost=realistic_entry_cost(mean_hourly_var, 2.0))

n_blocks = 4
block_size = len(df) // n_blocks


def annualized_sharpe(pnl: np.ndarray) -> float:
    if pnl.std() < 1e-16:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(365 * 24))


rows = []
for b in range(n_blocks):
    start, end = b * block_size, (b + 1) * block_size if b < n_blocks - 1 else len(df)
    block = df.iloc[start:end].reset_index(drop=True)
    w = (block["flag_lag"].to_numpy() == 1).astype(float)
    n_flagged = int(w.sum())

    bt_gross = backtest(block, w, VarSwapConfig(entry_cost=0.0))
    bt_vol = backtest(block, w, cost_cfg)
    bt_spot = backtest_spot_rebalance(block, w, trading_fee_bps=0.0005)

    g_t, g_p = stats.ttest_1samp(bt_gross["pnl"], 0.0) if n_flagged >= 2 else (np.nan, np.nan)
    v_t, v_p = stats.ttest_1samp(bt_vol["pnl"], 0.0) if n_flagged >= 2 else (np.nan, np.nan)

    rows.append(dict(
        block=f"{block['date'].min().date()} -> {block['date'].max().date()}",
        n_hours=len(block), n_flagged=n_flagged, flag_rate=w.mean(),
        gross_sharpe=annualized_sharpe(bt_gross["pnl"].to_numpy()), gross_p=float(g_p) if n_flagged >= 2 else np.nan,
        vol_cost_sharpe=annualized_sharpe(bt_vol["pnl"].to_numpy()), vol_cost_p=float(v_p) if n_flagged >= 2 else np.nan,
        spot_5bp_sharpe=annualized_sharpe(bt_spot["pnl"].to_numpy()),
        mean_ratio_on_flagged=(block.loc[w == 1, "realized_var"].mean() / block.loc[w == 1, "strike_var"].mean()
                                if n_flagged > 0 else np.nan),
    ))

table = pd.DataFrame(rows).set_index("block")
print("=== Hourly variance-swap, flag-only: 4-block robustness check ===")
print(table.to_string(float_format=lambda x: f"{x:.4f}"))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/variance_swap_multifold.csv")

# Pooled across all blocks (every flagged hour vs every unflagged hour, full period)
w_all = (df["flag_lag"].to_numpy() == 1).astype(float)
bt_all_gross = backtest(df, w_all, VarSwapConfig(entry_cost=0.0))
bt_all_vol = backtest(df, w_all, cost_cfg)
t_all, p_all = stats.ttest_1samp(bt_all_gross["pnl"], 0.0)
print(f"\n=== Pooled across the full period ({int(w_all.sum())} flagged hours of {len(df)}) ===")
print(f"gross sharpe={annualized_sharpe(bt_all_gross['pnl'].to_numpy()):.4f}, p={p_all:.5f}")
print(f"vol-cost(2pt) sharpe={annualized_sharpe(bt_all_vol['pnl'].to_numpy()):.4f}")
print("\nSaved reports/variance_swap_multifold.csv")
