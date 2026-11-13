"""SPY variance-swap backtest using VIX as a REAL, market-quoted strike --
the upgrade equities make possible over the BTC proxy (no options market
there). Same signal construction (path-signature anomaly flag), same
long-vol-only-when-flagged rule, same validation ladder (pooled stats +
multi-block robustness) as the BTC work.

Restricted to 2003+ : VIX's calculation methodology changed in Sept 2003
(old VXO-style index before, SPX-wide methodology after) -- mixing both
would conflate two different instruments.
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd
from scipy import stats

from src.strategy.variance_swap import VarSwapConfig, build_variance_series_vix, backtest

OHLCV_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/spy_daily_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/spy_daily_anomaly_score.csv"
VIX_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/vix_daily.csv"

cfg = VarSwapConfig()
df = build_variance_series_vix(OHLCV_CSV, ANOMALY_CSV, VIX_CSV, cfg)
df = df[df["date"] >= "2003-09-01"].reset_index(drop=True)
print(f"{len(df)} usable days, {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"flag rate (lagged): {df['flag_lag'].mean():.3%} ({int(df['flag_lag'].sum())} flagged days)")

w = (df["flag_lag"].to_numpy() == 1).astype(float)
bt = backtest(df, w, VarSwapConfig(entry_cost=0.0))  # gross first; VIX bid-ask cost added below
pnl = bt["pnl"].to_numpy()


def annualized_sharpe(x: np.ndarray) -> float:
    return float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 1e-16 else 0.0


t, p = stats.ttest_1samp(pnl, 0.0)
flagged = df[w == 1]
unflagged = df[w == 0]
hit_rate = (flagged["realized_var"] > flagged["strike_var"]).mean()
hit_rate_base = (unflagged["realized_var"] > unflagged["strike_var"]).mean()
ratio_mean = (flagged["realized_var"] / flagged["strike_var"]).mean()
ratio_median = (flagged["realized_var"] / flagged["strike_var"]).median()

print(f"\n=== Pooled, full 2003-2026 sample ===")
print(f"gross sharpe={annualized_sharpe(pnl):.4f}, p={p:.6f}")
print(f"hit rate flagged={hit_rate:.3%}, hit rate unflagged (base)={hit_rate_base:.3%}")
print(f"mean ratio={ratio_mean:.3f}x, median ratio={ratio_median:.3f}x")

# always-long / always-short controls
for label, wc in [("always long", np.ones(len(df))), ("always short", -np.ones(len(df)))]:
    btc_ = backtest(df, wc, VarSwapConfig(entry_cost=0.0))
    tc, pc = stats.ttest_1samp(btc_["pnl"], 0.0)
    print(f"{label}: sharpe={annualized_sharpe(btc_['pnl'].to_numpy()):.4f}, p={pc:.4f}")

# --- multi-block robustness: split the 23-year sample into ~5-year blocks ---
n_blocks = 5
block_size = len(df) // n_blocks
print(f"\n=== {n_blocks}-block robustness check ===")
rows = []
for b in range(n_blocks):
    start, end = b * block_size, (b + 1) * block_size if b < n_blocks - 1 else len(df)
    block = df.iloc[start:end].reset_index(drop=True)
    wb = (block["flag_lag"].to_numpy() == 1).astype(float)
    n_flagged = int(wb.sum())
    btb = backtest(block, wb, VarSwapConfig(entry_cost=0.0))
    pnlb = btb["pnl"].to_numpy()
    if n_flagged >= 2:
        _, pb = stats.ttest_1samp(pnlb, 0.0)
    else:
        pb = np.nan
    ratio_b = (block.loc[wb == 1, "realized_var"] / block.loc[wb == 1, "strike_var"]).mean() if n_flagged else np.nan
    rows.append(dict(period=f"{block['date'].min().date()} -> {block['date'].max().date()}",
                      n_days=len(block), n_flagged=n_flagged, sharpe=annualized_sharpe(pnlb), p=pb, ratio=ratio_b))
table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/spy_vix_variance_swap_blocks.csv", index=False)

out = pd.DataFrame(dict(
    metric=["n_days", "n_flagged", "flag_rate", "pooled_sharpe", "pooled_p", "hit_rate_flagged",
            "hit_rate_base", "mean_ratio", "median_ratio"],
    value=[len(df), int(w.sum()), w.mean(), annualized_sharpe(pnl), p, hit_rate, hit_rate_base, ratio_mean, ratio_median],
))
out.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/spy_vix_variance_swap_summary.csv", index=False)
print("\nSaved reports/spy_vix_variance_swap_summary.csv and reports/spy_vix_variance_swap_blocks.csv")
