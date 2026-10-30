"""Same synthetic variance-swap test as run_variance_swap.py, but on the
1-hour horizon instead of daily: 308 flagged windows instead of 17, giving
real statistical power to test "does long-vol-when-flagged have an edge."
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd
from scipy import stats

from src.strategy.variance_swap import (
    VarSwapConfig, build_variance_series, fit_vol_terciles, rule_based_weights, backtest, realistic_entry_cost,
)

OHLCV_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_hourly_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/one_hour_anomaly_score.csv"

cfg = VarSwapConfig(strike_window=24 * 5)  # 5-day trailing baseline, in hours
df = build_variance_series(OHLCV_CSV, ANOMALY_CSV, cfg)
mean_hourly_var = df["realized_var"].mean()
print(f"\nmean hourly realized var={mean_hourly_var:.8f} -> annualized vol="
      f"{(mean_hourly_var * 365 * 24) ** 0.5:.1%}")
for pts in [1.0, 2.0, 4.0, 8.0]:
    c = realistic_entry_cost(mean_hourly_var, vol_spread_points=pts)
    print(f"  {pts}-point vol spread -> entry_cost={c:.10f} "
          f"({c / mean_hourly_var:.3f}x mean hourly var, vs. old arbitrary 0.0002 = "
          f"{0.0002 / mean_hourly_var:.1f}x)")
print(f"{len(df)} usable hours, {df['date'].min()} -> {df['date'].max()}")

split = int(len(df) * 0.65)
purge = cfg.strike_window
train_df = df.iloc[:split].reset_index(drop=True)
test_df = df.iloc[split + purge:].reset_index(drop=True)
print(f"train: {train_df['date'].min()} -> {train_df['date'].max()} ({len(train_df)} hours)")
print(f"test:  {test_df['date'].min()} -> {test_df['date'].max()} ({len(test_df)} hours), purge={purge}h")

vol_terciles = fit_vol_terciles(train_df)
n_flagged = int((test_df["flag_lag"] == 1).sum())
print(f"test flag rate (lagged)={test_df['flag_lag'].mean():.3%} ({n_flagged} flagged hours)")


def annualized_sharpe(pnl: np.ndarray, periods_per_year: int) -> float:
    if pnl.std() < 1e-16:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))


def report(name: str, bt: pd.DataFrame) -> dict:
    pnl = bt["pnl"].to_numpy()
    gross = bt["gross_pnl"].to_numpy()
    t_stat, p_val = stats.ttest_1samp(pnl, 0.0)
    gt_stat, gp_val = stats.ttest_1samp(gross, 0.0)
    return dict(
        strategy=name,
        gross_total_pnl=float(gross.sum()),
        gross_sharpe_ann=annualized_sharpe(gross, 365 * 24),
        gross_p_value=float(gp_val),
        total_cost=float((gross - pnl).sum()),
        net_total_pnl=float(pnl.sum()),
        net_sharpe_ann=annualized_sharpe(pnl, 365 * 24),
        net_p_value=float(p_val),
        pct_hours_active=float((bt["weight"] != 0).mean()),
    )


w_rule = rule_based_weights(test_df, vol_terciles, w_flagged=1.0, w_calm=0.0, w_mid=0.0)
w_always_short = -np.ones(len(test_df))
w_always_long = np.ones(len(test_df))

rows = []
for pts in [1.0, 2.0, 4.0, 8.0]:
    cfg_pts = VarSwapConfig(strike_window=cfg.strike_window,
                             entry_cost=realistic_entry_cost(mean_hourly_var, vol_spread_points=pts))
    rows.append(report(f"Flag-only, {pts}pt vol spread", backtest(test_df, w_rule, cfg_pts)))
rows.append(report("Flag-only, GROSS (no cost)", backtest(test_df, w_rule, VarSwapConfig(entry_cost=0.0))))
rows.append(report("Always short vol, 2pt spread (no signal)",
                    backtest(test_df, w_always_short, VarSwapConfig(entry_cost=realistic_entry_cost(mean_hourly_var, 2.0)))))
rows.append(report("Always long vol, 2pt spread (no signal)",
                    backtest(test_df, w_always_long, VarSwapConfig(entry_cost=realistic_entry_cost(mean_hourly_var, 2.0)))))

table = pd.DataFrame(rows).set_index("strategy")
print("\n=== Out-of-sample variance-swap comparison (hourly), realistic vol-spread costs ===")
print(table.to_string(float_format=lambda x: f"{x:.8f}"))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/variance_swap_hourly_comparison.csv")
cfg = VarSwapConfig(strike_window=cfg.strike_window, entry_cost=realistic_entry_cost(mean_hourly_var, 2.0))

flagged = test_df[test_df["flag_lag"] == 1]
print(f"\nOn the {len(flagged)} flagged hours: mean realized_var={flagged['realized_var'].mean():.8f}, "
      f"mean strike_var={flagged['strike_var'].mean():.8f}, "
      f"ratio={flagged['realized_var'].mean()/flagged['strike_var'].mean():.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bt_rule = backtest(test_df, w_rule, cfg)
bt_short = backtest(test_df, w_always_short, cfg)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(test_df["date"], np.cumsum(bt_rule["pnl"]), label="Flag-only (long vol when flagged)")
ax.plot(test_df["date"], np.cumsum(bt_short["pnl"]), label="Always short vol", ls="--", color="gray")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Out-of-sample cumulative variance-swap P&L, hourly horizon (variance units)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/variance_swap_hourly_pnl.png", dpi=120)
print("\nSaved reports/variance_swap_hourly_comparison.csv and reports/variance_swap_hourly_pnl.png")
