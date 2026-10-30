"""Backtest the synthetic variance-swap strategy: trade realized volatility
directly (long when the anomaly flag fires, short only in the calmest
tercile to harvest the vol risk premium, flat otherwise) -- no RL, no
trend, no direction. See src/strategy/variance_swap.py's module docstring
for why this isn't mean-reversion despite superficially resembling one.

Same purged train/test split discipline as the RL strategy script: fit the
vol-tercile baseline on the training fold only, test out-of-sample on the
later ~35% of history.
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd
from scipy import stats

from src.strategy.variance_swap import VarSwapConfig, build_variance_series, fit_vol_terciles, rule_based_weights, backtest

OHLCV_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_daily_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/macro_daily_anomaly_score.csv"

cfg = VarSwapConfig()
df = build_variance_series(OHLCV_CSV, ANOMALY_CSV, cfg)
print(f"{len(df)} usable days, {df['date'].min().date()} -> {df['date'].max().date()}")

split = int(len(df) * 0.65)
purge = cfg.strike_window
train_df = df.iloc[:split].reset_index(drop=True)
test_df = df.iloc[split + purge:].reset_index(drop=True)
print(f"train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} ({len(train_df)} days)")
print(f"test:  {test_df['date'].min().date()} -> {test_df['date'].max().date()} ({len(test_df)} days), purge={purge}d")

vol_terciles = fit_vol_terciles(train_df)
print(f"vol (strike) terciles, fit on train only: {vol_terciles}")
print(f"test flag rate (lagged)={test_df['flag_lag'].mean():.3%}")


def annualized_sharpe(pnl: np.ndarray) -> float:
    if pnl.std() < 1e-14:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(365))


def report(name: str, bt: pd.DataFrame) -> dict:
    pnl = bt["pnl"].to_numpy()
    t_stat, p_val = stats.ttest_1samp(pnl, 0.0)
    return dict(
        strategy=name,
        total_pnl=float(pnl.sum()),
        daily_sharpe_ann=annualized_sharpe(pnl),
        mean_daily_pnl=float(pnl.mean()),
        pct_days_active=float((bt["weight"] != 0).mean()),
        turnover=float(np.abs(np.diff(bt["weight"].to_numpy(), prepend=0.0)).mean()),
        t_stat=float(t_stat),
        p_value=float(p_val),
    )


rows = []

w_rule = rule_based_weights(test_df, vol_terciles)
rows.append(report("Rule-based (long on flag, short in calmest tercile)", backtest(test_df, w_rule, cfg)))

w_flag_only = np.where(test_df["flag_lag"].to_numpy() == 1, 1.0, 0.0)
rows.append(report("Flag-only (long vol when flagged, else flat)", backtest(test_df, w_flag_only, cfg)))

w_always_short = -np.ones(len(test_df))
rows.append(report("Always short vol (sell the risk premium, no signal)", backtest(test_df, w_always_short, cfg)))

w_always_long = np.ones(len(test_df))
rows.append(report("Always long vol (no signal)", backtest(test_df, w_always_long, cfg)))

table = pd.DataFrame(rows).set_index("strategy")
print("\n=== Out-of-sample variance-swap comparison ===")
print(table.to_string(float_format=lambda x: f"{x:.6f}"))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/variance_swap_comparison.csv")

print("\n=== Sanity check: is the baseline forecast itself biased? ===")
print(f"mean realized_var={test_df['realized_var'].mean():.6f}, mean strike_var={test_df['strike_var'].mean():.6f}, "
      f"ratio={test_df['realized_var'].mean()/test_df['strike_var'].mean():.3f} (1.0 = unbiased baseline)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bt_rule = backtest(test_df, w_rule, cfg)
bt_short = backtest(test_df, w_always_short, cfg)
bt_flag = backtest(test_df, w_flag_only, cfg)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(test_df["date"], np.cumsum(bt_rule["pnl"]), label="Rule-based (flag + calm-harvest)")
ax.plot(test_df["date"], np.cumsum(bt_flag["pnl"]), label="Flag-only (long vol when flagged)")
ax.plot(test_df["date"], np.cumsum(bt_short["pnl"]), label="Always short vol", ls="--", color="gray")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Out-of-sample cumulative variance-swap P&L (variance units)")
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.fill_between(test_df["date"], 0, w_rule, step="mid", color="darkorange", alpha=0.6)
ax2.set_ylim(-1.2, 1.2)
ax2.set_title("Rule-based strategy weight (+1 = long vol, -1 = short vol)")
plt.tight_layout()
plt.savefig("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/variance_swap_pnl.png", dpi=120)
print("\nSaved reports/variance_swap_comparison.csv and reports/variance_swap_pnl.png")
