"""Honest cost check for the version of the vol trade buildable today with
zero new data: replicate "long realized variance" via spot/perp
rebalancing (no options, no variance swap), charged a real trading fee
every active period in proportion to that period's own price move -- see
src/strategy/variance_swap.py's spot_rebalancing_cost docstring for why
this is a structurally different (and structurally larger) cost than the
vol-spread model used in run_variance_swap_hourly.py.
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd
from scipy import stats

from src.strategy.variance_swap import (
    VarSwapConfig, build_variance_series, fit_vol_terciles, rule_based_weights, backtest_spot_rebalance,
)

OHLCV_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_hourly_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/one_hour_anomaly_score.csv"

cfg = VarSwapConfig(strike_window=24 * 5)
df = build_variance_series(OHLCV_CSV, ANOMALY_CSV, cfg)

split = int(len(df) * 0.65)
purge = cfg.strike_window
train_df = df.iloc[:split].reset_index(drop=True)
test_df = df.iloc[split + purge:].reset_index(drop=True)
vol_terciles = fit_vol_terciles(train_df)
n_flagged = int((test_df["flag_lag"] == 1).sum())
print(f"test: {test_df['date'].min()} -> {test_df['date'].max()} ({len(test_df)} hours, {n_flagged} flagged)")

w_rule = rule_based_weights(test_df, vol_terciles, w_flagged=1.0, w_calm=0.0, w_mid=0.0)


def annualized_sharpe(pnl: np.ndarray) -> float:
    if pnl.std() < 1e-16:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(365 * 24))


def report(name: str, bt: pd.DataFrame) -> dict:
    pnl = bt["pnl"].to_numpy()
    t_stat, p_val = stats.ttest_1samp(pnl, 0.0)
    return dict(
        strategy=name, net_total_pnl=float(pnl.sum()), net_sharpe=annualized_sharpe(pnl),
        p_value=float(p_val), total_cost=float((bt["gross_pnl"] - bt["pnl"]).sum()),
    )


# Realistic BTC perp/spot trading fees: 1bp (best-case maker, high-volume tier)
# through 5bp (standard taker on a major venue) per trade.
rows = [report("GROSS (no cost)", backtest_spot_rebalance(test_df, w_rule, trading_fee_bps=0.0))]
for bps in [0.0001, 0.0002, 0.0004, 0.0005, 0.001]:
    bt = backtest_spot_rebalance(test_df, w_rule, trading_fee_bps=bps)
    rows.append(report(f"{bps*10000:.1f}bp/trade spot rebalancing", bt))

table = pd.DataFrame(rows).set_index("strategy")
print("\n=== Spot-rebalancing cost sweep (hourly, out-of-sample) ===")
print(table.to_string(float_format=lambda x: f"{x:.8f}"))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/variance_swap_spot_cost_sweep.csv")
print("\nSaved reports/variance_swap_spot_cost_sweep.csv")
