"""Sweep min-hold (debounce) windows on the daily/macro variance-swap
strategy: same flag, same trigger, only the hold duration changes. Targets
the exact problem the hourly backtest showed -- isolated single-bar flags
paying entry+exit cost almost every time.
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd
from scipy import stats

from src.strategy.variance_swap import VarSwapConfig, build_variance_series, debounce_flag, backtest

OHLCV_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_daily_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/macro_daily_anomaly_score.csv"

cfg = VarSwapConfig()
df = build_variance_series(OHLCV_CSV, ANOMALY_CSV, cfg)
split = int(len(df) * 0.65)
purge = cfg.strike_window
test_df = df.iloc[split + purge:].reset_index(drop=True)
print(f"test: {test_df['date'].min().date()} -> {test_df['date'].max().date()} ({len(test_df)} days)")
print(f"test flag rate (lagged)={test_df['flag_lag'].mean():.3%} ({int(test_df['flag_lag'].sum())} flagged days)")


def annualized_sharpe(pnl: np.ndarray) -> float:
    if pnl.std() < 1e-16:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(365))


def report(name: str, bt: pd.DataFrame) -> dict:
    pnl, gross = bt["pnl"].to_numpy(), bt["gross_pnl"].to_numpy()
    _, gp = stats.ttest_1samp(gross, 0.0)
    _, np_ = stats.ttest_1samp(pnl, 0.0)
    return dict(
        strategy=name,
        gross_total_pnl=float(gross.sum()), gross_sharpe=annualized_sharpe(gross), gross_p=float(gp),
        total_cost=float((gross - pnl).sum()),
        net_total_pnl=float(pnl.sum()), net_sharpe=annualized_sharpe(pnl), net_p=float(np_),
        pct_days_active=float((bt["weight"] != 0).mean()), n_trades=int(np.sum(np.abs(np.diff(bt["weight"], prepend=0)) > 0)),
    )


rows = []
for min_hold in [1, 3, 5, 7, 10, 14, 21]:
    w = debounce_flag(test_df["flag_lag"].to_numpy(), min_hold)
    bt = backtest(test_df, w, cfg)
    rows.append(report(f"min_hold={min_hold}d", bt))

table = pd.DataFrame(rows).set_index("strategy")
print("\n=== Daily/macro variance-swap: debounce sweep (out-of-sample) ===")
print(table.to_string(float_format=lambda x: f"{x:.6f}"))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/variance_swap_debounce_sweep.csv")
print("\nSaved reports/variance_swap_debounce_sweep.csv")
