"""Train and evaluate the long/flat/short RL strategy, with two ablations:
does the anomaly-score feature help (the trading-strategy analogue of
Mihika's own detector "gate" test), and how much of any edge is just
trend-following that was already there without it.

Train/test split: a single purged walk-forward split (train on the earlier
~65% of history, a 20-day purge gap equal to the longest feature lookback,
then test on the later ~35%, which includes Terra/Luna, FTX and the 2023
banking crisis as well as the calmer 2024-2026 stretch) -- not multiple
folds, for simplicity; this is a real limitation, stated plainly in the
printed report, matching how this project has flagged the same limitation
elsewhere (one historical path, consecutive blocks, not independent draws).
"""
import sys

sys.path.insert(0, "/Users/aayushjain/Desktop/BNY_Mellon_Proj")

import numpy as np
import pandas as pd

from src.strategy.environment import EnvConfig, build_features, make_episode, encode_state, EXPOSURE_LEVELS
from src.strategy.q_learning import (
    QLearningConfig, train_q_learning, run_policy, rule_based_vol_target, trend_following_baseline,
)
from src.strategy.metrics import summarize

PRICE_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/btc_daily_ohlcv.csv"
ANOMALY_CSV = "/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/macro_daily_anomaly_score.csv"

cfg = EnvConfig()
df = build_features(PRICE_CSV, ANOMALY_CSV, cfg)
print(f"{len(df)} usable days, {df['date'].min().date()} -> {df['date'].max().date()}, "
      f"flag rate in full history={df['flag'].mean():.3%}")

split = int(len(df) * 0.65)
purge = 20
train_df = df.iloc[:split].reset_index(drop=True)
test_df = df.iloc[split + purge:].reset_index(drop=True)
print(f"train: {train_df['date'].min().date()} -> {train_df['date'].max().date()} ({len(train_df)} days)")
print(f"test:  {test_df['date'].min().date()} -> {test_df['date'].max().date()} ({len(test_df)} days), "
      f"purge gap={purge} days")

train_ep, thresholds = make_episode(train_df, cfg, thresholds=None)
test_ep, _ = make_episode(test_df, cfg, thresholds=thresholds)
print(f"vol terciles (fit on train only): {thresholds.vol_terciles}, "
      f"trend median: {thresholds.mom_slow_median:.4f}")
print(f"train flag rate (lagged)={train_ep.flag_lag.mean():.3%}, test flag rate (lagged)={test_ep.flag_lag.mean():.3%}")

qcfg = QLearningConfig()

print("\nTraining Q-learning WITH anomaly feature...")
Q_with = train_q_learning(train_ep, use_anomaly=True, qcfg=qcfg)
print("Training Q-learning WITHOUT anomaly feature (ablation: trend + vol only)...")
Q_without = train_q_learning(train_ep, use_anomaly=False, qcfg=qcfg)

results = {}
for label, (Q, use_anomaly) in {
    "RL (trend+vol+anomaly)": (Q_with, True),
    "RL (trend+vol only, ablation)": (Q_without, False),
}.items():
    out = run_policy(test_ep, Q, use_anomaly=use_anomaly)
    exposure = np.array([EXPOSURE_LEVELS[a] for a in out["actions"]])
    results[label] = summarize(label, out["returns"], exposure)
    results[label]["_out"] = out
    results[label]["_exposure"] = exposure

bh_returns = test_ep.returns  # m=1.0 always, no cost (never trades)
results["Buy and hold"] = summarize("Buy and hold", bh_returns, np.ones(len(bh_returns)))

rb = rule_based_vol_target(test_ep)
results["Rule-based vol-target (long-only, no RL)"] = summarize(
    "Rule-based vol-target (long-only, no RL)", rb["returns"], rb["actions_m"])

tf = trend_following_baseline(test_ep)
results["Trend-following only (no RL, no anomaly)"] = summarize(
    "Trend-following only (no RL, no anomaly)", tf["returns"], tf["actions_m"])

print("\n=== Out-of-sample comparison ===")
table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results.values()])
table = table.set_index("strategy")
print(table.to_string(float_format=lambda x: f"{x:.4f}"))
table.to_csv("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/rl_strategy_comparison.csv")

print("\n=== What the WITH-anomaly policy learned (greedy action per state) ===")
for flag in (0, 1):
    for vol_t in (0, 1, 2):
        row = []
        for trend in (0, 1):
            s = encode_state(flag, vol_t, trend, use_anomaly=True)
            row.append(EXPOSURE_LEVELS[int(np.argmax(Q_with[s]))])
        print(f"flag={flag} vol_tercile={vol_t} -> action by trend(down,up): {row}")

# --- the actual buy/sell signal: trade list on the OOS test period ---
print("\n=== RL (trend+vol+anomaly) trade list on the test period (position changes only) ===")
exposure = results["RL (trend+vol+anomaly)"]["_exposure"]
dates = test_ep.dates
prev = 0.0
n_trades = 0
for d, m in zip(dates, exposure):
    if m != prev:
        side = "BUY " if m > prev else "SELL"
        label = {-1.0: "SHORT", 0.0: "FLAT", 1.0: "LONG"}[m]
        print(f"  {d.date()}  {side} -> now {label}")
        n_trades += 1
        prev = m
print(f"{n_trades} position changes over {len(dates)} test days "
      f"(mean exposure={exposure.mean():.2f}, turnover={results['RL (trend+vol+anomaly)']['turnover']:.3f}/day)")

print("\nSaved comparison table to rl_strategy_comparison.csv")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0]
ax.plot(dates, np.exp(np.cumsum(results["RL (trend+vol+anomaly)"]["_out"]["returns"])), label="RL + anomaly")
ax.plot(dates, np.exp(np.cumsum(results["RL (trend+vol only, ablation)"]["_out"]["returns"])), label="RL, no anomaly")
ax.plot(dates, np.exp(np.cumsum(bh_returns)), label="Buy and hold", ls="--", color="gray")
ax.plot(dates, np.exp(np.cumsum(rb["returns"])), label="Rule-based vol-target", ls=":", color="black")
ax.plot(dates, np.exp(np.cumsum(tf["returns"])), label="Trend-following only", ls="-.", color="brown")
ax.set_yscale("log")
ax.set_title("Out-of-sample equity curves (log scale)")
ax.legend(fontsize=8)

ax2 = axes[1]
ax2.fill_between(dates, 0, exposure, step="mid", color="steelblue", alpha=0.6)
ax2.set_ylim(-1.2, 1.2)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(["SHORT", "FLAT", "LONG"])
ax2.set_title("RL (trend+vol+anomaly) position")
plt.tight_layout()
plt.savefig("/Users/aayushjain/Desktop/BNY_Mellon_Proj/reports/rl_strategy_equity_curves.png", dpi=120)
print("Saved equity-curve + position plot to reports/rl_strategy_equity_curves.png")
