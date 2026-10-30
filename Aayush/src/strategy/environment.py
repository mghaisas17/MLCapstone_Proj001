"""A discrete-state, discrete-action MDP that actually buys and sells BTC
(long / flat / short), learned by Q-learning from price history plus the
anomaly detector -- not a long-only leverage dial, and not mean-reversion.

Where direction comes from vs. where risk-sizing comes from, and why they're
kept separate: the validated anomaly signal (both Aayush's macro
recency-weighted expected-signature-distance score and Mihika's independent
signature-detector gate at the 1-day horizon) is that a window's *path
shape* looks unusual -- it was never shown to predict which way price moves
next, only that forward volatility tends to be elevated/surprising. Asking
it for direction would be asking a question it was never validated to
answer. Direction instead comes from the price history itself, via trend
(momentum) -- trend-following, the opposite of mean-reversion, and one of
the few things with any real documented edge in crypto. The RL agent is
handed both kinds of feature (trend from price history, regime risk from
the anomaly detector and realized vol) and learns from historical P&L which
combinations warrant being long, flat, or short.

State size, and why it's this small: an earlier version used a 54-state
table (flag x vol tercile x fast-momentum tercile x slow-momentum tercile).
On an out-of-sample backtest it churned constantly (turnover > 0.3/day,
419 position changes in 1502 test days) and lost money, including in the
ablation without the anomaly feature and in a plain trend-follower with no
RL at all -- and raising transaction costs 20x did not reduce the churn,
which rules out "flipping is just priced too cheap" and points at the real
cause: ~2,800 training days spread across 54 states (many further split
3 ways by action) leaves most cells with only a handful of visits, so the
learned greedy action per state is largely noise, not signal. Dropping fast
momentum entirely and collapsing slow momentum from 3 bins to 2 (binary
trend direction) cuts the table to 12 cells, which a backtest confirms is
enough to recover a non-degenerate, far less noisy policy (see
scripts/run_rl_strategy.py's printed comparison).

State (every feature lagged by one day -- known at the close of day t-1,
used to choose the position held through day t's return, so there is no
look-ahead):
    anomaly_flag_lag  in {0, 1}  -- the detector's alpha=0.01 flag
    vol_tercile_lag   in {0,1,2} -- trailing 20-day realized-vol tercile
    trend_up_lag      in {0, 1}  -- trailing `mom_slow_window`-day momentum
                                    (log return) above/below its median,
                                    median fit on the TRAINING fold only

Action: an index into `EXPOSURE_LEVELS` -- the actual position to hold for
the coming day. Negative = short, 0 = flat, positive = long: this is the
buy/sell signal (a BUY is any day the position moves up; a SELL is any day
it moves down, including going short).

Reward for the transition into day t:
    r_t = m_t * ret_t
          - txn_cost * |m_t - m_{t-1}|
          - risk_lambda * (m_t * vol_lag_t) ** 2
`ret_t` is realized only once the action is taken (not an input to the
state), so this is still a legitimate, non-anticipating reward. The variance
penalty is charged on the *lagged* (already-known) vol estimate, never on
realized vol_t itself -- penalizing large positions when the forecast ahead
of time was risky, not when it happened to pay off or not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EXPOSURE_LEVELS = (-1.0, 0.0, 1.0)


@dataclass
class EnvConfig:
    txn_cost: float = 0.0005       # 5bps per unit change in position
    risk_lambda: float = 0.1       # risk-aversion on (position * lagged vol)^2 -- see
                                    # q_learning.py's QLearningConfig docstring: this had to be
                                    # recalibrated down from an initial 3.0, which completely
                                    # swamped BTC's thin ~0.0014/day mean edge and collapsed
                                    # every policy to always-flat regardless of state
    vol_window: int = 20            # trailing realized-vol lookback, in days
    mom_slow_window: int = 20       # trend-direction lookback, in days


def build_features(price_csv: str, anomaly_csv: str, cfg: "EnvConfig | None" = None) -> pd.DataFrame:
    """Join daily close prices with the (already date-aligned) anomaly
    score/flag, and add the trailing realized-vol and momentum features
    computed purely from the price history. Everything that becomes part of
    the *state* is shifted by one day inside `make_episode`, not here --
    this just builds the raw time series."""
    cfg = cfg or EnvConfig()
    px = pd.read_csv(price_csv, parse_dates=["date"])
    px["date"] = pd.to_datetime(px["date"], utc=True).dt.tz_localize(None)
    px = px[["date", "close"]].sort_values("date").reset_index(drop=True)
    px["log_return"] = np.log(px["close"]).diff()
    px["realized_vol"] = px["log_return"].rolling(cfg.vol_window).std()
    px["mom_slow"] = np.log(px["close"]).diff(cfg.mom_slow_window)

    an = pd.read_csv(anomaly_csv, parse_dates=["date"])
    an["date"] = pd.to_datetime(an["date"], utc=True).dt.tz_localize(None)
    an = an[["date", "anomaly_score", "flag"]]

    df = px.merge(an, on="date", how="left").dropna(subset=["log_return", "realized_vol", "mom_slow"])
    df["flag"] = df["flag"].fillna(0).astype(int)
    df["anomaly_score"] = df["anomaly_score"].fillna(df["anomaly_score"].median())
    return df.reset_index(drop=True)


@dataclass
class Thresholds:
    vol_terciles: tuple[float, float]
    mom_slow_median: float


@dataclass
class Episode:
    """One contiguous slice of history, pre-discretized into states, ready
    to be replayed by the Q-learning loop. All bin edges are fixed (fit on
    the training fold) so a test episode's discretization never uses
    information from its own distribution."""
    dates: pd.DatetimeIndex
    returns: np.ndarray          # ret_t, realized return for day t
    flag_lag: np.ndarray         # known at t-1
    vol_tercile_lag: np.ndarray  # known at t-1, discretized with FIXED edges
    trend_up_lag: np.ndarray     # known at t-1, discretized with a FIXED median
    vol_lag: np.ndarray          # continuous trailing vol, known at t-1 (for the reward penalty)
    cfg: EnvConfig


def make_episode(df: pd.DataFrame, cfg: EnvConfig, thresholds: Thresholds | None = None) -> tuple[Episode, Thresholds]:
    """Build an Episode from a (already time-sliced) feature frame. If
    `thresholds` is None, fit them on this slice (call this on the TRAIN
    slice only); otherwise reuse the given edges (call this on validation/
    test slices with the train-fitted edges)."""
    vol_lag = df["realized_vol"].shift(1).to_numpy()
    mom_slow_lag = df["mom_slow"].shift(1).to_numpy()
    flag_lag = df["flag"].shift(1).fillna(0).to_numpy().astype(int)
    returns = df["log_return"].to_numpy()
    dates = pd.DatetimeIndex(df["date"])

    valid = ~(np.isnan(vol_lag) | np.isnan(mom_slow_lag))

    if thresholds is None:
        q1, q2 = np.nanquantile(vol_lag[valid], [1 / 3, 2 / 3])
        thresholds = Thresholds(vol_terciles=(float(q1), float(q2)),
                                 mom_slow_median=float(np.nanmedian(mom_slow_lag[valid])))

    vol_tercile_lag = np.digitize(vol_lag, list(thresholds.vol_terciles))
    trend_up_lag = (mom_slow_lag > thresholds.mom_slow_median).astype(int)

    ep = Episode(
        dates=dates[valid], returns=returns[valid], flag_lag=flag_lag[valid],
        vol_tercile_lag=vol_tercile_lag[valid].astype(int), trend_up_lag=trend_up_lag[valid],
        vol_lag=vol_lag[valid], cfg=cfg,
    )
    return ep, thresholds


def encode_state(flag_lag: int, vol_tercile_lag: int, trend_up_lag: int, use_anomaly: bool,
                  use_trend: bool = True) -> int:
    """Flatten the discrete state tuple to a single integer index for the
    Q-table: 2 (flag, or 1 without it) x 3 (vol tercile) x (2 if use_trend
    else 1) cells. `use_trend=False` drops momentum entirely -- direction
    then has to come purely from whatever the anomaly flag/vol regime alone
    implies, which is the direct test of "can the anomaly signal alone
    (no trend) justify a position, and which direction does it pick."""
    f = flag_lag if use_anomaly else 0
    t = trend_up_lag if use_trend else 0
    n_trend = 2 if use_trend else 1
    return (f * 3 + vol_tercile_lag) * n_trend + t


def n_states(use_anomaly: bool, use_trend: bool = True) -> int:
    n_flag = 2 if use_anomaly else 1
    n_trend = 2 if use_trend else 1
    return n_flag * 3 * n_trend


def step_reward(action_idx: int, prev_action_idx: int, ret_t: float, vol_lag_t: float, cfg: EnvConfig) -> float:
    m, m_prev = EXPOSURE_LEVELS[action_idx], EXPOSURE_LEVELS[prev_action_idx]
    pnl = m * ret_t
    cost = cfg.txn_cost * abs(m - m_prev)
    risk_penalty = cfg.risk_lambda * (m * vol_lag_t) ** 2
    return pnl - cost - risk_penalty
