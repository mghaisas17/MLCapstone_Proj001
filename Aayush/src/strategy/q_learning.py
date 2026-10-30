"""Tabular Q-learning for the long/flat/short MDP in `environment.py`.

Why tabular, not a deep policy: the whole dataset is ~4,000 days of ONE
historical BTC path. A deep RL agent (PPO/DQN with a neural net) has far
more parameters than independent data points here and would be trained
almost entirely on a single trajectory it will be evaluated against -- a
well-known failure mode in finance RL. Discretizing the state down to a
handful of bins (anomaly flag x vol tercile x trend direction) keeps the
parameter count (a lookup table of at most 12 states x 3 actions = 36
entries) well below the data size, and makes convergence and the policy
itself fully inspectable (print the table, see exactly what it learned),
which matters for an honest report of what the agent is and isn't doing.
(An earlier, larger 54-state version of this table overfit badly -- see
environment.py's module docstring for the backtest that showed it.)

Training loop: Q-learning is off-policy, so replaying the SAME fixed
historical return sequence for many epochs with epsilon-greedy exploration
(and the previous day's position carried over between steps, since it
affects the transaction-cost term) is a standard way to let tabular
Q-learning converge when there is only one real historical trajectory to
learn from. Position is NOT part of the state itself (see
`encode_state`'s docstring) -- carrying it between steps here is only for
charging the real cost of the agent's actual choices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.strategy.environment import EXPOSURE_LEVELS, Episode, encode_state, n_states, step_reward


@dataclass
class QLearningConfig:
    # A CONSTANT alpha keeps re-weighting toward recent noisy rewards and
    # never settles -- fatal here, since BTC's daily mean/std ratio is tiny
    # (~0.0014 / 0.039 on the training fold): a fixed-alpha estimate's own
    # noise floor is too high to ever resolve an edge that small. Each
    # (state, action) pair instead gets its own visit-count-based rate
    # alpha_t = 1 / (1 + N(s,a)), the textbook incremental sample-average --
    # provably converges to the true expected reward as N -> infinity,
    # which a constant rate does not.
    alpha_min: float = 0.01   # floor, so very-late updates still adapt a little
    gamma: float = 0.95
    epsilon_start: float = 0.3
    epsilon_end: float = 0.02
    n_epochs: int = 400
    seed: int = 0


def train_q_learning(ep: Episode, use_anomaly: bool, qcfg: QLearningConfig, use_trend: bool = True) -> np.ndarray:
    rng = np.random.default_rng(qcfg.seed)
    n_actions = len(EXPOSURE_LEVELS)
    n_s = n_states(use_anomaly, use_trend)
    Q = np.zeros((n_s, n_actions))
    visits = np.zeros((n_s, n_actions), dtype=np.int64)
    T = len(ep.returns)

    for epoch in range(qcfg.n_epochs):
        eps = qcfg.epsilon_start + (qcfg.epsilon_end - qcfg.epsilon_start) * epoch / max(1, qcfg.n_epochs - 1)
        prev_action = 0  # start flat each epoch (EXPOSURE_LEVELS[1] == 0.0 -- see note below)
        for t in range(T):
            s = encode_state(ep.flag_lag[t], ep.vol_tercile_lag[t], ep.trend_up_lag[t], use_anomaly, use_trend)
            if rng.random() < eps:
                a = rng.integers(n_actions)
            else:
                a = int(np.argmax(Q[s]))
            r = step_reward(a, prev_action, ep.returns[t], ep.vol_lag[t], ep.cfg)
            if t + 1 < T:
                s_next = encode_state(ep.flag_lag[t + 1], ep.vol_tercile_lag[t + 1], ep.trend_up_lag[t + 1],
                                       use_anomaly, use_trend)
                target = r + qcfg.gamma * np.max(Q[s_next])
            else:
                target = r
            visits[s, a] += 1
            alpha = max(qcfg.alpha_min, 1.0 / visits[s, a])
            Q[s, a] += alpha * (target - Q[s, a])
            prev_action = a
    return Q


def run_policy(ep: Episode, Q: np.ndarray, use_anomaly: bool, use_trend: bool = True) -> dict:
    """Greedy rollout of a trained (or hand-specified) Q-table on an episode
    (held-out test data, no exploration, no learning updates). Returns the
    realized daily strategy returns and the action sequence."""
    n_actions = len(EXPOSURE_LEVELS)
    T = len(ep.returns)
    prev_action = 0  # EXPOSURE_LEVELS[1] == 0.0: start flat
    actions = np.zeros(T, dtype=int)
    strat_returns = np.zeros(T)
    for t in range(T):
        s = encode_state(ep.flag_lag[t], ep.vol_tercile_lag[t], ep.trend_up_lag[t], use_anomaly, use_trend)
        a = int(np.argmax(Q[s]))
        m, m_prev = EXPOSURE_LEVELS[a], EXPOSURE_LEVELS[prev_action]
        strat_returns[t] = m * ep.returns[t] - ep.cfg.txn_cost * abs(m - m_prev)
        actions[t] = a
        prev_action = a
    return dict(dates=ep.dates, returns=strat_returns, actions=actions, raw_returns=ep.returns)


def rule_based_vol_target(ep: Episode, target_vol: float = 0.02, max_leverage: float = 2.0) -> dict:
    """Classical, non-RL, long-only baseline: m_t = clip(target_vol / vol_lag_t, 0, max_leverage).
    No anomaly feature, no direction, no learning -- the standard
    vol-targeting rule this project's RL agent needs to beat to justify the
    extra complexity."""
    T = len(ep.returns)
    strat_returns = np.zeros(T)
    actions_m = np.zeros(T)
    prev_m = 0.0
    for t in range(T):
        m = float(np.clip(target_vol / max(ep.vol_lag[t], 1e-6), 0.0, max_leverage))
        strat_returns[t] = m * ep.returns[t] - ep.cfg.txn_cost * abs(m - prev_m)
        actions_m[t] = m
        prev_m = m
    return dict(dates=ep.dates, returns=strat_returns, actions_m=actions_m, raw_returns=ep.returns)


def trend_following_baseline(ep: Episode) -> dict:
    """Classical, non-RL, non-anomaly baseline: m_t = +1 if trend_up_lag else -1,
    i.e. plain trend-following with no risk-sizing and no anomaly feature --
    isolates how much of the RL agent's edge (if any) is just "trend was
    already a signal" versus something the anomaly/vol conditioning adds."""
    T = len(ep.returns)
    strat_returns = np.zeros(T)
    actions_m = np.zeros(T)
    prev_m = 0.0
    for t in range(T):
        m = 1.0 if ep.trend_up_lag[t] else -1.0
        strat_returns[t] = m * ep.returns[t] - ep.cfg.txn_cost * abs(m - prev_m)
        actions_m[t] = m
        prev_m = m
    return dict(dates=ep.dates, returns=strat_returns, actions_m=actions_m, raw_returns=ep.returns)
