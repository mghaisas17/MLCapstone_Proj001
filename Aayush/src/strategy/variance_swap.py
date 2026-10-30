"""A synthetic variance-swap: trade realized volatility directly, with no
direction, no trend feature, and -- despite superficially resembling one --
not a mean-reversion bet.

The mechanism (why this isn't mean-reversion): a real variance swap pays
`notional * (realized_variance - strike_variance)` over its life. Its
market-making equivalent, delta-hedging a straddle, requires buying as
price falls and selling as it rises *relative to a reference that resets
after every rebalance* -- so the P&L comes from how much the market
oscillated (realized variance), never from a bet that price returns to any
fixed historical level. A mean-reversion trade bets on a level; this bets
on motion, full stop -- it pays off identically whether price round-trips
back to where it started or drifts off to a new price, as long as it got
there the choppy way.

We don't have an options market to read a real strike from (no Deribit/
options pipeline in this project yet), so the "strike" here is our own
trailing realized-variance baseline -- economically the same role a
market-quoted implied variance plays, just proxied from history instead of
priced by a market. That's a real limitation, stated plainly: this measures
whether the anomaly signal has any economic value for a vol trade, not a
literal tradable P&L (a real desk would strike against market-implied vol,
not its own forecast).

Daily realized variance uses Parkinson's high-low estimator (needs only
the OHLC we already have, no options), which is a standard, much
lower-noise single-day variance estimate than squared close-to-close
return:
    parkinson_var_t = (1 / (4 ln 2)) * (ln(high_t / low_t))^2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PARKINSON_CONST = 1.0 / (4.0 * np.log(2.0))


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    return PARKINSON_CONST * (np.log(high / low)) ** 2


@dataclass
class VarSwapConfig:
    strike_window: int = 20   # trailing days used for the "strike" (our own baseline forecast)
    entry_cost: float = 0.0002  # cost per unit of |change in weight| -- see realistic_entry_cost()
                                 # below for why this flat default is a placeholder, not a real number


def realistic_entry_cost(mean_period_var: float, vol_spread_points: float = 2.0,
                          periods_per_year: int = 365 * 24) -> float:
    """What a real market would actually charge to enter/exit this trade,
    derived from a bid-ask spread on *volatility* (the way variance/vol
    products are actually quoted), not guessed directly in variance units.

    A vol product's bid-ask is quoted in vol points (e.g. 55/57 implied
    vol). Converting a spread in vol to a spread in variance via
    d(var) = 2 * vol * d(vol), then back down from annualized to this
    series' own period length, gives a cost genuinely anchored to how
    these instruments trade -- `vol_spread_points` (in vol **percentage
    points**, e.g. 2.0 = a 2-point spread like "55/57") is the only
    judgment call, and it is a small, named one instead of an arbitrary
    constant buried in the P&L."""
    annualized_var = mean_period_var * periods_per_year
    annualized_vol = annualized_var ** 0.5
    d_vol = vol_spread_points / 100.0
    annualized_var_spread = 2 * annualized_vol * d_vol
    return annualized_var_spread / periods_per_year


def build_variance_series(ohlcv_csv: str, anomaly_csv: str, cfg: VarSwapConfig) -> pd.DataFrame:
    px = pd.read_csv(ohlcv_csv, parse_dates=["date"])
    px["date"] = pd.to_datetime(px["date"], utc=True).dt.tz_localize(None)
    px = px.sort_values("date").reset_index(drop=True)
    px["realized_var"] = parkinson_variance(px["high"], px["low"])
    px["log_return"] = np.log(px["close"]).diff()
    # The "strike": a naive trailing-average variance forecast, known at t-1 (no look-ahead) --
    # literally the simplest baseline forecast, standing in for a market-quoted implied variance.
    px["strike_var"] = px["realized_var"].rolling(cfg.strike_window).mean().shift(1)

    an = pd.read_csv(anomaly_csv, parse_dates=["date"])
    an["date"] = pd.to_datetime(an["date"], utc=True).dt.tz_localize(None)
    an = an[["date", "flag"]]

    df = px.merge(an, on="date", how="left").dropna(subset=["realized_var", "strike_var"])
    df["flag"] = df["flag"].fillna(0).astype(int)
    df["flag_lag"] = df["flag"].shift(1).fillna(0).astype(int)   # known at t-1, used for day t's weight
    # trailing-vol tercile, same convention as environment.py, fit on TRAIN only by the caller
    return df.reset_index(drop=True)


def fit_vol_terciles(df: pd.DataFrame) -> tuple[float, float]:
    q1, q2 = np.nanquantile(df["strike_var"], [1 / 3, 2 / 3])
    return float(q1), float(q2)


def rule_based_weights(df: pd.DataFrame, vol_terciles: tuple[float, float],
                        w_flagged: float = 1.0, w_calm: float = 0.0, w_mid: float = 0.0) -> np.ndarray:
    """The actual rule (no RL): go long variance when flagged (bet realized
    will surprise above our baseline); flat otherwise.

    `w_calm` defaults to 0, not negative: a first backtest tried shorting
    variance (harvesting the vol risk premium) in the calmest tercile, and
    it lost money at p=0.0025 -- short-vol strategies have known negative
    skew (small frequent wins, occasional large losses when "calm" still
    produces a surprise), and that showed up immediately even in the
    calmest tercile here. Rather than hand-tune it away, the honest move is
    to drop that leg: this now tests only the one thing actually validated
    -- does "long vol when flagged" have a real edge -- not a second,
    unvalidated hypothesis about selling calm-period vol."""
    q1, _ = vol_terciles
    calmest = df["strike_var"].shift(1) <= q1   # lagged, known at t-1
    w = np.where(df["flag_lag"] == 1, w_flagged, np.where(calmest, w_calm, w_mid))
    return w.astype(float)


def spot_rebalancing_cost(returns: np.ndarray, weight: np.ndarray, trading_fee_bps: float) -> np.ndarray:
    """The honest cost of the version of this trade actually buildable
    today with zero new data: no options, no variance swap -- replicate
    "long realized variance" by continuously rebalancing a spot/perp
    position (gamma scalping). That means a real trade *every active
    period*, not one at entry and one at exit: the standard discrete-
    hedging result is that the notional traded per period is proportional
    to that period's own price move, so the fee paid is
        cost_t = trading_fee_bps * |return_t|   (for each period the
                  position is active; 0 when flat)
    This is a structurally different, usually much larger, cost than
    `VarSwapConfig.entry_cost` (which only fires on a position *change* and
    is calibrated to a vol product's bid-ask spread, not a spot trading
    fee) -- the two are not interchangeable, and conflating them is exactly
    the mistake this module's cost history (see realistic_entry_cost) is
    trying to avoid repeating."""
    return trading_fee_bps * np.abs(returns) * (weight != 0)


def debounce_flag(flag_lag: np.ndarray, min_hold: int) -> np.ndarray:
    """Turn the raw (mostly single-bar) flag into a held position: once
    triggered, stay long for at least `min_hold` bars even after the flag
    itself clears, re-arming the hold clock on every new flag within that
    window. Same signal, same trigger -- this only changes how long each
    trade is held, directly targeting the problem the hourly backtest
    showed (isolated single-bar flags paying entry+exit cost almost every
    time). min_hold=1 reproduces the undebounced flag exactly."""
    held = np.zeros(len(flag_lag), dtype=float)
    remaining = 0
    for i, f in enumerate(flag_lag):
        if f:
            remaining = min_hold
        held[i] = 1.0 if remaining > 0 else 0.0
        remaining = max(0, remaining - 1)
    return held


def backtest(df: pd.DataFrame, weights: np.ndarray, cfg: VarSwapConfig) -> pd.DataFrame:
    """Cost model: a vol-product bid-ask spread (VarSwapConfig.entry_cost,
    see realistic_entry_cost) charged once per position change. Requires an
    actual options/variance-swap market -- see backtest_spot_rebalance for
    the alternative that's buildable with spot data alone today."""
    pnl = weights * (df["realized_var"].to_numpy() - df["strike_var"].to_numpy())
    turnover = np.abs(np.diff(weights, prepend=0.0))
    cost = cfg.entry_cost * turnover
    out = df[["date"]].copy()
    out["weight"] = weights
    out["pnl"] = pnl - cost
    out["gross_pnl"] = pnl
    return out


def backtest_spot_rebalance(df: pd.DataFrame, weights: np.ndarray, trading_fee_bps: float) -> pd.DataFrame:
    """Cost model: real spot/perp trading fees, charged every active period
    in proportion to that period's own price move (see
    spot_rebalancing_cost's docstring) -- the honest cost of the version of
    this trade buildable today with zero new data, no options market
    required."""
    returns = df["log_return"].to_numpy()
    pnl = weights * (df["realized_var"].to_numpy() - df["strike_var"].to_numpy())
    cost = spot_rebalancing_cost(returns, weights, trading_fee_bps)
    out = df[["date"]].copy()
    out["weight"] = weights
    out["pnl"] = pnl - cost
    out["gross_pnl"] = pnl
    return out
