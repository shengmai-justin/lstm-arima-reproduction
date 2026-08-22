"""Signal generation and backtesting (paper Sec. 4.8).

Unified forecast semantics: F(t) = price forecast made at the close of decision
day t (LSTM/hybrid: close at t+seq_len; ARIMA: close at t+1). Position for the
NEXT day: signal[t+1] = 1 if F(t) > P(t), else 0 (Long-Only) / -1 (Long-Short).

Low turnover follows naturally for LSTM/hybrid: F(t) targets close[t + seq_len],
so consecutive forecasts change slowly (turnover ~ 1/seq_len). ARIMA (h=1)
flips near-daily; with 0.1% costs per |position change| this is penalized
exactly as an implementable strategy would be (see gap G10 in the
README for the divergence from the paper's ARIMA row).

Transaction costs of 0.1% are charged on changes in position size.
"""
from __future__ import annotations

import pandas as pd

from . import config


def signals_from_forecast(forecast: pd.Series, close: pd.Series,
                          mode: str = "long_short") -> pd.Series:
    """Positions indexed by EARNING day t+1 from forecasts indexed by day t.

    forecast: F(t) stored at decision day t.
    close: full realized close series (provides P(t); index-aligned).
    """
    if mode not in {"long_only", "long_short"}:
        raise ValueError(f"unknown strategy mode: {mode}")
    cmp_close = close.reindex(forecast.index)
    pos = (forecast > cmp_close).astype(int)
    if mode == "long_short":
        pos = 2 * pos - 1
    # A missing forecast means "no opinion". Comparison with NaN is False, which
    # under long_short would otherwise become a maximum-conviction SHORT.
    pos = pos.where(forecast.notna() & cmp_close.notna(), 0)

    # Decision day t -> earning day t+1 on the asset's own trading calendar.
    # A positional shift within `forecast` would instead drop the position of
    # the LAST decision day, leaving the final OOS day of the run untraded.
    loc = close.index.get_indexer(forecast.index) + 1
    valid = (loc > 0) & (loc < len(close.index))  # -1 (unmatched) -> 0, filtered
    return pd.Series(pos.to_numpy()[valid], index=close.index[loc[valid]],
                     name=pos.name)


def strategy_returns(signal: pd.Series, market_ret: pd.Series,
                     cost: float = config.TRANSACTION_COST) -> pd.Series:
    """Daily net returns: position * market return - cost on position changes."""
    sig = signal.reindex(market_ret.index).fillna(0.0)
    trades = sig.diff().abs().fillna(sig.abs())
    return sig * market_ret - cost * trades


def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + returns).cumprod()


def buy_and_hold_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().fillna(0.0)
