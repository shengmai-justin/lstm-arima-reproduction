"""Performance metrics (paper Sec. 4.5).

All metrics are returned in percent units, matching the paper's tables.
Verified identities against Table 2 (S&P 500 B&H): ARC=7.52, ASD=19.58 ->
IR*=38.43, IR**=5.09.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def arc(returns: pd.Series) -> float:
    """Annualized return compounded (%)."""
    r = returns.dropna().to_numpy()
    if len(r) == 0:
        return float("nan")  # undefined, not "flat"
    growth = np.prod(1.0 + r)
    if growth <= 0:  # total wipe-out; annualization undefined
        return -100.0
    return (growth ** (config.TRADING_DAYS / len(r)) - 1.0) * 100.0


def asd(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns (%)."""
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        return float("nan")  # undefined, not "riskless"
    return float(np.sqrt(config.TRADING_DAYS) * np.std(r, ddof=1) * 100.0)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown of the equity curve (%)."""
    e = equity.dropna().to_numpy()
    if len(e) == 0:
        return float("nan")
    peaks = np.maximum.accumulate(e)
    dd = (peaks - e) / peaks
    return float(dd.max() * 100.0)


def max_loss_duration(equity: pd.Series) -> float:
    """Maximum loss duration (years): longest stretch between consecutive
    new highs of the equity curve (paper Sec. 4.5, Michańków et al. 2022).

    The end of the sample closes the final stretch: a curve still below its
    high-water mark on the last day has been underwater since that peak, and
    that stretch counts. Measuring only peak-to-peak gaps would report ~0 for
    a curve that peaks early and declines for the rest of the sample.
    """
    e = equity.dropna()
    if len(e) < 2:
        return 0.0

    # Longest stretch spent BELOW the high-water mark, measured peak-to-peak.
    # Working from the underwater days rather than from the gaps between new
    # highs matters: on a flat or monotonically rising curve every observation
    # is a new high, and the gaps between them are just weekends -- which gave
    # a spurious 0.0082-year floor to strategies that never lost anything.
    underwater = (e < e.cummax()).to_numpy()
    if not underwater.any():
        return 0.0

    idx = e.index.to_numpy()
    edges = np.diff(np.concatenate([[0], underwater.view(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)     # first underwater day of each run
    ends = np.flatnonzero(edges == -1) - 1  # last underwater day of each run
    # peak before the run -> peak that ends it (or the sample end, if the curve
    # never recovers: that stretch is still time spent underwater)
    prev_peak = idx[starts - 1]
    next_peak = idx[np.minimum(ends + 1, len(e) - 1)]
    gaps = next_peak - prev_peak

    if isinstance(e.index, pd.DatetimeIndex):
        return float(pd.Timedelta(gaps.max()).total_seconds() / 86400.0 / 365.25)
    # Positional index: gaps are observation counts. np.diff would otherwise
    # hand integers to pd.Timedelta, which reads them as NANOSECONDS and
    # silently reports ~0 years.
    return float(gaps.max()) / config.TRADING_DAYS


def information_ratio(returns: pd.Series) -> float:
    """IR* = ARC / ASD (%)."""
    a = asd(returns)
    if not a > 0:                      # includes NaN
        return float("nan") if np.isnan(a) else 0.0
    return arc(returns) / a * 100.0


def modified_information_ratio(returns: pd.Series, equity: pd.Series) -> float:
    """IR** = IR* * ARC * sign(ARC) / MD (%)."""
    ir = information_ratio(returns)
    a = arc(returns)
    md = max_drawdown(equity)
    if md == 0:
        # A strategy that never drew down has an unbounded IR**. Returning 0.0
        # ranked it BELOW every losing strategy in the paper's headline metric.
        return 0.0 if a == 0 else float(np.sign(a) * np.inf)
    return ir * a * np.sign(a) / md


def compute_all(returns: pd.Series, equity: pd.Series) -> dict:
    return {
        "ARC(%)": arc(returns),
        "ASD(%)": asd(returns),
        "MD(%)": max_drawdown(equity),
        "MLD": max_loss_duration(equity),
        "IR*(%)": information_ratio(returns),
        "IR**(%)": modified_information_ratio(returns, equity),
    }
