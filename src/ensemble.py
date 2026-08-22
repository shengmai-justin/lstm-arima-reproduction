"""Ensembled AIS (paper Sec. 7): invest $1 per index, equal 1/3 weights.

The ensemble equity curve is the mean of the three per-index equity curves,
computed over the common trading period 2005-01-25 -> 2023-08-30 with 0.1%
transaction costs. Averaging cumulative curves gives BUY-AND-HOLD weights that
drift with performance (terminal 52.9 / 21.1 / 26.0%), which is what "invest
$1 in each" means -- it is not a rebalanced 1/3 portfolio.
"""
from __future__ import annotations

import pandas as pd

from . import config, metrics


def ensemble_equity(equity_curves: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight average of per-index equity curves on their common days.

    Averaging equity curves IS the paper's "$1 in each" portfolio: the mean of
    the three curves equals portfolio_value/3 exactly. Note that this leaves the
    weights drifting (buy-and-hold), not rebalanced.

    Restricting to the common trading days is deliberate. Forward-filling across
    each market's holidays leaves terminal equity bit-identical -- an equity
    curve is cumulative, so sampling it on fewer dates loses no performance --
    but it inflates the observation count from 4581 to 4804 over the same 18.59
    years. ARC/ASD annualize with the paper's hard-coded 252, so that inflation
    alone moved ensemble IR** by ~7% (2.61 -> 2.43) without a cent changing
    hands. See the README for the measured comparison.
    """
    df = pd.concat(equity_curves, axis=1).sort_index().dropna()
    if df.empty:
        raise ValueError("equity curves do not overlap")
    df = df / df.iloc[0]  # renormalize each to 1 at the common start
    return df.mean(axis=1)


def ensemble_metrics(equity: pd.Series) -> dict:
    returns = equity.pct_change().fillna(0.0)
    return metrics.compute_all(returns, equity)


def load_equity_curve(index: str, model: str, mode: str,
                      variant: str = "base",
                      results_dir=config.RESULTS_DIR) -> pd.Series:
    path = results_dir / index / f"{model}_{variant}" / f"equity_{mode}.csv"
    eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    eq.name = f"{model}_{index}"
    return eq


def load_buy_hold(index: str, models, variant: str = "base",
                  results_dir=config.RESULTS_DIR) -> pd.Series:
    """Buy&Hold curve for an index, from whichever run wrote one.

    B&H depends only on the index and the OOS window, so it is identical across
    models and variants. Reading it from a hard-coded model directory breaks
    whenever that model was not run for the requested variant -- e.g. the
    LSTM-only dropout/batch variants of Sec. 6, which have no ARIMA counterpart.
    """
    for v in dict.fromkeys([variant, "base"]):
        for model in models:
            path = results_dir / index / f"{model}_{v}" / "equity_buy_hold.csv"
            if path.exists():
                eq = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
                eq.name = f"buy_hold_{index}"
                return eq
    raise FileNotFoundError(
        f"no equity_buy_hold.csv for {index} under variant '{variant}' or 'base' "
        f"(looked in models: {list(models)})")
