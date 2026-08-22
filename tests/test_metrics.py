"""Unit tests for performance metrics vs hand-computed values and paper identities."""
import numpy as np
import pandas as pd

from src import metrics


def _series(values, start="2020-01-01"):
    return pd.Series(values,
                     index=pd.date_range(start, periods=len(values), freq="D"))


def test_arc_constant_growth():
    # 1bp/day for 252 days -> ~2.55% annualized
    r = np.full(252, 0.0001)
    assert abs(metrics.arc(_series(r)) - ((1.0001 ** 252) - 1) * 100) < 1e-6


def test_arc_zero_returns():
    assert metrics.arc(_series(np.zeros(100))) == 0.0


def test_asd_matches_formula():
    r = np.random.default_rng(0).normal(0.001, 0.01, 500)
    expected = np.sqrt(252) * np.std(r, ddof=1) * 100
    assert abs(metrics.asd(_series(r)) - expected) < 1e-9


def test_ir_star_identity_paper_table2_bh():
    # Paper Table 2, S&P 500 B&H: ARC=7.52, ASD=19.58 -> IR* = 38.43
    assert abs(7.52 / 19.58 * 100 - 38.39) < 0.1  # 38.43 reported (rounding)


def test_ir_star_star_identity_paper():
    # IR** = IR* x ARC x sign(ARC) / MD
    # Paper Table 2 B&H: 38.43 * 7.52 * 1 / 56.78 = 5.087 ~ 5.09
    assert abs(38.43 * 7.52 / 56.78 - 5.09) < 0.01
    # Paper Table 3 FTSE LSTM-ARIMA LS: 60.92 * 10.98 / 40.17 = 16.65
    assert abs(60.92 * 10.98 / 40.17 - 16.65) < 0.01


def test_max_drawdown():
    # equity 1 -> 1.2 -> 0.6 -> 0.9 -> 1.3: max DD = 50%
    eq = _series([1.0, 1.2, 0.6, 0.9, 1.3])
    assert abs(metrics.max_drawdown(eq) - 50.0) < 1e-9


def test_max_loss_duration():
    # daily index: new highs at t=0 and t=10 -> 10 days / 365.25 years
    eq = pd.Series([1.0, 0.9, 0.95, 0.99, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1],
                   index=pd.date_range("2020-01-01", periods=11, freq="D"))
    assert abs(metrics.max_loss_duration(eq) - 10 / 365.25) < 1e-9


def test_max_loss_duration_counts_unrecovered_tail():
    """A curve still underwater on the last day has been losing since its peak.

    Peak-to-peak gaps alone would report ~1 day here (new highs on every day of
    the initial climb) and ignore the 900-day decline that follows.
    """
    eq = pd.Series(np.r_[np.linspace(1.0, 2.0, 100), np.linspace(2.0, 1.0, 900)],
                   index=pd.date_range("2020-01-01", periods=1000, freq="D"))
    assert abs(metrics.max_loss_duration(eq) - 899 / 365.25) < 1e-6


def test_max_loss_duration_monotone_decline():
    # never makes a new high after day 0: the whole sample is one stretch
    eq = _series(np.linspace(1.0, 0.5, 51))
    assert abs(metrics.max_loss_duration(eq) - 50 / 365.25) < 1e-9


def test_modified_ir_zero_when_flat():
    r = _series(np.zeros(50))
    eq = _series(np.ones(50))
    assert metrics.modified_information_ratio(r, eq) == 0.0


def test_compute_all_keys():
    out = metrics.compute_all(_series(np.random.default_rng(1).normal(0, 0.01, 300)),
                              _series(np.linspace(1, 2, 300)))
    assert set(out) == {"ARC(%)", "ASD(%)", "MD(%)", "MLD", "IR*(%)", "IR**(%)"}
