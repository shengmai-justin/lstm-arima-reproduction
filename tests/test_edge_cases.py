"""Edge cases surfaced by the audit: silent-wrong-answer paths."""
import numpy as np
import pandas as pd
import pytest

from src import backtest, ensemble, metrics, stats_tests, wfo


def _s(values, start="2020-01-01"):
    return pd.Series(values,
                     index=pd.date_range(start, periods=len(values), freq="D"))


# --------------------------- metrics --------------------------- #
def test_undefined_metrics_are_nan_not_zero():
    """An empty sample must not be reported as a flat, riskless strategy."""
    empty = pd.Series(dtype=float)
    assert np.isnan(metrics.arc(empty))
    assert np.isnan(metrics.asd(empty))
    assert np.isnan(metrics.max_drawdown(empty))


def test_drawdown_free_strategy_is_not_ranked_below_losers():
    """IR** = IR* * ARC / MD is unbounded at MD == 0. Returning 0.0 put a
    strategy that never lost money below every losing one."""
    winner = _s(np.full(500, 0.001))
    eq = backtest.equity_curve(winner)
    assert metrics.max_drawdown(eq) == 0.0
    assert metrics.modified_information_ratio(winner, eq) == np.inf

    loser = _s(np.full(500, -0.001))
    loser_eq = backtest.equity_curve(loser)
    assert metrics.modified_information_ratio(loser, loser_eq) < 0
    # and a genuinely flat strategy is still 0, not inf
    flat = _s(np.zeros(50))
    assert metrics.modified_information_ratio(flat, _s(np.ones(50))) == 0.0


def test_max_loss_duration_on_positional_index():
    """np.diff on an int index yields ints; pd.Timedelta reads those as
    NANOSECONDS and silently reported ~0 years."""
    eq = pd.Series([1.0, 0.5, 0.6, 1.2])          # underwater for 3 periods
    assert metrics.max_loss_duration(eq) == pytest.approx(3 / 252)


# --------------------------- backtest --------------------------- #
@pytest.mark.parametrize("mode,expected", [("long_short", 0), ("long_only", 0)])
def test_nan_forecast_stands_aside_rather_than_shorting(mode, expected):
    """`NaN > close` is False, which under long_short became a full SHORT."""
    dates = pd.date_range("2020-01-01", periods=3)
    close = pd.Series([10.0, 10.0, 10.0], index=dates)
    F = pd.Series([np.nan, 11.0, 9.0], index=dates)
    sig = backtest.signals_from_forecast(F, close, mode)
    assert sig.iloc[0] == expected            # the NaN day
    assert sig.iloc[1] == 1                   # 11 > 10 -> long either way


# --------------------------- stats_tests --------------------------- #
def test_regression_alpha_drops_nans_like_the_paired_test():
    """One NaN made every output NaN and `significant_10pct` silently False."""
    rng = np.random.default_rng(0)
    bench = _s(rng.normal(0, 0.01, 300))
    strat = _s(bench.to_numpy() * 0.5 + rng.normal(0.0005, 0.005, 300))

    clean = stats_tests.regression_alpha(strat, bench)
    dirty_bench = bench.copy()
    dirty_bench.iloc[100] = np.nan
    dirty = stats_tests.regression_alpha(strat, dirty_bench)
    assert np.isfinite(dirty["alpha"]) and np.isfinite(dirty["p_alpha_one_sided"])
    assert abs(dirty["beta"] - clean["beta"]) < 0.05


def test_regression_alpha_rejects_degenerate_samples():
    with pytest.raises(ValueError, match=">=3 paired observations"):
        stats_tests.regression_alpha(_s([0.01, 0.02]), _s([0.01, 0.02]))


# --------------------------- ensemble --------------------------- #
def test_ensemble_mean_equals_one_dollar_in_each():
    """Averaging equity curves IS the $1-per-index portfolio, /3."""
    idx = pd.date_range("2005-01-25", periods=400, freq="B")
    rng = np.random.default_rng(3)
    curves = {f"i{j}": pd.Series(np.cumprod(1 + rng.normal(3e-4, 0.01, 400)),
                                 index=idx) for j in range(3)}
    eq = ensemble.ensemble_equity(curves)
    portfolio = sum(c / c.iloc[0] for c in curves.values())
    np.testing.assert_allclose(eq.to_numpy(), (portfolio / 3).to_numpy())


def test_ensemble_rejects_non_overlapping_curves():
    a = pd.Series([1.0, 1.1], index=pd.date_range("2020-01-01", periods=2))
    b = pd.Series([1.0, 1.1], index=pd.date_range("2021-01-01", periods=2))
    with pytest.raises(ValueError, match="do not overlap"):
        ensemble.ensemble_equity({"a": a, "b": b})


# --------------------------- selection --------------------------- #
def test_selection_fallback_uses_lowest_val_loss():
    """With every IR_val == 0, ranking by |IR_train - IR_val| collapses to
    |IR_train| and would hand the walk to the WORST training model."""
    cands = [
        {"val_loss": 0.10, "ir_val": 0.0, "ir_train": 50.0, "ir_diff": 50.0},
        {"val_loss": 0.12, "ir_val": 0.0, "ir_train": 3.0, "ir_diff": 3.0},
    ]
    assert wfo._select_best_trial(cands)["val_loss"] == 0.10


def test_selection_prefers_smallest_ir_gap_among_eligible():
    cands = [
        {"val_loss": 0.10, "ir_val": 5.0, "ir_train": 40.0, "ir_diff": 35.0},
        {"val_loss": 0.11, "ir_val": 8.0, "ir_train": 9.0, "ir_diff": 1.0},
        {"val_loss": 0.09, "ir_val": 0.0, "ir_train": 1.0, "ir_diff": 1.0},
    ]
    assert wfo._select_best_trial(cands)["ir_diff"] == 1.0
    assert wfo._select_best_trial(cands)["ir_val"] == 8.0


def test_max_loss_duration_is_zero_when_never_underwater():
    """Every observation of a flat or rising curve is a new high, so gaps
    between highs are just weekends -- a spurious 0.0082-year floor."""
    idx = pd.bdate_range("2020-01-01", periods=250)
    assert metrics.max_loss_duration(pd.Series(np.ones(250), index=idx)) == 0.0
    assert metrics.max_loss_duration(
        pd.Series(np.linspace(1, 2, 250), index=idx)) == 0.0
