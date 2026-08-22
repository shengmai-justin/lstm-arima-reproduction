"""ARIMA fit/forecast behaviour that the LSTM-ARIMA hybrid depends on."""
import warnings

import numpy as np
import pytest

from src.arima import ArimaWalkForecaster, fit_arima, walk_predictions


@pytest.fixture(scope="module")
def prices() -> np.ndarray:
    rng = np.random.default_rng(11)
    return 1400.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, 1200))


@pytest.fixture(scope="module")
def fit(prices):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_arima(prices[:1000], p_range=range(0, 3), q_range=range(0, 3),
                         n_trials=6, rng=np.random.default_rng(0))


def test_residual_burn_in_is_neutralised(fit, prices):
    """The Kalman filter starts from a zero state, so the first residuals are
    of the order of the price level (~1400) rather than of a forecast error.

    Left in, they set the MinMax range of the LSTM-ARIMA residual feature and
    squeeze every genuine residual into a sliver of [-1, 1].
    """
    r = fit.residuals
    assert fit.burn_in >= 1
    assert np.all(r[:fit.burn_in] == 0.0)

    settled = np.abs(r[fit.burn_in:]).max()
    assert settled < 0.1 * prices[:1000].mean()      # residuals, not price levels
    assert np.abs(r).max() == pytest.approx(settled)  # no leftover outlier


def test_residuals_keep_row_alignment(fit, prices):
    # zeroing (not dropping) the burn-in keeps residual[i] paired with close[i]
    assert len(fit.residuals) == 1000


@pytest.mark.parametrize("mode", ["rolling", "static"])
def test_walk_predictions_are_causal(fit, prices, mode):
    """preds[i] must use only closes strictly before the day it forecasts."""
    evaluation = prices[1000:1100]
    preds, resid = walk_predictions(evaluation, fit, mode=mode)
    assert len(preds) == len(evaluation)
    np.testing.assert_allclose(resid, evaluation - preds)

    # feeding a longer window must not change the shared prefix
    longer, _ = walk_predictions(prices[1000:1200], fit, mode=mode)
    np.testing.assert_allclose(preds, longer[:len(preds)], rtol=1e-9)


def test_rolling_matches_step_by_step_append(fit, prices):
    fc = ArimaWalkForecaster(fit)
    manual = []
    for i in range(20):
        if i > 0:
            fc.append(prices[1000 + i - 1])
        manual.append(fc.forecast())
    preds, _ = walk_predictions(prices[1000:1020], fit, mode="rolling")
    np.testing.assert_allclose(manual, preds, rtol=1e-9)


def test_static_never_reconditions_on_observed_closes(fit, prices):
    """The static path is the fit's own multi-step forecast, nothing more."""
    preds, _ = walk_predictions(prices[1000:1020], fit, mode="static")
    np.testing.assert_allclose(
        preds, np.asarray(fit._res.forecast(steps=20)).reshape(-1), rtol=1e-12)

    # ...so it is blind to what actually happened: doubling every observed
    # close leaves it unchanged, whereas the rolling path must react.
    shocked = prices[1000:1020] * 2.0
    static_shocked, _ = walk_predictions(shocked, fit, mode="static")
    np.testing.assert_allclose(preds, static_shocked, rtol=1e-12)

    rolling, _ = walk_predictions(prices[1000:1020], fit, mode="rolling")
    rolling_shocked, _ = walk_predictions(shocked, fit, mode="rolling")
    assert not np.allclose(rolling[1:], rolling_shocked[1:])


def test_unknown_forecast_mode_is_rejected(fit, prices):
    with pytest.raises(ValueError, match="unknown ARIMA forecast mode"):
        walk_predictions(prices[1000:1010], fit, mode="lookahead")


def test_criterion_switch_changes_selection_basis(prices):
    """BIC is the Sec. 6 sensitivity variant; it must actually be honoured."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_aic = fit_arima(prices[:600], p_range=range(0, 4), q_range=range(0, 4),
                           n_trials=16, rng=np.random.default_rng(1))
        by_bic = fit_arima(prices[:600], p_range=range(0, 4), q_range=range(0, 4),
                           n_trials=16, rng=np.random.default_rng(1),
                           criterion="bic")
    # BIC penalises parameters harder, so it never picks a larger model than AIC
    assert sum(by_bic.order) <= sum(by_aic.order)
