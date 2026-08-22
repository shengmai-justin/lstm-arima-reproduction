"""Light CPU smoke tests: tiny walk-forward runs with few trials/epochs.

These validate wiring only (data -> WFO -> predictions -> backtest), NOT the
paper's numbers. Full-fidelity runs belong on the GPU server.
"""
import numpy as np
import pandas as pd
import pytest

from src import backtest, config, metrics, wfo


def _synthetic_features(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    """Random-walk closes with plausible volatility/volume features."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.01, n)
    close = 1000 * np.cumprod(1 + ret)
    vol = pd.Series(close).pct_change().pow(2).rolling(21).mean().pow(0.5).bfill() \
        * np.sqrt(252)
    volume = rng.uniform(1e9, 5e9, n)
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    return pd.DataFrame({"close": close, "volatility": vol.to_numpy(),
                         "volume": volume}, index=idx)


@pytest.fixture(scope="module")
def features():
    return _synthetic_features()


def test_signals_from_forecast_paper_rule():
    # F(t) stored at decision day t; position applies to day t+1.
    # Paper Sec 4.8: signal = 1 iff forecast > close at decision time.
    dates = pd.date_range("2020-01-01", periods=4)
    F = pd.Series([10.1, 10.2, 9.8, 9.9], index=dates)   # forecasts at t
    close = pd.Series([10.0, 10.0, 10.0, 10.0], index=dates)
    # positions for Jan2..Jan4: [1, 1, 0] (last decision day shifts out)
    assert list(backtest.signals_from_forecast(F, close, "long_only")) == [1, 1, 0]
    assert list(backtest.signals_from_forecast(F, close, "long_short")) == [1, 1, -1]


def test_forecast_rule_matches_model_selection_ir():
    """Backtest rule must equal the rule used inside WFO model selection."""
    rng = np.random.default_rng(3)
    n = 50
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    F = pd.Series(close.to_numpy() * (1 + rng.normal(0, 0.02, n)),
                  index=close.index)  # forecasts at decision days

    sig = backtest.signals_from_forecast(F, close, "long_short").to_numpy()
    wfo_pos = np.where(F.to_numpy()[:-1] > close.to_numpy()[:-1], 1.0, -1.0)
    np.testing.assert_allclose(sig, wfo_pos)


def test_strategy_returns_charge_costs_on_position_change():
    idx = pd.date_range("2020-01-01", periods=3)
    sig = pd.Series([1.0, 1.0, -1.0], index=idx)
    mkt = pd.Series([0.01, 0.02, -0.03], index=idx)
    rets = backtest.strategy_returns(sig, mkt, cost=0.001)
    # day1: 0.01 (open), day2: 0.02, day3: -1 * -0.03 - 0.001*2
    assert abs(rets.iloc[0] - (0.01 - 0.001)) < 1e-12
    assert abs(rets.iloc[1] - 0.02) < 1e-12
    assert abs(rets.iloc[2] - (0.03 - 0.002)) < 1e-12


def test_arima_wfo_smoke(features):
    out = wfo.run_arima_wfo(features, p_range=range(0, 3), q_range=range(0, 3),
                            criterion="aic", n_trials=3, max_walks=2)
    assert len(out.predictions) == 2 * config.OOS_DAYS
    assert out.predictions["pred"].notna().all()
    assert len(out.walks) == 2
    assert all(isinstance(w.chosen["order"], tuple) for w in out.walks)


@pytest.mark.filterwarnings("ignore:walk .*trials survived:RuntimeWarning")
def test_lstm_wfo_smoke(features, tmp_path):
    torch = pytest.importorskip("torch")
    out = wfo.run_lstm_wfo(features, hybrid=False, n_trials=2,
                           max_walks=1, device="cpu", seed=1, max_epochs=3)
    assert len(out.predictions) == config.OOS_DAYS
    assert np.isfinite(out.predictions["pred"]).all()
    assert out.walks[0].chosen["neurons"] > 0


@pytest.mark.filterwarnings("ignore:walk .*trials survived:RuntimeWarning")
def test_hybrid_wfo_smoke(features):
    pytest.importorskip("torch")
    out = wfo.run_lstm_wfo(features, hybrid=True, p_range=range(0, 2),
                           q_range=range(0, 2), n_trials=2, max_walks=1,
                           device="cpu", seed=2, max_epochs=2)
    assert len(out.predictions) == config.OOS_DAYS
    assert "order" in out.walks[0].chosen  # ARIMA info attached


def test_max_epochs_is_an_argument_not_module_state():
    """--epochs must reach joblib workers, which re-import config fresh."""
    import inspect

    from src import lstm
    assert "max_epochs" in inspect.signature(lstm.train_lstm).parameters
    assert "max_epochs" in inspect.signature(wfo.run_lstm_wfo).parameters


def test_end_to_end_backtest_from_predictions(features):
    out = wfo.run_arima_wfo(features, p_range=range(0, 2), q_range=range(0, 2),
                            n_trials=2, max_walks=1)
    pred_df = out.predictions  # decision-day indexed
    sig = backtest.signals_from_forecast(pred_df["pred"], features["close"],
                                         "long_short")
    market_ret = backtest.buy_and_hold_returns(features["close"]).reindex(sig.index)
    rets = backtest.strategy_returns(sig, market_ret)
    eq = backtest.equity_curve(rets)
    m = metrics.compute_all(rets, eq)
    assert set(m) == {"ARC(%)", "ASD(%)", "MD(%)", "MLD", "IR*(%)", "IR**(%)"}
    # earning days = the walk's OOS days exactly
    assert len(rets) == config.OOS_DAYS


def test_arima_walk_state_spans_the_validation_gap(features):
    """OOS forecasts must be conditioned on the days immediately before them.

    The IS window is train(1000) + validation(250); forecasting the OOS block
    straight off the training fit leaves the rolling state a year stale.
    """
    import warnings

    from src.arima import fit_arima, walk_predictions

    closes = features["close"].to_numpy()
    is_start = wfo.WARMUP_DAYS
    oos_start = is_start + config.IS_DAYS
    oos_end = oos_start + config.OOS_DAYS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_arima(closes[is_start:is_start + config.TRAIN_DAYS],
                        p_range=range(0, 3), q_range=range(0, 3), n_trials=4,
                        rng=np.random.default_rng(0))
        continuous, _ = walk_predictions(
            closes[is_start + config.TRAIN_DAYS:oos_end], fit, mode="rolling")
        out = wfo.run_arima_wfo(features, p_range=range(0, 3), q_range=range(0, 3),
                                n_trials=4, max_walks=1, forecast_mode="rolling")

    np.testing.assert_allclose(out.predictions["pred"].to_numpy(),
                               continuous[-config.OOS_DAYS:], rtol=1e-9)


def test_signals_cover_every_oos_day(features):
    """The final decision day must still earn; no OOS day may go untraded."""
    out = wfo.run_arima_wfo(features, p_range=range(0, 2), q_range=range(0, 2),
                            n_trials=2, max_walks=2)
    sig = backtest.signals_from_forecast(out.predictions["pred"],
                                         features["close"], "long_only")
    assert len(sig) == 2 * config.OOS_DAYS
    # earning days are exactly the day after each decision day
    decision = out.predictions.index
    assert sig.index[0] > decision[0] and sig.index[-1] > decision[-1]


@pytest.mark.filterwarnings("ignore:walk .*trials survived:RuntimeWarning")
def test_return_target_escapes_the_tanh_price_ceiling(features):
    """With target='level' the tanh head cannot emit a price above the training
    window's maximum, so a rising OOS block is mechanically short every day.
    Targeting the forward return removes the ceiling (the README's tanh-saturation section)."""
    pytest.importorskip("torch")
    closes = features["close"].to_numpy()
    train_max = closes[wfo.WARMUP_DAYS:wfo.WARMUP_DAYS + config.TRAIN_DAYS].max()

    lvl = wfo.run_lstm_wfo(features, n_trials=2, max_walks=1, device="cpu",
                           seed=0, max_epochs=2, target_mode="level")
    ret = wfo.run_lstm_wfo(features, n_trials=2, max_walks=1, device="cpu",
                           seed=0, max_epochs=2, target_mode="return")

    assert lvl.predictions["pred"].max() <= train_max + 1e-6
    assert ret.walks[0].chosen["target"] == "return"
    assert np.isfinite(ret.predictions["pred"]).all()
    # the return-mode forecast is anchored on each decision day's own close,
    # so it tracks the OOS level instead of being clamped to the training max
    assert ret.predictions["pred"].max() > lvl.predictions["pred"].max()


def test_unknown_target_mode_is_rejected(features):
    with pytest.raises(ValueError, match="unknown LSTM target mode"):
        wfo.run_lstm_wfo(features, n_trials=1, max_walks=1, device="cpu",
                         target_mode="logprice")
