"""ARIMA model with random-search order selection by AIC (paper Sec. 4.9.1).

- Orders p, q in [0, 6], d = 1, selected by the lowest AIC on the training window.
- 20 random trials per walk (out of the 49 combinations).
- Predicts the next-day closing price.

Efficiency notes:
- The random search parallelizes across candidate orders with joblib (n_jobs).
- Stationarity/invertibility constraints follow the statsmodels default
  (config.ARIMA_ENFORCE = True). Disabling them is faster but changes which
  order the AIC search picks in 5 of 6 walks, always toward the largest
  available order -- see config.py.
- ArimaWalkForecaster reuses the already-fitted training result (no duplicate
  MLE refit); statsmodels append(refit=False) extends it in O(p+q) per day.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from . import config


@dataclass
class ArimaResult:
    order: tuple[int, int, int]
    aic: float
    fitted_values: np.ndarray   # in-sample one-step-ahead predictions
    residuals: np.ndarray       # in-sample residuals (actual - fitted), burn-in zeroed
    burn_in: int = 0            # leading residuals neutralised (Kalman warm-up)
    _res: object = None         # live statsmodels results (not pickled by dataclass repr)


def _burn_in_length(res, order: tuple[int, int, int]) -> int:
    """Number of leading residuals produced before the Kalman filter is warm.

    statsmodels initialises the state at zero, so the first residuals are of
    the order of the price *level* rather than of a forecast error (e.g. 1409
    against a genuine residual range of 145 for ^GSPC). Left in place they
    dominate the MinMax range of the LSTM-ARIMA residual feature and compress
    every real residual into a sliver of [-1, 1].
    """
    p, d, q = order
    return int(max(getattr(res, "loglikelihood_burn", 0) or 0, d + max(p, q)))


def _fit_one(train_close: np.ndarray, order: tuple, trend: str | None,
             enforce: bool = config.ARIMA_ENFORCE) -> tuple | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(
                train_close, order=order, trend=trend,
                enforce_stationarity=enforce, enforce_invertibility=enforce,
            ).fit()
        return order, res.aic, res.bic, res
    except Exception:
        return None


def fit_arima(train_close: np.ndarray,
              p_range=config.ARIMA_P_RANGE,
              q_range=config.ARIMA_Q_RANGE,
              d: int = config.ARIMA_D,
              criterion: str = config.ARIMA_CRITERION,
              n_trials: int = config.N_TRIALS,
              rng: np.random.Generator | None = None,
              n_jobs: int = 1,
              trend: str | None = config.ARIMA_TREND,
              enforce: bool = config.ARIMA_ENFORCE) -> ArimaResult:
    """Random search over (p, q); returns the best-criterion fit.

    trend: 'c' adds a drift to the differenced model (linear drift in levels).
    statsmodels' default for d>0 is no drift, which produces near-random
    one-day signals; the paper's persistent ARIMA positions (Tables 2-4: LS
    ASD ~= market ASD) indicate a drift term (e.g. pmdarima's default
    with_intercept=True). See README 'What the paper leaves unspecified' G9.
    """
    rng = rng or np.random.default_rng()
    combos = [(p, d, q) for p in p_range for q in q_range]
    if n_trials < len(combos):
        idx = rng.choice(len(combos), size=n_trials, replace=False)
        combos = [combos[i] for i in idx]

    if n_jobs != 1 and len(combos) > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=min(n_jobs, len(combos)))(
            delayed(_fit_one)(train_close, order, trend, enforce)
            for order in combos)
        results = [r for r in results if r is not None]
    else:
        results = [r for r in (_fit_one(train_close, o, trend, enforce)
                               for o in combos) if r is not None]

    if not results:  # fallback: random walk
        rw = ARIMA(train_close, order=(0, d, 0), trend=trend,
                   enforce_stationarity=enforce,
                   enforce_invertibility=enforce).fit()
        results = [((0, d, 0), rw.aic, rw.bic, rw)]

    crit_index = {"aic": 1, "bic": 2}[criterion]
    order, _, _, res = min(results, key=lambda r: r[crit_index])

    burn = _burn_in_length(res, order)
    residuals = np.asarray(res.resid, dtype=float).copy()
    residuals[:burn] = 0.0  # neutral value: keeps row alignment, no range blow-up

    return ArimaResult(
        order=order,
        aic=float(res.aic),
        fitted_values=np.asarray(res.fittedvalues),
        residuals=residuals,
        burn_in=burn,
        _res=res,
    )


class ArimaWalkForecaster:
    """Rolling one-step-ahead forecaster wrapping a fitted training result.

    statsmodels `append()` extends the fitted sample with newly observed closes
    (no re-estimation), giving fast sequential one-step forecasts for the
    validation and OOS windows.

    Call sequence per day t (in strict chronological order):
        pred_t = forecast()          # P_hat(t+1) using data observed so far
        ... day t+1 close observed ...
        append(close_{t+1})          # before the next forecast()
    """

    def __init__(self, fit: ArimaResult):
        self.order = fit.order
        self._res = fit._res  # reuse the search's fitted result (no refit)

    def append(self, observed_close: float) -> None:
        self._res = self._res.append([observed_close], refit=False)

    def forecast(self) -> float:
        """P_hat(t+1) given all observations appended so far."""
        fc = self._res.forecast(steps=1)
        return float(np.asarray(fc).reshape(-1)[0])

    @property
    def residuals(self) -> np.ndarray:
        """Residuals over everything observed so far (train + appended)."""
        return np.asarray(self._res.resid)


def walk_predictions(oos_close: np.ndarray, fit: ArimaResult,
                     mode: str = config.ARIMA_FORECAST_MODE
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Predictions over an evaluation window, no look-ahead.

    Returns (preds, residuals) of length len(oos_close), residuals[i] being
    oos_close[i] - preds[i].

    mode='rolling' (default): preds[i] = E[close_i | train + oos_close[:i]].
        Each day is forecast one step ahead from everything observed so far.
    mode='static': a single multi-step forecast path taken from the end of the
        training window, with no re-conditioning on observed closes. See the
        module docstring and gap G10 in the README for why the paper's
        ARIMA rows are only reachable under something like this reading.
    """
    if mode == "static":
        preds = np.asarray(
            fit._res.forecast(steps=len(oos_close)), dtype=float).reshape(-1)
        return preds, oos_close - preds
    if mode != "rolling":
        raise ValueError(f"unknown ARIMA forecast mode: {mode}")

    fc = ArimaWalkForecaster(fit)
    preds = np.empty(len(oos_close))
    for i in range(len(oos_close)):
        if i > 0:
            fc.append(oos_close[i - 1])
        preds[i] = fc.forecast()
    residuals = oos_close - preds
    return preds, residuals
