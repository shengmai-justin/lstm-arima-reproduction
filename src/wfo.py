"""Walk-forward optimization engine (paper Sec. 4.4) for ARIMA, LSTM and LSTM-ARIMA.

Per walk k (step = 250 days, non-anchored), with W = WARMUP_DAYS:
    IS   rows [W + k*250, W + k*250+1250) = train [0,1000) + validation [1000,1250)
    OOS  rows [W + k*250+1250, W + k*250+1500)

Random search of 20 trials per walk; best model per paper Sec. 4.7:
among the 5 lowest validation-loss trials, pick the one minimizing
|IR(train) - IR(val)| subject to IR(val) != 0 (fallback: lowest val loss).

ARIMA models are selected by lowest AIC instead (paper Sec. 4.9.1).
"""
from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, config
from .arima import fit_arima, walk_predictions
from .lstm import (FeatureScaler, MinMaxScaler1D, build_sequences,
                   predict_scaled, sample_trials, train_lstm)
from .metrics import information_ratio

WALK_STEP = config.OOS_DAYS  # 250
# The paper's OOS starts (2005-01-25 GSPC, 2005-01-13 FTSE, 2004-12-28 FCHI)
# correspond exactly to row 1250 + max(seq_len) on Yahoo data. Rows [0, 21) are
# simply DISCARDED to line the grid up with those dates -- they are not a
# lookback buffer: the first training window's lookback is drawn from inside the
# IS window itself (walk 0's first sequence starts at is_start).
WARMUP_DAYS = max(config.LSTM_SEQ_LENGTHS)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ir_from_forecast(F: np.ndarray, decision_closes: np.ndarray) -> float:
    """IR* of the paper's rule using forecasts indexed by decision day.

    F has n entries (decision days d0..d_{n-1}); decision_closes holds the n+1
    closes of those days PLUS the next day (for the earned return).
    position for day d_i+1 = [F[i] > close(d_i)]; net returns include costs.
    """
    pos = np.where(F > decision_closes[:-1], 1.0, -1.0)
    rets = np.diff(decision_closes) / decision_closes[:-1]
    trades = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * rets - config.TRANSACTION_COST * trades
    if len(net) == 0:
        return 0.0
    return information_ratio(pd.Series(net))


def _to_price(scaled_pred: np.ndarray, scaler, decision_closes: np.ndarray,
              target_mode: str) -> np.ndarray:
    """Model output -> a price forecast comparable to P(t) under Sec. 4.8.

    Under "return" the network predicts the forward L-day return, so the price
    forecast is P(t)*(1+r_hat) and the signal F(t) > P(t) reduces to r_hat > 0.
    """
    out = scaler.inverse_transform(scaled_pred)
    if target_mode == "return":
        return decision_closes * (1.0 + out)
    return out


@dataclass
class WalkInfo:
    walk: int
    start: str
    end: str
    chosen: dict = field(default_factory=dict)
    n_trials: int = 0
    top5: list = field(default_factory=list)


@dataclass
class WfoOutput:
    predictions: pd.DataFrame   # columns: pred, close (OOS, all walks concatenated)
    walks: list  # WalkInfo


def _oos_slices(n_rows: int, max_walks: int | None = None):
    """(k, is_start, oos_start, oos_end) per walk; trailing partial OOS included.

    Walk k: IS rows [WARMUP + k*250, WARMUP + k*250 + 1250),
            OOS rows [WARMUP + k*250 + 1250, WARMUP + k*250 + 1500).
    """
    k = 0
    while WARMUP_DAYS + k * WALK_STEP + config.IS_DAYS < n_rows:
        is_start = WARMUP_DAYS + k * WALK_STEP
        oos_start = is_start + config.IS_DAYS
        oos_end = min(oos_start + config.OOS_DAYS, n_rows)
        yield k, is_start, oos_start, oos_end
        k += 1
        if max_walks is not None and k >= max_walks:
            break


def _select_best_trial(candidates: list[dict]) -> dict:
    """Paper Sec. 4.7: top-5 by val loss, min |IR_train - IR_val|, IR_val != 0.

    If no candidate clears the IR_val != 0 guard, fall back to the lowest
    validation loss. Ranking the ineligible pool by |IR_train - IR_val| instead
    would reduce to |IR_train| and hand the walk to the trial with the *worst*
    training performance.
    """
    top5 = sorted(candidates, key=lambda c: c["val_loss"])[:config.TOP_K_VAL_LOSS]
    eligible = [c for c in top5 if c["ir_val"] != 0.0]
    if not eligible:
        return top5[0]
    return min(eligible, key=lambda c: c["ir_diff"])


# --------------------------------------------------------------------------- #
# ARIMA walk-forward
# --------------------------------------------------------------------------- #
def run_arima_wfo(features: pd.DataFrame, *,
                  p_range=config.ARIMA_P_RANGE, q_range=config.ARIMA_Q_RANGE,
                  criterion: str = config.ARIMA_CRITERION,
                  n_trials: int = config.N_TRIALS, max_walks: int | None = None,
                  seed: int = 0, n_jobs: int = 1,
                  search_jobs: int = 1,
                  forecast_mode: str = config.ARIMA_FORECAST_MODE) -> WfoOutput:
    closes = features["close"].to_numpy()
    dates = features.index

    def _one_walk(k, is_start, oos_start, oos_end):
        # avoid oversubscription: walks already parallel when n_jobs > 1
        inner = 1 if n_jobs > 1 else search_jobs
        train = closes[is_start:is_start + config.TRAIN_DAYS]
        rng = np.random.default_rng(seed + k)
        fit = fit_arima(train, p_range=p_range, q_range=q_range,
                        criterion=criterion, n_trials=n_trials, rng=rng,
                        n_jobs=inner)

        # One-step forecasts for OOS targets [oos_start, oos_end), each made at
        # the previous close: decision days [oos_start-1, oos_end-2).
        # The rolling state must be carried through the 250 validation days that
        # sit between the training window and the OOS block, otherwise every
        # walk's first forecast is conditioned on data a year stale (~2% off on
        # ^GSPC). This mirrors _arima_residuals_per_walk on the hybrid path.
        n_oos = oos_end - oos_start
        preds, _ = walk_predictions(closes[is_start + config.TRAIN_DAYS:oos_end],
                                    fit, mode=forecast_mode)
        preds = preds[-n_oos:]
        info = WalkInfo(
            walk=k, start=str(dates[oos_start].date()), end=str(dates[oos_end - 1].date()),
            chosen={"order": fit.order, "aic": fit.aic, "forecast_mode": forecast_mode},
            n_trials=n_trials)
        return (k, pd.DataFrame(
            {"pred": preds, "close": closes[oos_start - 1:oos_end - 1]},
            index=dates[oos_start - 1:oos_end - 1]), info)  # decision-day index

    return _gather(_one_walk, features, max_walks, n_jobs)


# --------------------------------------------------------------------------- #
# LSTM / LSTM-ARIMA walk-forward
# --------------------------------------------------------------------------- #
def _arima_residuals_per_walk(closes: np.ndarray, is_start: int, oos_end: int,
                              p_range, q_range, criterion: str,
                              n_trials: int, rng,
                              search_jobs: int = 1) -> tuple[np.ndarray, dict]:
    """Residuals of the walk's best-AIC ARIMA for rows [is_start, oos_end).

    Train residuals come from the in-sample fit; val/OOS residuals from rolling
    one-step forecasts (each residual is known the moment its day's close is
    observed, so using it as a next-day feature leaks no information).
    """
    train = closes[is_start:is_start + config.TRAIN_DAYS]
    fit = fit_arima(train, p_range=p_range, q_range=q_range,
                    criterion=criterion, n_trials=n_trials, rng=rng,
                    n_jobs=search_jobs)
    train_resid = fit.residuals[-len(train):]

    # Always rolling here: the hybrid's 4th feature is meant to be a one-step
    # forecast ERROR. A static path would instead accumulate horizon drift and
    # carry no day-to-day information for the LSTM to exploit.
    _, ev_resid = walk_predictions(closes[is_start + config.TRAIN_DAYS:oos_end],
                                   fit, mode="rolling")

    return np.concatenate([train_resid, ev_resid]), {"order": fit.order, "aic": fit.aic}


def run_lstm_wfo(features: pd.DataFrame, *, hybrid: bool = False,
                 p_range=config.ARIMA_P_RANGE, q_range=config.ARIMA_Q_RANGE,
                 arima_criterion: str = config.ARIMA_CRITERION,
                 dropout: float = config.LSTM_DROPOUT,
                 batch_size: int = config.LSTM_BATCH_SIZE,
                 n_trials: int = config.N_TRIALS, max_walks: int | None = None,
                 device: str | None = None, seed: int = 0,
                 n_jobs: int = 1, search_jobs: int = 1, amp: bool = False,
                 verbose: bool = False,
                 max_epochs: int | None = None,
                 target_mode: str = config.LSTM_TARGET) -> WfoOutput:
    if target_mode not in {"level", "return"}:
        raise ValueError(f"unknown LSTM target mode: {target_mode}")
    closes = features["close"].to_numpy()
    dates = features.index
    base_raw = features[config.BASE_FEATURES].to_numpy(dtype=float)
    n_cols = len(config.HYBRID_FEATURES) if hybrid else len(config.BASE_FEATURES)

    def _one_walk(k, is_start, oos_start, oos_end):
        rng = np.random.default_rng(seed + k)
        # The ARIMA order search gets its OWN generator. Sharing `rng` would let
        # the hybrid's fit_arima consume draws before sample_trials runs, so the
        # hybrid and the plain LSTM would explore disjoint corners of the same
        # 216-point grid (measured config overlap 0.103, versus 0.093 expected
        # from two independent 20-of-216 draws -- i.e. no more than chance).
        # The paper's headline claim is LSTM-ARIMA > LSTM; with a shared rng
        # that comparison confounds the residual feature with search luck.
        arima_rng = np.random.default_rng(seed + k + 1_000_000)
        # avoid oversubscription: walks already parallel when n_jobs > 1
        inner = 1 if n_jobs > 1 else search_jobs
        extra = {}

        # Per-walk frame in walk-relative coordinates: rows [0, 1500) map to
        # absolute rows [is_start, oos_end). train=[0,1000), val=[1000,1250),
        # OOS=[1250, oos_end-is_start).
        walk_len = oos_end - is_start
        if hybrid:
            resid, arima_info = _arima_residuals_per_walk(
                closes, is_start, oos_end, p_range, q_range, arima_criterion,
                n_trials, arima_rng, search_jobs=inner)
            raw = np.column_stack([base_raw[is_start:oos_end], resid])
            extra["arima"] = arima_info
        else:
            raw = base_raw[is_start:oos_end].copy()
        assert raw.shape == (walk_len, n_cols)

        walk_closes = closes[is_start:oos_end]
        tr_lo, tr_hi = 0, config.TRAIN_DAYS
        va_lo, va_hi = tr_hi, config.IS_DAYS
        oos_lo, oos_hi = config.IS_DAYS, walk_len

        # MinMax scaling fit on the training window only (gap G1 in the README);
        # validation/OOS rows (including lookback context reaching into IS)
        # are transformed with the train-fitted scaler.
        scaler = FeatureScaler()
        full_scaled = np.empty_like(raw, dtype=np.float32)
        full_scaled[tr_lo:tr_hi] = scaler.fit_transform(raw[tr_lo:tr_hi])
        mask = np.ones(walk_len, dtype=bool)
        mask[tr_lo:tr_hi] = False
        full_scaled[mask] = scaler.transform(raw[mask])
        target_scaler = scaler.target_scaler

        def _decision_closes(lo, hi, L):
            """Closes of the decision days a build_sequences block produces,
            plus the next day (the one whose return is earned)."""
            return walk_closes[lo:hi - L + 1]

        candidates = []
        trials = sample_trials(rng, n_trials, dropout=dropout,
                               batch_size=batch_size)
        for t, trial in enumerate(trials):
            L = trial.seq_len  # lookback AND forecast horizon (paper Sec 4.9.2)
            # Target series in walk-relative coordinates. Under "return" it is
            # the BACKWARD L-day return at index k, so that build_sequences'
            # horizon=L lands on the FORWARD return from each decision day --
            # every window/index relationship below stays identical to "level".
            if target_mode == "return":
                fwd = np.full(walk_len, np.nan)
                fwd[L:] = walk_closes[L:] / walk_closes[:-L] - 1.0
                tgt_scaler = MinMaxScaler1D().fit(fwd[tr_lo + L:tr_hi])
                tgt = tgt_scaler.transform(fwd)
            else:
                tgt_scaler = target_scaler
                tgt = full_scaled[:, 0]

            try:
                # Supervised pairs: window [i, i+L) ending at decision day
                # t=i+L-1, target at t+L. Val windows may reach back into
                # train rows for context (features only, no leakage); val
                # targets stay inside the validation block.
                X_tr, y_tr = build_sequences(full_scaled[tr_lo:tr_hi, :],
                                             tgt[tr_lo:tr_hi], L, horizon=L)
                X_va, y_va = build_sequences(full_scaled[va_lo - L + 1:va_hi, :],
                                             tgt[va_lo - L + 1:va_hi],
                                             L, horizon=L)

                model, val_loss, tr_pred, va_pred = train_lstm(
                    X_tr, y_tr, X_va, y_va, trial, seed=seed * 1000 + k * 100 + t,
                    device=device, amp=amp, max_epochs=max_epochs)
            except Exception as exc:  # skip degenerate trials
                if verbose:
                    print(f"walk {k} trial {t} failed: {exc}")
                continue

            # Forecasts at consecutive decision days; compare F(t) > close(t),
            # earn t+1 (costs applied) - same rule as final evaluation.
            tr_dc = _decision_closes(L - 1, tr_hi, L)
            va_dc = _decision_closes(va_lo, va_hi, L)
            tr_price = _to_price(tr_pred, tgt_scaler, tr_dc[:-1], target_mode)
            va_price = _to_price(va_pred, tgt_scaler, va_dc[:-1], target_mode)
            ir_tr = _ir_from_forecast(tr_price, tr_dc)
            ir_va = _ir_from_forecast(va_price, va_dc)

            candidates.append({"trial": trial, "model": model, "val_loss": val_loss,
                               "ir_train": ir_tr, "ir_val": ir_va,
                               "ir_diff": abs(ir_tr - ir_va), "seq_len": L,
                               "tgt_scaler": tgt_scaler})

        if not candidates:
            raise RuntimeError(f"walk {k}: all trials failed")
        if len(candidates) < config.TOP_K_VAL_LOSS:
            # Sec. 4.7 ranks the 5 lowest-validation-loss trials. With fewer
            # survivors that selection is not really applied, and a partial
            # CUDA OOM across a long run would otherwise degrade some walks
            # silently -- walks.csv records n_trials but nothing flags it.
            warnings.warn(
                f"walk {k}: only {len(candidates)}/{len(trials)} trials "
                f"survived; Sec. 4.7 needs {config.TOP_K_VAL_LOSS}",
                RuntimeWarning, stacklevel=2)
        best = _select_best_trial(candidates)

        # Release non-best trial models (they may hold accelerator memory).
        for c in candidates:
            if c is not best:
                c["model"] = None

        # OOS forecasts for decision days [oos_lo-1, oos_hi-2] -> windows
        # starting at [oos_lo-L, oos_hi-L-1]; each earns the next day's return,
        # covering exactly this walk's OOS days [oos_lo, oos_hi-1].
        L = best["seq_len"]
        X_oo = np.lib.stride_tricks.sliding_window_view(
            full_scaled[oos_lo - L:oos_hi - 1, :],
            (L, full_scaled.shape[1]))[:, 0]
        X_oo = np.ascontiguousarray(X_oo, dtype=np.float32)
        oos_dc = walk_closes[oos_lo - 1:oos_hi - 1]   # decision-day closes
        oos_pred = _to_price(predict_scaled(best["model"], X_oo),
                             best["tgt_scaler"], oos_dc, target_mode)
        best["model"] = None  # walk output keeps only predictions, not weights

        chosen = {kk: getattr(best["trial"], kk) for kk in
                  ("neurons", "layers", "optimizer", "learning_rate", "seq_len",
                   "dropout", "batch_size")}
        chosen["target"] = target_mode
        chosen.update(extra.get("arima", {}))
        info = WalkInfo(
            walk=k, start=str(dates[oos_start].date()), end=str(dates[oos_end - 1].date()),
            chosen=chosen, n_trials=len(candidates),
            top5=[{"val_loss": c["val_loss"], "ir_train": c["ir_train"],
                   "ir_val": c["ir_val"], "ir_diff": c["ir_diff"],
                   "params": {kk: getattr(c["trial"], kk) for kk in
                              ("neurons", "layers", "optimizer", "learning_rate",
                               "seq_len")}} for c in
                 sorted(candidates, key=lambda c: c["val_loss"])[:config.TOP_K_VAL_LOSS]])
        if verbose:
            print(f"walk {k} [{info.start} -> {info.end}] chosen={chosen}")
        return (k, pd.DataFrame(
            {"pred": oos_pred, "close": walk_closes[oos_lo - 1:oos_hi - 1]},
            index=dates[oos_start - 1:oos_end - 1]), info)  # decision-day index

    return _gather(_one_walk, features, max_walks, n_jobs)


def _gather(_one_walk, features: pd.DataFrame, max_walks, n_jobs: int) -> WfoOutput:
    """Run walks (sequentially or in parallel worker processes) and merge.

    n_jobs > 1 parallelizes whole walks across processes — each walk is an
    independent data window. Use for CPU runs; keep 1 for single-GPU LSTM runs
    (one process owns the GPU) unless the device is CPU. Workers pin torch to a
    single thread to avoid oversubscription.
    """
    slices = list(_oos_slices(len(features), max_walks))

    if n_jobs != 1 and len(slices) > 1:
        from joblib import Parallel, delayed

        def _worker(*s):
            try:
                import torch

                torch.set_num_threads(1)
            except ImportError:
                pass
            return _one_walk(*s)

        results = Parallel(n_jobs=min(n_jobs, len(slices)))(
            delayed(_worker)(*s) for s in slices)
    else:
        results = [_one_walk(*s) for s in slices]

    results.sort(key=lambda r: r[0])
    return WfoOutput(pd.concat(r[1] for r in results), [r[2] for r in results])


# --------------------------------------------------------------------------- #
# shared entry point
# --------------------------------------------------------------------------- #
def run_wfo(features: pd.DataFrame, model: str, variant: str = "base",
            **kwargs) -> WfoOutput:
    variant_cfg = config.SENSITIVITY_VARIANTS.get(variant, {})

    if model == "arima":
        fn, extra = run_arima_wfo, {}
    elif model == "lstm":
        fn, extra = run_lstm_wfo, {"hybrid": False}
    elif model == "hybrid":
        fn, extra = run_lstm_wfo, {"hybrid": True}
    else:
        raise ValueError(f"unknown model: {model}")

    merged = {**_defaults_for(model, variant_cfg), **kwargs}
    accepted = set(inspect.signature(fn).parameters) | set(extra)
    merged = {kk: vv for kk, vv in merged.items() if kk in accepted}
    return fn(features, **extra, **merged)


def _defaults_for(model: str, variant_cfg: dict) -> dict:
    if model == "arima":
        return dict(
            p_range=variant_cfg.get("arima_p_range", config.ARIMA_P_RANGE),
            q_range=variant_cfg.get("arima_q_range", config.ARIMA_Q_RANGE),
            criterion=variant_cfg.get("arima_criterion", config.ARIMA_CRITERION),
        )
    return dict(
        p_range=variant_cfg.get("arima_p_range", config.ARIMA_P_RANGE),
        q_range=variant_cfg.get("arima_q_range", config.ARIMA_Q_RANGE),
        arima_criterion=variant_cfg.get("arima_criterion", config.ARIMA_CRITERION),
        dropout=variant_cfg.get("dropout", config.LSTM_DROPOUT),
        batch_size=variant_cfg.get("batch_size", config.LSTM_BATCH_SIZE),
    )
