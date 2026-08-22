"""Constants and search spaces for the LSTM-ARIMA reproduction (arXiv:2406.18206)."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

INDICES = {
    "^GSPC": {"name": "S&P 500", "vol_source": "vix"},
    "^FTSE": {"name": "FTSE 100", "vol_source": "realized"},
    "^FCHI": {"name": "CAC 40", "vol_source": "realized"},
}

VOL_TICKER = "^VIX"
DATA_START = "2000-01-01"
DATA_END = "2023-08-31"

# --- Walk-forward optimization (paper Sec. 4.4) ---
TRAIN_DAYS = 1000
VAL_DAYS = 250
OOS_DAYS = 250
IS_DAYS = TRAIN_DAYS + VAL_DAYS

TRANSACTION_COST = 0.001  # 0.1% per unit of traded exposure (paper Sec. 5)

TRADING_DAYS = 252

# --- Random search (paper Sec. 4.9) ---
N_TRIALS = 20
TOP_K_VAL_LOSS = 5  # paper Sec. 4.7: 5 lowest validation-loss models

# --- ARIMA (paper Sec. 4.9.1) ---
ARIMA_P_RANGE = range(0, 7)   # 0..6
ARIMA_Q_RANGE = range(0, 7)   # 0..6
ARIMA_D = 1
ARIMA_CRITERION = "aic"       # "aic" (base) or "bic" (sensitivity)
# 't' = linear trend in levels == drift/constant in the differenced equation
# (statsmodels disallows trend='c' when d>0). The paper's persistent ARIMA
# positions (Tables 2-4: Long-Short ASD ~= market ASD, ARC ~= drifted B&H)
# imply a drift term was included (gap G9 in the README).
ARIMA_TREND = "t"

# How ARIMA produces the evaluation-window forecasts (gap G10 in the README).
# The paper (Sec. 4.3 step 5, Sec. 4.9.1) says only "fit the best model and
# execute predictions"; it never states whether the model is re-conditioned on
# observed closes as the OOS window advances.
#   "static"  - a single multi-step path from the end of the training window,
#               not re-conditioned as the window advances. Signals are then
#               "price below the extrapolated trend", which turns over slowly.
#               This is the default because it is the only reading that
#               reproduces the paper: it recovers the ARIMA Long-Only ASD
#               signature (^GSPC 14.83 vs paper 14.45, market 19.58) and moves
#               5 of 6 ARIMA IR** cells from wildly-off to close.
#   "rolling" - one-step-ahead, re-conditioned each day. The implementable
#               reading and the more defensible strategy, but signals flip on
#               ~50% of days, costing ~25%/yr at 0.1%, which drives every
#               ARIMA row deeply negative (^GSPC Long-Short IR** -17.2).
# The hybrid's residual feature always uses rolling regardless (see wfo.py).
ARIMA_FORECAST_MODE = "static"

# Whether ARIMA fits are constrained to stationary AR / invertible MA regions.
# statsmodels defaults to True and the paper never says otherwise. Turning it
# off was originally done for speed, but it is NOT free: measured over the
# first 6 ^GSPC walks, the two settings select the same (p,d,q) in only 1 of 6,
# and the unconstrained search systematically lands on the largest available
# order (3,1,3) -- it reaches lower AIC by wandering into exactly the
# non-stationary region the constraint exists to exclude. Keep True.
ARIMA_ENFORCE = True

# --- LSTM (paper Sec. 4.9.2) ---
LSTM_NEURONS = [25, 50, 75, 100, 250, 500]
LSTM_LAYERS = [1, 2]
LSTM_OPTIMIZERS = ["adam", "nadam", "adagrad"]
LSTM_LEARNING_RATES = [0.01, 0.0001]
LSTM_SEQ_LENGTHS = [7, 14, 21]
LSTM_DROPOUT = 0.075          # base case
LSTM_BATCH_SIZE = 32          # base case
LSTM_MAX_EPOCHS = 100
LSTM_PATIENCE = 10            # early stopping patience on validation loss
LSTM_LOSS = "mse"

# What the LSTM/hybrid regresses on.
#   "level"  - the close at t+seq_len, MinMax-scaled on the training window.
#              Faithful to Sec. 4.9.2, but the tanh head then cannot emit a
#              price above the training-window maximum, so the Sec. 4.8 rule
#              (F(t) > P(t)) is mechanically short on every OOS day where the
#              index has risen past that maximum -- 55.5% of ^GSPC OOS days,
#              and 100% of 7 of the 19 walks. See the README's tanh-saturation section.
#   "return" - the forward seq_len-day return. The signal becomes r_hat > 0,
#              which has no price ceiling. A DELIBERATE DEVIATION from the
#              paper, kept as a control so the saturation effect is measurable
#              rather than merely argued.
# Note the close FEATURE is still a scaled level either way; only the target
# changes. Feature extrapolation is far less harmful than output saturation.
LSTM_TARGET = "level"

# --- Sensitivity variants (paper Sec. 6) ---
SENSITIVITY_VARIANTS = {
    "base": {},
    "dropout_0.05": {"dropout": 0.05},
    "dropout_0.10": {"dropout": 0.10},
    "batch_16": {"batch_size": 16},
    "batch_64": {"batch_size": 64},
    "arima_orders_0_3": {"arima_p_range": range(0, 4), "arima_q_range": range(0, 4)},
    "arima_bic": {"arima_criterion": "bic"},
}

BASE_FEATURES = ["close", "volatility", "volume"]          # LSTM (paper Sec. 4.9.2)
HYBRID_FEATURES = BASE_FEATURES + ["arima_residual"]       # LSTM-ARIMA (paper Sec. 4.9.3)


def seed_everything(seed: int) -> None:
    import os
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
