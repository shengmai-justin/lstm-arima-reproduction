"""PyTorch LSTM model and trainer (paper Sec. 4.9.2, TensorFlow -> PyTorch port).

Architecture (paper): stacked LSTM (1-2 layers, sigmoid gate activations are the
PyTorch default), dropout between LSTM layers, linear head with tanh output
activation. Features/target are MinMax-scaled to [-1, 1] per walk to make the
tanh output usable (gap G1 in the README); predictions are inverse-transformed.

Training (paper): MSE loss, batch size 32, max 100 epochs, early stopping on
validation loss with patience 10. Optimizer/learning rate from the random search.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import config


@dataclass
class TrialSpace:
    """Sampled hyperparameter set for one random-search trial (paper Sec. 4.9.2)."""
    neurons: int
    layers: int
    optimizer: str
    learning_rate: float
    seq_len: int
    dropout: float = config.LSTM_DROPOUT
    batch_size: int = config.LSTM_BATCH_SIZE

    @classmethod
    def sample(cls, rng: np.random.Generator, **overrides) -> "TrialSpace":
        defaults = dict(
            neurons=int(rng.choice(config.LSTM_NEURONS)),
            layers=int(rng.choice(config.LSTM_LAYERS)),
            optimizer=str(rng.choice(config.LSTM_OPTIMIZERS)),
            learning_rate=float(rng.choice(config.LSTM_LEARNING_RATES)),
            seq_len=int(rng.choice(config.LSTM_SEQ_LENGTHS)),
        )
        defaults.update(overrides)
        return cls(**defaults)


class LSTMRegressor(nn.Module):
    """Paper Sec. 4.9.2: 1-2 LSTM layers, dropout, linear head with tanh output.

    Gate activations are sigmoid and the change gate is tanh, which is what the
    paper's own Eqs. 9-12 specify and what PyTorch's nn.LSTM hardcodes.

    Dropout needs care. Keras' `LSTM(dropout=r)` -- what the paper used -- drops
    the layer's *input* connections with a mask held constant across timesteps,
    and applies to a single-layer model just as much as to a stacked one.
    PyTorch's `nn.LSTM(dropout=r)` instead drops only *between* stacked layers
    and is silently a no-op when num_layers == 1. Since the search samples
    layers from {1, 2}, relying on nn.LSTM alone would leave ~50% of trials with
    no dropout at all -- and would make the Sec. 6 dropout sensitivity panels
    (0.05 / 0.075 / 0.1, i.e. RQ3) meaningless for half the models.
    """

    def __init__(self, n_features: int, neurons: int, layers: int, dropout: float):
        super().__init__()
        self.dropout_p = float(dropout)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=neurons,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,  # between stacked layers
        )
        self.head = nn.Linear(neurons, 1)
        self._apply_unit_forget_bias()

    def _apply_unit_forget_bias(self) -> None:
        """Keras' LSTM sets the forget-gate bias to 1 (`unit_forget_bias=True`,
        its default); PyTorch initialises every bias uniformly near 0.

        A forget gate starting at sigmoid(0)=0.5 instead of sigmoid(1)=0.73
        discards cell state faster early in training, which works against the
        long-memory behaviour the paper leans on. PyTorch keeps two
        bias vectors that are summed, and the gate order is (i, f, g, o), so the
        1.0 goes in the input-hidden half and the hidden-hidden half stays 0.
        """
        h = self.lstm.hidden_size
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if name.startswith("bias_ih"):
                    param[h:2 * h].fill_(1.0)
                elif name.startswith("bias_hh"):
                    param[h:2 * h].fill_(0.0)

    def _input_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Keras-style variational dropout on the input sequence.

        One mask per (sample, feature), reused at every timestep.
        """
        if not self.training or self.dropout_p <= 0.0:
            return x
        keep = 1.0 - self.dropout_p
        mask = torch.empty(x.shape[0], 1, x.shape[2],
                           device=x.device, dtype=x.dtype)
        mask.bernoulli_(keep).div_(keep)
        return x * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(self._input_dropout(x))
        return torch.tanh(self.head(out[:, -1, :]))


SEARCH_KEY_FIELDS = ("neurons", "layers", "optimizer", "learning_rate", "seq_len")


def grid_size() -> int:
    """Number of distinct hyperparameter combinations in the Sec. 4.9.2 space."""
    return (len(config.LSTM_NEURONS) * len(config.LSTM_LAYERS)
            * len(config.LSTM_OPTIMIZERS) * len(config.LSTM_LEARNING_RATES)
            * len(config.LSTM_SEQ_LENGTHS))


def sample_trials(rng: np.random.Generator, n_trials: int,
                  **overrides) -> list[TrialSpace]:
    """`n_trials` DISTINCT hyperparameter sets (paper Sec. 4.9: "we conduct 20
    trials on a randomly chosen set").

    Drawing each field independently samples the 216-point grid *with*
    replacement, so 56.5% of walks waste at least one trial re-training a
    configuration already tried (0.80 of 20 on average). sklearn's
    RandomizedSearchCV and Keras Tuner both de-duplicate over a finite grid,
    and `fit_arima` in this repo already uses `replace=False`.

    `overrides` (dropout, batch_size) are fixed by the paper rather than
    searched, so they are excluded from the uniqueness key.
    """
    wanted = min(n_trials, grid_size())
    trials: list[TrialSpace] = []
    seen: set[tuple] = set()
    for _ in range(wanted * 200):
        if len(trials) == wanted:
            break
        trial = TrialSpace.sample(rng, **overrides)
        key = tuple(getattr(trial, f) for f in SEARCH_KEY_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        trials.append(trial)
    return trials


# Keras uses epsilon=1e-7 for Adam/Nadam/Adagrad; PyTorch defaults to 1e-8
# (1e-10 for Adagrad). The paper ran on TensorFlow, so match Keras.
KERAS_EPS = 1e-7
# Keras' Adagrad starts its accumulator at 0.1; PyTorch starts at 0, which makes
# the very first update lr*sign(grad) for every weight -- a full 0.01 step at the
# paper's higher learning rate.
KERAS_ADAGRAD_ACCUM = 0.1


def _make_optimizer(name: str, params, lr: float):
    """Optimizers from paper Sec. 4.9.2, with Keras' defaults rather than
    PyTorch's for the hyperparameters the paper does not pin down."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, eps=KERAS_EPS)
    if name == "nadam":
        return torch.optim.NAdam(params, lr=lr, eps=KERAS_EPS)
    if name == "adagrad":
        return torch.optim.Adagrad(
            params, lr=lr, eps=KERAS_EPS,
            initial_accumulator_value=KERAS_ADAGRAD_ACCUM)
    raise ValueError(f"unknown optimizer: {name}")


def build_sequences(features: np.ndarray, target: np.ndarray, seq_len: int,
                    horizon: int = 1):
    """Sliding-window supervised pairs with a configurable forecast horizon.

    X[i] = features[i : i+seq_len] (window ends at day i+seq_len-1 = decision
    day t), y[i] = target[t + horizon]. The paper (Sec. 4.9.2/4.9.3) predicts
    the closing price at t + seq_len, i.e. horizon = seq_len for LSTM/hybrid.
    """
    h = horizon
    n = len(features) - seq_len - h + 1
    if n <= 0:
        raise ValueError("not enough data for the requested sequence length")
    X = np.lib.stride_tricks.sliding_window_view(features, (seq_len, features.shape[1]))
    X = np.ascontiguousarray(X[:n, 0])  # (n, seq_len, n_features)
    y = target[seq_len - 1 + h : seq_len - 1 + h + n]
    return X.astype(np.float32), y.astype(np.float32)


class MinMaxScaler1D:
    """Scale to [-1, 1] using min/max of the fit sample (gap G1)."""

    def __init__(self):
        self.lo = self.hi = None

    def fit(self, x: np.ndarray) -> "MinMaxScaler1D":
        self.lo, self.hi = float(np.nanmin(x)), float(np.nanmax(x))
        if self.hi == self.lo:
            self.hi = self.lo + 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x + 1.0) * (self.hi - self.lo) / 2.0 + self.lo


class FeatureScaler:
    """Per-column MinMax scaling of a 2-D feature matrix, fit on train only."""

    def __init__(self):
        self.scalers: list[MinMaxScaler1D] = []

    def fit(self, feats: np.ndarray) -> "FeatureScaler":
        self.scalers = [MinMaxScaler1D().fit(feats[:, j]) for j in range(feats.shape[1])]
        return self

    def fit_transform(self, feats: np.ndarray) -> np.ndarray:
        return self.fit(feats).transform(feats)

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [self.scalers[j].transform(feats[:, j]) for j in range(feats.shape[1])]
        )

    @property
    def target_scaler(self) -> MinMaxScaler1D:
        return self.scalers[0]  # column 0 is the close price


def train_lstm(train_X: np.ndarray, train_y: np.ndarray,
               val_X: np.ndarray, val_y: np.ndarray,
               trial: TrialSpace, seed: int = 0,
               device: str | None = None, amp: bool = False,
               verbose: bool = False, max_epochs: int | None = None):
    """Train one trial; returns (model, best_val_loss, train_preds_scaled, val_preds_scaled).

    Early stopping restores the best-validation-loss weights (paper: Keras
    EarlyStop on validation loss, patience 10, max 100 epochs).

    `max_epochs` is an explicit argument rather than a read of the module
    constant so that it survives into joblib worker processes, which re-import
    `config` fresh and would otherwise silently ignore a caller's override.

    Efficiency: tensors are moved to the device once; batch indices stay on the
    device (no host<->device round-trips); optional AMP autocast on CUDA.
    """
    config.seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = amp and device.startswith("cuda")

    model = LSTMRegressor(train_X.shape[2], trial.neurons, trial.layers,
                          trial.dropout).to(device)
    opt = _make_optimizer(trial.optimizer, model.parameters(), trial.learning_rate)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    tx = torch.as_tensor(train_X, device=device)
    ty = torch.as_tensor(train_y, device=device).unsqueeze(1)
    vx = torch.as_tensor(val_X, device=device)
    vy = torch.as_tensor(val_y, device=device).unsqueeze(1)

    n = len(tx)
    best_val, best_state, patience = np.inf, None, 0
    rng = np.random.default_rng(seed)
    epochs = config.LSTM_MAX_EPOCHS if max_epochs is None else max_epochs
    for epoch in range(epochs):
        model.train()
        order = torch.as_tensor(rng.permutation(n), device=device)
        for start in range(0, n, trial.batch_size):
            idx = order[start:start + trial.batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(tx[idx]), ty[idx])
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            val_loss = float(loss_fn(model(vx), vy))
        if verbose:
            print(f"epoch {epoch + 1}: val_loss={val_loss:.6f}")
        if val_loss < best_val - 1e-12:
            best_val, patience = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.LSTM_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(tx).squeeze(1).float().cpu().numpy()
        val_pred = model(vx).squeeze(1).float().cpu().numpy()
    return model, best_val, train_pred, val_pred


def predict_scaled(model, X: np.ndarray, device: str | None = None) -> np.ndarray:
    """Deterministic forward pass.

    Forces eval mode rather than trusting the caller: dropout is now active for
    single-layer models too, so a model left in train mode would return a
    different forecast on every call (measured spread 0.05 on a [-1, 1] target).
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(torch.as_tensor(X.astype(np.float32), device=device))
    finally:
        model.train(was_training)
    return out.squeeze(1).cpu().numpy()
