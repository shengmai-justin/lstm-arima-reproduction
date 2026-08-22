"""LSTM architecture conformance with paper Sec. 4.9.2."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.lstm import LSTMRegressor, MinMaxScaler1D, build_sequences  # noqa: E402


@pytest.mark.parametrize("layers", [1, 2])
def test_dropout_is_active_for_every_layer_count(layers):
    """Paper fixes dropout at 0.075 and samples layers from {1, 2}.

    PyTorch's nn.LSTM(dropout=) is a no-op at num_layers == 1, so relying on it
    alone would silently disable dropout for ~half the random-search trials and
    void the Sec. 6 dropout panels.
    """
    torch.manual_seed(0)
    model = LSTMRegressor(4, 32, layers, dropout=0.5)
    x = torch.randn(64, 21, 4)

    model.train()
    outs = [model(x) for _ in range(8)]
    spread = torch.stack(outs).std(dim=0).mean().item()
    assert spread > 1e-6, "dropout had no effect in train mode"

    model.eval()
    a, b = model(x), model(x)
    torch.testing.assert_close(a, b)  # deterministic at eval time


@pytest.mark.parametrize("layers", [1, 2])
def test_zero_dropout_is_deterministic_in_train_mode(layers):
    torch.manual_seed(0)
    model = LSTMRegressor(4, 16, layers, dropout=0.0).train()
    x = torch.randn(8, 14, 4)
    torch.testing.assert_close(model(x), model(x))


def test_input_dropout_mask_is_constant_across_timesteps():
    """Keras LSTM(dropout=) reuses one mask per sample for the whole sequence."""
    torch.manual_seed(0)
    model = LSTMRegressor(4, 8, 1, dropout=0.5).train()
    x = torch.ones(256, 10, 4)
    dropped = model._input_dropout(x)
    # every timestep of a given (sample, feature) must be identical
    assert torch.equal(dropped, dropped[:, :1, :].expand_as(dropped))
    # and the mask must actually be dropping something, rescaled to keep the mean
    assert 0.0 < (dropped == 0).float().mean().item() < 1.0
    assert abs(dropped.mean().item() - 1.0) < 0.1


def test_output_activation_is_tanh():
    """Paper Sec. 4.9.2: Output Layer Activation Function: tanh."""
    model = LSTMRegressor(4, 8, 1, dropout=0.0).eval()
    with torch.no_grad():
        out = model(torch.randn(32, 7, 4) * 50)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_build_sequences_uses_seq_len_horizon():
    """Paper Sec. 4.9.2 predicts the close at t + sequence_length (gap G2)."""
    feats = np.arange(40, dtype=float).reshape(20, 2)
    target = np.arange(20, dtype=float)
    L = 5
    X, y = build_sequences(feats, target, L, horizon=L)
    assert X.shape == (len(y), L, 2)
    # window i ends on decision day i+L-1; its target is close[i+L-1+L]
    for i in range(len(y)):
        assert X[i, -1, 0] == feats[i + L - 1, 0]
        assert y[i] == target[i + L - 1 + L]


def test_minmax_scaler_round_trips_to_the_paper_range():
    x = np.array([10.0, 20.0, 30.0])
    s = MinMaxScaler1D().fit(x)
    z = s.transform(x)
    assert z.min() == -1.0 and z.max() == 1.0  # tanh-compatible range (gap G1)
    np.testing.assert_allclose(s.inverse_transform(z), x)


def test_minmax_scaler_handles_constant_columns():
    s = MinMaxScaler1D().fit(np.zeros(5))
    assert np.all(np.isfinite(s.transform(np.zeros(5))))


def test_forget_gate_bias_starts_at_one():
    """Keras' LSTM defaults to unit_forget_bias=True; PyTorch does not."""
    for layers in (1, 2):
        m = LSTMRegressor(4, 32, layers, dropout=0.0)
        h = m.lstm.hidden_size
        for name, param in m.lstm.named_parameters():
            if name.startswith("bias_ih"):
                torch.testing.assert_close(
                    param.data[h:2 * h], torch.ones(h))
            elif name.startswith("bias_hh"):
                torch.testing.assert_close(
                    param.data[h:2 * h], torch.zeros(h))
        # only the forget slice is touched; the rest keeps PyTorch's init
        b = dict(m.lstm.named_parameters())["bias_ih_l0"].data
        assert not torch.allclose(b[:h], torch.ones(h))


def test_optimizers_use_keras_defaults():
    """The paper ran on TensorFlow; it pins lr but not eps/accumulator."""
    from src.lstm import KERAS_ADAGRAD_ACCUM, KERAS_EPS, _make_optimizer

    m = LSTMRegressor(4, 8, 1, 0.0)
    for name in ("adam", "nadam", "adagrad"):
        g = _make_optimizer(name, m.parameters(), 0.01).param_groups[0]
        assert g["eps"] == KERAS_EPS, name
    g = _make_optimizer("adagrad", m.parameters(), 0.01).param_groups[0]
    assert g["initial_accumulator_value"] == KERAS_ADAGRAD_ACCUM


def test_unknown_optimizer_is_rejected():
    from src.lstm import _make_optimizer

    m = LSTMRegressor(4, 8, 1, 0.0)
    with pytest.raises(ValueError, match="unknown optimizer"):
        _make_optimizer("rmsprop", m.parameters(), 0.01)


def test_predict_scaled_is_deterministic_regardless_of_caller_mode():
    """Dropout is now live for 1-layer models, so a train-mode model would
    otherwise return a different forecast on every call."""
    from src.lstm import predict_scaled

    m = LSTMRegressor(4, 32, 1, dropout=0.5).train()
    X = np.random.default_rng(0).normal(size=(16, 21, 4)).astype(np.float32)
    np.testing.assert_allclose(predict_scaled(m, X), predict_scaled(m, X))
    assert m.training, "predict_scaled must restore the caller's mode"


def test_random_search_draws_distinct_configurations():
    """Paper Sec. 4.9: 20 trials on a randomly chosen set. Drawing each field
    independently samples the 216-point grid WITH replacement, wasting a trial
    on 56% of walks."""
    from src.lstm import SEARCH_KEY_FIELDS, sample_trials

    for seed in range(25):
        trials = sample_trials(np.random.default_rng(seed), 20,
                               dropout=0.075, batch_size=32)
        assert len(trials) == 20
        keys = [tuple(getattr(t, f) for f in SEARCH_KEY_FIELDS) for t in trials]
        assert len(set(keys)) == 20, f"duplicate config at seed {seed}"
        # fixed (non-searched) params are still applied
        assert all(t.dropout == 0.075 and t.batch_size == 32 for t in trials)


def test_random_search_cannot_exceed_the_grid():
    from src.lstm import grid_size, sample_trials

    trials = sample_trials(np.random.default_rng(0), grid_size() + 50)
    assert len(trials) == grid_size()
