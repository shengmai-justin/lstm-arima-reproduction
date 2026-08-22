# Reproducing "LSTM-ARIMA as a Hybrid Approach in Algorithmic Investment Strategies"

An independent reproduction of **Kashif & Ślepaczuk (2024)**,
[arXiv:2406.18206](https://arxiv.org/abs/2406.18206) (q-fin.TR).

The paper builds three algorithmic investment strategies — ARIMA, LSTM, and a
hybrid LSTM-ARIMA that feeds ARIMA residuals to the LSTM as a fourth input —
and evaluates them on the S&P 500, FTSE 100 and CAC 40 over 2000-01 → 2023-08
with walk-forward optimization. Its headline claim is that **LSTM-ARIMA obtains
the highest modified information ratio (IR\*\*) in every index × strategy
combination.**

This repository is a from-scratch PyTorch implementation of that pipeline,
written to be checkable rather than merely runnable. **Read
[Known deviations](#known-deviations-from-the-paper) and
[What the paper leaves unspecified](#what-the-paper-leaves-unspecified) before
comparing any number here against the paper.**

---

## Status

| Component | State |
|---|---|
| Data pipeline | ✅ Reproduces the paper's Table 1 exactly (all 8 statistics × 3 indices) |
| Walk-forward grid | ✅ Reproduces the paper's OOS start dates exactly |
| Performance metrics | ✅ Reproduce the paper's Buy&Hold rows to 2 dp |
| ARIMA strategy | ⚠️ 5 of 6 cells close; ^GSPC Long-Short unexplained (see below) |
| LSTM / LSTM-ARIMA | ⏳ Not yet validated at full scale |
| Test suite | 58 tests, all passing |

**The paper's central claim has not yet been tested by this reproduction.**
Buy&Hold matching is evidence that the data and the metric formulas are right;
it says nothing about whether the paper's method works.

---

## What the paper leaves unspecified

These are genuine holes in the paper, not implementation choices. Each one had
to be decided before the pipeline could run at all, and each decision changes
the numbers.

| # | Underspecified in the paper | What this repo does |
|---|---|---|
| **G1** | Feature/target scaling is never mentioned — yet the output activation is fixed to `tanh`, which cannot represent a raw price | MinMax-scale features and target to [-1, 1], fit on **each walk's training window only**. Fitting on IS or full data would leak. See [tanh saturation](#the-tanh-saturation-problem) for the consequence |
| **G2** | §4.9.2 says the target is the close at `t + sequence_length`; §4.8 says the signal compares `P̂(t+1)` to `P(t)`. These contradict each other | Follow §4.9.2 for the target (horizon = `seq_len`) and §4.8 for the signal. Turnover then scales as `1/seq_len`, which is what produces the long flat stretches visible in the paper's own Figs. 5–7 |
| **G3** | The model-selection criterion (§4.7) is defined in terms of "IR2", which is never defined anywhere in the paper | Read as IR\* (Eq. 21), computed from the train/validation equity curves under the Long-Short rule. Applied consistently at selection time and evaluation time |
| **G4** | The realized-volatility formula (Eq. 1) is written over the whole sample, which is unusable in a walk-forward setting | Rolling 21-day window, past data only |
| **G6** | How ARIMA residuals are aligned with the LSTM's input rows is not described | Residuals from the walk's best-AIC fit, aligned by date. Train rows use in-sample residuals; validation/OOS rows use rolling one-step forecast errors, each known at its own close |
| **G7** | Keras specifics (initialization, shuffling, optimizer epsilons) are inherited silently from TensorFlow | Matched deliberately where it matters: `unit_forget_bias=True`, Keras' Adagrad accumulator (0.1) and epsilon (1e-7), Keras-style variational input dropout |
| **G9** | Whether the ARIMA includes a drift term is not stated, and statsmodels defaults to none when `d > 0` | `trend='t'` (a linear trend in levels is a drift in the differenced equation, i.e. the `c` of the paper's Eq. 8) |
| **G10** | Whether ARIMA is re-conditioned on observed closes as the OOS window advances is never stated | **Both readings implemented**, switchable. See below — this one decides whether the ARIMA rows are reproducible at all |

### G10 in detail: the ARIMA forecast mode

The paper says only "fit the best model and execute predictions". Two readings:

- **`static`** (default) — one multi-step forecast path per walk, taken from the
  end of the training window.
- **`rolling`** — one-step-ahead, re-conditioned on each observed close. The
  implementable reading, and the more defensible strategy.

Under `rolling`, positions flip on ~50% of days, which costs ~25%/yr at the
paper's 0.1% transaction cost and drives every ARIMA row deeply negative. Only
`static` reproduces the paper:

| index | strategy | rolling IR\*\* | static IR\*\* | paper IR\*\* |
|---|---|---|---|---|
| ^GSPC | Long-Only | −2.99 | **0.73** | 0.53 |
| ^GSPC | Long-Short | −17.21 | −1.28 | 7.13 |
| ^FTSE | Long-Only | −8.61 | **0.13** | −1.91 |
| ^FTSE | Long-Short | −28.28 | **−0.26** | 0.07 |
| ^FCHI | Long-Only | −7.67 | **0.92** | −1.93 |
| ^FCHI | Long-Short | −27.15 | **0.02** | −0.21 |

`static` also recovers the Long-Only ASD signature in all six cells (^GSPC
14.83 vs the paper's 14.45, against a market ASD of 19.58) and the Table 6
regression betas (0.573 vs 0.555). It is therefore the default. The hybrid's
residual feature always uses `rolling`, since it is meant to be a one-step
forecast *error*.

```bash
python scripts/run_experiment.py --model arima --index ^GSPC --arima-forecast rolling
```

---

## The tanh saturation problem

This is the most important thing to understand before reading any LSTM result
from this repository or from the paper.

Three individually reasonable constraints combine into a dead end:

1. The paper fixes the output activation to `tanh`, so the network's output is
   bounded to (−1, 1).
2. To predict a price of ~1500 with a bounded output, the target must be
   scaled. The paper never says how (G1); MinMax is the standard choice.
3. The scaler must be fit on the training window only, or the model sees the
   future.

Inverting the scaling, a bounded output maps back to a price strictly inside
`(train_min, train_max)`. **The model can never predict a price above the
highest price it saw in training.** The §4.8 rule is `long iff F(t) > P(t)`, so
on every OOS day where the index has already risen past that maximum, the rule
is mechanically short regardless of what the network learned.

Measured on ^GSPC, per walk:

```
walk                     train window  train_max     OOS closes   OOS above ceiling
   9  2009-01-14~2013-01-03                 1466    1742~2091            100.0%
  12  2012-01-05~2015-12-24                 2131    2239~2690            100.0%
  16  2015-12-28~2019-12-16                 3191    3647~4705            100.0%

overall: 2600/4682 = 55.5% of OOS days are above the model's output ceiling
         7 of 19 walks are at 100%
```

A short smoke run on walk 0 shows the mechanism directly:

```
target   forecast range     actual OOS closes   days long
level    1115.6 – 1181.2    1138 – 1294           0.0%
return   1142.0 – 1298.7    1138 – 1294          99.6%
```

This is a consequence of the paper's own specification, not a bug in this
implementation, and it plausibly explains the long flat stretches in the
paper's Figs. 5–7. It is left as the default for fidelity. A control is
provided:

```bash
python scripts/run_experiment.py --model lstm --index ^GSPC --target return
```

`--target return` regresses the forward `seq_len`-day return instead, so the
signal reduces to `r̂ > 0` and no price ceiling exists. This is a **deliberate
deviation** from the paper, kept so the effect is measurable rather than
merely argued. Results land in a separate `*_return` directory.

---

## Known deviations from the paper

Beyond the interpretation of the gaps above, this implementation deliberately
differs in the following ways. All were made to fix defects that would
otherwise invalidate the comparison the paper is trying to make.

**The hybrid and the plain LSTM search the same hyperparameter grid.** The
random search draws 20 of 216 configurations per walk. If the ARIMA order
search shares the walk's random generator, it consumes draws first and the
hybrid ends up exploring a different corner of the grid — measured overlap
0.103 against 0.093 expected from two independent draws, i.e. no more than
chance. Any hybrid-vs-plain difference would then confound the residual feature
with search luck, and the paper's headline claim (RQ2) could not be answered
either way. ARIMA now gets its own generator; overlap is 1.000.

**Dropout applies to single-layer models.** Keras' `LSTM(dropout=r)` drops the
layer's input connections; PyTorch's `nn.LSTM(dropout=r)` drops only *between*
stacked layers and is a silent no-op at `num_layers == 1`. Since the search
samples `layers ∈ {1, 2}`, a naive port leaves ~half of every random search
unregularised and makes the §6 dropout panels (the evidence for RQ3) measure
nothing on those trials. Keras-style variational input dropout is implemented
explicitly.

**ARIMA fits are constrained.** `enforce_stationarity` / `enforce_invertibility`
follow the statsmodels default of `True`. Disabling them is faster, but over the
first 6 ^GSPC walks the two settings agree on the selected order in only 1 of 6,
and the unconstrained search always lands on the largest available order — it
reaches lower AIC by entering exactly the non-stationary region the constraint
exists to exclude.

**Early stopping restores the best weights.** Keras' `EarlyStopping` defaults to
`restore_best_weights=False`, keeping the weights from `patience` epochs *after*
the best. The paper describes the mechanism as "optimize the number of epochs
based on the validation loss", and §4.7 then ranks trials by a validation loss
the returned model must actually have achieved, so best-weight restoration is
used.

**Maximum loss duration measures time underwater.** MLD is computed as the
longest stretch strictly below the high-water mark, peak to peak, and the end of
the sample closes an unrecovered stretch. Measuring only gaps between new highs
reports ~0 for a curve that peaks early and declines to the end, and gives a
spurious ~0.008-year floor to strategies that never lose.

---

## Errors found in the paper

Recorded because they affect what "matching the paper" can mean, not as
criticism of the authors.

- **Table 7, S&P 500 base-case rows** duplicate the LSTM-ARIMA row of Table 2
  (4.32 / 11.14 / 28.95 / 1.67 / 38.79 / 5.79) instead of the ARIMA row. The
  FTSE and CAC blocks of the same table correctly echo Tables 3–4, so it is a
  copy-paste error confined to the S&P block.
- **Table 5** reports the ARIMA Long-Only paired t-test as significant with
  H₁: μ_strategy − μ_benchmark > 0 (p = 0.0152 for FTSE, 0.0086 for CAC), while
  Tables 3–4 give that same strategy a *lower* ARC **and** a *lower* ASD than
  Buy&Hold. Higher mean daily return, lower volatility, and lower compounded
  return cannot hold simultaneously.
- **Table 2, S&P 500 Buy&Hold MLD = 1.65 years.** This repo reproduces that
  row's ARC (7.57 vs 7.52), ASD (19.58 vs 19.58) and MD (56.78 vs 56.78) — so
  it is the same equity curve — yet the S&P peaked in Oct 2007 and did not
  recover until Mar 2013, about 5.4 years. This repo reports 5.47. The FTSE
  (5.93 vs 5.94) and CAC (13.86 vs 14.04) values do match.
- **§4.9.2 vs §4.8** disagree on the forecast horizon (gap G2 above).

Also worth knowing when reading the paper's headline: **IR\*\* = ARC²/(ASD·MD)**
is quadratic in return, so it amplifies small differences. The ensemble's
"IR\*\* of 70.54%" is a ratio, not a return — the underlying ARC is 11.82%.

---

## Layout

```
src/
  config.py       constants, search spaces, and every switchable interpretation
  data.py         download, cleansing, feature engineering (paper §3, §4.9.2)
  arima.py        ARIMA fit, order search, rolling/static forecasting (§4.9.1)
  lstm.py         PyTorch LSTM, Keras-compatible defaults, trial sampling (§4.9.2)
  wfo.py          walk-forward engine for all three models (§4.4, §4.7)
  backtest.py     signal generation and transaction costs (§4.8)
  metrics.py      ARC, ASD, MD, MLD, IR*, IR** (§4.5)
  stats_tests.py  paired t-test and regression alpha (§5.2)
  ensemble.py     equal-weight ensemble AIS (§7)
scripts/
  download_data.py    fetch and cache the four Yahoo series
  run_experiment.py   one (model × index × variant) run
  run_sensitivity.py  sweep the §6 variants for one model × index
  run_all.sh          the full grid, one thread per job, resumable
  run_ensemble.py     §7 ensemble
  make_report.py      paper-style tables and equity-curve plots
tests/                58 tests
```

## Running it

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest tests/ -q                      # 58 passed

python scripts/download_data.py       # cached in data/ (gitignored)
scripts/run_all.sh                    # the full grid; -n for a dry run
python scripts/run_ensemble.py
python scripts/make_report.py
```

### One thread per job, not one job with many threads

The models are small — 3k to 3M parameters, batch 32, 4 input features — so the
matrix multiplies are far below the size where threading pays. Measured per
optimizer step:

| model | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| 25 × 1 layer | 0.98 ms | 1.01 | 1.01 |
| 100 × 2 layers | **3.58 ms** | 5.01 | 5.70 |
| 250 × 2 layers | **9.27 ms** | 9.95 | 10.58 |
| 500 × 2 layers | 25.00 ms | **24.32** | 25.62 |

Four threads made the 100×2 model 59% *slower*. `run_all.sh` pins each worker to
one thread and parallelises across jobs. A GPU does not help for the same
reason: batch 32 × hidden ≤ 500 cannot saturate one.

The whole grid is about 11 core-hours — roughly 40 minutes on an 18-core
machine. Finished jobs are skipped on re-run, so it is interruptible.

## Data

Four Yahoo Finance series (`^GSPC`, `^FTSE`, `^FCHI`, `^VIX`), daily,
2000-01-01 → 2023-08-31, cached as CSV under `data/` (gitignored). Yahoo revises
history, so re-downloading later can produce a slightly different series — ship
the cached CSVs rather than re-fetching if you need results to stay comparable.

Note that `^FCHI` carries 21.3% zero-volume days on current Yahoo data. Volume
is a paper feature and is kept as-is; the paper does not discuss data quality
here.

## Citation

```bibtex
@article{kashif2024lstmarima,
  title   = {LSTM-ARIMA as a Hybrid Approach in Algorithmic Investment Strategies},
  author  = {Kashif, Kamil and {\'S}lepaczuk, Robert},
  journal = {arXiv preprint arXiv:2406.18206},
  year    = {2024}
}
```

This repository is an independent reproduction and is not affiliated with the
authors.
