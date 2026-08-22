# Revisions: audit against arXiv:2406.18206

Record of what was checked against the paper (Kashif & Ślepaczuk 2024), what
was wrong, and what changed. Every claim below was verified by running code on
CPU; no GPU or heavy CPU work was involved.

---

## 1. What already matched the paper

Verified before changing anything — these needed no work:

| Check | Paper | Repo | |
|---|---|---|---|
| Row counts (Table 1) | 5953 / 5975 / 6049 | 5953 / 5975 / 6049 | exact |
| First OOS date ^GSPC | 2005-01-25 | 2005-01-25 | exact |
| First OOS date ^FTSE | 2005-01-13 | 2005-01-13 | exact |
| First OOS date ^FCHI | 2004-12-28 | 2004-12-28 | exact |
| Realized vol (Eq. 1) | √(Σ R²/N)·√252, N=21 | same | exact |
| ARC / ASD / MD (Eq. 15–19) | — | — | reproduce Table 2 B&H to 2 dp |
| IR\* = ARC/ASD (Eq. 21) | 38.43 | 38.66 | ✓ |
| IR\*\* = IR\*·ARC·sign/MD (Eq. 22) | 5.09 | 5.15 | ✓ |
| WFO grid: 1000 train / 250 val / 250 OOS, non-anchored | 19 walks | 19 walks | exact |
| Search spaces (Sec. 4.9.1–4.9.3) | — | — | match |
| Sec. 4.7 selection rule | top-5 val loss, min \|ΔIR\|, IR_val≠0 | same | match |
| Sec. 4.8 signal rule, 0.1% costs | — | — | match |

## 2. Correctness bugs found and fixed

### 2.1 ARIMA forecast state skipped the validation window — `src/wfo.py`

`run_arima_wfo` fitted ARIMA on the 1000-day training window and then fed
`walk_predictions` the **OOS block directly**, jumping over the 250 validation
days that sit between them. Every walk's first forecast was therefore
conditioned on state a year stale, and the rolling `append` chain started from a
250-day discontinuity.

```
^GSPC walk 0, first OOS forecast:   1143.94  (stale)
                                    1163.32  (continuous)
actual close:                       1168.41
```

The hybrid path (`_arima_residuals_per_walk`) already did this correctly — the
two code paths disagreed with each other. Fixed to carry the state through the
validation block; regression test asserts the two paths now agree.

### 2.2 MLD ignored an unrecovered tail — `src/metrics.py`

`max_loss_duration` measured only gaps *between* consecutive new equity highs.
A curve that peaks early and declines for the rest of the sample never opens a
second high, so the entire decline was invisible:

```
equity rises for 100 days, then falls for 900
before: 0.003 years        after: 2.462 years
```

MLD is a reported column in every table of the paper (2–4, 7–10), and most
strategies end below their high-water mark, so this silently understated a
headline metric almost everywhere. Fixed by closing the final stretch at the
end of the sample.

*Related finding (paper-side, not fixable here):* the paper's own S&P 500 B&H
MLD of **1.65 years** is not credible. Our B&H reproduces its ARC (7.57 vs
7.52), ASD (19.58 vs 19.58) and MD (56.78 vs 56.78) — so the equity curve is
the same one — yet the S&P peaked in Oct 2007 and did not recover until Mar
2013, i.e. ~5.4 years. We report 5.47. Treat the paper's 1.65 as an error.

### 2.3 ARIMA residual burn-in destroyed the hybrid's key feature — `src/arima.py`

statsmodels initialises the Kalman filter from a zero state, so the first
residuals are of the order of the **price level**, not of a forecast error:

```
^GSPC train window:  resid[0] = 1409.46   (train[0] = 1409.12)
genuine residual range after warm-up:  144.78
raw range including burn-in:          1487.92     -> 10.3x inflation
```

Those residuals feed the LSTM-ARIMA as its 4th input and are MinMax-scaled to
[-1, 1] alongside the other features. The burn-in outliers therefore *set the
scale*, compressing every genuine residual into roughly 10% of the usable
range — gutting the one feature that distinguishes the hybrid from plain LSTM,
which is the entire subject of the paper.

Fixed by zeroing the burn-in residuals (using `loglikelihood_burn`, floored at
`d + max(p, q)`). Zeroing rather than dropping keeps `residual[i]` aligned with
`close[i]`.

### 2.4 The last OOS day was never traded — `src/backtest.py`

`signals_from_forecast` mapped decision day → earning day with a *positional*
`.shift(1)` inside the forecast index, which discards the final decision day's
position instead of applying it to the next trading day. This was the cause of
the pre-existing failing test (`249 != 250`). Positions are now mapped onto the
asset's own trading calendar.

### 2.5 `--epochs` was silently ignored under `--jobs > 1` — `src/lstm.py`, `src/wfo.py`, `scripts/run_experiment.py`

`run_experiment.py` set `config.LSTM_MAX_EPOCHS = args.epochs`. joblib workers
re-import `config` fresh, so they never saw it:

```
parent: 3    workers: [100, 100]
```

`--epochs 5 --jobs 4` trained 100 epochs per trial. `max_epochs` is now an
explicit argument threaded to `train_lstm`.

### 2.6 Ensemble Buy&Hold was read from a hard-coded ARIMA directory — `src/ensemble.py`, `scripts/run_ensemble.py`

`run_ensemble.py` loaded the B&H curve from `results/<index>/arima_<variant>/`.
B&H does not depend on the model at all, and the Sec. 6 variants
(`dropout_*`, `batch_*`) are LSTM-only and have no ARIMA counterpart — so any
sensitivity ensemble crashed. Added `ensemble.load_buy_hold()`, which takes the
curve from whichever run produced one and falls back to the base variant.

Also: `ensemble_equity` intersected the three indices' trading days with
`dropna()`, discarding each index's performance on days another market was
shut. Now forward-fills across differing holiday calendars instead.

### 2.7 Dropout was inert for half the random-search trials — `src/lstm.py`

The paper fixes dropout at 0.075 (Sec. 4.9.2) and samples `layers` from
`{1, 2}`. The model used `nn.LSTM(dropout=r)`, which in PyTorch drops only
**between stacked layers** and is a silent no-op at `num_layers == 1`:

```
layers=1: nn.LSTM.dropout = 0.0     <- ~50% of sampled trials
layers=2: nn.LSTM.dropout = 0.075
```

Keras' `LSTM(dropout=r)` — what the paper used — drops the layer's *input*
connections with a mask held constant across timesteps, and applies to a
single-layer model just as much as a stacked one. So roughly half of every
random search ran with no regularisation at all, and, worse, the Sec. 6
dropout panels (0.05 / 0.075 / 0.1 — the evidence for **RQ3**) were measuring
nothing on those trials.

Fixed by adding Keras-style variational input dropout, active at any layer
count, alongside the existing inter-layer dropout.

### 2.8 Unconstrained ARIMA fits changed the selected order — `src/arima.py`

`OUTLINE.md` claimed `enforce_stationarity=False, enforce_invertibility=False`
was a free speed win with "negligible effect on AIC ranking". Measured over the
first 6 ^GSPC walks with orders 0–3:

```
walk 0  (2,1,3) vs (2,1,2)      walk 3  (1,1,3) vs (1,1,1)
walk 1  (3,1,3) vs (0,1,0)      walk 4  (3,1,3) vs (2,1,1)
walk 2  (3,1,3) vs (0,1,1)      walk 5  (2,1,3) vs (2,1,3)   -> agree 1/6
```

The unconstrained search systematically lands on the **largest available
order**, reaching lower AIC by wandering into exactly the non-stationary /
non-invertible region the constraint exists to exclude — the overfitting the
paper's own Sec. 6.1 speculates about. statsmodels defaults to enforcing and
the paper never says otherwise, so `config.ARIMA_ENFORCE = True` is now the
default. Cost: a full ARIMA job goes 27 s → 34–46 s per index. Downstream
effect on the metrics is small (^GSPC Long-Only IR\*\* 0.73 → 0.79), but the
documented justification was wrong and is retracted.

### 2.9 Keras -> PyTorch semantic drift in the training stack — `src/lstm.py`

The paper ran on TensorFlow (Sec. 4.9). Three Keras defaults it therefore
inherited silently were not reproduced by the PyTorch port:

| | Keras (what the paper got) | PyTorch (what we had) |
|---|---|---|
| LSTM forget-gate bias | `unit_forget_bias=True` -> **1.0** | uniform, measured **-0.0004** |
| Adagrad accumulator | `initial_accumulator_value=0.1` | **0** |
| Adam / Nadam / Adagrad epsilon | `1e-7` | `1e-8` / `1e-10` |

The forget bias is the one that matters. At bias 0 the gate opens at
sigmoid(0)=0.5 instead of sigmoid(1)=0.73, so the cell sheds state faster
early in training -- working directly against the long-memory behaviour the
paper's whole argument rests on. The Adagrad accumulator is next: starting at
0 makes the very first update `lr*sign(grad)` for every weight, i.e. a full
0.01 step at the paper's higher learning rate. All three now follow Keras.

### 2.10 `predict_scaled` trusted the caller's train/eval mode — `src/lstm.py`

It never called `model.eval()`. That was harmless while dropout only existed
between stacked layers, but §2.7 made dropout live for single-layer models too,
so a model left in train mode returns a different forecast on every call
(measured spread **0.05** on a target scaled to [-1, 1]). Now forces eval and
restores the caller's mode.

### 2.11 Early stopping: a deliberate divergence, recorded

The paper says it used "the Keras EarlyStop function ... setting the patience
to 10 epochs". Keras' `EarlyStopping` defaults to
**`restore_best_weights=False`** -- it keeps the weights from the *last* epoch,
10 epochs after the best one, not the best. Our trainer restores the best
weights.

We keep restore-best deliberately: the paper describes the mechanism as
"optimize the number of epochs based on the validation loss", which only makes
sense if the best epoch is the one used, and Sec. 4.7 then ranks trials by
validation loss -- a ranking that is incoherent if the returned model is not
the one that achieved that loss. Flagged here because it is a real fork in the
road, not because the evidence is ambiguous.

### 2.12 The hybrid and the plain LSTM searched disjoint hyperparameter grids — `src/wfo.py`

**The most consequential defect found.** `_one_walk` built one generator per
walk and handed that same object to `fit_arima` on the hybrid path. `fit_arima`
consumes 20 draws (`rng.choice(49, size=20, replace=False)`) *before*
`sample_trials` runs, so the hybrid's random search started from a different
point in the stream:

```
seed 0, walk 0        plain                     hybrid
trial 0    (500, 2, nadam,  0.01,    7)    (25,  1, adagrad, 0.0001, 21)
trial 1    (25,  1, adam,   0.01,   21)    (100, 2, adagrad, 0.0001, 14)
trial 2    (100, 2, nadam,  0.0001, 21)    (100, 2, adagrad, 0.01,   21)

config overlap across all 19 ^GSPC walks at n_trials=20:  0.103
expected from two independent 20-of-216 draws:            0.093
```

The two searches shared **no more configurations than chance**. The paper's
entire headline claim is LSTM-ARIMA > LSTM; under a shared generator any
measured difference confounds the ARIMA residual feature with hyperparameter
search luck, and the reproduction could not have answered RQ2 either way.

Fixed by giving the ARIMA order search its own generator
(`default_rng(seed + k + 1_000_000)`). Overlap is now **1.000** — hybrid and
plain train on the identical 20 configurations, so the only difference between
them is the 4th input feature.

### 2.13 Silent-wrong-answer paths in the metrics and tests

Found by an independent audit pass; all reproduced before fixing.

- **`modified_information_ratio` returned 0.0 when MD == 0.** IR\*\* is
  unbounded for a drawdown-free strategy, so a strategy that never lost money
  was ranked *below every losing strategy* in the paper's headline metric.
  Now returns signed infinity; a genuinely flat strategy still returns 0.
- **`arc` / `asd` / `max_drawdown` returned 0.0 on an empty sample**, making
  "no data" indistinguishable from "flat, riskless". Now NaN.
- **`max_loss_duration` had a 0.0082-year floor for curves that never lost.**
  Every observation of a flat or rising curve is a new high, so the gaps
  between consecutive highs were just weekends. It now measures the longest
  stretch spent *below* the high-water mark, peak to peak. Real Buy&Hold values
  are unchanged (^GSPC 5.4675, ^FTSE 5.9302, ^FCHI 13.8563); only
  never-underwater curves move, and they move to 0.
- **`max_loss_duration` returned ~0 on a positional index.** `np.diff` yields
  integers and `pd.Timedelta(3)` is 3 *nanoseconds*. Now divides by
  `TRADING_DAYS` when the index is not a `DatetimeIndex`.
- **A NaN forecast became a maximum-conviction SHORT.** `NaN > close` is False,
  which under long_short is -1. A failed fit now stands aside (0).
- **`regression_alpha` never dropped NaNs.** One NaN made every output NaN and
  `significant_10pct` silently False (`nan < 0.10` is False), and it would have
  run on a different sample than `paired_ttest`, which does drop them. Now
  aligned and dropped, with a guard for n < 3. Also switched `1 - t.cdf` to
  `t.sf`, which does not underflow in the far tail.
- **`_select_best_trial`'s fallback contradicted its docstring.** With every
  `IR_val == 0`, ranking by `|IR_train - IR_val|` collapses to `|IR_train|` and
  hands the walk to the trial with the *worst* training performance. Now falls
  back to lowest validation loss, as documented. (Latent: `information_ratio`
  only returns exactly 0 for a degenerate series.)
- **A walk could silently complete on one surviving trial.** `except Exception:
  continue` in the trial loop meant a partial CUDA OOM would quietly reduce
  Sec. 4.7's top-5 pool to an arbitrary single model, visible only in
  `walks.csv`. Now warns when fewer than `TOP_K_VAL_LOSS` survive.

### 2.14 Report and ensemble bookkeeping

- **`make_report.base_tables` emitted the Buy&Hold row once per model** (its
  `break` left only the inner `iterrows` loop), and the `paper_IR**` column was
  **always blank for Buy&Hold** because `PAPER_IR2` is keyed by strategy mode
  and no `(index, "buy_hold")` key exists. The benchmark row — the one row
  checkable against the paper without a model run — was never compared. Both
  fixed; it now reads 5.15 vs 5.09, 0.68 vs 0.66, 0.98 vs 0.98.
- **`run_ensemble` built the Buy&Hold ensemble inside the mode loop**, emitting
  two byte-identical rows mislabeled "Long Only" / "Long Short" plus two
  identical CSVs. Paper Table 10 has one. Now computed once.
- **My own §2.6 `ffill` change is reverted.** The audit disproved its premise:
  terminal equity is bit-identical with or without it, because an equity curve
  is cumulative and sampling it on fewer dates loses no performance. What the
  ffill *did* do was inflate the observation count from 4581 to 4804 over the
  same 18.59 years, and since ARC/ASD annualize with the paper's hard-coded 252,
  that alone moved ensemble IR\*\* by ~7% (2.61 -> 2.43) without a cent
  changing hands. Back to the common-days intersection.

  The residual issue is inherent to the paper: 252 is hard-coded, but the
  ensemble's common calendar runs ~246 observations/year. We keep the paper's
  constant rather than invent a new convention.

## 3. The main alignment gap: how ARIMA forecasts the OOS block (gap G10)

The paper never states whether the fitted ARIMA is re-conditioned on observed
closes as the OOS window advances. It matters enormously, and it was the reason
the repo's ARIMA rows looked nothing like Tables 2–4.

Measured turnover under the repo's original (rolling) reading:

```
^GSPC  position changes on 49.1% of days  ->  24.8%/yr in costs at 0.1%
^FTSE  49.7%                              ->  25.1%/yr
^FCHI  52.0%                              ->  26.2%/yr
```

A 25%/yr cost drag cannot produce the paper's ARIMA numbers, so the paper's
ARIMA must turn over slowly. Both readings are now implemented and switchable
(`config.ARIMA_FORECAST_MODE`, `--arima-forecast`). Full CPU runs of all three
indices under each:

| index | strat | rolling ARC/ASD/IR\*\* | static ARC/ASD/IR\*\* | paper ARC/ASD/IR\*\* |
|---|---|---|---|---|
| ^GSPC | LO | -5.42 / 14.97 / **-2.99** | 2.38 / 14.83 / **0.73** | 1.89 / 14.45 / **0.53** |
| ^GSPC | LS | -18.17 / 19.65 / **-17.21** | -4.14 / 19.59 / **-1.28** | 8.66 / 19.19 / **7.13** |
| ^FTSE | LO | -10.38 / 14.31 / **-8.61** | 0.95 / 14.48 / **0.13** | -3.78 / 12.88 / **-1.91** |
| ^FTSE | LS | -22.54 / 18.11 / **-28.28** | -1.66 / 18.04 / **-0.26** | 0.84 / 18.04 / **0.07** |
| ^FCHI | LO | -10.43 / 15.97 / **-7.67** | 2.93 / 16.66 / **0.92** | -4.38 / 15.14 / **-1.93** |
| ^FCHI | LS | -24.10 / 21.50 / **-27.15** | 0.49 / 21.44 / **0.02** | -1.81 / 21.43 / **-0.21** |

Static reproduces the ASD signature in all six cells (the Long-Only ASD well
below market ASD is the fingerprint of partial exposure) and lands within ~1–2
IR\*\* points in five of six. Rolling is 10–30 IR\*\* points off everywhere.
**`static` is therefore the default**; `rolling` remains available and is the
more defensible reading for a strategy anyone would actually trade.

The hybrid's residual feature always uses `rolling` regardless — it is meant to
be a one-step forecast *error*, and a static path would just accumulate horizon
drift.

The one cell static does not explain is ^GSPC Long-Short (paper 7.13). Note the
paper claims ARIMA Long-Short beats B&H there (ARC 8.66 vs 7.52) while turning
over enough to pay costs; we could not reproduce that under either reading.

## 4. Internal inconsistencies in the paper (recorded, not reproduced)

- **Table 7, S&P 500 base-case rows** duplicate the LSTM-ARIMA row of Table 2
  (4.32 / 11.14 / 28.95 / 1.67 / 38.79 / 5.79) instead of the ARIMA row. The
  FTSE and CAC blocks of Table 7 correctly echo Tables 3–4, so this is a
  copy-paste error confined to the S&P block.
- **Table 5** reports the ARIMA Long-Only paired t-test as significant with
  H₁: μ_strategy − μ_benchmark > 0 (p = 0.0152 for FTSE, 0.0086 for CAC), yet
  the same ARIMA Long-Only has *lower* ARC and *lower* ASD than B&H in Tables
  3–4. Higher mean daily return, lower volatility and lower compounded return
  cannot hold simultaneously.
- **Sec. 4.9.2 vs Sec. 4.8**: the feature list says the target is the close at
  `t + sequence_length`, while the strategy section compares `P̂(t+1)` to
  `P(t)`. The repo implements horizon = `sequence_length` (see OUTLINE gap G2);
  this is also what produces the long flat stretches visible in the paper's
  Figures 5–7.
- **Table 2 S&P 500 B&H MLD = 1.65 years** — see §2.2.

## 5. Documentation corrected

`OUTLINE.md` had drifted from the code: gap G2 described a next-day horizon the
code does not use, G3 claimed a config switch that does not exist, G5 described
`^GSPC` volume as unusable (Yahoo now returns real volume — 0% zeros), and gaps
G9/G10 were cited by `config.py` and `backtest.py` but absent from the file.
All corrected, and the verification section now carries measured numbers.

## 6. Test suite

15 passed / 1 failed  →  **29 passed**. New coverage: ARIMA burn-in
neutralisation, row alignment, causality of both forecast modes, static-vs-
rolling behavioural difference, rejection of unknown modes, BIC vs AIC
selection, the validation-gap regression, full OOS-day coverage, MLD tail and
monotone-decline cases, and `max_epochs` being an argument rather than module
state.

## 7. Open risk on the LSTM path: tanh output saturation

Not a bug — a structural consequence of the paper's own spec — but anyone
running the GPU jobs needs to know about it first.

The paper fixes the output activation to **tanh** (Sec. 4.9.2), and gap G1 has
us MinMax-scale the target to [-1, 1] fitted on each walk's *training* window.
Any OOS close outside that training price range is therefore **unreachable**:
the head saturates at ±1 and the inverse transform returns the training
min/max. In a trending market that means the forecast pins to the training
high, which sits *below* the current price, so the §4.8 rule emits a
persistent short/flat signal that reflects saturation rather than learning.

Measured share of OOS closes outside the training window's range:

| index | walks | mean | median | walks >50% saturated |
|---|---|---|---|---|
| ^GSPC | 19 | 58.7% | 80.4% | 12 / 19 |
| ^FTSE | 19 | 27.8% | 5.2% | 4 / 19 |
| ^FCHI | 20 | 38.6% | 26.1% | 8 / 20 |

On the S&P 500 the median walk has **80% of its OOS days unreachable**. This
plausibly explains the long flat stretches in the paper's Figs. 5–7, so it may
well be what the authors' code did too.

`config.LSTM_TARGET` (`--target`) now carries both readings so the effect is
measurable rather than merely argued. `level` stays the default and is faithful
to Sec. 4.9.2. `return` regresses the forward `seq_len`-day return instead, so
the Sec. 4.8 rule reduces to `r_hat > 0` and no price ceiling exists; the
forecast is reported back as `P(t)*(1+r_hat)` so every downstream stage is
unchanged. Results land in a `*_return` directory. Measured on ^GSPC walk 0
(3 trials, 4 epochs — undertrained, but the mechanism is the point):

```
target   forecast range      actual OOS closes   days long
level    1115.6 – 1181.2     1138 – 1294           0.0%
return   1142.0 – 1298.7     1138 – 1294          99.6%
```

The `level` forecast is structurally clamped below the range it is trying to
predict. Note the close *feature* is still a scaled level under both settings —
only the target changes; feature extrapolation is far less harmful than a
saturated output. Switching the target is a **deliberate deviation** from the
paper, recorded here, not a silent correction.

## 8. Not run locally (per instruction: no GPU, no heavy CPU)

All LSTM and LSTM-ARIMA walk-forward runs — 2 of the 3 models, and the paper's
entire headline claim. The fixes in §2.3 (residual scaling) and §2.5 (epochs)
affect those paths and are covered by unit tests only; they need a server run
to confirm end-to-end. §8 of `OUTLINE.md` has the commands.

## 9. Known-unresolved discrepancies

- **^GSPC ARIMA Long-Short**: paper reports IR\*\* 7.13 (base), 7.62 (orders
  0–3), 7.86 (BIC); we get -1.28, -1.35, -1.52 under `static` and -17.2 under
  `rolling`. The gap is systematic across every variant, while everything else
  about ^GSPC lines up (B&H exact, Long-Only within 0.2, regression beta 0.573
  vs 0.555). Unexplained.
- **^FTSE / ^FCHI ARIMA Long-Only sign**: ours mildly positive (0.13–0.92),
  paper mildly negative (-1.23 to -2.86). Small magnitudes either way.
- **Minor code smells left alone** (cosmetic, no effect on results):
  `make_report.plot_equity_curves` reads `run` outside the loop that defines
  it; `ArimaWalkForecaster.residuals` is unused; `data._fetch_yf`'s
  `lru_cache` is redundant given the CSV cache; the `^GSPC` zero-volume guard
  in `build_features` is now dead code (Yahoo returns 0% zeros).
