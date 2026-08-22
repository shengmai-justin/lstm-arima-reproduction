# Reproduction Outline: LSTM-ARIMA as a Hybrid Approach in Algorithmic Investment Strategies

Source paper: Kashif & Ślepaczuk (2024), arXiv:2406.18206 (q-fin.TR)
Reproduction target: full pipeline — ARIMA / LSTM / LSTM-ARIMA AIS on S&P 500, FTSE 100, CAC 40.

---

## 1. Paper Structure

| Section | Content |
|---|---|
| 1 Introduction | Motivation, hypothesis (LSTM-ARIMA beats individuals), 4 research questions |
| 2 Literature | ARIMA / ML (LSTM, SVM, RF) / hybrid models overview |
| 3 Data | 3 indices, yfinance, 2000-01 → 2023-08-30, descriptive stats, volatility measures |
| 4 Methodology | ARIMA, LSTM, LSTM-ARIMA, WFO, metrics, strategy rules, hyperparameter tuning |
| 5 Empirical Results | Base-case OOS results (Tables 2–4), statistical significance (Tables 5–6) |
| 6 Sensitivity Analysis | Dropout / batch size / ARIMA orders / BIC variants (Tables 7–9) |
| 7 Ensembled AIS | Equal-weight (1/3) ensemble of the three indices (Table 10) |
| 8 Conclusion | LSTM-ARIMA outperforms on IR** everywhere; not robust to hyperparams (RQ3) |

## 2. Core Design of the Study

### 2.1 Data
- Tickers: `^GSPC`, `^FTSE`, `^FCHI` (daily, from yfinance), ~2000-01-03/04 → 2023-08-30.
- Counts in paper: GSPC 5953, FTSE 5975, FCHI 6049.
- Volatility feature: VIX (`^VIX`) for S&P 500; 21-day annualized realized volatility for FTSE/CAC.
- LSTM inputs: close, volatility, volume (+ ARIMA residual for the hybrid).

### 2.2 Walk-Forward Optimization (non-anchored)
- IS window = 1250 trading days (train 1000 + validation 250); OOS window = 250 days.
- Windows slide forward by 250 days; IS window is non-anchored (moves with each walk).
- Verified against paper: the first OOS day = row 1250 + 21 of Yahoo data (the 21-day
  seq_len warm-up), reproducing the paper's OOS starts exactly (2005-01-25 ^GSPC,
  2005-01-13 ^FTSE, 2004-12-28 ^FCHI).
- Random search of 20 trials re-run at EVERY walk; best model per walk selected by §4.7 criterion.
- Concatenated OOS periods form the evaluation sample (~19 walks per index).

### 2.3 Models
1. **ARIMA(p,1,q)**: p,q ∈ [0,6], selected by lowest AIC on the 1000-day training window
   (paper: 20 random trials out of the 49 combos). Predicts next-day close.
2. **LSTM**: predicts next-day close from a lookback window (seq_len ∈ {7,14,21}).
   - Random search: neurons ∈ {25,50,75,100,250,500}, layers ∈ {1,2},
     optimizer ∈ {Adam, Nadam, Adagrad}, lr ∈ {0.01, 0.0001}, seq_len ∈ {7,14,21}.
   - Fixed (base case): dropout 0.075, batch size 32, MSE loss, max 100 epochs,
     early stopping on validation loss w/ patience 10.
3. **LSTM-ARIMA**: same as LSTM but adds ARIMA residuals as a 4th input feature.
   ARIMA is re-fitted (best AIC) each walk; its residuals feed the LSTM.

### 2.4 Best-Model Criterion (paper §4.7)
From the 20 random-search trials take the 5 with lowest validation loss; compute "IR2"
(paper notation; we interpret as IR* — see §4 gap G3) on train and on validation equity
curves; pick the model minimizing |IR2(train) − IR2(val)| subject to IR2(val) ≠ 0.

### 2.5 Trading Rules (paper §4.8)
- Signal: Long (1) if P̂(t+1) > P(t), else 0 (Long-Only) or −1 (Long-Short).
- Transaction costs: 0.1% per unit of traded exposure (applied on position changes).
- Benchmark: Buy & Hold on the same index over the same OOS period.

### 2.6 Performance Metrics (paper §4.5)
- ARC = (∏(1+Rt))^(252/N) − 1
- ASD = sqrt(252) · std(Rt)
- MD = max peak-to-trough drawdown of the equity curve
- MLD = longest time (years) between consecutive new equity-curve highs
- IR* = ARC / ASD
- IR** = IR* × ARC × sign(ARC) / MD   (all in %, verified against paper Tables 2–4)

### 2.7 Statistical Significance (paper §5.2)
- Paired t-test on daily returns (strategy − B&H), H1: mean > 0, α = 10%.
- OLS: R_strategy = α + β·R_BH + ε, right-tailed test on α (H1: α > 0).

### 2.8 Sensitivity Analysis (paper §6)
- ARIMA: order range (0–3,1,0–3); AIC → BIC.
- LSTM & LSTM-ARIMA: dropout 0.05 / 0.10 (base 0.075); batch 16 / 64 (base 32).

### 2.9 Ensembled AIS (paper §7)
- Invest $1 per index, equal weights 1/3, 2005-01-25 → 2023-08-30, costs 0.1%.
- Equity curve = mean of the three per-index equity curves; metrics recomputed.
- Paper's headline: LSTM-ARIMA Long-Short ensemble IR** = 70.54%.

## 3. Key Paper Numbers to Compare Against (base case, IR** %)

| Index | Strategy | B&H | ARIMA | LSTM | LSTM-ARIMA |
|---|---|---|---|---|---|
| S&P 500 | Long-Only | 5.09 | 0.53 | 1.94 | 5.79 |
| S&P 500 | Long-Short | 5.09 | 7.13 | 3.87 | 7.18 |
| FTSE 100 | Long-Only | 0.66 | −1.91 | 1.44 | 7.19 |
| FTSE 100 | Long-Short | 0.66 | 0.07 | 0.67 | 16.65 |
| CAC 40 | Long-Only | 0.98 | −1.93 | 1.43 | 3.04 |
| CAC 40 | Long-Short | 0.98 | −0.21 | 0.97 | 14.29 |
| Ensemble | Long-Only | 1.70 | −0.84 | 3.27 | 9.64 |
| Ensemble | Long-Short | 1.70 | 2.64 | 7.25 | 70.54 |

## 4. Reproduction Gaps and Our Assumptions

| # | Gap in paper | Our assumption |
|---|---|---|
| G1 | No feature scaling mentioned, yet output activation is tanh | MinMax scale all features & target to [−1,1], fit on each walk's training window only; inverse-transform predictions |
| G2 | "Predict closing price at time t + sequence_length" (§4.9.2) vs §4.8 signal "P̂(t+1) vs P(t)" | Follow §4.9.2 literally: the LSTM/hybrid target is `close[t + seq_len]`, so horizon = seq_len (`build_sequences(..., horizon=L)`). The signal still follows §4.8 — compare that forecast to `P(t)` and hold one day. This yields turnover ~1/seq_len, which is what produces the long flat stretches in the paper's Figs. 5–7. The 21-day warm-up before the first walk reproduces the paper's exact OOS start dates |
| G3 | "IR2" in best-model criterion never defined | Read as IR\* (Eq. 21), computed from the train/validation equity curves under the **Long-Short** rule (`wfo._ir_from_forecast`). Not configurable — one rule is applied consistently at selection time and at evaluation time, which `tests/test_smoke.py::test_forecast_rule_matches_model_selection_ir` pins |
| G4 | Realized-vol formula is whole-sample; not usable in walk-forward | Rolling 21-day realized vol using only past data |
| G5 | Paper implies `^GSPC` volume may be unusable | Not an issue on current Yahoo data: `^GSPC` returns real volume with **0% zeros** (measured). Volume is kept as-is for all three indices, per the paper's feature list. (`^FCHI` carries 21% zero-volume days; also kept.) `data.build_features` retains a guard that zeroes the column only if it is >50% zeros |
| G6 | ARIMA residual alignment details unspecified | Residuals from the walk's best-AIC fit, aligned by date, available for IS+OOS of that walk. Kalman warm-up residuals are zeroed — see REVISIONS.md §2.3 for why leaving them in destroys the feature's scale |
| G7 | Keras specifics (weight init, shuffling, etc.) | PyTorch defaults, shuffled train batches; fixed seeds per trial for reproducibility |
| G8 | Paper Table 7 S&P500 "base case" rows duplicate Table 2 LSTM-ARIMA values | Treated as a paper typo (the FTSE/CAC blocks of Table 7 correctly echo Tables 3–4); our sensitivity table uses genuinely re-run base cases |
| G9 | ARIMA drift term unspecified; statsmodels defaults to none for d>0 | `trend='t'` (linear trend in levels == drift in the differenced equation; statsmodels disallows `trend='c'` when d>0). Matches the persistent-position profile of the paper's ARIMA rows |
| G10 | Whether ARIMA is re-conditioned on observed closes across the OOS block is never stated | Both readings implemented, switchable via `config.ARIMA_FORECAST_MODE` / `--arima-forecast`. **Default `static`** (one multi-step path per walk): it is the only reading that reproduces Tables 2–4 — rolling one-step turns over on ~50% of days, costing ~25%/yr at 0.1% and driving every ARIMA row deeply negative. Full comparison table in REVISIONS.md §3. The hybrid's residual feature always uses `rolling` |

## 5. Repository Layout

```
reproduce_LSTM/
├── OUTLINE.md              ← this file
├── REVISIONS.md            ← audit against the paper: bugs found, fixes, evidence
├── DEPLOY.md               ← how to run the sweep (local / RunPod / sharded)
├── requirements.txt
├── data/                   ← downloaded parquet/csv (gitignored)
├── results/                ← metrics, equity curves, plots, tables (gitignored)
├── src/
│   ├── config.py           ← constants: paths, tickers, windows, search spaces
│   ├── data.py             ← download/clean, volatility features
│   ├── metrics.py          ← ARC/ASD/MD/MLD/IR*/IR**
│   ├── backtest.py         ← signals, returns, equity curves, transaction costs
│   ├── arima.py            ← ARIMA random search + residuals
│   ├── lstm.py             ← PyTorch LSTM model/trainer (CPU/GPU agnostic)
│   ├── wfo.py              ← walk-forward engine for all 3 model types
│   ├── stats_tests.py      ← paired t-test, OLS alpha test
│   └── ensemble.py         ← equal-weight ensemble of the 3 indices
├── scripts/
│   ├── download_data.py
│   ├── run_experiment.py   ← one (model × index × variant) job; building block for server runs
│   ├── run_sensitivity.py  ← sweeps the §6 variants
│   ├── run_ensemble.py     ← combines per-index results into §7 outputs
│   └── make_report.py      ← Tables 2–10 style CSV/markdown + equity-curve plots
└── tests/
    ├── test_metrics.py     ← unit tests vs hand-computed values & paper identities
    ├── test_arima.py       ← burn-in, causality, static vs rolling, AIC/BIC
    └── test_smoke.py       ← light CPU smoke: tiny WFO runs (1 walk, few trials/epochs)
```

## 6. Efficiency Notes (optimizations over the naive implementation)

| Where | Optimization | Effect |
|---|---|---|
| ARIMA per walk | Reuse the search's fitted `ARIMAResults` for rolling forecasts (no duplicate MLE refit); `append(refit=False)` extends in O(p+q)/day | ~2× faster ARIMA walks |
| ARIMA random search | `joblib` parallelism across candidate orders (`n_jobs`) | search 5.8s → 2.1s per walk (4 jobs) |
| ~~ARIMA fits~~ | ~~`enforce_stationarity=False, enforce_invertibility=False` (negligible effect on AIC ranking)~~ **RETRACTED — the claim was false.** Measured over the first 6 ^GSPC walks the two settings pick the same order in only **1 of 6**, and the unconstrained search always lands on the largest available order. `config.ARIMA_ENFORCE = True` (the statsmodels default) is now used | full ARIMA job 27s → 34–46s per index; downstream IR** shift is small (^GSPC LO 0.73 → 0.79) |
| Walk level | Whole walks are independent windows → optional process parallelism (`--jobs`); workers pin `torch.set_num_threads(1)` | 4 walks: 15.9s → 7.5s (8 jobs) |
| LSTM training | Tensors moved to device once; batch indices kept on-device (no host round-trips); `zero_grad(set_to_none=True)`; cudnn benchmark on CUDA; optional AMP (`--amp`, CUDA only) | trial 3-20s CPU; server GPU runs scale further |
| Memory | Non-best trial models released after model selection; only predictions (not weights) leave a walk | keeps 20-trial searches memory-flat |
| Data | yfinance responses cached to `data/*.csv` on first download | repeat runs do no network I/O |

Caveat: `--jobs > 1` changes float reduction order (thread count) → results can
differ from the sequential run at float-noise level (amplified by early stopping
and model selection). Default `--jobs 1` is bit-reproducible (verified). Use
`--jobs N` when speed matters more than bit-equality.

## 7. Validation Against the Paper (already verified locally)

- Yahoo row counts match exactly: GSPC 5953, FTSE 5975, FCHI 6049.
- Walk grid reproduces the paper's OOS start dates exactly (2005-01-25 / 2005-01-13 /
  2004-12-28) and end 2023-08-30; 19 walks per index.
- Buy&Hold metrics vs paper Tables 2-4 (GSPC 7.57/19.58/56.78 → IR** 5.15 vs 5.09;
  FTSE 2.42/18.03/47.83 → 0.68 vs 0.66; FCHI 3.53/21.44/59.16 → 0.98 vs 0.98).
  Only MLD disagrees (5.47 vs the paper's 1.65 on GSPC) — since ARC, ASD and MD
  all match to 2 dp, the equity curve is the same one, and the S&P's 2007 peak
  did not recover until 2013. The paper's 1.65 appears to be an error.
- ARIMA rows under the default `static` reading of G10 land within ~1–2 IR** of
  the paper in 5 of 6 index × strategy cells, and reproduce the Long-Only ASD
  signature in all 6. The exception is GSPC Long-Short (paper 7.13), which we
  could not reproduce under either reading. See REVISIONS.md §3.
- LSTM and LSTM-ARIMA rows are NOT yet verified — those runs belong on the GPU
  server (§8). The qualitative target is LSTM-ARIMA > LSTM/ARIMA on IR**.

## 8. Execution Plan

1. **Local (light, CPU)**: `download_data.py`; `pytest tests/`; ARIMA base +
   sensitivity variants (fast): `run_sensitivity.py --model arima --index ^GSPC --jobs 8`.
   A full 19-walk ARIMA run is ~27 s per index at `--jobs 6`. To reproduce the
   gap-G10 comparison, add `--arima-forecast rolling` (results land in a
   `arima_<variant>_rolling/` directory alongside the default `static` run).
2. **GPU server (heavy)**: for each index × {lstm, hybrid} × {base, drop0.05,
   drop0.1, batch16, batch64}: `run_experiment.py --model <m> --index <i>
   --variant <v> --device cuda --amp`. If the box is CPU-only, add `--jobs N`
   for walk-level parallelism. Expected runtime similar to paper (~3-4 h per
   (model, index)) on CPU; substantially less on GPU.
3. **Post-processing**: `run_ensemble.py`, `make_report.py` → compare against §3 targets.

## 9. Success Criteria

- Pipeline runs end-to-end for all 3 models × 3 indices without manual intervention.
- Metrics/tables reproduce the paper's structure (Tables 2–4, 5–6, 7–10) and plots (Fig. 5-type equity curves).
- Qualitative conclusions match: LSTM-ARIMA achieves the best IR** per index in most
  strategy/index combinations. Exact numeric equality is NOT expected (paper gaps G1–G10,
  library differences TF↔PyTorch, random search seeds, yfinance data revisions).
- See REVISIONS.md for the audit trail: what was verified against the paper,
  the six correctness bugs fixed, and the paper's own internal inconsistencies.
