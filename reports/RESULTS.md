# Reproduction results

Full out-of-sample run of the pipeline against Kashif & Ślepaczuk (2024),
[arXiv:2406.18206](https://arxiv.org/abs/2406.18206), from a single 39-job sweep.

**Read [the caveats](#caveats-read-before-quoting-anything-here) before quoting
any number here.** Several of them are large enough to change what the numbers
mean.

Provenance: the tables in [`tables/`](tables/) are produced by
`scripts/make_report.py` and back every headline figure. Two blocks — the
realised-signal table and the walk-13 detail — were computed ad hoc from
per-job `predictions.csv` files, which are ~34 MB and not committed; they are
labelled where they appear.

---

## Headline

**LSTM-ARIMA takes the top IR\*\* on 2 of the 3 indices, not all 3.** It loses
on the S&P 500 and wins on the FTSE 100 and the CAC 40.

The paper's directional claim therefore partly survives an independent
implementation, but with three important qualifications:

1. The margin is far smaller. Across the 60 comparable non-ARIMA cells our
   IR\*\* runs a **median 1.03 points below** the paper's.
2. It is **not robust to hyperparameters** — LSTM-ARIMA beats plain LSTM in
   only 8 of 15 independent (index × variant) comparisons, and the S&P 500
   verdict *flips* on a single dropout change.
3. Only one cell reaches statistical significance, and it would not survive
   any correction for multiple testing.
4. The winning set is **not stable**: under a `--target return` control the
   hybrid still wins 4 of 6 cells, but on a different pair of indices.

### A note on counting

Long-Only and Long-Short are **not two independent tests**. Both derive from the
same forecast series: `src/backtest.py` computes `pos = (forecast > close)` and
then `2·pos − 1`, so the Long-Short signal is an affine transform of the
Long-Only one. Verified directly — the FTSE LSTM-ARIMA regression alphas are
0.00011872 and 0.00023765, exactly 2×, and in all 15 (index × variant) pairs the
two strategies pick the same winner.

This report therefore counts **independent comparisons** (index × variant), not
cells. Where cell counts appear they are given as "4 of 6 cells = 2 of 3
indices" so they can be compared with the paper's own presentation.

---

## What was verified first

Buy&Hold is the benchmark, not a result of the paper's method, so matching it
only shows the data pipeline and the metric formulas are right.

| index | our IR\*\* | paper IR\*\* |
|---|---|---|
| S&P 500 | 5.15 | 5.09 |
| FTSE 100 | 0.68 | 0.66 |
| CAC 40 | 0.98 | 0.98 |

Descriptive statistics reproduce the paper's Table 1 to printed precision for
all three indices, and the walk-forward grid reproduces its OOS start dates
exactly (2005-01-25 / 2005-01-13 / 2004-12-28). One Buy&Hold figure does *not*
match: the S&P 500 MLD, 5.47 against the paper's 1.65 — we believe the paper is
wrong there, and the README sets out why.

## Base case (paper Tables 2–4)

IR\*\* (%). **Bold** marks the best of the three models in each row. The ARIMA
column uses `ARIMA_FORECAST_MODE = "static"`; see [the ARIMA
caveat](#the-arima-column-was-selected-for-agreement-with-the-paper).

| index | strategy | ARIMA | LSTM | LSTM-ARIMA | B&H | paper's LSTM | paper's LSTM-ARIMA |
|---|---|---|---|---|---|---|---|
| S&P 500 | Long-Only | **0.87** | 0.25 | −0.02 | 5.15 | 1.94 | 5.79 |
| S&P 500 | Long-Short | **−1.00** | −3.02 | −5.43 | 5.15 | 3.87 | 7.18 |
| FTSE 100 | Long-Only | 0.09 | 0.63 | **4.51** | 0.68 | 1.44 | 7.19 |
| FTSE 100 | Long-Short | −0.38 | −0.14 | **2.12** | 0.68 | 0.67 | 16.65 |
| CAC 40 | Long-Only | 0.96 | 0.45 | **1.97** | 0.98 | 1.43 | 3.04 |
| CAC 40 | Long-Short | 0.03 | −0.49 | **0.24** | 0.98 | 0.97 | 14.29 |

LSTM-ARIMA wins 4 of 6 cells = **2 of 3 indices**. It beats Buy&Hold in 3 of 6
cells (both FTSE, plus CAC Long-Only).

The strongest single cell is **FTSE 100 Long-Only LSTM-ARIMA: ARC 4.25% against
Buy&Hold's 2.42%, with maximum drawdown 31.7% against 47.8%.** Three caveats
attach to it and should be read alongside:

- It was selected after the fact as the best of six.
- The lower drawdown partly reflects lower exposure — the strategy is long on
  only 30.4% of days, and anything flat most of the time shows a small MD.
- Under the `batch_64` variant the **plain LSTM** posts a higher ARC (5.09%) and
  a higher IR\*\* (6.20) on the same index and strategy. The margin here is
  therefore not attributable to the residual feature; changing the batch size
  buys more than adding ARIMA residuals does.

## The tanh output ceiling: real, verifiable, and not the explanation

The README describes a structural consequence of the paper's fixed `tanh` output
combined with a training-window scaler: the inverse-transformed forecast can
never exceed the training window's maximum close, so on any OOS day already
above that maximum the §4.8 rule `F(t) > P(t)` is false and the position is
short regardless of what the network learned.

That mechanism is real and we confirmed it exactly. Forecasts never exceed the
ceiling in **19 of 19 walks** — the ceiling is not merely rarely crossed, it is
mathematically uncrossable. Walk 13 is fully saturated: its training window tops
out at 2271.72, its OOS year trades between 2546 and 2931, and the forecast is
**exactly 2271.72 on all 250 days** for both neural models. How often the
ceiling binds:

| index | OOS days above the ceiling | walks fully pinned |
|---|---|---|
| S&P 500 | 55.5% (2600/4682) | 7 / 19 |
| FTSE 100 | 25.7% (1210/4704) | 2 / 19 |
| CAC 40 | 35.4% (1691/4778) | 1 / 20 |

*Realised S&P 500 signals (computed ad hoc from uncommitted `predictions.csv`):*

| model | forecasts pinned at the ceiling | days long | days short |
|---|---|---|---|
| LSTM | 5.3% | 18.2% | 81.8% |
| LSTM-ARIMA | 5.3% | 16.9% | 83.1% |
| ARIMA (unscaled, no ceiling) | 57.7% | 32.5% | 67.5% |

The neural models are short on 82–83% of S&P 500 days, in an index that rose
3.88× over the evaluation window.

**But the ceiling cannot be what makes LSTM-ARIMA lose on the S&P 500**, and it
is worth being explicit that our own sensitivity data says so:

| S&P 500 variant | Long-Only: hybrid vs LSTM | Long-Short: hybrid vs LSTM |
|---|---|---|
| base (dropout 0.075) | −0.03 vs 0.25 — loses | −5.43 vs −3.02 — loses |
| **dropout 0.05** | **4.01 vs 0.24 — wins** | **−0.94 vs −3.13 — wins** |
| dropout 0.10 | 0.23 vs 0.21 — wins | −2.55 vs −3.18 — wins |
| batch 16 | 0.00 vs 0.61 — loses | −5.19 vs −1.83 — loses |
| batch 64 | 0.47 vs 1.32 — loses | −2.35 vs −1.30 — loses |

The ceiling is a property of the price series and the scaler. It is **identical
across every one of these variants**. Yet changing dropout from 0.075 to 0.05
flips both S&P 500 cells from hybrid losses to hybrid wins. A cause that does
not vary cannot explain an outcome that does.

Two further points against reading the ceiling as the explanation:

- It applies **equally to both neural models**, so it cannot discriminate the
  hybrid from the plain LSTM — but the S&P 500 failure is precisely the hybrid
  losing to the plain LSTM.
- Full-year saturation is not S&P-specific. FTSE walk 13 is pinned at exactly
  7104.00 for all 250 days too — on the index where the hybrid wins.

The honest reading: **the ceiling is a genuine defect that depresses all neural
results, most severely on the S&P 500; the hybrid-versus-LSTM verdict is
governed by something else, and on this evidence that something else is
hyperparameter luck.**

## Sensitivity (paper Tables 7–9)

Counting independent (index × variant) comparisons:

```
LSTM-ARIMA beats LSTM in 8 of 15
```

| variant | LSTM-ARIMA beats LSTM |
|---|---|
| dropout 0.05 | 3 / 3 |
| base (dropout 0.075) | 2 / 3 |
| dropout 0.10 | 1 / 3 |
| batch 16 | 1 / 3 |
| batch 64 | 1 / 3 |

Slightly better than a coin flip. The paper reaches the same qualitative
conclusion about its own results in §6 (RQ3), so this is agreement rather than
divergence — but the practical implication is blunt: **outside the base case and
the dropout-0.05 variant, the ARIMA residual feature buys nothing measurable
here.**

Against the paper, over the 60 non-ARIMA cells that have a published
counterpart:

```
median difference   −1.03 IR** points
mean difference     −2.00
cells where we are higher   19 / 60
```

(Including the 18 ARIMA cells as well, over all 78: median −0.61, mean −1.68,
30/78 higher.)

## Statistical significance (paper Tables 5–6)

**Paired t-test**, H₁: μ_strategy − μ_benchmark > 0, α = 10%. No cell is
significant; the smallest p is 0.376.

That one-sided framing hides a real finding, so state it directly: **on the S&P
500 all three models underperform Buy&Hold significantly at conventional
two-sided levels** — LSTM-ARIMA t = −2.66 (p ≈ 0.008), LSTM t = −2.09
(p ≈ 0.037), ARIMA t = −1.89 (p ≈ 0.058). This is not an absence of evidence; it
is evidence of underperformance.

The paper reports ARIMA Long-Only as significant for all three indices. We do
not reproduce that. (The README documents why that Table 5 result is also
internally inconsistent with the paper's own Tables 3–4.)

**Regression alpha**, R_strategy = α + β·R_benchmark, right-tailed test on α.
One model clears 10%:

| index | strategy | model | α | t | p |
|---|---|---|---|---|---|
| FTSE 100 | Long-Only | LSTM-ARIMA | 0.000119 | 1.4315 | 0.0762 |
| FTSE 100 | Long-Short | LSTM-ARIMA | 0.000238 | 1.4327 | 0.0760 |

These are one result, not two — the Long-Short row is the same forecast series
doubled. Everything else has p ≥ 0.21.

**This should not be over-read.** Nine independent tests at α = 0.10 produce
about one false positive under the null, which is exactly what we observe. A
Bonferroni threshold at nine tests would be p < 0.011; 0.076 misses it by
sevenfold, and the paper's own corresponding value (0.0268, quoted from its
Table 6) misses it too. What can be said is that the one cell clearing the
uncorrected threshold is **the same cell in both the paper and this
reproduction** — suggestive, and worth a targeted follow-up, but not on its own
evidence of a real effect.

Long-Short betas: the two neural models are near zero (−0.12 to +0.007), but the
**ARIMA Long-Short strategies retain significantly positive beta** (0.19 to 0.28,
t = 13 to 20) — they are net long over the sample rather than market-neutral.

## Assessment

| claim | verdict |
|---|---|
| Data pipeline and metrics reproduce | ✅ Table 1 to printed precision; Buy&Hold within 0.06 IR\*\*, except the S&P MLD (we believe a paper error) |
| LSTM-ARIMA > LSTM and ARIMA | ⚠️ 2 of 3 indices in the base case; 8 of 15 across variants |
| LSTM-ARIMA > Buy&Hold | ⚠️ 3 of 6 cells |
| Magnitudes match the paper | ❌ median 1.03 IR\*\* points lower |
| Robust to hyperparameters | ❌ — and the paper agrees (RQ3) |
| Statistically significant | ❌ one uncorrected result; would not survive correction |

**The qualitative direction of the paper's finding survives an independent
implementation on two of three indices, but the effect is much smaller than
reported, is not robust to hyperparameters, and is not statistically significant
once multiple testing is accounted for.**

---

## Caveats: read before quoting anything here

**Single seed, single random-search draw.** Every LSTM and LSTM-ARIMA number
comes from `--seed 0` and one 20-of-216 hyperparameter draw per walk, which
bears no relation to the paper's draw. No seed-variation study was run. The
scale matters: across the five hyperparameter variants, S&P 500 LSTM-ARIMA
Long-Only spans −0.02 to +4.01 IR\*\*, while the FTSE base-case
hybrid-over-LSTM margin this report calls a win is 3.89. **The effects being
reported are the same size as the run-to-run spread.** Individual cell verdicts
should be read as one draw, not as a measurement.

**Price returns only; not investable performance.** `src/data.py` uses
`auto_adjust=False`, so Buy&Hold **excludes dividends** — roughly 2%/yr on the
S&P 500 and 3.5–4%/yr on the FTSE 100. Idle cash earns 0% while the Long-Only
strategies sit flat most days. Shorting is frictionless beyond the 0.1% cost. And
none of the three indices is directly tradeable. These choices follow the paper,
but they mean the return comparisons here are not investable performance, and
the FTSE "beats Buy&Hold" result in particular is measured against a benchmark
missing most of its return.

### The ARIMA column was selected for agreement with the paper

The paper never states whether the fitted ARIMA is re-conditioned on observed
closes as the OOS window advances (README gap G10). This repo implements both
readings. The ARIMA column above uses `static`, which was made the default
*because it reproduces the paper* — the README's own words for the alternative
`rolling` are "the implementable reading, and the more defensible strategy".
Note also that the hybrid's residual feature always uses `rolling`, so the
"LSTM-ARIMA vs ARIMA" comparison above is against a different ARIMA
specification than the one inside the hybrid. Under `rolling`, every ARIMA cell
is deeply negative (−0.26 to −28.3).

**The severity of the tanh ceiling is our choice, not the paper's.** The paper
fixes `tanh`; it specifies no scaling at all. MinMax to [−1, 1] fit on the
training window is the most common leak-free choice, but it is not the only one
— mapping the training range to a sub-interval of tanh's codomain (e.g.
[−0.5, 0.5]), scaling log prices, or regressing returns would all relax the
ceiling substantially. The ceiling's *existence* follows from the paper's fixed
`tanh`; its *severity* follows from our scaler.

## The return-target control

`--target return` regresses the forward `seq_len`-day return instead of the
price level, so the signal reduces to `r̂ > 0` and the ceiling does not exist.
Six jobs, base variant, same seed and same environment.

**The control works as designed.** Short-day share falls sharply, most where the
ceiling bound hardest:

| index | ceiling binds | LSTM short: level → return | LSTM-ARIMA short: level → return |
|---|---|---|---|
| S&P 500 | 55.5% | 81.8% → **50.5%** | 83.1% → **49.1%** |
| FTSE 100 | 25.7% | 72.7% → 55.5% | 69.6% → 57.4% |
| CAC 40 | 35.4% | 75.0% → 60.1% | 72.6% → 55.8% |

And the change in absolute performance tracks how tightly the ceiling bound:

| index | ceiling binds | LSTM-ARIMA IR\*\* level → return |
|---|---|---|
| S&P 500 | 55.5% | −0.02 → 0.49 (**+0.52**), −5.43 → −1.46 (**+3.97**) |
| CAC 40 | 35.4% | 1.97 → −0.00 (−1.98), 0.24 → −1.75 (−1.99) |
| FTSE 100 | 25.7% | 4.51 → −0.16 (−4.67), 2.12 → −2.09 (−4.21) |

So the ceiling is confirmed as a genuine, quantified drag on the S&P 500 — the
index where it binds on 55.5% of days is the only one that improves when it is
removed.

**But it does not rescue the paper's claim, and it makes the instability
worse.** Full comparison, IR\*\* (%), `*` marking LSTM-ARIMA beating LSTM:

| index | strategy | LSTM level | LSTM return | LSTM-ARIMA level | LSTM-ARIMA return | paper |
|---|---|---|---|---|---|---|
| S&P 500 | Long-Only | 0.25 | 0.00 | −0.02 | **0.49\*** | 5.79 |
| S&P 500 | Long-Short | −3.02 | −3.77 | −5.43 | **−1.46\*** | 7.18 |
| FTSE 100 | Long-Only | 0.63 | 0.56 | **4.51\*** | −0.16 | 7.19 |
| FTSE 100 | Long-Short | −0.14 | 0.00 | **2.12\*** | −2.09 | 16.65 |
| CAC 40 | Long-Only | 0.45 | −0.12 | **1.97\*** | **−0.00\*** | 3.04 |
| CAC 40 | Long-Short | −0.49 | −3.16 | **0.24\*** | **−1.75\*** | 14.29 |

LSTM-ARIMA wins 4 of 6 cells under **both** targets — but **on different
indices**. Under the level target it wins FTSE 100 and CAC 40; under the return
target it wins S&P 500 and CAC 40. Changing one specification the paper never
states moves the winning set wholesale while leaving the count untouched.

Nothing under either target approaches the paper's magnitudes: the best
return-target cell is 0.49 against the paper's 5.79.

**This is the strongest single piece of evidence in the report.** Combined with
the dropout-0.05 flip, the hybrid-versus-LSTM verdict has now been shown to
change under two independent perturbations — a hyperparameter the paper *does*
specify, and a target definition it does *not*. On this evidence, the ordering
between LSTM-ARIMA and plain LSTM in any single cell is not a measurement of the
residual feature's value; it is within the noise of the specification choices.

## Open

- **^GSPC ARIMA Long-Short** remains unexplained: −1.00 against the paper's
  7.13, with the gap stable at −8.1 to −9.0 across all three ARIMA variants
  while the Long-Only cells agree within 0.4.
- **No seed-variation study.** Everything here is one draw. The two
  perturbations above establish that cell-level verdicts are unstable; a proper
  seed sweep would quantify the noise band directly. That is the obvious next
  step and it was not run.

## Run details

```
hardware      RunPod: RTX 4090, AMD EPYC 75F3 (128 vCPU), 503 GB
software      Python 3.11.10, torch 2.4.1+cu124, pandas 3.0.5, numpy 2.4.6
sweep         39 jobs level-target + 6 return-target control + 3 rolling-ARIMA,
              all seed 0, same environment
concurrency   12 processes on one GPU
result        48/48 completed, 0 failures
tests         58 passing
```

Job durations sum to 20.96 h of single-process work; the longest single job was
69 minutes. Per-trial cost was 1.64 s on the RTX 4090, against 3.45 s on an
Apple M5 Pro performance core and 33.67 s on an EPYC thread — see the README's
performance section.

Raw per-job outputs (predictions, equity curves, per-walk chosen
hyperparameters) are ~34 MB and not committed. The five summary tables this
report is built from are in [`tables/`](tables/).
