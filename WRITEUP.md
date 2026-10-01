# Path Signatures for Market Anomaly Detection and Volatility Forecasting — Progress Writeup

This covers what the team has tried so far, why, and what came out of it. It
draws on Aayush's and Mihika's code, notebooks and saved outputs. Mark's
directory is not covered. All numbers come from notebook outputs and CSVs in
the repo. Where a number was not saved, the text says so.

## 1. Project motivation

Anomaly detection in financial markets usually relies on volatility models,
statistical process control, or ML on handcrafted features. Those summarize a
window of prices with pointwise statistics (return, vol, range) and discard the
order in which things happened. **Path signatures** (iterated integrals from
rough path theory) encode the geometry of an ordered path and are
"universal" features of it. The reference paper, Gasteratos et al., *Novelty
Detection on Path Space* (arXiv:2512.03243), uses them for novelty detection
and hypothesis testing. It validates on synthetic Brownian-motion-with-a-spike
data and nanopore RNA signals, **not** on finance. Applying it to market data is
this project's own contribution.

Goals:
- Flag known disruptions: the 2020 COVID crash, the 2021 retail-mania run-up and
  crash, the 2022 Terra/Luna and FTX crashes, and the 2023 banking crisis.
- Detect bull/bear and high/low-volatility regime transitions.
- Test whether signatures carry information beyond standard summary statistics.

Work so far splits into three threads:

| Thread | Owner | Question |
|---|---|---|
| A. Planning, data sources, signature benchmarking | team / Mihika | Do signatures capture path shape that summary stats miss? How deep? |
| B. Daily-horizon novelty detection on BTC | Aayush | Can the paper's methods flag historical events and regime changes? |
| C. Short-horizon anomaly prediction and volatility forecasting | Mihika | Do signatures help predict abnormal windows or future volatility? |

## 2. Data

| Source | What it gives | Used for |
|---|---|---|
| Yahoo Finance (`yfinance`) BTC-USD | Daily OHLCV from 2014-09 to 2026-09 (4,369 rows; the volatility study uses 2015-01 onward) | Macro-horizon detection (B); daily/weekly/monthly vol forecasting (C) |
| Binance public archive (`data.binance.vision`) | Trade-by-trade ticks and 1-min klines, no API key | Intraday and micro horizons (B); 5-min/1-hour anomaly and vol work (C) |

Why this mix: Yahoo never exposes trade-level data. It only stores pre-built
bars (daily back to ~2014, hourly ~2 years, 1-minute ~7 days). So it gives the
multi-year history needed to cover all the labeled events. Binance gives real
tick data, but a day of BTCUSDT trades is roughly 900k rows. Tick pulls are
therefore short windows only. The volatility study (C) streams each Binance
day into 5-second and 1-minute bars and keeps only those (2026-07-01 to
2026-09-15), because loading raw ticks for that range crashed the notebook
kernel. In Aayush's pipeline the 2023-10-01 to 2023-10-05
tick reference alone was ~98.8M trades.

Dukascopy (via the TheoryCraft library) and CoinDesk were also researched as
sources for tick/second-level and bid/ask data. Neither ended up feeding any
experiment. The code uses Yahoo and Binance only.

## 3. Thread A — Benchmarking signatures on synthetic paths (Mihika, `explore.ipynb`)

**Motivation.** Before using signatures for anomaly detection, check that they
capture path shape at all, and pick a truncation depth.

**What was done.**
- Wrote a signature implementation based on Chen's identity.
- Generated six synthetic path types, 200 each (1,200 total), with noise: uptrend,
  downtrend, crash-then-recovery, spike-then-reversal, oscillating, flat.
- Classified the type from signature features with logistic regression
  (5-fold stratified CV), at depths 1–4.

**Result.**

| Depth | # features | Accuracy |
|---|---|---|
| 1 | 2 | 49.8% |
| 2 | 6 | 82.6% |
| 3 | 14 | **100%** |
| 4 | 30 | 100% |

Depth 3 matches depth 4 at roughly half the feature count, so **depth 3** was
chosen for later work. Depth 1 is essentially chance. This is expected, since
level 1 only holds net increments, and several classes (crash-recovery,
spike-reversal, oscillating, flat) all end near zero net change. Caveat: the synthetic classes are idealized piecewise-linear
shapes, so 100% shows signatures *can* separate shapes, not that they will on
noisy market data.

Not done: the planned comparison of signatures against summary stats and raw
paths *on the synthetic data* (the plan listed it). The summary-vs-signature
comparison was run on real data instead (§5).

## 4. Thread B — Novelty detection with the paper's methods (Aayush)

### 4.1 Pipeline

Raw bars → rolling windows → multichannel path (time augmentation, cumulative
log-return, basepoint) → per-channel normalization fit on calm periods only →
truncated signature → one-class/novelty scorer.

Design choices and why:
- **Three horizons** (micro: 1000-trade windows; intraday: 48 hourly bars; macro:
  30 daily bars) to test multiple scales. Only **macro** was taken through full
  evaluation. Micro and intraday feature matrices exist (5,947×39 and 75×120)
  but were never scored or event-validated.
- **Pure-NumPy signatures** (Chen's identity) instead of `iisignature`/`esig`,
  because the C++ toolchain on that machine could not compile them. The
  implementation was checked against hand-computed values and concatenation
  invariance. The same limitation led to a hand-rolled Gaussian HMM in place of
  `hmmlearn`.
- **Macro channels:** time, log-return, volume, log-range, overnight gap.
  Depth 4, giving 780 signature features per window. 434 windows with a stride
  of 10 days.
- **Reference ("normal") data.** The first version calibrated on a single
  ~2.5-month 2023 stretch. That left only a few dozen windows and caused
  warnings (LOF `n_neighbors` override, rank-deficient EllipticEnvelope). It was
  replaced by **13 auto-detected calm stretches (2015–2026)**: low-vol runs of
  at least 45 days and at least 30 days clear of any labeled event. These give 52 train
  and 80 held-out calibration windows (132 of 434). Calibration is on held-out
  windows, not the training ones. The earlier version's threshold had been
  calibrated in-sample.

### 4.2 Models tried

| Model | Idea | Status |
|---|---|---|
| Distance to expected signature (ESD) | `‖S_N(x) − E[S_N(X)]‖`, isotropic, grows with any deviation | Best-behaved signature model |
| Conformance score | Mahalanobis distance under the reference covariance | Matches ESD |
| CVaR-OCSVM (Thm 2.6) | Shuffle-product reduction of a CVaR one-class SVM to a function of the expected signature | Failed on real data (see below) |
| Weibull tail + BH-FDR | Paper's §4.1 thresholding and multiple-testing control | Used for thresholds |
| Signature-MMD | Two-sample MMD of a trailing batch of 5 windows vs reference | Added after a June 2025 paper (Alden et al.) |
| Recency-weighted ESD, step=1 | Exponentially down-weight older increments (half-life 8 days) and score every day | Best signature variant |

### 4.3 Validation, following the paper's order

**Step 1, synthetic spike injection.** Inject a spike of known size into
held-out reference paths and measure AUROC against clean paths. Magnitudes are
scaled to the channel's own std. The first try used fixed absolute sizes, and
those were negligible next to real 30-day BTC swings, so AUROC came out flat.

| Model | AUROC at 0× | at 8× std |
|---|---|---|
| ESD | 0.509 | 0.979 |
| Conformance | 0.506 | 0.976 |
| Recency-weighted ESD | 0.485 | 0.997 |
| Signature-MMD | 0.509 | 0.979 |
| CVaR-OCSVM | 0.489 | **0.013** (score runs the wrong way) |

ESD, conformance and MMD rise monotonically with spike size, so the pipeline
works. The CVaR-OCSVM AUROC falls toward 0, meaning anomalies receive *lower*
scores than normals.

**Step 2, historical events (alpha = 0.01 Weibull threshold).** "Detected"
means the score crossed the threshold anywhere in the labeled event window.
Lag is measured from the event start.

| Event | ESD (10-day step) | Recency-weighted ESD (daily) |
|---|---|---|
| COVID crash | Yes, 27d | Yes, 21d |
| 2021 mania | Yes, 1d | Yes, 1d |
| Terra/Luna | Yes, 16d | Yes, 8d |
| FTX | Yes, 12d | Yes, 8d |
| 2023 banking | **Missed** | **Missed** |
| False-positive rate outside events | **21.6%** | **6.0%** |
| Bull/bear transition hit rate (±15d) | 28.9% | 38.2% |

Conformance gives the same detections as ESD (FPR 20.6%). CVaR-OCSVM detects
0/5, with FPR 3.0% and a 2.6% hit rate.

**CVaR-OCSVM.** It was verified against the shuffle identity and small
synthetic tests. On real BTC data the optimizer pushed `‖w‖` to its bound
instead of settling, and the Weibull fit failed (scores go negative for normal
points). The likely cause is that a degree-2 polynomial is a poor surrogate for
the hinge over fat-tailed return ranges. It also had to run at depth 2 instead of
4, because the shuffle powers need signature levels up to depth × degree.
Conclusion: treated as a negative result.

### 4.4 Breadth: is a signature even the right representation?

Other methods were run on the same event/regime setup. Thresholds here are the
empirical 99th percentile of reference scores.

| Method (features) | Events | FPR | Bull/bear hit |
|---|---|---|---|
| One-class SVM (7 engineered stats) | 5/5 | 34.5% | 59.2% |
| Isolation Forest (engineered) | 2/5 | 7.7% | 15.8% |
| LOF (engineered) | 2/5 | 15.9% | 17.1% |
| Elliptic Envelope (engineered) | 1/5 | 3.2% | 9.2% |
| ECOD (engineered) | 1/5 | 1.7% | 11.8% |
| Matrix profile discord (stumpy, raw returns) | 1/5 | 1.1% | 22.4% |
| Gaussian HMM (hand-rolled, high-vol state) | **5/5**, 0–3d lag | **56.2%** | 100% |
| Signature-MMD | 4/5 | 31.8% | 44.7% |
| Two-stage ensemble (HMM flags, ESD confirms within ±15d) | 4/5 | 23.3% | 60.5% |
| PELT changepoints (`ruptures`, pen=10) | 0/5 | n/a | n/a |

Changepoint penalty sweep: pen=10 found only 3 changepoints in 11 years and none
near an event. Lowering it to pen=1 gave 5/5 nearby events but 132 changepoints,
88.6% of them outside every event window (about one every 33 days). That "5/5" is
trivial and not a real detection.

Hyperparameter sweep, recency half-life (daily ESD):

| Half-life | Events | FPR |
|---|---|---|
| 4d | 3/5 | 4.1% |
| 6d | 3/5 | 4.8% |
| **8d** | 4/5 | 6.0% |
| 10d | 4/5 | 7.3% |
| 12d | 4/5 | 8.7% |
| 15d | 4/5 | 11.0% |

The 8-day default had been chosen as the midpoint of an agreed 7–10 day range.
The sweep supports it as a reasonable operating point.

### 4.5 Takeaways for Thread B

1. Signature distance scores pass the synthetic check cleanly, and recency
   weighting plus daily scoring is the best signature variant (4/5 events, 6%
   FPR, lag cut from 27d to 21d on COVID and from 16d to 8d on Terra/Luna).
2. Events, not FPR alone, are the wrong yardstick. Event windows are 1–5
   months long, so any scorer that fires often "detects" them. The HMM and the
   engineered-feature OCSVM score 5/5 only by flagging 56% and 34% of all days. The
   signature scorers are much more selective, but also slower and still miss
   the 2023 banking crisis.
3. Calibration is the weak point. Realized FPR (6–22%) sits far above the 1%
   nominal alpha, because of a small number of overlapping calibration windows.
   Daily scoring partly fixes this by keeping calibration on the coarse stride.
4. There is no evidence yet that signatures *beat* simple alternatives on
   detection. A plain HMM is faster, and engineered-feature methods are in the
   same range. Hyperparameters (half-life, changepoint penalty) were tuned
   against the same five events they are evaluated on, so the numbers are
   optimistic.
5. Notable extra: the score also spikes around the 2017–18 crypto bubble crash,
   which was not a labeled event. That suggests real signal beyond the chosen
   labels. It was a visual observation on the plots, not quantified.

## 5. Thread C — Short-horizon anomaly prediction and volatility forecasting (Mihika)

### 5.1 5-minute anomaly prediction (`explore.ipynb`)

**Setup.** One week of Binance BTCUSDT trades, cut into 2,016 non-overlapping
5-minute windows (64 resampled points each).

Paths: price (time + log-price) and price + signed order-flow imbalance, each
with and without lead-lag. Summary-stat baselines: return, realized vol, range,
and imbalance.

**Exploratory comparison (no labels).** Mahalanobis scores
(Ledoit-Wolf covariance) were computed for summary stats and for two
signature sets. The aim was to find windows that signatures rank as unusual and
summary stats do not, and to inspect them against the average path and its
25–75% band.

**Supervised test: predict whether the *next* window is abnormal.** "Abnormal"
means a 5-minute return beyond ±2 std of the training period (4.6% of windows,
93 total). One-class SVM and Isolation Forest were trained only on normal
current windows, with a chronological split of 1,410 train and 605 test pairs
(45 abnormal in test).

| Features | Model | Bal. acc. | F1 | ROC AUC |
|---|---|---|---|---|
| Price lead-lag signature | OCSVM | 0.646 | 0.267 | 0.617 |
| Summary: price | IsoForest | 0.645 | 0.265 | **0.689** |
| Price signature | OCSVM | 0.627 | 0.248 | 0.638 |
| Price + imbalance lead-lag signature | OCSVM | 0.587 | 0.193 | 0.626 |
| Summary: price + imbalance | OCSVM | 0.564 | 0.176 | 0.595 |

**Takeaway.** Signatures did not clearly beat summary statistics. The best F1 is
a signature set, but the best AUC is a summary-stat set. Everything sits at
AUC 0.60–0.69 with precision around 0.13–0.19 against a 7.4% test base rate.
Adding order-flow imbalance tended to *hurt*. Predicting an extreme next-5-min
return from the current window is hard, and no representation solved it. The
results come from a single week of data and a single split, so the model rankings
are within noise.

**Follow-up: 30-minute realized-vol forecast** (same data). HAR (R² 0.339, RMSE
0.1004) beat both a lead-lag signature model (R² 0.103) and HAR plus a
signature residual correction (R² 0.294, slightly *worse* than HAR alone). This
was the first signal that classical HAR is hard to beat at short horizons, and it
led to the dedicated forecasting study below.

### 5.2 Volatility forecasting experiment (`volatility_experiments.ipynb`, `experiment.py`, `reporting.py`)

**Motivation.** Anomaly labels are arbitrary, but future realized volatility is
an objective target. It gives a cleaner test of whether signature features carry
predictive information, against standard econometric baselines.

**History.** The first version used two notebooks with different folds, caps and
settings. Its walk-forward folds were not purged, and the path dimensions were
chosen on the same holdout the comparison was scored on. Both notebooks were
replaced by one runner. Everything below comes from the run in the committed
notebook (`results/main`). Its numbers supersede the earlier ones.

**Design.**
- **Data.** Yahoo BTC-USD daily bars from 2015-01 to 2026-09 for the 1/7/30-day
  horizons, with no cap and no stride. Binance BTCUSDT from 2026-07-01 to
  2026-09-15 (77 days) for 5 min and 1 hour, streamed one day at a time into
  5-second and 1-minute bars. The intraday horizons use a sampling stride so each
  has about 4,000 forecast rows.
- **Target.** Realized volatility over the next horizon (Garman-Klass variance
  for daily bars, close-to-close for intraday).
- **CV.** Three expanding-window walk-forward folds on four consecutive blocks.
  Block 0 is train-only. Any training row whose label window reaches the first
  test time is purged. Every model sees identical folds, and Ridge's alpha is
  chosen by an inner purged CV.
- **Signature specification, fixed in advance.** Depth 3, lead-lag plus time
  augmentation, full path: price, activity, and signed flow (Binance) or high-low
  range (Yahoo). That is 399 signature features. The ablations below did not
  choose it.
- **Models.** Mean, HAR-RV-L, GARCH(1,1), and Ridge, XGBoost and MLP on summary
  stats, on signatures, or on both, plus a signature LSTM. In the run reported in
  §5.3, Ridge's alpha was tuned but XGBoost (500 trees, depth 3), the MLP
  (128-64-16, 200 epochs) and the LSTM used fixed hyperparameters, and GARCH was
  fit once and held fixed. §5.4 explains why that limits the conclusions, and
  the code has since been changed (see "Fixes since this run" there).
- **Tests.** Diebold–Mariano on pooled out-of-sample errors, with the lag set to
  the horizon length in forecast rows.

| Horizon | Rows | Independent targets (approx.) | Fold test periods |
|---|---|---|---|
| 5 min | 3,992 | ~3,990 (non-overlapping), but all from 77 days | Jul 20–Aug 8, Aug 8–27, Aug 27–Sep 15, 2026 |
| 1 hour | 3,907 | ~1,300 | same |
| 1 day | 4,262 | ~4,260 | Dec 2017–Nov 2020, Nov 2020–Oct 2023, Oct 2023–Sep 2026 |
| 7 day | 4,256 | ~600 | same |
| 30 day | 4,233 | ~140 | same |

The daily folds cover very different eras. Fold 1 contains the 2018 bear market
and the 2020 COVID crash. Fold 2 contains the 2021 run-up, the 2022 crashes,
the FTX collapse and the 2023 banking crisis. Fold 3 is the recent, lower-volatility period.

### 5.3 Results

*The results in this section come from the run before the fixes described in
§5.4: fixed hyperparameters for XGBoost, MLP and LSTM, a fixed-parameter GARCH,
and no signature-free controls.*

**Model ranking** (fold-mean RMSE; lower is better; `%HAR` is the RMSE
improvement over HAR-RV-L). `p` is the pooled Diebold–Mariano p-value against HAR-RV-L.

| Horizon | Best models | RMSE (%HAR, p) | HAR-RV-L | Mean | Worst |
|---|---|---|---|---|---|
| 5 min | Ridge stats+sig / Ridge stats / Ridge sig | 4.11e-4 (+12.3%, 0.004) / 4.25e-4 (+9.2%, 0.025) / 4.43e-4 (+5.5%, 0.20) | 4.68e-4 | 6.13e-4 | Sig. LSTM 7.19e-4 (−54%) |
| 1 hour | **HAR-RV-L** | 1.49e-3 | best | 2.06e-3 | Ridge sig 1.83e-3 (−22%, p<1e-10) |
| 1 day | Ridge stats+sig / Ridge sig / XGB sig | 1.500e-2 (+8.5%, 5e-4) / 1.502e-2 (+8.4%, 8e-4) / 1.519e-2 (+7.3%, 0.002) | 1.639e-2 | 1.862e-2 | MLP stats 2.31e-2 |
| 7 day | Ridge sig / Ridge stats+sig | 3.06e-2 (+12.6%, 2e-6) / 3.11e-2 (+11.4%, 5e-4) | 3.51e-2 | 3.95e-2 | XGB stats 4.89e-2 (−40%) |
| 30 day | Ridge stats+sig / Ridge sig | 5.65e-2 (+13.0%, 0.011) / 5.67e-2 (+12.7%, 0.025) | 6.50e-2 | 6.88e-2 | XGB stats 1.06e-1 (−63%) |

- **Daily and longer.** Ridge on signatures (alone or with stats) is the best or
  second-best model at 1, 7 and 30 days. Ridge on stats alone is statistically
  indistinguishable from HAR (p = 0.66 at 1 day, 0.74 at 7 days, 0.97 at 30 days).
  At 5 min, Ridge on stats+signatures is nominally best, but the difference
  from Ridge on stats alone is not significant (p = 0.54).
- **1 hour.** HAR-RV-L is best and significantly better than every other model.
- **Nonlinear models.** XGBoost, MLP and LSTM beat HAR only sporadically and
  never significantly at 7 or 30 days. XGBoost and MLP on stats alone are far worse
  than the mean forecast at 7 and 30 days.
- **GARCH(1,1).** Worse than the mean forecast at 7 days (−3%) and 30 days (−35%),
  and only 2.6% better than the mean at 1 day.
- **At 30 days HAR itself is barely better than predicting the mean** (5.7% lower RMSE; p = 0.40).

**Fold stability.** Ridge on signatures beats the mean forecast in all three
folds at every horizon. HAR-RV-L is worse than the mean in fold 3 at 30 days,
and GARCH is worse than the mean in most folds. The LSTM is unstable: in fold 1
its RMSE is 2.5× the mean forecast at 5 min and 1.5× at 1 hour.

**Signature-construction ablations** (Ridge on signatures, same folds; % change
in RMSE against each control; positive is better). These are descriptive. They
did not select the main specification.

| | 5 min | 1 hour | 1 day | 7 day | 30 day |
|---|---|---|---|---|---|
| Levels 1–2 vs level 1 | +11.5 | +7.8 | +7.8 | +8.0 | +6.0 |
| Levels 1–3 vs level 1 | +28.9 | +11.6 | +12.1 | +14.7 | +8.8 |
| Lead-lag + time vs raw path | +9.3 | +5.4 | +9.4 | +7.5 | +0.3 |
| Time only vs raw path | −0.6 | +2.8 | +4.8 | +6.7 | −0.4 |
| Price + activity vs price only | +11.4 | +17.1 | +4.0 | +14.6 | −1.5 |
| Full path vs price only | +11.2 | +17.2 | +12.7 | +26.8 | +10.1 |

**Calm vs stressed periods.** Terciles of realized volatility in the
out-of-sample target. Relative to HAR, Ridge on signatures does much better in
the low and middle terciles at 1, 7 and 30 days (roughly +30–45%), but is worse
than HAR in the highest-volatility tercile (roughly −5% to −20%). The nonlinear
models lose heavily in the lowest tercile at 5 min and 1 hour.

### 5.4 What the evidence supports

**Reasonably supported.**
1. **At daily to weekly horizons a regularized linear model on depth-3
   signatures forecasts volatility better than HAR-RV-L.**
   - The gain is 8% at 1 day and 12.6% at 7 days, with p-values of 1e-3 to 1e-6.
   - It holds in all three folds, which cover different eras of BTC.
   - The 1-day test has about 4,260 targets and the 7-day test about 600.
   - It survives purging, which removed the concern that the earlier result came
     from label overlap. The first, unpurged run gave a similar direction, but
     many things changed at once, so the two cannot be compared cleanly.
2. **Signatures do not help at 1 hour**, and beat HAR at 5 min only when combined
   with summary stats, where they add nothing demonstrable over stats alone.
3. **The main spec was fixed in advance**, so the headline numbers were not
   picked from the ablations.
4. **Using only the price path throws away most of the benefit.** In the
   dimension ablation, price-only is the worst at every horizon. Activity and
   range or flow channels carry much of the gain.

**Suggestive, but not established.**
- **30 days.** There are only about 140 independent targets, the p-values are
  0.011–0.025, and HAR is barely better than the mean. With about 50
  model-and-horizon comparisons against HAR, a single p of 0.02 is not strong
  evidence. A 10% gain with this much noise could well shrink on new data.
- **5 min.** Ridge on signatures alone is not significantly better than HAR
  (p = 0.20), and the data cover only 77 days of one market regime.

**Why the signature result should not be over-read (in the run above).**
- **No dimension-matched control.** Ridge on 399 signature features was compared
  with HAR (4 features) and Ridge on 11 summary stats, never with a
  signature-free feature set of the same size. The gain could come from the
  signature acting as a rich multi-scale realized-volatility feature set.
  Depth also matters: more levels helped monotonically up to the deepest tested
  level (3), so we do not know where the gain saturates.
- **Unequal tuning.** Ridge's alpha was tuned by CV. XGBoost, the MLP and the
  LSTM used fixed, untuned hyperparameters. "Linear beats nonlinear" was partly a
  tuning-effort result, not a conclusion that nonlinear models cannot help.
- **GARCH was probably handicapped.** It used parameters fit once on the
  training period and held fixed. In a market whose volatility fell over time,
  long-horizon forecasts revert to a long-run variance estimated from earlier,
  more volatile years, which would explain its overshoot at 30 days (−35%
  against the mean). This was a hypothesis and had not been checked, so the
  result does not show that signatures beat GARCH in general.
- **One asset, one history, three folds.** The folds are consecutive blocks of
  one time series, not independent draws. Consistency across three eras is
  reassuring, but it says nothing about other assets.
- **RMSE is dominated by crisis periods**, so pooled results lean on fold 1
  (2018–2020) and its large errors. The loss differential is not stationary
  across folds, which weakens the Diebold–Mariano p-values. A robust loss such as QLIKE
  was dropped earlier and should be restored.
- **Stress periods.** The calm-vs-stressed picture is conditioned on the
  realized outcome, so models that smooth forecasts look good in calm periods and
  poor in spikes by construction. Even so, signature Ridge does not beat HAR when
  volatility is highest, which is where forecasts matter most for risk. This
  motivates the anomaly flag below.
- **Short intraday window.** 5 min and 1 hour come from 77 days of one regime,
  so those results are the weakest in the study.

**Fixes since this run (in the code; results pending a re-run).** The first
three issues above are addressed in `experiment.py`, `models_and_metrics.py`,
`forecast_data.py` and `reporting.py`, and verified for wiring on synthetic data
only:
1. **Signature-free controls**, run through the same Ridge and the same folds.
   - `Ridge | lag bank (matched)`: the same base channels at the same resolution,
     as a flat bank of raw, squared, cross-channel and lagged increments with
     *exactly as many columns as the signature* (399 here).
   - `Ridge | multiscale stats`: a hand-built set of multi-scale realized
     variance, absolute return, range, volume, signed return, downside
     semivariance, largest move and flow at five look-back scales.
   - `rp.control_table` reports signature vs each control (RMSE, % difference,
     Diebold–Mariano p). If the controls match the signature, the gain is not
     specific to signatures.
2. **Equal tuning.** XGBoost, the MLP and the LSTM now get a small grid with
   early stopping, scored on a purged inner holdout (the last 20% of the
   training rows, with overlapping labels removed). XGBoost: depth × L2 with
   early stopping on the number of trees. MLP: weight decay × two architectures.
   LSTM: weight decay × two hidden sizes. Each is retrained on all training rows
   with the chosen settings. Switch off with `tune_nonlinear=False`.
3. **GARCH.** `GARCH(1,1) | fixed` keeps the original behaviour for comparison.
   `GARCH(1,1) | rolling refit` refits every 30 rows on the trailing 1,000
   returns, using only past returns. `rp.bias_table` reports mean forecast ÷ mean
   realized volatility per fold, which tests the stale-long-run-variance
   hypothesis directly. On a synthetic series whose volatility level drops, the
   fixed version overshoots by 14% and the rolling version is unbiased; that
   shows the mechanism, not that it explains the real result.

**What would still strengthen the conclusions.** Depth 4; the same experiment on
a second asset such as ETH and a longer Binance history; QLIKE or MAE alongside
RMSE; per-fold Diebold–Mariano tests.

**How to read the next run.** If `Ridge | signature` beats both controls
significantly, the case for signature structure specifically is much stronger.
If the lag bank or multi-scale set match it, the honest conclusion is that a
rich multi-scale feature set helps, with no evidence for signatures in
particular. If tuned XGBoost/MLP/LSTM close the gap to Ridge, the earlier
"linear wins" statement should be dropped. If rolling GARCH is near HAR, its
earlier result was a handicap rather than a model weakness.

### 5.5 HMM anomaly flag as a forecasting input (implemented, not yet run on real data)

**Idea.** A Gaussian HMM, fit on as many features as we like, outputs a
"stress or not" flag. The flag then becomes an extra input to the volatility
forecasts, to see whether regime information improves them, in particular in the
high-volatility periods where signature Ridge underperforms HAR.

**Implementation** (`anomaly_flag.py`, wired into `experiment.py`):
- A 3-state diagonal-Gaussian HMM, with the highest-volatility state treated as
  "stress". Features: return, log range, relative volume, signed-flow imbalance
  (Binance only) and trailing volatility.
- Only **filtered** (forward) probabilities are used, never smoothed ones. The
  HMM and its scaler are refit in each fold on observations that ended before the
  test block, and a row only sees state bars completed by its timestamp.
- Four new models, each paired with an otherwise identical model: HAR-RV-L + flag,
  Ridge signature + flag, XGBoost signature + flag (flag and stress probability as
  columns), and Ridge signature with the stress probability as an additional
  signature **path dimension**.
- Diagnostics: how often the flag is on per fold, episode lengths, and whether
  volatility is higher while it is on. The results can be compared with and
  without the flag, split by flagged and unflagged rows.
- Status: verified for wiring on synthetic data only. No results yet.
  Horizons already in `results/main` are skipped on re-run, so use a new
  `output_dir` (or delete the horizon folders) to include the flag models.
- Caveat: rows in a fold's training set are flagged by an HMM fit on data that
  includes them, whereas test rows are flagged out of sample. This is a mild
  mismatch but a real one.

## 6. Cross-cutting issues and limitations

- **Single asset.** Everything is BTC. There is no FX/equity test, although the
  original plan proposed EUR/USD and Dukascopy data.
- **Labeled-event evaluation is weak.** There are only five events, and
  detection is "any crossing in a multi-month window". Tuning on those events
  inflates results. Better scoring would use per-day flag rates, precision/recall
  against regime labels, or an out-of-sample asset.
- **Calibration.** FPRs of 6–22% against a 1% target show the thresholds are not
  yet trustworthy. Longer or less-overlapping calibration sets would help.
- **Compute and environment.** Pure-NumPy signatures and a hand-rolled HMM on
  one machine; depth 4 was never run in the volatility study (it crashed in the
  first version). These limit depth and scale.
- **Volatility study (Thread C).** One asset and three folds; the 5-min and
  1-hour results rest on 77 days; RMSE only, so crisis periods dominate. The
  missing signature-free control, unequal tuning and fixed-parameter GARCH are
  fixed in the code but not yet re-run. See §5.4.
- **Unfinished.** Micro and intraday horizons are not evaluated. The summary
  table has no rows for them. CVaR-OCSVM does not work on real data. P&L
  evaluation (in the task plan) has not been started. The anomaly threads (B and C)
  do not yet share a common evaluation, and no comparison of signature
  anomaly scores against volatility-based scores has been run.

## 7. Where things stand

| Question | Current answer |
|---|---|
| Do signatures separate path shapes? | Yes on synthetic paths; depth 3 is sufficient |
| Do signature novelty scores respond to injected anomalies? | Yes (ESD/conformance AUROC 0.98–1.0 at large spikes) |
| Do they flag real historical events? | 4/5, with lags of 1–21 days and 6% FPR in the best configuration; the 2023 banking crisis is missed |
| Do they beat simple baselines at detection? | Not shown. An HMM is faster and engineered-feature one-class SVM is competitive, both at much higher FPR |
| Do signatures predict abnormal 5-min windows better than summary stats? | No, with a small single-week sample |
| Do signatures improve volatility forecasts? | At 1 and 7 days, yes: Ridge on signatures is 8% and 12.6% better than HAR in RMSE (p ≤ 1e-3), consistent across three eras. At 30 days it is ~13% better but only weakly significant (p ≈ 0.01–0.03). At 5 min only combined with stats, and not beyond stats alone. At 1 hour HAR wins. Whether signature *geometry* is the reason is untested in the reported run; matched controls, equal tuning and a rolling GARCH are now in the code and need a re-run. |
| Does an HMM anomaly flag help volatility forecasts? | Implemented and tested for wiring on synthetic data; not yet run on real data |
| Paper's CVaR-OCSVM on market data? | Does not work as implemented |
