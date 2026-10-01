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
| Yahoo Finance (`yfinance`) BTC-USD | Daily OHLCV from 2014-09 to 2026-09, 4,369 rows | Macro-horizon detection (B); daily/weekly/monthly vol forecasting (C) |
| Binance public archive (`data.binance.vision`) | Trade-by-trade ticks and 1-min klines, no API key | Intraday and micro horizons (B); 5-min/1-hour anomaly and vol work (C) |

Why this mix: Yahoo never exposes trade-level data. It only stores pre-built
bars (daily back to ~2014, hourly ~2 years, 1-minute ~7 days). So it gives the
multi-year history needed to cover all the labeled events. Binance gives real
tick data, but a day of BTCUSDT trades is roughly 900k rows. Tick pulls are
therefore short windows only. In Aayush's pipeline the 2023-10-01 to 2023-10-05
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

### 5.2 Volatility forecasting across five horizons (`volatility_forecasting.ipynb`, `forecast_data.py`, `models_and_metrics.py`)

**Motivation.** Anomaly labels are arbitrary, but future realized volatility is
an objective target. This gives a cleaner test of whether signature features
carry predictive information, against standard econometric baselines.

**Design.**
- Horizons: 5 min and 1 hour (Binance ticks/bars, Aug 2026); 1, 7 and 30 days
  (Yahoo, 2020–2026).
- Target: realized variance over the forecast horizon (Garman-Klass for daily
  bars).
- Paths: price, activity (volume), and signed flow (Binance) or high-low range
  (Yahoo, which has no signed flow). Lead-lag and time augmentation. Depth 3
  (399 signature features in the compact run).
- Baselines: HAR-RV-L (log-variance HAR with a leverage term, per-horizon direct
  regression), GARCH(1,1), XGBoost on stats, MLP on stats.
- Signature models: Ridge, XGBoost, MLP (stats + signature), and an LSTM over a
  sequence of sub-window signatures.
- Evaluation: chronological splits; walk-forward expanding-window backtest for
  the full comparison. Diebold–Mariano tests against the best model.
- The earlier "Random Forest on stats" baseline was replaced by XGBoost on stats,
  so the stats-vs-signature comparison changes only the features, not the model
  family.
- Depth 4 is commented out because it crashes on the machine.

**Experiment 1, baseline comparison (RMSE; lower is better).**

| Horizon | HAR-RV-L | XGB (stats) | Sig. Ridge | XGB (sig.) | Winner |
|---|---|---|---|---|---|
| 5 min | 6.12e-4 | **4.19e-4** | 4.67e-4 | 4.25e-4 | XGB stats, XGB sig. a close second |
| 1 hour | **1.31e-3** | 1.66e-3 | 1.90e-3 | 1.51e-3 | HAR |
| 1 day | 1.06e-2 | 1.24e-2 | **9.56e-3** | 1.01e-2 | Sig. Ridge |
| 7 day | 2.24e-2 | 3.99e-2 | **1.92e-2** | 2.36e-2 | Sig. Ridge |
| 30 day | 4.80e-2 | 7.95e-2 | **3.99e-2** | 4.57e-2 | Sig. Ridge |

**Experiment 5, full walk-forward model comparison (1/7/30 day).**
Signature Ridge ranks first at every horizon. The Diebold–Mariano p-value is the
test against it.

| Horizon | Sig. Ridge RMSE | HAR-RV-L (p) | Sig. LSTM (p) | GARCH(1,1) (p) | XGB sig. (p) |
|---|---|---|---|---|---|
| 1 day | 0.01298 | 0.01384 (3e-5) | 0.01749 (3e-8) | 0.01734 (≈0) | 0.01712 (1e-10) |
| 7 day | 0.02450 | 0.02943 (2e-10) | 0.03338 (9e-6) | 0.03782 (≈0) | 0.03956 (1e-6) |
| 30 day | 0.05203 | 0.05825 (0.025) | 0.06016 (0.041) | 0.09063 (≈0) | 0.06927 (8e-4) |

The two stats-only models (XGBoost and MLP on stats) are the worst at all three horizons. The HAR
advantage for signatures is about 6% at 1 day, 17% at 7 days, and 11% at 30
days. That is significant at the 5% level at every horizon, though only weakly
at 30 days (p≈0.025 against HAR, p≈0.04 against the LSTM).

**Experiment 3, augmentation ablation.** Lead-lag + time was best for XGBoost at
5 min, 1 hour and 7 days, and for Ridge at 5 min, 1 hour and 1 day. Time-only was
best for Ridge at 7 and 30 days (by under 1% at 7 days). No augmentation was best
for XGBoost at 30 days, where lead-lag + time was the worst setting. Overall
effects are small and horizon-dependent.

**Experiment 4, path dimensions.** Adding activity helped slightly at 5 min
(best overall RMSE 4.16e-4, price + activity with XGBoost) but did not
consistently help elsewhere. Price-only was best for 30-day XGBoost (0.0396
vs 0.0457 for the full path). The flow
(imbalance) channel never improved on price + activity. Daily range gave a
small gain for Ridge at 1 and 7 days. For Experiment 5 the notebook picks, per
horizon, the dimension setting with the lowest Experiment 4 RMSE.

**Experiment 2, depth 3 vs 4.** Not run, because depth 4 crashes. Depth 3 rests
on the synthetic test in §3 only.

### 5.3 Revised, more conservative evaluation (`volatility_forecasting_visuals.ipynb`)

**Motivation.** The first notebook picked per-horizon dimension settings and then
reported results on the same data. The second notebook is built to be harder to
fool:
- two **purged** expanding-window folds, dropping any training row whose target
  reaches into the validation block;
- a no-information **Mean** baseline in every comparison;
- data volume capped at 1,200 observations per horizon (laptop memory);
- configuration diagnostics run with Ridge only, so a representation change is
  not confused with a model change;
- the main model table fixed before the diagnostics are examined.

It adds heatmaps of % improvement over the mean forecast, fold-stability plots,
actual-vs-predicted timelines, cumulative squared-error advantage over HAR,
RMSE by realized-vol regime, and a configuration-driver heatmap (signature
level, augmentation, dimensions). **The numeric outputs were not saved to the
repo** (it writes to `outputs_small/`, which is not committed), and the figures
are not stored, so there are no results to report here. The notebook's own
text says two folds on a capped dataset are for "directional analysis" and that
its conclusions should be confirmed on more data before being treated as final.

### 5.4 Takeaways for Thread C

1. At daily-to-monthly horizons, a ridge regression on depth-3 signatures gives
   a consistent, statistically significant RMSE reduction over HAR-RV-L, GARCH
   and the other learners. This is the project's clearest positive result for
   signatures.
2. At 5 min and 1 hour the benefit disappears. Signatures roughly tie XGBoost on
   summary stats at 5 min and lose to HAR at 1 hour.
3. A simple, heavily regularized linear model on signatures beats the flexible
   models (XGBoost, MLP, LSTM) on the same features. With limited observations
   per horizon (about 1,000 in the compact run) against hundreds of signature
   features, heavy regularization is plausible as the reason, though this was not
   tested.
4. Augmentation and dimension choices produce small, inconsistent differences.
   They are not a reliable source of gains.
5. Caveats on the evidence: the Experiment 1–5 numbers use a chronological split
   or a walk-forward backtest on one asset and one period. The path-dimension
   choice was made on the same data it was scored on. The revised notebook
   (§5.3) exists to address this, but its results are not saved.

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
  one machine; depth-4 crashes on another. These limit depth and scale.
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
| Do signatures improve volatility forecasts? | Yes at 1/7/30 days (Ridge, significant vs HAR/GARCH); no at 5 min / 1 hour |
| Paper's CVaR-OCSVM on market data? | Does not work as implemented |
