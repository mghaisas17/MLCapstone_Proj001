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
chosen on the same holdout the comparison was scored on. Both were replaced by one
runner. A first purged run left three weaknesses: no signature-free control,
fixed (untuned) hyperparameters for XGBoost, MLP and LSTM, and a GARCH fit once
and held fixed. A second run, reported here, fixes those and adds an HMM anomaly
flag. Everything below comes from that run in the committed notebook. Its numbers
supersede the earlier ones.

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
  test time is purged. Every model sees identical folds.
- **Signature specification, fixed in advance.** Depth 3, lead-lag plus time
  augmentation, full path: price, activity, and signed flow (Binance) or high-low
  range (Yahoo). That is 399 signature features. The ablations below did not
  choose it.
- **Models.** Mean, HAR-RV-L, GARCH(1,1) in two versions, and Ridge, XGBoost and
  MLP on summary stats, on signatures, or on both, plus a signature LSTM.
- **Equal tuning.** Ridge picks its alpha by an inner purged walk-forward CV.
  XGBoost, the MLP and the LSTM now get a small grid with early stopping,
  scored on a purged inner holdout (the last 20% of the training rows).
- **Signature-free controls**, fed to the same Ridge:
  - a *lag bank* with exactly as many columns as the signature (399 for the
    daily specs): raw, squared, cross-channel and lagged increments of the same
    channels at the same resolution;
  - a hand-built *multi-scale stats* set: realized variance, absolute return,
    range, volume, signed return, downside semivariance, largest move and flow
    at five look-back scales.
- **GARCH.** `fixed` is fit once on the training period. `rolling refit` refits
  every 30 rows on the trailing 1,000 returns, using only past data.
- **HMM anomaly flag.** See §5.5.
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

**Model ranking** (fold-mean RMSE; lower is better; `%HAR` is the RMSE
improvement over HAR-RV-L). `p` is the pooled Diebold–Mariano p-value against
HAR-RV-L. HAR-RV-L, the mean forecast and all Ridge models give the same numbers
as in the earlier run, as they should, since nothing about them changed.

| Horizon | Top models | RMSE (%HAR, p) | HAR-RV-L | Mean |
|---|---|---|---|---|
| 5 min | Ridge multiscale / Ridge stats+sig / Ridge stats / Ridge sig | 4.08e-4 (+12.8%, 0.004) / 4.11e-4 (+12.3%, 0.004) / 4.25e-4 (+9.2%, 0.025) / 4.43e-4 (+5.5%, 0.20) | 4.68e-4 | 6.13e-4 |
| 1 hour | Ridge multiscale ≈ **HAR-RV-L** | 1.478e-3 (+0.9%, 0.79) | 1.491e-3 | 2.06e-3 |
| 1 day | Ridge lag bank / LSTM / Ridge stats+sig / Ridge sig | 1.459e-2 (+11.0%, 1e-9) / 1.490e-2 (+9.1%, 3e-5) / 1.500e-2 (+8.5%, 5e-4) / 1.502e-2 (+8.4%, 8e-4) | 1.639e-2 | 1.862e-2 |
| 7 day | Ridge sig / LSTM / Ridge stats+sig / Ridge lag bank | 3.063e-2 (+12.6%, 2e-6) / 3.105e-2 (+11.4%, 2e-8) / 3.106e-2 (+11.4%, 5e-4) / 3.203e-2 (+8.6%, 0.021) | 3.506e-2 | 3.949e-2 |
| 30 day | Ridge stats+sig / Ridge sig / Ridge lag bank | 5.65e-2 (+13.0%, 0.011) / 5.67e-2 (+12.7%, 0.025) / 5.72e-2 (+12.0%, 0.015) | 6.50e-2 | 6.88e-2 |

- **1 day.** The top nine models (lag bank, LSTM, Ridge stats+sig, Ridge sig, MLP,
  XGBoost, and the flag variants) are statistically tied: none differs from the
  best at p < 0.07. Ridge on stats alone and HAR are indistinguishable from each
  other (p = 0.66).
- **7 days.** Ridge sig, the LSTM and Ridge stats+sig are tied (p ≥ 0.39 against the best).
- **30 days.** Only the Ridge models beat HAR significantly. The tuned LSTM, MLP and XGBoost are
  not distinguishable from HAR (p = 0.87, 0.63, 0.98). HAR itself is not
  distinguishable from the mean forecast (p = 0.40).
- **1 hour.** HAR-RV-L and Ridge on multi-scale stats tie for first; every other model except HAR + flag is 16–38% worse (p < 1e-6).
- **Fold stability.** Ridge on signatures beats the mean forecast in every fold
  at every horizon. HAR is worse than the mean in fold 3 at 30 days.

**Do signatures matter? Signatures vs signature-free controls** (same Ridge, same
folds). `Pct_sig_better` > 0 means signatures have the lower RMSE.

| Horizon | vs lag bank (matched) | vs multi-scale stats | vs summary stats |
|---|---|---|---|
| 5 min | **+14.4% (p = 0.025)** | −8.3% (p = 0.10) | −4.7% (p = 0.34) |
| 1 hour | −1.4% (p = 0.55) | **−22.3% (p ≈ 0)** | −6.2% (p = 0.004) |
| 1 day | −2.8% (p = 0.22) | **+8.8% (p ≈ 0)** | **+7.2% (p ≈ 0)** |
| 7 day | +4.9% (p = 0.11) | **+16.7% (p = 4e-8)** | **+10.8% (p = 2e-5)** |
| 30 day | +0.5% (p = 0.72) | **+21.8% (p = 5e-4)** | **+10.5% (p = 0.018)** |

**Equal tuning.** Fold-mean `%HAR`, before (fixed hyperparameters) → after tuning:

| Model | 5 min | 1 hour | 1 day | 7 day | 30 day |
|---|---|---|---|---|---|
| LSTM | −53.5 → +1.4 | −36.5 → −19.6 | +0.8 → **+9.1** | +5.0 → **+11.4** | +2.7 → +3.7 |
| MLP stats+sig | −27.5 → −2.0 | −37.6 → −36.0 | −9.2 → **+7.7** | +5.4 → +7.6 | −15.9 → +1.6 |
| XGBoost sig | −12.7 → −14.2 | −21.6 → −20.9 | +7.3 → +7.2 | +4.7 → +6.0 | −1.6 → +2.8 |

**GARCH and forecast bias.** Mean forecast ÷ mean realized volatility, pooled
(1.0 is unbiased):

| Horizon | GARCH fixed | GARCH rolling | HAR-RV-L | Ridge sig |
|---|---|---|---|---|
| 1 day | 1.35 | 1.33 | 1.24 | 0.93 |
| 7 day | 1.30 | 1.25 | 1.19 | 0.98 |
| 30 day | 1.41 | 1.27 | 1.19 | 0.97 |

Rolling refit cuts GARCH's 30-day RMSE from 9.3e-2 to 7.2e-2, but it is still
worse than the mean forecast (−5%) and HAR (−12%). Its overshoot falls in fold 3
(1.44 → 1.10 at 30 days) but not in fold 1 (1.47 either way).

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
out-of-sample target. Relative to HAR, the signature, lag-bank and tuned
nonlinear models do much better in the low and middle terciles at 1, 7 and 30
days (roughly +30–45%), but are worse than HAR in the highest-volatility tercile
(roughly −5% to −20%). The flag does not change this (§5.5).

### 5.4 What the evidence supports

**Supported.**
1. **At 1 and 7 days, Ridge on the full multi-channel path beats HAR-RV-L.**
   The gain is 8–11% at 1 day and 9–13% at 7 days, with p-values from 1e-3 to
   1e-9, in all three folds covering different eras. The 1-day test has about
   4,260 targets and the 7-day test about 600. It survives purging.
2. **Hand-built multi-scale statistics do not reproduce that gain** at daily or
   longer horizons: Ridge on them is no better than HAR at 1 day and worse than
   HAR at 7 and 30 days (p = 0.037 and 0.008). Aggregated volatility, range and volume
   features are not what carries the gain.
3. **At 1 hour and 5 min, signatures are not the best choice.** At 1 hour, HAR
   or multi-scale stats win and signatures are 22% worse than the latter. At 5
   min, Ridge on multi-scale stats or on stats+signatures is nominally best, but
   signatures alone are not significantly better than HAR (p = 0.20) and
   not better than stats (p = 0.34).
4. **Using only the price path throws away most of the benefit** (price-only is
   the worst in the dimension ablation at every horizon).

**What changed from the first purged run.**
- **Signature geometry is not shown to matter.** The size-matched lag bank ties
  Ridge on signatures at 1 day (nominally better, p = 0.22), 1 hour, 7 days
  (signatures +4.9%, p = 0.11) and 30 days (p = 0.72). Signatures beat the lag
  bank significantly only at 5 min (+14.4%, p = 0.025), where they do not beat
  HAR. The gain over HAR comes from giving a regularized linear model the
  within-window path of increments across price, activity and range, and a
  signature is not required for that. A plausible reading, which we did not test,
  is that HAR fixes its lag weights as averages over 1, 7 and 30 days, whereas
  Ridge on lags learns free weights.
- **"Linear beats nonlinear" no longer holds.** With tuning, the LSTM and MLP
  caught up at 1 and 7 days and tie Ridge there. XGBoost barely improved and
  stays weaker. Ridge still wins at 30 days, plausibly because heavy
  regularization suits the smallest sample, but that is not tested.
- **GARCH is only partly explained by a stale long-run variance.** Rolling refit
  helped, but GARCH still overshoots realized volatility by 25–33% and still
  loses to HAR and the mean at 30 days.

**Forecast calibration may explain part of the gap to HAR and GARCH.** HAR
overshoots realized volatility by 19–24% (and by 28% at 5 min), and GARCH by 25–41%, while Ridge
sits near 0.93–0.98. Both HAR and GARCH forecast variance and are then
transformed to volatility, which biases the result upward (Jensen's
inequality), whereas Ridge is trained directly on the volatility target and
RMSE. So part of "Ridge beats HAR/GARCH" may be a calibration difference rather
than better information. A bias-calibrated HAR and GARCH, or HAR fit directly on
the volatility level, have not been tried and would test this.

**Suggestive, but not established.**
- **30 days.** There are only about 140 independent targets, the p-values are
  0.011–0.025, and HAR is not distinguishable from the mean. With about 50
  model-and-horizon comparisons against HAR, a single p of 0.02 is not strong
  evidence.
- **5 min.** The 5-min data cover only 77 days of one market regime.

**Remaining limitations.**
- One asset, one history, three folds. The folds are consecutive blocks of one
  time series, not independent draws.
- RMSE is dominated by crisis periods, so pooled results lean on fold 1
  (2018–2020). The loss differential is not stationary across folds, which
  weakens the Diebold–Mariano p-values. QLIKE or MAE should be added.
- Depth 4 was never run. More levels helped monotonically up to the deepest
  tested level (3), so we do not know where the gain saturates.
- The calm-vs-stressed picture is conditioned on the realized outcome, so
  smoothing models look good in calm periods and poor in spikes by construction.
  Even so, none of the signature-type models beat HAR when volatility is highest,
  which is where forecasts matter most for risk.

**Next experiments, in order of value.**
(a) Bias-calibrated HAR and GARCH, and HAR fit on the volatility level, to
see how much of the gap to Ridge is calibration.
(b) The same experiment on a second asset such as ETH, and a longer Binance
history.
(c) QLIKE or MAE alongside RMSE, and per-fold Diebold–Mariano tests.
(d) A richer non-signature distributed-lag control (for example, Ridge on lags
of several variables with more lags than 29), to test the free-lag-weights reading.

### 5.5 HMM anomaly flag as a forecasting input

**Idea.** A Gaussian HMM, fit on as many features as we like, outputs a
"stress or not" flag. The flag then becomes an extra input to the volatility
forecasts, to see whether regime information improves them, in particular in the
high-volatility periods where the signature models underperform HAR.

**Implementation** (`anomaly_flag.py`, wired into `experiment.py`):
- A 3-state diagonal-Gaussian HMM, with the highest-volatility state treated as
  "stress". Features: return, log range, relative volume, signed-flow imbalance
  (Binance only) and trailing volatility.
- Only **filtered** (forward) probabilities are used, never smoothed ones. The
  HMM and its scaler are refit in each fold on observations that ended before the
  test block, and a row only sees state bars completed by its timestamp.
- Four models, each paired with an otherwise identical one: HAR-RV-L + flag,
  Ridge signature + flag, XGBoost signature + flag (flag and stress probability
  as columns), and Ridge signature with the stress probability as an extra
  signature **path dimension** (`stress dim`).
- Caveat: rows in a fold's training set are flagged by an HMM fit on data that
  includes them, whereas test rows are flagged out of sample.

**Does the flag identify a stress regime? Yes, roughly.** Share of out-of-sample
rows flagged, and mean realized volatility while flagged ÷ while not flagged:

| Horizon | Share flagged, folds 1 / 2 / 3 | Volatility ratio, folds 1 / 2 / 3 |
|---|---|---|
| 5 min | 13% / 33% / 18% | 2.1 / 3.2 / 2.2 |
| 1 hour | 17% / 34% / 22% | 1.5 / 2.5 / 1.5 |
| 1 day | 32% / 17% / 7% | 2.0 / 2.0 / 1.6 |
| 7 day | 32% / 17% / 6% | 1.7 / 1.8 / 1.4 |
| 30 day | 33% / 17% / 7% | 1.4 / 1.4 / 1.2 |

On the daily timeline the flag clusters around late 2018, mid-2019, March 2020
(COVID) and early to mid 2021, and is thin after 2022 (the 2022 crashes and the
2023 banking crisis are only sparsely flagged). The flag is on about a fifth of
the time overall, so it behaves like a "volatile regime" indicator, not a rare
anomaly. In daily fold 3 it fires only 7% of the time, a calmer era than the
training data.

**Does it improve the forecasts? Not at 1 day or longer.** Change in pooled RMSE
from adding the flag (positive = the flag helps; DM p-value of the flag version
against the original):

| Pair | 5 min | 1 hour | 1 day | 7 day | 30 day |
|---|---|---|---|---|---|
| HAR-RV-L + flag | −0.1% (0.98) | −0.1% (0.73) | −0.1% (0.69) | −0.2% (0.12) | −0.2% (0.13) |
| Ridge sig + flag | +1.2% (0.80) | **+1.3% (2e-6)** | 0.0% (0.83) | +0.1% (0.39) | −0.1% (0.016) |
| Ridge sig + stress dim | +1.9% (0.67) | **+2.7% (0.009)** | −0.9% (0.16) | −4.9% (0.10) | **−3.4% (0.002)** |
| XGBoost sig + flag | 0.0% (0.99) | **+3.6% (5e-12)** | +1.2% (0.051) | −0.6% (0.28) | −0.4% (0.18) |

- At 1 day and longer, the flag has no effect on HAR or Ridge. Adding it as a
  path dimension makes things worse at 7 and 30 days, since it grows the
  signature from 399 to 819 features with few independent targets.
- The only reliable gains are at 1 hour, for models that were already losing:
  the best flag model there is still about 16% worse than plain HAR.
- Splitting by flagged and unflagged rows does not show a concentrated gain in
  stressed periods at 1 day or longer, and the signature models are still worse than
  HAR in the highest-volatility tercile.
- The likely reason is that the HMM sees return, range and trailing volatility,
  which HAR and Ridge already contain, so the flag mostly restates what they know.

**Conclusion for the flag.** It is a reasonable descriptive regime indicator,
and it gives no forecasting benefit at daily or longer horizons. Ways to make it
carry new information would be to give the HMM inputs the forecasters do not
use, or to use it for gating or risk decisions, not as an extra regressor.

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
  1-hour results rest on 77 days; RMSE only, so crisis periods dominate; HAR
  and GARCH may be penalised by calibration (they overshoot realized volatility
  by 19–41%); depth 4 was never run. See §5.4.
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
| Do signatures improve volatility forecasts? | Not specifically. A regularized linear model on the full multi-channel path beats HAR by 8–11% at 1 day and 9–13% at 7 days (p ≤ 1e-3, consistent across three eras), and by ~13% at 30 days (p ≈ 0.01–0.03). But a size-matched lag bank matches the signatures at every horizon but 5 min, so signature geometry is not shown to matter. At 1 hour HAR or multi-scale stats win. |
| Is Ridge better than nonlinear models? | No longer. With equal tuning, the LSTM and MLP tie Ridge at 1 and 7 days; Ridge still wins at 30 days. |
| Is GARCH a fair baseline? | Rolling refit helps but GARCH (and HAR) still overshoot realized volatility by 19–41%. Part of the gap to Ridge may be calibration; a bias-corrected HAR/GARCH has not been tried. |
| Does an HMM anomaly flag help volatility forecasts? | It identifies a stress regime (volatility 1.2–3.2× higher while flagged), but adds nothing at 1 day or longer; small gains at 1 hour only for models that still lose to HAR. As a path dimension it hurts at 7 and 30 days. |
| Paper's CVaR-OCSVM on market data? | Does not work as implemented |
