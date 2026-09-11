# Path-Signature Novelty Detection on BTC Market Data

A group-project pipeline applying **"Novelty detection on path space"**
(Gasteratos, Jacquier, Lemercier, Lyons, Salvi — arXiv:2512.03243, source
included in `arXiv-2512.03243v1/`) to real Bitcoin market data: detecting
the 2020 COVID crash, the 2021 meme-stock-era retail mania, the 2022
crypto crashes (Terra/Luna, FTX), the 2023 banking crisis, and bull/bear +
volatility-regime transitions, using path-signature features and one-class
novelty detection.

The paper itself only validates its methods on synthetic
Brownian-motion-with-a-spike data and real molecular-biology (nanopore
RNA) signals — applying it to financial markets is this project's own
contribution.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the notebooks in order (`jupyter notebook` or
`jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`):

1. `01_data_ingestion.ipynb` — pull Yahoo daily OHLCV + Binance ticks/klines.
2. `02_path_construction_and_signatures.ipynb` — build signature feature matrices for all three horizons.
3. `03_baseline_models.ipynb` — sklearn baselines + classical volatility z-score.
4. `04_paper_faithful_models.ipynb` — the paper's own test statistics.
5. `05_synthetic_and_event_evaluation.ipynb` — sanity check + historical event/regime evaluation.

## Data sources

Yahoo Finance never exposes trade-level data at any lookback — it only
stores pre-built bars (daily back to ~2014, hourly for ~2 years, 1-minute
for ~7 days). So:

- **Yahoo Finance** (`yfinance`) supplies daily BTC-USD OHLCV, the
  multi-year backbone for the macro horizon and the event/regime study.
- **Binance's free public archive** (`data.binance.vision`, no API key)
  supplies real trade-by-trade tick data and 1-minute klines for the
  micro/intraday horizons. Tick data is pulled only for a short reference
  window (`tick_reference_period` in `configs/horizons.yaml`), not the
  full history — a single day of BTCUSDT trades is ~900k rows.

## Three horizons (`configs/horizons.yaml`)

| Horizon   | Source                  | Window          | Targets |
|-----------|--------------------------|-----------------|---------|
| micro     | Binance ticks             | 1000 trades     | flash-crash-style microstructure anomalies |
| intraday  | Binance 1m klines → 1h    | 48 hours        | intraday regime breaks |
| macro     | Yahoo daily               | 30 trading days | bull/bear transitions, multi-week structural breaks |

## Pipeline

Raw data → rolling windows (`src/paths/windows.py`) → multichannel path
(`src/paths/transforms.py`: time-augmentation, cumulative log-return,
basepoint, optional invisibility-reset/lead-lag) → per-channel
increment-scale normalization fit on a calm reference period only → **path
signature** (`src/features/signatures.py`) → one-class novelty model.

### Signature computation: pure NumPy, not iisignature

This machine's C++ toolchain is missing standard library headers (a broken
Command Line Tools install — `xcode-select --install` would likely fix
it), so `iisignature`/`esig`/`pySigLib` cannot compile here.
`src/features/signatures.py` instead computes truncated signatures
directly via Chen's identity (the same algorithm those libraries use
internally), verified against hand-computed values and against
concatenation-invariance. Entirely adequate at the depths (2-4) and
channel counts (2-3) this project uses; swap in a compiled backend later
if useful once the toolchain is fixed.

### Models

**Layer 1 — standard baselines** (`src/models/baselines.py`): sklearn
`OneClassSVM`, `IsolationForest`, `LocalOutlierFactor`, `EllipticEnvelope`
on signature features, plus a classical realized-volatility z-score
baseline that skips signatures entirely (the pointwise-statistics foil).

**Layer 2 — paper-faithful methods**, built against formulas verified
directly from the paper's LaTeX source (`arXiv-2512.03243v1/main.tex`):

- `src/models/expected_signature.py` — distance-to-expected-signature
  statistic, `f(x) = ||S_N(x) - E_mu[S_N(X)]||`. Isotropic by
  construction, so it grows with *any* deviation direction — the most
  reliable statistic in this project's own testing.
- `src/models/conformance.py` — conformance score (Mahalanobis distance
  under the reference covariance; the Gaussian-reference special case of
  the paper's "variance norm").
- `src/models/shuffle_algebra.py` + `src/models/cvar_ocsvm.py` — the
  Theorem 2.6 CVaR-OCSVM, which uses the shuffle product to turn a
  regularized-CVaR one-class-SVM objective into a deterministic function
  of the expected signature. Verified against the defining shuffle
  identity `<w1,S(x)><w2,S(x)> = <w1⊔⊔w2,S(x)>` and against small
  synthetic tests. **Known limitation on real data**: the optimizer tends
  to push `||w||` toward its bound rather than settle at a clean interior
  optimum, most likely because a degree-2 polynomial poorly approximates
  the CVaR hinge over the wide score range real, fat-tailed BTC returns
  produce (see the docstring in `cvar_ocsvm.py` for what to try next).
- `src/models/tail_bounds.py` — Weibull tail-fit + empirical p-values,
  Benjamini-Hochberg FDR correction across all windows tested (mirrors the
  paper's own Section 4.1 methodology). Note: with only ~35-40
  non-degenerate calibration windows from a single ~2.5-month reference
  period, statistical power for BH-controlled discovery is genuinely
  limited — a longer or less-overlapping calibration set would help.

## Results so far (macro horizon, distance-to-expected-signature)

- **Synthetic sanity check passes**: injected-spike AUROC rises cleanly
  from ~0.49 (no spike) to ~0.92 (spike = 8x the channel's natural std),
  confirming the pipeline behaves correctly before trusting real events.
- **3 of 5 labeled events detected** at a Weibull-fit alpha=0.01
  threshold: COVID crash (27-day lag), the 2021 BTC retail-mania
  runup/crash (1-day lag), the FTX collapse (12-day lag). The Terra/Luna
  crash and the 2023 banking crisis were missed at this threshold/horizon.
- **False positive rate outside labeled events: ~4.7%** — higher than the
  nominal alpha=0.01 target, consistent with the calibration-sample-size
  caveat above.
- The score also spikes sharply around the 2017-2018 crypto bubble
  crash — not one of the four labeled events, but a well-known real
  volatile period, a good sign of genuine signal beyond the specific
  labels chosen.

See `reports/event_detection_macro.png` for the full score timeline.

## What's demonstrated vs. what's left as follow-on work

Notebooks 03-05 demonstrate the full pattern (fit → score → validate) on
the **macro horizon** with the **distance-to-expected-signature**
statistic. Repeating that same pattern for the **intraday** and **micro**
horizons, and for the other models (sklearn baselines, conformance score,
CVaR-OCSVM), to fill out `src/evaluation/metrics.py`'s
`summarize_horizon_model` comparison table across all horizon x model
combinations, is the natural next step — every function needed for that
(`raw_paths_for_horizon`, `fit_reference_scale`, `build_feature_frame`, all
five models, `auroc_vs_spike_magnitude`, `evaluate_all_events`,
`bull_bear_labels`/`volatility_regime_labels`) already exists and is
tested; it's a matter of looping the notebook 05 pattern over the other
horizon/model combinations.

## Project layout

```
configs/horizons.yaml          # horizons, reference/tick-reference periods, labeled events
src/ingestion/                 # yahoo.py, binance.py
src/paths/                     # windows.py, transforms.py
src/features/                  # signatures.py (pure NumPy), pipeline.py
src/models/                    # baselines, expected_signature, conformance,
                                # shuffle_algebra + cvar_ocsvm, tail_bounds
src/evaluation/                # synthetic.py, events.py, regimes.py, metrics.py
notebooks/                     # 01-05, run in order
data/raw/, data/processed/     # gitignored, populated by running the notebooks
reports/                       # plots + summary_table.csv from notebook runs
```
