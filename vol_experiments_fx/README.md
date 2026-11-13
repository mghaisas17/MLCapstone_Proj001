# vol_experiments_fx

EUR/USD volatility forecasting and anomaly detection with path signatures, using channels built
for FX. These are cleaned, FX-only versions of `Mihika/volatility_experiments.ipynb` and
`anomaly/anomaly_experiments.ipynb`; the originals are unchanged.

| File | What it is |
|---|---|
| `anomaly_fx.ipynb` | Anomaly study: does a signature anomaly score carry information beyond the HAR forecast? Run first. |
| `vol_fx.ipynb` | Volatility and volatility-direction forecasts, channel ablations, and the anomaly score in the forecasts |
| `fx_lab.py` | Shared settings (data, end date, channels, model lists), the direction target, and the tables |
| `smoke_test.py` | Checks everything on synthetic data in about a minute: `python vol_experiments_fx/smoke_test.py` |

**New FX channels** (`forecast_data.fx_hourly_channels`, past data only): the time-of-day-adjusted
log price (`price_sa`) and range (`range_sa`), each hour scaled by how volatile that hour of the week
usually is in New York time, and the abnormal bid-ask spread (`spread`, log spread minus its usual
level for that hour).

**Data**: Dukascopy BID and ASK hourly candles from 2012 to the end of the last finished month,
cached in `Mihika/data/dukascopy/EURUSD_1h_ba_*.parquet` (separate from the mid-only cache the other
notebooks use). The first run downloads about 360 files under Dukascopy's rate limit, which can take
an hour.

**Outputs** go to `results/` and the feature cache to `cache/` (both git-ignored). Every run resumes:
finished horizons are skipped, so re-running a notebook only computes what is missing. Set
`QUICK = True` in a notebook to check everything on 2021+ data first (results in `*_quick`).

**More pairs**: section 9 of `vol_fx.ipynb` runs the same experiment on GBP/USD, USD/JPY and AUD/USD
(results in `results/pairs/<pair>/`; EUR/USD reuses the main run). The first run downloads about 360
files per pair. Section 10 runs the **pairs together**: each pair's path also carries the other pairs'
prices (and, in a second variant, a dollar factor), to see whether the signature's cross terms find
information one pair alone does not have (results in `results/cross/<pair>/`).

## Notes for the write-up

Conclusions from the EUR/USD run (Oct 2026) to carry into the report. Each holds on both BTC and EUR/USD
unless noted. Re-check them against the other pairs once section 9 has run.

- **Drop neural networks on signatures.** The MLP on the hourly signature is far worse than Ridge on the
  same features, on both assets (EUR/USD: −59% at 1 day and −42% at 5 days vs HAR; Ridge −1.2% / −4.6%).
- **The time-of-day adjustment is not a consistent improvement.** On EUR/USD the original channels (price,
  activity, range) beat the adjusted ones (+1.7% vs HAR at 1 day, against −0.0% adjusted and −1.2% adjusted
  + spread), and on USD/JPY too; on AUD/USD the adjusted channels did better at 5 days (−4.3% vs −11.2%).
  No channel set wins on every pair, so say "no consistent gain", not "keep the original channels".
- **Don't put signature features in a direction classifier.** Adding them to the HAR logistic model lowers the
  AUC (0.793 → 0.714 at 1 day, 0.745 → 0.680 at 5 days): about 800 features overfit. Use the volatility
  forecast as the direction score instead (signature forecast 0.788 / 0.751, HAR forecast 0.793 / 0.754).
  Holds on all four pairs: the signature classifier is lower on every pair and horizon (0.60–0.75 against
  0.72–0.80 for HAR alone).

Across the four pairs (section 9):

- **The signature never beats HAR**, and it is significantly worse than calibrated HAR on every pair at 1 day.
- **It beats the matched lag bank** on EUR/USD, USD/JPY and AUD/USD (+6% to +12%), as on BTC: the
  signature is the better summary of the path wherever the run is stable.
- **HAR on hourly RV ties HAR on every pair** (−1.2% to +0.2%): the hourly path adds little at 1 and 5 days
  across FX, not only on EUR/USD.
- **GBP/USD is an outlier** (signature −36% / −51% vs HAR, the lag bank also bad, HAR-type models fine),
  most likely a few extreme days such as the 2016 flash crash. Check `L.worst_errors` in section 9
  before quoting GBP/USD; section 10 clips the Ridge features to guard against it.

The sponsor report is `fx_signature_report.html`.
