#NOTES
---------------
This document is intended to document all the notes that the team has for review for the mentors as well as to keep track of all information.

## Data Sources

### 1. Dukascopy Bank Data

* **Source:** Dukascopy Bank SA historical market data, accessed through the open-source **TheoryCraft Dukascopy** library.
* **GitHub:** https://github.com/theorycraft-trading/dukascopy
* The library provides a relatively simple CLI-based interface for downloading historical market data.

#### Data Available

| Asset Class              | Approx. Coverage | Examples                       |
| ------------------------ | ---------------: | ------------------------------ |
| Forex Majors             |                7 | EUR/USD, GBP/USD, USD/JPY      |
| Forex Crosses            |             290+ | EUR/GBP, AUD/NZD, GBP/JPY      |
| Metals                   |              50+ | XAU/USD, XAG/USD, XPT/USD      |
| Stocks                   |           1,000+ | AAPL.US/USD, TSLA.US/USD       |
| Indices                  |               21 | USA30.IDX/USD, USATECH.IDX/USD |
| Commodities              |              10+ | BRENT.CMD/USD, COPPER.CMD/USD  |
| Agricultural Commodities |                6 | COCOA.CMD/USD, COFFEE.CMD/USX  |
| Crypto                   |               33 | BTC/USD, ETH/USD               |

#### Advantages

* Supports **tick-level and second-level data**
* Includes **bid and ask prices**, 
* Straightforward to use through the command-line interface.
* Forex majors may provide a useful starting universe because of their liquidity and continuous trading characteristics.

#### Potential Issues

* The TheoryCraft library is still a **work in progress**, so its API and usage requirements may change.
* If the library becomes unreliable or incompatible, the data may need to be downloaded directly from Dukascopy.

---

### 2. Crypto Data — CoinDesk

* CoinDesk provides historical cryptocurrency market data across:

  * Spot markets
  * Futures
  * Options
* Available data includes **OHLC/OHLCV-style market data** across a range of crypto instruments.

#### Advantages

* Crypto markets trade continuously, which may simplify some aspects of high-frequency time-series construction.
* Could provide useful datasets for studying major abnormal market events.

#### Potential Issues

* API access is **rate limited**.
* Downloading large amounts of high-frequency historical data may require additional scripting and data-management work.

# Path Signature Benchmarking Plan

The goal of this stage is to understand whether path signatures provide a useful representation of financial time-series paths before applying them to anomaly detection.

## 1. Synthetic Path Experiments

* Generate simple synthetic paths with different shapes:

  * Gradual upward trend
  * Gradual downward trend
  * Crash followed by recovery
  * Spike followed by reversal
  * Oscillating / choppy path
  * Low-volatility flat path
* Construct some paths to have similar summary statistics, such as:

  * Similar total return
  * Similar realized volatility
* Compute path signatures for each synthetic path.
* Use PCA to visualize the resulting signature vectors.
* Check whether different path types naturally separate in signature space.

**Main question:**
Can signatures distinguish paths that look different even when basic statistics are similar?

---

## 2. Choosing Signature Depth

* Compute signatures using different truncation levels:

  * Level 1
  * Level 2
  * Level 3
* Use the signature vectors to classify the known synthetic path types.
* Keep the classification model simple, such as logistic regression.
* Compare:

  * Classification accuracy
  * Number of signature features
  * Computation time

**Main question:**
How much additional predictive value do higher signature levels provide relative to their computational cost?

---

## 3. Summary Statistics vs. Signatures

* Represent each synthetic path using standard financial features such as:

  * Total return
  * Realized volatility
  * Range
  * Maximum drawdown
  * Autocorrelation
* Compare classification performance using:

  * Standard summary features
  * Raw path observations
  * Path-signature features
* Use the same classifier across representations so the comparison focuses on the feature representation.

**Main question:**
Do signatures capture path information that is not contained in traditional summary statistics?

---

## 4. Initial Test on Real Market Data

* Start with one liquid instrument, for example EUR/USD.
* Divide tick- or second-level data into fixed windows, initially using something simple such as 5-minute windows.
* Represent each window using:

  * Standard financial summary statistics
  * Path-signature features
* Select several interesting market windows, such as:

  * Large price move
  * Sharp reversal
  * High-volatility period
* Find the nearest historical windows under each representation.
* Plot and visually compare the resulting neighbors.

**Main question:**
Do signature-based similarities identify historical windows with more similar path shapes than conventional features?

---

## Expected Outcome

These experiments should help us determine:

* Whether path signatures meaningfully capture path shape.
* Which signature depth provides a reasonable accuracy/computation tradeoff.
* Whether signatures provide information beyond standard financial features.
* Which signature specification should be carried forward into the anomaly-detection stage.

