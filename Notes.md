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


