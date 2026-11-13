"""Horizons, the representation / detector grid, and run settings."""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class HorizonSpec:
    key: str
    label: str
    bar_freq: str  # bars the windows are made of
    window: int  # default window length in bars
    horizon: str  # forecast horizon the anomaly is judged against
    sample_every: str  # spacing between scored rows
    har_windows: tuple  # look-backs of the HAR-RV baseline, as durations
    path_points: int  # signature paths are subsampled to at most this many points


HORIZONS = {
    "1h": HorizonSpec("1h", "1 hour", "5min", 288, "1h", "1h", ("1h", "1D", "7D"), 96),
    "1d": HorizonSpec("1d", "1 day", "1h", 168, "1D", "1D", ("1D", "7D", "30D"), 96),
    "7d": HorizonSpec("7d", "7 day", "1h", 720, "7D", "1D", ("1D", "7D", "30D"), 96),
}

# FX trades only on weekdays and every duration here is counted in hourly BARS, so "5D" is 120
# bars = 5 trading days. Windows: one trading week (1 day) and one trading month (5 days).
FX_HORIZONS = {
    "1d": HorizonSpec("1d", "1 day", "1h", 120, "1D", "1D", ("1D", "5D", "22D"), 96),
    "5d": HorizonSpec("5d", "5 day", "1h", 528, "5D", "1D", ("1D", "5D", "22D"), 96),
}


def horizon_specs(asset: str = "crypto") -> dict:
    if asset not in ("crypto", "fx"):
        raise ValueError(f"asset must be 'crypto' or 'fx', got {asset!r}")
    return FX_HORIZONS if asset == "fx" else HORIZONS

CHANNEL_SETS = {
    "price": ("price",),
    "price+volume": ("price", "activity"),
    "price+volume+range+flow": ("price", "activity", "range", "flow"),
    # FX channels (need AnomalyConfig.fx_channels): time-of-day-adjusted price and range, and the
    # abnormal bid-ask spread (see forecast_data.fx_hourly_channels)
    "price_sa": ("price_sa",),
    "price_sa+spread": ("price_sa", "spread"),
    "price_sa+volume+range_sa+spread": ("price_sa", "activity", "range_sa", "spread"),
}
ORIGINAL_CHANNEL_SETS = ("price", "price+volume", "price+volume+range+flow")

# Labeled crises (from Aayush/configs/horizons.yaml), used only for the timeline case study.
FX_EVENTS = {
    "SNB floor removal": ("2015-01-15", "2015-01-31"),
    "Brexit vote": ("2016-06-23", "2016-07-15"),
    "COVID crash": ("2020-03-01", "2020-04-15"),
    "EUR/USD below parity": ("2022-07-12", "2022-10-31"),
}
EVENTS = {
    "COVID crash": ("2020-02-20", "2020-04-30"),
    "2021 run-up and crash": ("2021-01-01", "2021-05-31"),
    "Terra/Luna": ("2022-05-01", "2022-05-31"),
    "FTX": ("2022-11-01", "2022-11-30"),
    "2023 banking crisis": ("2023-03-01", "2023-03-31"),
}


@dataclass(frozen=True)
class RepConfig:
    """How a window becomes a path (shared by the signature and its controls)."""
    channels: str  # key of CHANNEL_SETS
    normalize: str  # "none" | "trailing_vol" (divide by the vol measured before the window)
    window_mult: float  # window = round(spec.window * window_mult)
    lead_lag: bool = True

    @property
    def name(self) -> str:
        aug = "leadlag" if self.lead_lag else "time"
        return f"{self.channels}|{self.normalize}|w{self.window_mult:g}|{aug}"


@dataclass
class AnomalyConfig:
    horizons: tuple = ("1h", "1d", "7d")  # finished horizons are skipped (resume), not re-run
    asset: str = "crypto"  # "fx": Dukascopy hourly bars of `symbol` (e.g. EURUSD), FX_HORIZONS
    n_folds: int = 3
    min_train_rows: int = 200
    start: str = "2017-08-01"
    end: str | None = None  # None -> today
    symbol: str = "BTCUSDT"
    # FX only: also load the bid-ask spread and the time-of-day-adjusted channels, which the
    # price_sa / range_sa / spread channel sets need
    fx_channels: bool = False
    target_rows: dict = field(default_factory=lambda: {"1h": 20000, "1d": None, "7d": None})

    # Stage A grid (cheap detectors)
    channel_sets: tuple = ORIGINAL_CHANNEL_SETS
    normalizations: tuple = ("none", "trailing_vol")
    window_mults: tuple = (0.5, 1.0, 2.0)
    lead_lag: tuple = (True, False)
    depths: tuple = (2, 3, 4)
    max_sig_features: int = 3000  # skip depths whose signature is longer than this
    stage_a_detectors: tuple = ("mahalanobis", "iforest")
    # Stage B: the expensive detectors on the top configs (+ the best of each control)
    stage_b_top_k: int = 5
    stage_b_detectors: tuple = ("autoencoder", "hmm")
    hmm_states: int = 3
    hmm_components: int = 5
    ae_epochs: int = 60
    ae_patience: int = 8

    surprise_quantile: float = 0.8  # top 20% of log(realized / HAR forecast) = "missed events"
    flag_rate: float = 0.05  # flag = score above the 95th percentile of training scores
    n_bootstrap: int = 500
    seed: int = 0

    data_dir: str = "data"
    output_dir: str = "results/anomaly"
    resume: bool = True

    def rep_configs(self) -> list:
        return [
            RepConfig(c, n, w, ll)
            for c, n, w, ll in itertools.product(self.channel_sets, self.normalizations, self.window_mults, self.lead_lag)
        ]

    def quick(self) -> "AnomalyConfig":
        """Small grid on recent data, to check that everything runs."""
        return replace(
            self, start="2023-01-01", target_rows={"1h": 3000, "1d": None, "7d": None},
            channel_sets=("price", "price+volume+range+flow"), window_mults=(1.0,), lead_lag=(True,),
            depths=(2, 3), stage_b_top_k=2, ae_epochs=15, n_bootstrap=100,
            output_dir=self.output_dir + "_quick",
        )

    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path) -> "AnomalyConfig":
        d = json.loads(Path(path).read_text())
        for k in ("horizons", "channel_sets", "normalizations", "window_mults", "lead_lag", "depths",
                  "stage_a_detectors", "stage_b_detectors"):
            if k in d:  # older config.json files lack newer fields; the defaults apply
                d[k] = tuple(d[k])
        return cls(**d)
