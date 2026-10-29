"""Wiring check for the anomaly study on synthetic data (no network, tiny grid).

    python -m anomaly.smoke_test        # from the repo root

Says nothing about real markets: it checks no look-ahead, that every detector x representation
runs, the synthetic order-only test, resume, selection/gate files and the report functions.
"""
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

try:
    import iisignature  # noqa: F401
except ImportError:  # pure-NumPy stand-in (handles a batch of paths) where the library won't build
    import types

    def _sig1(path, depth):
        d = path.shape[1]
        total = [np.array([1.0])] + [np.zeros(d**k) for k in range(1, depth + 1)]
        for dx in np.diff(path, axis=0):
            seg, term = [np.array([1.0])], np.array([1.0])
            for k in range(1, depth + 1):
                term = np.kron(term, dx) / k
                seg.append(term)
            total = [sum(np.kron(total[k], seg[n - k]) for k in range(n + 1)) for n in range(depth + 1)]
        return np.concatenate(total[1:])

    def _sig(path, depth):
        path = np.asarray(path, float)
        return np.stack([_sig1(p, depth) for p in path]) if path.ndim == 3 else _sig1(path, depth)

    shim = types.ModuleType("iisignature")
    shim.sig = _sig
    shim.siglength = lambda d, m: sum(d**k for k in range(1, m + 1))
    sys.modules["iisignature"] = shim

import anomaly  # noqa: F401  (puts Mihika/ on the path)
from anomaly import config as C
from anomaly import report as R
from anomaly import run as RUN
from anomaly.detectors import make
from anomaly.evaluate import reorder_injection
from anomaly.features import build_rows, extract_windows, representations

rng = np.random.default_rng(0)


def ok(msg):
    print("PASS", msg)


def synth_bars(n, freq, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    h = np.zeros(n)
    for t in range(1, n):  # GARCH-like log variance
        h[t] = 0.97 * h[t - 1] + 0.25 * rng.normal()
    r = rng.normal(0, 1, n) * 0.002 * np.exp(h / 2)
    close = 30_000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 1, n)) * 0.001 * np.exp(h / 2) * close
    dv = rng.lognormal(10, 0.5, n) * np.exp(h / 4)
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                       "low": np.minimum(open_, close) - spread, "close": close, "volume": dv / close,
                       "dollar_volume": dv, "signed_dollar_flow": dv * rng.uniform(-0.3, 0.3, n)}, index=idx)
    df.index.name = "timestamp"
    return df


def test_no_lookahead():
    spec = C.HORIZONS["1d"]
    bars = synth_bars(24 * 120, "1h")
    rows = build_rows(bars, spec, 336, None)
    rep = C.RepConfig("price+volume+range+flow", "trailing_vol", 1.0, True)
    feats = lambda b: representations(extract_windows(b, rows.pos, 168, rep.normalize), rep, spec, (2, 3), 3000)
    base = feats(bars)
    T = bars.index[len(bars) // 2]
    tampered = bars.copy()
    cols = ["open", "high", "low", "close"]
    tampered.loc[tampered.index > T, cols] *= 3.0
    tampered.loc[tampered.index > T, "dollar_volume"] *= 5.0
    alt = feats(tampered)
    past = np.asarray(rows.times <= T)
    assert past.sum() > 10 and (~past).sum() > 10
    for name in base:
        assert np.allclose(base[name][past], alt[name][past]), f"{name} uses future bars"
    rows_alt = build_rows(tampered, spec, 336, None)
    assert np.allclose(rows.har_X[past], rows_alt.har_X[past]), "HAR inputs use future bars"
    ok(f"no look-ahead: {sorted(base)} unchanged for rows known before the tampered bars")


def test_synthetic_order_only():
    spec = C.HORIZONS["1d"]
    bars = synth_bars(24 * 200, "1h")
    rows = build_rows(bars, spec, 336, None)
    rep = C.RepConfig("price", "trailing_vol", 1.0, True)
    cfg = C.AnomalyConfig()
    win = extract_windows(bars, rows.pos, 168, rep.normalize)
    X = representations(win, rep, spec, (3,), 3000)
    n = len(rows.pos)
    train, test = np.arange(n // 2), np.arange(n // 2, n)
    sub, injected = reorder_injection(win, test, seed=1)
    Xs = representations(sub, rep, spec, (3,), 3000)
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for name in ("signature d3", "stats", "lag bank"):
        model = make("mahalanobis", cfg).fit(X[name][train])
        aucs[name] = roc_auc_score(injected, model.score(Xs[name]))
    assert aucs["signature d3"] > 0.7, aucs
    ok("order-only injected windows (same returns, different shape): AUC " +
       ", ".join(f"{k} {v:.2f}" for k, v in aucs.items()))


def test_pipeline(tmp):
    bars = {"5min": synth_bars(12 * 24 * 70, "5min"), "1h": synth_bars(24 * 900, "1h")}
    RUN.fetch_binance_klines_hourly = lambda *a, freq="1h", **k: bars[freq]
    cfg = C.AnomalyConfig(
        target_rows={"1h": 900, "1d": None}, min_train_rows=100,
        channel_sets=("price", "price+volume+range+flow"), window_mults=(1.0,), lead_lag=(True,),
        depths=(2, 3), stage_b_top_k=1, ae_epochs=3, n_bootstrap=30,
        data_dir=str(tmp / "data"), output_dir=str(tmp / "results"),
    )
    root = RUN.run_all(cfg)
    res = R.load_results(root)
    assert set(res.meta) == {"1h", "1d"}
    kinds = set(res.metrics.Kind)
    assert kinds == {"signature", "lag bank", "stats", "raw", "baseline"}, kinds
    assert {"mahalanobis", "iforest", "autoencoder", "none"} <= set(res.metrics.Detector)
    dup = res.metrics.duplicated(["Horizon", "Config", "Detector", "Fold"])
    assert not dup.any(), res.metrics[dup].head()
    assert res.metrics.AUC_surprise_strat.between(0, 1).all()
    for hk, sel in res.selection.items():
        assert sel["best_signature"]["Kind"] == "signature" and sel["best_signature_free"]["Kind"] != "signature"
        assert "passed" in sel["gate"] and np.isfinite(sel["gate"]["bootstrap_last_fold"]["diff"])
        assert all("AUC_injected" in v for v in sel["synthetic_order_only"].values())
    assert (Path(root) / "best_configs.json").exists()
    ok("pipeline: both horizons, every kind x detector, no duplicate fits, selection + gate + synthetic")

    before = (Path(root) / "1d" / "meta.json").stat().st_mtime
    RUN.run_all(cfg)
    assert (Path(root) / "1d" / "meta.json").stat().st_mtime == before
    import shutil

    (Path(root) / "1h" / "meta.json").unlink()  # interrupted horizon: parts are reused
    RUN.run_all(cfg)
    res2 = R.load_results(root)
    assert not res2.metrics.duplicated(["Horizon", "Config", "Detector", "Fold"]).any()
    ok("resume: finished horizons skipped, an interrupted horizon reuses its saved parts")

    for t in (R.leaderboard(res), R.kind_table(res), R.gate_table(res), R.synthetic_table(res)):
        assert len(t) > 0
    import matplotlib.pyplot as plt

    for fig in (R.plot_kind_heatmap(res), R.plot_timeline(res, "1d"), R.plot_timeline(res, "1h")):
        fig.savefig(tmp / "f.png", dpi=40)
        plt.close(fig)
    ok("report tables and figures")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_no_lookahead()
        test_synthetic_order_only()
        test_pipeline(tmp)
    print("\nALL ANOMALY SMOKE CHECKS PASSED")
