"""Wiring check on synthetic data (no network, tiny models).

Verifies that the loader, purged CV, model registry, result files and every
report function run end to end. It does NOT say anything about forecast quality.

    python smoke_test.py
"""
import io
import itertools
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

try:
    import iisignature  # noqa: F401
except ImportError:  # pure-NumPy stand-in so the pipeline can run where it will not compile
    import types

    def _sig(path, depth):
        path = np.asarray(path, float)
        d = path.shape[1]
        total = [np.array([1.0])] + [np.zeros(d**k) for k in range(1, depth + 1)]
        for dx in np.diff(path, axis=0):  # Chen's identity: total <- total (x) exp(dx)
            seg, term = [np.array([1.0])], np.array([1.0])
            for k in range(1, depth + 1):
                term = np.kron(term, dx) / k  # dx^(x)k / k!
                seg.append(term)
            total = [sum(np.kron(total[k], seg[n - k]) for k in range(n + 1)) for n in range(depth + 1)]
        return np.concatenate(total[1:])

    shim = types.ModuleType("iisignature")
    shim.sig = _sig
    shim.siglength = lambda d, m: sum(d**k for k in range(1, m + 1))
    sys.modules["iisignature"] = shim

import experiment as ex
import forecast_data as fd
import reporting as rp

rng = np.random.default_rng(0)


def ok(msg):
    print("PASS", msg)


# ---------------------------------------------------------------- loader
def fake_zip(day, n=20_000, unit="us"):
    t0 = pd.Timestamp(day, tz="UTC")
    secs = np.sort(rng.uniform(0, 86_400, n))
    ts = (t0.value // 1000 + (secs * 1e6).astype(np.int64)) if unit == "us" else (t0.value // 10**6 + (secs * 1e3).astype(np.int64))
    price = 30_000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    qty = rng.uniform(0.001, 2, n)
    df = pd.DataFrame({
        "id": np.arange(n), "price": price, "qty": qty, "quote": price * qty,
        "ts": ts, "maker": rng.random(n) < 0.5, "match": True,
    })
    df["maker"] = df["maker"].map({True: "True", False: "False"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("trades.csv", df.to_csv(index=False, header=False))
    return buf.getvalue(), df


def test_loader(tmp):
    content, raw = fake_zip("2025-03-01")

    class Resp:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_content(self, chunk_size):
            for i in range(0, len(content), 4096):
                yield content[i:i + 4096]

    calls = []
    def fake_get(url, **kw):
        calls.append(url)
        return Resp()

    fd.requests.get = fake_get
    got = fd.fetch_binance_bars("2025-03-01", "2025-03-01", freqs=("5s", "1min"),
                                data_dir=tmp / "bars", chunksize=1500)  # many chunk boundaries
    raw.columns = fd.BINANCE_COLUMNS
    raw["is_buyer_maker"] = raw["is_buyer_maker"] == "True"  # what read_csv yields for the real files
    expected = fd.binance_to_bars(fd.prepare_binance_trades(raw), "5s")
    g = got["5s"].loc[expected.index.intersection(got["5s"].index)]
    e = expected.loc[g.index]
    for col in e.columns:
        np.testing.assert_allclose(g[col].to_numpy(), e[col].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=col)
    assert got["1min"].index.freq is not None or len(got["1min"]) > 1000
    n_calls = len(calls)
    fd.fetch_binance_bars("2025-03-01", "2025-03-01", freqs=("5s", "1min"), data_dir=tmp / "bars")
    assert len(calls) == n_calls, "cached day was downloaded again"
    ok("chunked loader matches full-day resample; cached day not re-downloaded")


# ---------------------------------------------------------------- CV
def test_purge():
    n, h = 400, 30
    times = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    target_end = times + pd.Timedelta(days=h)
    for f in ex.purged_walk_forward(times, target_end, 3, 20):
        assert (target_end[f.train] < times[f.test[0]]).all()
        assert f.train.max() < f.test.min()
        assert len(set(f.train) & set(f.test)) == 0
    ok("purged folds never train on a label that reaches the test block")


# ---------------------------------------------------------------- synthetic bars
def synth_bars(n, freq, binance):
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    vol = 0.002 * np.exp(0.6 * np.cumsum(rng.normal(0, 0.05, n)) / np.sqrt(np.arange(1, n + 1)))
    r = rng.normal(0, 1, n) * vol
    close = 30_000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 1, n)) * vol * close
    df = pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
        "close": close, "volume": rng.uniform(1, 10, n),
    }, index=idx)
    df["dollar_volume"] = df["volume"] * df["close"]
    if binance:
        df["signed_dollar_flow"] = rng.normal(0, 1, n) * df["dollar_volume"] * 0.2
    df.index.name = "timestamp"
    return df


def test_pipeline(tmp):
    synthetic = {
        "5s": synth_bars(3 * 17_280, "5s", True),
        "1min": synth_bars(14 * 1_440, "1min", True),
    }
    ex.fetch_binance_bars = lambda *a, **k: synthetic
    ex.load_or_download_yahoo = lambda *a, **k: synth_bars(900, "1D", False)

    cfg = ex.ExperimentConfig(
        target_rows={h: 300 for h in ex.HORIZON_LABELS},
        torch_epochs=2, xgb_estimators=10,
        data_dir=str(tmp / "data"), cache_dir=str(tmp / "cache"), output_dir=str(tmp / "results"),
    )
    root = ex.run_all(cfg)
    res = rp.load_results(root)

    assert set(res.meta) == set(ex.HORIZON_LABELS), res.meta.keys()
    assert res.metrics.Fold.nunique() == cfg.n_folds
    # GARCH only on daily horizons; every other model on every horizon
    garch = set(res.metrics[res.metrics.Model == "GARCH(1,1)"].HorizonKey)
    assert garch == set(ex.DAILY_HORIZONS), garch
    assert np.isfinite(res.metrics.RMSE).all()
    for m in ex.MODELS:
        assert m in set(res.metrics.Model), m
    assert {"Signature level", "Augmentation", "Dimensions"} <= set(res.ablations.Experiment)
    ok("all horizons x folds x models ran; GARCH daily-only; ablations present")

    # cache: second build must come from disk
    n_cached = len(list(Path(cfg.cache_dir).glob("*.npz")))
    assert n_cached >= 5
    ok(f"feature cache written ({n_cached} files)")

    # resume: nothing is recomputed
    before = (root / "1d" / "meta.json").stat().st_mtime
    ex.run_all(cfg)
    assert (root / "1d" / "meta.json").stat().st_mtime == before
    ok("resume skips completed horizons")

    summ = rp.summary_table(res)
    assert {"Mean_RMSE", "Pct_vs_HAR", "Rank"} <= set(summ.columns)
    for hk in ex.HORIZON_LABELS:
        t = rp.dm_table(res, hk)
        assert t.Best.sum() == 1 and len(t) >= 9
    ok("summary + Diebold-Mariano tables")

    import matplotlib.pyplot as plt
    for fn in (rp.plot_relative_heatmap, rp.plot_fold_stability, rp.plot_predictions,
               rp.plot_cumulative_advantage, rp.plot_regimes, rp.plot_ablations):
        fig = fn(res)
        fig.savefig(tmp / f"{fn.__name__}.png", dpi=40)
        plt.close(fig)
    ok("all six figures render")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_purge()
        test_loader(tmp)
        test_pipeline(tmp)
    print("\nALL SMOKE CHECKS PASSED")
