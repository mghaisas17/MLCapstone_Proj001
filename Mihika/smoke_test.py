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

import anomaly_flag as af
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


# ---------------------------------------------------------------- HMM
def test_hmm():
    T = 4000
    state = np.zeros(T, int)
    for t in range(1, T):
        state[t] = state[t - 1] if rng.random() < 0.97 else 1 - state[t - 1]
    sd = np.where(state == 1, 3.0, 0.7)[:, None]
    X = rng.normal(0, 1, (T, 3)) * sd
    hmm = af.DiagGaussianHMM(2, seed=0).fit(X[:2500], restarts=3)
    stress = int(np.argmax(hmm.vars_.sum(1)))
    p = hmm.filter(X)[:, stress]
    acc = ((p[2500:] > 0.5) == (state[2500:] == 1)).mean()
    assert acc > 0.9, acc
    assert np.allclose(hmm.filter(X[:1500]), hmm.filter(X)[:1500]), "filter must be causal"
    ok(f"HMM recovers a 2-regime process out of sample (acc={acc:.2f}); filtered probs are causal")


def test_to_ns():
    i = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    a = af.to_ns(i)
    assert a.dtype == np.dtype("datetime64[ns]") and (pd.DatetimeIndex(a).tz_localize("UTC") == i).all()
    ok("datetime handling is independent of pandas resolution")


# ---------------------------------------------------------------- controls / tuning / GARCH
def test_controls_and_tuning():
    import models_and_metrics as mm
    for horizon, n_bars, freq in (("1d", 400, "1D"), ("5m", 4000, "5s")):
        bars = synth_bars(n_bars, freq, horizon == "5m")
        spec = ex.base_spec_for(horizon, bars, ex.ExperimentConfig(target_rows={horizon: 40}))
        ds = fd.build_dataset(bars, spec)
        assert ds.X_lag.shape == ds.X_sig.shape, (ds.X_lag.shape, ds.X_sig.shape)
        assert ds.X_multi.shape[0] == len(ds.y) and np.isfinite(ds.X_multi).all() and np.isfinite(ds.X_lag).all()
    ok("lag-bank control has exactly as many columns as the signature (daily and intraday specs)")

    n = 300
    times = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    target_end = times + pd.Timedelta(days=10)
    fit, val = mm.inner_holdout(times, target_end)
    assert (target_end[fit] < times[val[0]]).all() and fit.max() < val.min()
    X = rng.normal(size=(n, 12))
    y = np.abs(X[:, 0]) + 0.1 * rng.normal(size=n)
    xgb = mm.fit_xgb_tuned(X, y, fit, val, max_estimators=40)
    mlp = mm.fit_mlp_tuned(X, y, fit, val, max_epochs=4, patience=2)
    lstm = mm.fit_lstm_tuned(X.reshape(n, 3, 4), y, fit, val, max_epochs=4, patience=2)
    for model, Xt in ((xgb, X), (mlp, X), (lstm, X.reshape(n, 3, 4))):
        assert np.isfinite(model.predict(Xt)).all() and model.tuned_
    ok("tuned XGBoost / MLP / LSTM fit on a purged inner holdout and predict")


def test_garch_rolling():
    import models_and_metrics as mm
    n = 1500
    sd = np.where(np.arange(n) < 600, 0.05, 0.01)  # volatility level drops: a stale long-run variance
    r = pd.Series(rng.normal(0, 1, n) * sd, index=pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC"))
    anchors = r.index[1000:1300:5]
    h = 30
    y = np.array([np.sqrt((r.loc[a:].iloc[1:h + 1] ** 2).sum()) for a in anchors])
    roll = mm.evaluate_garch_rolling(r, anchors, y, h, window=400, refit_every=30).predictions
    fixed = mm.evaluate_garch(r.loc[: r.index[999]], r, anchors, y, h).predictions
    assert np.isfinite(roll).all() and np.isfinite(fixed).all()
    b_roll, b_fix = roll.mean() / y.mean(), fixed.mean() / y.mean()
    ok(f"rolling GARCH runs (mean forecast / realized: rolling {b_roll:.2f} vs fixed {b_fix:.2f} on a series with a volatility-level drop)")


def test_flag_accuracy():
    n = 600
    t = pd.date_range("2024", periods=n, freq="D", tz="UTC")
    vol = np.exp(np.cumsum(rng.normal(0, 0.1, n)) * 0.3) * rng.lognormal(0, 0.3, n)
    p = np.clip((vol - vol.min()) / np.ptp(vol), 0, 1)
    fold = np.repeat([1, 2, 3], n // 3)
    base = {"Horizon": "1 day", "HorizonKey": "1d", "Fold": fold, "Time": t}
    preds = pd.DataFrame({**base, "Model": "HAR-RV-L", "Actual": vol, "Prediction": vol * rng.lognormal(0, 0.3, n)})
    mk = lambda pp: rp.Results(pd.DataFrame(), preds, pd.DataFrame(), {"1d": {"horizon_rows": 1}},
                               pd.DataFrame({**base, "p_stress": pp, "flag": (pp > 0.6).astype(float)}), pd.DataFrame())
    good = rp.flag_accuracy_table(mk(p)).query("Fold == 'All'").iloc[0]
    bad = rp.flag_accuracy_table(mk(rng.random(n))).query("Fold == 'All'").iloc[0]
    assert good.AUC_flag > 0.8 and abs(bad.AUC_flag - 0.5) < 0.1 and good.Lift > 2
    ok(f"flag accuracy table separates an informative flag (AUC {good.AUC_flag:.2f}) from a random one ({bad.AUC_flag:.2f})")


# ---------------------------------------------------------------- intraday paths
def hourly_and_daily(n_days=400):
    h = synth_bars(24 * n_days, "1h", True)
    d = h.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                              "volume": "sum", "dollar_volume": "sum"})
    return h, d


def test_klines(tmp):
    t0 = pd.Timestamp("2025-03-01", tz="UTC")
    minutes = pd.date_range(t0, periods=3 * 1440, freq="min")  # 3 days of 1-minute klines
    price = 30_000 * np.exp(np.cumsum(rng.normal(0, 3e-4, len(minutes))))
    qty, quote = rng.uniform(0.1, 3, len(minutes)), None
    quote = price * qty
    buy_q = quote * rng.uniform(0.2, 0.8, len(minutes))
    df = pd.DataFrame({
        "t": 0,
        "o": price, "h": price * 1.001, "l": price * 0.999, "c": price, "v": qty, "ct": 0,
        "q": quote, "n": 10, "tb": qty * 0.5, "tq": buy_q, "ig": 0,
    })
    df["t"] = (minutes - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(microseconds=1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("k.csv", df.to_csv(index=False, header=False))
    content = buf.getvalue()

    class Resp:
        def __init__(self, status): self.status_code = status
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_content(self, chunk_size):
            for i in range(0, len(content), 8192):
                yield content[i:i + 8192]

    calls = []
    def fake_get(url, **kw):
        calls.append(url)
        return Resp(404 if "/monthly/" in url else 200)  # no monthly archive yet -> daily fallback

    fd.requests.get = fake_get
    out = fd.fetch_binance_klines_hourly("2025-03-01", "2025-03-01", data_dir=tmp / "klines")
    assert any("/daily/" in u for u in calls) and any("/monthly/" in u for u in calls)
    first = out.iloc[:24]
    day = df.iloc[:1440]
    assert np.isclose(first["dollar_volume"].sum(), day["q"].sum())
    assert np.isclose(first["signed_dollar_flow"].sum(), (2 * day["tq"] - day["q"]).sum())
    assert np.isclose(first["close"].iloc[0], day["c"].iloc[59]) and np.isclose(first["high"].iloc[0], day["h"].iloc[:60].max())
    n = len(calls)
    fd.fetch_binance_klines_hourly("2025-03-01", "2025-03-01", data_dir=tmp / "klines")
    assert len([u for u in calls[n:] if "/daily/" in u]) == 0, "cached day downloaded again"
    ok("klines -> hourly bars: dollar volume, signed taker flow, OHLC; monthly->daily fallback; cache")


def test_intraday_no_lookahead():
    hourly, daily = hourly_and_daily(300)
    spec = ex.replace(fd.SPECS["1d_i"], intraday_path_points=24, sample_stride=7)
    base = fd.build_dataset(daily, spec, intraday=hourly)
    assert base.X_isig.shape == base.X_ilag.shape and base.X_isig.shape[0] == len(base.y)
    # rows only exist once a full intraday window is available
    assert base.times[0] >= hourly.index[0] + pd.Timedelta("30D") - pd.Timedelta("1D")
    T = hourly.index[24 * 200]
    tampered = hourly.copy()
    cols = ["open", "high", "low", "close"]
    tampered.loc[tampered.index > T, cols] = tampered.loc[tampered.index > T, cols] * 3.0  # change only the future
    alt = fd.build_dataset(daily, spec, intraday=tampered)
    known = base.times + pd.Timedelta("1D")
    past = np.asarray(known <= T)
    assert past.sum() > 5 and (~past).sum() > 5
    for name in ("X_isig", "X_ilag", "X_imulti", "X_harrv"):
        a, b = getattr(base, name), getattr(alt, name)
        assert np.allclose(a[past], b[past]), f"{name} changed when only future intraday bars changed"
    assert not np.allclose(base.X_isig[~past], alt.X_isig[~past])
    ok("intraday windows use no bars after the row's known time (tampering with the future changes nothing earlier)")


def test_intraday_pipeline(tmp):
    hourly, daily = hourly_and_daily(400)
    ex.load_or_download_yahoo = lambda *a, **k: daily
    ex.fetch_binance_klines_hourly = lambda *a, **k: hourly
    models = ("Mean", "HAR-RV-L", "Ridge | stats", "Ridge | signature", "GARCH(1,1) | rolling refit",
              "Ridge | intraday signature", "Ridge | intraday lag bank (matched)", "Ridge | intraday multiscale stats",
              "HAR-RV-L (intraday RV)", "Ridge | intraday signature + HAR-RV")
    cfg = ex.ExperimentConfig(
        horizons=("1d_i", "7d_i"), models=models, torch_epochs=2, xgb_estimators=10, target_rows={},
        spec_overrides={k: {"intraday_path_points": 24} for k in ("1d_i", "7d_i")},
        data_dir=str(tmp / "data"), cache_dir=str(tmp / "cache"), output_dir=str(tmp / "results_intraday"),
    )
    root = ex.run_all(cfg)
    res = rp.load_results(root)
    assert set(res.meta) == {"1d_i", "7d_i"} and all(m["intraday_paths"] for m in res.meta.values())
    assert set(res.metrics.Model) == set(models), set(models) ^ set(res.metrics.Model)
    assert res.flags.empty and res.ablations.empty
    assert np.isfinite(res.metrics.RMSE).all()
    # rows are the same for every model on a horizon (same target, same rows)
    for hk in ("1d_i", "7d_i"):
        p = res.predictions[res.predictions.HorizonKey == hk]
        assert p.groupby("Model").Time.apply(lambda t: tuple(t)).nunique() == 1
    tab = rp.intraday_table(res)
    assert {"Versus", "Pct_better", "DM_p"} <= set(tab.columns) and np.isfinite(tab.Pct_better).all()
    assert set(rp.summary_table(res).Horizon.astype(str)) == {"1 day (intraday paths)", "7 day (intraday paths)"}
    ok("intraday-path horizons: same rows for every model, new models + controls run, comparison table")


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


def _dump(cfg, tmp):
    p = tmp / "variant_config.json"
    cfg.to_json(p)
    return p


def test_pipeline(tmp):
    synthetic = {
        "5s": synth_bars(3 * 17_280, "5s", True),
        "1min": synth_bars(14 * 1_440, "1min", True),
    }
    ex.fetch_binance_bars = lambda *a, **k: synthetic
    ex.load_or_download_yahoo = lambda *a, **k: synth_bars(900, "1D", False)

    cfg = ex.ExperimentConfig(
        target_rows={h: 300 for h in ex.MAIN_HORIZONS},
        torch_epochs=2, xgb_estimators=10,
        data_dir=str(tmp / "data"), cache_dir=str(tmp / "cache"), output_dir=str(tmp / "results"),
    )
    root = ex.run_all(cfg)
    res = rp.load_results(root)

    assert set(res.meta) == set(ex.MAIN_HORIZONS), res.meta.keys()
    assert res.metrics.Fold.nunique() == cfg.n_folds
    # GARCH only on daily horizons; every other model on every horizon
    for g in ("GARCH(1,1) | fixed", "GARCH(1,1) | rolling refit"):
        garch = set(res.metrics[res.metrics.Model == g].HorizonKey)
        assert garch == set(ex.DAILY_HORIZONS) & set(ex.MAIN_HORIZONS), (g, garch)
    assert np.isfinite(res.metrics.RMSE).all()
    for m, spec in ex.MODELS.items():
        if not spec.needs_intraday:  # intraday-path models run only on the *_i horizons
            assert m in set(res.metrics.Model), m
    assert not any(ex.MODELS[m].needs_intraday for m in set(res.metrics.Model))
    assert {"Signature level", "Augmentation", "Dimensions"} <= set(res.ablations.Experiment)
    flag_models = {n for n, m in ex.MODELS.items() if m.needs_flag}
    assert flag_models <= set(res.metrics.Model), flag_models - set(res.metrics.Model)
    assert not res.flags.empty and not res.flag_diag.empty
    assert set(res.flags.flag.unique()) <= {0.0, 1.0}
    assert res.flags.p_stress.between(0, 1).all()
    assert len(res.flags) == len(res.predictions[res.predictions.Model == "Mean"])
    ok("all horizons x folds x models ran; GARCH daily-only; ablations present")

    # cache: second build must come from disk
    n_cached = len(list(Path(cfg.cache_dir).glob("*.npz")))
    assert n_cached >= 5
    ok(f"feature cache written ({n_cached} files)")
    bars = ex.load_bars("1d", cfg)
    spec = ex.base_spec_for("1d", bars, cfg)
    again = ex.get_dataset(bars, spec, cfg, include_seq=True, sub_window=cfg.lstm_sub_window["1d"])
    again2 = ex.get_dataset(bars, spec, cfg, include_seq=True, sub_window=cfg.lstm_sub_window["1d"])
    assert (again.times == again2.times).all() and again.times.tz is not None
    assert again.times[0] >= bars.index[0] and again.times[-1] <= bars.index[-1], "cached times corrupted"
    ok("cached features reload with correct timestamps")

    # resume: nothing is recomputed
    before = (root / "1d" / "meta.json").stat().st_mtime
    ex.run_all(cfg)
    assert (root / "1d" / "meta.json").stat().st_mtime == before
    ok("resume skips completed horizons")

    summ = rp.summary_table(res)
    assert {"Mean_RMSE", "Pct_vs_HAR", "Rank"} <= set(summ.columns)
    for hk in ex.MAIN_HORIZONS:
        t = rp.dm_table(res, hk)
        assert t.Best.sum() == 1 and len(t) >= 9
    ok("summary + Diebold-Mariano tables")

    # a signature variant through the same pipeline, then compared against the main run
    from dataclasses import replace as _r
    variant = _r(
        cfg, horizons=("1d",), spec_overrides={"1d": {"depth": 2, "dims": ["price", "activity"]}},
        models=("Mean", "HAR-RV-L", "Ridge | signature"), run_ablations=False, use_flag=False,
        output_dir=str(tmp / "results_variant"),
    )
    ex.run_all(_r(variant, spec_overrides=ex.ExperimentConfig.from_json(_dump(variant, tmp)).spec_overrides))
    vres = rp.load_results(variant.output_dir)
    assert set(vres.metrics.Model) == {"Mean", "HAR-RV-L", "Ridge | signature"}
    cmp_all = rp.compare_runs({"main": root, "depth2": variant.output_dir}, folds=None)
    cmp_f12 = rp.compare_runs({"main": root, "depth2": variant.output_dir}, folds=(1, 2))
    assert "1 day" in cmp_all.columns and set(cmp_all.index) == {"main", "depth2"}
    assert np.isfinite(cmp_f12.loc["depth2", "1 day"])
    n_main = ex.get_dataset(ex.load_bars("1d", cfg), ex.base_spec_for("1d", ex.load_bars("1d", cfg), cfg), cfg,
                            include_seq=False, sub_window="5D").X_sig.shape[1]
    n_var = ex.get_dataset(ex.load_bars("1d", variant), ex.base_spec_for("1d", ex.load_bars("1d", variant), variant),
                           variant, include_seq=False, sub_window="5D").X_sig.shape[1]
    assert n_var < n_main, (n_var, n_main)
    try:
        ex.base_spec_for("1d", ex.load_bars("1d", cfg), _r(cfg, spec_overrides={"1d": {"bogus": 1}}))
        raise AssertionError("unknown override accepted")
    except ValueError:
        pass
    ok(f"spec_overrides: variant run + compare_runs ({n_main} -> {n_var} signature columns); bad keys rejected")

    ct = rp.control_table(res)
    assert set(ct.Control) == set(rp.CONTROLS) and np.isfinite(ct.RMSE_control).all()
    bt = rp.bias_table(res)
    assert {"Fold 1", "Fold 3", "Pooled"} <= set(bt.columns) and np.isfinite(bt.Pooled).all()
    ok("control table (signature vs signature-free) and bias table")

    eff = rp.flag_effect_table(res)
    assert len(eff) == 4 * len(ex.MAIN_HORIZONS) and np.isfinite(eff.RMSE_base_all).all()
    diag = rp.flag_diagnostics_table(res)
    assert (diag.Share_flagged_test.between(0, 1)).all()
    acc = rp.flag_accuracy_table(res)
    assert (acc.Fold == "All").sum() == len(ex.MAIN_HORIZONS) and acc.Share_flagged.between(0, 1).all()
    assert acc.Precision.dropna().between(0, 1).all() and acc.AUC_flag.dropna().between(0, 1).all()
    ok("flag models, out-of-sample flags, diagnostics and flag-effect table")

    import matplotlib.pyplot as plt
    for fn in (rp.plot_flag_timeline, rp.plot_relative_heatmap, rp.plot_fold_stability, rp.plot_predictions,
               rp.plot_cumulative_advantage, rp.plot_regimes, rp.plot_ablations):
        fig = fn(res)
        fig.savefig(tmp / f"{fn.__name__}.png", dpi=40)
        plt.close(fig)
    ok("all figures render")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_purge()
        test_hmm()
        test_to_ns()
        test_garch_rolling()
        test_flag_accuracy()
        test_loader(tmp)
        test_klines(tmp)
        test_intraday_no_lookahead()
        test_controls_and_tuning()
        test_pipeline(tmp)
        test_intraday_pipeline(tmp)
    print("\nALL SMOKE CHECKS PASSED")
