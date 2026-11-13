"""End-to-end check of the FX notebooks' code on synthetic data (no downloads, about a minute).

    python vol_experiments_fx/smoke_test.py      # from the repo root

Covers: the time-of-day-adjusted channels (no look-ahead, the clock pattern removed), the new path
dims, the vol run with the channel ablations, the direction target, the anomaly study with the new
channel sets, and the anomaly score in the vol forecasts.
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fx_lab as L  # noqa: E402
import experiment as ex  # noqa: E402
import forecast_data as fd  # noqa: E402
import reporting as rp  # noqa: E402
from anomaly import evaluate as AE, features as AF, report as AR, run as ARUN  # noqa: E402


def ok(msg: str) -> None:
    print(f"PASS {msg}", flush=True)


def synthetic_fx(n_weeks: int = 70, seed: int = 3) -> pd.DataFrame:
    """EUR/USD-like hourly bars on FX trading hours, with volatility and spreads that follow the
    New York clock (busy London/NY overlap, wide spreads at the 17:00 rollover)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-02 22:00", periods=24 * 7 * n_weeks, freq="h", tz="UTC")
    idx = idx[(idx + pd.Timedelta(hours=2)).dayofweek < 5]
    ny = idx.tz_convert("America/New_York").hour.to_numpy()
    clock = 1.0 + 1.5 * np.exp(-0.5 * ((ny - 9) / 2.5) ** 2)  # volatility peak around the NY morning
    regime = np.exp(np.cumsum(rng.normal(0, 0.02, len(idx))))  # slowly changing volatility level
    regime = regime / regime.mean()
    r = rng.standard_normal(len(idx)) * 4e-4 * clock * regime
    close = 1.1 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[1.1], close[:-1]])
    wiggle = np.abs(rng.standard_normal(len(idx))) * 2e-4 * clock * regime
    h = pd.DataFrame({"open": open_, "close": close,
                      "high": np.maximum(open_, close) * (1 + wiggle), "low": np.minimum(open_, close) * (1 - wiggle),
                      "volume": rng.gamma(2.0, 500.0, len(idx)) * clock}, index=pd.DatetimeIndex(idx, name="timestamp"))
    h["spread"] = 1e-5 * (1 + 4 * (ny == 17)) * np.exp(rng.normal(0, 0.2, len(idx)))
    return fd._fx_bar_columns(h)


def test_channels(h: pd.DataFrame) -> pd.DataFrame:
    c = fd.fx_hourly_channels(h)
    assert {"ret_sa", "range_sa", "spread_sa"} <= set(c.columns) and c[["ret_sa", "range_sa", "spread_sa"]].notna().all().all()
    # no look-ahead: changing the future leaves every earlier value unchanged
    cut = len(h) // 2
    h2 = h.copy()
    h2.iloc[cut:, h2.columns.get_indexer(["close", "high", "low", "spread"])] *= 1.3
    c2 = fd.fx_hourly_channels(h2)
    for col in ("ret_sa", "range_sa", "spread_sa"):
        assert np.allclose(c[col].iloc[:cut], c2[col].iloc[:cut]), f"{col} uses future bars"
    # the planted clock pattern is (mostly) removed after the warm-up
    late = c.iloc[len(c) // 3:]
    ny = pd.DatetimeIndex(late.index).tz_convert("America/New_York").hour
    raw = np.log(late.close).diff().abs().groupby(ny).mean()
    adj = late.ret_sa.abs().groupby(ny).mean()
    spread_raw = np.log(late.spread).groupby(ny).mean()
    assert raw.max() / raw.min() > 1.8 and adj.max() / adj.min() < 1.35, (raw.max() / raw.min(), adj.max() / adj.min())
    assert spread_raw.max() - spread_raw.min() > 1.0 and late.spread_sa.groupby(ny).mean().abs().max() < 0.15
    # path dims
    win = c.iloc[-100:]
    path = fd.make_base_path(win, ("price_sa", "activity", "range_sa", "spread"))  # columns: activity, price_sa, range_sa, spread
    assert path.shape == (100, 4) and np.allclose(path[0, 1], 0) and np.isclose(path[-1, 1], win.ret_sa.iloc[1:].sum())
    assert np.isclose(path[-1, 2], win.range_sa.sum()) and np.isclose(path[-1, 3], win.spread_sa.mean())
    fig = L.plot_seasonality(c)
    plt.close(fig)
    ok("time-of-day channels: past data only, clock pattern removed, new path dims, seasonality figure")
    return c


def test_vol_and_direction(tmp: Path, h: pd.DataFrame):
    ex._INTRADAY_MEMO.clear()
    over = {k: {"intraday_dims": L.MAIN_DIMS, "intraday_path_points": 24} for k in L.HORIZONS}
    cfg = L.vol_config(fx_start="2022-01-01", intraday_end=None, torch_epochs=2, spec_overrides=over,
                       data_dir=str(tmp / "data"), cache_dir=str(tmp / "cache"), output_dir=str(tmp / "vol"))
    res = rp.load_results(ex.run_all(cfg))
    assert set(res.meta) == set(L.HORIZONS) and set(res.metrics.Model) == set(cfg.models)
    assert np.isfinite(res.metrics.RMSE).all()
    abl = L.ablation_table(res)
    assert set(abl.Experiment) == {"Signature level", "Augmentation", "Channels"}
    assert set(abl[abl.Experiment == "Channels"].Setting) == set(L.CHANNEL_ABLATIONS)
    main = abl[(abl.Experiment == "Channels") & (abl.Setting == "Adjusted + spread (main)")].Mean_RMSE.to_numpy()
    full = res.metrics[res.metrics.Model == "Ridge | intraday signature"].groupby("Horizon", sort=False).RMSE.mean()
    assert np.allclose(np.sort(main), np.sort(full.to_numpy())), "the main channel setting is the main run's model"
    assert len(rp.pair_table(res, L.VOL_COMPARISONS)) and len(rp.dm_table(res, "1d_i"))
    for fig in (rp.plot_pair_skill(res, rp.as_pairs(L.VOL_COMPARISONS)),
                rp.plot_calibration(res, models=("HAR-RV-L", "HAR-RV-L calibrated", "Ridge | intraday signature"))):
        plt.close(fig)
    ok("vol run: hourly paths with the new channels, channel/level/augmentation ablations, comparisons")

    droot = L.run_direction(cfg)
    df, meta = L.load_direction(droot)
    assert set(df.Source) == {"vol forecast", "classifier"} and set(L.DIRECTION_FEATURES) <= set(df.Model)
    ds, spec, folds, hs = L.horizon_dataset(cfg, "5d_i")
    past = L.past_vol(ds, spec, hs)
    # the past-vol level equals the realized vol of the previous h daily bars
    bars = fd.add_variance_increment(ex.load_bars("5d_i", cfg), method=spec.variance_method)
    i = int(np.flatnonzero(bars.index == ds.times[-1])[0])
    assert np.isclose(past[-1], np.sqrt(bars.var_inc.iloc[i - hs + 1: i + 1].sum()), rtol=1e-4)
    t = L.direction_table(df, meta, n_boot=50)
    assert t.AUC.between(0, 1).all() and t.dropna(subset=["AUC_vs_ref"]).shape[0] > 0
    plt.close(L.plot_direction(t))
    assert L.run_direction(cfg) == droot  # finished horizons are skipped
    ok("vol direction: from the vol forecasts and from logistic classifiers, past-vol level, AUC table")
    return cfg


def test_pairs(tmp: Path, cfg: ex.ExperimentConfig):
    """Two pairs through the cross-pair section (synthetic bars stand in for every symbol)."""
    assert L.pair_vol_config("EURUSD").output_dir == L.vol_config().output_dir
    assert L.pair_vol_config("USDJPY").fx_symbol == "USDJPY" and "pairs/USDJPY/vol" in L.pair_vol_config("USDJPY").output_dir
    assert L.pair_vol_config("USDJPY", quick=True).output_dir.endswith("vol_quick")
    keep = {k: getattr(cfg, k) for k in ("fx_start", "intraday_end", "torch_epochs", "spec_overrides", "data_dir", "cache_dir")}
    cfgs = {"EURUSD": cfg}
    ex._INTRADAY_MEMO.clear()
    cfgs.update(L.run_pairs(("GBPUSD",), subprocess_per_horizon=False, output_dir=str(tmp / "pairs" / "GBPUSD" / "vol"), **keep))
    assert cfgs["GBPUSD"].fx_symbol == "GBPUSD" and (tmp / "pairs" / "GBPUSD" / "direction").exists()
    results = L.load_pairs(cfgs)
    level = L.pairs_level_table(results)
    assert set(level.Pair) == {"EURUSD", "GBPUSD"} and level["Signature, original channels"].notna().all()
    tests = L.pairs_test_table(results)
    assert {"HAR-RV-L", "Ridge | intraday lag bank (matched)"} <= set(tests.columns.get_level_values(0))
    d = L.pairs_direction_table(cfgs, n_boot=20)
    assert len(d) == 4 and d.notna().all().all()
    plt.close(L.plot_pairs(level))
    ok("more pairs: per-pair runs and direction, cross-pair level / test / direction tables, figure")


def test_cross(tmp: Path, h: pd.DataFrame, cfg: ex.ExperimentConfig):
    """Pairs together: other pairs' prices and the dollar factor in the path, three runs merged."""
    rng = np.random.default_rng(7)

    def fake(*a, symbol="EURUSD", **k):  # each pair: the synthetic bars times its own random walk
        if symbol == "EURUSD":
            return h
        walk = np.exp(np.cumsum(rng.normal(0, 1e-3, len(h))))
        out = h.copy()
        for c in ("open", "high", "low", "close"):
            out[c] = h[c] * walk * (100 if "JPY" in symbol else 1)
        return out.iloc[3:]  # a slightly different calendar: the first hours are missing

    # the channels: past data only, dollar sign convention, path dims
    others = {s: fake(symbol=s) for s in ("GBPUSD", "USDJPY")}
    c = fd.fx_cross_channels(h, "EURUSD", others)
    assert np.allclose(c["lclose__USDJPY"].dropna(), np.log(others["USDJPY"].close).reindex(c.index).dropna())
    assert c["lclose__GBPUSD"].iloc[:3].isna().all() and c["usd_log"].notna().all()
    r = np.log(h.close).diff()
    m = pd.concat([-r, -c["lclose__GBPUSD"].diff(), c["lclose__USDJPY"].diff()], axis=1).mean(axis=1)
    assert np.allclose(c.usd_log.diff().iloc[5:], m.iloc[5:])
    assert fd.usd_side("USDJPY") == 1 and fd.usd_side("AUDUSD") == -1 and fd.usd_side("EURGBP") == 0
    path = fd.make_base_path(c.iloc[:30], ("price", "activity", "range", "x:GBPUSD", "x:USDJPY", "usd"))
    assert path.shape == (30, 6) and np.isfinite(path).all() and np.allclose(path[0, 3:], 0)
    assert L.cross_dims("EURUSD", "pairs") == L.OWN_DIMS + ("x:GBPUSD", "x:USDJPY", "x:AUDUSD")

    saved = ex.fetch_dukascopy_hourly
    ex.fetch_dukascopy_hourly = fake
    try:
        ex._INTRADAY_MEMO.clear()
        keep = {k: getattr(cfg, k) for k in ("fx_start", "intraday_end", "torch_epochs", "data_dir", "cache_dir")}
        cfgs = {}
        for v in L.CROSS_VARIANTS:
            pts = {hk: {"intraday_dims": L.cross_dims("EURUSD", v), "intraday_path_points": 24} for hk in L.HORIZONS}
            cc = L.cross_config("EURUSD", v, output_dir=str(tmp / "cross" / v), spec_overrides=pts, **keep)
            assert cc.ridge_clip == L.RIDGE_CLIP and (v == "own") == (not cc.cross_pairs)
            ex.run_all(cc)
            cfgs[v] = cc
    finally:
        ex.fetch_dukascopy_hourly = saved
    res = L.load_cross({"EURUSD": cfgs})["EURUSD"]
    assert {"Signature | own pair", "Signature | + dollar factor", "Signature | + other pairs", "HAR-RV-L"} <= set(res.metrics.Model)
    level = L.cross_level_table({"EURUSD": res})
    assert level.drop(columns=["Pair", "Horizon"]).notna().all().all()
    t = L.cross_test_table({"EURUSD": res})
    assert "Signature | own pair" in set(t.columns.get_level_values(0))
    plt.close(L.plot_cross(level))
    assert len(L.worst_errors(res, "Signature | + other pairs", n=3)) == 6
    # clipping: an extreme test row cannot push the forecast far outside the training range
    X = rng.normal(size=(300, 5)); y = X[:, 0] + rng.normal(0, .1, 300)
    t_ = pd.date_range("2020", periods=300, freq="D").to_numpy()
    mdl = ex.fit_ridge(X, y, t_, t_, clip=5.0)
    assert abs(mdl.predict(np.array([[1e6, 0, 0, 0, 0]]))[0]) < 10
    ok("pairs together: cross-pair channels, dollar factor, three runs merged, tables, clipped Ridge")


def test_anomaly(tmp: Path, h: pd.DataFrame, cfg: ex.ExperimentConfig):
    acfg = L.anomaly_config(start="2022-01-01", end=None, min_train_rows=60, n_bootstrap=20, stage_b_top_k=1,
                            data_dir=str(tmp / "data"), output_dir=str(tmp / "anomaly"))
    root = ARUN.run_all(acfg)
    ares = AR.load_results(root)
    assert set(ares.meta) == {"1d", "5d"}
    keys = set(ares.metrics.Config)
    for ch in L.ANOMALY_CHANNELS:
        assert any(k.startswith("signature") and f"| {ch}|" in k for k in keys), ch
    assert len(AR.gate_table(ares)) and len(AR.synthetic_table(ares))
    ct = L.channel_table(ares)
    assert set(ct.Channels) == set(L.ANOMALY_CHANNELS) and ct.Select.between(0, 1).all()
    plt.close(AR.plot_timeline(ares, "5d"))

    # the order-only injection moves the adjusted returns with the raw ones
    spec = ARUN.horizon_specs("fx")["1d"]
    bars = ARUN.load_bars(spec, acfg)
    rows = AF.build_rows(bars, spec, spec.window, None)
    win = AF.extract_windows(bars, rows.pos[:40], spec.window, "trailing_vol")
    sub, inj = AE.reorder_injection(win, np.arange(40), seed=1)
    j = int(np.flatnonzero(inj)[0])
    order = np.argsort(win["r"][j], kind="stable")
    assert np.allclose(np.sort(sub["r_sa"][j]), np.sort(win["r_sa"][j]))
    assert np.allclose(sub["r_sa"][j], win["r_sa"][j][order]) or np.allclose(sub["r_sa"][j], win["r_sa"][j][order[::-1]])
    ok("anomaly study: the new FX channel sets, gate, synthetic check (adjusted returns re-ordered too)")

    vcfg = replace(cfg, use_anomaly=True, anomaly_selection=str(Path(root) / "best_configs.json"),
                   models=("Mean", "HAR-RV-L", "HAR-RV-L calibrated", *L.ANOMALY_MODELS), run_ablations=False,
                   output_dir=str(tmp / "vol_anomaly"))
    vres = rp.load_results(ex.run_all(vcfg))
    assert set(L.ANOMALY_MODELS) <= set(vres.metrics.Model) and np.isfinite(vres.metrics.RMSE).all()
    assert len(rp.anomaly_effect_table(vres))
    plt.close(rp.plot_anomaly_response(vres))
    ok("anomaly score in the vol forecasts (feature + residual), from the new anomaly study")


if __name__ == "__main__":
    h = synthetic_fx()
    for mod in (ex, ARUN, L):  # every loader gets the synthetic bars (with the spread)
        mod.fetch_dukascopy_hourly = lambda *a, **k: h
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_channels(h)
        cfg = test_vol_and_direction(tmp, h)
        test_pairs(tmp, cfg)
        test_cross(tmp, h, cfg)
        test_anomaly(tmp, h, cfg)
    print("\nALL FX SMOKE CHECKS PASSED")
