"""
weight_spread_ratio.py
======================

Per-catchment RATIO of the WEIGHTED ensemble's 10-90 % inter-percentile spread
of the projected AnnMax change to the same spread obtained when the SAME weight
vector is randomly re-assigned to the climate models (co-author request,
11 Aug 2026):

    R_shuffled(catchment) =            q90(x, w) - q10(x, w)
                            ------------------------------------------------
                            mean over S shuffles of  q90(x, w_perm) - q10(x, w_perm)

with

  x        the per-climate-model projected AnnMax change at that catchment
           (members averaged within a model first: one model, one vote —
           the project-wide convention, hydro_indices_change.aggregate),
  w        the frozen weights of model_weights.load_frozen() (BMA on TCR by
           default; weights=kde selects the density-ratio variant),
  w_perm   the same weight VALUES permuted across the models (the weight
           vector is shuffled, so the weight distribution is preserved and
           only the model->weight assignment is destroyed),
  q10/q90  weighted quantiles under the definition already used for the
           10-90 % band of Fig. 3b,d (hydro_indices_change._weighted_quantile:
           linear interpolation of the weighted ECDF at its midpoints).

Reading the map:  R < 1  the TCR-informed weights compress the climate-model
spread MORE than a random assignment of the same weights;  R = 1  the weights
buy nothing over a random assignment;  R > 1  they widen the spread.

The permutation is drawn ONCE per shuffle and applied at every catchment
(weights are a property of the model, not of the catchment), so the S draws are
shared across catchments; the per-catchment denominator is still the mean of S
independent draws of that catchment's shuffled spread.

Byproduct, written to the per-engine CSV: the one-sided permutation p-value
    p = (#{shuffles with spread <= observed weighted spread} + 1) / (S + 1)
i.e. the per-catchment weight-permutation null — the weighting-mode analogue of
the random-removal null of bootstrap_random_removal.py (which exits under
method=weighting). Add-one correction as in that script.

Also computed, for context and for the comparison the manuscript needs (the
attached template plots exactly this second quantity: "model subset uncertainty
/ all models uncertainty"):

    R_equal    = spread(weighted)  / spread(equal weights, all models)
    R_screened = spread(TCR<=2.2 models only) / spread(equal weights, all models)

All three ratios use ONE quantile estimator (the weighted one, with equal
weights standing in for the unweighted case), so numerator and denominator are
always commensurate. That estimator is the type-5/Harrell midpoint rule, which
differs marginally from the pandas `quantile()` (type 7) used in the unweighted
branch of Fig. 3b,d — spreads here are therefore not bit-identical to that
figure, but every ratio is internally consistent.

Inputs  : fig_data/fig05_hydro_change_{hbv,lstm}.parquet (Stage-1 per-
          realization caches; method-independent, never re-tagged)
          fig_data/model_weights_wt.csv (frozen weights)
Outputs : fig_data/annmax_spread_ratio_{scen}_{period}_{engine}_n<N>sta_wt.csv
          fig_data/annmax_spread_ratio_summary_{scen}_{period}_wt.csv
          hydro_indices/maps/annmax_spread_ratio_{scen}_{period}_n<N>sta_wt.png
          hydro_indices/maps/annmax_spread_ratio_compare_{...}_wt.png
          (_wtkde instead of _wt under weights=kde)

Run (via slurm — slurm/weight_spread_ratio.sh; no python from the login shell):

    python paper/figures/scripts/weight_spread_ratio.py \
        [weights=bma|kde] [shuffles=1000] [scenario=ssp585] [period=2081-2100] \
        [metric=peakflow_change_pct] [lim=0.25,1.75]

`method=weighting` is forced: the whole analysis is defined only for a weighted
ensemble, so the script sets it rather than accepting method=removal.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

_THIS = os.path.dirname(os.path.abspath(__file__))            # paper/figures/scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
sys.path.insert(0, _THIS)

from config import CONFIG
from realization_blocklist import apply_blocklist
from figstyle import add_panel_letters
import ensemble_method as em

# fig02 owns the model universe and the TCR screen; it parses sys.argv at import
# time, so neutralise argv around the import (same guard hydro_indices_change uses).
_argv_bak = sys.argv
sys.argv = sys.argv[:1]
try:
    from fig02_precip_change import FULL_MODELS, REDUCED_MODELS, EXCLUDED_MODELS
finally:
    sys.argv = _argv_bak

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
ROOT      = os.path.join(CONFIG.out_path, CONFIG.delta_method)
CACHE_DIR = os.path.join(ROOT, "fig_data")
MAPS_DIR  = os.path.join(os.path.dirname(_THIS), "hydro_indices", "maps")

ENGINES   = ("hbv", "lstm")
ENGINE_LABEL = {"hbv": "HBV", "lstm": "LSTM"}

SCENARIO  = "ssp585"          # same focus as Fig. 3 / the significance battery
PERIOD    = "2081-2100"
METRIC    = "peakflow_change_pct"       # AnnMax, per cent (Fig. 3b,d quantity)
METRIC_LABEL = "AnnMax"

BAND      = (0.10, 0.90)      # the "10-90 % CI" of the request
N_SHUFFLES   = 1000           # denominator = mean over this many weight shuffles
SHUFFLE_SEED = 20260811       # frozen, stamped into the summary CSV
MIN_MODELS   = 10             # catchments with fewer models present are dropped
CHECK_ROWS   = 200            # rows used by the vectorised-quantile self-check

CMAP      = "PRGn_r"          # green below 1, white at 1, purple above 1
LIM       = None              # (lo, hi) colour limits; None = data-driven
MARKER_S  = 4


# ===========================================================================
#  weighted quantiles
# ===========================================================================
def _weighted_quantile_ref(v, w, qs):
    """VERBATIM copy of hydro_indices_change._weighted_quantile — kept here as
    the reference the vectorised routine is checked against, so the two can
    never drift apart silently."""
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    if v.size == 0 or w.sum() <= 0:
        return np.full(len(np.atleast_1d(qs)), np.nan)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.atleast_1d(qs), cw, v)


def _wq_rows(xs, ws, qs):
    """Weighted quantiles of every ROW, vectorised.

    xs : (n, M) each row sorted ascending with NaN in the trailing slots
    ws : (n, M) non-negative weights aligned to xs, 0 in the NaN slots
    qs : sequence of probabilities
    Returns (len(qs), n).

    Same estimator as _weighted_quantile_ref: linear interpolation of the
    weighted ECDF evaluated at the midpoints cw = (cumsum(w) - w/2)/sum(w),
    clamped to the first/last value outside [cw_0, cw_k-1].
    """
    n, _ = xs.shape
    valid = np.isfinite(xs)
    k = valid.sum(axis=1)                       # models present per catchment
    tot = ws.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cw = (np.cumsum(ws, axis=1) - 0.5 * ws) / np.where(tot > 0, tot, np.nan)
    # invalid slots must never be found by the search, and must never be
    # interpolated onto: push them to +inf.
    cw_search = np.where(valid, cw, np.inf)
    idx_last = np.maximum(k - 1, 0)[:, None]
    x_first = xs[:, 0]
    x_last = np.take_along_axis(xs, idx_last, axis=1)[:, 0]

    out = np.empty((len(qs), n), float)
    for i, q in enumerate(qs):
        j = (cw_search < q).sum(axis=1)         # 0 .. k
        jh = np.clip(j, 1, np.maximum(k - 1, 1))[:, None]
        jl = jh - 1
        xl = np.take_along_axis(xs, jl, axis=1)[:, 0]
        xh = np.take_along_axis(xs, jh, axis=1)[:, 0]
        cl = np.take_along_axis(cw, jl, axis=1)[:, 0]
        ch = np.take_along_axis(cw, jh, axis=1)[:, 0]
        d = ch - cl
        t = np.where(d > 0, (q - cl) / np.where(d > 0, d, 1.0), 0.0)
        v = xl + t * (xh - xl)
        v = np.where(j <= 0, x_first, v)        # below the first midpoint
        v = np.where(j >= k, x_last, v)         # above the last midpoint
        out[i] = np.where(k > 0, v, np.nan)
    return out


def _check_wq(xs, ws, qs, rng, n_rows=CHECK_ROWS):
    """Assert the vectorised quantiles equal the reference implementation on a
    random sample of rows. Cheap, and it makes the estimator claim testable."""
    n = xs.shape[0]
    rows = rng.choice(n, size=min(n_rows, n), replace=False)
    fast = _wq_rows(xs[rows], ws[rows], qs)
    ref = np.array([_weighted_quantile_ref(xs[r], np.where(np.isfinite(xs[r]),
                                                           ws[r], np.nan), qs)
                    for r in rows]).T
    bad = np.nanmax(np.abs(fast - ref))
    if not np.isfinite(bad) or bad > 1e-9:
        raise AssertionError(f"vectorised weighted quantile differs from the "
                             f"reference by {bad:g} on {len(rows)} rows")
    logger.info(f"  weighted-quantile self-check passed on {len(rows)} rows "
                f"(max |diff| = {bad:.2e})")


# ===========================================================================
#  data
# ===========================================================================
def _cache_path(engine):
    return os.path.join(CACHE_DIR, f"fig05_hydro_change_{engine}.parquet")


def load_matrix(engine, scenario, period, metric):
    """(stations, lon, lat, models, X) for one engine.

    X is (n_catchments, n_models) of the per-climate-model projected change in
    `metric` at (scenario, period): members averaged within a model first.
    NaN marks a model absent at that catchment."""
    path = _cache_path(engine)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — build the Stage-1 cache "
                                "with hydro_indices_change.py first")
    cols = ["station", "lon", "lat", "scenario", "model", "member",
            "period", metric]
    flt = [[("scenario", "==", scenario), ("period", "==", period)]]
    try:
        df = pd.read_parquet(path, columns=cols, filters=flt)
    except Exception as e:                       # old pyarrow: read then filter
        logger.warning(f"  pushdown filter failed ({e}); reading in full")
        df = pd.read_parquet(path, columns=cols)
        df = df[(df["scenario"].astype(str) == scenario)
                & (df["period"].astype(str) == period)]
    logger.info(f"[{engine}] {len(df)} cached realization rows at "
                f"{scenario}/{period}")
    df = apply_blocklist(df, label=f"weight_spread_ratio {engine}")
    df = df[df["model"].isin(FULL_MODELS)].dropna(subset=[metric])
    if df.empty:
        raise RuntimeError(f"[{engine}] no rows for {scenario}/{period}")

    per_model = df.groupby(["station", "model"], as_index=False)[metric].mean()
    piv = per_model.pivot(index="station", columns="model", values=metric)
    piv = piv.sort_index()
    coords = (df.groupby("station")[["lon", "lat"]].first()
                .reindex(piv.index))

    models = list(piv.columns)
    X = piv.to_numpy(dtype=float)
    k = np.isfinite(X).sum(axis=1)
    keep = k >= MIN_MODELS
    if (~keep).any():
        logger.info(f"[{engine}] dropped {int((~keep).sum())} catchments with "
                    f"< {MIN_MODELS} climate models present")
    logger.info(f"[{engine}] {int(keep.sum())} catchments x {len(models)} "
                f"climate models; models present per catchment: "
                f"median {int(np.median(k[keep]))}, min {int(k[keep].min())}")
    return (piv.index.to_numpy()[keep], coords["lon"].to_numpy()[keep],
            coords["lat"].to_numpy()[keep], models, X[keep])


# ===========================================================================
#  spreads and ratios
# ===========================================================================
def _spread(xs, ws, band=BAND):
    q = _wq_rows(xs, ws, band)
    return q[1] - q[0]


def spread_ratios(X, models, n_shuffles=N_SHUFFLES, seed=SHUFFLE_SEED):
    """All spreads and ratios for one engine's (n, M) model matrix.

    Returns a dict of 1-D arrays, all of length n."""
    order = np.argsort(X, axis=1, kind="stable")        # NaN sorts last
    xs = np.take_along_axis(X, order, axis=1)
    valid = np.isfinite(xs)

    w = em.weight_array(models)                 # frozen weights, this model set
    ws = np.where(valid, w[order], 0.0)
    we = valid.astype(float)                    # equal weights, all models

    rng = np.random.default_rng(seed)
    _check_wq(xs, ws, BAND, rng)

    s_w = _spread(xs, ws)
    s_eq = _spread(xs, we)

    # Screened ensemble, equal weights over the retained models. Resolved from
    # em HERE, not from the REDUCED_MODELS imported at module load: main()
    # calls em.set_method("weighting"), which resets SCREEN to "hot" and METRIC
    # to "tcr", so the imported constant can be a stale two-sided or ECS set
    # under a _wt tag.
    retained_now = set(em.retained_models())
    ired = [i for i, m in enumerate(models) if m in retained_now]
    Xr = X[:, ired]
    xr = np.take_along_axis(Xr, np.argsort(Xr, axis=1, kind="stable"), axis=1)
    s_red = _spread(xr, np.isfinite(xr).astype(float))

    # --- the null: the same weight VALUES, randomly re-assigned to models ----
    acc = np.zeros(len(s_w))
    acc2 = np.zeros(len(s_w))
    below = np.zeros(len(s_w))
    for s in range(n_shuffles):
        wp = rng.permutation(w)
        sp = _spread(xs, np.where(valid, wp[order], 0.0))
        acc += sp
        acc2 += sp * sp
        below += (sp <= s_w)
        if (s + 1) % 200 == 0:
            logger.info(f"  shuffle {s + 1}/{n_shuffles}")
    s_sh = acc / n_shuffles
    var = np.maximum(acc2 / n_shuffles - s_sh ** 2, 0.0)

    def _ratio(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(b > 0, a / np.where(b > 0, b, np.nan), np.nan)
        return r

    return {
        "n_models": np.isfinite(X).sum(axis=1),
        "spread_weighted": s_w,
        "spread_shuffled_mean": s_sh,
        "spread_shuffled_sd": np.sqrt(var),
        "spread_equal": s_eq,
        "spread_screened": s_red,
        "ratio_shuffled": _ratio(s_w, s_sh),
        "ratio_equal": _ratio(s_w, s_eq),
        "ratio_screened": _ratio(s_red, s_eq),
        # one-sided: how often a random weight assignment compresses the
        # ensemble at least as much as the TCR-informed one (add-one corrected)
        "p_perm": (below + 1.0) / (n_shuffles + 1.0),
    }


def compute(scenario, period, metric, n_shuffles):
    """Per-engine result frames + the summary table."""
    out = {}
    rows = []
    for engine in ENGINES:
        stn, lon, lat, models, X = load_matrix(engine, scenario, period, metric)
        res = spread_ratios(X, models, n_shuffles=n_shuffles)
        d = pd.DataFrame({"station": stn, "lon": lon, "lat": lat, **res})
        out[engine] = d
        csv = em.tag(os.path.join(
            CACHE_DIR, f"annmax_spread_ratio_{scenario}_{period}_{engine}"
                       f"_n{len(d)}sta.csv"))
        d.to_csv(csv, index=False)
        logger.info(f"[{engine}] -> {csv}")

        row = {"engine": engine, "scenario": scenario, "period": period,
               "metric": metric, "n_sta": len(d), "n_models": len(models),
               "weights": em.variant(), "n_shuffles": n_shuffles,
               "seed": SHUFFLE_SEED,
               "n_eff_weights": em.n_eff_present(models)}
        for c in ("ratio_shuffled", "ratio_equal", "ratio_screened"):
            v = d[c].to_numpy()
            row[f"{c}_median"] = float(np.nanmedian(v))
            row[f"{c}_p25"] = float(np.nanpercentile(v, 25))
            row[f"{c}_p75"] = float(np.nanpercentile(v, 75))
            ok = np.isfinite(v)
            row[f"{c}_frac_below1"] = float(np.mean(v[ok] < 1.0))
        row["p_perm_median"] = float(np.nanmedian(d["p_perm"]))
        row["frac_p_perm_lt_0.05"] = float(np.nanmean(d["p_perm"] < 0.05))
        row["spread_weighted_median"] = float(np.nanmedian(d["spread_weighted"]))
        row["spread_equal_median"] = float(np.nanmedian(d["spread_equal"]))
        rows.append(row)
        logger.info(
            f"[{engine}] median ratios — weighted/shuffled "
            f"{row['ratio_shuffled_median']:.4f}, weighted/equal "
            f"{row['ratio_equal_median']:.4f}, screened/equal "
            f"{row['ratio_screened_median']:.4f}; median permutation "
            f"p = {row['p_perm_median']:.3f}")

    summ = pd.DataFrame(rows)
    scsv = em.tag(os.path.join(
        CACHE_DIR, f"annmax_spread_ratio_summary_{scenario}_{period}.csv"))
    summ.to_csv(scsv, index=False)
    logger.info(f"summary -> {scsv}")
    print(summ.to_string(index=False))
    return out, summ


# ===========================================================================
#  figures
# ===========================================================================
def _global_ax(fig, spec):
    """Robinson map axes (mirrors gev_fdr_dipole._global_ax, incl. its
    no-cartopy fallback)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax = fig.add_subplot(spec, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.5")
        return ax, dict(transform=ccrs.PlateCarree())
    except Exception:
        ax = fig.add_subplot(spec)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        return ax, {}


def _limits(arrays, override=None, lo_pct=2.0, hi_pct=98.0):
    """Colour limits centred on 1 and symmetric in log-ratio, from the
    lo/hi percentiles of the pooled data (so both hydrological models share
    one scale). `override` is an explicit (lo, hi) from the CLI."""
    if override is not None:
        return float(override[0]), float(override[1])
    v = np.concatenate([np.asarray(a, float) for a in arrays])
    v = v[np.isfinite(v) & (v > 0)]
    lo = np.percentile(v, lo_pct)
    hi = np.percentile(v, hi_pct)
    m = max(abs(np.log(lo)), abs(np.log(hi)), 0.02)
    m = np.ceil(m * 100.0) / 100.0
    return float(np.exp(-m)), float(np.exp(m))


def _map_panel(ax, tk, d, col, norm, title):
    sc = ax.scatter(d["lon"], d["lat"], c=d[col], s=MARKER_S, cmap=CMAP,
                    norm=norm, linewidths=0, rasterized=True, **tk)
    ax.set_title(title, fontsize=10)
    return sc


def plot_ratio_maps(data, scenario, period, lim=None):
    """The requested figure: weighted / shuffled-weight 10-90 % spread ratio,
    one map per hydrological model."""
    engines = [e for e in ENGINES if e in data]
    lo, hi = _limits([data[e]["ratio_shuffled"] for e in engines], lim)
    norm = TwoSlopeNorm(vmin=lo, vcenter=1.0, vmax=hi)
    n_sta = max(len(data[e]) for e in engines)

    fig = plt.figure(figsize=(7.2, 3.7 * len(engines) + 0.9))
    gs = fig.add_gridspec(len(engines), 1, hspace=0.06)
    axes, sc = [], None
    for i, e in enumerate(engines):
        ax, tk = _global_ax(fig, gs[i, 0])
        med = float(np.nanmedian(data[e]["ratio_shuffled"]))
        sc = _map_panel(ax, tk, data[e], "ratio_shuffled", norm,
                        f"{ENGINE_LABEL[e]} (median ratio {med:.3f}, "
                        f"n = {len(data[e]):,})")
        axes.append(ax)

    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.05,
                      pad=0.03, aspect=40, extend="both")
    cb.set_label(f"Ratio of weighted-ensemble 10 to 90% spread of projected "
                 f"{METRIC_LABEL} change\nto shuffled-weight spread",
                 fontsize=10)
    cb.ax.tick_params(labelsize=9)
    add_panel_letters(axes)
    out = em.tag(os.path.join(
        MAPS_DIR, f"annmax_spread_ratio_{scenario}_{period}_n{n_sta}sta.png"))
    os.makedirs(MAPS_DIR, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"figure -> {out}")
    return out


def plot_compare_maps(data, scenario, period, lim=None):
    """Companion figure: the same ratio (left) beside the exclusion-mode
    equivalent, screened-ensemble spread / full-ensemble spread (right) —
    the quantity the co-author's template plots."""
    engines = [e for e in ENGINES if e in data]
    cols = [("ratio_shuffled",
             "Weighted / shuffled-weight spread"),
            ("ratio_screened",
             "Screened / full-ensemble spread")]
    norms = []
    for c, _ in cols:
        lo, hi = _limits([data[e][c] for e in engines],
                         lim if c == "ratio_shuffled" else None)
        norms.append(TwoSlopeNorm(vmin=lo, vcenter=1.0, vmax=hi))
    n_sta = max(len(data[e]) for e in engines)

    fig = plt.figure(figsize=(12.4, 3.4 * len(engines) + 0.9))
    gs = fig.add_gridspec(len(engines), 2, hspace=0.06, wspace=0.04)
    axes = []
    mappables = [None, None]
    col_axes = [[], []]
    for i, e in enumerate(engines):
        for j, (c, ttl) in enumerate(cols):
            ax, tk = _global_ax(fig, gs[i, j])
            med = float(np.nanmedian(data[e][c]))
            mappables[j] = _map_panel(ax, tk, data[e], c, norms[j],
                                      f"{ENGINE_LABEL[e]}: {ttl.lower()} "
                                      f"(median {med:.3f})")
            axes.append(ax)
            col_axes[j].append(ax)
    for j, (c, ttl) in enumerate(cols):
        cb = fig.colorbar(mappables[j], ax=col_axes[j], orientation="horizontal",
                          fraction=0.05, pad=0.03, aspect=35, extend="both")
        cb.set_label(f"{ttl} ({METRIC_LABEL} change, 10 to 90%)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    add_panel_letters(axes)
    out = em.tag(os.path.join(
        MAPS_DIR,
        f"annmax_spread_ratio_compare_{scenario}_{period}_n{n_sta}sta.png"))
    os.makedirs(MAPS_DIR, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"figure -> {out}")
    return out


# ===========================================================================
#  CLI
# ===========================================================================
def _parse(argv):
    opt = {"scenario": SCENARIO, "period": PERIOD, "metric": METRIC,
           "shuffles": N_SHUFFLES, "lim": LIM}
    for tok in argv:
        if "=" not in tok:
            raise SystemExit(f"unknown argument {tok!r} (expected key=value)")
        k, v = tok.split("=", 1)
        if k == "shuffles":
            opt["shuffles"] = int(v)
        elif k == "lim":
            lo, hi = (float(x) for x in v.split(","))
            opt["lim"] = (lo, hi)
        elif k in opt:
            opt[k] = v
        else:
            raise SystemExit(f"unknown option {k!r}")
    return opt


def main(argv):
    argv = em.reject_cli_tokens(argv)
    if not em.is_weighting():
        em.set_method("weighting")      # the analysis is weighting-only
    # Provenance: logged AFTER the weighting-only switch above, so the line
    # records the treatment this run actually used.
    em.log_treatment(logger)
    opt = _parse(argv)
    logger.info(f"weights = {em.variant()} (suffix {em.suffix()!r}), "
                f"{opt['shuffles']} shuffles, seed {SHUFFLE_SEED}, "
                f"band {BAND}, metric {opt['metric']}, "
                f"{opt['scenario']}/{opt['period']}")
    logger.info(f"model universe: {len(FULL_MODELS)} models, "
                f"{len(em.excluded_models())} excluded / "
                f"{len(em.retained_models())} retained")
    data, _ = compute(opt["scenario"], opt["period"], opt["metric"],
                      opt["shuffles"])
    plot_ratio_maps(data, opt["scenario"], opt["period"], opt["lim"])
    plot_compare_maps(data, opt["scenario"], opt["period"], opt["lim"])


if __name__ == "__main__":
    main(sys.argv[1:])
