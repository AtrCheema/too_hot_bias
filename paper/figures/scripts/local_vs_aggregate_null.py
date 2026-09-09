"""
local_vs_aggregate_null.py
==========================

Two tests answering the co-author critique of the "real in the aggregate,
invisible locally" story (comments of 7 Aug 2026), using ONE consistent
yardstick at both spatial scales. No new simulations: both tests read only the
existing Stage-1 caches
    <CONFIG.out_path>/<delta_method>/fig_data/fig02_change.parquet   (climate)
    <CONFIG.out_path>/<delta_method>/fig_data/stats_change.parquet   (flow)

TEST 1 - per-catchment random-removal null
------------------------------------------
The manuscript's per-catchment yardstick was the bias-to-spread ratio r =
|reduced - full| / sigma_model against the threshold 1. That threshold is
unreachable by construction: with p = h/N the fraction removed and D =
mean(retained) - mean(hot), the shift is p*D while sigma^2_model =
sigma^2_within + p(1-p)*D^2, so r <= sqrt(p/(1-p)) < 1 whenever fewer than
half the ensemble is removed (co-author comment 1; p = 15/32 gives a ceiling
of 0.94). The fix implemented here replaces the absolute threshold with the
SAME random-removal null already used for the global aggregate (Table S1 /
bootstrap_random_removal.py): at EVERY catchment, the observed shift
(reduced - full) is compared against the shifts produced by B removals of the
same number of models drawn at random - family-block (genealogy-aware,
PRIMARY) and uniform (reference) - and given a two-sided empirical p-value
    p_s = (#{|null shift_s| >= |observed shift_s|} + 1) / (B + 1).
The SAME B draws are applied to all catchments (one field-consistent removal
per draw), and the network-aggregate shift of each draw is accumulated
alongside, so the local and aggregate verdicts come from literally the same
null realisations. Per-catchment p-values are then controlled with
Benjamini-Hochberg across catchments at q = 0.05 and q = 0.10 (the Wilks 2016
alpha_FDR = 2*alpha convention used everywhere else in the paper), with field
significance = at least one BH rejection at q = 0.10 - mirroring
gev_fdr_dipole.fdr_field_significance. Expected story (co-author): the typical
catchment's shift sits INSIDE its null (screening moves it no further than
removing any h models would) while the aggregate shift falls OUTSIDE.

B default is 50,000 (not the 10,000 of Table S1) because BH couples the
attainable per-catchment p-resolution to the rejection threshold: with
~20,600 catchments at q = 0.10 the smallest BH threshold is ~5e-7, and an
empirical p can never go below 1/(B+1), so a larger B makes the BH step-up
less conservative. The add-one correction keeps every p-value valid at any B.

TEST 2 - aggregate bias vs aggregate internal variability
---------------------------------------------------------
Locally the paper compares the screening bias against internal variability
(pooled within-model spread; Hawkins & Sutton decomposition) and the bias
loses everywhere except temperature. The aggregate claim, however, rested on
the different random-removal yardstick (co-author comment 2). This test
applies the LOCAL comparison at the aggregate scale: the network-aggregate
change of every realization (model, member) is computed over one COMMON
complete-case station set, the aggregate bias (reduced - full, model
democracy) is compared against
    sigma_int_agg = pooled within-model std across members of the AGGREGATE
                    change (models with >= 2 members),
which automatically carries the spatial correlation of internal variability
(each member is one coherent global field - no sqrt(n_catchments) shortcut).
The between-model spread of the aggregate is reported alongside, corrected
for the internal-noise inflation of finite member counts exactly as
variance_partition() does per catchment. Both the network MEAN (the
manuscript's aggregate estimator) and the network MEDIAN (the robust Table-S1
estimator; relative flow changes blow up on near-zero-baseline catchments)
are carried throughout. The local counterpart (per-catchment |bias| /
sigma_internal, median and IQR across catchments) is computed from the SAME
complete-case matrix so the two scales differ only in the aggregation step.

Outputs
-------
CSVs (data) -> <fig_data>/
    percatchment_removal_null_<dataset>[_<engine>]_<col>_<scen>_<per>_n<N>sta.csv
    percatchment_removal_null_summary_<scen>_<per>.csv
    aggregate_internal_variability_<scen>_<per>.csv
Figures -> paper/figures/significance/local_vs_aggregate/
    percat_null_pvals_<scen>_<per>.png        ECDFs of per-catchment p-values
    chain_local_vs_aggregate_<scen>_<per>.png Test-1 summary along the chain
    aggregate_snr_<scen>_<per>.png            Test-2 aggregate vs local S/N

Run (via slurm only):
    python paper/figures/scripts/local_vs_aggregate_null.py test1
    python paper/figures/scripts/local_vs_aggregate_null.py test2
    python paper/figures/scripts/local_vs_aggregate_null.py all
    python paper/figures/scripts/local_vs_aggregate_null.py all 20000   # custom B
    python paper/figures/scripts/local_vs_aggregate_null.py fig5  # plot only:
        main-text Fig. 5 (two-scale summary) from the saved CSVs, no recompute
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))            # paper/figures/scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))  # repo root
sys.path.insert(0, _THIS)

# Reuse the Table-S1 null machinery verbatim: model universe, screen, genealogy
# families and the family-block draw. This guarantees the per-catchment null is
# THE SAME null as the aggregate one in the supplement.
import ensemble_method as em
import bootstrap_random_removal as brr
from bootstrap_random_removal import (FULL_MODELS, EXCLUDED_MODELS,
                                      RETAINED_MODELS, TOO_HOT_TCR_THRESHOLD,
                                      family_of, family_block_keep,
                                      per_station_model_means,
                                      CACHE_DIR, CLIMATE_CACHE, FLOW_CACHE)
from realization_blocklist import apply_blocklist
from flow_metrics import BY_COLUMN
from figstyle import add_panel_letters

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

FIG_DIR = os.path.join(os.path.dirname(_THIS), "significance",
                       "local_vs_aggregate")

FOCUS_SCEN = "ssp585"
FOCUS_PER  = "2081-2100"
SEED       = 42
B_DEFAULT  = 50000          # see docstring: BH resolution, not Table-S1 parity
CHUNK      = 2000           # draws per matmul chunk (memory: n_sta x CHUNK)
ALPHA_RAW  = 0.05
ALPHA_FDR  = (0.05, 0.10)   # 0.10 = Wilks (2016) alpha_FDR = 2*alpha_global

# Columns per dataset: the Table-S1 set minus flashiness (persistence index,
# excluded by the delta-method caveat; it never enters a null figure either).
CLIMATE_COLS = ["temp_change_degC", "temp_q99_change_degC",
                "precip_change_pct", "precip_q99_change_pct"]
FLOW_COLS    = ["delta_rel_mean_annual_max", "delta_rel_q95",
                "delta_rel_q99", "delta_rel_qmean", "delta_rel_q05",
                "delta_rel_flood_q20"]

# Cells added after the 31 Aug 2026 _tcr_vl run. run_test1 walks the columns
# with ONE shared generator, so inserting a cell would re-phase the draws of
# every cell after it and move published p-values by Monte-Carlo noise. These
# cells therefore take their own generator, seeded explicitly here, and are
# appended at the END of their list; the shared stream is then untouched and
# the pre-existing rows reproduce bit-identically.
LATE_CELL_SEED = {("flow", "hbv",  "delta_rel_flood_q20"): 4201,
                  ("flow", "lstm", "delta_rel_flood_q20"): 4202}

CLIMATE_LABEL = {"temp_change_degC":      ("Temp mean",   "°C"),
                 "temp_q99_change_degC":  ("Temp Q99",    "°C"),
                 "precip_change_pct":     ("Precip mean", "%"),
                 "precip_q99_change_pct": ("Precip Q99",  "%")}

def col_label(dataset, col):
    """(short label, units) for a cache column."""
    if dataset == "climate":
        return CLIMATE_LABEL[col]
    m = BY_COLUMN.get(col)
    return (m.label if m else col, m.units if m else "%")

# Chain order of the two summary figures (label, dataset, column). Flow rows
# are drawn once per engine. Q99 flow stays in the CSVs but is dropped from
# the figures (nested with AnnMax/Q95; keeps the chain readable).
CHAIN = [("Temp mean",   "climate", "temp_change_degC"),
         ("Precip mean", "climate", "precip_change_pct"),
         ("Precip Q99",  "climate", "precip_q99_change_pct"),
         ("AnnMax",      "flow",    "delta_rel_mean_annual_max"),
         ("Q95",         "flow",    "delta_rel_q95"),
         ("Mean runoff", "flow",    "delta_rel_qmean"),
         ("Q05",         "flow",    "delta_rel_q05")]

# Palette shared verbatim with the S/N cascade figure
# (analyze_full_vs_reduced_significance.CASCADE_COL_* / the axvspan bands), so
# the two chain figures read as one system: gray-blue forcing, blue HBV, red
# LSTM, pale-yellow forcing band, pale-blue impact band.
COL_FORCING  = "#5B6B73"
COL_HBV      = "#3B6FB6"
COL_LSTM     = "#C24B4B"
BAND_FORCING = "#F2C94C"      # alpha 0.10 in the cascade
BAND_IMPACT  = "#2D9CDB"      # alpha 0.08 in the cascade
BAR_ALPHA    = 0.75           # cascade bar alpha

ENGINE_COLOR = {"": COL_FORCING, "hbv": COL_HBV, "lstm": COL_LSTM}
SIG_INT_COLOR, SIG_MOD_COLOR = "#2ca02c", "#d62728"   # green/red, as supplement


def cascade_bands(ax, x_split, xlim):
    """Pale forcing / impact background bands plus the faint y-grid, exactly as
    make_snr_cascade() draws them. `x_split` is the boundary between the last
    forcing slot and the first impact slot."""
    ax.axvspan(xlim[0], x_split, color=BAND_FORCING, alpha=0.10, zorder=0)
    ax.axvspan(x_split, xlim[1], color=BAND_IMPACT,  alpha=0.08, zorder=0)
    ax.set_xlim(*xlim)
    ax.grid(axis="y", color="0.92", lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
#  shared loading
# ---------------------------------------------------------------------------
def load_focus(dataset, cols):
    """Focus-cell (scenario, period) rows of one cache, blocklisted, restricted
    to FULL_MODELS, flow rescaled to %. Columns: ids + requested metrics."""
    cache  = CLIMATE_CACHE if dataset == "climate" else FLOW_CACHE
    if not os.path.exists(cache):
        sys.exit(f"cache not found: {cache}")
    import pyarrow.parquet as pq
    schema = set(pq.ParquetFile(cache).schema_arrow.names)
    cols   = [c for c in cols if c in schema]
    if not cols:
        sys.exit(f"none of the requested columns exist in {cache}")
    idcols = ["station", "scenario", "period", "model", "member"]
    idcols += [c for c in ("engine", "lon", "lat") if c in schema]
    df = pd.read_parquet(cache, columns=idcols + cols,
                         filters=[("scenario", "=", FOCUS_SCEN),
                                  ("period",   "=", FOCUS_PER)])
    if df.empty:
        sys.exit(f"{cache}: no rows for {FOCUS_SCEN}/{FOCUS_PER} - "
                 "check the scenario/period literals")
    df = apply_blocklist(df, label=f"{dataset} local-vs-aggregate cache")
    df = df[df["model"].isin(FULL_MODELS)]
    if dataset == "flow":
        df[cols] = df[cols].astype(float) * 100.0     # fractions -> %
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"[{dataset}] {len(df):,} rows | {df['station'].nunique():,} stations "
          f"| {df['model'].nunique()} models | cols={cols}")
    return df, cols


def masked_mean(M, keep):
    """Ensemble mean per station over kept model columns (NaN-aware).
    M: (S x K) with NaN where a model is absent; keep: (K x B) 0/1."""
    val = np.where(np.isnan(M), 0.0, M)
    has = (~np.isnan(M)).astype(float)
    n   = has @ keep
    n   = np.where(n > 0, n, np.nan)
    return (val @ keep) / n


def bh_reject(p, q):
    """Benjamini-Hochberg step-up at level q. Returns (n_rejected, p_threshold).
    Mirrors gev_fdr_dipole._bh_reject."""
    p = np.sort(np.asarray(p, float)[np.isfinite(p)])
    m = len(p)
    if m == 0:
        return 0, 0.0
    k  = np.arange(1, m + 1)
    ok = p <= q * k / m
    if not ok.any():
        return 0, 0.0
    kmax = int(k[ok].max())
    return kmax, float(p[kmax - 1])


def fmt_p(p, B):
    """Honest p-string at the add-one resolution limit (as in Table S1)."""
    if p <= 1.0 / (B + 1) + 1e-12:
        return f"p < {1.0 / B:.1g}"
    if p < 1e-3:
        return f"p = {p:.1g}"
    return f"p = {p:.3f}"


# ---------------------------------------------------------------------------
#  TEST 1 - per-catchment random-removal null
# ---------------------------------------------------------------------------
def percatchment_null_cell(mm, rng, B):
    """One (engine, column) cell: per-catchment two-sided empirical p-values
    under the family-block and uniform removal nulls, plus the aggregate
    (network mean / median) verdicts from the SAME draws.

    mm : (station x model) DataFrame of member-mean changes (model democracy).
    Returns None if the screen split is degenerate."""
    models = list(mm.columns)
    K      = len(models)
    fam    = [family_of(m) for m in models]
    weighting = em.is_weighting()

    if weighting:
        # No hot/retained split exists. The treatment column is the frozen
        # weight vector and the null PERMUTES it: masked_mean computes
        # (val @ keep) / (has @ keep), so a general non-negative column is the
        # weighted mean renormalised per station over the models present there.
        # The screen is exactly the 0/1 limit of this, which is why one code
        # path serves both treatments.
        if K < 4:
            return None
        h = 0
        w = em.weight_array(models)
        keep_obs = w.reshape(K, 1).copy()
    else:
        hot_ix = np.array([i for i, m in enumerate(models)
                           if m in EXCLUDED_MODELS])
        h      = len(hot_ix)
        if h < 2 or K - h < 2:
            return None
        keep_obs = np.ones((K, 1)); keep_obs[hot_ix, 0] = 0.0

    M   = mm.to_numpy(float)                          # (S x K)
    S   = M.shape[0]

    keep_full = np.ones((K, 1))
    full_mean = masked_mean(M, keep_full)[:, 0]       # (S,)
    # reduced - full under removal, weighted - full under weighting
    obs_shift = masked_mean(M, keep_obs)[:, 0] - full_mean
    abs_obs   = np.abs(obs_shift)

    # aggregate observed statistics on the SAME station set
    obs_agg_med  = float(np.nanmedian(obs_shift))
    obs_agg_mean = float(np.nanmean(obs_shift))

    n_distinct = np.nan
    if weighting:
        # Reuse the AGGREGATE table's own permutation constructors, so the
        # local and aggregate nulls are the same construction by definition.
        from weight_permutation_null import (uniform_permute,
                                             family_block_permute)
        keep_fam, n_distinct = family_block_permute(fam, w, B, rng)
        keep_unif = uniform_permute(w, B, rng)
    else:
        # the two removal nulls; each column of keep removes exactly h models
        keep_fam  = family_block_keep(fam, h, B, rng)
        keep_unif = np.ones((K, B))
        for b in range(B):
            keep_unif[rng.choice(K, size=h, replace=False), b] = 0.0

    out = {}
    for tag, keep in (("fam", keep_fam), ("unif", keep_unif)):
        exceed   = np.zeros(S, dtype=np.int64)        # per catchment
        valid    = np.zeros(S, dtype=np.int64)        # draws with a defined shift
        agg_med  = np.empty(B)
        agg_mean = np.empty(B)
        for lo in range(0, B, CHUNK):
            hi = min(lo + CHUNK, B)
            null_shift = masked_mean(M, keep[:, lo:hi]) - full_mean[:, None]
            # A draw whose removal set covers EVERY model present at a sparse
            # station has no defined shift there (NaN). Count exceedances only
            # over defined draws, and divide by the per-station defined count,
            # otherwise those NaNs would silently deflate the p-value.
            finite  = np.isfinite(null_shift)
            valid  += finite.sum(axis=1).astype(np.int64)
            exceed += np.nansum(np.abs(null_shift) >= abs_obs[:, None],
                                axis=1).astype(np.int64)
            # aggregate: identical draws, aggregated over the network
            agg_med[lo:hi]  = np.nanmedian(null_shift, axis=0)
            agg_mean[lo:hi] = np.nanmean(null_shift, axis=0)
        p_cat = (exceed + 1) / (valid + 1)            # add-one correction
        # a station with < 90 % defined draws gets no p (too coarse to trust)
        p_cat = np.where(valid >= 0.9 * B, p_cat, np.nan)
        p_agg_med  = (int(np.sum(np.abs(agg_med)  >= abs(obs_agg_med)))  + 1) / (B + 1)
        p_agg_mean = (int(np.sum(np.abs(agg_mean) >= abs(obs_agg_mean))) + 1) / (B + 1)
        out[tag] = {"p_cat": p_cat, "p_agg_med": p_agg_med,
                    "p_agg_mean": p_agg_mean,
                    "null_agg_med": agg_med, "null_agg_mean": agg_mean}

    return {"stations": mm.index.to_numpy(), "n_sta": S, "n_models": K,
            "h_removed": h, "B": B, "obs_shift": obs_shift,
            "obs_agg_med": obs_agg_med, "obs_agg_mean": obs_agg_mean,
            "null_kind": "weight_permutation" if weighting else "random_removal",
            "n_eff": em.n_eff_present(models) if weighting else float(K - h),
            "n_perm_distinct": n_distinct,
            "fam": out["fam"], "unif": out["unif"]}


def run_test1(B):
    rng  = np.random.default_rng(SEED)
    os.makedirs(FIG_DIR, exist_ok=True)
    summary, panels = [], []

    for dataset, cols_req in (("climate", CLIMATE_COLS), ("flow", FLOW_COLS)):
        df, cols = load_focus(dataset, cols_req)
        engines = sorted(df["engine"].unique()) if "engine" in df.columns else [""]
        for eng in engines:
            sub_e = df[df["engine"] == eng] if eng else df
            for col in cols:
                mm = per_station_model_means(sub_e, col)
                if mm.empty:
                    print(f"  [{dataset}/{eng or '-'}] {col}: no eligible "
                          "stations, skipped")
                    continue
                seed_late = LATE_CELL_SEED.get((dataset, eng, col))
                cell_rng = (np.random.default_rng(seed_late)
                            if seed_late is not None else rng)
                res = percatchment_null_cell(mm, cell_rng, B)
                if res is None:
                    continue
                label, units = col_label(dataset, col)

                # ---- per-catchment CSV --------------------------------------
                cat = pd.DataFrame({"station": res["stations"],
                                    "shift_red_minus_full": res["obs_shift"],
                                    "p_fam":  res["fam"]["p_cat"],
                                    "p_unif": res["unif"]["p_cat"]})
                if {"lon", "lat"} <= set(sub_e.columns):
                    coords = (sub_e.groupby("station")[["lon", "lat"]]
                              .first().reset_index())
                    cat = cat.merge(coords, on="station", how="left")
                tag = f"{dataset}{'_' + eng if eng else ''}_{col}"
                out_csv = em.tag(os.path.join(
                    CACHE_DIR, f"percatchment_removal_null_{tag}_"
                               f"{FOCUS_SCEN}_{FOCUS_PER}_n{res['n_sta']}sta.csv"))
                cat.to_csv(out_csv, index=False)

                # ---- summary row: local fractions + aggregate verdicts ------
                row = {"dataset": dataset, "engine": eng or "-", "column": col,
                       "label": label, "units": units,
                       "n_sta": res["n_sta"], "n_models": res["n_models"],
                       "h_removed": res["h_removed"], "B": B,
                       # what the null actually IS in this run: the file stem
                       # says "removal" for backward compatibility with
                       # screen_effect_maps.py, so the kind is recorded here.
                       "null_kind": res["null_kind"],
                       "treatment": em.suffix() or "removal_tcr_hot",
                       "n_eff": res["n_eff"],
                       "n_perm_distinct": res["n_perm_distinct"],
                       "obs_agg_median": res["obs_agg_med"],
                       "obs_agg_mean":   res["obs_agg_mean"]}
                for tag2 in ("fam", "unif"):
                    p = res[tag2]["p_cat"]
                    pf = p[np.isfinite(p)]            # stations with a valid p
                    row[f"n_sta_valid_{tag2}"] = int(len(pf))
                    row[f"frac_raw_{tag2}"] = float(np.mean(pf < ALPHA_RAW))
                    row[f"p_cat_median_{tag2}"] = float(np.median(pf))
                    for q in ALPHA_FDR:
                        n_rej, thr = bh_reject(pf, q)
                        row[f"frac_BH_q{q:.2f}_{tag2}"] = n_rej / max(len(pf), 1)
                        row[f"p_thresh_BH_q{q:.2f}_{tag2}"] = thr
                    row[f"field_significant_{tag2}"] = \
                        bool(row[f"frac_BH_q{ALPHA_FDR[1]:.2f}_{tag2}"] > 0)
                    row[f"p_agg_median_{tag2}"] = res[tag2]["p_agg_med"]
                    row[f"p_agg_mean_{tag2}"]   = res[tag2]["p_agg_mean"]
                summary.append(row)
                panels.append({"row": row, "p_fam": res["fam"]["p_cat"],
                               "p_unif": res["unif"]["p_cat"]})
                print(f"  [{dataset}/{eng or '-'}] {label}: "
                      f"local raw {row['frac_raw_fam']:.1%}, "
                      f"BH q0.10 {row['frac_BH_q0.10_fam']:.2%}, "
                      f"aggregate(median) {fmt_p(row['p_agg_median_fam'], B)} | "
                      f"n={res['n_sta']:,}")

    summ = pd.DataFrame(summary)
    out_csv = em.tag(os.path.join(CACHE_DIR, "percatchment_removal_null_summary_"
                                             f"{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    summ.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(summ)} cells)")
    figure_pval_ecdfs(panels)
    figure_chain_summary(summ)


def figure_pval_ecdfs(panels):
    """ECDFs of the per-catchment empirical p-values, one panel per cell. A
    curve on the 1:1 diagonal = the screen is indistinguishable from a random
    removal at every catchment (p uniform); mass near p=0 = locally special."""
    # keep the figure to the chain cells, in chain order
    keyed = {(d["row"]["dataset"], d["row"]["engine"], d["row"]["column"]): d
             for d in panels}
    order = []
    for lab, ds, col in CHAIN:
        if ds == "climate":
            order.append((lab, keyed.get((ds, "-", col))))
        else:
            for eng in ("hbv", "lstm"):
                d = keyed.get((ds, eng, col))
                order.append((f"{lab} ({eng.upper()})", d))
    order = [(lab, d) for lab, d in order if d is not None]
    n = len(order)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, (lab, d) in zip(axes.ravel(), order):
        row = d["row"]
        for p, color, lw, name in ((d["p_fam"], "firebrick", 1.8,
                                    "Family-block null"),
                                   (d["p_unif"], "0.45", 1.0, "Uniform null")):
            ps = np.sort(p[np.isfinite(p)])
            ax.plot(ps, np.arange(1, len(ps) + 1) / len(ps),
                    color=color, lw=lw, label=name)
        ax.plot([0, 1], [0, 1], color="0.75", lw=0.8, ls="--",
                label="Uniform p (indistinguishable)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(lab, fontsize=11)
        ax.text(0.97, 0.08,
                f"Raw {row['frac_raw_fam']:.0%}\n"
                f"BH q0.10 {row['frac_BH_q0.10_fam']:.1%}\n"
                f"Agg {fmt_p(row['p_agg_median_fam'], row['B'])}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
    for r in range(nrow):
        axes[r][0].set_ylabel("Fraction of catchments")
    for c in range(ncol):
        axes[-1][c].set_xlabel("Per-catchment empirical p")
    axes[0][0].legend(fontsize=8, loc="upper left", frameon=False)
    add_panel_letters(axes)
    fig.tight_layout()
    out = em.tag(os.path.join(
        FIG_DIR, f"percat_null_pvals_{FOCUS_SCEN}_{FOCUS_PER}.png"))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_chain_summary(summ):
    """Test-1 story figure: along the T -> P -> Q chain, the percentage of
    catchments where the screen is distinguishable from a random removal
    (BH q=0.10, family-block null), with the aggregate verdict of the SAME
    null draws annotated above each stage."""
    stages, hbv_vals, lstm_vals, clim_vals, agg_notes = [], [], [], [], []
    B = int(summ["B"].iloc[0])
    for lab, ds, col in CHAIN:
        if ds == "flow" and col == "delta_rel_q99":
            continue
        stages.append(lab)
        if ds == "climate":
            r = summ[(summ.dataset == "climate") & (summ.column == col)]
            clim_vals.append(100 * float(r["frac_BH_q0.10_fam"].iloc[0])
                             if len(r) else np.nan)
            hbv_vals.append(np.nan); lstm_vals.append(np.nan)
            agg_notes.append(fmt_p(float(r["p_agg_median_fam"].iloc[0]), B)
                             if len(r) else "")
        else:
            note = []
            for eng, dest in (("hbv", hbv_vals), ("lstm", lstm_vals)):
                r = summ[(summ.dataset == "flow") & (summ.engine == eng)
                         & (summ.column == col)]
                dest.append(100 * float(r["frac_BH_q0.10_fam"].iloc[0])
                            if len(r) else np.nan)
                if len(r):
                    note.append(f"{eng.upper()} "
                                f"{fmt_p(float(r['p_agg_median_fam'].iloc[0]), B)}")
            clim_vals.append(np.nan)
            agg_notes.append("\n".join(note))

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    w = 0.34
    ax.bar(x, clim_vals, width=0.55, color="0.55", label="Climate forcing")
    ax.bar(x - w / 2, hbv_vals, width=w, color=ENGINE_COLOR["hbv"], label="HBV")
    ax.bar(x + w / 2, lstm_vals, width=w, color=ENGINE_COLOR["lstm"],
           label="LSTM")
    top = np.nanmax(np.vstack([clim_vals, hbv_vals, lstm_vals])) if stages else 1
    for xi, note in zip(x, agg_notes):
        ax.text(xi, top * 1.06, note, ha="center", va="bottom", fontsize=8.5,
                color="0.15")
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Catchments where screen differs from\n"
                  "random removal (%, BH q=0.10)", fontsize=11)
    ax.set_ylim(0, top * 1.35 if np.isfinite(top) and top > 0 else 1)
    ax.set_title(f"Same null, two scales ({FOCUS_SCEN.upper()}, {FOCUS_PER}): "
                 "per-catchment fractions (bars) and network-aggregate "
                 "verdict (text)", fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    out = em.tag(os.path.join(
        FIG_DIR, f"chain_local_vs_aggregate_{FOCUS_SCEN}_{FOCUS_PER}.png"))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
#  TEST 2 - aggregate bias vs aggregate internal variability
# ---------------------------------------------------------------------------
MIN_REALIZATION_COVERAGE = 0.99   # of the best-covered realization's stations


def complete_matrix(sub, col):
    """(station x realization) complete-case matrix for one cell.

    Realizations covering < MIN_REALIZATION_COVERAGE of the best-covered
    realization's station count are dropped first (they would shrink the
    common station set), then stations with any remaining NaN are dropped, so
    EVERY kept realization aggregates the IDENTICAL station set. Returns
    (matrix DataFrame, model list per column)."""
    piv = (sub.dropna(subset=[col])
              .pivot_table(index="station", columns=["model", "member"],
                           values=col, aggfunc="mean", observed=True))
    if piv.empty:
        return piv, []
    cov = piv.notna().sum(axis=0)
    keep_r = cov >= MIN_REALIZATION_COVERAGE * cov.max()
    n_drop_r = int((~keep_r).sum())
    piv = piv.loc[:, keep_r]
    before = len(piv)
    piv = piv.dropna(axis=0, how="any")
    if n_drop_r or before - len(piv):
        print(f"    coverage control: dropped {n_drop_r} realizations, "
              f"{before - len(piv):,} stations -> {len(piv):,} stations x "
              f"{piv.shape[1]} realizations")
    models = [c[0] for c in piv.columns]
    return piv, models


def aggregate_cell(sub, col):
    """One (engine, column) cell of Test 2. Returns a dict of aggregate and
    local statistics, or None if degenerate."""
    piv, models = complete_matrix(sub, col)
    if piv.empty or len(set(models)) < 4:
        return None
    R      = piv.to_numpy(float)                     # (S x n_realizations)
    m_arr  = np.asarray(models, dtype=object)
    uniq   = list(dict.fromkeys(models))
    hot    = [m for m in uniq if m in EXCLUDED_MODELS]
    ret    = [m for m in uniq if m not in EXCLUDED_MODELS]
    if len(hot) < 2 or len(ret) < 2:
        return None

    out = {"n_sta": R.shape[0], "n_realizations": R.shape[1],
           "n_models": len(uniq), "h_removed": len(hot)}

    # ---- LOCAL statistics from the SAME matrix (per catchment, then median) --
    # per-catchment per-model member means -> model-democracy bias; pooled
    # within-model variance (>=2 members) = internal variability (Hawkins-Sutton)
    mem_cols = {m: np.where(m_arr == m)[0] for m in uniq}
    Mmod = np.column_stack([R[:, ix].mean(axis=1) for m, ix in mem_cols.items()])
    ret_ix = np.array([i for i, m in enumerate(uniq) if m in ret])
    bias_local = Mmod[:, ret_ix].mean(axis=1) - Mmod.mean(axis=1)   # reduced - full
    ss = np.zeros(R.shape[0]); dof = 0
    for m, ix in mem_cols.items():
        if len(ix) >= 2:
            ss  += R[:, ix].var(axis=1, ddof=1) * (len(ix) - 1)
            dof += len(ix) - 1
    sig_int_local = np.sqrt(ss / dof) if dof > 0 else np.full(R.shape[0], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_local = np.abs(bias_local) / sig_int_local
    ratio_local = ratio_local[np.isfinite(ratio_local)]
    if len(ratio_local) == 0:
        return None
    out.update(local_ratio_median=float(np.median(ratio_local)),
               local_ratio_q25=float(np.percentile(ratio_local, 25)),
               local_ratio_q75=float(np.percentile(ratio_local, 75)),
               dof_internal=int(dof))

    # ---- AGGREGATE statistics: network mean and network median --------------
    for agg_name, agg_fn in (("mean", np.mean), ("median", np.median)):
        a = agg_fn(R, axis=0)                        # (n_realizations,)
        A = {m: a[ix].mean() for m, ix in mem_cols.items()}   # per-model agg
        full_agg = float(np.mean([A[m] for m in uniq]))
        red_agg  = float(np.mean([A[m] for m in ret]))
        bias_agg = red_agg - full_agg                # reduced - full
        # pooled within-model variance of the aggregate series
        ss_a, dof_a, counts = 0.0, 0, []
        for m, ix in mem_cols.items():
            counts.append(len(ix))
            if len(ix) >= 2:
                ss_a  += a[ix].var(ddof=1) * (len(ix) - 1)
                dof_a += len(ix) - 1
        s2_int = ss_a / dof_a if dof_a > 0 else np.nan
        # between-model spread of the aggregate, internal-noise corrected
        # (mirrors variance_partition in the significance script)
        s2_between = float(np.var([A[m] for m in uniq], ddof=1))
        if np.isfinite(s2_int):
            s2_mod = max(0.0, s2_between
                         - s2_int * float(np.mean([1.0 / c for c in counts])))
        else:
            s2_mod = s2_between
        sig_int, sig_mod = np.sqrt(s2_int), np.sqrt(s2_mod)
        ratio_int = abs(bias_agg) / sig_int if sig_int > 0 else np.nan
        # chi-square 95% CI on sigma_int -> CI on the ratio
        if HAVE_SCIPY and dof_a > 0 and sig_int > 0:
            lo = np.sqrt(s2_int * dof_a / sps.chi2.ppf(0.975, dof_a))
            hi = np.sqrt(s2_int * dof_a / sps.chi2.ppf(0.025, dof_a))
            ci = (abs(bias_agg) / hi, abs(bias_agg) / lo)
        else:
            ci = (np.nan, np.nan)
        out.update({f"agg_{agg_name}_bias": bias_agg,
                    f"agg_{agg_name}_sigma_int": float(sig_int),
                    f"agg_{agg_name}_sigma_model": float(sig_mod),
                    f"agg_{agg_name}_ratio_int": float(ratio_int),
                    f"agg_{agg_name}_ratio_int_lo": float(ci[0]),
                    f"agg_{agg_name}_ratio_int_hi": float(ci[1]),
                    f"agg_{agg_name}_dof_internal": int(dof_a)})
    return out


def run_test2():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []
    for dataset, cols_req in (("climate", CLIMATE_COLS), ("flow", FLOW_COLS)):
        df, cols = load_focus(dataset, cols_req)
        engines = sorted(df["engine"].unique()) if "engine" in df.columns else [""]
        for eng in engines:
            sub_e = df[df["engine"] == eng] if eng else df
            for col in cols:
                res = aggregate_cell(sub_e[["station", "model", "member", col]],
                                     col)
                if res is None:
                    print(f"  [{dataset}/{eng or '-'}] {col}: degenerate cell, "
                          "skipped")
                    continue
                label, units = col_label(dataset, col)
                row = {"dataset": dataset, "engine": eng or "-", "column": col,
                       "label": label, "units": units, **res}
                rows.append(row)
                print(f"  [{dataset}/{eng or '-'}] {label}: "
                      f"agg bias(median est.) {res['agg_median_bias']:+.3f} {units}, "
                      f"sigma_int {res['agg_median_sigma_int']:.3f}, "
                      f"ratio {res['agg_median_ratio_int']:.1f}x "
                      f"(local median {res['local_ratio_median']:.2f}x) | "
                      f"n={res['n_sta']:,} sta, {res['n_realizations']} realiz.")
    out = pd.DataFrame(rows)
    out_csv = em.tag(os.path.join(CACHE_DIR, "aggregate_internal_variability_"
                                             f"{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(out)} cells)")
    figure_aggregate_snr(out)


def figure_aggregate_snr(out):
    """Test-2 figure. Panels a-c: aggregate |bias| against the aggregate
    internal-variability and (noise-corrected) model spread, per variable
    group (absolute units, log scale). Panel d: the local-vs-aggregate
    bias-to-internal-variability ratio along the chain (dimensionless; the
    ratio = 1 line is meaningful here - unlike the model-spread ratio, this
    denominator carries no algebraic ceiling)."""
    fig = plt.figure(figsize=(12.6, 7.6))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42,
                           wspace=0.30)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[0, 1])
    ax_f = fig.add_subplot(gs[0, 2])
    ax_r = fig.add_subplot(gs[1, :])

    def bars(ax, sel, title, unit):
        sel = sel.reset_index(drop=True)
        x = np.arange(len(sel))
        w = 0.30
        ax.bar(x - w, sel["agg_median_sigma_int"], width=w,
               color=SIG_INT_COLOR, label="Aggregate $\\sigma$ internal")
        ax.bar(x, sel["agg_median_sigma_model"], width=w,
               color=SIG_MOD_COLOR, label="Aggregate $\\sigma$ model")
        ax.bar(x + w, sel["agg_median_bias"].abs(), width=w,
               color="0.25", label="Aggregate |bias|")
        ax.set_yscale("log")
        labs = [f"{r.label}" + (f"\n({r.engine.upper()})"
                                if r.engine != "-" else "")
                for r in sel.itertuples()]
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
        ax.set_ylabel(f"Network-median aggregate ({unit})", fontsize=10)
        ax.set_title(title, fontsize=11)

    clim = out[out.dataset == "climate"]
    bars(ax_t, clim[clim.units == "°C"], "Temperature", "°C")
    bars(ax_p, clim[clim.units == "%"], "Precipitation", "%")
    flow_rows = out[(out.dataset == "flow")
                    & (out.column.isin(["delta_rel_mean_annual_max",
                                        "delta_rel_qmean"]))]
    bars(ax_f, flow_rows, "Streamflow", "%")
    ax_t.legend(fontsize=8.5, frameon=False, loc="upper right")

    # ---- panel d: local vs aggregate ratio along the chain ------------------
    xt, xl = [], []
    xpos = 0
    for lab, ds, col in CHAIN:
        cells = (out[(out.dataset == ds) & (out.column == col)]
                 .sort_values("engine"))
        for r in cells.itertuples():
            color = ENGINE_COLOR.get(r.engine if r.engine != "-" else "", "0.3")
            # local: median per-catchment ratio with IQR
            ax_r.errorbar([xpos - 0.12], [r.local_ratio_median],
                          yerr=[[r.local_ratio_median - r.local_ratio_q25],
                                [r.local_ratio_q75 - r.local_ratio_median]],
                          fmt="o", color=color, ms=5, capsize=2, lw=1.1)
            # aggregate: ratio with chi-square CI, larger open marker
            ax_r.errorbar([xpos + 0.12], [r.agg_median_ratio_int],
                          yerr=[[max(0.0, r.agg_median_ratio_int
                                     - r.agg_median_ratio_int_lo)],
                                [max(0.0, r.agg_median_ratio_int_hi
                                     - r.agg_median_ratio_int)]],
                          fmt="D", mfc="none", color=color, ms=7, capsize=2,
                          lw=1.3)
            ax_r.plot([xpos - 0.12, xpos + 0.12],
                      [r.local_ratio_median, r.agg_median_ratio_int],
                      color=color, lw=0.8, alpha=0.6)
            xt.append(xpos)
            xl.append(lab if r.engine == "-" else f"{lab}\n({r.engine.upper()})")
            xpos += 1
    ax_r.axhline(1.0, color="0.4", lw=1.0, ls="--")
    ax_r.text(xpos - 0.45, 1.05, "Bias = internal variability", fontsize=9,
              ha="right", va="bottom", color="0.3")
    ax_r.set_yscale("log")
    ax_r.set_xticks(xt); ax_r.set_xticklabels(xl, fontsize=9.5)
    ax_r.set_ylabel("|bias| / $\\sigma$ internal", fontsize=11)
    ax_r.set_title("Same comparison, two scales: per catchment "
                   "(filled, median and IQR) vs network aggregate "
                   "(open, 95% CI)", fontsize=11)
    from matplotlib.lines import Line2D
    ax_r.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="0.3",
               label="Per catchment (median, IQR)"),
        Line2D([], [], marker="D", mfc="none", ls="none", color="0.3",
               label="Network aggregate (95% CI)"),
        Line2D([], [], marker="s", ls="none", color=ENGINE_COLOR["hbv"],
               label="HBV"),
        Line2D([], [], marker="s", ls="none", color=ENGINE_COLOR["lstm"],
               label="LSTM")], fontsize=9, frameon=False, ncol=4,
        loc="upper left")
    add_panel_letters([ax_t, ax_p, ax_f, ax_r])
    out_png = em.tag(os.path.join(
        FIG_DIR, f"aggregate_snr_{FOCUS_SCEN}_{FOCUS_PER}.png"))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"  wrote {out_png}")


# ---------------------------------------------------------------------------
#  FIG. 5 - main-text figure assembled from the saved CSVs (no recompute)
# ---------------------------------------------------------------------------
def figure_fig5():
    """Main-text Fig. 5 from the Test-2 CSV: a SINGLE panel showing
    |bias| / sigma_internal per catchment (filled, median and IQR) and for the
    across-catchment aggregate (open, median estimator with the chi-square 95%
    CI), along the T -> P -> Q chain. The ratio = 1 line is meaningful for this
    denominator (no algebraic ceiling).

    The former panel b (percentage of catchments where the too-hot and
    retained groups differ, two-sided Welch raw p < 0.05) was dropped on
    2026-08-09; that content lives in the S/N cascade figure. The
    per-catchment removal null (Test 1) is supplement TABLE material
    (tab:si_localnull), not a panel. Being single-panel, the figure carries no
    panel letter (figstyle house rule).
    """
    # em.tag() MUST match figure_aggregate_snr's tagged writer: without it a
    # two-sided run draws main-text Fig. 5 from the untagged hot-only cells
    t2_csv = em.tag(os.path.join(CACHE_DIR, "aggregate_internal_variability_"
                                            f"{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    if not os.path.exists(t2_csv):
        sys.exit(f"missing {t2_csv}")
    aggv = pd.read_csv(t2_csv)
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax_a = plt.subplots(figsize=(12.6, 5.4))

    # Old-Fig-5 stage set (exact continuity with snr_cascade): both
    # temperature and both precipitation stages in the old order, and the
    # four flow indices without Q05 (Q05 stays in the SI tables).
    fig5_chain = [("Temp mean",   "climate", "temp_change_degC"),
                  ("Temp Q99",    "climate", "temp_q99_change_degC"),
                  ("Precip Q99",  "climate", "precip_q99_change_pct"),
                  ("Precip mean", "climate", "precip_change_pct"),
                  ("AnnMax",      "flow",    "delta_rel_mean_annual_max"),
                  ("Q99",         "flow",    "delta_rel_q99"),
                  ("Q95",         "flow",    "delta_rel_q95"),
                  ("Mean runoff", "flow",    "delta_rel_qmean")]

    # ---- bias-to-internal-variability ratio at two scales -----------------
    xt, xl = [], []
    xpos = 0
    x_split_a = None            # boundary between the forcing and impact bands
    for lab, ds, col in fig5_chain:
        if ds == "flow" and x_split_a is None:
            x_split_a = xpos - 0.5
        cells = (aggv[(aggv.dataset == ds) & (aggv.column == col)]
                 .sort_values("engine"))
        for r in cells.itertuples():
            color = ENGINE_COLOR.get(r.engine if r.engine != "-" else "",
                                     COL_FORCING)
            ax_a.errorbar([xpos - 0.12], [r.local_ratio_median],
                          yerr=[[r.local_ratio_median - r.local_ratio_q25],
                                [r.local_ratio_q75 - r.local_ratio_median]],
                          fmt="o", color=color, ms=6, capsize=2, lw=1.2)
            ax_a.errorbar([xpos + 0.12], [r.agg_median_ratio_int],
                          yerr=[[max(0.0, r.agg_median_ratio_int
                                     - r.agg_median_ratio_int_lo)],
                                [max(0.0, r.agg_median_ratio_int_hi
                                     - r.agg_median_ratio_int)]],
                          fmt="D", mfc="none", color=color, ms=8, capsize=2,
                          lw=1.4)
            ax_a.plot([xpos - 0.12, xpos + 0.12],
                      [r.local_ratio_median, r.agg_median_ratio_int],
                      color=color, lw=0.9, alpha=0.6)
            xt.append(xpos)
            # multi-line labels throughout so adjacent ticks cannot collide
            if r.engine == "-":
                xl.append(lab.replace(" ", "\n", 1))
            else:
                xl.append(lab.replace(" ", "\n", 1) + f"\n({r.engine.upper()})")
            xpos += 1
    ax_a.axhline(1.0, color="0.25", lw=1.1, ls="--")
    ax_a.set_yscale("log")
    # log spacing (values span 0.09-5.6) but readable scalar ticks
    from matplotlib import ticker
    ax_a.set_ylim(0.07, 7)
    ax_a.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.2, 0.5, 1, 2, 5]))
    ax_a.yaxis.set_major_formatter(ticker.FixedFormatter(
        ["0.1", "0.2", "0.5", "1", "2", "5"]))
    ax_a.yaxis.set_minor_locator(ticker.NullLocator())
    ax_a.set_xticks(xt)
    ax_a.set_xticklabels(xl, fontsize=11.5)
    ax_a.tick_params(axis="y", labelsize=11)
    ax_a.set_ylabel("|bias| / $\\sigma$ internal", fontsize=13)
    ax_a.set_title("The correction against internal variability at two scales",
                   fontsize=12.5)
    cascade_bands(ax_a, x_split_a if x_split_a is not None else xpos - 0.5,
                  (-0.6, xpos - 0.4))
    ax_a.text(xpos - 0.5, 1.03, "Bias = internal variability", fontsize=11,
              ha="right", va="bottom", color="0.25")
    from matplotlib.lines import Line2D
    ax_a.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="0.35",
               label="Per catchment (median, IQR)"),
        Line2D([], [], marker="D", mfc="none", ls="none", color="0.35",
               label="Across-catchment aggregate (95% CI)"),
        Line2D([], [], marker="s", ls="none", color=ENGINE_COLOR["hbv"],
               label="HBV"),
        Line2D([], [], marker="s", ls="none", color=ENGINE_COLOR["lstm"],
               label="LSTM")], fontsize=11, ncol=2, loc="upper right",
        framealpha=0.9, edgecolor="none")

    # single panel: no panel letter (add_panel_letters is a no-op below two
    # panels; the call is dropped so the intent is explicit)
    out = em.tag(os.path.join(
        FIG_DIR, f"fig05_two_scales_{FOCUS_SCEN}_{FOCUS_PER}.png"))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    em.reject_cli_tokens(sys.argv[1:])
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment()
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    what = argv[0].lower() if argv else "all"
    B    = int(argv[1]) if len(argv) > 1 else B_DEFAULT
    print(f"FULL={len(FULL_MODELS)}  EXCLUDED={len(EXCLUDED_MODELS)} "
          f"({em.screen_expr()})  RETAINED={len(RETAINED_MODELS)}  "
          f"B={B}  focus={FOCUS_SCEN}/{FOCUS_PER}")
    if what in ("test1", "all"):
        kind = ("weight-permutation" if em.is_weighting() else "random-removal")
        print(f"\n=== TEST 1: per-catchment {kind} null ===")
        run_test1(B)
    if what in ("test2", "all"):
        if em.is_weighting():
            # aggregate_cell() splits the realizations into a too-hot and a
            # retained group, which does not exist under weighting. Converting
            # it is a separate job; skip rather than write a removal contrast
            # under a weighting tag.
            print("\n=== TEST 2: SKIPPED under METHOD='weighting' ===")
            print("aggregate_cell() contrasts a too-hot against a retained "
                  "group; no such split exists for a weighted ensemble. "
                  "Table S3 comes from TEST 1, which is converted.")
        else:
            print("\n=== TEST 2: aggregate bias vs aggregate internal "
                  "variability ===")
            run_test2()
    if what == "fig5":
        print("\n=== FIG 5: two-scale summary from saved CSVs ===")
        figure_fig5()
    print("\ndone.")
