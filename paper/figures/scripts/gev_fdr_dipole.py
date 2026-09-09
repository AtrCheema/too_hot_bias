"""
gev_fdr_dipole.py
=================

The three review-gate analyses for the too-hot-models paper, in ONE file with
one function per step (merged from four single-purpose scripts, 5 Jul 2026).
Everything runs from EXISTING caches — no new GEV fits, no delta/hydrology
recompute. Steps and their data flow:

  baseline_cache  Tiny stage-1 cache of the BASELINE (observed-forcing) GEV
                  flood return levels flood_q10/q20/q50 (mm/day) per station
                  and engine, read column-selectively from each station's
                  stats.parquet. Needed because build_streamflow_cache.py keeps
                  only future delta_* rows. The GEV fits themselves were made
                  by backfill_streamflow_stats.py (flood_return_levels():
                  L-moment GEV on annual block maxima; Coles 2001; Katz,
                  Parlange & Naveau 2002).
                  -> fig_data/flood_rl_baseline.parquet

  gev             Design-flood (return-period) translation of the too-hot
                  peak-flow bias: members -> model -> ensemble means (one
                  model one vote, as everywhere else) of the cached
                  delta_rel_flood_q{20,50}; absolute future return levels via
                  Q_T,ens = Q_T,base * (1 + delta); then each ensemble's
                  return-level curve interpolated linearly in the Gumbel
                  reduced variate y(T) = -ln(-ln(1-1/T)) through its two
                  fitted points (T=20, Q20), (T=50, Q50) (the straight-line
                  EV1 representation on Gumbel paper; Coles 2001 Sect. 3.1.3)
                  to give T*: the return period, on the SCREENED ensemble's
                  curve, of the RAW ensemble's future 20-yr flood. With
                  ~20-year windows Q20 is the defensible headline level (Q100
                  is deliberately never fitted); T* off the [20, 50] anchor
                  pair is a mild extrapolation along that line — state it.
                  Q10 is carried too, as the third FITTED anchor of the Fig. 4a
                  curve; it is joined left and never gates the catchment set,
                  so no headline RL20 number moves.
                  -> fig_data/gev_translation_*.csv,
                     hydro_indices/summary/gev_translation_*.png

  fdr             Multiple-testing / field-significance control: Benjamini-
                  Hochberg FDR (Benjamini & Hochberg 1995, JRSS-B 57, 289) on
                  the per-catchment p-values of every cached (metric, engine)
                  cell, at alpha_FDR = 0.05 and at 0.10 — the Wilks (2016,
                  BAMS 97, 2263) rule alpha_FDR = 2*alpha_global for
                  positively spatially correlated fields; field significance
                  = >=1 BH rejection at the Wilks level (Wilks 2006, JAMC 45,
                  1181). Reads ONLY the signif_*.csv focus files written by
                  analyze_full_vs_reduced_significance.py (disjoint hot-vs-
                  retained Welch test on one-value-per-model means).
                  -> fig_data/fdr_field_significance_*.csv

  dipole          Quantifies the nival/pluvial sign dipole of the peak-flow
                  bias (report.tex P18b / hydro_indices finding 11). Primary
                  inference is CONTINUOUS: Spearman of the per-catchment bias
                  (full - reduced, delta_rel_mean_annual_max, %) vs snowfall
                  fraction, plus a PARTIAL Spearman controlling |latitude|
                  (rank-residual method, identical to the transmission scan);
                  decile-binned medians and fraction-negative with Wilson 95%
                  score intervals (Wilson 1927; Brown, Cai & DasGupta 2001).
                  Descriptive supplements: rain-dominated (snowfrac <= 0.10)
                  vs snow-dominated (>= 0.50) Mann-Whitney U (snow-fraction
                  regime classification after Berghuijs et al. 2014, Nat.
                  Clim. Change 4, 583) and a Koeppen A-E breakdown; a
                  significant-only sensitivity repeats the tests.
                  -> fig_data/dipole_peakflow_*.csv,
                     hydro_indices/summary/dipole_peakflow_snowfrac_*.png

Prerequisites: `gev` needs `baseline_cache` first; `fdr`/`dipole` need the
focus CSVs (run analyze_full_vs_reduced_significance.py 'all'/'flow' and, for
the return-level cells, on delta_rel_flood_q20 / delta_rel_flood_q50).

  gev_map         MAIN-FIGURE composite (narrative4 Fig 3): global maps of
                  the per-catchment 20-yr return-level bias (full − reduced,
                  pp) for HBV and LSTM + the T* distribution. Reads the
                  per-engine gev_translation CSVs written by `gev` and joins
                  lon/lat from static.csv — no recompute.
                  -> hydro_indices/maps/gev_translation_maps_*.png

  atlas           MAIN-FIGURE composite (narrative4 Fig 5): the two-engine ×
                  four-index too-hot bias atlas — global maps of
                  full − reduced for peak flow (AnnMax), high flow (Q99),
                  low flow (Q05) and mean runoff, HBV and LSTM, with
                  non-significant catchments shown in grey (significance
                  masking). Reads ONLY the cached signif_*.csv focus files.
                  -> hydro_indices/maps/too_hot_bias_atlas_*.png

  atlas_extra     SUPPLEMENTARY companions to `atlas` (S19-S21): the same
                  two-engine bias maps for the NINE Table-S6 streamflow
                  metrics the main text does not map, in three family
                  figures of three metrics each — high flows and floods
                  (Q95, RL10, RL50), average conditions and flow-duration-
                  curve shape (Q50, Q/P, FDCslope), and low flow, variability
                  and flashiness (AnnMin, SD, FlashRB). Identical conventions
                  to `atlas`: grey = not significant, one colour scale per
                  metric row shared by the two hydrological models only.
                  Also writes a per-metric statistics table so the captions
                  can quote median bias / significant fraction / median S/N
                  instead of describing the maps qualitatively.
                  -> hydro_indices/maps/too_hot_bias_atlas_{highflow,central,
                     lowflow}_*.png, fig_data/atlas_extra_stats_*.csv

  change_maps     SUPPLEMENTARY maps of the PROJECTED CHANGE itself (not the
                  full - reduced difference): for each of the 14 Table-S6
                  indices, the full-ensemble and reduced-ensemble projected
                  change, two hydrological models, two metrics per figure
                  (7 figures). One colour scale per metric across all four of
                  its panels, no significance mask (the cached test compares
                  ensemble compositions, not a change against zero). Reads the
                  same signif_*.csv focus files — the full-ensemble mean is
                  recovered from the two disjoint groups, so nothing is
                  recomputed. Optional CLI argument selects metrics by key.
                  -> hydro_indices/maps/projected_change_maps_*.png,
                     fig_data/change_maps_stats_*.csv

Run (via slurm — slurm/gev_translation.sh, slurm/fdr_dipole.sh,
slurm/paper_maps.sh):
    python paper/figures/scripts/gev_fdr_dipole.py baseline_cache [max_stations]
    python paper/figures/scripts/gev_fdr_dipole.py gev
    python paper/figures/scripts/gev_fdr_dipole.py gev_map
    python paper/figures/scripts/gev_fdr_dipole.py fdr
    python paper/figures/scripts/gev_fdr_dipole.py dipole
    python paper/figures/scripts/gev_fdr_dipole.py atlas
    python paper/figures/scripts/gev_fdr_dipole.py atlas_extra
    python paper/figures/scripts/gev_fdr_dipole.py change_maps
    python paper/figures/scripts/gev_fdr_dipole.py change_maps annmax,rl20
    python paper/figures/scripts/gev_fdr_dipole.py all      # all in order
"""

import os
import re
import sys
import glob
import logging

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))           # paper/figures/scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))  # repo root
sys.path.insert(0, _THIS)                                     # fig01_model_screening etc.
from config import CONFIG
# Nature-style bold top-left panel letters (multi-panel figures only)
from figstyle import add_panel_letters, mfmt, percentile_label, plabel
# method switch: removal (default; behaviour unchanged) vs weighting
# (weighted ensemble replaces the reduced one; outputs tagged _wt/_wtkde)
import ensemble_method as em

# Screen-aware label text: "too-hot" under SCREEN="hot", but the excluded set
# is not too-hot under SCREEN="both", so every figure/label string uses these
# instead of hardcoding the word.
_XHOT  = em.excluded_label()
_XHOTT = em.excluded_label(title=True)


def _lbl_second():
    """Legend name of the non-full ensemble.

    This used to hardcode "weighted (BMA)", which named Massoud's scheme on
    EVERY weighting run: the gaussjoint figures went out labelled BMA, the one
    scheme main4.tex contrasts the kernel against. The variant is already
    carried by the tag() suffix in the file name, so the legend states the
    contrast only, as main4.tex does ("the full (equal-weight) and weighted
    ensembles")."""
    return "weighted" if em.is_weighting() else "reduced (screened)"


def _word_screened():
    """The verb/adjective used in labels: 'screened' or 'weighted'."""
    return "weighted" if em.is_weighting() else "screened"

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- shared paths & conventions (identical to the other fig scripts) -------
ROOT       = os.path.join(CONFIG.out_path, CONFIG.delta_method)
CACHE_DIR  = os.path.join(ROOT, "fig_data")      # same as analyze_*/build_* caches
FLOW_CACHE = os.path.join(CACHE_DIR, "stats_change.parquet")
BASE_CACHE = os.path.join(CACHE_DIR, "flood_rl_baseline.parquet")
HYDRO_DIR      = os.path.join(os.path.dirname(_THIS), "hydro_indices", "summary")
HYDRO_MAPS_DIR = os.path.join(os.path.dirname(_THIS), "hydro_indices", "maps")
STATIC_CSV = os.path.join(CONFIG.gscad_path, "dataframes", "static.csv")

FOCUS_SCEN = "ssp585"            # mirror analyze_full_vs_reduced_significance.py
FOCUS_PER  = "2081-2100"
ENGINES    = ("hbv", "lstm")
ECOL       = {"hbv": "#3B6FB6", "lstm": "#C24B4B"}


def _latest_focus_csv(pattern):
    """Newest focus CSV in CACHE_DIR matching `pattern`, else None."""
    hits = glob.glob(os.path.join(CACHE_DIR, pattern))
    return max(hits, key=os.path.getmtime) if hits else None


def _global_ax(fig, spec):
    """Global map axes on `spec` (a SubplotSpec): Robinson via cartopy when
    available, plain lon/lat rectangle otherwise. Returns (ax, scatter_kwargs)
    — the same fallback pattern as the analyzer's transmission figure."""
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


def _station_coords():
    """station (str) -> (lon, lat) Series pair from static.csv (the same
    'long'/'lat' columns build_streamflow_cache.py uses)."""
    st = pd.read_csv(STATIC_CSV, comment="#", index_col=0)
    lon = st["long"].astype(float)
    lat = st["lat"].astype(float)
    lon.index = lon.index.astype(str)
    lat.index = lat.index.astype(str)
    return lon, lat


# =============================================================================
# STEP 1 — baseline_cache
# =============================================================================
# flood_q10 comes from the SAME L-moment GEV fit as q20/q50 but was written by
# a later backfill round, so it is read opportunistically: a station that
# predates that round still yields its q20/q50 row with base_flood_q10 = NaN.
# Keeping those stations is deliberate — dropping them would change the
# catchment count the manuscript's figure filenames carry.
_BASE_READ_COLS    = ["scenario", "engine", "flood_q20", "flood_q50"]
_BASE_READ_COLS_Q10 = _BASE_READ_COLS + ["flood_q10"]


def _baseline_read_one(f):
    """(status, station, rows) for one stats.parquet. Top-level so a fork Pool
    can dispatch it."""
    stn = os.path.basename(os.path.dirname(f))
    try:
        sdf = pd.read_parquet(f, columns=_BASE_READ_COLS_Q10)
    except Exception:
        try:                       # station predates the flood_q10 backfill
            sdf = pd.read_parquet(f, columns=_BASE_READ_COLS)
            sdf["flood_q10"] = np.nan
        except Exception:
            # stations from before the flood backfill have no flood_* columns
            return ("no_flood_cols", stn, None)
    base = sdf[sdf["scenario"].astype(str) == "baseline"]
    if base.empty:
        return ("no_baseline_row", stn, None)
    rows = []
    for eng, sub in base.groupby("engine"):
        q20 = pd.to_numeric(sub["flood_q20"], errors="coerce").dropna()
        q50 = pd.to_numeric(sub["flood_q50"], errors="coerce").dropna()
        q10 = pd.to_numeric(sub["flood_q10"], errors="coerce").dropna()
        if q20.empty or q50.empty:
            continue                       # GEV fit failed / < min_years
        rows.append({"station": stn, "engine": str(eng),
                     "base_flood_q10": float(q10.iloc[0]) if not q10.empty
                                       else np.nan,
                     "base_flood_q20": float(q20.iloc[0]),
                     "base_flood_q50": float(q50.iloc[0])})
    if not rows:
        return ("no_valid_fit", stn, None)
    return ("ok", stn, rows)


def build_flood_baseline_cache(max_stations=None):
    """Cache baseline flood_q20/q50 per station x engine -> BASE_CACHE."""
    from multiprocessing import Pool

    # derive workers from the slurm allocation (project rule: never hardcode)
    workers = (int(os.environ["SLURM_CPUS_PER_TASK"])
               if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit() else 1)
    files = sorted(glob.glob(os.path.join(ROOT, "*", "stats.parquet")))
    if max_stations is not None:
        files = files[:max_stations]
    logger.info(f"{len(files)} station stats.parquet under {ROOT} "
                f"| workers={workers}")

    skip = {"no_flood_cols": 0, "no_baseline_row": 0, "no_valid_fit": 0}
    frames = []

    def _collect(i, status, rows):
        if status == "ok":
            frames.extend(rows)
        else:
            skip[status] += 1
        if (i + 1) % 1000 == 0:
            logger.info(f"  [{i+1}/{len(files)}] scanned")

    if workers > 1:
        with Pool(workers) as pool:
            for i, (status, _stn, rows) in enumerate(
                    pool.imap_unordered(_baseline_read_one, files, chunksize=64)):
                _collect(i, status, rows)
    else:
        for i, f in enumerate(files):
            status, _stn, rows = _baseline_read_one(f)
            _collect(i, status, rows)

    if not frames:
        raise RuntimeError("no station produced baseline flood return levels "
                           "— has backfill_streamflow_stats.py run?")
    df = pd.DataFrame(frames)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(BASE_CACHE)

    logger.info("flood baseline cache accounting:")
    logger.info(f"  scanned                : {len(files)}")
    for k, v in skip.items():
        logger.info(f"  skipped [{k}] : {v}")
    logger.info(f"  stations cached        : {df['station'].nunique()}")
    logger.info(f"  rows (station x engine): {len(df)}")
    logger.info(f"  rows with flood_q10    : {int(df['base_flood_q10'].notna().sum())}"
                f" ({100 * df['base_flood_q10'].notna().mean():.1f}%)")
    logger.info(f"  Q20 median (mm/day)    : "
                + ", ".join(f"{e}={g['base_flood_q20'].median():.2f}"
                            for e, g in df.groupby('engine')))
    logger.info(f"Cached -> {BASE_CACHE}")


# =============================================================================
# STEP 2 — gev (return-period translation)
# =============================================================================
# implied return periods outside this window are unreliable extrapolations of
# the two-point Gumbel line; they are CLIPPED and the clipped share reported.
T_MIN, T_MAX = 1.5, 500.0
Y10 = -np.log(-np.log(1 - 1 / 10.0))       # Gumbel reduced variate at T=10
Y20 = -np.log(-np.log(1 - 1 / 20.0))       # Gumbel reduced variate at T=20
Y50 = -np.log(-np.log(1 - 1 / 50.0))


def _implied_T(q, q20, q50):
    """Return period of level q on the Gumbel line through (20, q20), (50, q50).
    Vectorised; NaN where the curve is degenerate (q50 <= q20)."""
    q, q20, q50 = (np.asarray(v, float) for v in (q, q20, q50))
    b = (q50 - q20) / (Y50 - Y20)
    a = q20 - b * Y20
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = (q - a) / b
        T = 1.0 / (1.0 - np.exp(-np.exp(-y)))
    return np.where((b > 0) & np.isfinite(T), T, np.nan)


def _ensemble_means(df, col, too_hot):
    """station x (full, reduced) means of `col`: members -> model mean ->
    unweighted mean over models (one model one vote). Under method=weighting
    the 'reduced' column is the WEIGHTED model mean (weights renormalised
    over the models present per station); the column name is kept so every
    downstream consumer works unchanged."""
    mm = (df.dropna(subset=[col])
            .groupby(["station", "model"])[col].mean().reset_index())
    mm["is_hot"] = mm["model"].isin(too_hot)
    g = mm.groupby("station")
    if em.is_weighting():
        mm["_w"] = mm["model"].map(em.get_weights())
        mm["_wx"] = mm["_w"] * mm[col]
        gs = mm.groupby("station")
        out = pd.DataFrame({
            "full":     g[col].mean(),
            "reduced":  gs["_wx"].sum() / gs["_w"].sum(),
            "n_models": g[col].size(),
            "n_hot":    (mm[mm["is_hot"]].groupby("station")[col].size()
                         .reindex(g[col].size().index).fillna(0).astype(int)),
        })
        return out[out["n_models"] >= 4]
    out = pd.DataFrame({
        "full":     g[col].mean(),
        "reduced":  mm[~mm["is_hot"]].groupby("station")[col].mean(),
        "n_models": g[col].size(),
        "n_hot":    mm[mm["is_hot"]].groupby("station")[col].size(),
    })
    # need both groups represented, as in the significance test
    return out[(out["n_hot"] >= 2) & (out["n_models"] - out["n_hot"] >= 2)]


def gev_return_period_translation():
    """Translate the cached return-level changes into return-period language;
    per-catchment CSVs + summary CSV + figure. Needs BASE_CACHE."""
    # Model universe + screen come from ensemble_method.py (single source of
    # truth). Resolved HERE rather than at module level because
    # em.full_models() opens a station's mean_deltas.nc, which the fdr/dipole
    # steps must not depend on.
    from realization_blocklist import apply_blocklist

    full_models = list(em.full_models())
    too_hot     = em.excluded_models()   # active screen: hot only, or hot+cold
    retained    = em.retained_models()

    for f, maker in ((FLOW_CACHE, "build_streamflow_cache.py"),
                     (BASE_CACHE, "gev_fdr_dipole.py baseline_cache")):
        if not os.path.exists(f):
            sys.exit(f"missing {f} — run {maker} first.")

    cols = ["station", "scenario", "model", "member", "period", "engine",
            "delta_rel_flood_q10", "delta_rel_flood_q20", "delta_rel_flood_q50"]
    try:                       # row-group filter push-down (pyarrow)
        df = pd.read_parquet(
            FLOW_CACHE, columns=cols,
            filters=[("scenario", "==", FOCUS_SCEN), ("period", "==", FOCUS_PER)])
    except (TypeError, ValueError) as e:
        print(f"(parquet filters unsupported here — {e}; falling back to "
              "column-selective full read)")
        df = pd.read_parquet(FLOW_CACHE, columns=cols)
        df = df[(df["scenario"].astype(str) == FOCUS_SCEN)
                & (df["period"].astype(str) == FOCUS_PER)]
    df = apply_blocklist(df, label="gev translation")
    df = df[df["model"].isin(full_models)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"{FOCUS_SCEN} {FOCUS_PER}: {len(df)} realization rows, "
          f"{df['station'].nunique()} stations | FULL={len(full_models)} "
          f"EXCLUDED={len(too_hot)} RETAINED={len(retained)}")

    base = pd.read_parquet(BASE_CACHE)
    if "base_flood_q10" not in base.columns:   # cache predates the 3rd anchor
        base["base_flood_q10"] = np.nan
        print("baseline cache has no flood_q10 — Fig. 4a will fall back to its "
              "20-50 segment; rerun baseline_cache to add the third anchor.")

    summary_rows, per_eng_frames = [], {}
    for eng in ENGINES:
        de = df[df["engine"].astype(str) == eng]
        if de.empty:
            print(f"[{eng}] no rows; skipping.")
            continue
        e20 = _ensemble_means(de, "delta_rel_flood_q20", too_hot).rename(
            columns={"full": "full_d20", "reduced": "reduced_d20"})
        e50 = _ensemble_means(de, "delta_rel_flood_q50", too_hot)[
            ["full", "reduced"]].rename(
            columns={"full": "full_d50", "reduced": "reduced_d50"})
        # the 10-yr level is a THIRD fitted anchor for the Fig. 4a curve only;
        # it is joined left so that a catchment missing it keeps its q20/q50
        # row and the headline RL20 numbers stay on the same catchment set.
        e10 = _ensemble_means(de, "delta_rel_flood_q10", too_hot)[
            ["full", "reduced"]].rename(
            columns={"full": "full_d10", "reduced": "reduced_d10"})
        m = e20.join(e50, how="inner").join(e10, how="left")
        m = m.reset_index().merge(base[base["engine"] == eng], on="station")
        n_join = len(m)

        # absolute future return levels per ensemble (deltas are FRACTIONS)
        m["Q20_full"] = m["base_flood_q20"] * (1 + m["full_d20"])
        m["Q20_red"]  = m["base_flood_q20"] * (1 + m["reduced_d20"])
        m["Q50_full"] = m["base_flood_q50"] * (1 + m["full_d50"])
        m["Q50_red"]  = m["base_flood_q50"] * (1 + m["reduced_d50"])
        m["Q10_full"] = m["base_flood_q10"] * (1 + m["full_d10"])
        m["Q10_red"]  = m["base_flood_q10"] * (1 + m["reduced_d10"])
        # the q10 columns are NOT in `ok`: they feed the Fig. 4a curve only, and
        # gating on them would move the catchment count behind every headline
        # RL20 number and every figure filename.
        ok = (m[["Q20_full", "Q20_red", "Q50_full", "Q50_red"]] > 0).all(axis=1) \
            & (m["Q50_full"] > m["Q20_full"]) & (m["Q50_red"] > m["Q20_red"])
        n_degenerate = int((~ok).sum())
        m = m[ok].copy()

        m["T_screened_of_rawQ20"] = _implied_T(m["Q20_full"], m["Q20_red"], m["Q50_red"])
        m["T_raw_of_screenedQ20"] = _implied_T(m["Q20_red"], m["Q20_full"], m["Q50_full"])
        n_clip = int(((m["T_screened_of_rawQ20"] < T_MIN)
                      | (m["T_screened_of_rawQ20"] > T_MAX)).sum())
        for c in ("T_screened_of_rawQ20", "T_raw_of_screenedQ20"):
            m[c] = m[c].clip(T_MIN, T_MAX)

        # headline numbers (percent for readability; deltas stored as fractions)
        m["d20_full_pct"] = m["full_d20"] * 100
        m["d20_red_pct"]  = m["reduced_d20"] * 100
        m["rl20_bias_pp"] = m["d20_full_pct"] - m["d20_red_pct"]
        gm_full, gm_red = m["d20_full_pct"].mean(), m["d20_red_pct"].mean()
        share = 100 * (gm_full - gm_red) / gm_full if gm_full != 0 else np.nan
        q25, q75 = m["T_screened_of_rawQ20"].quantile([0.25, 0.75])

        summary_rows.append({
            "engine": eng, "n_catchments": len(m),
            "n_joined": n_join, "n_degenerate_curve": n_degenerate,
            "n_T_clipped": n_clip,
            "n_with_q10_anchor": int(m["Q10_full"].notna().sum()),
            "RL20_change_full_globalmean_pct": gm_full,
            "RL20_change_reduced_globalmean_pct": gm_red,
            "RL20_bias_globalmean_pp": gm_full - gm_red,
            "RL20_bias_median_pp": float(m["rl20_bias_pp"].median()),
            "share_of_RL20_intensification_pct": share,
            "T_screened_of_rawQ20_median_yr": float(m["T_screened_of_rawQ20"].median()),
            "T_screened_of_rawQ20_iqr_yr": f"{q25:.1f}-{q75:.1f}",
            "frac_catchments_T_gt_20": float((m["T_screened_of_rawQ20"] > 20).mean()),
            "frac_catchments_T_gt_30": float((m["T_screened_of_rawQ20"] > 30).mean()),
        })
        per_eng_frames[eng] = m

        keep = ["station", "n_models", "n_hot",
                "base_flood_q10", "base_flood_q20", "base_flood_q50",
                "d20_full_pct", "d20_red_pct", "rl20_bias_pp",
                "Q10_full", "Q10_red", "Q20_full", "Q20_red",
                "Q50_full", "Q50_red",
                "T_screened_of_rawQ20", "T_raw_of_screenedQ20"]
        out_csv = em.tag(os.path.join(
            CACHE_DIR, f"gev_translation_{FOCUS_SCEN}_{FOCUS_PER}_{eng}"
                       f"_n{len(m)}sta.csv"))
        m[[c for c in keep if c in m.columns]].to_csv(out_csv, index=False)
        print(f"[{eng}] per-catchment table -> {out_csv}")

    if not summary_rows:
        sys.exit("no engine produced results.")
    summary = pd.DataFrame(summary_rows)
    s_csv = em.tag(os.path.join(
        CACHE_DIR, f"gev_translation_{FOCUS_SCEN}_{FOCUS_PER}_summary.csv"))
    summary.to_csv(s_csv, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.2f}"):
        print("\n== GEV return-period translation summary ==")
        print(summary.to_string(index=False))
    print(f"\nwritten -> {s_csv}")

    # ---- figure: distribution of T* per engine ---------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(figure skipped: {e})")
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bins = np.geomspace(T_MIN, T_MAX, 60)
    for eng, m in per_eng_frames.items():
        axA.hist(m["T_screened_of_rawQ20"], bins=bins, histtype="step",
                 lw=1.6, color=ECOL[eng], density=True,
                 label=f"{eng.upper()} (median "
                       f"{m['T_screened_of_rawQ20'].median():.0f} yr)")
        axB.hist((BIAS_SIGN * m["rl20_bias_pp"]).clip(-30, 30), bins=60,
                 histtype="step",
                 lw=1.6, color=ECOL[eng], density=True,
                 label=f"{eng.upper()} (median "
                       f"{(BIAS_SIGN * m['rl20_bias_pp']).median():+.1f} pp)")
    axA.axvline(20, color="0.4", lw=0.8, ls="--")
    axA.set_xscale("log")
    axA.set_xlabel("return period of the raw-ensemble 20-yr flood\n"
                   "on the screened ensemble's curve (yr)")
    axA.set_ylabel("density")
    axA.set_title("implied return period T*")
    axB.axvline(0, color="0.4", lw=0.8, ls="--")
    axB.set_xlabel(f"{_XHOT} bias of the 20-yr return-level change\n"
                   f"({SIGNCONV}, pp)")
    axB.set_title("RL20 bias")
    for ax in (axA, axB):
        ax.grid(color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    add_panel_letters([axA, axB])
    n = max(len(m) for m in per_eng_frames.values())
    #fig.suptitle(f"Design-flood translation of the too-hot bias "
    #             f"({FOCUS_SCEN} {FOCUS_PER})", y=1.02)
    os.makedirs(HYDRO_DIR, exist_ok=True)
    out = em.tag(os.path.join(HYDRO_DIR,
                              f"gev_translation_{FOCUS_SCEN}_{FOCUS_PER}"
                              f"_n{n}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


# =============================================================================
# STEP 2b — gev_map (narrative4 Fig 3: RL20-bias maps + T* distribution)
# =============================================================================
def gev_translation_maps():
    """Main-figure composite from the per-engine gev_translation CSVs, 2 x 2:
    row 1 (a, b) per-catchment RL20 bias maps (full − reduced, pp) for
    HBV/LSTM with a shared colorbar under the maps; row 2 (c) the T*
    distribution and (d) the RL20-inflation distribution (values beyond
    ±30 pp are clipped into the outermost bins; the exceedance share is
    stated in the legend instead of piling up at the axis edge). No
    recompute — CSV + static.csv only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon_s, lat_s = _station_coords()
    per_eng = {}
    for eng in ENGINES:
        path = _latest_focus_csv(
            f"gev_translation_{FOCUS_SCEN}_{FOCUS_PER}_{eng}"
            f"_n*sta{em.suffix()}.csv")
        if path is None:
            print(f"[{eng}] no gev_translation CSV — run the 'gev' step first;"
                  " skipping engine.")
            continue
        d = pd.read_csv(path)
        d["station"] = d["station"].astype(str)
        d["lon"] = d["station"].map(lon_s)
        d["lat"] = d["station"].map(lat_s)
        d = d[np.isfinite(d["lon"]) & np.isfinite(d["rl20_bias_pp"])]
        per_eng[eng] = d
        print(f"[{eng}] {os.path.basename(path)} -> {len(d)} catchments mapped")
    if not per_eng:
        sys.exit("gev_map: no engine data found.")

    allb = np.concatenate([m["rl20_bias_pp"].to_numpy(float)
                           for m in per_eng.values()])
    lim = min(float(np.nanpercentile(np.abs(allb), 98)), 25.0) or 1.0

    fig = plt.figure(figsize=(15, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 0.55],
                          hspace=0.28, wspace=0.06)
    sc, map_axes = None, []
    for i, (eng, m) in enumerate(per_eng.items()):
        ax, tk = _global_ax(fig, gs[0, i])
        map_axes.append(ax)
        sc = ax.scatter(m["lon"], m["lat"], c=BIAS_SIGN * m["rl20_bias_pp"],
                        s=4,
                        cmap="RdYlBu_r", vmin=-lim, vmax=lim, linewidths=0,
                        rasterized=True, **tk)
        ax.set_title(f"{eng.upper()}: 20-yr return-level "
                     f"bias, {SIGNCONV} (median "
                     f"{(BIAS_SIGN * m['rl20_bias_pp']).median():+.1f} pp)",
                     fontsize=10)
    fig.colorbar(sc, ax=map_axes, orientation="horizontal", fraction=0.05,
                 pad=0.04, aspect=50, extend="both",
                 label=f"{_XHOT} bias of the 20-yr flood return-level change "
                       f"({SIGNCONV}, pp)")

    # bottom row (c, d): narrower panels with a wide central gap so panel d's
    # y-tick labels clear panel c. The plots live in the two wide columns
    # (1 and 3); columns 0, 2, 4 are spacers framing them.
    gs_bot = gs[1, :].subgridspec(
        1, 5, width_ratios=[0.38, 1.12, 0.28, 1.12, 0.38], wspace=0.0)

    # (c) implied return period T*
    axC = fig.add_subplot(gs_bot[0, 1])
    bins = np.geomspace(T_MIN, T_MAX, 60)
    for eng, m in per_eng.items():
        t = m["T_screened_of_rawQ20"].to_numpy(float)
        t = t[np.isfinite(t)]
        axC.hist(t, bins=bins, histtype="step", lw=1.6, color=ECOL[eng],
                 density=True,
                 label=f"{eng.upper()} (median {np.median(t):.0f} yr)")
    axC.axvline(20, color="0.4", lw=0.8, ls="--")
    axC.set_xscale("log")
    axC.set_xlabel("return period of the raw-ensemble 20-yr flood\n"
                   f"on the {_word_screened()} ensemble's curve (yr)")
    axC.set_ylabel("density")
    axC.set_title("implied return period T*", fontsize=10)

    # (d) RL20 inflation, clipped display with exceedance stated in legend
    axD = fig.add_subplot(gs_bot[0, 3])
    dlim = 30.0
    for eng, m in per_eng.items():
        b = BIAS_SIGN * m["rl20_bias_pp"].to_numpy(float)
        b = b[np.isfinite(b)]
        if BIAS_SIGN > 0:
            out_txt = f"{100.0 * float(np.mean(b > dlim)):.0f}% > +{dlim:.0f}"
        else:
            out_txt = f"{100.0 * float(np.mean(b < -dlim)):.0f}% < −{dlim:.0f}"
        axD.hist(np.clip(b, -dlim, dlim), bins=60, histtype="step", lw=1.6,
                 color=ECOL[eng], density=True,
                 label=f"{eng.upper()} (median {np.median(b):+.1f} pp; "
                       f"{out_txt})")
    axD.axvline(0, color="0.4", lw=0.8, ls="--")
    axD.set_xlabel(f"{_XHOT} bias of the 20-yr return-level change\n"
                   f"({SIGNCONV}, pp; display clipped at ±{dlim:.0f})")
    axD.set_ylabel("density")
    axD.set_title("RL20 bias across catchments", fontsize=10)

    for ax in (axC, axD):
        ax.grid(color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper left")

    # explicit order: the two engine maps (top row) are a,b; the two histograms
    # (bottom row) are c,d -- the letters the titles used to carry.
    add_panel_letters(map_axes + [axC, axD])
    n = max(len(m) for m in per_eng.values())
    fig.suptitle(f"Design-flood translation of the {_XHOT} bias "
                 f"({FOCUS_SCEN} {plabel(FOCUS_PER)})", y=0.985)
    os.makedirs(HYDRO_MAPS_DIR, exist_ok=True)
    out = em.tag(os.path.join(HYDRO_MAPS_DIR,
                              f"gev_translation_maps_{FOCUS_SCEN}_"
                              f"{FOCUS_PER}_n{n}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


# =============================================================================
# STEP 2c — fig4 (design-flood MAIN figure: concept + magnitude +
#           translation + geography). Five panels since 7 Sep 2026: a, b, c on
#           row 1 and the two RL20-bias maps as d, e on row 2. Additive: reads
#           ONLY the caches written by `gev` (per-engine gev_translation CSVs
#           + summary) and static.csv; touches neither
#           gev_return_period_translation() nor gev_translation_maps().
# =============================================================================
# ensemble colours for the full (raw) vs reduced (screened) contrast — kept
# separate from ECOL (which colours the two engines) so the two encodings do
# not collide in this figure.
C_FULL = "#C24B4B"       # red   = full / raw ensemble
C_REDU = "#3B6FB6"       # blue  = reduced / screened ensemble


def _tint(hexcol, frac):
    """Blend a colour `frac` of the way towards white (frac=0 -> unchanged)."""
    r, g, b = (int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return tuple(c + (1.0 - c) * frac for c in (r, g, b))


# Panel (b) only. The bar tops are joined by a black double-headed arrow that
# marks the too-hot share; at full saturation the bars swallowed it. The fills
# are therefore whitened and the saturated hue is kept as the bar edge, so the
# full/weighted identity still matches the saturated lines of panel (a).
C_FULL_FILL = _tint(C_FULL, 0.58)
C_REDU_FILL = _tint(C_REDU, 0.58)

# Fig. 4 ran its legends at 8 pt and its panel titles at 9.5 pt, i.e. it fought
# the figstyle house sizes (legend 10, title 12) downward and printed smaller
# than every other figure in the paper. They now sit at or above the house
# sizes: dropping the sixth panel widened the canvas to 16.5 in, which made the
# previous 10/11 pt read small against the larger panels.
PANEL_LEGEND_FS = 11
PANEL_TITLE_FS  = 12
PANEL_ANNOT_FS  = 11     # in-panel text: the T* label (a) and the shares (b)

# Panel (c) display range (yr). The T* histogram is binned over the full
# T_MIN-T_MAX span; these bound the AXIS only, so nothing is dropped.
C_TLO, C_THI = 5.0, 100.0


def gev_translation_figure4():
    """Design-flood MAIN figure (Fig. 4), five panels on 2 rows:
      row 1
      (a) two-engine-mean return-level curve, full vs reduced, piecewise-linear
          in the Gumbel reduced variate through the three FITTED anchors
          (T = 10, 20, 50) and drawn only over that range, with the raw 20-yr
          flood -> T* translation on it. Anchors are across-catchment MEDIANS
          of the level normalised by the catchment's own present-day 20-yr
          flood, so (a) is a median composite while (b) plots global means;
      (b) full vs reduced projected 20-yr return-level change per engine,
          with the share contributed by the too-hot models annotated;
      (c) the per-catchment implied-return-period T* distribution;
      row 2
      (d, e) the per-engine (HBV, LSTM) per-catchment RL20 bias, mapped.

    An earlier six-panel version carried a further panel between (c) and the
    maps: the across-catchment histogram of the RL20 bias. It was dropped on
    7 Sep 2026 as doubly redundant. That histogram was the marginal
    distribution of exactly the field the two maps show, and the map titles
    already print its two medians; the same per-catchment distribution is also
    already a box in Fig. 3a, which plots all 14 curated indices including the
    three return levels. Its only unique content was the fraction of catchments
    beyond the +/-30 pp display clip, which is a property of the clipping rather
    than a result. No number in main4.tex came from it.

    Every panel comes from the `gev` caches — no recompute."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- per-engine translation tables + the summary -----------------------
    per_eng = {}
    for eng in ENGINES:
        path = _latest_focus_csv(
            f"gev_translation_{FOCUS_SCEN}_{FOCUS_PER}_{eng}"
            f"_n*sta{em.suffix()}.csv")
        if path is None:
            print(f"[{eng}] no gev_translation CSV — run the 'gev' step first;"
                  " skipping engine.")
            continue
        d = pd.read_csv(path)
        d["station"] = d["station"].astype(str)
        per_eng[eng] = d
        print(f"[{eng}] {os.path.basename(path)} -> {len(d)} catchments")
    if not per_eng:
        sys.exit("fig4: no gev_translation CSVs — run the 'gev' step first.")

    s_csv = _latest_focus_csv(
        f"gev_translation_{FOCUS_SCEN}_{FOCUS_PER}_summary{em.suffix()}.csv")
    summary = pd.read_csv(s_csv) if s_csv else None

    # normalised return-level anchors (present-day 20-yr flood = 1.0), median
    # across catchments per engine, then the two-engine mean for the schematic.
    def _anchors(d):
        b20 = d["base_flood_q20"].to_numpy(float)
        ok = b20 > 0
        b20 = b20[ok]
        col = lambda c: d[c].to_numpy(float)[ok] / b20
        return dict(base10=np.nanmedian(col("base_flood_q10")),
                    base50=np.nanmedian(col("base_flood_q50")),
                    full10=np.nanmedian(col("Q10_full")),
                    full20=np.nanmedian(col("Q20_full")),
                    full50=np.nanmedian(col("Q50_full")),
                    red10=np.nanmedian(col("Q10_red")),
                    red20=np.nanmedian(col("Q20_red")),
                    red50=np.nanmedian(col("Q50_red")))
    A = [_anchors(d) for d in per_eng.values()]
    anc = {k: float(np.mean([a[k] for a in A])) for k in A[0]}

    # Six gridspec columns: row 1 splits into three equal analysis panels
    # (0:2, 2:4, 4:6) and row 2 into two equal maps (0:3, 3:6).
    fig = plt.figure(figsize=(16.5, 9.8))
    gs = fig.add_gridspec(2, 6, hspace=0.30, wspace=0.62,
                          height_ratios=[1.0, 1.02])

    # (a) return-level curve ------------------------------------------------
    # Drawn ONLY over the fitted range: the three anchors are the 10-, 20- and
    # 50-yr levels the L-moment GEV actually produced (the 100-yr level is
    # never fitted, Methods), joined piecewise-linearly in the Gumbel reduced
    # variate. Piecewise (not one refitted line) is required: it leaves the
    # 20-50 segment identical to the two-anchor line the method uses for T*,
    # so the T* marker stays exactly on the drawn curve. An earlier version
    # extended this line from 2 to 100 yr; below ~4 yr that extrapolation put
    # both future curves BELOW present-day, contradicting the +18-25% peak-flow
    # (mean annual maximum, ~2.33-yr recurrence) increase reported in Fig. 3.
    axA = fig.add_subplot(gs[0, 0:2])
    have10 = all(np.isfinite([anc["base10"], anc["full10"], anc["red10"]]))
    mono = have10 and (anc["base10"] < 1.0 and anc["full10"] < anc["full20"]
                       and anc["red10"] < anc["red20"])
    if have10 and not mono:
        print("fig4a: 10-yr anchors are not monotone at the median catchment "
              "— falling back to the 20-50 segment.")
    T_LO = 10.0 if mono else 20.0
    Tgrid = np.geomspace(T_LO, 50.0, 200)
    yv = -np.log(-np.log(1 - 1.0 / Tgrid))

    def _curve(v10, v20, v50):
        """Piecewise-linear through the fitted anchors in the Gumbel reduced
        variate; falls back to the (20, 50) segment when no 10-yr anchor."""
        if mono:
            return np.interp(yv, [Y10, Y20, Y50], [v10, v20, v50])
        b = (v50 - v20) / (Y50 - Y20)
        return (v20 - b * Y20) + b * yv
    axA.plot(Tgrid, _curve(anc["base10"], 1.0, anc["base50"]), color="0.6",
             lw=1.4, ls=":", label="present-day")
    axA.plot(Tgrid, _curve(anc["red10"], anc["red20"], anc["red50"]),
             color=C_REDU, lw=1.8, label=_lbl_second())
    axA.plot(Tgrid, _curve(anc["full10"], anc["full20"], anc["full50"]),
             color=C_FULL, lw=1.8, label="full (raw)")
    # every anchor is now an endpoint or the marked T=20 line, so the axis
    # limits themselves show the reader where the curve is pinned.
    # the raw-ensemble future 20-yr level, and its return period on the
    # screened curve (T*), computed exactly as the method does (_implied_T).
    L = anc["full20"]
    Tstar = float(_implied_T(np.array([L]), np.array([anc["red20"]]),
                             np.array([anc["red50"]]))[0])
    axA.scatter([20], [L], color=C_FULL, zorder=5, s=30)
    if np.isfinite(Tstar) and T_LO <= Tstar <= 50.0:
        axA.annotate("", xy=(Tstar, L), xytext=(20, L),
                     arrowprops=dict(arrowstyle="->", color="0.25", lw=1.5))
        axA.scatter([Tstar], [L], color=C_REDU, zorder=5, s=30)
        axA.text(Tstar * 1.06, L, f"$T^{{*}}\\approx{Tstar:.0f}$ yr",
                 fontsize=PANEL_ANNOT_FS, va="center")
    elif np.isfinite(Tstar):
        print(f"fig4a: T*={Tstar:.1f} yr is outside the fitted [{T_LO:.0f}, 50] "
              "range — arrow suppressed rather than drawn off a fitted curve.")
    axA.axvline(20, color="0.8", lw=0.8, ls="--")
    axA.set_xscale("log")
    # a log axis over a single decade defaults to almost no labelled ticks
    axA.set_xlim(T_LO, 50.0)
    ticks = [t for t in (10, 15, 20, 30, 40, 50) if T_LO <= t <= 50]
    axA.set_xticks(ticks)
    axA.set_xticks([], minor=True)
    axA.set_xticklabels([str(t) for t in ticks])
    axA.set_xlabel("Return period (yr)")
    axA.set_ylabel("Flood return level\n(relative to present-day 20-yr flood)")
    # two lines: the three-across row leaves ~5.5 in per panel, too narrow for
    # any of these titles on one line at the house size.
    axA.set_title(f"The raw 20-yr flood is a rarer event\nonce "
                  f"{_word_screened()}", fontsize=PANEL_TITLE_FS)
    axA.legend(fontsize=PANEL_LEGEND_FS, loc="upper left")
    axA.grid(color="0.92", lw=0.5); axA.set_axisbelow(True)

    # (b) full vs reduced RL20 change per engine ----------------------------
    axB = fig.add_subplot(gs[0, 2:4])
    if summary is not None and not summary.empty:
        engs = [e for e in ENGINES if e in set(summary["engine"].astype(str))]
        cell = lambda e, c: float(
            summary.loc[summary["engine"].astype(str) == e, c].iloc[0])
        x = np.arange(len(engs)); w = 0.22
        off = 0.20               # > w/2, so a white lane is left at xi for the
                                 # double-headed bias arrow
        full = [cell(e, "RL20_change_full_globalmean_pct") for e in engs]
        red = [cell(e, "RL20_change_reduced_globalmean_pct") for e in engs]
        shr = [cell(e, "share_of_RL20_intensification_pct") for e in engs]
        axB.bar(x - off, full, w, color=C_FULL_FILL, edgecolor=C_FULL,
                linewidth=1.0, label="full (raw)")
        axB.bar(x + off, red, w, color=C_REDU_FILL, edgecolor=C_REDU,
                linewidth=1.0, label=_lbl_second())
        for xi, fv, rv, sv in zip(x, full, red, shr):
            axB.annotate("", xy=(xi, fv), xytext=(xi, rv),
                         arrowprops=dict(arrowstyle="<->", color="black",
                                         lw=1.8, shrinkA=0, shrinkB=0,
                                         mutation_scale=16), zorder=6)
            # "artefact" would assert that the screened ensemble is the truth;
            # the annotation states the SOURCE of the share instead, which is
            # what the arithmetic supports (see the caption of Fig. 4).
            axB.text(xi, max(fv, rv) + 0.8, f"{sv:.0f}%\nfrom {_XHOT} models",
                     ha="center", va="bottom", fontsize=PANEL_ANNOT_FS)
        axB.set_xticks(x); axB.set_xticklabels([e.upper() for e in engs])
        axB.set_ylabel("Projected 20-yr return-level change (%)")
        axB.set_ylim(0, max(full) * 1.32)
        # opaque: the legend overlaps the weighted bar, and against the
        # whitened fill a semi-transparent box no longer separates.
        axB.legend(fontsize=PANEL_LEGEND_FS, loc="lower right",
                   framealpha=1.0)
        axB.grid(axis="y", color="0.92", lw=0.5); axB.set_axisbelow(True)
    axB.set_title(f"Full vs {_word_screened()}\n20-yr return-level change",
                  fontsize=PANEL_TITLE_FS)

    # (c) implied return period T* distribution -----------------------------
    # The view is restricted to C_TLO-C_THI yr. Over the full T_MIN-T_MAX
    # binning range the mass between 10 and 40 yr collapsed into three or four
    # bins and the panel read as a bare spike, hiding its actual message: the
    # translation is TIGHTLY clustered near 21-22 yr. The bins still span the
    # full range and the medians are still taken over every finite value, so
    # this restricts the AXIS only -- no catchment is dropped and the plotted
    # density is the true density, not one renormalised over a subset. The
    # fraction left off-screen is printed to the log rather than annotated, and
    # the 30-yr exceedance the manuscript quotes stays inside the view.
    axC = fig.add_subplot(gs[0, 4:6])
    bins = np.geomspace(T_MIN, T_MAX, 140)   # ~4% wide: resolves the peak
    for eng, d in per_eng.items():
        t = d["T_screened_of_rawQ20"].to_numpy(float)
        t = t[np.isfinite(t)]
        off = 100.0 * float(np.mean((t < C_TLO) | (t > C_THI)))
        print(f"fig4c [{eng}]: {off:.2f}% of catchments fall outside the "
              f"{C_TLO:.0f}-{C_THI:.0f} yr display range (axis limit only; "
              "binning and medians use every value)")
        axC.hist(t, bins=bins, histtype="step", lw=1.6, color=ECOL[eng],
                 density=True,
                 label=f"{eng.upper()} (median {np.median(t):.0f} yr)")
    axC.axvline(20, color="0.4", lw=0.8, ls="--")
    # the narrower three-across row brings the upper-left legend down onto the
    # tallest bin, so give the peak explicit headroom.
    axC.set_ylim(0, axC.get_ylim()[1] * 1.25)
    axC.set_xscale("log")
    axC.set_xlim(C_TLO, C_THI)
    # a log axis over ~1.5 decades defaults to two labelled ticks (10, 100)
    # the tick list must not reach below C_TLO: Axis.set_ticks widens the
    # view interval to span the ticks, which would undo the set_xlim above.
    axC.set_xticks([5, 10, 20, 30, 50, 100])
    axC.set_xticks([], minor=True)
    axC.set_xticklabels(["5", "10", "20", "30", "50", "100"])
    axC.set_xlabel("Return period of the raw-ensemble 20-yr flood\n"
                   f"on the {_word_screened()} ensemble's curve (yr)")
    axC.set_ylabel("Density")
    # (a) draws the 20 -> T* translation at the median catchment; (c) is that
    # same translation across every catchment, so the title says so rather than
    # naming the symbol (the x label and the caption define T*).
    axC.set_title("How much rarer,\ncatchment by catchment",
                  fontsize=PANEL_TITLE_FS)
    axC.legend(fontsize=PANEL_LEGEND_FS, loc="upper left")
    axC.grid(color="0.92", lw=0.5); axC.set_axisbelow(True)

    # (d, e) per-engine RL20 bias maps -------------------------------------
    # BIAS_SIGN flips the plotted orientation only (reduced − full when −1).
    lon_s, lat_s = _station_coords()
    allb = np.concatenate([d["rl20_bias_pp"].to_numpy(float)
                           for d in per_eng.values()])
    mlim = min(float(np.nanpercentile(np.abs(allb), 98)), 25.0) or 1.0
    sc, map_axes = None, []
    for j, (eng, d) in enumerate(per_eng.items()):
        dd = d.copy()
        dd["lon"] = dd["station"].map(lon_s)
        dd["lat"] = dd["station"].map(lat_s)
        dd = dd[np.isfinite(dd["lon"]) & np.isfinite(dd["rl20_bias_pp"])]
        dd["bias_pp"] = BIAS_SIGN * dd["rl20_bias_pp"]
        ax, tk = _global_ax(fig, gs[1, 3 * j:3 * (j + 1)])
        map_axes.append(ax)
        sc = ax.scatter(dd["lon"], dd["lat"], c=dd["bias_pp"], s=4,
                        cmap="RdYlBu_r", vmin=-mlim, vmax=mlim, linewidths=0,
                        rasterized=True, **tk)
        # mfmt: SIGNCONV in the same title is written with a true minus sign,
        # so the median must be too.
        ax.set_title(f"{eng.upper()}: RL20 bias, {SIGNCONV} "
                     f"(median {mfmt(dd['bias_pp'].median(), '+.1f')} pp)",
                     fontsize=PANEL_TITLE_FS)
    if sc is not None:
        fig.colorbar(sc, ax=map_axes, orientation="horizontal", fraction=0.05,
                     pad=0.04, aspect=55, extend="both",
                     label=f"{_XHOTT} bias of the 20-yr return-level change "
                           f"({SIGNCONV}, pp)")

    # explicit order: the three analysis panels are a-c and the two engine maps
    # on the bottom row are d,e -- the letters the titles used to carry.
    add_panel_letters([axA, axB, axC] + map_axes)
    n = max(len(d) for d in per_eng.values())
    os.makedirs(HYDRO_MAPS_DIR, exist_ok=True)
    out = em.tag(os.path.join(HYDRO_MAPS_DIR,
                              f"gev_translation_fig4_{FOCUS_SCEN}_"
                              f"{FOCUS_PER}_n{n}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


# =============================================================================
# STEP 2d — rl50_check (diagnostic, NO figure): does the design-flood too-hot
#           bias grow toward the tail? Compares the RL50 bias (full - reduced)
#           against the RL20 bias per engine, from the same per-engine
#           gev_translation CSVs (base_flood_q50, Q50_full, Q50_red). Additive.
# =============================================================================
def rl50_check():
    """Compare the projected 50-yr vs 20-yr return-level too-hot bias per
    engine; print a table and write fig_data/rl50_vs_rl20_check_*.csv. This
    tests whether 'the inflation grows toward the design tail' is defensible."""
    rows = []
    for eng in ENGINES:
        path = _latest_focus_csv(
            f"gev_translation_{FOCUS_SCEN}_{FOCUS_PER}_{eng}"
            f"_n*sta{em.suffix()}.csv")
        if path is None:
            print(f"[{eng}] no gev_translation CSV; skipping.")
            continue
        d = pd.read_csv(path)
        for c in ("base_flood_q50", "Q50_full", "Q50_red",
                  "d20_full_pct", "d20_red_pct", "rl20_bias_pp"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d[(d["base_flood_q50"] > 0) & np.isfinite(d["Q50_full"])
              & np.isfinite(d["Q50_red"])].copy()
        # recover the 50-yr return-level change from the cached anchors
        d["d50_full_pct"] = (d["Q50_full"] / d["base_flood_q50"] - 1.0) * 100
        d["d50_red_pct"] = (d["Q50_red"] / d["base_flood_q50"] - 1.0) * 100
        d["rl50_bias_pp"] = d["d50_full_pct"] - d["d50_red_pct"]
        f20, r20 = d["d20_full_pct"].mean(), d["d20_red_pct"].mean()
        f50, r50 = d["d50_full_pct"].mean(), d["d50_red_pct"].mean()
        rows.append({
            "engine": eng, "n": len(d),
            "RL20_full_mean_pct": f20, "RL20_reduced_mean_pct": r20,
            "RL50_full_mean_pct": f50, "RL50_reduced_mean_pct": r50,
            "RL20_bias_mean_pp": d["rl20_bias_pp"].mean(),
            "RL50_bias_mean_pp": d["rl50_bias_pp"].mean(),
            "RL20_bias_median_pp": d["rl20_bias_pp"].median(),
            "RL50_bias_median_pp": d["rl50_bias_pp"].median(),
            "RL50_gt_RL20_mean": bool(d["rl50_bias_pp"].mean()
                                      > d["rl20_bias_pp"].mean()),
            "RL50_gt_RL20_median": bool(d["rl50_bias_pp"].median()
                                        > d["rl20_bias_pp"].median()),
            "share20_artefact_pct": 100 * (f20 - r20) / f20 if f20 else np.nan,
            "share50_artefact_pct": 100 * (f50 - r50) / f50 if f50 else np.nan,
        })
    if not rows:
        sys.exit("rl50_check: no per-engine gev_translation CSVs found.")
    out = pd.DataFrame(rows)
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_csv = em.tag(os.path.join(
        CACHE_DIR, f"rl50_vs_rl20_check_{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    out.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.2f}"):
        print(f"\n== RL50 vs RL20 {_XHOT} bias (does the design-flood inflation "
              "grow toward the tail?) ==")
        print(out.to_string(index=False))
    print(f"\nwritten -> {out_csv}")
    print("\nReading guide: only if RL50_bias > RL20_bias for BOTH engines on "
          "BOTH the mean and the median is the sentence 'the inflation grows "
          "toward the design tail' defensible for the main text; a gap under "
          "~0.5 pp is partly a construction of the two-anchor Gumbel line and "
          "belongs (if anywhere) in one SI clause.")


# =============================================================================
# STEP 5 — atlas (narrative4 Fig 5: two-engine x four-index bias maps)
# =============================================================================
# columns of the atlas: (signif CSV column, panel label); biases in the focus
# CSVs are PERCENT (the analyzer rescales delta_rel_* x100 before export).
def _atlas_label(col, fallback):
    """Panel title for `col`, taken from figstyle.PERCENTILE_LABELS when that
    table covers the index and from `fallback` when it does not (AnnMax and the
    other unambiguous indices are deliberately outside the table). This is the
    call pattern figstyle's own docstring prescribes, and it exists because the
    label used to be hard-coded here: q99 was written "High flow (Q99)" while
    every other module -- flow_metrics.py, ATLAS_EXTRA below, figstyle -- calls
    it a VERY high flow and reserves "High flow" for q95. The wrong string was
    drawn into the atlas panel titles and copied from there into the
    supplement caption, so one index answered to two names. Sourcing the label
    instead of restating it removes that failure mode for good."""
    lab = percentile_label(col)
    return lab.long if lab else fallback


ATLAS_INDICES = [
    ("delta_rel_mean_annual_max", _atlas_label("delta_rel_mean_annual_max",
                                               "Mean annual maximum")),
    ("delta_rel_q99",             _atlas_label("delta_rel_q99",
                                               "Very high flow (Q99)")),
    ("delta_rel_q05",             _atlas_label("delta_rel_q05",
                                               "Low flow (Q05)")),
    ("delta_rel_qmean",           _atlas_label("delta_rel_qmean",
                                               "Mean runoff")),
]
# per-column cap on the symmetric colour limit (%), mirroring the analyzer's
# COLORBAR_LIM_CAP: low-flow relative changes blow up on ~0 baselines.
ATLAS_LIM_CAP = {"delta_rel_q05": 40.0}
# diverging colormap for the atlas: yellow (not white) at 0 so zero-bias
# catchments read as a distinct hue rather than blank; positive (full > reduced)
# = red, negative = blue (RdYlBu reversed).
ATLAS_CMAP = "RdYlBu_r"


def bias_atlas():
    """Main-figure composite: global maps of the too-hot bias
    (full − reduced, %) for four flow indices x two engines, from the cached
    signif CSVs; non-significant catchments in grey (significance masking)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}                       # (col, eng) -> DataFrame
    for col, _lab in ATLAS_INDICES:
        for eng in ENGINES:
            path = _latest_focus_csv(
                f"signif_{col}_{eng}_{FOCUS_SCEN}_{FOCUS_PER}"
                f"_n*sta{em.suffix()}.csv")
            if path is None:
                print(f"[{eng}] no focus CSV for {col}; panel will be empty.")
                continue
            d = pd.read_csv(path, usecols=["lon", "lat", "full_minus_reduced",
                                           "significant"])
            d = d[np.isfinite(d["full_minus_reduced"])]
            # plotted orientation only; the CSV column stays full - reduced
            d["bias"] = BIAS_SIGN * d["full_minus_reduced"]
            data[(col, eng)] = d
    if not data:
        sys.exit("atlas: no focus CSVs found — run the flow significance "
                 "pass first.")

    # transposed layout: rows = the four flow indices, columns = the two engines.
    # height/hspace chosen so the ~2:1 Robinson maps nearly fill their cells while
    # leaving a gap above each map for its title (no overlap, no large gaps).
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(len(ATLAS_INDICES), 2, hspace=0.16, wspace=0.04)
    all_axes = []          # every map panel, row-major, for the panel letters
    for i, (col, lab) in enumerate(ATLAS_INDICES):
        # shared symmetric scale per row (both engines)
        pool = [data[(col, e)]["bias"].to_numpy(float)
                for e in ENGINES if (col, e) in data]
        vals = np.concatenate(pool) if pool else np.array([1.0])
        lim = float(np.nanpercentile(np.abs(vals), 98)) or 1.0
        lim = min(lim, ATLAS_LIM_CAP.get(col, lim))
        sc, row_axes = None, []
        for j, eng in enumerate(ENGINES):
            ax, tk = _global_ax(fig, gs[i, j])
            row_axes.append(ax)
            all_axes.append(ax)
            if (col, eng) not in data:
                ax.set_title(f"{eng.upper()}: {lab} (missing)", fontsize=8)
                continue
            d = data[(col, eng)]
            ns, sg = d[~d["significant"]], d[d["significant"]]
            ax.scatter(ns["lon"], ns["lat"], s=1.2, c="0.82", linewidths=0,
                       rasterized=True, **tk)
            sc = ax.scatter(sg["lon"], sg["lat"], c=sg["bias"],
                            s=4, cmap=ATLAS_CMAP, vmin=-lim, vmax=lim,
                            linewidths=0, rasterized=True, **tk)
            frac = 100.0 * len(sg) / len(d) if len(d) else 0.0
            ax.set_title(f"{eng.upper()}: {lab}  ({frac:.0f}% significant)",
                         fontsize=8.5)
        if sc is not None:
            fig.colorbar(sc, ax=row_axes, orientation="vertical",
                         fraction=0.025, pad=0.02, shrink=0.7, extend="both",
                         label=BIAS_LABEL)
    # all_axes is filled row-major (index x engine), matching the reading order.
    add_panel_letters(all_axes)
    fig.suptitle(f"{_XHOTT} bias ({SIGNCONV}) across the flow spectrum: "
                 f"grey = not significant ({FOCUS_SCEN} {plabel(FOCUS_PER)})",
                 y=0.995)
    n = max(len(d) for d in data.values())
    os.makedirs(HYDRO_MAPS_DIR, exist_ok=True)
    out = em.tag(os.path.join(HYDRO_MAPS_DIR,
                              f"too_hot_bias_atlas_{FOCUS_SCEN}_"
                              f"{FOCUS_PER}_n{n}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


# =============================================================================
# STEP 5b — atlas_extra (supplementary S19-S21: the nine unmapped metrics)
# =============================================================================
# Same object and same conventions as bias_atlas(); only the metric list and
# the grouping differ. Written as its own function rather than a parameter of
# bias_atlas() so that editing the supplement can never disturb main-text
# Fig. 6.
#
# SIGN CONVENTION. Every panel here plots BIAS_SIGN * full_minus_reduced.
# DECIDED 2 Aug 2026: the manuscript reports every ensemble difference as
# reduced - full (Prof. Beck's 10 Jul request), so BIAS_SIGN = -1: negative
# values at the flood indices mean the too-hot models inflate the projection
# and screening lowers it. The flip is PRESENTATION-ONLY - the signif_*.csv
# focus files, the gev_translation CSVs and every stored cache keep the
# original full_minus_reduced column untouched (consumers read that column by
# name), and the Welch p, |bias|/spread ratio and significance masks are
# invariant under the flip. Applied in the same pass to bias_atlas(),
# gev_translation_figure4(), atlas_extra, the dipole, fig02, fig3_combined,
# the tcr sweep and the bootstrap null plots, plus the main4/supplement text.
BIAS_SIGN  = -1.0
# Name of the non-full ensemble INSIDE the sign convention. Under
# METHOD='weighting' nothing is removed, so "reduced" is the wrong word and
# contradicts main4.tex, which writes every difference as "weighted − full".
_SECOND    = "weighted" if em.is_weighting() else "reduced"
BIAS_LABEL = f"bias (pp, full − {_SECOND})" if BIAS_SIGN > 0 else \
             f"bias (pp, {_SECOND} − full)"
# Orientation string for axis labels / titles that spell the convention out.
SIGNCONV   = f"full − {_SECOND}" if BIAS_SIGN > 0 else f"{_SECOND} − full"

# Display cap on the symmetric colour limit, applied as
# min(98th percentile of |value|, cap) exactly as ATLAS_LIM_CAP is for Q05.
# The cap governs the DISPLAY only; the logged statistics are uncapped and the
# log records whether it actually bound.
#
# The two figure families need DIFFERENT caps because they plot different
# quantities. The 98th percentile already discards the pathological tail, so a
# cap is warranted only where the p98 is itself absurd — which depends on the
# quantity, not just the metric:
#   bias   Q50 p98 = 19-22 pp, AnnMin p98 = 610 pp, so only AnnMin binds; the
#          other entries are harmless guards against a future rebuild.
#   change Q05 p98 = 18,615 % and AnnMin p98 = 4,384 % (HBV, near-zero
#          baselines: report.tex P20), against medians of -35 % and -22 %, so
#          those two need the cap. Q50 (p98 = 76 %), Q/P (50 %) and the FDC
#          slope (57 %) do not, and capping them would clip real signal.
BIAS_LIM_CAP = {
    "delta_rel_q50":              40.0,
    "delta_rel_mean_annual_min":  40.0,
    "delta_rel_slope_fdc":        40.0,
    "delta_rel_runoff_ratio":     40.0,
}
CHANGE_LIM_CAP = {
    "delta_rel_q05":              40.0,
    "delta_rel_mean_annual_min":  40.0,
}


def _sym_limit(arrays, col, caps):
    """Symmetric colour limit for one metric: the 98th percentile of the
    absolute value pooled over `arrays`, capped per the `caps` mapping.
    Returns (limit_used, uncapped_p98) so the caller can log the difference."""
    pool = [a for a in arrays if len(a)]
    vals = np.concatenate(pool) if pool else np.array([1.0])
    p98 = float(np.nanpercentile(np.abs(vals), 98))
    lim = p98 if (np.isfinite(p98) and p98 > 0) else 1.0
    cap = caps.get(col)
    if cap is not None and lim > cap:
        logger.info(f"{col}: colour limit capped {lim:.1f} -> {cap:.1f} "
                    f"(display only)")
        lim = cap
    return lim, p98


def _map_scatter(ax, tk, d, valcol, lim, title, mask=False):
    """Draw one global map panel and return the mappable for the colorbar.

    mask=True reproduces the Fig. 6 convention: catchments failing the
    per-catchment Welch test are drawn first as a flat grey underlay and only
    the significant ones are coloured. Grey is a mask, not a data value, so it
    never becomes a colorbar entry. It applies to the FULL - REDUCED bias
    only: the test compares the two ensemble compositions, so it says nothing
    about whether a projected change differs from zero and must not be used to
    mask a projected-change panel."""
    if mask:
        ns, shown = d[~d["significant"]], d[d["significant"]]
        ax.scatter(ns["lon"], ns["lat"], s=1.2, c="0.82", linewidths=0,
                   rasterized=True, **tk)
    else:
        shown = d
    sc = ax.scatter(shown["lon"], shown["lat"], c=shown[valcol], s=4,
                    cmap=ATLAS_CMAP, vmin=-lim, vmax=lim, linewidths=0,
                    rasterized=True, **tk)
    ax.set_title(title, fontsize=7.5)
    return sc

# Three supplementary figures of three metrics each, grouped by the Table S6
# families. (stem, suptitle, [(signif-CSV column, panel label), ...]).
# Panel labels are kept short: the bold panel letter sits at the axes' top-left
# corner, so a long centred title runs into it. The full definitions live in
# Supplementary Table S6, which the captions cite.
EXTRA_ATLASES = [
    ("highflow",
     "High flows and design floods",
     # q95 and q50 below carry the Q-symbol ambiguity and are sourced from
     # figstyle via _atlas_label, as ATLAS_INDICES already is; RL10, RL50,
     # Q/P, the FDC slope, AnnMin, the flow SD and FlashRB do not and keep
     # their literal labels.
     [("delta_rel_q95",        _atlas_label("delta_rel_q95",
                                            "High flow (Q95)")),
      ("delta_rel_flood_q10",  "10-year flood (RL10)"),
      ("delta_rel_flood_q50",  "50-year flood (RL50)")]),
    ("central",
     "Average conditions and flow-duration-curve shape",
     [("delta_rel_q50",         _atlas_label("delta_rel_q50",
                                             "Median flow (Q50)")),
      ("delta_rel_runoff_ratio","Runoff ratio (Q/P)"),
      ("delta_rel_slope_fdc",   "FDC slope")]),
    ("lowflow",
     "Low flow, flow variability and flashiness",
     [("delta_rel_mean_annual_min", "Mean annual minimum (AnnMin)"),
      ("delta_rel_qstd",            "Flow SD"),
      ("delta_rel_flashiness_rb",   "Flashiness (FlashRB)")]),
]


def _load_metric(col):
    """{engine: DataFrame} for one metric, from the cached focus CSVs.

    Adds three derived columns on top of what the analyzer wrote:

      bias     BIAS_SIGN * full_minus_reduced (percentage points)
      reduced  the reduced-ensemble projected change (%). The reduced
               ensemble IS the retained (TCR <= 2.2 K) model set, so this is
               retained_mean unchanged.
      full     the full-ensemble projected change (%), recovered as the
               count-weighted mean of the two disjoint groups the analyzer
               tested, (h*hot + r*retained)/(h+r). This is the same
               one-model-one-vote mean used everywhere else, and it satisfies
               full - reduced = (h/N)(hot - retained), the identity the
               analyzer's own full_minus_reduced column is built from.

    One loader for both the bias maps and the projected-change maps."""
    out = {}
    for eng in ENGINES:
        path = _latest_focus_csv(
            f"signif_{col}_{eng}_{FOCUS_SCEN}_{FOCUS_PER}"
            f"_n*sta{em.suffix()}.csv")
        if path is None:
            logger.warning(f"[{eng}] no focus CSV for {col}; panel left empty")
            continue
        # branch on the file's ACTUAL schema (header sniff), not on the em
        # state: a stale or mismatched file then degrades to a loud warning
        # naming the path instead of a bare pandas usecols ValueError.
        head_cols = set(pd.read_csv(path, nrows=0).columns)
        has_weighted = "weighted_mean" in head_cols
        if has_weighted != em.is_weighting():
            logger.warning(f"[{eng}] {os.path.basename(path)} schema "
                           f"({'weighted' if has_weighted else 'removal'}) "
                           f"does not match method={em.method()} — reading by "
                           "its own schema; check the glob/battery outputs")
        if has_weighted:
            # weighted focus files carry the two ensemble means explicitly
            d = pd.read_csv(path, usecols=["lon", "lat", "full_mean",
                                           "weighted_mean",
                                           "full_minus_reduced",
                                           "signal_to_noise", "significant"])
            d = d[np.isfinite(d["full_minus_reduced"])]
            d["bias"] = BIAS_SIGN * d["full_minus_reduced"]
            d["reduced"] = d["weighted_mean"]
            d["full"] = d["full_mean"]
        else:
            d = pd.read_csv(path, usecols=["lon", "lat", "n_hot", "n_retained",
                                           "hot_mean", "retained_mean",
                                           "full_minus_reduced",
                                           "signal_to_noise", "significant"])
            d = d[np.isfinite(d["full_minus_reduced"])]
            d["bias"] = BIAS_SIGN * d["full_minus_reduced"]
            n_tot = d["n_hot"] + d["n_retained"]
            d["reduced"] = d["retained_mean"]
            d["full"] = (d["n_hot"] * d["hot_mean"]
                         + d["n_retained"] * d["retained_mean"]) / n_tot
        out[eng] = d
    return out


def _extra_atlas_stats(col, label, data):
    """Per-metric caption statistics, uncapped. One row per engine."""
    rows = []
    for eng, d in data.items():
        sig = d[d["significant"]]
        rows.append({
            "metric": col, "label": label, "engine": eng,
            "n": int(len(d)),
            "median_bias_pp": float(d["bias"].median()),
            "mean_bias_pp": float(d["bias"].mean()),
            "pct_significant": float(100.0 * len(sig) / len(d)) if len(d) else np.nan,
            "median_bias_where_sig_pp": float(sig["bias"].median()) if len(sig) else np.nan,
            "pct_positive_where_sig": float(100.0 * (sig["bias"] > 0).mean()) if len(sig) else np.nan,
            "median_signal_to_noise": float(d["signal_to_noise"].median()),
            "p98_abs_bias_pp": float(np.nanpercentile(np.abs(d["bias"]), 98)),
        })
    return rows


def bias_atlas_extra():
    """Supplementary S19-S21: two-engine bias maps for the nine Table-S6
    streamflow metrics that the main text does not map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_stats = []
    for stem, suptitle, rows in EXTRA_ATLASES:
        data = {col: _load_metric(col) for col, _lab in rows}
        if not any(data.values()):
            logger.error(f"atlas_extra[{stem}]: no focus CSVs found; skipping")
            continue

        # 3 rows x 2 engine columns; height scaled from the 4-row main atlas
        # (10 x 11) so the ~2:1 Robinson maps keep the same aspect and the
        # per-row title gap is unchanged.
        fig = plt.figure(figsize=(10, 8.4))
        gs = fig.add_gridspec(len(rows), 2, hspace=0.16, wspace=0.04)
        all_axes = []
        for i, (col, lab) in enumerate(rows):
            per_eng = data[col]
            all_stats += _extra_atlas_stats(col, lab, per_eng)

            # ONE colour scale per metric row, shared by the two hydrological
            # models and by nothing else: the two models must be directly
            # comparable (report.tex P21 — the LSTM dampens the bias relative
            # to HBV), which independent per-panel scales would conceal.
            lim, _p98 = _sym_limit([per_eng[e]["bias"].to_numpy(float)
                                    for e in ENGINES if e in per_eng],
                                   col, BIAS_LIM_CAP)

            sc, row_axes = None, []
            for j, eng in enumerate(ENGINES):
                ax, tk = _global_ax(fig, gs[i, j])
                row_axes.append(ax)
                all_axes.append(ax)
                if eng not in per_eng:
                    ax.set_title(f"{eng.upper()}: {lab} (missing)", fontsize=8)
                    continue
                d = per_eng[eng]
                frac = 100.0 * d["significant"].sum() / len(d) if len(d) else 0.0
                sc = _map_scatter(
                    ax, tk, d, "bias", lim,
                    f"{eng.upper()}: {lab}  ({frac:.0f}% significant)",
                    mask=True)
            if sc is not None:
                fig.colorbar(sc, ax=row_axes, orientation="vertical",
                             fraction=0.025, pad=0.02, shrink=0.7,
                             extend="both", label=BIAS_LABEL)

        # x just outside the axes so the letter clears the centred title.
        add_panel_letters(all_axes, x=-0.04)
        fig.suptitle(f"{suptitle}: {_XHOT} bias, grey = not significant "
                     f"({FOCUS_SCEN} {plabel(FOCUS_PER)})", y=0.995)
        n = max((len(d) for per_eng in data.values()
                 for d in per_eng.values()), default=0)
        os.makedirs(HYDRO_MAPS_DIR, exist_ok=True)
        out = em.tag(os.path.join(HYDRO_MAPS_DIR,
                                  f"too_hot_bias_atlas_{stem}_{FOCUS_SCEN}_"
                                  f"{FOCUS_PER}_n{n}sta.png"))
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {out}")

    if not all_stats:
        sys.exit("atlas_extra: no metric produced statistics — are the focus "
                 "CSVs present?")
    stats = pd.DataFrame(all_stats)
    os.makedirs(CACHE_DIR, exist_ok=True)
    s_csv = em.tag(os.path.join(
        CACHE_DIR, f"atlas_extra_stats_{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    stats.to_csv(s_csv, index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print("\n== atlas_extra per-metric statistics (uncapped) ==")
        print(stats.drop(columns=["label"]).to_string(index=False))
    print(f"\nwritten -> {s_csv}")
    print("\nReading guide: quote median_bias_pp and pct_significant in the "
          "captions. Q50 / AnnMin / FDCslope / Q-over-P are relative changes "
          "on near-zero baselines (report.tex P20) — if p98_abs_bias_pp is far "
          "above the 40 pp display cap for those rows, the caption must say "
          "the sign pattern is the information, not the magnitude.")


# =============================================================================
# STEP 5c — change_maps (supplementary: the PROJECTED CHANGE, full vs reduced)
# =============================================================================
# Answers the second of Prof. Beck's 10 Jul requests ("maps for all metrics of
# the projected change"); atlas / atlas_extra answer the first (the full -
# reduced difference). The two families are deliberately disjoint: NO
# difference panel appears here, because the difference is already mapped for
# all 14 metrics — five in the main text (Fig. 4d,e for RL20; Fig. 6 for
# AnnMax, Q99, Q05, mean runoff) and nine in Supplementary Figs. S19-S21 —
# and there with the significance mask that an unmasked difference panel would
# lack.
#
# Layout, one figure per CHANGE_METRICS_PER_FIG metrics: columns are the two
# hydrological models, and each metric occupies two rows, full above reduced.
# Sample: 2081-2100 against the 2001-2020 reference, SSP5-8.5, matching Fig. 2.
BASE_PER = "2001-2020"
CHANGE_METRICS_PER_FIG = 2

# The 14 curated indices of Supplementary Table S6, in that table's order; the
# figures are consecutive chunks of this one list, so adding or reordering a
# metric needs no other edit. `key` is the CLI selector.
METRICS_S6 = [
    # The four percentile entries take their title from
    # figstyle.PERCENTILE_LABELS via _atlas_label, exactly as ATLAS_INDICES
    # does: a bare "Q95" collides with the flow-duration convention, in which
    # Q95 is a LOW flow, whereas here q95 = quantile(0.95), a high flow. The
    # other ten indices never carried that ambiguity, are outside the table on
    # purpose, and keep the wording the supplement captions already use --
    # "mean annual runoff", not figstyle's shorter "Mean runoff".
    ("qmean",        "delta_rel_qmean",            "Mean annual runoff"),
    ("q50",          "delta_rel_q50",
     _atlas_label("delta_rel_q50", "Median flow (Q50)")),
    ("runoff_ratio", "delta_rel_runoff_ratio",     "Runoff ratio (Q/P)"),
    ("q95",          "delta_rel_q95",
     _atlas_label("delta_rel_q95", "High flow (Q95)")),
    ("q99",          "delta_rel_q99",
     _atlas_label("delta_rel_q99", "Very high flow (Q99)")),
    ("annmax",       "delta_rel_mean_annual_max",  "Mean annual maximum (AnnMax)"),
    ("rl10",         "delta_rel_flood_q10",        "10-year flood (RL10)"),
    ("rl20",         "delta_rel_flood_q20",        "20-year flood (RL20)"),
    ("rl50",         "delta_rel_flood_q50",        "50-year flood (RL50)"),
    ("q05",          "delta_rel_q05",
     _atlas_label("delta_rel_q05", "Low flow (Q05)")),
    ("annmin",       "delta_rel_mean_annual_min",  "Mean annual minimum (AnnMin)"),
    ("qstd",         "delta_rel_qstd",             "Flow SD"),
    ("fdcslope",     "delta_rel_slope_fdc",        "FDC slope"),
    ("flashrb",      "delta_rel_flashiness_rb",    "Flashiness (FlashRB)"),
]
# Row titles of the change maps. The second row is the WEIGHTED ensemble under
# method=weighting -- the cache column keeps the name "reduced" only so
# downstream code works unchanged -- so the title is sourced from _SECOND
# rather than hard-coding "reduced ensemble", which named a treatment that was
# not applied. BIAS_LABEL and SIGNCONV above already read from _SECOND.
ENSEMBLES = (("full", "full ensemble"), ("reduced", f"{_SECOND} ensemble"))


def _select_metrics(keys=None):
    """METRICS_S6 filtered to `keys` (a list of CLI selectors), order
    preserved. None selects all. Unknown keys are a hard error rather than a
    silent empty figure."""
    if not keys:
        return list(METRICS_S6)
    known = {k for k, _c, _l in METRICS_S6}
    unknown = [k for k in keys if k not in known]
    if unknown:
        sys.exit(f"change_maps: unknown metric key(s) {unknown}. "
                 f"Known: {sorted(known)}")
    return [m for m in METRICS_S6 if m[0] in set(keys)]


def _change_stats(col, label, per_eng):
    """Per-metric caption statistics for the projected change, uncapped."""
    rows = []
    for eng, d in per_eng.items():
        rows.append({
            "metric": col, "label": label, "engine": eng, "n": int(len(d)),
            "median_full_pct": float(d["full"].median()),
            "median_reduced_pct": float(d["reduced"].median()),
            "mean_full_pct": float(d["full"].mean()),
            "mean_reduced_pct": float(d["reduced"].mean()),
            "median_bias_pp": float(d["bias"].median()),
            "p98_abs_change_pct": float(np.nanpercentile(
                np.abs(np.r_[d["full"], d["reduced"]]), 98)),
        })
    return rows


def projected_change_maps(keys=None, per_fig=CHANGE_METRICS_PER_FIG):
    """Supplementary projected-change maps: full and reduced ensembles, two
    hydrological models, for the Table-S6 indices selected by `keys`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _select_metrics(keys)
    chunks = [metrics[i:i + per_fig] for i in range(0, len(metrics), per_fig)]
    logger.info(f"change_maps: {len(metrics)} metrics -> {len(chunks)} figures")

    all_stats = []
    for chunk in chunks:
        data = {col: _load_metric(col) for _k, col, _lab in chunk}
        if not any(data.values()):
            logger.error(f"change_maps: no focus CSVs for "
                         f"{[k for k, _c, _l in chunk]}; skipping figure")
            continue

        # 2 rows per metric (full above reduced) x 2 engine columns. The height
        # per row matches the 4-row main atlas so the ~2:1 Robinson maps keep
        # their aspect and every title clears the bold panel letter.
        n_rows = 2 * len(chunk)
        fig = plt.figure(figsize=(10, 2.75 * n_rows))
        gs = fig.add_gridspec(n_rows, 2, hspace=0.16, wspace=0.04)
        all_axes = []
        for i, (_key, col, lab) in enumerate(chunk):
            per_eng = data[col]
            all_stats += _change_stats(col, lab, per_eng)

            # ONE colour scale for all four panels of a metric: the figure
            # exists to compare the screened against the unscreened
            # projection, and separate scales would make two different
            # projections look identical. Pooled over both ensembles and both
            # hydrological models for exactly that reason.
            lim, p98 = _sym_limit(
                [per_eng[e][ens].to_numpy(float)
                 for e in ENGINES if e in per_eng for ens, _t in ENSEMBLES],
                col, CHANGE_LIM_CAP)
            logger.info(f"{col}: colour limit +/-{lim:.1f} % "
                        f"(uncapped p98 {p98:.1f} %)")

            sc, block_axes = None, []
            for r, (ens, ens_title) in enumerate(ENSEMBLES):
                for j, eng in enumerate(ENGINES):
                    ax, tk = _global_ax(fig, gs[2 * i + r, j])
                    block_axes.append(ax)
                    all_axes.append(ax)
                    if eng not in per_eng:
                        ax.set_title(f"{eng.upper()}: {lab} (missing)",
                                     fontsize=8)
                        continue
                    # No significance mask here: the cached test compares the
                    # two ensemble COMPOSITIONS, not a change against zero.
                    sc = _map_scatter(ax, tk, per_eng[eng], ens, lim,
                                      f"{eng.upper()}: {lab}, {ens_title}")
            if sc is not None:
                fig.colorbar(sc, ax=block_axes, orientation="vertical",
                             fraction=0.025, pad=0.02, shrink=0.7,
                             extend="both", label="projected change (%)")

        add_panel_letters(all_axes, x=-0.04)
        # "full versus {_SECOND}" for the same reason the row titles are: under
        # method=weighting the second ensemble is weighted, not reduced. The
        # supplement captions already read "full versus weighted".
        fig.suptitle(f"Projected change, {plabel(FOCUS_PER)} versus "
                     f"{plabel(BASE_PER)} "
                     f"({FOCUS_SCEN}): full versus {_SECOND} ensemble",
                     y=0.995)
        n = max((len(d) for per_eng in data.values()
                 for d in per_eng.values()), default=0)
        stem = "_".join(k for k, _c, _l in chunk)
        os.makedirs(HYDRO_MAPS_DIR, exist_ok=True)
        out = em.tag(os.path.join(HYDRO_MAPS_DIR,
                                  f"projected_change_maps_{stem}_{FOCUS_SCEN}_"
                                  f"{FOCUS_PER}_n{n}sta.png"))
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {out}")

    if not all_stats:
        sys.exit("change_maps: no metric produced statistics — are the focus "
                 "CSVs present?")
    stats = pd.DataFrame(all_stats)
    os.makedirs(CACHE_DIR, exist_ok=True)
    s_csv = em.tag(os.path.join(
        CACHE_DIR, f"change_maps_stats_{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    stats.to_csv(s_csv, index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print("\n== change_maps per-metric statistics (uncapped) ==")
        print(stats.drop(columns=["label"]).to_string(index=False))
    print(f"\nwritten -> {s_csv}")
    print("\nReading guide: median_full_pct and median_reduced_pct are the "
          "projected changes an unscreened and a screened assessment would "
          "report; their difference is the bias mapped in Fig. 6 and "
          "Supplementary Figs. S19-S21. Where the two medians are close, the "
          "two maps will look identical — that is the masking result, not a "
          "plotting fault.")


# -----------------------------------------------------------------------------
# annmax_maps -- Fig. 3c,e rendered as maps
# -----------------------------------------------------------------------------
# Fig. 3c,e (hydro_indices_change.fig3_combined) plot the ACROSS-CATCHMENT
# DISTRIBUTION of the per-catchment ensemble-mean AnnMax change, full against
# reduced, one panel per hydrological model. The mapped quantity here is that
# same per-catchment scalar: _load_metric's 'full' and 'reduced' columns are
# the one-model-one-vote ensemble means, identical to _station_means() in
# hydro_indices_change.py. Nothing is recomputed; only the display changes
# from a density on a value axis to a value on a map.
#
# What a map cannot carry is the overlap of the two distributions, which is
# the point of the density panels. Each map therefore keeps a small inset of
# its own across-catchment distribution on a colour scale and an x-range
# shared by all four panels, so the density reading survives the transform.
#
# Kept separate from projected_change_maps() on purpose: that step rewrites
# fig_data/change_maps_stats_*.csv, the provenance file for the Supplementary
# projected-change captions, and running it for one metric would truncate it.
ANNMAX_COL   = "delta_rel_mean_annual_max"
ANNMAX_LABEL = "Mean annual maximum daily flow (AnnMax)"
ANNMAX_ENS_C = {"full": "firebrick", "reduced": "steelblue"}   # as Fig. 3b-e
ANNMAX_INSET_PCT = (1.0, 99.0)     # SPREAD_CLIP_PCT of the density panels
# Display cap on the colour scale. The uncapped p98 of |change| is ~118 %,
# but the 10th-90th percentile of the mapped values spans about -24 to +58 %,
# so an uncapped scale spends its range on a thin tail and leaves the bulk of
# the catchments in one pale band. Capped LOCALLY rather than in
# CHANGE_LIM_CAP: that dict is shared with projected_change_maps(), and adding
# AnnMax to it would silently rescale Supplementary Fig. chgq99annmax whose
# caption declares no cap. The colorbar carries extend='both' and the fraction
# of catchments beyond the cap is written to the stats CSV.
ANNMAX_LIM_CAP = 60.0


def _annmax_inset(ax, vals, colour, xlim, fs=7.5):
    """Across-catchment density of the mapped values, inset in a map panel.

    Reproduces the Fig. 3c,e reading at reduced size: filled histogram over a
    range shared by all panels, a dashed line at the across-catchment mean and
    a solid reference at zero change. Drawn defensively -- an inset that fails
    on a projected axes must not cost the figure."""
    import numpy as np
    try:
        axin = ax.inset_axes([0.015, 0.06, 0.27, 0.21])
    except Exception as exc:                      # pragma: no cover
        logger.warning(f"annmax_maps: inset unavailable ({exc}); skipped")
        return
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = xlim
    h, edg = np.histogram(v, bins=np.linspace(lo, hi, 60), range=(lo, hi),
                          density=True)
    x = 0.5 * (edg[:-1] + edg[1:])
    axin.fill_between(x, 0, h, color=colour, alpha=0.55, linewidth=0)
    axin.plot(x, h, color=colour, lw=0.9)
    axin.axvline(0.0, color="0.45", lw=0.6, zorder=0)
    axin.axvline(float(np.mean(v)), color=colour, ls="--", lw=1.0)
    axin.set_xlim(lo, hi)
    axin.set_ylim(bottom=0.0)
    axin.set_yticks([])
    axin.set_xticks([lo, 0.0, hi])
    axin.set_xticklabels([f"{lo:.0f}", "0", f"{hi:.0f}"], fontsize=fs)
    axin.tick_params(axis="x", length=2, pad=1)
    for side in ("top", "right", "left"):
        axin.spines[side].set_visible(False)
    axin.spines["bottom"].set_linewidth(0.6)
    axin.patch.set_alpha(0.75)
    axin.set_xlabel("Change (%)", fontsize=fs, labelpad=1)


def annmax_projection_maps():
    """Standalone 2x2 map of the projected AnnMax change: rows are the full and
    the reduced ensemble, columns the two hydrological models.

    One colour scale over all four panels, pooled across both ensembles and
    both hydrological models, because the figure exists to compare a screened
    against an unscreened projection and per-panel scaling would make two
    different projections look alike. No significance mask: the cached Welch
    test compares the two ensemble compositions, not a change against zero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_eng = _load_metric(ANNMAX_COL)
    engines = [e for e in ENGINES if e in per_eng]
    if not engines:
        sys.exit(f"annmax_maps: no focus CSV for {ANNMAX_COL} at "
                 f"{FOCUS_SCEN}/{FOCUS_PER}; run the significance step first.")

    pooled = [per_eng[e][ens].to_numpy(float)
              for e in engines for ens, _t in ENSEMBLES]
    lim, p98 = _sym_limit(pooled, ANNMAX_COL, {ANNMAX_COL: ANNMAX_LIM_CAP})
    logger.info(f"annmax_maps: colour limit +/-{lim:.1f} % "
                f"(uncapped p98 {p98:.1f} %)")

    # Inset x-range shared by the four panels, from the FULL ensemble as in
    # _spread_density: a per-panel range would make one distribution look
    # wider than another purely through its axis.
    ref = np.concatenate([per_eng[e]["full"].to_numpy(float)
                          for e in engines])
    ref = ref[np.isfinite(ref)]
    ilo, ihi = np.percentile(ref, list(ANNMAX_INSET_PCT))

    fig = plt.figure(figsize=(11.0, 6.8))
    gs = fig.add_gridspec(len(ENSEMBLES), len(engines),
                          hspace=0.14, wspace=0.04)
    axes, sc, rows = [], None, []
    for r, (ens, ens_title) in enumerate(ENSEMBLES):
        for j, eng in enumerate(engines):
            ax, tk = _global_ax(fig, gs[r, j])
            axes.append(ax)
            d = per_eng[eng]
            sc = _map_scatter(ax, tk, d, ens, lim,
                              f"{eng.upper()}: {ens_title.capitalize()}")
            ax.set_title(f"{eng.upper()}: {ens_title.capitalize()}",
                         fontsize=10)
            _annmax_inset(ax, d[ens].to_numpy(float), ANNMAX_ENS_C[ens],
                          (ilo, ihi))
            v = d[ens].to_numpy(float)
            v = v[np.isfinite(v)]
            rows.append({"engine": eng, "ensemble": ens, "n": int(v.size),
                         "mean_pct": float(np.mean(v)),
                         "median_pct": float(np.median(v)),
                         "p10_pct": float(np.percentile(v, 10)),
                         "p90_pct": float(np.percentile(v, 90)),
                         "colour_cap_pct": float(lim),
                         "frac_beyond_cap_pct": float(
                             100.0 * np.mean(np.abs(v) > lim))})

    cb = fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.022,
                      pad=0.02, shrink=0.72, extend="both")
    cb.set_label("Projected change in AnnMax (%)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    add_panel_letters(axes, x=-0.04)
    scen_lbl = FOCUS_SCEN.upper().replace("SSP585", "SSP5-8.5")
    fig.suptitle(f"Projected change in the {ANNMAX_LABEL[0].lower()}"
                 f"{ANNMAX_LABEL[1:]}, {plabel(FOCUS_PER)} versus "
                 f"{plabel(BASE_PER)} "
                 f"({scen_lbl})", y=0.97, fontsize=11)

    os.makedirs(HYDRO_MAPS_DIR, exist_ok=True)
    out = em.tag(os.path.join(
        HYDRO_MAPS_DIR,
        f"annmax_projection_maps_{FOCUS_SCEN}_{FOCUS_PER}"
        f"_n{max(r_['n'] for r_ in rows)}sta.png"))
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")

    stats = pd.DataFrame(rows)
    os.makedirs(CACHE_DIR, exist_ok=True)
    s_csv = em.tag(os.path.join(
        CACHE_DIR, f"annmax_projection_maps_stats_{FOCUS_SCEN}_"
                   f"{FOCUS_PER}.csv"))
    stats.to_csv(s_csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print("\n== annmax_maps per-panel across-catchment statistics ==")
        print(stats.to_string(index=False))
    print(f"\nwritten -> {s_csv}")


# =============================================================================
# STEP 3 — fdr (multiple-testing / field significance)
# =============================================================================
ALPHA_RAW = 0.05
ALPHA_FDR = (0.05, 0.10)       # 0.10 = Wilks (2016) alpha_FDR = 2*alpha_global
def _fname_re():
    """Focus-CSV filename regex, built lazily so em.suffix() (set from the
    CLI in __main__, after import) is honoured."""
    return re.compile(rf"^signif_(?P<col>.+?)_{FOCUS_SCEN}_{FOCUS_PER}"
                      rf"_n(?P<n>\d+)sta{re.escape(em.suffix())}\.csv$")


def _bh_reject(p, q):
    """Benjamini-Hochberg step-up at level q on finite p-values.
    Returns (n_rejected, largest rejected p; 0.0 when none)."""
    p = np.sort(np.asarray(p, float)[np.isfinite(p)])
    m = len(p)
    if m == 0:
        return 0, 0.0
    k = np.arange(1, m + 1)
    ok = p <= q * k / m
    if not ok.any():
        return 0, 0.0
    kmax = int(k[ok].max())
    return kmax, float(p[kmax - 1])


def _discover_focus_csvs():
    """Newest focus CSV per (metric, engine) at the focus scenario/period.
    The engine ('hbv'/'lstm') is a suffix of the metric part of the filename;
    climate columns (precip_*, temp_*) carry no engine."""
    latest = {}
    for f in glob.glob(os.path.join(
            CACHE_DIR,
            f"signif_*_{FOCUS_SCEN}_{FOCUS_PER}_n*sta{em.suffix()}.csv")):
        mt = _fname_re().match(os.path.basename(f))
        if not mt:
            continue
        col, engine = mt.group("col"), ""
        for eng in ENGINES:
            if col.endswith(f"_{eng}"):
                col, engine = col[: -len(eng) - 1], eng
                break
        key = (col, engine)
        if key not in latest or os.path.getmtime(f) > os.path.getmtime(latest[key]):
            latest[key] = f
    return latest


def fdr_field_significance():
    """BH-FDR + Wilks field significance for every cached significance cell."""
    files = _discover_focus_csvs()
    if not files:
        sys.exit(f"no signif_*_{FOCUS_SCEN}_{FOCUS_PER}_n*sta{em.suffix()}.csv "
                 "found in "
                 f"{CACHE_DIR} — run the significance passes first.")
    print(f"FDR / field significance | {FOCUS_SCEN} {FOCUS_PER} | "
          f"{len(files)} (metric, engine) cells from\n  {CACHE_DIR}\n")

    rows = []
    for (col, engine), f in sorted(files.items()):
        d = pd.read_csv(f, usecols=lambda c: c in ("p", "significant"))
        p = d["p"].to_numpy(float)
        n = int(np.isfinite(p).sum())
        if n == 0:
            continue
        rec = {"metric": col, "engine": engine or "-", "n_tests": n,
               f"frac_raw_p<{ALPHA_RAW}": float(np.mean(p[np.isfinite(p)] < ALPHA_RAW)),
               "source": os.path.basename(f)}
        for q in ALPHA_FDR:
            n_rej, thr = _bh_reject(p, q)
            rec[f"frac_BH_q{q:.2f}"] = n_rej / n
            rec[f"p_thresh_BH_q{q:.2f}"] = thr
        # field significance per Wilks (2016): >=1 BH rejection at q = 2*alpha
        rec["field_significant"] = bool(rec[f"frac_BH_q{ALPHA_FDR[1]:.2f}"] > 0)
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["engine", "metric"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_csv = em.tag(os.path.join(
        CACHE_DIR, f"fdr_field_significance_{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    out.to_csv(out_csv, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(out.drop(columns="source").to_string(index=False))
    print(f"\nwritten -> {out_csv}")
    print("\nReading guide: frac_raw is the pointwise p<0.05 fraction quoted "
          "so far; frac_BH_q0.05 is the FDR-controlled fraction; frac_BH_q0.10 "
          "is the Wilks (2016) alpha_FDR = 2*alpha convention for spatially "
          "correlated fields — quote frac_raw alongside frac_BH_q0.10 in the "
          "paper and state field significance from the flag.")


# =============================================================================
# STEP 4 — dipole (nival/pluvial peak-flow bias dipole, P18b)
# =============================================================================
FLOW_COL = "delta_rel_mean_annual_max"        # peak flow (mean annual maximum)
# descriptive regime classes (Berghuijs et al. 2014-style snow-fraction split;
# adjust freely — the inference rests on the continuous statistics, not these):
RAIN_MAX, SNOW_MIN = 0.10, 0.50


def _load_static_attrs():
    """station -> aridity, snowfrac, kgz (mirrors the analyzer's loader)."""
    st = pd.read_csv(STATIC_CSV, comment="#", index_col=0).rename_axis("station")
    st = st.reset_index()
    out = pd.DataFrame({"station": st["station"].astype(str)})
    out["aridity"]  = pd.to_numeric(st["aridity_index_mswx_mswepv3"], errors="coerce")
    out["snowfrac"] = pd.to_numeric(st["WorldClim_V2_snowfall_fraction_P.mat"],
                                    errors="coerce")
    out["kgz"]      = st["KGZ_major"].astype(str).str.strip()
    return out


def _partial_spearman(x, y, z):
    """Spearman partial correlation of x, y controlling z (rank residuals) —
    identical to the transmission scan's implementation."""
    from scipy import stats as sps
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    n = int(len(x))
    if n < 10:
        return (np.nan, np.nan, n)
    rx, ry, rz = sps.rankdata(x), sps.rankdata(y), sps.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = sps.pearsonr(ex, ey)
    return (float(r), float(p), n)


def _wilson(k, n, z=1.96):
    """Wilson 95% score interval for a binomial fraction."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    hw  = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (ctr - hw, ctr + hw)


def _dipole_engine(eng, attrs):
    from scipy import stats as sps
    path = _latest_focus_csv(
        f"signif_{FLOW_COL}_{eng}_{FOCUS_SCEN}_{FOCUS_PER}"
        f"_n*sta{em.suffix()}.csv")
    if path is None:
        print(f"[{eng}] no focus CSV for {FLOW_COL} — run the flow "
              "significance pass first; skipping engine.")
        return None, None, None
    d = pd.read_csv(path)
    d["station"] = d["station"].astype(str)
    d = d.merge(attrs, on="station", how="left")
    d["abslat"] = d["lat"].abs()
    d = d[np.isfinite(d["full_minus_reduced"]) & np.isfinite(d["snowfrac"])].copy()
    # plotted/tested orientation (reduced − full when BIAS_SIGN = -1): the
    # spearman/partial correlations and frac_negative below flip with it,
    # while every p-value is invariant. CSV column stays full - reduced.
    d["bias"] = BIAS_SIGN * d["full_minus_reduced"]         # percent (see analyzer)
    print(f"[{eng}] {os.path.basename(path)} -> {len(d)} catchments with "
          f"finite bias + snow fraction")

    summary_rows, bin_rows = [], []
    for subset, label in ((d, "all"), (d[d["significant"]], "significant-only")):
        n = len(subset)
        if n < 50:
            continue
        r_raw, p_raw = sps.spearmanr(subset["snowfrac"], subset["bias"],
                                     nan_policy="omit")[:2]
        r_par, p_par, n_par = _partial_spearman(subset["snowfrac"],
                                                subset["bias"], subset["abslat"])
        rain = subset.loc[subset["snowfrac"] <= RAIN_MAX, "bias"]
        snow = subset.loc[subset["snowfrac"] >= SNOW_MIN, "bias"]
        if len(rain) >= 20 and len(snow) >= 20:
            _u, p_mwu = sps.mannwhitneyu(snow, rain, alternative="two-sided")
        else:
            p_mwu = np.nan
        summary_rows.append({
            "engine": eng, "subset": label, "n": n,
            "spearman_r": float(r_raw), "spearman_p": float(p_raw),
            "partial_r_ctrl_abslat": r_par, "partial_p": p_par, "partial_n": n_par,
            f"median_bias_rain(snowfrac<={RAIN_MAX})": float(rain.median()) if len(rain) else np.nan,
            f"median_bias_snow(snowfrac>={SNOW_MIN})": float(snow.median()) if len(snow) else np.nan,
            "n_rain": int(len(rain)), "n_snow": int(len(snow)),
            "mannwhitney_p": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
            "frac_negative_rain": float((rain < 0).mean()) if len(rain) else np.nan,
            "frac_negative_snow": float((snow < 0).mean()) if len(snow) else np.nan,
        })

    # decile bins of snow fraction (all catchments): dose-response curve
    dd = d.dropna(subset=["snowfrac", "bias"])
    edges = np.unique(np.nanpercentile(dd["snowfrac"], np.arange(0, 101, 10)))
    if len(edges) >= 3:
        b = np.clip(np.digitize(dd["snowfrac"].to_numpy(float), edges[1:-1]),
                    0, len(edges) - 2)
        for k in range(len(edges) - 1):
            sel = dd[b == k]
            if len(sel) < 30:
                continue
            kneg = int((sel["bias"] < 0).sum())
            lo, hi = _wilson(kneg, len(sel))
            bin_rows.append({
                "engine": eng, "bin": k, "n": int(len(sel)),
                "snowfrac_median": float(sel["snowfrac"].median()),
                "bias_median": float(sel["bias"].median()),
                "frac_negative": kneg / len(sel),
                "frac_negative_lo95": lo, "frac_negative_hi95": hi,
            })

    # Koeppen major zones (descriptive)
    for z in ("A", "B", "C", "D", "E"):
        v = d.loc[d["kgz"] == z, "bias"]
        if len(v) < 30:
            continue
        summary_rows.append({
            "engine": eng, "subset": f"koeppen_{z}", "n": int(len(v)),
            "median_bias_zone": float(v.median()),
            "frac_negative_zone": float((v < 0).mean()),
        })
    return summary_rows, bin_rows, len(d)


def _dipole_figure(bins, n_by_eng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for eng in ENGINES:
        bb = bins[bins["engine"] == eng].sort_values("snowfrac_median")
        if bb.empty:
            continue
        axA.plot(bb["snowfrac_median"], bb["bias_median"], marker="o",
                 color=ECOL[eng], label=eng.upper())
        axB.plot(bb["snowfrac_median"], bb["frac_negative"] * 100, marker="o",
                 color=ECOL[eng], label=eng.upper())
        axB.fill_between(bb["snowfrac_median"],
                         bb["frac_negative_lo95"] * 100,
                         bb["frac_negative_hi95"] * 100,
                         color=ECOL[eng], alpha=0.18, lw=0)
    axA.axhline(0, color="0.4", lw=0.8, ls="--")
    axA.set_xlabel("snowfall fraction of precipitation")
    axA.set_ylabel(f"median AnnMax {_XHOT} bias ({SIGNCONV}, %)")
    axA.set_title("bias vs snow fraction (decile medians)")
    axB.axhline(50, color="0.4", lw=0.8, ls="--")
    axB.set_xlabel("snowfall fraction of precipitation")
    axB.set_ylabel("catchments with negative bias (%)")
    axB.set_title("sign of the bias (Wilson 95% band)")
    for ax in (axA, axB):
        ax.grid(color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    add_panel_letters([axA, axB])
    n = max(n_by_eng.values())
    fig.suptitle(f"Nival/pluvial dipole of the mean annual maximum {_XHOT} bias "
                 f"({FOCUS_SCEN} {plabel(FOCUS_PER)})", y=1.02)
    os.makedirs(HYDRO_DIR, exist_ok=True)
    out = em.tag(os.path.join(HYDRO_DIR,
                              f"dipole_peakflow_snowfrac_{FOCUS_SCEN}_"
                              f"{FOCUS_PER}_n{n}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


def quantify_peakflow_dipole():
    """Test the P18b nival/pluvial dipole; CSVs + figure + decision rule."""
    attrs = _load_static_attrs()
    print(f"static attributes for {len(attrs)} stations from {STATIC_CSV}\n")
    all_summary, all_bins, n_by_eng = [], [], {}
    for eng in ENGINES:
        s, b, n = _dipole_engine(eng, attrs)
        if s:
            all_summary += s
        if b:
            all_bins += b
        if n:
            n_by_eng[eng] = n
    if not all_summary:
        sys.exit("no engine produced results — are the focus CSVs present?")

    summary = pd.DataFrame(all_summary)
    bins    = pd.DataFrame(all_bins)
    os.makedirs(CACHE_DIR, exist_ok=True)
    s_csv = em.tag(os.path.join(
        CACHE_DIR, f"dipole_peakflow_{FOCUS_SCEN}_{FOCUS_PER}_summary.csv"))
    b_csv = em.tag(os.path.join(
        CACHE_DIR, f"dipole_peakflow_{FOCUS_SCEN}_{FOCUS_PER}_bins.csv"))
    summary.to_csv(s_csv, index=False)
    bins.to_csv(b_csv, index=False)

    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print("\n== dipole summary ==")
        print(summary.to_string(index=False))
    print(f"\nwritten -> {s_csv}\nwritten -> {b_csv}")
    if not bins.empty:
        _dipole_figure(bins, n_by_eng)
    print("\nReading guide (bias plotted as BIAS_SIGN * full_minus_reduced): "
          "the dipole is a TESTED result if (i) spearman_r and "
          "partial_r_ctrl_abslat carry the nival-reversal sign with small p "
          "for BOTH engines (negative under full - reduced, positive under "
          "reduced - full), and (ii) the frac_negative curve in panel (b) "
          "crosses 50% toward high snow fraction (rising under "
          "full - reduced, falling under reduced - full). If the partial "
          "correlation collapses toward 0, the map pattern is a latitude "
          "artefact and P18b must stay out of the manuscript claims.")


# =============================================================================
# dispatcher
# =============================================================================
STEPS = ("baseline_cache", "gev", "gev_map", "fig4", "rl50", "fdr", "dipole",
         "atlas", "atlas_extra", "change_maps", "annmax_maps", "all")

if __name__ == "__main__":
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment(logger)
    # method=removal|weighting and weights=bma|kde are consumed first; the
    # steps below index sys.argv directly, so it is rewritten in place.
    sys.argv = sys.argv[:1] + em.reject_cli_tokens(sys.argv[1:])
    logger.info(f"ensemble method = {em.method()}"
                + (f" [{em.variant()}], suffix '{em.suffix()}'"
                   if em.is_weighting() else ""))
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step not in STEPS:
        sys.exit(f"usage: python {os.path.basename(__file__)} "
                 f"{{{'|'.join(STEPS)}}} [max_stations for baseline_cache]")
    if step in ("baseline_cache", "all"):
        cap = int(sys.argv[2]) if step == "baseline_cache" and len(sys.argv) > 2 else None
        build_flood_baseline_cache(max_stations=cap)
    if step in ("gev", "all"):
        gev_return_period_translation()
    if step in ("gev_map", "all"):
        gev_translation_maps()
    if step in ("fig4", "all"):
        gev_translation_figure4()
    if step in ("rl50", "all"):
        rl50_check()
    if step in ("fdr", "all"):
        fdr_field_significance()
    if step in ("dipole", "all"):
        quantify_peakflow_dipole()
    if step in ("atlas", "all"):
        bias_atlas()
    if step in ("atlas_extra", "all"):
        bias_atlas_extra()
    if step in ("change_maps", "all"):
        # optional 2nd arg: comma-separated METRICS_S6 keys, e.g. "annmax,rl20"
        sel = (sys.argv[2].split(",")
               if step == "change_maps" and len(sys.argv) > 2 else None)
        projected_change_maps(keys=sel)
    if step in ("annmax_maps", "all"):
        annmax_projection_maps()
