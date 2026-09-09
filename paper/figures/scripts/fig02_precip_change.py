"""
Figure 2 — climate-change signal, FULL vs REDUCED (too-hot-removed) ensemble.

Reproduces the map panels of Fig. 2 in the manuscript. The layout is set by
MAP_SHOW_SECOND_ENSEMBLE (see USER SETTINGS):

    False (default, 2x2 grid; column 1 precipitation change %, column 2
    temperature change °C):
      Row 1  full ensemble
      Row 2  difference (weighted − full)
    True (3x2 grid), the pre-9 Sep 2026 figure, adds the middle row:
      Row 2  weighted (or reduced) ensemble

What the "change" means here
----------------------------
It is the *forcing change that actually drives the runoff projections*. For
every (station, scenario, model, member) we take the catchment-mean qqmap_day
deltas saved in CONFIG.out_path/CONFIG.delta_method/<station>/mean_deltas.nc and apply them to the
observed reference climate EXACTLY the way apply_deltas_stn.py does:

  * precipitation : apply_qqmap_day_delta_pr() to MSWEP, then
                    precip_change_pct = (mean(future) / mean(obs) - 1) * 100
  * temperature   : apply_qqmap_day_delta_temp() to MSWX Tmin & Tmax, take
                    tas = (tmin + tmax)/2 (as build_future_data does), then
                    temp_change_degC  = mean(future_tas) - mean(obs_tas)

over the historical window (CONFIG.historical_period). This is consistent with
the hydrology, unlike a raw mean of the delta arrays.

Two-stage design (so you can tweak the model lists cheaply)
-----------------------------------------------------------
Stage 1 (expensive): loop over all stations / realizations / scenarios
        (SCENARIOS) / future periods (PERIODS), compute the per-realization
        precipitation and temperature change and cache them to a long-format
        parquet (CACHE_FPATH). Skipped if the cache exists, RECOMPUTE is
        False, it has the temperature column, and it already covers every
        scenario in SCENARIOS and period in PERIODS.
Stage 2 (cheap): load the cache, split realizations into FULL vs REDUCED
        using the editable model lists below, aggregate, and plot.

So: edit FULL_MODELS / EXCLUDED_MODELS, keep RECOMPUTE=False, and rerun — only
the aggregation + plot is redone (seconds).

Run (python only):
    python paper/figures/scripts/fig02_precip_change.py          # all stations, all figures
    python paper/figures/scripts/fig02_precip_change.py 200      # first 200 stations
    python paper/figures/scripts/fig02_precip_change.py --max-stations 200

MAX_STATIONS caps Stage 1 and is self-isolating: a capped run reads and writes
its OWN `fig02_change_max<N>.parquet` and `fig02_dryspell_max<N>.parquet`, never
the shared caches every downstream script reads, and its figures are named
`n<N>sta`. Both `_max<N>` files are test artefacts — delete them afterwards.

Selecting individual figures (Stage 2 only)
-------------------------------------------
`--only` renders a subset of this script's figures and skips the rest. The
targets are

    maps                the precip/temp map figure, one per
                        MAP_SCENARIOS x MAP_PERIODS (the slow part of Stage 2)
    model_boxplots      boxplots_by_model, one figure per MAP_SCENARIOS
    scenario_boxplots   boxplots_by_scenario
    spectrum            spectrum_plot (percentile spectrum)
    dryspell_spectrum   dryspell_spectrum_plot

    python paper/fig02_precip_change.py --only spectrum
    python paper/fig02_precip_change.py --only model_boxplots
    python paper/fig02_precip_change.py --only maps,spectrum
    python paper/fig02_precip_change.py --list

A `--only` run is PLOT-ONLY by construction: both expensive paths are refused —
Stage 1 (compute_cache) and the dry-spell scan (load_dryspell_table) — so the
per-realization parquet (CACHE_FPATH) and, for dryspell_spectrum, the dry-spell
parquet must already exist and are read, never rebuilt. A cache that is stale
(missing a scenario, period or percentile column) therefore aborts the run
rather than quietly plotting from it. Without `--only` every figure is drawn,
which is the previous behaviour and what slurm/fig02.sh and slurm/fig02_replot.sh
do.

`--recompute` forces the Stage-1 rebuild that RECOMPUTE=True does; it cannot be
combined with `--only`.
"""

import os
import sys
import glob
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap

# this file lives in paper/figures/scripts/; config.py and apply_deltas_utils.py
# live in the repo root, three dirs up, so put that on sys.path (same pattern as
# fig01_model_screening.py). fig01_model_screening.py is in the same dir so it is
# already importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import CONFIG
from apply_deltas_utils import apply_qqmap_day_delta_pr, apply_qqmap_day_delta_temp
# single source of truth for the model universe + literature ECS/TCR/trend
from fig01_model_screening import METRICS, models_in_results
# single source of truth for the corrupt/degenerate realizations to exclude
from realization_blocklist import BLOCKLIST, is_blocked as _is_blocked, apply_blocklist
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

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===========================================================================
#  USER SETTINGS  (edit these)
# ===========================================================================

# Scenarios computed & cached in Stage 1. The maps and the per-model
# boxplot are generated for each scenario in MAP_SCENARIOS; the per-scenario
# boxplot uses all SCENARIOS.
SCENARIOS     = ["ssp126", "ssp245", "ssp370", "ssp585"]
MAP_SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]    # focus scenarios for the maps

# Future periods to compute & cache in Stage 1. Defaults to every period in
# CONFIG.future_periods (e.g. ["2041-2060", "2081-2100"]); each must be a
# `period` label in mean_deltas.nc.
PERIODS = ['2041-2060', '2081-2100']
# A map figure is produced for every MAP_SCENARIOS × MAP_PERIODS combo
# (3 × 2 = 6 figures with the current settings). The per-model boxplot is
# produced once per scenario in MAP_SCENARIOS (already covers both periods).
MAP_PERIODS = list(PERIODS)

# --- model universe & the active screen ------------------------------------
# Both come from ensemble_method.py, the single source of truth for the
# treatment: edit SCREEN / METHOD / VARIANT at the top of that file to change
# what the "reduced" ensemble is. There is no command-line switch.
#   FULL_MODELS     every model in the results (one station's mean_deltas.nc,
#                   so the universe matches Fig. 1 exactly)
#   EXCLUDED_MODELS the models the active screen drops. Under SCREEN='hot' this
#                   is the too-hot set (TCR > 2.2 K); under SCREEN='both' it
#                   also carries the too-cold set (TCR < 1.4 K) -- hence
#                   "excluded" rather than "too_hot".
#   REDUCED_MODELS  FULL_MODELS minus EXCLUDED_MODELS.
# A model missing from METRICS fails loudly inside em.full_models().
FULL_MODELS_STATION   = em.FULL_MODELS_STATION
TOO_HOT_TCR_THRESHOLD = em.TCR_UPPER      # K; kept for plot labels and sweeps
FULL_MODELS     = list(em.full_models())
EXCLUDED_MODELS = em.excluded_models()
REDUCED_MODELS  = em.retained_models()
_tcr            = em.tcr                  # TCR (K) of one model, from METRICS

# Realization BLOCKLIST / is_blocked() / apply_blocklist() now live in
# realization_blocklist.py (imported above) so the SAME exclusion is shared by
# fig02, fig05, build_streamflow_cache and the significance script.

# How to combine realizations into an ensemble value per catchment:
#   "model_mean"       : mean over members within a model, then mean over
#                        models (one model = one vote; "model democracy").
#   "realization_mean" : plain mean over all realizations in the set.
AGGREGATION = "model_mean"

# First-N stations only (int) for a quick test; None = all stations.
# Also overridable from the command line: `python fig02_precip_change.py 200`
# or `--max-stations 200`. Parsed in _parse_cli(), which runs ONLY under
# `__main__` — importing this module (hydro_indices_change.py, baseline.py,
# replot drivers) never touches sys.argv.
MAX_STATIONS = None

# Recompute the per-realization cache even if it already exists.
# `--recompute` on the command line sets this for one run.
RECOMPUTE = False

# Set True by a `--only` run. It makes Stage 1 an error rather than a silent
# hour-long rebuild: a figure-subset run is by definition a replot off the
# existing cache. Enforced inside compute_cache(), so it covers BOTH ways
# main() can reach Stage 1 (cache absent, and cache present but stale/missing
# a column).
PLOT_ONLY = False

# Stage 1 parallelism: number of worker processes. Derived from the SLURM
# allocation rather than hardcoded — `#SBATCH --cpus-per-task=N` makes SLURM
# export SLURM_CPUS_PER_TASK=N, so the worker count always matches the cores
# actually granted (cpus via SLURM, not os.cpu_count()) and can't drift out of
# sync with the job script. Off SLURM (local) it falls back to os.cpu_count()-1.
# Each worker opens its own xarray handles to MSWEP/MSWX once (in
# `_worker_init`), then loops over its stations independently. Fork start
# method is used so workers inherit FULL_MODELS / METRICS / SCENARIOS
# without re-running module-import-time setup (which opens netCDF files).
NUM_WORKERS = (int(os.environ["SLURM_CPUS_PER_TASK"])
               if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit()
               else max(1, (os.cpu_count() or 1) - 1))

# horizontal / vertical gap between map panels, as a fraction of the average
# panel size (smaller = tighter). HSPACE alone does not control the visible row
# gap -- see MAP_FIG_H below.
WSPACE, HSPACE = 0.03, 0.09

# Height (in) of the 3x2 map figure; width is MAP_FIG_W. A Robinson map is
# ~1.97x wider than tall, and a GeoAxes keeps that aspect by shrinking its box,
# so every panel is drawn as a map centred in a taller cell. The dead space
# above and below each map is what reads as an oversized row gap, and it is set
# by the figure's height, NOT by HSPACE and NOT by the colourbar length
# (Colorbar `shrink` resizes only the bar's own slot, never the parent axes).
# Lowering MAP_FIG_H squeezes each cell towards the map's own height and so
# closes the gap. Do not go far below ~10: once a cell is shorter than the map
# it holds, the map becomes height-limited and shrinks sideways instead, which
# opens a gap between the COLUMNS.
MAP_FIG_W, MAP_FIG_H = 13.0, 10.8

# Draw the second-ensemble row (weighted, or reduced under removal) in the map
# figure. False gives the 2x2 layout adopted on 9 Sep 2026 after a co-author
# review: the weighted panel differs from the full panel by ~0.5 units on a
# 3-6.5 colour scale, which is not resolvable by eye, so the figure keeps one
# full-ensemble panel per variable for context and the difference panel that
# carries the result. Row order is unchanged (full on top, difference below),
# every panel keeps the colour scale, inset and title it had in the 3-row
# layout, and MAP_FIG_H is scaled by the row count so the per-row height is
# identical. Set True to restore the 3x2 figure.
MAP_SHOW_SECOND_ENSEMBLE = False

# Length of each map colourbar as a fraction of the panel group it annotates
# (matplotlib Colorbar `shrink`). 1.0 = full height. This is purely cosmetic:
# it does not move the panels, so it cannot be used to tighten the row gap.
CBAR_SHRINK = 1.0

# difference colorbars use TwoSlopeNorm (0 at the centre, asymmetric ends);
# force this many tick labels on EACH side of 0 so the two halves look
# balanced regardless of how asymmetric the data range is.
DIFF_TICKS_PER_SIDE = 3

# Gray band around 0 in the "greyified" diverging cmaps used for the
# precipitation column. All three are fractions of the FULL colorbar range,
# measured from its centre:
#   GREY_HALFWIDTH   half-width of the pure-gray core.
#   GREY_FADE_SIGMA  scale of the Gaussian tail the gray decays along outside
#                    the core. The tail has zero slope where it leaves the
#                    core and no cut-off where it ends, so the bar carries no
#                    edge at either point; the gray is at half strength
#                    ~0.83*sigma out and is imperceptible by ~1.6*sigma.
#                    Larger = longer, gentler fade, but the gray then tints a
#                    wider band of genuinely non-zero values.
#   GREY_SRC_PAD     how much of the source cmap's own near-white centre to
#                    drop before sampling. This is the main control on how
#                    ABRUPT the fade looks: the gray has to travel the whole
#                    colour distance to whatever the source offers at the end
#                    of the tail, so a large pad forces a saturated colour to
#                    appear within a short stretch of the bar. Keep it small
#                    and let the source's own centre-to-end ramp do the work.
GREY_HALFWIDTH  = 0.02
GREY_FADE_SIGMA = 0.10
GREY_SRC_PAD    = 0.06

# Background fill colour for the LAND polygons on the maps (matplotlib gray
# string, 0=black -> 1=white). Higher = lighter. Tune freely.
LAND_FACECOLOR = "0.96"

# inset Köppen-Geiger boxplot on each map: one box per KGZ_major zone,
# placed in the empty lower-left (ocean) corner of the Robinson map.
ADD_ZONE_BOXPLOTS = True
KG_ZONES = ["A", "B", "C", "D", "E"]   # tropical, arid, temperate, cold, polar
# inset position within each map axes (axes fraction: x0, y0, w, h). y0 sits
# low in the axes because the Robinson outline curves away from the axes corners
# there, leaving ocean/white space the inset can occupy without covering data.
BOX_RECT = [0.50, 0.08, 0.30, 0.30]
# tick-label size for the inset, applied to BOTH axes so they match.
INSET_TICK_FS = 9
# gap (points) between the A/B/C/D/E labels and the inset x-axis.
INSET_XTICK_PAD = 3
# opacity of the inset's background only (boxes/whiskers stay opaque):
# 0.0 = fully transparent (markers below fully visible), 1.0 = solid white.
BOX_FACE_ALPHA = 0.0

# main figure: the precip/temp map, one per MAP_SCENARIOS x MAP_PERIODS.
# The cartopy rendering is the slow part of Stage 2, which is why `--only` can
# switch it off while keeping the boxplots/spectra.
MAKE_MAPS = True

# extra figure: one box per model (distribution across catchments), 2×N_period
# grid, one per scenario in MAP_SCENARIOS.
MAKE_MODEL_BOXPLOTS = True
MODEL_BOX_SORT = "tcr"   # "tcr" = hottest model first; "full" = FULL_MODELS order
# Horizontal spacing between box centres in DATA coordinates (default was 1.0
# from integer positions). Smaller -> boxes packed more tightly with less
# empty space between them. Must stay > MODEL_BOX_W to keep the boxes apart.
MODEL_BOX_STEP = 0.55
# Box width in the same DATA coordinates. Only the MODEL_BOX_W / MODEL_BOX_STEP
# ratio matters visually (0.47 here): the axes are rescaled to the model count,
# so a smaller ratio buys panel width back and lets the tick labels and legend
# be set in a larger point size without the figure growing.
MODEL_BOX_W = 0.26

# extra figure: per-scenario full vs reduced boxes (distribution across
# catchments) + global-mean markers, 2×N_period grid, all SCENARIOS.
MAKE_SCENARIO_BOXPLOTS = True

# Extra cache columns: changes at higher percentiles of the DAILY distribution
# (in addition to the mean, which is the existing precip_change_pct /
# temp_change_degC).
#   - PRECIP percentiles are taken on all daily values (incl. dry days);
#     Q90/Q95/Q99 should be > 0 even in arid catchments, but if the obs
#     baseline percentile is 0 the change-percent is undefined and stored as
#     NaN (handled gracefully by the spectrum plot).
#   - TEMP percentiles are taken on daily mean temperature (well-defined
#     everywhere; additive change, °C).
# A different list here forces a Stage-1 recompute (handled by the cache
# guard below).
PRECIP_PERCENTILES = [90, 95, 99]
TEMP_PERCENTILES   = [5, 50, 90, 95, 99]

# Subset of SCENARIOS shown in the spectrum plot (SCENARIOS stays the cache's
# scope; the spectrum just filters it down for readability). The spectrum
# drops SSP1-1.9 because it has the smallest ensemble and the lines tend to
# clutter the top corner without adding signal.
SPECTRUM_SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]

# SIGN CONVENTION (2 Aug 2026, matching gev_fdr_dipole.py). The manuscript
# reports every ensemble difference as reduced − full: negative values mean
# the too-hot models inflate the projection. BIAS_SIGN is applied at the
# single point where the plotted difference is formed (ensemble_table's
# "diff" column), so the map difference panels, the percentile spectrum and
# the dry-spell spectrum all flip together. The Stage-1 parquet cache stores
# per-realization changes and is orientation-free.
BIAS_SIGN = -1.0
# Name of the non-full ensemble INSIDE the sign convention. Under
# METHOD='weighting' nothing is removed, so "reduced" is the wrong word and
# contradicts main4.tex, which writes every difference as "weighted − full".
# Read by the map difference panels and the two spectrum figures, NOT by the
# per-model boxplots, so those figures need a rerun for this to reach paper.
_SECOND   = "weighted" if em.is_weighting() else "reduced"
SIGNCONV  = f"full − {_SECOND}" if BIAS_SIGN > 0 else f"{_SECOND} − full"

# Additional plot ("spectrum"): how the too-hot bias (reduced − full) scales
# with the percentile of the daily distribution. CC scaling implies extremes
# inflate more than the mean — this figure should make that visible.
MAKE_SPECTRUM_PLOT = True

# Companion plot for the dry-spell side of the precip distribution.
# Reads precip_dry_spell_mean / _max / n_precip_dry_spells from each
# station's stats.parquet (populated by update_stats.py) -- so no Stage-1
# rebuild is needed. A separate auxiliary cache parquet is written once
# (DRYSPELL_CACHE_FPATH); subsequent runs reuse it.
MAKE_DRYSPELL_SPECTRUM = True

# values below this are treated as the -9991 "outside grid / all-NaN"
# sentinel set in _catch_mean_per_sample(); such realizations are unusable.
SENTINEL_BELOW = -100.0

# this script lives in paper/figures/scripts/, so the figures dir is its parent's parent
FIG_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# cache now holds ALL scenarios & periods, so its name is scenario/period-
# independent (one recompute after this change: the old cache name won't match).
CACHE_FPATH = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data", "fig02_change.parquet")
# per-script PNG output dir (matches the fig04 -> baseline/ convention).
# Keeping figures/ uncluttered makes it easy to wipe & re-render just this
# script's outputs without touching the parquet caches in fig_data/.
FIG_OUT_DIR = os.path.join(FIG_DIR, "fig02")
# {n} is filled in main() with the number of stations actually plotted, so the
# filename reflects the map's sample size (cache name stays count-independent).
OUT_PNG_TMPL = os.path.join(
    FIG_OUT_DIR,
    "precip_temp_change_{scenario}_{period}_n{n}sta.png")
MODEL_BOX_PNG_TMPL = os.path.join(
    FIG_OUT_DIR, "boxplots_by_model_{scenario}_n{n}sta.png")
SCENARIO_BOX_PNG_TMPL = os.path.join(
    FIG_OUT_DIR, "boxplots_by_scenario_n{n}sta.png")
SPECTRUM_PNG_TMPL = os.path.join(
    FIG_OUT_DIR, "spectrum_n{n}sta.png")
DRYSPELL_CACHE_FPATH = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data", "fig02_dryspell.parquet")
DRYSPELL_PNG_TMPL    = os.path.join(
    FIG_OUT_DIR, "dryspell_spectrum_n{n}sta.png")

# ===========================================================================
#  Stage 1 — per-realization precipitation (%) and temperature (°C) change
# ===========================================================================


def _delta_by_month(ds_sel, var, sample_month, sample_rank):
    """Rebuild the {month: rank-ordered delta array} dict for `var` from the
    flat 'sample' axis, mirroring save_mean_deltas_qqmap_day_as_nc's layout."""
    vals = np.asarray(ds_sel[var].values, dtype=np.float32)  # (sample,)
    by_month = {}
    for m in range(1, 13):
        sel = sample_month == m
        if not np.any(sel):
            continue
        idx = np.where(sel)[0]
        idx = idx[np.argsort(sample_rank[idx])]
        by_month[int(m)] = vals[idx]
    return by_month, vals


def _open_var(path):
    """Open a gscad daily_dataframes netCDF and return its 'var1' DataArray
    (all stations, lazily)."""
    return xr.open_dataset(path)["var1"]


def _obs_series(da, stn, time_format=None):
    """Select one station, return a datetime-indexed float32 Series."""
    s = da.sel(stations=stn).to_pandas().sort_index()
    s.index = (pd.to_datetime(s.index, format=time_format)
               if time_format else pd.to_datetime(s.index))
    return s.sort_index().astype("float32")


# --- per-worker reference state, populated by `_worker_init` --------------
# Stays empty until a worker (or the main process in the serial path) calls
# `_worker_init`. Holds the lazily-opened obs DataArrays so a worker can
# process many stations without reopening MSWEP/MSWX between them.
_REF = {}


def _worker_init(gscad_path, hist_start_str, hist_end_str):
    """Called once per worker (or once in the main process for the serial
    path). Opens reference climate datasets + static table and stashes them
    in `_REF` so `_process_station` can reuse them across many stations."""
    ddf = os.path.join(gscad_path, "dataframes", "daily_dataframes")
    _REF["mswep"]     = _open_var(os.path.join(ddf, "MSWEP_V280_Past.nc"))
    _REF["mswx_tmin"] = _open_var(os.path.join(ddf, "MSWX_V100_Past_Tmin.nc"))
    _REF["mswx_tmax"] = _open_var(os.path.join(ddf, "MSWX_V100_Past_Tmax.nc"))
    _REF["static"]    = pd.read_csv(
        os.path.join(gscad_path, "dataframes", "static.csv"),
        comment="#", index_col=0)
    _REF["obs_stations"] = (
        set(np.atleast_1d(_REF["mswep"]["stations"].values).astype(str))
        & set(np.atleast_1d(_REF["mswx_tmin"]["stations"].values).astype(str))
        & set(np.atleast_1d(_REF["mswx_tmax"]["stations"].values).astype(str)))
    _REF["hist_start"] = pd.Timestamp(hist_start_str)
    _REF["hist_end"]   = pd.Timestamp(hist_end_str)


def _process_station(f):
    """Worker entry point — process ONE station's mean_deltas.nc.

    Returns a tuple `(records, skip_key)` where `records` is a list of dicts
    ready for the cache, and `skip_key` is one of the keys of the `skip`
    counter in `compute_cache` (or None if the station produced records).
    Takes only a string `f` (path) so the Pool can pickle it. Reference
    climate is read from module-level `_REF` populated by `_worker_init`.
    Any unexpected exception is caught here so a Pool worker never crashes.
    """
    stn = os.path.basename(os.path.dirname(f))
    try:
        if stn not in _REF["obs_stations"] or stn not in _REF["static"].index:
            return [], "not_in_obs_static"

        try:
            ds = xr.open_dataset(f, engine="netcdf4")
        except Exception as e:
            logger.warning(f"  {stn}: cannot open ({e})")
            return [], "open_fail"

        try:
            file_periods  = set(np.atleast_1d(ds["period"].values).astype(str))
            avail_periods = [p for p in PERIODS if p in file_periods]
            if not avail_periods:
                return [], "no_period"

            sample_month = np.asarray(ds["sample_month"].values, dtype=np.int16)
            sample_rank  = np.asarray(ds["sample_rank"].values,  dtype=np.int64)

            try:
                obs_pr   = _obs_series(_REF["mswep"], stn).loc[
                    _REF["hist_start"]:_REF["hist_end"]]
                obs_tmin = _obs_series(_REF["mswx_tmin"], stn, "%Y%j").loc[
                    _REF["hist_start"]:_REF["hist_end"]]
                obs_tmax = _obs_series(_REF["mswx_tmax"], stn, "%Y%j").loc[
                    _REF["hist_start"]:_REF["hist_end"]]
            except Exception as e:
                logger.warning(f"  {stn}: obs load failed ({e})")
                return [], "bad_ref"

            base_pr  = float(obs_pr.mean())
            obs_tas  = (obs_tmin + obs_tmax) / 2.0
            base_tas = float(obs_tas.mean())
            if (not np.isfinite(base_pr) or base_pr <= 0
                    or not np.isfinite(base_tas)):
                return [], "bad_ref"

            # obs percentiles depend only on the historical window; precompute
            # once per station and reuse across every realization × period.
            obs_pr_pcts  = {q: float(obs_pr.quantile(q / 100.0))
                            for q in PRECIP_PERCENTILES}
            obs_tas_pcts = {q: float(obs_tas.quantile(q / 100.0))
                            for q in TEMP_PERCENTILES}

            lon = float(_REF["static"].loc[stn, "long"])
            lat = float(_REF["static"].loc[stn, "lat"])

            records = []
            for real in np.atleast_1d(ds["realization"].values).astype(str):
                scn, model, member = str(real).split("_", 2)
                if scn not in SCENARIOS or model not in FULL_MODELS:
                    continue

                ds_r = ds[["pr", "tmin", "tmax", "pr_wetfreq"]].sel(
                    realization=real)

                for per in avail_periods:
                    # skip blocklisted realizations (corrupt / degenerate delta)
                    # before the expensive apply step
                    if _is_blocked(model, member, scn, per):
                        continue
                    ds_sel = ds_r.sel(period=per)

                    pr_by_month,   pr_vals   = _delta_by_month(ds_sel, "pr",   sample_month, sample_rank)
                    tmin_by_month, tmin_vals = _delta_by_month(ds_sel, "tmin", sample_month, sample_rank)
                    tmax_by_month, tmax_vals = _delta_by_month(ds_sel, "tmax", sample_month, sample_rank)

                    if any((not np.isfinite(v).all()) or np.any(v < SENTINEL_BELOW)
                           for v in (pr_vals, tmin_vals, tmax_vals)):
                        continue

                    wetfreq = np.asarray(ds_sel["pr_wetfreq"].values,
                                         dtype=np.float32)
                    try:
                        fut_pr   = apply_qqmap_day_delta_pr(obs_pr, pr_by_month, wetfreq)
                        fut_tmin = apply_qqmap_day_delta_temp(obs_tmin, tmin_by_month)
                        fut_tmax = apply_qqmap_day_delta_temp(obs_tmax, tmax_by_month)
                    except Exception as e:
                        logger.warning(f"    {stn} {real} {per}: apply failed ({e})")
                        continue

                    fut_tas = (fut_tmin + fut_tmax) / 2.0
                    rec = {
                        "station": stn, "lon": lon, "lat": lat,
                        "scenario": scn, "model": model, "member": member,
                        "period": per,
                        "precip_change_pct":
                            (float(fut_pr.mean()) / base_pr - 1.0) * 100.0,
                        "temp_change_degC":
                            float(fut_tas.mean()) - base_tas,
                    }
                    # higher-percentile changes of the daily distributions
                    for q in PRECIP_PERCENTILES:
                        bq = obs_pr_pcts[q]
                        if np.isfinite(bq) and bq > 0:
                            fq = float(fut_pr.quantile(q / 100.0))
                            rec[f"precip_q{q}_change_pct"] = \
                                (fq / bq - 1.0) * 100.0
                        else:
                            rec[f"precip_q{q}_change_pct"] = np.nan
                    for q in TEMP_PERCENTILES:
                        bq = obs_tas_pcts[q]
                        if np.isfinite(bq):
                            fq = float(fut_tas.quantile(q / 100.0))
                            rec[f"temp_q{q}_change_degC"] = fq - bq
                        else:
                            rec[f"temp_q{q}_change_degC"] = np.nan
                    records.append(rec)

            return records, (None if records else "no_matching_realizations")
        finally:
            ds.close()
    except Exception as e:
        # last-resort safety net so a Pool worker never crashes mid-run
        logger.warning(f"  {stn}: unexpected worker error ({e!r})")
        return [], "open_fail"


def compute_cache():
    if PLOT_ONLY:
        msg = ("Stage 1 was triggered during a --only (plot-only) run: the "
               f"cache {CACHE_FPATH} is absent or does not cover the requested "
               "scenarios/periods/columns. This script never rebuilds it in "
               "--only mode. Run `sbatch slurm/fig02.sh` first, then rerun "
               "with --only.")
        # Also log it: main() prints WHICH check failed to stdout just above,
        # and the slurm scripts send stdout and stderr to different files.
        logger.error(msg)
        raise SystemExit(msg)

    n_dirs = len(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "")))
    md_files = sorted(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "mean_deltas.nc")))
    if MAX_STATIONS is not None:
        md_files = md_files[:MAX_STATIONS]

    n_workers = (NUM_WORKERS if NUM_WORKERS is not None
                 else max(1, (os.cpu_count() or 1) - 1))
    logger.info(f"{n_dirs} station dirs in out_path; "
                f"{len(md_files)} have mean_deltas.nc and will be scanned "
                f"(scenarios={SCENARIOS}, periods={PERIODS}"
                f"{'' if MAX_STATIONS is None else f', capped at {MAX_STATIONS}'}"
                f"; {n_workers} worker process{'es' if n_workers > 1 else ''})")

    skip = {"not_in_obs_static": 0, "open_fail": 0, "no_period": 0,
            "bad_ref": 0, "no_matching_realizations": 0}
    records = []

    init_args = (CONFIG.gscad_path,
                 CONFIG.historical_period[0],
                 CONFIG.historical_period[1])

    if n_workers <= 1:
        # serial — useful for debugging tracebacks (workers swallow them).
        _worker_init(*init_args)
        results_iter = (_process_station(f) for f in md_files)
        pool = None
    else:
        # `fork` so workers inherit FULL_MODELS / METRICS / SCENARIOS instead
        # of re-running module-import-time setup (which opens netCDF files).
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=n_workers,
                        initializer=_worker_init,
                        initargs=init_args)
        results_iter = pool.imap_unordered(_process_station, md_files,
                                            chunksize=1)

    try:
        for i, (rec, skip_key) in enumerate(results_iter):
            if rec:
                records.extend(rec)
            if skip_key:
                skip[skip_key] += 1
            if (i + 1) % 200 == 0:
                logger.info(f"  [{i+1}/{len(md_files)}] stations completed; "
                            f"{len(records)} rows so far")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No records produced — check "
                           "SCENARIOS/PERIODS/FULL_MODELS and CONFIG.out_path.")
    n_used = df["station"].nunique()
    logger.info("Stage 1 station accounting:")
    logger.info(f"  scanned (had mean_deltas.nc) : {len(md_files)}")
    for reason, n in skip.items():
        logger.info(f"  skipped [{reason}] : {n}")
    logger.info(f"  USED (>=1 valid realization) : {n_used}")
    logger.info(f"  -> {len(df)} (station,realization) rows cached")

    os.makedirs(os.path.dirname(CACHE_FPATH), exist_ok=True)
    # parquet only: CLIMATE_CACHE downstream (significance, bootstrap) reads
    # fig02_change.parquet via read_parquet; nothing reads a .csv twin.
    df.to_parquet(CACHE_FPATH)
    logger.info(f"Cached -> {CACHE_FPATH}")
    return df


# ===========================================================================
#  Stage 2 — aggregate to FULL / REDUCED ensembles and plot
# ===========================================================================


def aggregate(df, models, col):
    """Per-station ensemble mean of `col` over `models`."""
    sub = df[df["model"].isin(models)].copy()
    if AGGREGATION == "model_mean":
        # members -> model, then models -> ensemble (one model = one vote)
        per_model = (sub.groupby(["station", "lon", "lat", "model"], as_index=False)[col]
                        .mean())
        out = (per_model.groupby(["station", "lon", "lat"], as_index=False)[col]
                        .mean())
    elif AGGREGATION == "realization_mean":
        out = sub.groupby(["station", "lon", "lat"], as_index=False)[col].mean()
    else:
        raise ValueError(f"Unknown AGGREGATION={AGGREGATION!r}")
    return out


def aggregate_weighted(df, col):
    """Per-station WEIGHTED ensemble mean of `col` over FULL_MODELS: members
    -> model mean, then a weighted model mean with the frozen weights
    renormalised over the models present at each station (identical
    convention to model_weights.renormalize)."""
    if AGGREGATION != "model_mean":
        raise ValueError("weighting requires AGGREGATION='model_mean'")
    sub = df[df["model"].isin(FULL_MODELS)]
    per_model = (sub.groupby(["station", "lon", "lat", "model"],
                             as_index=False)[col].mean()
                    .dropna(subset=[col]))
    w = em.get_weights()
    per_model["_w"] = per_model["model"].map(w)
    per_model["_wx"] = per_model["_w"] * per_model[col]
    g = (per_model.groupby(["station", "lon", "lat"], as_index=False)
                  [["_wx", "_w"]].sum())
    g[col] = g["_wx"] / g["_w"]
    return g[["station", "lon", "lat", col]]


def ensemble_table(df, col):
    """Return per-station full/reduced/diff for value column `col`.

    Under method=weighting the "reduced" column holds the WEIGHTED ensemble
    mean instead (internal column name kept so every downstream plot works
    unchanged; display labels use em.second_label()).

    "diff" carries the MANUSCRIPT orientation (BIAS_SIGN * (full − reduced),
    i.e. reduced − full when BIAS_SIGN = -1); every plot in this file reads
    it, so the convention is flipped here and nowhere else."""
    full = aggregate(df, FULL_MODELS, col).rename(columns={col: "full"})
    if em.is_weighting():
        red = aggregate_weighted(df, col).rename(columns={col: "reduced"})
    else:
        red = aggregate(df, REDUCED_MODELS, col).rename(columns={col: "reduced"})
    m = full.merge(red, on=["station", "lon", "lat"], how="inner")
    m["diff"] = BIAS_SIGN * (m["full"] - m["reduced"])
    return m


def _ens2_count_label(models_present):
    """Legend fragment for the second ensemble: 'reduced=N' under removal,
    'N_eff=X.X' under weighting (Kish size over the present models)."""
    if em.is_weighting():
        return f"N_eff={em.n_eff_present(models_present):.1f}"
    return f"reduced={len(set(models_present) & set(REDUCED_MODELS))}"


def _greyify_diverging(name, gray=(0.78, 0.78, 0.78),
                       halfwidth=GREY_HALFWIDTH,
                       sigma=GREY_FADE_SIGMA,
                       src_pad=GREY_SRC_PAD, n=512):
    """Replace the white centre of a diverging matplotlib colormap with a
    gray band around 0, so values near zero stay visible against a white/ocean
    background without the gray bleeding far into either side.

    The gray is blended in rather than spliced in: each of `n` samples is a
    weighted mix of the source colour and `gray`. The weight is 1 across the
    pure-gray core (|x - 0.5| <= halfwidth) and decays along a Gaussian tail of
    scale `sigma` outside it. A Gaussian is flat at its peak, so the weight
    leaves the core with zero slope, and it has no cut-off, so it never has to
    return to 0 over a fixed window. The bar therefore carries no edge where
    the gray begins and none where it ends, and since the weight reads only
    |x - 0.5| the gray fades out at the same rate towards both ends.

    `src_pad` rescales the source before sampling so that its own near-white
    centre, spanning [0.5 - src_pad, 0.5 + src_pad], is dropped. Keep it small:
    the fade has to cross the whole colour distance from `gray` to whatever the
    source offers where the tail dies out, so a large pad puts a saturated
    colour a short way off the centre and makes the fade look abrupt however
    smooth the weight is. The endpoint colours of the source are preserved.
    """
    src = plt.get_cmap(name)
    x = np.linspace(0.0, 1.0, n)
    u = np.abs(x - 0.5)                          # 0 at the centre, 0.5 at the ends
    # |x - 0.5| in [0, 0.5] -> [src_pad, 0.5], i.e. skip the pale centre
    xs = 0.5 + np.sign(x - 0.5) * (src_pad + (0.5 - src_pad) * (u / 0.5))
    rgb = np.asarray(src(xs))[:, :3]
    t = np.maximum(u - halfwidth, 0.0) / max(sigma, 1e-9)
    w = np.exp(-t * t)                           # 1 across the core, then Gaussian
    g = np.asarray(gray, dtype=float)
    out = rgb * (1.0 - w)[:, None] + g[None, :] * w[:, None]
    return LinearSegmentedColormap.from_list(f"{name}_grey", out, N=n)


def _diff_norm_cmap(vals, greyify=False):
    """Colour map + norm for an ensemble-difference field (the "diff" column
    of ensemble_table, i.e. BIAS_SIGN * (full − reduced)).

    The full − reduced difference equals h/(r+h) · (mean of excluded
    "too-hot" models − mean of retained models), so under BIAS_SIGN = +1 it
    is positive wherever the too-hot models warm/wet more than the retained
    ones — which, for temperature, is almost everywhere (too-hot == high
    TCR); under BIAS_SIGN = -1 the same field is one-signed NEGATIVE. A
    symmetric ±dlim scale therefore wastes half the colour bar. We use
    robust (2nd/98th percentile) limits and pick:
      * a sequential map when the field is essentially one-signed (<2 % of
        the opposite sign) — Reds for positive fields, Blues_r for negative
        fields (the mirror of Reds: dark at the extreme, pale at 0) — which
        spends the whole colour range on the real spread;
      * a 0-centred diverging map (RdBu_r, TwoSlopeNorm) when both signs are
        genuinely present — 0 stays at the centre, sign preserved.
    If `greyify=True`, the diverging case uses RdBu_r with a gray centre
    (via `_greyify_diverging`) instead of white. Robust limits clip the
    most extreme ~4 % of catchments to the end colours.
    """
    v = np.asarray(vals, dtype=float)
    lo = float(np.nanpercentile(v, 2))
    hi = float(np.nanpercentile(v, 98))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = -1.0, 1.0
    if lo >= 0:                                    # essentially one-signed, +
        return "Reds", Normalize(vmin=0.0, vmax=max(hi, 1e-9))
    if hi <= 0:                                    # essentially one-signed, −
        return "Blues_r", Normalize(vmin=min(lo, -1e-9), vmax=0.0)
    cmap = _greyify_diverging("RdBu_r") if greyify else "RdBu_r"
    return cmap, TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


def _scatter(ax, lon, lat, val, cmap, title, vmin=None, vmax=None, norm=None):
    try:
        import cartopy.crs as ccrs           # optional; nicer maps if present
        import cartopy.feature as cfeature
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor=LAND_FACECOLOR, zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.5", zorder=1)
        kw = dict(transform=ccrs.PlateCarree())
    except Exception:
        ax.set_facecolor("0.97")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        kw = {}
    if norm is not None:                       # norm overrides vmin/vmax
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    sc = ax.scatter(lon, lat, c=val, s=4, cmap=cmap,
                     linewidths=0, rasterized=True, **kw)
    ax.set_title(title, fontsize=10)
    return sc


def _add_zone_box(ax, zones, values):
    """Inset boxplot of `values` split by KGZ_major zone, drawn in the
    empty lower-left corner of the map axes `ax`."""
    zones  = np.asarray(zones)
    values = np.asarray(values, dtype=float)
    inset = ax.inset_axes(BOX_RECT)
    counts = []
    for i, z in enumerate(KG_ZONES, start=1):
        v = values[(zones == z) & np.isfinite(values)]
        counts.append(len(v))
        if v.size == 0:                       # boxplot errors on empty input
            continue
        bp = inset.boxplot(v, positions=[i], widths=0.62,
                           showfliers=False, patch_artist=True)
        bp["boxes"][0].set(facecolor="0.85", linewidth=0.5)
        bp["medians"][0].set(color="black", linewidth=0.9)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(linewidth=0.5)
    inset.axhline(0.0, color="0.55", lw=0.6, zorder=0)   # sign reference
    allv = values[np.isfinite(values)]
    if allv.size:                              # robust y-limits
        lo, hi = np.nanpercentile(allv, [1, 99])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            inset.set_ylim(lo, hi)
    inset.set_xlim(0.5, len(KG_ZONES) + 0.5)
    inset.set_xticks(range(1, len(KG_ZONES) + 1))
    # Zone letter only. The per-zone catchment counts used to sit on a second
    # line under each letter, but at the inset scale they rendered ~7 pt and the
    # five 3-5 digit numbers ran into one another. They are logged instead, so
    # the caption can still quote n per zone.
    inset.set_xticklabels(KG_ZONES, fontsize=INSET_TICK_FS)
    logger.info("  inset KG counts: "
                + ", ".join(f"{z}={c}" for z, c in zip(KG_ZONES, counts)))
    # Both axes carry the SAME label size: the inset is small enough that two
    # sizes read as an inconsistency rather than a hierarchy.
    inset.tick_params(axis="y", labelsize=INSET_TICK_FS, length=2, pad=1)
    inset.tick_params(axis="x", labelsize=INSET_TICK_FS, length=0,
                      pad=INSET_XTICK_PAD)
    inset.set_facecolor((1, 1, 1, 0.0))       # readable over the ocean
    for side, sp in inset.spines.items():
        if side in ("top", "right"):
            sp.set_visible(False)            # open box: drop upper & right
        else:
            sp.set_linewidth(0.4)
    return inset


def _column(axes, m, vmin, vmax, cmap, labels, cbar_label, dcbar_label,
            cbar_side="right", greyify_centre=False):
    """Fill ONE column of the map figure. `axes` is either a 3-element array
    holding (full, reduced, diff) or, when MAP_SHOW_SECOND_ENSEMBLE is False,
    a 2-element array holding (full, diff); `labels` has one entry per panel
    in the same order. The full/reduced colourbar and the independent diff
    colourbar are placed on `cbar_side` ("left" or "right") of the column. If
    `greyify_centre=True`, the diverging full/reduced cmap and (when
    applicable) the diff cmap get a gray centre instead of white so values
    near zero stand out — used for the precipitation column."""
    if greyify_centre and isinstance(cmap, str):
        cmap = _greyify_diverging(cmap)
    three_row = len(axes) == 3
    cols = ("full", "reduced", "diff") if three_row else ("full", "diff")
    s = _scatter(axes[0], m.lon, m.lat, m["full"],    cmap, labels[0],
                 vmin=vmin, vmax=vmax)
    if three_row:
        _scatter(axes[1], m.lon, m.lat, m["reduced"], cmap, labels[1],
                 vmin=vmin, vmax=vmax)
    dcmap, dnorm = _diff_norm_cmap(m["diff"], greyify=greyify_centre)
    sd = _scatter(axes[-1], m.lon, m.lat, m["diff"], dcmap, labels[-1],
                  norm=dnorm)
    if ADD_ZONE_BOXPLOTS and "zone" in m.columns:
        for ax, col in zip(axes, cols):
            _add_zone_box(ax, m["zone"].values, m[col].values)
    fig = axes[0].figure
    # the first colourbar annotates every panel sharing the full-ensemble
    # scale: two panels in the 3-row layout, the single full panel in the
    # 2-row one. Its aspect then matches the diff bar so both read alike.
    fig.colorbar(s, ax=(axes[:2] if three_row else axes[0]),
                 location=cbar_side, fraction=0.045,
                 pad=0.02, aspect=30 if three_row else 15,
                 shrink=CBAR_SHRINK, label=cbar_label)
    cb = fig.colorbar(sd, ax=axes[-1], location=cbar_side, fraction=0.045,
                       pad=0.02, aspect=15, shrink=CBAR_SHRINK,
                       label=dcbar_label)
    if isinstance(dnorm, TwoSlopeNorm):
        # equal number of ticks on each side of 0 (values aren't round, but
        # the count is balanced — the two TwoSlopeNorm halves are equal width)
        k = DIFF_TICKS_PER_SIDE
        ticks = np.r_[np.linspace(dnorm.vmin, 0.0, k + 1)[:-1],
                      0.0,
                      np.linspace(0.0, dnorm.vmax, k + 1)[1:]]
        cb.set_ticks(ticks)
        # mfmt, not f"{t:.2g}": these labels replace the ones matplotlib would
        # have drawn, and half of them are negative, so a plain format would
        # put a hyphen where the two colourbars beside it carry a minus sign.
        cb.set_ticklabels([mfmt(t, ".2g") for t in ticks])


def boxplots_by_model(df, scenario):
    """Extra figure: one box per model showing the distribution *across
    catchments* of that model's per-catchment change (ensemble members are
    averaged within the model first, so each catchment contributes once per
    model & period). 2×N_period grid: rows = precipitation change (%) /
    temperature change (°C); columns = the future periods in PERIODS. Same
    model order in every panel; y shared per row so periods are directly
    comparable. "Too hot" models are coloured distinctly so the screening is
    visible. `df` should be a single-scenario slice of the Stage-1 cache (all
    periods kept); `scenario` is the SSP label used for the filename + log.
    """
    # members -> (station, model, period): one value per catchment per model
    pm = (df[df["model"].isin(FULL_MODELS)]
          .groupby(["station", "model", "period"], as_index=False)
          [["precip_change_pct", "temp_change_degC"]].mean())
    pm["period"] = pm["period"].astype(str)
    present = set(pm["model"])

    if MODEL_BOX_SORT == "tcr":   # hottest first, by the ACTIVE screening metric
        # em.metric_value, not _tcr: the boxes are coloured by EXCLUDED_MODELS
        # (the active screen), so sorting by TCR under METRIC='ecs' would
        # scatter the excluded boxes through the panel instead of grouping them
        # and the figure would read as an incoherent screen.
        order = sorted(FULL_MODELS, key=lambda m: em.metric_value(m),
                       reverse=True)
    else:
        order = list(FULL_MODELS)
    models = [m for m in order if m in present]
    periods = [p for p in PERIODS if p in set(pm["period"])]
    if not models or not periods:
        logger.warning("boxplots_by_model: no models/periods with data, skipping")
        return None

    nsta = df["station"].nunique()
    too_hot = set(EXCLUDED_MODELS)
    rows = [("precip_change_pct", "Precipitation change (%)"),
            ("temp_change_degC",  "Temperature change (°C)")]

    # Width per column scales with model count. Narrower boxes (MODEL_BOX_W)
    # allow a narrower column, so the whole figure shrinks and every label
    # occupies a larger fraction of it once the PNG is scaled to the page
    # width. 0.20 in per model leaves ~0.21 in of x pitch inside the axes,
    # which clears the ~0.19 in line height of a 12 pt rotated tick label.
    col_w = max(4.5, 0.20 * len(models) + 1.3)
    fig, axes = plt.subplots(
        len(rows), len(periods),
        figsize=(col_w * len(periods), 8.0),
        sharex=True, sharey="row", squeeze=False)

    for r, (col, ylab) in enumerate(rows):
        for c, per in enumerate(periods):
            ax = axes[r][c]
            sub = pm[pm["period"] == per]
            for i, m in enumerate(models, start=1):
                v = sub.loc[sub["model"] == m, col].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size == 0:                # boxplot errors on empty input
                    continue
                bp = ax.boxplot(v, positions=[i * MODEL_BOX_STEP],
                                widths=MODEL_BOX_W,
                                showfliers=False, patch_artist=True)
                bp["boxes"][0].set(
                    facecolor="firebrick" if m in too_hot else "steelblue",
                    alpha=0.75, linewidth=0.6)
                bp["medians"][0].set(color="black", linewidth=1.0)
                for w in bp["whiskers"] + bp["caps"]:
                    w.set(linewidth=0.5)
            ax.axhline(0.0, color="0.5", lw=0.7, zorder=0)   # sign reference
            ax.set_xlim(0.5 * MODEL_BOX_STEP,
                        (len(models) + 0.5) * MODEL_BOX_STEP)
            ax.grid(axis="y", color="0.92", lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=13)
            if r == 0:
                ax.set_title(plabel(per), fontsize=15)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=15)

    for c in range(len(periods)):
        axes[-1][c].set_xticks([i * MODEL_BOX_STEP
                                for i in range(1, len(models) + 1)])
        axes[-1][c].set_xticklabels(models, rotation=90, fontsize=12)

    # Legend wording follows the ACTIVE treatment. Under METHOD='weighting'
    # no climate model is dropped, so "excluded from reduced" / "retained"
    # describes an analysis this run did not perform. The colour split is the
    # screen threshold either way; what changes is what happens to the red
    # set. All 15 too-hot models carry less than one equal vote under the
    # Gaussian TCR kernel, so "down-weighted" holds for every one of them,
    # while the blue set is mixed (some up-weighted, some down) and is
    # therefore named neutrally.
    from matplotlib.patches import Patch
    if em.is_weighting():
        lab_hot, lab_rest = (f'"{_XHOT}" ({em.screen_expr()}), down-weighted',
                             "Other climate models")
    else:
        lab_hot, lab_rest = f'"{_XHOT}" (excluded from reduced)', "retained"
    axes[0][-1].legend(
        handles=[Patch(facecolor="firebrick", alpha=0.75, label=lab_hot),
                 Patch(facecolor="steelblue", alpha=0.75, label=lab_rest)],
        loc="upper right", fontsize=13, frameon=False)

    out = em.tag(MODEL_BOX_PNG_TMPL.format(scenario=scenario, n=nsta))
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    add_panel_letters(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(models)} models × {len(periods)} periods "
                f"{periods}, {nsta} catchments, {scenario.upper()})")
    return out


def boxplots_by_scenario(df):
    """Summary figure: per scenario, the distribution ACROSS CATCHMENTS of
    the per-catchment ensemble-mean change, FULL vs REDUCED (too-hot
    removed), with the global mean (mean over catchments) overlaid as a
    marker. 2×N_period grid: rows = precipitation change (%) / temperature
    change (°C); columns = PERIODS. Each panel has one full+reduced box pair
    per scenario in SCENARIOS. Uses the full Stage-1 cache `df`.
    """
    scenarios = [s for s in SCENARIOS if s in set(df["scenario"].astype(str))]
    periods   = [p for p in PERIODS   if p in set(df["period"].astype(str))]
    if not scenarios or not periods:
        logger.warning("boxplots_by_scenario: no scenarios/periods in cache, "
                        "skipping")
        return None

    rows = [("precip_change_pct", "Precipitation change (%)"),
            ("temp_change_degC",  "Temperature change (°C)")]

    # per (col, period, scenario): per-station FULL & REDUCED ensemble means
    # (same model-democracy aggregation as the maps, via ensemble_table)
    data = {}
    for p in periods:
        for s in scenarios:
            sub = df[(df["scenario"].astype(str) == s)
                     & (df["period"].astype(str) == p)]
            if sub.empty:
                continue
            for col, _ in rows:
                et = ensemble_table(sub, col)
                data[(col, p, s)] = (et["full"].to_numpy(dtype=float),
                                     et["reduced"].to_numpy(dtype=float))

    nsta = df["station"].nunique()
    FULL_C, REDUCED_C = "firebrick", "steelblue"   # full=reddish, reduced=bluish
    dx = 0.19

    col_w = max(5.5, 0.9 * len(scenarios) + 1.5)
    fig, axes = plt.subplots(len(rows), len(periods),
                             figsize=(col_w * len(periods), 7.2),
                             sharex=True, sharey="row", squeeze=False)

    for r, (col, ylab) in enumerate(rows):
        # robust shared y-limits across the whole row (1st-99th pct)
        rv = [np.concatenate([a, b]) for (cc, _, _), (a, b) in data.items()
              if cc == col]
        rv = np.concatenate(rv) if rv else np.array([])
        rv = rv[np.isfinite(rv)]
        ylim = None
        if rv.size:
            lo, hi = np.nanpercentile(rv, [1, 99])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                ylim = (lo, hi)

        for c, per in enumerate(periods):
            ax = axes[r][c]
            for k, s in enumerate(scenarios, start=1):
                pair = data.get((col, per, s))
                if pair is None:
                    continue
                for vals, xoff, color in ((pair[0], -dx, FULL_C),
                                          (pair[1], +dx, REDUCED_C)):
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:        # boxplot errors on empty input
                        continue
                    bp = ax.boxplot(vals, positions=[k + xoff], widths=0.32,
                                    showfliers=False, patch_artist=True)
                    bp["boxes"][0].set(facecolor=color, alpha=0.7,
                                       linewidth=0.6)
                    bp["medians"][0].set(color="black", linewidth=1.0)
                    for w in bp["whiskers"] + bp["caps"]:
                        w.set(linewidth=0.5)
                    ax.scatter([k + xoff], [np.nanmean(vals)], marker="D",
                               s=22, facecolor="white", edgecolor="black",
                               linewidth=0.7, zorder=5)        # global mean
            ax.axhline(0.0, color="0.5", lw=0.7, zorder=0)     # sign reference
            ax.set_xlim(0.5, len(scenarios) + 0.5)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", color="0.92", lw=0.5)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(plabel(per), fontsize=11)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=11)

    for c in range(len(periods)):
        axes[-1][c].set_xticks(range(1, len(scenarios) + 1))
        axes[-1][c].set_xticklabels([s.upper() for s in scenarios], fontsize=9)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axes[0][0].legend(
        handles=[Patch(facecolor=FULL_C, alpha=0.7, label="full ensemble"),
                 # not "weighted (BMA on TCR)": that named Massoud's scheme on
                 # every weighting run, including the gaussjoint kernel that
                 # main4.tex contrasts against BMA. The variant lives in the
                 # tag() suffix of the file name.
                 Patch(facecolor=REDUCED_C, alpha=0.7,
                       label=("weighted" if em.is_weighting()
                              else f"reduced ({_XHOT} removed)")),
                 Line2D([0], [0], marker="D", linestyle="None",
                        markerfacecolor="white", markeredgecolor="black",
                        label="global mean")],
        loc="best", fontsize=8, framealpha=0.7)

    out = em.tag(SCENARIO_BOX_PNG_TMPL.format(n=nsta))
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    add_panel_letters(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(scenarios)} scenarios × {len(periods)} "
                f"periods, full vs reduced, {nsta} catchments)")
    return out


def spectrum_plot(df):
    """Percentile-spectrum of the too-hot bias.

    For each cached period, plot the global-mean (across catchments) of the
    full − reduced difference as a function of the percentile of the daily
    distribution, one line per scenario, for precipitation (left column) and
    temperature (right column). If the line rises from Mean → Q90 → Q95 →
    Q99, the bias is concentrated at the tails (Clausius–Clapeyron
    expectation for precipitation extremes).

    Uses the full Stage-1 cache (all scenarios, all periods) but plots only
    `SPECTRUM_SCENARIOS` on the x-axis. The cache must carry the extra
    `precip_q{N}_change_pct` / `temp_q{N}_change_degC` columns — the cache
    guard above forces a recompute if any are missing. Dry-spell metrics
    live in their OWN figure (see `dryspell_spectrum_plot`).
    """
    # Spectrum is shown for a curated subset of scenarios (drops SSP1-1.9 by
    # default; configurable via SPECTRUM_SCENARIOS). Cache is unaffected.
    spectrum_scen = [s for s in SPECTRUM_SCENARIOS
                     if s in set(df["scenario"].astype(str))]
    scenarios = spectrum_scen
    periods   = [p for p in PERIODS   if p in set(df["period"].astype(str))]
    if not scenarios or not periods:
        logger.warning("spectrum_plot: no scenarios/periods, skipping")
        return None

    # column specs per panel: ordered list of (xtick-label, dataframe column).
    # Order = Mean -> low percentiles -> high percentiles, so the visual slope
    # makes physical sense (tail amplification reads as a rising curve).
    #
    # Tick labels come from figstyle.PERCENTILE_LABELS, the project-wide table
    # adopted on 2026-08-17. Q is reserved for discharge there, so a percentile
    # of temperature is T_p95 and one of precipitation is P_p95; a bare "Q95"
    # on these axes would both put a discharge symbol on a temperature and
    # collide with the flow-duration reading of Q95 (a LOW flow). _psym raises
    # rather than falling back, so a new percentile must be added to the table
    # instead of being hand-typed here.
    def _psym(col):
        lab = percentile_label(col)
        if lab is None:
            raise KeyError(f"{col!r} is not in figstyle.PERCENTILE_LABELS; "
                           "add it there rather than typing a bare percentile "
                           "symbol into spectrum_plot")
        return lab.short

    precip_specs = ([(_psym("precip_change_pct"), "precip_change_pct")]
                    + [(_psym(f"precip_q{q}_change_pct"),
                        f"precip_q{q}_change_pct")
                       for q in sorted(PRECIP_PERCENTILES)])
    # Temperature column: the 5th percentile is placed before the mean
    # (T_p05, T_mean, T_p50, T_p90, ...).
    temp_specs   = ([(_psym("temp_q5_change_degC"), "temp_q5_change_degC")]
                    if 5 in TEMP_PERCENTILES else []) \
                   + [(_psym("temp_change_degC"), "temp_change_degC")] \
                   + [(_psym(f"temp_q{q}_change_degC"),
                       f"temp_q{q}_change_degC")
                      for q in sorted(TEMP_PERCENTILES) if q != 5]
    panel_meta = [
        (f"Δ Precipitation change (%, {SIGNCONV})",
         "Daily precipitation statistic", precip_specs),
        (f"Δ Temperature change (°C, {SIGNCONV})",
         "Daily mean-temperature statistic", temp_specs),
    ]

    nsta = df["station"].nunique()
    cmap = plt.get_cmap("viridis")
    nscol = max(1, len(scenarios) - 1)
    scen_colors = {s: cmap(i / nscol) for i, s in enumerate(scenarios)}

    # Per-scenario count of distinct models present in the cache (FULL set
    # = every contributing model since Stage 1 filtered by FULL_MODELS;
    # REDUCED = same minus the too-hot list). Shown in the legend so the
    # reader can see that scenarios with smaller ensembles (e.g. SSP1-1.9)
    # have less coverage than SSP5-8.5.
    full_set = set(FULL_MODELS)
    scen_counts = {}
    for s in scenarios:
        models_s = set(df.loc[df["scenario"].astype(str) == s, "model"]
                       .astype(str).unique()) & full_set
        scen_counts[s] = (len(models_s), _ens2_count_label(models_s))

    fig, axes = plt.subplots(len(periods), 2,
                              figsize=(11.5, 4.2 * len(periods)),
                              squeeze=False)

    for r, period in enumerate(periods):
        for c, (ylab, xlab, specs) in enumerate(panel_meta):
            ax = axes[r][c]
            xpos    = np.arange(len(specs))
            xlabels = [name for name, _ in specs]

            for s in scenarios:
                sub = df[(df["scenario"].astype(str) == s)
                         & (df["period"].astype(str) == period)]
                if sub.empty:
                    continue
                ys = []
                for _, col_name in specs:
                    if col_name not in sub.columns:
                        ys.append(np.nan)
                        continue
                    et = ensemble_table(sub, col_name)
                    ys.append(float(et["diff"].mean()) if len(et)
                              else np.nan)
                n_full, lbl2 = scen_counts[s]
                lbl = f"{s.upper()}  (full={n_full}, {lbl2})"
                ax.plot(xpos, ys, marker="o",
                        color=scen_colors[s], label=lbl,
                        linewidth=1.5, markersize=5)

            ax.axhline(0.0, color="0.5", lw=0.6, zorder=0)
            ax.set_xticks(xpos)
            ax.set_xticklabels(xlabels)
            ax.set_xlabel(xlab, fontsize=12)
            ax.set_ylabel(ylab, fontsize=12)
            ax.set_title(plabel(period), fontsize=11)
            ax.grid(axis="y", color="0.92", lw=0.5)
            ax.set_axisbelow(True)

    # one shared legend in the top-right panel
    axes[0][-1].legend(loc="best", fontsize=10, frameon=False,
                       title="Scenario", title_fontsize=11)

    out = em.tag(SPECTRUM_PNG_TMPL.format(n=nsta))
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    add_panel_letters(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(scenarios)} scenarios × {len(periods)} "
                f"periods × {len(precip_specs)} precip / {len(temp_specs)} "
                f"temp points, {nsta} catchments)")
    return out


# ---------------------------------------------------------------------------
#  Dry-spell spectrum (separate figure, separate data source)
# ---------------------------------------------------------------------------
#  Reads precip_dry_spell_mean / _max / n_precip_dry_spells from each
#  station's stats.parquet (populated by `update_stats.py`). This avoids a
#  Stage-1 recompute -- the values are already on disk; we just join them.

DRYSPELL_RAW_COLS  = ["precip_dry_spell_mean",
                      "precip_dry_spell_max",
                      "n_precip_dry_spells"]
DRYSPELL_CHG_COLS  = ["precip_dryspell_mean_change_pct",
                      "precip_dryspell_max_change_pct",
                      "precip_n_dryspell_change_pct"]


def _read_station_dryspell(station_dir):
    """Read ONE station's stats.parquet and return a long-format DataFrame
    of dry-spell *percent change* rows for each (scenario, model, member,
    period). Engine-independent (precip dry-spell is the same for hbv/lstm),
    so we filter to engine='hbv' and drop the engine column.

    Returns None if the file is missing / malformed / has no usable baseline.
    Designed to be Pool-pickleable (takes a string, returns a small DataFrame).
    """
    try:
        stn = os.path.basename(station_dir.rstrip("/"))
        sp = os.path.join(station_dir, "stats.parquet")
        if not os.path.exists(sp):
            return None
        s = pd.read_parquet(sp)
        for c in DRYSPELL_RAW_COLS + ["scenario", "model", "member",
                                       "period", "engine"]:
            if c not in s.columns:
                return None
        s = s[s["engine"].astype(str) == "hbv"]
        base = s[(s["scenario"].astype(str) == "baseline")
                 & (s["period"].astype(str) == "historical")]
        if base.empty:
            return None
        bm = float(base["precip_dry_spell_mean"].iloc[0])
        bx = float(base["precip_dry_spell_max"].iloc[0])
        bn = float(base["n_precip_dry_spells"].iloc[0])
        if not (np.isfinite(bm) and bm > 0
                and np.isfinite(bx) and bx > 0
                and np.isfinite(bn) and bn > 0):
            return None

        fut = s[s["scenario"].astype(str) != "baseline"].copy()
        if fut.empty:
            return None
        fut["precip_dryspell_mean_change_pct"] = (
            fut["precip_dry_spell_mean"].astype(float) / bm - 1.0) * 100.0
        fut["precip_dryspell_max_change_pct"]  = (
            fut["precip_dry_spell_max"].astype(float)  / bx - 1.0) * 100.0
        fut["precip_n_dryspell_change_pct"]    = (
            fut["n_precip_dry_spells"].astype(float)   / bn - 1.0) * 100.0
        fut["station"] = stn
        return fut[["station", "scenario", "model", "member", "period"]
                   + DRYSPELL_CHG_COLS]
    except Exception:
        return None


def load_dryspell_table(df, force=False):
    """Build (or load) the long-format dry-spell change table. If a parquet
    cache exists at DRYSPELL_CACHE_FPATH and covers every station present in
    `df`, it is reused; otherwise the per-station stats.parquet files are
    scanned in parallel (Pool of NUM_WORKERS) and the result is cached.

    `force=True` rebuilds the cache unconditionally."""
    needed_stations = sorted(df["station"].astype(str).unique())
    if (not force and os.path.exists(DRYSPELL_CACHE_FPATH)):
        try:
            ds = pd.read_parquet(DRYSPELL_CACHE_FPATH)
            have = set(ds["station"].astype(str).unique())
            if set(needed_stations).issubset(have):
                logger.info(
                    f"Loaded cached dry-spell table {DRYSPELL_CACHE_FPATH} "
                    f"({len(ds)} rows, {len(have)} stations)")
                return ds
            logger.info(
                f"  dry-spell cache covers {len(have)} stations but {df['station'].nunique()} "
                "needed -> rebuilding")
        except Exception as e:
            logger.warning(f"  dry-spell cache unreadable ({e}) -> rebuilding")

    # A --only run promises "no rebuild". The dry-spell scan is the SECOND
    # expensive path in this script (a Pool over every station's stats.parquet),
    # so it is refused there for the same reason compute_cache() is.
    if PLOT_ONLY:
        msg = ("The dry-spell table would be rebuilt during a --only "
               f"(plot-only) run: {DRYSPELL_CACHE_FPATH} is missing, "
               f"unreadable, or does not cover all {len(needed_stations)} "
               "stations in the cache. This scan reads every station's "
               "stats.parquet, so --only refuses it. Run `sbatch "
               "slurm/fig02.sh` (or slurm/fig02_replot.sh) to refresh the "
               "dry-spell cache, then rerun with --only.")
        # logged as well as raised: the reason line above goes to stdout, the
        # SystemExit to stderr, and slurm keeps those in separate files.
        logger.error(msg)
        raise SystemExit(msg)

    # Build by parallel scan. Station dirs are CONFIG.out_path/CONFIG.delta_method/<stn>/.
    station_dirs = [os.path.join(CONFIG.out_path, CONFIG.delta_method, s) for s in needed_stations]
    n_workers = (NUM_WORKERS if NUM_WORKERS is not None
                 else max(1, (os.cpu_count() or 1) - 1))
    logger.info(f"Scanning {len(station_dirs)} stats.parquet files for "
                f"dry-spell stats ({n_workers} worker process"
                f"{'es' if n_workers > 1 else ''})")

    if n_workers <= 1:
        results = (_read_station_dryspell(d) for d in station_dirs)
        pool = None
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=n_workers)
        results = pool.imap_unordered(_read_station_dryspell, station_dirs,
                                       chunksize=4)

    parts = []
    try:
        for i, r in enumerate(results):
            if r is not None and len(r):
                parts.append(r)
            if (i + 1) % 500 == 0:
                logger.info(f"  [{i+1}/{len(station_dirs)}] stations scanned, "
                            f"{sum(len(p) for p in parts)} rows so far")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if not parts:
        logger.warning("load_dryspell_table: no usable rows found")
        return pd.DataFrame(columns=["station", "scenario", "model", "member",
                                      "period"] + DRYSPELL_CHG_COLS)
    ds = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(DRYSPELL_CACHE_FPATH), exist_ok=True)
    ds.to_parquet(DRYSPELL_CACHE_FPATH)
    logger.info(f"Cached dry-spell table -> {DRYSPELL_CACHE_FPATH} "
                f"({len(ds)} rows, {ds['station'].nunique()} stations)")
    return ds


def dryspell_spectrum_plot(df):
    """Dry-spell-spectrum: bias of full ensemble vs reduced ensemble as a
    function of the dry-spell *metric* (mean spell length, max spell length =
    CDD, count of spells), for each cached period. One line per scenario in
    SPECTRUM_SCENARIOS, one panel per period. Reads dry-spell data from the
    per-station stats.parquet files via `load_dryspell_table` (no Stage-1
    rebuild needed).
    """
    # Attach the dry-spell change columns onto the main long table by inner
    # join on (station, scenario, model, member, period). The main df brings
    # lon/lat so ensemble_table works without further joins.
    ds = load_dryspell_table(df)
    if ds.empty:
        logger.warning("dryspell_spectrum_plot: empty dry-spell table, skipping")
        return None
    merged = df[["station", "lon", "lat", "scenario", "model", "member",
                 "period"]].merge(
        ds, on=["station", "scenario", "model", "member", "period"],
        how="inner")
    if merged.empty:
        logger.warning("dryspell_spectrum_plot: no overlap with main cache, skipping")
        return None

    spectrum_scen = [s for s in SPECTRUM_SCENARIOS
                     if s in set(merged["scenario"].astype(str))]
    periods = [p for p in PERIODS
               if p in set(merged["period"].astype(str))]
    if not spectrum_scen or not periods:
        logger.warning("dryspell_spectrum_plot: no scenarios/periods, skipping")
        return None

    specs = [("DS-mean", "precip_dryspell_mean_change_pct"),
             ("DS-max",  "precip_dryspell_max_change_pct"),
             ("n DS",    "precip_n_dryspell_change_pct")]

    nsta = merged["station"].nunique()
    cmap = plt.get_cmap("viridis")
    nscol = max(1, len(spectrum_scen) - 1)
    scen_colors = {s: cmap(i / nscol) for i, s in enumerate(spectrum_scen)}

    # legend counts on the merged data so they reflect what's actually plotted
    full_set = set(FULL_MODELS)
    scen_counts = {}
    for s in spectrum_scen:
        models_s = set(merged.loc[merged["scenario"].astype(str) == s, "model"]
                       .astype(str).unique()) & full_set
        scen_counts[s] = (len(models_s), _ens2_count_label(models_s))

    fig, axes = plt.subplots(1, len(periods),
                              figsize=(5.5 * len(periods), 4.4),
                              squeeze=False, sharey=True)

    for c, period in enumerate(periods):
        ax = axes[0][c]
        xpos = np.arange(len(specs))
        xlabels = [name for name, _ in specs]

        for s in spectrum_scen:
            sub = merged[(merged["scenario"].astype(str) == s)
                         & (merged["period"].astype(str) == period)]
            if sub.empty:
                continue
            ys = []
            for _, col_name in specs:
                if col_name not in sub.columns:
                    ys.append(np.nan)
                    continue
                et = ensemble_table(sub, col_name)
                ys.append(float(et["diff"].mean()) if len(et) else np.nan)
            n_full, lbl2 = scen_counts[s]
            lbl = f"{s.upper()}  (full={n_full}, {lbl2})"
            ax.plot(xpos, ys, marker="o",
                    color=scen_colors[s], label=lbl,
                    linewidth=1.5, markersize=5)

        ax.axhline(0.0, color="0.5", lw=0.6, zorder=0)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Precipitation dry-spell metric "
                      "(DS = consecutive dry days)", fontsize=10)
        if c == 0:
            ax.set_ylabel(f"Δ Dry-spell change (%, {SIGNCONV})",
                          fontsize=10)
        ax.set_title(plabel(period), fontsize=11)
        ax.grid(axis="y", color="0.92", lw=0.5)
        ax.set_axisbelow(True)

    axes[0][-1].legend(loc="best", fontsize=8, frameon=False,
                       title="Scenario")

    out = em.tag(DRYSPELL_PNG_TMPL.format(n=nsta))
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    add_panel_letters(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(spectrum_scen)} scenarios × "
                f"{len(periods)} periods × {len(specs)} dry-spell metrics, "
                f"{nsta} catchments)")
    return out


def main():
    if os.path.exists(CACHE_FPATH) and not RECOMPUTE:
        logger.info(f"Loading cached per-realization table {CACHE_FPATH} "
                    "(set RECOMPUTE=True to rebuild)")
        df = pd.read_parquet(CACHE_FPATH)
        have_p = set(df["period"].astype(str))   if "period"   in df.columns else set()
        have_s = set(df["scenario"].astype(str)) if "scenario" in df.columns else set()
        if "temp_change_degC" not in df.columns:
            logger.info("  cache lacks 'temp_change_degC' (old precipitation-only "
                        "run) -> recomputing")
            df = compute_cache()
        elif not set(PERIODS).issubset(have_p):
            logger.info(f"  cache periods {sorted(have_p)} miss some of {PERIODS} "
                        "-> recomputing")
            df = compute_cache()
        elif not set(SCENARIOS).issubset(have_s):
            logger.info(f"  cache scenarios {sorted(have_s)} miss some of "
                        f"{SCENARIOS} -> recomputing")
            df = compute_cache()
        elif any(c not in df.columns for c in (
                [f"precip_q{q}_change_pct" for q in PRECIP_PERCENTILES]
                + [f"temp_q{q}_change_degC" for q in TEMP_PERCENTILES])):
            missing = [c for c in (
                [f"precip_q{q}_change_pct" for q in PRECIP_PERCENTILES]
                + [f"temp_q{q}_change_degC" for q in TEMP_PERCENTILES])
                if c not in df.columns]
            logger.info(f"  cache missing {len(missing)} percentile column(s) "
                        f"({missing[:3]}{'...' if len(missing) > 3 else ''}) "
                        "-> recomputing")
            df = compute_cache()
        else:
            # FULL_MODELS is now auto-derived from models_in_results(), so it
            # can grow when new mean_deltas.nc files appear. We only warn
            # here (a recompute would be ~5× longer per added model — too
            # harsh to trigger silently). Set RECOMPUTE=True to absorb them.
            have_m = set(df["model"].astype(str)) if "model" in df.columns else set()
            new_models = sorted(set(FULL_MODELS) - have_m)
            if new_models:
                logger.warning(
                    f"  cache is missing {len(new_models)} FULL_MODELS that "
                    f"were just discovered: {new_models}. They will be "
                    "excluded from the figures until you run with "
                    "RECOMPUTE=True."
                )
    else:
        df = compute_cache()

    # Drop blocklisted realizations (corrupt downloads / degenerate qqmap_day
    # deltas). Applied here too — not just in Stage 1 — so an EXISTING cache is
    # cleaned at plot time without forcing a full recompute.
    df = apply_blocklist(df)

    # KG zone is a per-station attribute (independent of scenario/period) so
    # load it once and reuse across every (scenario, period) map. Only the map
    # panels carry the inset, so a run with the maps switched off does not read
    # static.csv at all.
    zser = None
    if ADD_ZONE_BOXPLOTS and MAKE_MAPS:
        static = pd.read_csv(os.path.join(CONFIG.gscad_path, "dataframes",
                                          "static.csv"), comment="#", index_col=0)
        if "KGZ_major" in static.columns:
            zser = static["KGZ_major"].astype(str).str.strip()
        else:
            logger.warning("static.csv has no 'KGZ_major' column -> "
                           "skipping inset boxplots")

    logger.info(f"Stage 2: {df['station'].nunique()} stations in cache "
                f"(scenarios {sorted(set(df['scenario'].astype(str)))}, periods "
                f"{sorted(set(df['period'].astype(str)))}) | "
                + (f"maps will be drawn for {MAP_SCENARIOS} × {MAP_PERIODS}"
                   if MAKE_MAPS else "maps skipped (MAKE_MAPS is False)"))
    logger.info(f"figures selected: {', '.join(selected_targets()) or 'none'}")
    logger.info(f"FULL models ({len(FULL_MODELS)}): {FULL_MODELS}")
    logger.info(f"REDUCED models ({len(REDUCED_MODELS)}): {REDUCED_MODELS}")
    logger.info(f"Excluded as {_XHOT}: {sorted(set(FULL_MODELS) - set(REDUCED_MODELS))}")

    proj = {}
    if MAKE_MAPS:
        try:
            import cartopy.crs as ccrs
            proj = dict(subplot_kw={"projection": ccrs.Robinson()})
        except Exception:
            proj = {}

    # ---- one map figure per (MAP_SCENARIOS × MAP_PERIODS) combo ------------
    # MAKE_MAPS gates the loop through its iterable so the body keeps its
    # original indentation (and therefore stays diff-clean).
    for map_scen in (MAP_SCENARIOS if MAKE_MAPS else []):
        for map_per in MAP_PERIODS:
            dmap = df[(df["period"].astype(str)   == map_per)
                      & (df["scenario"].astype(str) == map_scen)]
            if dmap.empty:
                logger.warning(f"  [{map_scen}/{map_per}] no cached rows -> "
                               "skipping this map")
                continue

            mp = ensemble_table(dmap, "precip_change_pct")
            mt = ensemble_table(dmap, "temp_change_degC")
            if zser is not None:
                for tbl in (mp, mt):
                    tbl["zone"] = tbl["station"].map(zser)

            logger.info(f"[{map_scen} {map_per}] precip plotted={len(mp)} | "
                        f"temp plotted={len(mt)} | "
                        f"global mean precip change (%): "
                        f"full={mp['full'].mean():.2f} "
                        f"reduced={mp['reduced'].mean():.2f} "
                        f"diff={mp['diff'].mean():.2f}; "
                        f"global mean temp change (degC): "
                        f"full={mt['full'].mean():.2f} "
                        f"reduced={mt['reduced'].mean():.2f} "
                        f"diff={mt['diff'].mean():.2f}")

            # robust per-map scales (precipitation symmetric BrBG, temperature
            # sequential YlOrRd; diff panels get their own 0-aware scale)
            p_lim = float(np.nanpercentile(
                np.abs(np.r_[mp["full"], mp["reduced"]]), 98))
            t_lo  = float(np.nanpercentile(
                np.r_[mt["full"], mt["reduced"]], 2))
            t_hi  = float(np.nanpercentile(
                np.r_[mt["full"], mt["reduced"]], 98))

            # 3x2 with the second-ensemble row, 2x2 without it. The height is
            # scaled by the row count so each map cell keeps the height tuned
            # for MAP_FIG_H, and with it the row gap MAP_FIG_H controls.
            nrows = 3 if MAP_SHOW_SECOND_ENSEMBLE else 2
            fig, axes = plt.subplots(
                nrows, 2, figsize=(MAP_FIG_W, MAP_FIG_H * nrows / 3.0),
                gridspec_kw=dict(wspace=WSPACE, hspace=HSPACE), **proj)

            # Panel letters are NOT part of these titles any more: add_panel_letters
            # below stamps the bold top-left letter (Nature style). Row-major order
            # over the grid gives a,b / c,d (/ e,f) down the rows, so the letters
            # follow the reading order of the panels that are actually drawn.
            _lbl2 = em.second_label().lower()
            _sgn2 = (f"{_lbl2} − full" if BIAS_SIGN < 0 else f"full − {_lbl2}")
            _plabels = ["Precipitation change (%): full ensemble",
                        f"Precipitation change (%): {_lbl2} ensemble",
                        f"Precipitation change (%): difference ({_sgn2})"]
            _tlabels = ["Temperature change (°C): full ensemble",
                        f"Temperature change (°C): {_lbl2} ensemble",
                        f"Temperature change (°C): difference ({_sgn2})"]
            if not MAP_SHOW_SECOND_ENSEMBLE:      # drop the middle-row title
                _plabels = [_plabels[0], _plabels[2]]
                _tlabels = [_tlabels[0], _tlabels[2]]

            _column(axes[:, 0], mp, -p_lim, p_lim, "BrBG",
                    labels=_plabels,
                    cbar_label="Precipitation change (%)",
                    dcbar_label="Δ Precipitation change (%)",
                    cbar_side="left", greyify_centre=True)

            _column(axes[:, 1], mt, t_lo, t_hi, "YlOrRd",
                    labels=_tlabels,
                    cbar_label="Temperature change (°C)",
                    dcbar_label="Δ Temperature change (°C)",
                    cbar_side="right", greyify_centre=False)

            out_png = em.tag(OUT_PNG_TMPL.format(scenario=map_scen,
                                                 period=map_per, n=len(mp)))
            os.makedirs(FIG_OUT_DIR, exist_ok=True)
            add_panel_letters(axes)
            fig.savefig(out_png, dpi=250, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {out_png}")

    # ---- per-model boxplots: one figure per MAP_SCENARIOS (covers periods)
    if MAKE_MODEL_BOXPLOTS:
        for map_scen in MAP_SCENARIOS:
            sub = df[df["scenario"].astype(str) == map_scen]
            if sub.empty:
                logger.warning(f"  [{map_scen}] no rows for per-model boxplot")
                continue
            boxplots_by_model(sub, scenario=map_scen)

    if MAKE_SCENARIO_BOXPLOTS:
        boxplots_by_scenario(df)

    if MAKE_SPECTRUM_PLOT:
        spectrum_plot(df)

    if MAKE_DRYSPELL_SPECTRUM:
        dryspell_spectrum_plot(df)


# ---------------------------------------------------------------------------
#  command line
# ---------------------------------------------------------------------------
# Figure target -> the module toggle it drives. Keys are the `--only` names.
FIGURE_TARGETS = {
    "maps":              "MAKE_MAPS",
    "model_boxplots":    "MAKE_MODEL_BOXPLOTS",
    "scenario_boxplots": "MAKE_SCENARIO_BOXPLOTS",
    "spectrum":          "MAKE_SPECTRUM_PLOT",
    "dryspell_spectrum": "MAKE_DRYSPELL_SPECTRUM",
}


def selected_targets():
    """Names of the figure targets whose toggle is currently True."""
    return [name for name, flag in FIGURE_TARGETS.items() if globals()[flag]]


def _parse_cli(argv):
    """Apply the command line to the module-level settings.

    Called ONLY from the `__main__` block: importing this module leaves
    MAX_STATIONS / RECOMPUTE / the MAKE_* toggles at their edited-in-file
    defaults, whatever sys.argv happens to hold in the importing script.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="fig02_precip_change.py",
        description="Figure 2 (precipitation/temperature change, full vs "
                    "reduced ensemble). With no arguments it builds the "
                    "Stage-1 cache if needed and draws every figure.")
    ap.add_argument("max_stations_pos", nargs="?", type=int, default=None,
                    metavar="MAX_STATIONS",
                    help="legacy positional form of --max-stations")
    ap.add_argument("--max-stations", type=int, default=None,
                    help="use only the first N stations (quick test)")
    ap.add_argument("--only", action="append", default=None, metavar="TARGET",
                    help="draw only these figures (comma-separated, or repeat "
                         "the flag). Choices: "
                         + ", ".join(FIGURE_TARGETS) + ". Implies plot-only: "
                         "the Stage-1 cache is read, never rebuilt.")
    ap.add_argument("--recompute", action="store_true",
                    help="force the Stage-1 rebuild (same as RECOMPUTE=True)")
    ap.add_argument("--list", action="store_true",
                    help="list the figure targets and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("figure targets for --only:")
        for name, flag in FIGURE_TARGETS.items():
            print(f"  {name:<18} -> {flag}")
        raise SystemExit(0)

    if args.max_stations is not None and args.max_stations_pos is not None:
        ap.error("give MAX_STATIONS either positionally or as --max-stations, "
                 "not both")
    n = (args.max_stations if args.max_stations is not None
         else args.max_stations_pos)
    if n is not None:
        if n <= 0:
            ap.error("MAX_STATIONS must be a positive integer")
        # A capped Stage 1 writes an N-station table. Sending that to the SHARED
        # cache path would leave every downstream script reading N catchments
        # while believing it has the full network, so a capped run gets its own
        # pair of cache files — ALWAYS, not only under --recompute. Stage 1 is
        # reachable with RECOMPUTE=False too (cache absent, or present but
        # missing a scenario/period/percentile column), which is exactly when a
        # "quick test with 200" is typed.
        root, ext = os.path.splitext(CACHE_FPATH)
        globals()["CACHE_FPATH"] = f"{root}_max{n}{ext}"
        # The dry-spell table is keyed off the same station set, so it must be
        # redirected with it (load_dryspell_table caches whatever it scanned).
        droot, dext = os.path.splitext(DRYSPELL_CACHE_FPATH)
        globals()["DRYSPELL_CACHE_FPATH"] = f"{droot}_max{n}{dext}"
        print(f"MAX_STATIONS={n}: this run reads and writes its own caches "
              f"{CACHE_FPATH} and {DRYSPELL_CACHE_FPATH}; the shared "
              f"{root}{ext} and {droot}{dext} are untouched. Its figures are "
              f"named n{n}sta and are test artefacts — delete both _max{n} "
              "caches when done.")
        globals()["MAX_STATIONS"] = n

    if args.only:
        if args.recompute:
            ap.error("--only is plot-only and cannot be combined with "
                     "--recompute")
        if n is not None:
            ap.error("--only replots the existing full cache; --max-stations "
                     "would mislabel the output PNGs with a truncated station "
                     "count. Drop one of the two.")
        wanted = [t.strip() for chunk in args.only for t in chunk.split(",")
                  if t.strip()]
        if not wanted:
            # `--only ""` / `--only ,` would otherwise switch every figure off
            # and exit 0 after loading the whole cache — a silent no-op that
            # looks like a successful job.
            ap.error("--only was given no target. Choices: "
                     + ", ".join(FIGURE_TARGETS))
        unknown = [t for t in wanted if t not in FIGURE_TARGETS]
        if unknown:
            ap.error(f"unknown --only target(s): {', '.join(unknown)}. "
                     f"Choices: {', '.join(FIGURE_TARGETS)}")
        for name, flag in FIGURE_TARGETS.items():
            globals()[flag] = name in wanted
        globals()["PLOT_ONLY"] = True

    if args.recompute:
        globals()["RECOMPUTE"] = True


if __name__ == "__main__":
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment(logger)
    # method=removal|weighting and weights=bma|kde are consumed here; the
    # remaining tokens go through the unchanged argparse CLI.
    _parse_cli(em.reject_cli_tokens(sys.argv[1:]))
    logger.info(f"ensemble method = {em.method()}"
                + (f" [{em.variant()}], suffix '{em.suffix()}'"
                   if em.is_weighting() else ""))
    main()
