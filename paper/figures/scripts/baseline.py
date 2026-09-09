"""
Figure 4 — present-day (baseline) climatology of the hydrological indices.

Manuscript spec: "Impact of full vs reduced ensemble on changes in
hydrological indices in future" subsection of too_hot_models.tex (labelled
"Figure 4" in the current outline). Q05 stands in for the .tex's "dry spell
duration" per user request. Two output families come out of this script:

  * Combined 3x2 plot: `baseline_indices_n{n}sta.png`
        rows = the original three indices (mean annual runoff, peak-flow
        magnitude, low-flow Q05); columns = engines (HBV, LSTM). Layout
        preserved per user request — driven by INDICES[:N_COMBINED].
  * Thematic combined plots (COMBINED_GROUPS): rows = indices, cols = engines.
        `lowflow_dryspell_n{n}sta.png` (4x2) groups Q05 with the dry-spell mean
        duration and count and the runoff ratio Q/P — Q05 *is* the dry-day
        threshold (qsim < baseline Q05), so the first three rows read
        top-to-bottom as a causal chain (threshold -> persistence -> frequency)
        and the runoff ratio gives the water-balance context.
        `flow_quantiles_flashiness_n{n}sta.png`
        (4x2) groups the mid/high flow quantiles (Q50, Q95, Q99) with the
        Richards-Baker flashiness index. `flow_variability_extremes_n{n}sta.png`
        (3x2) groups daily-flow std, number of flood events and annual-minimum
        flow. `fdc_flood_signatures_n{n}sta.png` (4x2) groups the FDC slope, the
        Q5/Q95 flow ratio and the 20- and 10-year GEV flood return levels.
        `lowflow_waterbalance_n{n}sta.png` (4x2) groups the 7Q10 design low
        flow, the 7-day low flow (MAM7), the low-flow deficit and the maximum
        dry-spell length — the drought-severity family (magnitude, volume and
        duration extreme). Its first three rows and the fdc_flood_signatures
        figure carry order-independent (distributional) signatures, which —
        unlike flashiness, the maximum dry-spell length and the dry-spell /
        flood-event counts — transfer faithfully through the qqmap delta method.
  * Per-index 1x2 plots: `{slug}_n{n}sta.png`, one file per index. Currently
        every index slug is covered by the combined baseline_indices trio or a
        COMBINED_GROUPS figure (see SKIP_PER_INDEX_SLUGS), so NO standalone
        per-index PNG is written; the loop is kept so an individual index can be
        re-enabled by removing its slug from SKIP_PER_INDEX_SLUGS.
  * Boxplot summaries (BOXPLOT_GROUPS), two figures, one small panel per index
        (grid of panels). `index_boxplots_tabS6_n{n}sta.png` carries the 14
        indices of the manuscript's streamflow-index table (Supplementary Table
        S6, paper/tables/streamflow_indices_table.tex) in that table's order;
        `index_boxplots_extra_n{n}sta.png` carries every other index this script
        computes — the remaining 15 members of the 29-metric streamflow registry
        (paper/figures/scripts/flow_metrics.py) plus the dry-spell count and the
        flood-event count, which are baseline-only diagnostics outside that
        registry. Each panel holds an HBV and an LSTM box on its OWN y-axis
        (units-appropriate; log or linear by the same `_robust_norm` rule the
        maps use), giving an at-a-glance HBV-vs-LSTM comparison for every index
        without forcing the different units onto one shared scale.

Output dirs: PNGs -> `paper/figures/baseline/`; parquet/csv cache ->
`paper/figures/fig_data/` (shared cache convention with fig02/fig05). The
two-dir split means `rm -rf paper/figures/baseline/` rebuilds the plots
without losing the expensive Stage-1 cache.

This is the *absolute* present-day state of the same indices whose future
*% change* Figure 5 (hydro_indices_change.py) shows — it gives the
reader the spatial context / denominator for those relative changes. The
baseline run uses observed MSWEP/MSWX forcing (no CMIP6 deltas), so each
index is a single value per catchment: ensemble- and scenario-independent,
hence one map per (engine, index) rather than the full/reduced/diff triplets.

A column shares one log colour scale across the HBV and LSTM rows so the two
engines are directly comparable for that index (the three columns are
different indices with different units, so they cannot share a scale).

Where the values come from
--------------------------
Every station's stats.parquet has a baseline row
(scenario=model=member="baseline", period="historical"), one per engine. We
read every scalar baseline statistic produced by the apply_deltas pipeline,
for each ENGINE:
  * Mean annual runoff       (mm yr-1)  <- qmean * DAYS_PER_YEAR
  * Daily-flow std           (mm day-1) <- qstd
  * Peak-flow magnitude      (mm day-1) <- mean_annual_max  (mean annual max)
  * Annual minimum flow      (mm day-1) <- mean_annual_min  (mean annual min)
  * Low flow  Q05            (mm day-1) <- q05  (5th-percentile daily flow)
  * Median    Q50            (mm day-1) <- q50
  * High flow Q95            (mm day-1) <- q95
  * Very-high flow Q99       (mm day-1) <- q99  (99th-percentile)
  * Mean dry-spell duration  (days)     <- dry_spell_mean
  * Maximum dry-spell length (days)     <- dry_spell_max
  * Number of dry spells     (count)    <- n_dry_spells
  * Richards-Baker flashiness (-)       <- flashiness_rb
  * Number of flood events   (count)    <- n_flood_events
  * FDC slope                (-)        <- slope_fdc
  * Q5/Q95 flow ratio        (-)        <- q5_q95_ratio
  * 20-year flood Q20        (mm day-1) <- flood_q20
  * 10-year flood Q10        (mm day-1) <- flood_q10
  * 7-day low flow  MAM7     (mm day-1) <- mam7
  * 7Q10 low flow            (mm day-1) <- q7_10
  * Low-flow deficit         (mm yr-1)  <- low_q_deficit
  * Runoff ratio  Q/P        (-)        <- runoff_ratio
If stats.parquet is missing/empty for a station, the original six baseline
*flow* scalars are recomputed from that engine's baseline series in
qsim.nc (exact, since they are deterministic from that series). The
dry-spell / flashiness / flood-event metrics are NOT recomputed in the
qsim.nc fallback (the original event-extraction logic lives in the
apply_deltas pipeline, not here); those stations get NaN for those
columns and are silently dropped by the per-index `dropna` filter. In
practice every station has stats.parquet, so the fallback only matters
for edge cases.

Two-stage design (same as fig02 / fig05)
----------------------------------------
Stage 1 (cheap, but cached for consistency): scan every station, pull the
        three baseline values for BOTH engines + lon/lat, cache parquet/csv.
Stage 2 (cheap): load the cache and plot.
Keep RECOMPUTE = False to re-style the plot in seconds.

Run (python only):
    python paper/figures/scripts/baseline.py            # all stations
    python paper/figures/scripts/baseline.py 200        # quick test: first 200
"""

import os
import re
import sys
import glob
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# add BOTH paper/figures/scripts/ (this dir, for the fig02 sibling import) and the
# repo root (three dirs up, where config.py / apply_deltas_utils.py live) to
# sys.path, so the script imports cleanly when launched as
# `python paper/figures/scripts/figXX.py` (mirrors fig02).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config import CONFIG
# Nature-style bold top-left panel letters (multi-panel figures only)
from figstyle import add_panel_letters, percentile_label

# Reuse fig02's map-drawing helper so all figures share one map style.
# fig02 reads sys.argv at import time (MAX_STATIONS); neutralise argv first.
_argv_bak = sys.argv
sys.argv = sys.argv[:1]
try:
    from fig02_precip_change import _scatter
finally:
    sys.argv = _argv_bak

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===========================================================================
#  USER SETTINGS  (edit these)
# ===========================================================================

# One row per engine, in this order. stats.parquet / qsim.nc carry both.
ENGINES = ["hbv", "lstm"]

# qmean is mean *daily* flow (mm day-1); multiply by this for mm yr-1.
DAYS_PER_YEAR = 365.25

# First-N stations only (int) for a quick test; None = all. Also overridable
# from the command line:  python baseline.py 200
# --boxplots-only regenerates ONLY the index_boxplots_n{n}sta.png summary from
# the existing cache, skipping the expensive cartopy map families — a quick
# restyle of that one figure.
BOXPLOTS_ONLY = "--boxplots-only" in sys.argv
_pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAX_STATIONS = int(_pos_args[0]) if _pos_args else None

# Recompute the per-station cache even if it already exists.
RECOMPUTE = False

# Stage-1 parallelism: number of worker processes (fork Pool over stations).
# Derived from the SLURM allocation (SLURM_CPUS_PER_TASK, set by
# --cpus-per-task) so it always matches the cores granted and never drifts from
# the job script; off SLURM (local) falls back to os.cpu_count()-1.
NUM_WORKERS = (int(os.environ["SLURM_CPUS_PER_TASK"])
               if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit()
               else max(1, (os.cpu_count() or 1) - 1))

# Horizontal / vertical gap between map panels. Robinson axes are
# equal-aspect, so the map is centred in its box; FIG_H below is sized so the
# box height ~= the map height (otherwise a tall box leaves a big empty band
# that no amount of HSPACE shrinking removes). Keep HSPACE small.
WSPACE, HSPACE = 0.03, 0.02

# Figure width (inches). All fig04 PNGs share this width; the height of each
# layout is derived from FIG_W and the Robinson ~0.507 height/width ratio so
# the maps fill their axes box without leaving a vertical band.
FIG_W = 13.0

# Inset Köppen-Geiger boxplot on each map: one box per major KG zone, drawn
# in the empty lower-left (ocean) corner of the Robinson map. Adapted from
# fig02_precip_change.py (copied, not imported, by request). NOTE the two
# differences from fig02, forced by the data being absolute magnitudes
# rather than signed changes: (1) the inset y-axis is LOG when the map is
# log (these indices span orders of magnitude — a linear box collapses to a
# sliver); (2) the 0-reference line fig02 draws is dropped (0 is not a
# meaningful reference for a positive magnitude and is invalid on a log
# axis). On a log axis non-positive values (e.g. Q05 = 0 for ephemeral
# rivers) cannot be shown and are excluded; the per-zone n under each box
# reports how many catchments actually contributed.
ADD_ZONE_BOXPLOTS = False
KG_ZONES = ["A", "B", "C", "D", "E"]   # tropical, arid, temperate, cold, polar
# inset position within each map axes (axes fraction: x0, y0, w, h)
BOX_RECT = [0.50, 0.09, 0.30, 0.30]
# opacity of the inset background only (boxes/whiskers stay opaque):
# 0.0 = fully transparent (markers below visible), 1.0 = solid white.
BOX_FACE_ALPHA = 0.0

# PNGs (combined 3x2 and per-index 1x2) go to `paper/figures/baseline/`.
# Parquet/csv cache goes to `paper/figures/fig_data/` — the shared cache
# convention with fig02/fig05. Splitting the dirs means that wiping
# `figures/baseline/` to re-render plots does NOT lose the expensive Stage-1
# cache, and that LaTeX never sees `.parquet/.csv` in its include path.
# this script lives in paper/figures/scripts/, so the figures root is its parent's parent
_FIGROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_FIGROOT, "baseline")
CACHE_DIR = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")  # per-method cache on /ibex (avoids cross-method contamination)
CACHE_FPATH = os.path.join(CACHE_DIR, "fig04_baseline.parquet")
# {n} filled in main() with the station count actually plotted (known only
# after Stage 1); the cache name stays fixed so the skip path can resolve it.
OUT_PNG_TMPL = os.path.join(
    FIG_DIR, "baseline_indices_n{n}sta.png")

# Indices: (base column, panel label, colormap, slug). All are absolute
# magnitudes, strongly right-skewed across catchments -> sequential cmap +
# log scale. Per-engine cache columns are "<engine>_<base>". The `slug` is
# used in the per-index PNG filename `fig04_{slug}_n{n}sta.png`.
#
# The first N_COMBINED entries make up the combined 3x2 PNG (those are the
# three "Figure 4" indices the .tex outline specifies — preserved). All
# entries get a 1x2 per-index PNG; extras below N_COMBINED only appear in
# the per-index family.
INDICES = [
    # --- combined-plot trio (the .tex's Figure 4) ---
    ("runoff_mm_yr",     "Mean annual runoff (mm yr$^{-1}$)",      "YlGnBu", "runoff"),
    ("peakflow_mm_day",  "Mean annual maximum (mm day$^{-1}$)",    "YlOrRd", "peakflow"),
    # The percentile rows take their symbol from figstyle's project-wide table
    # rather than a hand-typed "Q05"/"Q50"/"Q95"/"Q99". A bare Q95 collides with
    # the flow-duration convention, in which Q95 is a LOW flow, whereas here
    # q95 = quantile(0.95), a high flow.
    ("q05_mm_day",       f"Low flow {percentile_label('q05').short} "
                         "(mm day$^{-1}$)",                        "Blues",  "q05"),
    # --- flow distribution extras ---
    ("annmin_mm_day",    "Mean annual minimum (mm day$^{-1}$)",    "PuBu",   "annmin"),
    ("q50_mm_day",       f"Median flow {percentile_label('q50').short} "
                         "(mm day$^{-1}$)",                        "Greens", "q50"),
    ("q95_mm_day",       f"High flow {percentile_label('q95').short} "
                         "(mm day$^{-1}$)",                        "OrRd",   "q95"),
    ("q99_mm_day",       f"Very high flow {percentile_label('q99').short} "
                         "(mm day$^{-1}$)",                        "Reds",   "q99"),
    ("qstd_mm_day",      "Daily-flow std (mm day$^{-1}$)",         "Greys",  "qstd"),
    # --- dry-spell metrics (days / count) ---
    ("dryspell_mean",    "Mean dry-spell duration (days)",         "YlOrBr", "dryspell_mean"),
    ("dryspell_max",     "Maximum dry-spell length (days)",        "Oranges","dryspell_max"),
    ("n_dryspell",       "Number of dry spells (-)",               "Reds",   "n_dryspell"),
    # --- flashiness & flood frequency ---
    ("flashiness",       "Richards–Baker flashiness index (-)",    "RdPu",   "flashiness"),
    ("n_flood",          "Number of flood events (-)",             "Purples","n_flood"),
    # --- order-independent FDC-shape / flood-magnitude signatures. Unlike the
    #     Richards-Baker index and the event counts above, these are functions of
    #     the flow DISTRIBUTION only (no day-to-day sequencing), so they transfer
    #     faithfully through the qqmap delta method into the Figure-5 change maps.
    ("slope_fdc",        "FDC slope (-)",                          "BuPu",   "slope_fdc"),
    # Written high-over-low. apply_deltas_utils.flow_duration_signatures builds
    # this as quantile(0.95) / quantile(0.05), i.e. high flow over low flow, but
    # the old label "Q5/Q95" was written in the EXCEEDANCE convention while every
    # neighbouring index on the same axis was written in the percentile one, so
    # the same symbol Q95 meant a high flow in one panel and a low flow in the
    # next. The value is unchanged; only the naming was inverted.
    ("q5_q95_ratio",     f"{percentile_label('q5_q95_ratio').short} "
                         "flow ratio (-)",                         "PuRd",   "q5_q95_ratio"),
    # Flood return levels use the RL notation of the manuscript's streamflow-index
    # table, NOT "Q20"/"Q10" — those would collide with the flow-duration-curve
    # percentiles Q05/Q50/Q95/Q99 above, which are a different quantity.
    ("flood_q20",        "20-year flood RL20 (mm day$^{-1}$)",     "Reds",   "flood_q20"),
    ("flood_q10",        "10-year flood RL10 (mm day$^{-1}$)",     "OrRd",   "flood_q10"),
    # --- low-flow drought / water-balance signatures (order-independent
    #     counterparts of the dry-spell family; likewise transfer faithfully). ---
    ("mam7",             "7-day low flow MAM7 (mm day$^{-1}$)",    "Blues",  "mam7"),
    ("q7_10",            "7Q10 low flow (mm day$^{-1}$)",          "GnBu",   "q7_10"),
    ("low_q_deficit",    "Low-flow deficit (mm yr$^{-1}$)",        "Oranges","low_q_deficit"),
    ("runoff_ratio",     "Runoff ratio Q/P (-)",                   "Greens", "runoff_ratio"),
    # --- remaining members of the 29-metric streamflow registry
    #     (paper/figures/scripts/flow_metrics.py FLOW_METRICS). These are carried
    #     ONLY by the boxplot summaries (all ten are in SKIP_PER_INDEX_SLUGS and
    #     in no COMBINED_GROUPS entry), so that every metric whose future change
    #     the manuscript analyses also has its present-day distribution shown. ---
    ("flood_q50",        "50-year flood RL50 (mm day$^{-1}$)",     "Reds",   "flood_q50"),
    ("q7min_min",        "Minimum 7-day flow (mm day$^{-1}$)",     "PuBu",   "q7min_min"),
    # NOTE low_q_freq at BASELINE is ~0.05 x 365 = 18.3 d yr-1 for every
    # catchment BY CONSTRUCTION (the threshold is that catchment's own baseline
    # Q05), so its box is near-degenerate; it is kept for completeness of the
    # registry and only becomes informative as a future-change metric.
    ("low_q_freq",       "Low-flow day frequency (d yr$^{-1}$)",   "YlOrBr", "low_q_freq"),
    ("high_q_freq",      "High-flow day frequency (d yr$^{-1}$)",  "Purples","high_q_freq"),
    ("high_q_dur",       "High-flow event duration (days)",        "BuPu",   "high_q_dur"),
    # Same inversion as q5_q95_ratio above: built as quantile(0.90) /
    # quantile(0.10), so high over low in the percentile convention.
    ("q10_q90_ratio",    f"{percentile_label('q10_q90_ratio').short} "
                         "flow ratio (-)",                         "PuRd",   "q10_q90_ratio"),
    ("flashiness_event", "Event flashiness (mm day$^{-2}$)",       "RdPu",   "flashiness_event"),
    ("flashiness_frt",   "Flood rising time FRT (days)",           "BuPu",   "flashiness_frt"),
    ("flashiness_sfd",   "Storm-flow depth SFD (mm day$^{-1}$)",   "OrRd",   "flashiness_sfd"),
    ("hfd_mean",         "Half-flow date (day of water year)",     "cividis","hfd_mean"),
]
N_COMBINED = 3   # first N_COMBINED entries of INDICES drive the combined PNG

# Thematic combined figures beyond the .tex trio. Each entry is
# (output-filename slug, [index slugs in row order], shared_cmap). Rows reuse
# the INDICES definitions (label / colour-scale) resolved by slug, so a single
# source of truth drives every figure. `shared_cmap` (when not None) overrides
# every row's per-index colormap so the whole figure uses ONE palette and the
# rows become easy to compare at a glance; pass None to keep each row's own
# INDICES cmap. The low-flow / dry-spell group puts Q05 — which *defines* the
# dry-day threshold (a dry day is qsim < baseline Q05) — above the two spell
# metrics it controls, so the first three rows read as a causal chain
# (threshold -> persistence -> frequency), with the runoff ratio Q/P last as
# the water-balance context. cmap=None (rather than one shared warm "dryness"
# palette) because the runoff ratio is a WETNESS measure: a dark brown high
# value would read as dry.
COMBINED_GROUPS = [
    ("lowflow_dryspell",
     ["q05", "dryspell_mean", "n_dryspell", "runoff_ratio"], None),
    # Mid/high flow quantiles (Q50, Q95, Q99) + Richards-Baker flashiness, as a
    # 4x2 (rows = indices, cols = engines). cmap=None -> each row keeps its own
    # INDICES colormap (these are different quantities / units, so no shared
    # palette, unlike lowflow_dryspell).
    ("flow_quantiles_flashiness",
     ["q50", "q95", "q99", "flashiness"], None),
    # Flow-variability & extremes group: daily-flow std (variability), number
    # of flood events (high-flow frequency) and annual-minimum flow (low-flow
    # extreme), as a 3x2 (rows = indices, cols = engines). Replaces the former
    # standalone qstd/n_flood/annmin per-index PNGs. cmap=None -> each row keeps
    # its own INDICES colormap (different quantities / units, no shared palette).
    ("flow_variability_extremes",
     ["qstd", "n_flood", "annmin"], None),
    # FDC-shape & flood-magnitude signatures: FDC slope, high/low percentile
    # ratio (Q5/Q95) and the 20- and 10-year GEV flood return levels, as a 4x2
    # (rows = indices, cols = engines). All are order-independent (distribution
    # only), so unlike Richards-Baker flashiness / flood-event count they transfer
    # faithfully through the qqmap delta method. cmap=None -> per-row cmap.
    ("fdc_flood_signatures",
     ["slope_fdc", "q5_q95_ratio", "flood_q20", "flood_q10"], None),
    # Low-flow drought severity: the 7Q10 design low flow (10-year-recurrence
    # 7-day low), the mean annual 7-day low flow (MAM7), the mean annual
    # low-flow deficit volume and the maximum dry-spell length, as a 4x2 — the
    # magnitude, volume and duration-extreme views of the same drought. The
    # first three are order-independent (distributional) and so transfer
    # faithfully through the qqmap delta method; the maximum dry-spell length
    # does NOT (it depends on day-to-day sequencing), so treat that row as
    # baseline context rather than as a transferable change signal.
    # (Low-flow frequency — days per year below the baseline Q05 — is
    # deliberately NOT here: it is ~5% x 365 = 18.3 d/yr for every catchment BY
    # CONSTRUCTION at baseline, so it is a flat, uninformative map; it only
    # becomes meaningful as a future-change metric in Figure 5.)
    # cmap=None -> per-row cmap.
    ("lowflow_waterbalance",
     ["q7_10", "mam7", "low_q_deficit", "dryspell_max"], None),
]

# Per-index 1x2 PNGs are skipped for these slugs: each already appears either
# in the combined baseline_indices trio (runoff/peakflow/q05) or as a row of a
# COMBINED_GROUPS figure above, so the standalone files are redundant. With the
# trio + every COMBINED_GROUPS slug listed here, no standalone per-index PNG is
# written at all (the per-index loop below becomes a no-op); the machinery is
# kept so an individual index can be re-enabled by removing its slug.
SKIP_PER_INDEX_SLUGS = {
    "runoff", "peakflow", "q05",                     # -> baseline_indices trio
    "dryspell_mean", "n_dryspell", "runoff_ratio",   # -> lowflow_dryspell
    "q50", "q95", "q99", "flashiness",               # -> flow_quantiles_flashiness
    "qstd", "n_flood", "annmin",                     # -> flow_variability_extremes
    "slope_fdc", "q5_q95_ratio", "flood_q20", "flood_q10",  # -> fdc_flood_signatures
    "q7_10", "mam7", "low_q_deficit", "dryspell_max",       # -> lowflow_waterbalance
    # registry completions: boxplot summaries only, no map of their own
    "flood_q50", "q7min_min", "low_q_freq", "high_q_freq", "high_q_dur",
    "q10_q90_ratio", "flashiness_event", "flashiness_frt", "flashiness_sfd",
    "hfd_mean",
}

# --- Index boxplot summaries (two figures) ---
# One small panel PER index in a grid of boxplot panels. Each
# panel holds one HBV and one LSTM box and gets its OWN y-axis, so every index
# is shown on a scale appropriate to its units (mm yr-1, mm day-1, days, counts,
# dimensionless) instead of being forced onto a single shared axis. The panel
# title is the index's INDICES label; the x ticks are the engine names.
# Per-engine box fill colours (one hue per engine):
ENGINE_BOX_COLORS = {"hbv": "#4878CF", "lstm": "#EE854A"}
# A log axis cannot show <=0. Instead of DROPPING those catchments, clip every
# value below this small positive floor UP to it (np.maximum), so the low tail
# (incl. LSTM's negative low flows, interpretation §4) piles visibly at the
# bottom of the box rather than vanishing and biasing the box upward. 0.01
# matches the Q05 map's Q05_FLOOR ("effectively zero" daily flow); it only bites
# the sub-0.01 low-flow indices (Q05, annmin, Q50) and is a no-op for the
# larger-magnitude indices (runoff, peakflow, Q95, Q99, qstd).
BOX_FLOOR = 0.01

# The boxplot summary is split into TWO figures, by slug:
#   * BOXPLOT_GROUPS[0] — the 14 indices listed in the manuscript's streamflow-
#     index table (Supplementary Table S6, paper/tables/streamflow_indices_table.tex),
#     in that table's own order (average conditions -> high flows/floods ->
#     low flows -> distribution shape -> flashiness).
#   * BOXPLOT_GROUPS[1] — every OTHER index this script carries: the remaining 15
#     members of the 29-metric streamflow registry (flow_metrics.FLOW_METRICS)
#     plus two baseline-only diagnostics that are not in that registry (the
#     dry-spell count and the flood-event count).
# Each entry is (filename slug, [index slugs in panel order]).
BOXPLOT_GROUPS = [
    ("index_boxplots_tabS6", [
        # average conditions
        "runoff", "q50", "runoff_ratio",
        # high flows and floods
        "q95", "q99", "peakflow", "flood_q10", "flood_q20", "flood_q50",
        # low flows and drought
        "q05", "annmin",
        # distribution shape
        "qstd", "slope_fdc",
        # hydrograph flashiness
        "flashiness",
    ]),
    ("index_boxplots_extra", [
        # low flow / drought
        "mam7", "q7_10", "q7min_min", "low_q_freq", "low_q_deficit",
        "dryspell_mean", "dryspell_max", "n_dryspell",
        # high flow
        "high_q_freq", "high_q_dur", "n_flood",
        # distribution shape
        "q5_q95_ratio", "q10_q90_ratio",
        # event flashiness and timing
        "flashiness_event", "flashiness_frt", "flashiness_sfd", "hfd_mean",
    ]),
]

# ===========================================================================
#  Stage 1 — per-station baseline values (both engines) -> cache
# ===========================================================================


def _qsim_baseline_series(qvar) -> pd.Series:
    """Baseline realization from qsim.nc as a clean datetime-indexed Series."""
    s = (qvar.sel(scenario="baseline", model="baseline",
                  member="baseline", period="historical")
              .to_series().dropna())
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def _from_series(q: pd.Series) -> dict:
    """All baseline indices, recomputed from a qsim series (qsim.nc fallback
    path). Mirrors the scalar fields in `compute_flow_stats` for the basic
    flow stats; dry-spell / flashiness / flood-event metrics are left NaN
    here (their event-extraction logic lives in the apply_deltas pipeline,
    not here, and the fallback is rarely triggered in practice)."""
    am  = q.groupby(q.index.year).max()
    amn = q.groupby(q.index.year).min()
    return {
        "runoff_mm_yr":    float(q.mean()) * DAYS_PER_YEAR,
        "qstd_mm_day":     float(q.std()),
        "peakflow_mm_day": float(am.mean())  if len(am)  else np.nan,
        "annmin_mm_day":   float(amn.mean()) if len(amn) else np.nan,
        "q05_mm_day":      float(q.quantile(0.05)),
        "q50_mm_day":      float(q.quantile(0.50)),
        "q95_mm_day":      float(q.quantile(0.95)),
        "q99_mm_day":      float(q.quantile(0.99)),
        "dryspell_mean":   np.nan,
        "dryspell_max":    np.nan,
        "n_dryspell":      np.nan,
        "flashiness":      np.nan,
        "n_flood":         np.nan,
        # marginal signatures — left NaN in the qsim.nc fallback (their fitting /
        # threshold logic lives in the apply_deltas pipeline, and every station
        # has stats.parquet in practice, so this path is essentially never hit)
        "slope_fdc":       np.nan,
        "q5_q95_ratio":    np.nan,
        "flood_q20":       np.nan,
        "flood_q10":       np.nan,
        "mam7":            np.nan,
        "q7_10":           np.nan,
        "low_q_deficit":   np.nan,
        "runoff_ratio":    np.nan,
        # remaining members of the 29-metric registry (boxplot-only)
        "flood_q50":       np.nan,
        "q7min_min":       np.nan,
        "low_q_freq":      np.nan,
        "high_q_freq":     np.nan,
        "high_q_dur":      np.nan,
        "q10_q90_ratio":   np.nan,
        "flashiness_event": np.nan,
        "flashiness_frt":  np.nan,
        "flashiness_sfd":  np.nan,
        "hfd_mean":        np.nan,
    }


def _to_num(x):
    """None/NaN-safe float. The marginal signatures (slope_fdc, flood_qT, mam7,
    runoff_ratio, ...) are legitimately ``None`` in the pipeline for degenerate
    catchments (e.g. slope_fdc where a percentile flow is <= 0, or a failed GEV
    fit); ``float(None)`` would raise and kill the worker, so convert None /
    non-finite to ``np.nan`` (silently dropped later by the per-index dropna)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _engine_vals_from_stats(stats, engine):
    """All baseline scalars for `engine` from a stats frame, or None if the
    baseline row is absent. Pulls every scalar baseline field the apply_deltas
    pipeline writes: basic flow stats (qmean, qstd, q05/q50/q95,
    mean_annual_max/_min), dry-spell metrics (mean, max, count), the
    Richards-Baker flashiness index and the flood-event count, plus the
    order-independent marginal signatures (FDC slope, Q5/Q95 ratio, GEV flood
    return levels Q10/Q20, 7-day low flow MAM7, low-flow frequency/deficit and
    the runoff ratio Q/P)."""
    if stats is None:
        return None
    b = stats[(stats["engine"] == engine)
              & (stats["scenario"] == "baseline")
              & (stats["period"].astype(str) == "historical")]
    if b.empty:
        return None
    r = b.iloc[0]
    return {
        "runoff_mm_yr":    float(r.get("qmean", np.nan)) * DAYS_PER_YEAR,
        "qstd_mm_day":     float(r.get("qstd", np.nan)),
        "peakflow_mm_day": float(r.get("mean_annual_max", np.nan)),
        "annmin_mm_day":   float(r.get("mean_annual_min", np.nan)),
        "q05_mm_day":      float(r.get("q05", np.nan)),
        "q50_mm_day":      float(r.get("q50", np.nan)),
        "q95_mm_day":      float(r.get("q95", np.nan)),
        "q99_mm_day":      float(r.get("q99", np.nan)),
        "dryspell_mean":   float(r.get("dry_spell_mean", np.nan)),
        "dryspell_max":    float(r.get("dry_spell_max", np.nan)),
        "n_dryspell":      float(r.get("n_dry_spells", np.nan)),
        "flashiness":      float(r.get("flashiness_rb", np.nan)),
        "n_flood":         float(r.get("n_flood_events", np.nan)),
        # order-independent FDC-shape / flood-magnitude signatures (None-safe:
        # these keys are legitimately None for degenerate catchments)
        "slope_fdc":       _to_num(r.get("slope_fdc")),
        "q5_q95_ratio":    _to_num(r.get("q5_q95_ratio")),
        "flood_q20":       _to_num(r.get("flood_q20")),
        "flood_q10":       _to_num(r.get("flood_q10")),
        # low-flow drought / water-balance signatures
        "mam7":            _to_num(r.get("mam7")),
        "q7_10":           _to_num(r.get("q7_10")),
        "low_q_deficit":   _to_num(r.get("low_q_deficit")),
        "runoff_ratio":    _to_num(r.get("runoff_ratio")),
        # remaining members of the 29-metric streamflow registry (boxplot-only).
        # Written by the live pipeline (flood_return_levels, fdc_signatures,
        # low_flow_signatures, high_flow_frequency, flashiness_stats,
        # half_flow_date) and, for older stations, by backfill_streamflow_stats.py.
        "flood_q50":       _to_num(r.get("flood_q50")),
        "q7min_min":       _to_num(r.get("q7min_min")),
        "low_q_freq":      _to_num(r.get("low_q_freq")),
        "high_q_freq":     _to_num(r.get("high_q_freq")),
        "high_q_dur":      _to_num(r.get("high_q_dur")),
        "q10_q90_ratio":   _to_num(r.get("q10_q90_ratio")),
        "flashiness_event": _to_num(r.get("flashiness_event")),
        "flashiness_frt":  _to_num(r.get("flashiness_event_frt")),
        "flashiness_sfd":  _to_num(r.get("flashiness_event_sfd")),
        "hfd_mean":        _to_num(r.get("hfd_mean")),
    }


_F04_REF = {}


def _f04_worker_init(gscad_path):
    """Per-worker init (fork Pool): load the static table once."""
    _F04_REF["static"] = pd.read_csv(
        os.path.join(gscad_path, "dataframes", "static.csv"),
        comment="#", index_col=0)


def _f04_process_station(stn):
    """Worker entry point — pull both engines' baseline indices for ONE
    station. Returns (rec_or_None, acc); acc carries the skip / per-engine
    source counts to fold into the global tally. Body mirrors the old serial
    loop verbatim. Reference static table comes from _F04_REF."""
    static = _F04_REF["static"]
    acc = {"not_in_static": 0, "no_baseline_any_engine": 0}
    for e in ENGINES:
        acc[f"{e}_stats"] = 0
        acc[f"{e}_qsim"] = 0
        acc[f"{e}_ok"] = 0

    sdir = os.path.join(CONFIG.out_path, CONFIG.delta_method, stn)
    if stn not in static.index:
        acc["not_in_static"] += 1
        return None, acc
    lon = float(static.loc[stn, "long"])
    lat = float(static.loc[stn, "lat"])

    # read stats once per station, reuse for every engine
    stats = None
    spq = os.path.join(sdir, "stats.parquet")
    try:
        if os.path.isfile(spq):
            stats = pd.read_parquet(spq)
    except Exception as e:
        logger.warning(f"  {stn}: stats read failed ({e})")

    # qsim.nc opened lazily, at most once per station, reused per engine
    qf = os.path.join(sdir, "qsim.nc")
    qds = None

    rec = {"station": stn, "lon": lon, "lat": lat}
    any_engine = False
    try:
        for engine in ENGINES:
            vals = _engine_vals_from_stats(stats, engine)
            if vals is not None:
                acc[f"{engine}_stats"] += 1
            else:
                if qds is None and os.path.isfile(qf):
                    try:
                        qds = xr.open_dataset(qf)
                    except Exception as e:
                        logger.warning(f"  {stn}: qsim open failed ({e})")
                        qds = False  # don't retry this station
                if qds not in (None, False) and f"{engine}_qsim" in qds:
                    try:
                        qb = _qsim_baseline_series(qds[f"{engine}_qsim"])
                        if len(qb):
                            vals = _from_series(qb)
                            acc[f"{engine}_qsim"] += 1
                    except Exception as e:
                        logger.warning(f"  {stn}: {engine} qsim fallback "
                                       f"failed ({e})")
            if vals is None:
                continue
            if np.any(np.isfinite(list(vals.values()))):
                rec.update({f"{engine}_{k}": v for k, v in vals.items()})
                any_engine = True
                acc[f"{engine}_ok"] += 1
    finally:
        if qds not in (None, False):
            qds.close()

    if not any_engine:
        acc["no_baseline_any_engine"] += 1
        return None, acc
    return rec, acc


def compute_cache() -> pd.DataFrame:
    stat_files = sorted(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "stats.parquet")))
    qsim_files = sorted(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "qsim.nc")))
    stations = sorted({os.path.basename(os.path.dirname(p))
                       for p in (stat_files + qsim_files)})
    if MAX_STATIONS is not None:
        stations = stations[:MAX_STATIONS]
    n_workers = (NUM_WORKERS if NUM_WORKERS is not None
                 else max(1, (os.cpu_count() or 1) - 1))
    logger.info(f"{len(stations)} station dirs to scan (engines={ENGINES}"
                f"{'' if MAX_STATIONS is None else f', capped at {MAX_STATIONS}'}"
                f"; {n_workers} worker process{'es' if n_workers > 1 else ''})")

    skip = {"not_in_static": 0, "no_baseline_any_engine": 0}
    src = {f"{e}_stats": 0 for e in ENGINES}
    src.update({f"{e}_qsim": 0 for e in ENGINES})
    n_engine_ok = {e: 0 for e in ENGINES}
    records = []

    if n_workers <= 1:
        _f04_worker_init(CONFIG.gscad_path)
        results_iter = (_f04_process_station(s) for s in stations)
        pool = None
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=n_workers,
                        initializer=_f04_worker_init,
                        initargs=(CONFIG.gscad_path,))
        results_iter = pool.imap_unordered(_f04_process_station, stations,
                                           chunksize=8)

    try:
        for j, (rec, acc) in enumerate(results_iter):
            if rec is not None:
                records.append(rec)
            skip["not_in_static"] += acc["not_in_static"]
            skip["no_baseline_any_engine"] += acc["no_baseline_any_engine"]
            for e in ENGINES:
                src[f"{e}_stats"] += acc[f"{e}_stats"]
                src[f"{e}_qsim"] += acc[f"{e}_qsim"]
                n_engine_ok[e] += acc[f"{e}_ok"]
            if (j + 1) % 500 == 0:
                logger.info(f"  [{j+1}/{len(stations)}] processed")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No baseline records produced — check ENGINES and "
                           "CONFIG.out_path.")
    logger.info("Stage 1 station accounting:")
    logger.info(f"  scanned : {len(stations)}")
    for reason, n in skip.items():
        logger.info(f"  skipped [{reason}] : {n}")
    for e in ENGINES:
        logger.info(f"  {e}: {n_engine_ok[e]} stations "
                    f"(stats={src[f'{e}_stats']}, qsim={src[f'{e}_qsim']})")
    logger.info(f"  USED (>=1 engine) : {df['station'].nunique()} stations")

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(CACHE_FPATH)
    df.to_csv(CACHE_FPATH.replace(".parquet", ".csv"), index=False)
    logger.info(f"Cached -> {CACHE_FPATH}")
    return df


# ===========================================================================
#  Stage 2 — plot (2x3: engine rows, index columns)
# ===========================================================================


# Indices whose 2nd-98th percentile range spans less than this many decades
# get a LINEAR colour scale instead of LogNorm. Reason: matplotlib's
# LogLocator only places major ticks at integer powers of ten, so a colorbar
# spanning <1.5 decades shows only 0-1 labelled ticks (the rest are minor
# "3x10^k" annotations). The 1.5-decade threshold pushes flashiness, n_flood,
# and the dry-spell count/duration metrics onto a linear scale (5+ readable
# plain-decimal ticks) while leaving runoff / peakflow / Q05 / Q50 / Q95 /
# annmin / qstd on log (their 3-6 decade spread truly needs it).
LOG_SPAN_THRESHOLD = 1.5

# Q05 colorbar floor: in extremely arid catchments Q05 of streamflow can be
# ~1e‑9 mm day‑1 (effectively zero), which pins the LogNorm lower end and
# squashes the readable range. Force the bar to start at Q05_FLOOR; below‑
# floor catchments render in Q05_UNDER_COLOR and the bar gets a downward
# arrow (extend='min'). Hard‑coded to Q05 only — kept deliberately minimal.
Q05_FLOOR       = 0.01      # mm day‑1
Q05_UNDER_COLOR = "#dcdcdc" # pale neutral grey, distinct from the "Blues" cmap

# --- Heavy-tail colorbar clamps ------------------------------------------
# The maps share ONE log colour scale across HBV and LSTM so the two engines
# are directly comparable within a panel. For heavy-tailed indices that shared
# scale is blown out by one engine's extreme tail, washing the other engine's
# panel into a single colour: (i) the low-flow magnitudes (MAM7, 7Q10, low-flow
# deficit) have a near-zero arid tail that pins the LogNorm minimum ~10 decades
# down; (ii) the Q5/Q95 ratio explodes to ~1e7 where the low flow approaches
# zero. Clamp such colorbars to at most SCALE_MAX_DECADES decades — a FLOOR
# (raise vmin, arrow down) for the low-flow magnitudes, a CAP (lower vmax, arrow
# up) for the ratio — so the bulk of the distribution is resolved and the
# extreme tail collapses into the under/over colour. Q05 keeps its own explicit
# 0.01 "effectively-zero daily flow" floor (a physical threshold, not a decade
# rule).
SCALE_MAX_DECADES  = 3.0
FLOOR_DECADE_SLUGS = {"mam7", "q7_10", "low_q_deficit"}
CAP_DECADE_SLUGS   = {"q5_q95_ratio"}
SCALE_UNDER_COLOR  = "#dcdcdc"   # below-floor (effectively dry) catchments
SCALE_OVER_COLOR   = "#3f007d"   # above-cap (extreme intermittent) catchments


def _scale_override(norm, cmap, slug):
    """Apply the per-index floor/cap clamp to a colour norm.

    Returns ``(norm, cmap, extend)``. A no-op (``extend='neither'``) for a
    linear norm or an unlisted slug, so most indices are untouched. Q05 keeps
    its explicit floor; the FLOOR/CAP slugs are clamped to SCALE_MAX_DECADES
    and get the under/over colour + matching colorbar arrow.
    """
    if not isinstance(norm, LogNorm):
        return norm, cmap, "neither"
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    if slug == "q05":
        cmap = plt.get_cmap(cmap).with_extremes(under=Q05_UNDER_COLOR)
        return LogNorm(vmin=Q05_FLOOR, vmax=vmax), cmap, "min"
    if slug in FLOOR_DECADE_SLUGS:
        vmin_new = max(vmin, vmax / 10.0 ** SCALE_MAX_DECADES)
        if vmin_new > vmin:
            cmap = plt.get_cmap(cmap).with_extremes(under=SCALE_UNDER_COLOR)
            return LogNorm(vmin=vmin_new, vmax=vmax), cmap, "min"
    if slug in CAP_DECADE_SLUGS:
        vmax_new = min(vmax, vmin * 10.0 ** SCALE_MAX_DECADES)
        if vmax_new < vmax:
            cmap = plt.get_cmap(cmap).with_extremes(over=SCALE_OVER_COLOR)
            return LogNorm(vmin=vmin, vmax=vmax_new), cmap, "max"
    return norm, cmap, "neither"


def _robust_norm(vals):
    """Robust colour norm — log when justified by the data span, linear
    otherwise.

    Returns a ``LogNorm`` over the 2nd / 98th percentile of the positive
    values when (a) the data is mostly positive AND (b) those positives span
    at least ``LOG_SPAN_THRESHOLD`` decades. Otherwise returns a linear
    ``Normalize`` over the 2-98 percentile of the full finite distribution.
    The linear fallback fires for indices like flashiness, n_flood and the
    dry-spell counts/durations, whose 2-98 range is under one decade and
    produces a near-empty log colorbar.
    """
    v = np.asarray(vals, dtype=float)
    finite = v[np.isfinite(v)]
    pos = finite[finite > 0]
    use_log = (
        pos.size >= max(10, 0.2 * finite.size)
        and np.log10(np.nanpercentile(pos, 98) /
                     max(np.nanpercentile(pos, 2), 1e-30))
            >= LOG_SPAN_THRESHOLD
    )
    if use_log:
        vmin = float(np.nanpercentile(pos, 2))
        vmax = float(np.nanpercentile(pos, 98))
        if not (np.isfinite(vmin) and vmin > 0):
            vmin = float(pos.min())
        if not (np.isfinite(vmax) and vmax > vmin):
            vmax = vmin * 10.0
        return LogNorm(vmin=vmin, vmax=vmax)
    lo = float(np.nanpercentile(finite, 2))
    hi = float(np.nanpercentile(finite, 98))
    if not (np.isfinite(hi) and hi > lo):
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi)


def _fmt_decimal(x):
    """Plain-decimal label, no '1e+03' switch. Used for LogNorm colorbar
    ticks where matplotlib would otherwise render '10**k' / '1e3'."""
    if not np.isfinite(x) or x == 0:
        return f"{x:g}"
    a = abs(x)
    if a >= 100:
        return f"{x:.0f}"
    if a >= 10:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.2f}"
    if a >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2g}"


def _three_ticks_log_cbar(cb, norm):
    """Pin exactly three tick labels on a LogNorm colorbar: vmin (start),
    geometric mean (visual middle), vmax (end), formatted as plain decimals.
    No-op when ``norm`` is linear — matplotlib's default ScalarFormatter
    already gives readable ticks for narrow-range data."""
    if not isinstance(norm, LogNorm):
        return
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    geo = (vmin * vmax) ** 0.5
    ticks = [vmin, geo, vmax]
    cb.set_ticks(ticks, labels=[_fmt_decimal(t) for t in ticks])
    cb.minorticks_off()


def _add_zone_box(ax, zones, values, log):
    """Inset boxplot of `values` split by major Köppen-Geiger zone, drawn in
    the empty lower-left corner of the map axes `ax`.

    Adapted from fig02_precip_change._add_zone_box. Differences (see the
    ADD_ZONE_BOXPLOTS comment): a log y-axis when `log` is True, no 0-line,
    and non-positive values dropped on the log axis (they cannot be drawn).
    """
    zones  = np.asarray(zones)
    values = np.asarray(values, dtype=float)
    inset = ax.inset_axes(BOX_RECT)
    counts = []
    for i, z in enumerate(KG_ZONES, start=1):
        v = values[(zones == z) & np.isfinite(values)]
        if log:
            v = v[v > 0]                      # log axis cannot show <= 0
        counts.append(len(v))
        if v.size == 0:                       # boxplot errors on empty input
            continue
        bp = inset.boxplot(v, positions=[i], widths=0.62,
                           showfliers=False, patch_artist=True)
        bp["boxes"][0].set(facecolor="0.85", linewidth=0.5)
        bp["medians"][0].set(color="black", linewidth=0.9)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(linewidth=0.5)
    if log:
        inset.set_yscale("log")
    allv = values[np.isfinite(values)]
    if log:
        allv = allv[allv > 0]
    if allv.size:                             # robust y-limits
        lo, hi = np.nanpercentile(allv, [1, 99])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            inset.set_ylim(lo, hi)
    inset.set_xlim(0.5, len(KG_ZONES) + 0.5)
    inset.set_xticks(range(1, len(KG_ZONES) + 1))
    inset.set_xticklabels([f"{z}\n{c}" for z, c in zip(KG_ZONES, counts)],
                          fontsize=7)
    inset.tick_params(axis="y", labelsize=7, length=2, pad=1)
    inset.tick_params(axis="x", length=0, pad=1)
    inset.set_facecolor((1, 1, 1, BOX_FACE_ALPHA))
    for side, sp in inset.spines.items():
        if side in ("top", "right"):
            sp.set_visible(False)             # open box: drop upper & right
        else:
            sp.set_linewidth(0.4)
    return inset


def plot_index_pair(df, col, label, cmap, slug, n_sta, proj):
    """Single-index, two-engine map: a 1x2 figure for one hydrological index
    (HBV left, LSTM right), saved as `fig04_{slug}_n{n}sta.png`. The combined
    3x2 figure in main() carries all three indices at once; this companion
    lets the manuscript embed each index alone at higher per-map size.

    Same colour-scale convention as the corresponding row of the combined
    figure: one log scale built from both engines so the two panels stay
    directly comparable, with one colorbar to the right of the row.
    """
    fig, axes = plt.subplots(
        1, len(ENGINES),
        figsize=(FIG_W, (FIG_W / len(ENGINES)) * 0.507 + 0.8),
        gridspec_kw=dict(wspace=WSPACE), **proj,
    )
    axes = np.atleast_1d(axes)

    stacked = np.concatenate([
        df[f"{e}_{col}"].to_numpy(dtype=float) for e in ENGINES])
    norm = _robust_norm(stacked)
    use_log = isinstance(norm, LogNorm)

    # Heavy-tail floor/cap clamp (Q05 floor, low-flow-magnitude floors, Q5/Q95
    # cap); no-op for linear norms and unlisted slugs.
    norm, cmap, extend = _scale_override(norm, cmap, slug)

    sc = None
    # The panel letter comes solely from add_panel_letters() below (bold, top
    # left); the title carries only the index and the engine, so a panel is not
    # lettered twice.
    for c, engine in enumerate(ENGINES):
        ec = f"{engine}_{col}"
        m = df.dropna(subset=[ec])
        m = m[np.isfinite(m[ec])]
        sc = _scatter(axes[c], m.lon, m.lat, m[ec].to_numpy(), cmap,
                      f"{label}, {engine.upper()}", norm=norm)
        if ADD_ZONE_BOXPLOTS and "zone" in m.columns:
            _add_zone_box(axes[c], m["zone"].to_numpy(),
                          m[ec].to_numpy(), log=use_log)

    cb = fig.colorbar(sc, ax=list(axes), location="right",
                      fraction=0.025, pad=0.02, aspect=20, label=label,
                      extend=extend)
    _three_ticks_log_cbar(cb, norm)

    out_png = os.path.join(FIG_DIR, f"{slug}_n{n_sta}sta.png")
    add_panel_letters(axes)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_png}")


def plot_combined(df, indices, out_png, proj, cmap_override=None):
    """Multi-index combined figure: rows = indices, columns = engines (HBV,
    LSTM). Each index ROW shares one colour scale across both engines (so the
    columns stay directly comparable), with a single colorbar to the right of
    the row. Drives both the .tex "Figure 4" trio (runoff/peakflow/Q05) and the
    thematic COMBINED_GROUPS (e.g. the low-flow / dry-spell 4x2). `out_png` is
    the fully-resolved output path (n{stations}sta already substituted); panel
    letters restart at (a) per figure.

    `cmap_override` (a colormap name): when given, every row uses this single
    colormap instead of its per-index INDICES cmap, so a thematic figure reads
    with one consistent palette top-to-bottom (each row keeps its own norm /
    colorbar — only the palette is shared, since the rows have different units
    and cannot share a scale). None keeps each row's own cmap."""
    nrow, ncol = len(indices), len(ENGINES)
    # per-column width ~= FIG_W/ncol, Robinson height ~= 0.507 * width.
    FIG_H = nrow * (FIG_W / ncol) * 0.507 + 1.0
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W, FIG_H),
                             gridspec_kw=dict(wspace=WSPACE, hspace=0.12),
                             **proj)
    axes = np.atleast_2d(axes)

    # No figure title — the caption carries this in the manuscript. The panel
    # letter comes solely from add_panel_letters() below (bold, top left); the
    # title carries only the index and the engine, so a panel is not lettered
    # twice.
    for r, (col, label, cmap, slug) in enumerate(indices):
        # optional single shared palette across all rows (COMBINED_GROUPS): use
        # it instead of this row's per-index cmap so the thematic figure reads
        # with one consistent colormap top-to-bottom (each row still keeps its
        # own norm/colorbar — only the palette is shared).
        if cmap_override is not None:
            cmap = cmap_override
        # one colour scale per index ROW, from BOTH engines, so the columns compare
        stacked = np.concatenate([
            df[f"{e}_{col}"].to_numpy(dtype=float) for e in ENGINES])
        norm = _robust_norm(stacked)
        # inset boxplot y-scale follows the map: log iff the map is log, so
        # the two views of the same index stay visually consistent.
        use_log = isinstance(norm, LogNorm)

        # Heavy-tail floor/cap clamp (Q05 floor, low-flow-magnitude floors,
        # Q5/Q95 cap); no-op for linear norms and unlisted slugs.
        norm, cmap, extend = _scale_override(norm, cmap, slug)

        sc = None
        for c, engine in enumerate(ENGINES):
            ec = f"{engine}_{col}"
            m = df.dropna(subset=[ec])
            m = m[np.isfinite(m[ec])]
            sc = _scatter(axes[r, c], m.lon, m.lat, m[ec].to_numpy(),
                          cmap, f"{label}, {engine.upper()}", norm=norm)
            if ADD_ZONE_BOXPLOTS and "zone" in m.columns:
                _add_zone_box(axes[r, c], m["zone"].to_numpy(),
                              m[ec].to_numpy(), log=use_log)

        # single colorbar for the index ROW, spanning both engine columns
        cb = fig.colorbar(sc, ax=list(axes[r, :]), location="right",
                          fraction=0.025, pad=0.02, aspect=30, label=label,
                          extend=extend)
        _three_ticks_log_cbar(cb, norm)

    os.makedirs(FIG_DIR, exist_ok=True)
    add_panel_letters(axes)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_png}")


_MATHTEXT_RE = re.compile(r"\$[^$]*\$")
_TRAILING_UNIT_RE = re.compile(r"^(.*?)\s*(\([^()]*\))$")


def _visible_len(s):
    """Rendered character width of `s`, counting a mathtext run (e.g.
    `$^{-1}$`) as the ~2 characters it actually draws rather than its 7 source
    characters. Used only to wrap panel titles, so an approximation suffices."""
    return len(_MATHTEXT_RE.sub("--", s))


def _wrap_panel_title(label, width=20):
    """Break an INDICES label into <= `width`-character lines.

    The boxplot panels are narrow (a two-box comparison needs little width), so
    a one-line title such as "Richards-Baker flashiness index (-)" would run
    into its neighbours. Words are packed greedily by `_visible_len`, and a
    trailing parenthesised unit is treated as ONE unbreakable token so a unit
    never splits across lines ("(day of water year)" stays intact, and a short
    "(-)" still rides along on the last text line).
    """
    m = _TRAILING_UNIT_RE.match(label)
    words = (m.group(1).split() + [m.group(2)]) if m else label.split()
    lines, cur = [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if cur and _visible_len(cand) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def plot_index_boxplots(df, n_sta, entries, fname_slug):
    """One figure, one panel PER index in `entries`: each panel shows an HBV box
    and an LSTM box for that index, on its OWN y-axis. Saved as
    `{fname_slug}_n{n}sta.png`. `entries` is a list of INDICES tuples (see
    BOXPLOT_GROUPS, which splits the indices into the manuscript-table set and
    the remainder).

    Per-index panels (rather than one shared axis) are used precisely because
    the indices carry different units and span different ranges — each panel
    auto-scales to its own index, so the HBV-vs-LSTM contrast is legible for
    every index, narrow-range (flashiness) and wide-range (Q05) alike. Each
    panel's y-axis is log or linear by the SAME rule the maps use
    (`_robust_norm`: log when the pooled positive values span >=
    LOG_SPAN_THRESHOLD decades), so the boxplot view and the map view of an
    index share one scale convention. Boxes = IQR, whiskers = 5-95th pct,
    fliers hidden. A log axis cannot show <=0, so on log panels the lower tail
    is CLIPPED up to BOX_FLOOR (np.maximum) rather than dropped: every value
    below the floor — including LSTM's negative low flows (interpretation §4) —
    piles at the floor, so the box keeps the full catchment count and the
    low-flow pile-up stays visible (a flat box/whisker bottom). The count
    clipped per panel/engine is logged and annotated.
    """
    entries = list(entries)
    n_idx = len(entries)
    ncol = 5
    nrow = int(np.ceil(n_idx / ncol))
    # Canvas: a two-box comparison needs little width, so each panel column is
    # ~1.8 in and the per-index titles are WRAPPED (`_wrap_panel_title`) to fit
    # it. The narrow canvas is what makes the tick / title / annotation type
    # legible once the PNG is scaled to a journal page: the point sizes below
    # are unchanged, but they now occupy ~1.4x more of the figure width than
    # they did on the former 12.5 in canvas. The per-row height leaves a band
    # above every panel for the bold panel letter, which sits ABOVE the centred
    # (up to 2-line) title instead of overlaying it.
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.0, 2.3 * nrow),
                             squeeze=False)
    axes_flat = axes.ravel()

    for k, (col, label, _cmap, slug) in enumerate(entries):
        ax = axes_flat[k]
        # log-vs-linear decision shared with the maps (one scale convention)
        stacked = np.concatenate([
            df[f"{e}_{col}"].to_numpy(dtype=float) for e in ENGINES])
        use_log = isinstance(_robust_norm(stacked), LogNorm)

        data, box_colors, ticklabels, clips = [], [], [], []
        for engine in ENGINES:
            v = df[f"{engine}_{col}"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if use_log:
                n_clip = int((v < BOX_FLOOR).sum())
                if n_clip:
                    logger.info(f"  boxplot {engine} {slug}: clipped {n_clip} "
                                f"value(s) < {BOX_FLOOR} up to {BOX_FLOOR} "
                                f"(log panel)")
                    clips.append((engine.upper(), n_clip))
                v = np.maximum(v, BOX_FLOOR)     # clip low tail up to the floor
            data.append(v)
            box_colors.append(ENGINE_BOX_COLORS.get(engine, "0.6"))
            ticklabels.append(engine.upper())

        # Two engine boxes per panel, drawn narrow and close together so the
        # pair reads as a single comparison (narrow boxes + small centre gap).
        # The box width, the centre gap and the x limits are scaled together
        # (0.20 / 0.15 / +-0.40, i.e. a box spans 25% of the axes width), which
        # halves the drawn box width relative to the earlier wide canvas and is
        # what lets the figure be 9.0 in rather than 12.5 in wide.
        n_eng = len(ENGINES)
        half_gap = 0.15                       # box-centre offset from panel centre
        positions = ([1.0 - half_gap + 2 * half_gap * j / (n_eng - 1)
                      for j in range(n_eng)] if n_eng > 1 else [1.0])
        bp = ax.boxplot(data, positions=positions, widths=0.20,
                        showfliers=False, patch_artist=True, whis=(5, 95))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set(facecolor=c, edgecolor="black", linewidth=0.6, alpha=0.85)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.2)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(linewidth=0.6, color="black")

        if use_log:
            ax.set_yscale("log")
        # Wrapped title (the panel is ~1.8 in wide) with a small pad, so the
        # bold panel letter (added later ABOVE the title) clears it instead of
        # overlaying it.
        ax.set_title(_wrap_panel_title(label), fontsize=9, pad=2)
        ax.set_xticks(positions)
        # The x-tick labels (engine names) are identical in every panel, so show
        # them only on the bottom-most plotted panel of each column and blank
        # them elsewhere to free vertical space; engine identity on the blanked
        # panels is carried by the box colour (see the figure legend). The
        # `k + ncol >= n_idx` test picks the lowest plotted panel per column, so
        # it handles the ragged last row correctly.
        if k + ncol >= n_idx:
            ax.set_xticklabels(ticklabels, fontsize=8.5)
        else:
            ax.set_xticklabels([])
        ax.set_xlim(1.0 - 0.40, 1.0 + 0.40)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", which="major", linestyle=":", linewidth=0.4,
                alpha=0.6)
        ax.set_axisbelow(True)
        if clips:
            # Flag how many (and which engine) piled at BOX_FLOOR, so the flat
            # box/whisker bottom is read as "values clipped to 0.01" (incl.
            # LSTM's negative low flows, interpretation §4), not as a real low
            # whisker. Two short lines rather than one long one: a single line
            # would run past the right edge of these narrow panels.
            ax.text(0.04, 0.97,
                    f"clipped to {BOX_FLOOR}\n"
                    + ", ".join(f"{e} {n}" for e, n in clips),
                    transform=ax.transAxes, fontsize=6.5, va="top",
                    ha="left", color="#555555", linespacing=1.1,
                    # semi-opaque backing: the note sits where the upper
                    # whisker often runs, and unbacked text over the whisker
                    # line is hard to read at print size.
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=0.8))

    # hide any unused trailing panels in the grid
    for k in range(n_idx, len(axes_flat)):
        axes_flat[k].set_visible(False)

    # hidden trailing cells of the grid (n_idx is rarely a multiple of ncol) —
    # the last one hosts the legend below.
    n_hidden = len(axes_flat) - n_idx

    # h_pad widens the inter-row gap so the panel letters (placed above each
    # title) sit clear of the panel above; rect's bottom strip holds the
    # footnote and its top strip holds the panel letters of the first row,
    # which tight_layout cannot see (they are drawn after it, in axes coords
    # above each title). That strip is a FIXED 0.30 in, converted to the
    # fraction this figure's height implies, so a 3-row and a 4-row grid
    # reserve the same absolute space rather than the same proportion.
    fig_h = fig.get_figheight()
    fig.tight_layout(h_pad=3.0, rect=[0, 0.012, 1, 1 - 0.30 / fig_h])

    # Single figure legend (engine -> box colour). With the x-tick labels shown
    # only on the bottom row, this keeps every panel's two boxes identifiable.
    # It goes INSIDE the last (empty) grid cell — bottom-right — which no panel
    # uses; drawn after tight_layout so that cell's final position is known.
    # Only if the grid happens to be exactly full does it fall back to the
    # top-right strip, where it has to share space with the panel letters.
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=ENGINE_BOX_COLORS.get(e, "0.6"),
                            edgecolor="black", alpha=0.85, label=e.upper())
                      for e in ENGINES]
    if n_hidden:
        slot = axes_flat[-1].get_position()
        fig.legend(handles=legend_handles, loc="center",
                   bbox_to_anchor=(slot.x0, slot.y0, slot.width, slot.height),
                   bbox_transform=fig.transFigure, ncol=1,
                   frameon=False, fontsize=11, handlelength=1.6)
    else:
        fig.legend(handles=legend_handles, loc="upper right",
                   bbox_to_anchor=(0.995, 0.999), ncol=len(ENGINES),
                   frameon=False, fontsize=9)
    fig.text(0.005, 0.004,
             "box = IQR · whiskers = 5–95th pct · fliers hidden · "
             "log y where span >= 1.5 decades, else linear · "
             f"on log panels values < {BOX_FLOOR} are clipped to {BOX_FLOOR}",
             fontsize=7, color="0.4")

    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = os.path.join(FIG_DIR, f"{fname_slug}_n{n_sta}sta.png")
    # Panel letters ABOVE the title so the bold "a)" never overlays it. y=1.26
    # (not 1.12) because the titles are now wrapped onto up to two lines; the
    # letters share one y, so they stay aligned across the whole grid.
    add_panel_letters(axes, y=1.26)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_png}")


def _plot_boxplot_groups(df, n_sta):
    """Draw every BOXPLOT_GROUPS figure and verify the split covers INDICES.

    The two groups must partition INDICES exactly: an index named in no group
    would silently disappear from the summary, and one named twice would be
    plotted twice. Both cases are logged as warnings rather than raising, so a
    long run still produces its figures.
    """
    by_slug = {entry[3]: entry for entry in INDICES}
    listed = [s for _, slugs in BOXPLOT_GROUPS for s in slugs]
    unknown = [s for s in listed if s not in by_slug]
    if unknown:
        raise KeyError(f"BOXPLOT_GROUPS names unknown index slug(s): {unknown}")
    missing = [s for s in by_slug if s not in listed]
    dupes = sorted({s for s in listed if listed.count(s) > 1})
    if missing:
        logger.warning(f"BOXPLOT_GROUPS omits index slug(s): {missing}")
    if dupes:
        logger.warning(f"BOXPLOT_GROUPS repeats index slug(s): {dupes}")

    for fname_slug, slugs in BOXPLOT_GROUPS:
        logger.info(f"boxplot figure {fname_slug}: {len(slugs)} indices")
        plot_index_boxplots(df, n_sta, [by_slug[s] for s in slugs], fname_slug)


def main():
    if os.path.exists(CACHE_FPATH) and not RECOMPUTE:
        logger.info(f"Loading cached baseline table {CACHE_FPATH} "
                    "(set RECOMPUTE=True to rebuild)")
        df = pd.read_parquet(CACHE_FPATH)
        need = [f"{e}_{c}" for e in ENGINES for c, *_ in INDICES]
        missing = [c for c in need if c not in df.columns]
        if missing:
            logger.info(f"  cache lacks {missing} -> recomputing")
            df = compute_cache()
    else:
        df = compute_cache()

    n_sta = int(df["station"].nunique())
    logger.info(f"Stage 2: plotting {n_sta} stations (engines={ENGINES})")
    for e in ENGINES:
        for col, label, *_ in INDICES:
            ec = f"{e}_{col}"
            s = df[ec].replace([np.inf, -np.inf], np.nan).dropna()
            logger.info(f"  {e} {label}: n={len(s)}  median={s.median():.3g}  "
                        f"p05={s.quantile(.05):.3g}  p95={s.quantile(.95):.3g}")

    # --boxplots-only: regenerate just the index_boxplots summary from the cache
    # and skip the expensive cartopy map families (combined trio, thematic
    # groups, per-index pairs).
    if BOXPLOTS_ONLY:
        logger.info("BOXPLOTS_ONLY: regenerating the index boxplot figures only "
                    "(skipping the map families)")
        _plot_boxplot_groups(df, n_sta)
        return

    # attach Köppen-Geiger major zone per station for the inset boxplots
    # (done here, not in the cache, so changing KG_ZONES needs no recompute)
    if ADD_ZONE_BOXPLOTS:
        static = pd.read_csv(os.path.join(CONFIG.gscad_path, "dataframes",
                                          "static.csv"), comment="#", index_col=0)
        if "KGZ_major" in static.columns:
            zser = static["KGZ_major"].astype(str).str.strip()
            df["zone"] = df["station"].map(zser)
            kept = df["zone"].isin(KG_ZONES).sum()
            logger.info(f"KG zones attached: {kept}/{len(df)} stations fall in "
                        f"{KG_ZONES}; per-zone counts shown under each boxplot")
        else:
            logger.warning("static.csv has no 'KGZ_major' column -> "
                           "skipping inset boxplots")

    try:
        import cartopy.crs as ccrs
        proj = dict(subplot_kw={"projection": ccrs.Robinson()})
    except Exception:
        proj = {}

    # Layout: rows = indices, columns = engines (HBV, LSTM) -> 2 panels per row
    # (fig02-style). Each index ROW shares one colour scale across both engines
    # (so the engines stay directly comparable), with a single colorbar to the
    # right of the row. All combined figures go through plot_combined().

    # (1) The .tex "Figure 4" trio (runoff, peakflow, Q05) — layout preserved.
    plot_combined(df, list(INDICES[:N_COMBINED]),
                  OUT_PNG_TMPL.format(n=n_sta), proj)

    # (2) Thematic combined figures (e.g. the low-flow / dry-spell 4x2). Rows
    # are resolved from INDICES by slug so cmap/label/scale live in one place.
    by_slug = {entry[3]: entry for entry in INDICES}
    for fname_slug, row_slugs, group_cmap in COMBINED_GROUPS:
        rows = [by_slug[s] for s in row_slugs]
        out_png = os.path.join(FIG_DIR, f"{fname_slug}_n{n_sta}sta.png")
        plot_combined(df, rows, out_png, proj, cmap_override=group_cmap)

    # Per-index companion plots (one 1x2 PNG per hydrological index, HBV vs
    # LSTM). Reuses df, the same colour-scale logic and the same FIG_DIR.
    # Dry-spell slugs are skipped — they already live in the lowflow_dryspell
    # combined figure, so a standalone copy would be redundant.
    for col, label, cmap, slug in INDICES:
        if slug in SKIP_PER_INDEX_SLUGS:
            continue
        plot_index_pair(df, col, label, cmap, slug, n_sta, proj)

    # (3) Boxplot summaries: one panel per index, each with an HBV and an LSTM
    # box on its own units-appropriate y-axis (engine comparison), split into the
    # manuscript-table set and the remainder (BOXPLOT_GROUPS).
    _plot_boxplot_groups(df, n_sta)


if __name__ == "__main__":
    main()
