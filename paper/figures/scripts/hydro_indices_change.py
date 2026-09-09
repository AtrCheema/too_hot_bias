"""
Figure — Impact of full vs reduced (too-hot-removed) ensemble on the
projected change in hydrological indices.

Reproduces the 9-panel figure of the
"Impact of full vs reduced ensemble on changes in hydrological indices in
future" subsection of too_hot_mdoels.tex (it is labelled "Figure 5" in the
current outline; the 6-panel precip/temp "Figure 2" is fig02_precip_change.py
— the user referred to this one loosely as "Fig2"):

    Row 1  mean annual runoff change (%)        (a) full  (b) reduced  (c) diff
    Row 2  mean annual maximum change (%)       (d) full  (e) reduced  (f) diff
    Row 3  dry-spell duration change (%)        (g) full  (h) reduced  (i) diff

where "diff" = full − reduced (the bias introduced by keeping the too-hot
models), and the focus scenario is SSP5-8.5 (the .tex says "Let's focus on
SSP5-8.5 in this section").

Indices and where they come from
--------------------------------
* Mean annual runoff      -> stats.parquet column ``delta_rel_qmean``
                             (qmean = mean daily flow; its relative change is
                             the relative change in mean annual runoff).
* Peak-flow magnitude     -> stats.parquet column ``delta_rel_mean_annual_max``
                             (mean of the annual maxima).
  Both ``delta_rel_*`` are the per-realization future-vs-baseline relative
  changes already computed inside apply_deltas_stn.py
  (add_change_vs_baseline), so they are exactly the "% change in the future"
  the figure needs. They are multiplied by 100 here.
* Dry-spell duration      -> NOT stored in stats.* (compute_flow_stats does
                             not produce it), so it is computed from qsim.nc:
                             a "dry day" is a day with q <= threshold, the
                             threshold is the baseline (historical) Q-quantile
                             DRYSPELL_QUANTILE (default Q10), held FIXED for
                             both baseline and future so the change reflects a
                             flow-regime shift, not a moving threshold. The
                             metric is the mean over years of the longest
                             consecutive dry run, and the % change is
                             (future − baseline) / baseline * 100.

Set INDEX_SOURCE = "qsim" to recompute runoff and peak-flow from qsim.nc as
well (same series used for the dry spell), instead of trusting stats.parquet.

The "too hot" model screening, the aggregation rule and the shared map
helpers are imported from fig02_precip_change.py because they DEFINE the
study and must be identical in both figures (`FULL_MODELS`,
`EXCLUDED_MODELS`, `REDUCED_MODELS`, `AGGREGATION`, `_scatter`,
`_diff_norm_cmap`). Presentation choices (`SCENARIOS`, `MAP_SCENARIOS`,
`ALL_PERIODS`, `PERIODS`) are LOCAL here — fig05 can show a different
scenario subset / focus / period selection than fig02 without one
breaking the other. ALL_PERIODS still comes from CONFIG.future_periods.

What one run produces (per engine in ENGINES)
---------------------------------------------
1. The 9-panel index-change MAP (rows = indices, cols = full/reduced/diff),
   one figure per scenario in MAP_SCENARIOS × per period in PERIODS.
2. A per-scenario summary BOXPLOT figure: rows = the INDICES, columns = the
   two future periods (mid- & end-century), and in each panel a full +
   reduced box pair per scenario (5 scenarios -> 10 boxes) with the global
   mean marker. This is the fig02.boxplots_by_scenario analogue and is how
   the other scenarios are shown without a wall of maps.
So with two engines you get (2 × len(MAP_SCENARIOS) × len(PERIODS)) maps +
2 boxplot figures. Failures for one combination are logged; the rest run.

Two-stage design (same as fig02), ONE cache per engine
------------------------------------------------------
Stage 1 (expensive): per engine, loop over every station and compute the
        per-realization % changes for ALL SCENARIOS x ALL_PERIODS in a
        single pass (qsim.nc + the baseline are opened once per station),
        cached to a long parquet/csv named per engine only.
Stage 2 (cheap): slice the cache (MAP_SCENARIOS for the maps; everything
        for the boxplot), split into FULL vs REDUCED, aggregate, plot.
Keep RECOMPUTE = False and rerun -> only the (seconds-long) plots are redone.

Run (python only):
    python hydro_indices_change.py                 # every section, all stations
    python hydro_indices_change.py bias_ranking    # one section only
    python hydro_indices_change.py maps spread     # several sections
    python hydro_indices_change.py --max-stations 200   # quick test: first 200
Sections: maps, scenario_boxplots, diff_maps, value_maps, engine_disagreement,
    bias_ranking, engine_boxplots, spread (or "all"). Parsed in __main__.
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
import cmcrameri.cm  # noqa: F401  registers 'cmc.*' colormaps with matplotlib
from matplotlib.colors import TwoSlopeNorm, ListedColormap, to_rgb

# add BOTH paper/figures/scripts/ (this dir, for the fig02 sibling import) and the
# repo root (three dirs up, where config.py / apply_deltas_utils.py live) to
# sys.path, so the script imports cleanly when launched as
# `python paper/figures/scripts/figXX.py` (mirrors fig02).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config import CONFIG
# same realization exclusion list fig02/build_streamflow/significance use
from realization_blocklist import apply_blocklist
# registry of EVERY change metric in stats.parquet (29 streamflow + 12 precip
# indices), shared with build_streamflow_cache.py and the significance script.
# The Stage-1 cache below carries all of them; the per-index MAP figures stay on
# the curated DIFF_INDICES subset, but the cross-index bias ranking spans the lot.
from flow_metrics import FLOW_METRICS, PRECIP_INDEX_METRICS
# Nature-style bold top-left panel letters (multi-panel figures only) and the
# project-wide percentile-index label table (see pretty_index)
from figstyle import add_panel_letters, percentile_label, plabel

# Reuse ONLY what genuinely defines the study (model screening + aggregation
# rule) and the visual helpers, so the two figures cannot disagree about what
# counts as "too hot" or look stylistically different. Scenarios and periods
# are NOT imported — they're presentation choices, defined locally below, so
# fig05 can plot a different scenario subset / focus than fig02 without
# either figure breaking the other. fig02 reads sys.argv at import time
# (MAX_STATIONS); neutralise argv during the import.
_argv_bak = sys.argv
sys.argv = sys.argv[:1]
try:
    from fig02_precip_change import (
        FULL_MODELS, EXCLUDED_MODELS, REDUCED_MODELS, AGGREGATION,
        _scatter, _diff_norm_cmap, aggregate_weighted,
    )
finally:
    sys.argv = _argv_bak

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

# SIGN CONVENTION (2 Aug 2026, matching gev_fdr_dipole.py / fig02). The
# manuscript reports every ensemble difference as reduced − full, so the
# main-text Fig. 3 bias boxplots (panel a) are drawn as
# BIAS_SIGN * (full − reduced). The flip is applied ONLY at that drawing
# site: ensemble_table's "diff" column, the regime-share CSVs
# (share = (full − reduced)/full, a positive removal share) and the
# per-index diagnostic maps keep the stored full − reduced orientation.
BIAS_SIGN = -1.0
# Name of the non-full ensemble INSIDE the sign convention. Under
# METHOD='weighting' nothing is removed, so "reduced" is the wrong word and
# contradicts main4.tex, which writes every difference as "weighted − full".
_SECOND   = "weighted" if em.is_weighting() else "reduced"
SIGNCONV  = f"full − {_SECOND}" if BIAS_SIGN > 0 else f"{_SECOND} − full"

# The renderable figure families ("sections"). Argument parsing lives in
# __main__: run with no section to draw them all (the fig05*.sh slurm jobs), or
# name one/several to restrict the run so a slurm job can regenerate a single
# family cheaply (e.g. `python hydro_indices_change.py bias_ranking`).
SECTIONS = [
    "maps",                 # 9-panel per-(scenario, period) change maps
    "scenario_boxplots",    # per-engine scenario-gradient boxplots
    "diff_maps",            # per-index 4x2 full-minus-reduced difference maps
    "value_maps",           # per-index 4x2 full & reduced value maps
    "engine_disagreement",  # (LSTM - HBV) of the full-minus-reduced diff maps
    "bias_ranking",         # cross-index too-hot bias-ranking summary
    "engine_boxplots",      # combined HBV+LSTM full-vs-reduced boxplots
    "spread",               # full-vs-reduced ensemble-spread figures
    "fig3_combined",        # main-text Fig. 3: bias ranking + peak-flow spread
    "regime_strata",        # artefact share by Koeppen zone / aridity / snow (CSV only)
]


# ===========================================================================
#  USER SETTINGS  (edit these; model screening + aggregation live in fig02)
# ===========================================================================

# Hydrological engines to plot. One figure (+ cache) is produced per engine
# per future period automatically when the script is run — no manual editing
# / re-invocation needed. stats.parquet/qsim.nc carry both engines.
ENGINES = ["hbv", "lstm"
           ]

# Scenarios this figure considers. Independent of fig02's SCENARIOS list —
# edit freely; changes do NOT affect fig02.
SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]

# Scenarios for the 9-panel MAP figure — one map per (engine, scenario,
# period). Independent of fig02. Must be a subset of SCENARIOS. Generating
# more is cheap because Stage 1 is cached; only the plotting reruns.
MAP_SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
MAP_SCENARIOS = [s for s in MAP_SCENARIOS if s in SCENARIOS] or [SCENARIOS[-1]]

# ALL_PERIODS = every future period in the pipeline (apply_deltas_stn.py
# labels them "{start_year}-{end_year}", e.g. "2041-2060", "2081-2100").
# The Stage-1 cache ALWAYS covers all of these x all SCENARIOS so the
# per-scenario boxplot has both periods.
ALL_PERIODS = [f"{p[0].split('-')[0]}-{p[1].split('-')[0]}"
               for p in CONFIG.future_periods]

# PERIODS = the subset of ALL_PERIODS for which the 9-panel MAP figure is
# drawn (one map per engine per period). Narrow this to cut map output; it
# does NOT affect the cache or the boxplot (those use ALL_PERIODS).
PERIODS = ALL_PERIODS         # both mid- and end-of-century
PERIODS = [p for p in PERIODS if p in ALL_PERIODS] or ALL_PERIODS

# Per-scenario summary boxplot (fig02.boxplots_by_scenario analogue): one
# figure per engine, rows = the INDICES, columns = the two periods, and in
# each panel a full + reduced box pair per scenario (reddish = full,
# bluish = reduced) with the global-mean marker overlaid.
FULL_BOX_C, REDUCED_BOX_C = "firebrick", "steelblue"

# Combined full-vs-reduced boxplot with BOTH engines in ONE figure (HBV +
# LSTM paired inside every panel). The scenario is fixed to the section's
# focus (BOX_ENGINES_SCENARIO; the .tex "Let's focus on SSP5-8.5") so the
# figure stays readable: rows = CORE_BOX_INDICES × cols = ALL_PERIODS, four
# boxes per panel (HBV {full, reduced}, then LSTM {full, reduced}). The
# scenario gradient itself is already shown by boxplots_by_scenario; this
# figure isolates the engine-dependence of the too-hot bias at the focus
# scenario. Needs both engine caches, so it is drawn after both are loaded.
# One combined figure per scenario listed here (each a separate PNG). Kept to
# the paper's highlighted scenarios so the set stays small; must be a subset
# of SCENARIOS.
BOX_ENGINES_SCENARIOS = ["ssp585", "ssp245"]
BOX_ENGINES_SCENARIOS = [s for s in BOX_ENGINES_SCENARIOS if s in SCENARIOS]
# Curated 6-index subset (one representative per family) for the combined
# figure; each tuple is (cache column, panel/row label). Labels are split
# across two lines ("\n") so they fit the compact row height as rotated
# y-axis labels without overlapping the neighbouring panel.
CORE_BOX_INDICES = [
    ("runoff_change_pct",        "Mean annual runoff\nchange (%)"),
    ("peakflow_change_pct",      "Mean annual maximum\nchange (%)"),
    # The two percentile rows take their wording from figstyle's project-wide
    # table rather than a hand-typed "Q95"/"Q05". A bare Q95 collides with the
    # flow-duration convention, in which Q95 is a LOW flow, while here
    # q95 = quantile(0.95), a high flow.
    ("q95_change_pct",           f"{percentile_label('q95').long}\nchange (%)"),
    ("q05_change_pct",           f"{percentile_label('q05').long}\nchange (%)"),
    ("flashiness_rb_change_pct", "Richards–Baker flashiness\nchange (%)"),
    ("dry_spell_max_change_pct", "Longest dry-spell\nchange (%)"),
]
# boxplots_by_scenario splits the 6 core indices across several PNGs, this many
# index-rows per figure (6 rows in one figure was too cramped to read). 3 -> two
# files per engine (…_set1 = first 3 indices, …_set2 = next 3).
BY_SCENARIO_ROWS_PER_FIG = 3

# Spread-of-projections figure (spread_projection). For ONE index at ONE
# (scenario, period) it shows the full vs reduced ensemble spread, full drawn
# BEHIND and reduced IN FRONT, in TWO complementary panels per engine (rows =
# HBV, LSTM):
#   (a)  per-catchment SPREAD_BAND_PCT ensemble-spread band across catchments
#        sorted by their own median — shows WHERE on the response spectrum, and
#        on WHICH edge, the too-hot models add spread;
#   (b)  across-catchment density of the per-catchment ensemble mean — shows the
#        distributional SHIFT / skew reduction (finding 9).
# (The earlier pooled-per-model density panel was dropped: its width conflated
# spatial and inter-model variance and it isolated neither the shift nor its
# location.) The projection unit is the MODEL (members averaged within a model
# first), so "reduced" is exactly "full minus the too-hot models", matching the
# maps. Reuses the per-engine cache already loaded for every other figure (no
# recompute) and the FULL_BOX_C / REDUCED_BOX_C colours.
SPREAD_SCENARIO = "ssp585"
SPREAD_PERIOD   = "2081-2100"
SPREAD_INDICES  = [
    ("peakflow_change_pct", "Mean annual maximum change (%)"),
    ("runoff_change_pct",   "Mean annual runoff change (%)"),
]
SPREAD_BAND_PCT = (0.10, 0.90)   # ensemble spread band percentiles (panel a)
SPREAD_CLIP_PCT = (1.0, 99.0)    # robust x-range for the density panel (pct)
# Legibility-only smoothing of the panel-(a) band edges & median line: centred
# rolling median over this many sorted catchments (they are already ordered, so
# this only removes per-catchment pixel striping; it does NOT move the story).
# 0 disables it. ~2 % of the catchment count is a light, honest window.
SPREAD_BAND_SMOOTH = 120

# Cross-index bias ranking summary (one figure per (scenario, period) pair
# listed here). Each figure compares the too-hot bias (full − reduced)
# magnitude AND its share of the full-ensemble signal across all
# DIFF_INDICES, with HBV and LSTM shown side-by-side. Independent of
# MAP_SCENARIOS / PERIODS so the ranking can focus on whichever scenarios
# the paper highlights.
BIAS_RANKING_TARGETS = [("ssp245", "2081-2100"),
                        ("ssp585", "2081-2100")]
ENGINE_HBV_C, ENGINE_LSTM_C = "#d95f02", "#1f78b4"   # HBV orange, LSTM blue

# ---------------------------------------------------------------------------
# Regime-stratified artefact share (the `regime_strata` section)
# ---------------------------------------------------------------------------
# The manuscript quotes ONE global artefact share per flood index (report.tex
# P15: screening removes 29 % (HBV) / 23 % (LSTM) of the projected peak-flow
# intensification at SSP5-8.5 / 2081-2100). A global mean is only meaningful if
# the underlying share is reasonably uniform, and three independent lines say it
# may not be: Rahimpour Asenjan et al. 2023 (HESS) found the hot-vs-normal
# streamflow response flips sign between North American regions;
# Castaneda-Gonzalez et al. 2023 (Clim. Change) found weighting moves peak flow
# 3-5 % in snow-dominated vs 16-37 % in rain-dominated basins; and our own R4
# result has flood transmission running 0.50-0.54 in the most humid aridity
# decile against 0.24-0.26 near PET/P ~ 1. This section recomputes the SAME
# statistic within climate-regime strata so that stability can be reported
# rather than assumed. It writes a CSV to CACHE_DIR (fig_data) and draws no
# figure; the numbers feed a supplementary table.
REGIME_STRATA_TARGETS = [("ssp585", "2081-2100")]

# Slugs from the flow_metrics registry. The two flood indices carry the
# headline; q99 and mean runoff are the contrast pair (the manuscript claims the
# screening reaches the flood tail but not the water balance).
REGIME_STRATA_SLUGS = ["peakflow", "flood_q20", "q99", "runoff"]

# Static catchment attributes defining the strata. Deliberately the SAME
# static.csv columns the transmission analysis uses (TRANSMISSION_ATTRS in
# analyze_full_vs_reduced_significance.py), so the regime axes of the two
# results are directly comparable.
KGZ_LABELS = {"A": "A tropical", "B": "B arid", "C": "C temperate",
              "D": "D continental", "E": "E polar"}

# Snow fraction of precipitation (0-1). The 0.5 cut isolates the snow-dominated
# basins in which the snow-to-rain shift can invert the sign of the projected
# peak-flow change.
SNOW_BINS = [-0.001, 0.05, 0.20, 0.50, 1.001]
SNOW_LABELS = ["snow <5%", "snow 5-20%", "snow 20-50%", "snow >50%"]

# A stratum needs at least this many catchments before its mean is reported as a
# share, and the full ensemble must project an intensification of at least this
# many percentage points before a share is defined at all (see
# _regime_share_row for why the guard is one-sided).
REGIME_MIN_N = 100
REGIME_MIN_DENOM_PCT = 2.0

ARIDITY_N_DECILES = 10

# ---------------------------------------------------------------------------
# Main-text Fig. 3 — ONE canvas carrying both halves of the flow-bias result
# ---------------------------------------------------------------------------
# main4.tex previously stacked two separately rendered PNGs (the bias ranking
# and the peak-flow spread) inside one \begin{figure}. Each PNG carries its own
# panel letters, so the compiled figure showed a),b) followed by a),b),c),d),
# and the two PNGs were typeset at different scales, so their fonts did not
# match. This section draws the panels on a single canvas instead. Current
# layout (FIG3_SPREAD_BANDS = False):
#     a) cross-index too-hot bias boxplots       (bias-ranking top panel)
#     b) HBV  per-catchment ensemble-mean distribution
#     c) LSTM per-catchment ensemble-mean distribution
# The bias-share bar panel and the two ensemble-spread band panels are not
# drawn here; see fig3_combined() for why.
# The two standalone figures (bias_ranking / spread sections) are unchanged and
# still produced; this is an additional output.
FIG3_SCENARIO = "ssp585"
FIG3_PERIOD   = "2081-2100"
FIG3_SPREAD_INDEX = ("peakflow_change_pct", "Mean annual maximum change (%)")
# Canvas: 12.4 in wide keeps the 18 rotated index labels of panel (b) at the
# same density as the standalone figure (9.9 in for the same 18 indices, plus
# the wider spread panels below); 14.6 in tall reproduces to ~180 x 212 mm at
# the Nature Water double-column width, inside the 183 x 247 mm page limit.
FIG3_FIGSIZE = (12.4, 15.2)
# Vertical split between the ranking block (a, b) and the spread block (c-f).
# The gap between the blocks carries panel (b)'s rotated index labels, which
# is why the ranking block's gross share exceeds what its two panels occupy.
FIG3_HEIGHT_RATIOS = (4.20, 5.30)
# Co-author review, 9 Sep 2026: the two ensemble-spread BAND panels (the old
# (b) and (d), HBV and LSTM) are dropped from main-text Fig. 3. The weighted
# 10-90% band sits inside the full band by less than 10% of the band width, so
# a reader cannot resolve the blue band inside the red one at print size. The
# information the panels carried is already in the per-catchment
# ensemble-mean distributions that stay. Set True to restore the five-panel
# layout; the STANDALONE spread figures (`spread` section) always keep both
# columns and are unaffected by this switch.
FIG3_SPREAD_BANDS = False
# Canvas for the three-panel layout (FIG3_SPREAD_BANDS = False): the spread
# block holds ONE row (HBV and LSTM density side by side) instead of two.
# The three numbers below are solved, not guessed, so that every kept panel
# prints at exactly the size it had in the five-panel figure:
#     ranking block   4.20 units -> 5.60 in, as before
#     density panel   2.266 units -> 3.02 in, as each density panel was before
#     block gap       hspace 0.587 -> 2.53 in, as before
# The gap matters: it carries panel (a)'s rotated index labels. Scaling it
# down with the shorter canvas ran those labels into the titles of (b) and (c).
FIG3_FIGSIZE_NOBANDS = (12.4, 11.15)
FIG3_HEIGHT_RATIOS_NOBANDS = (4.20, 2.266)
FIG3_HSPACE_NOBANDS = 0.587

# Where the runoff & peak-flow % changes come from:
#   "stats" : stats.parquet delta_rel_qmean / delta_rel_mean_annual_max
#             (the metric apply_deltas_stn.py officially recorded).
#   "qsim"  : recompute them from qsim.nc (same series as the dry spell).
# Dry-spell duration is ALWAYS computed from qsim.nc (not in stats).
INDEX_SOURCE = "stats"

# Dry-spell threshold = this quantile of the BASELINE (historical) flow,
# held fixed for baseline and future. 0.10 -> Q10 (the .tex example);
# 0.25 -> Q25; 0.0 -> zero-flow days (q <= min, i.e. effectively q == 0).
DRYSPELL_QUANTILE = 0.10

# First-N stations only for a quick test; None = all. __main__ overrides this
# from --max-stations; the module default keeps it defined for importers, which
# call the figure functions directly without going through the CLI.
MAX_STATIONS = None

# Recompute the per-realization cache even if it already exists.
RECOMPUTE = False

# Stage-1 parallelism: number of worker processes (fork Pool over stations).
# Derived from the SLURM allocation (SLURM_CPUS_PER_TASK, set by
# --cpus-per-task) so it always matches the cores granted and never drifts from
# the job script; off SLURM (local) falls back to os.cpu_count()-1.
NUM_WORKERS = (int(os.environ["SLURM_CPUS_PER_TASK"])
               if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit()
               else max(1, (os.cpu_count() or 1) - 1))

# horizontal / vertical gap between map panels (smaller = tighter)
WSPACE, HSPACE = 0.0, 0.18      # columns abut (no inter-column gap)

# Marker styling. Default colormaps put value~0 at WHITE, invisible on the
# light land background. Primary fix here: recolour the colormap centre
# (value~0) to GREY (see ZERO_GREY below) so near-zero catchments stay
# visible WITHOUT a per-marker edge (edges add clutter at high point
# density). The edge is left as an optional fallback (MARKER_EDGE=None=off).
MARKER_SIZE    = 4        # point area (fig02's _scatter default is 4)
MARKER_EDGE    = None     # edge colour, e.g. "0.35"; None = no edge
MARKER_EDGE_LW = 0.20     # edge line width (only if MARKER_EDGE set)

# Grey-centre recolouring of the (diverging) colormaps. Only the band of
# normalized-colormap positions within +/- GREY_HALFWIDTH of the centre is
# blended toward ZERO_GREY; the saturated extremes keep the original hues,
# so the colormap identity (BrBG / PuOr / RdBu ...) is preserved.
ZERO_GREY      = "0.55"   # grey the centre maps to ("0"=black .. "1"=white)
GREY_HALFWIDTH = 0.08     # half-width of the grey band (colormap units, 0..0.5)

# fig05 PNGs live under paper/figures/hydro_indices/ (every figure here is
# part of the "hydrological-indices" set: 9-panel maps, per-scenario
# boxplots, 4×2 diff figures). CACHE_DIR is intentionally kept at
# paper/figures/fig_data/ so cache files stay co-located with all other
# figure-data parquets (fig02, fig04, …) and so renaming the figure subdir
# never invalidates a cache.
# this script lives in paper/figures/scripts/, so the figures root is its parent's parent
_PAPER_FIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR   = os.path.join(_PAPER_FIG_ROOT, "hydro_indices")
CACHE_DIR = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")  # per-method cache on /ibex (avoids cross-method contamination)

# The figure set is filed BY ROLE into subfolders of FIG_DIR (the flat
# directory held ~120 PNGs, which was unwieldy). Each writer os.makedirs its
# own target, so the tree is created on demand:
#   summary/                 cross-index distribution & ranking panels
#                            (boxplots_by_scenario, full-vs-reduced engine
#                            boxplots, bias-ranking) — the paper candidates.
#   maps/change_9panel/      the 9-panel full|reduced|diff change maps.
#   maps/per_index_full/     per-index 4×2 full-ensemble % change maps.
#   maps/per_index_reduced/  per-index 4×2 reduced-ensemble % change maps.
#   maps/per_index_diff/     per-index 4×2 too-hot bias (full−reduced) maps.
#   maps/per_index_enginediff/ per-index 4×2 (LSTM−HBV of the bias) maps.
SUMMARY_DIR        = os.path.join(FIG_DIR, "summary")
_MAPS_DIR          = os.path.join(FIG_DIR, "maps")
MAP_9PANEL_DIR     = os.path.join(_MAPS_DIR, "change_9panel")
MAP_FULL_DIR       = os.path.join(_MAPS_DIR, "per_index_full")
MAP_REDUCED_DIR    = os.path.join(_MAPS_DIR, "per_index_reduced")
MAP_DIFF_DIR       = os.path.join(_MAPS_DIR, "per_index_diff")
MAP_ENGINEDIFF_DIR = os.path.join(_MAPS_DIR, "per_index_enginediff")


def _cache_fpath(engine: str) -> str:
    """Per-engine Stage-1 cache, holding ALL SCENARIOS x ALL_PERIODS (so the
    maps and the per-scenario boxplot share one cache, like fig02). Name is
    scenario/period/count-independent so the skip-recompute path resolves it
    before those are known. Lives under FIG_DIR/fig_data/ alongside the
    figure PNGs."""
    return os.path.join(CACHE_DIR, f"fig05_hydro_change_{engine}.parquet")


def _out_png(engine: str, scenario: str, period: str, n_sta: int) -> str:
    """9-panel MAP figure path for one (engine, scenario, period); the
    station count is appended once it is known (after Stage 1)."""
    return os.path.join(
        MAP_9PANEL_DIR,
        f"hydro_indices_change_{engine}_{scenario}_{period}_n{n_sta}sta.png")


def _box_png(engine: str, n_sta: int, part: int = None) -> str:
    """Per-engine scenario-gradient boxplot figure path. `part` (1-based) tags
    the index-chunk when the core indices are split across several figures
    (BY_SCENARIO_ROWS_PER_FIG rows each) so each file stays readable."""
    suffix = f"_set{part}" if part is not None else ""
    return os.path.join(
        SUMMARY_DIR,
        f"hydro_boxplots_by_scenario_{engine}{suffix}_n{n_sta}sta.png")

# Indices: (cache column, panel label, colormap). The cmap is diverging
# centred on 0 (used for the full/reduced columns under a symmetric
# TwoSlopeNorm). `RdYlBu_r` is the matplotlib yellow-centre diverging map:
# saturated bright red (positive) ↔ light yellow (≈0) ↔ saturated bright
# blue (negative); no pure-white centre and — unlike Crameri's perceptually
# uniform berlin/vik/lisbon family — the extremes are bright rather than
# near-black, so dense maps don't read as "dominated by black".
INDICES = [
    ("runoff_change_pct",   "Mean annual runoff change (%)",  "RdYlBu_r"),
    ("peakflow_change_pct", "Mean annual maximum change (%)", "RdYlBu_r"),
    #("dryspell_change_pct", "Dry-spell duration change (%)", "RdYlBu_r"),
    ("q05_change_pct",      "Low-flow (Q05) change (%)",      "RdYlBu_r"),
]

# Per-index 4 (scenarios) × 2 (periods) DIFF-ONLY map figures (one PNG per
# (index, engine)). Each row corresponds to one column tuple here; `slug`
# goes into the filename. Extends INDICES with high-flow (Q95), median flow
# (Q50), annual-minimum, flow variability (σ), Richards-Baker flashiness,
# and the proper longest-dry-spell metric. Source = the delta_rel_* columns
# in stats.csv (delta_rel_flashiness_rb, delta_rel_dry_spell_max, etc.).
DIFF_INDICES = [
    # Magnitude / quantile indices (qmean .. mean_annual_min)
    ("runoff_change_pct",              "Mean annual runoff change (%)",          "runoff"),
    ("peakflow_change_pct",            "Mean annual maximum change (%)",         "peakflow"),
    ("q99_change_pct",                 "Very-high-flow (Q99) change (%)",        "q99"),
    ("q95_change_pct",                 "High-flow (Q95) change (%)",             "q95"),
    ("q50_change_pct",                 "Median flow (Q50) change (%)",           "q50"),
    ("q05_change_pct",                 "Low-flow (Q05) change (%)",              "q05"),
    ("annual_min_change_pct",          "Mean annual minimum change (%)",         "annual_min"),
    # Variability / flashiness
    ("qstd_change_pct",                "Flow variability (σ) change (%)",        "qstd"),
    ("flashiness_rb_change_pct",       "Richards-Baker flashiness change (%)",   "flashiness_rb"),
    ("flashiness_event_change_pct",    "Event flashiness change (%)",            "flashiness_event"),
    ("flashiness_event_frt_change_pct","Flood rising time (FRT) change (%)",     "flashiness_event_frt"),
    ("flashiness_event_sfd_change_pct","Storm flow depth (SFD) change (%)",      "flashiness_event_sfd"),
    # Drought persistence (dry-spell length, defined per apply_deltas_utils.dry_spell_stats)
    ("dry_spell_max_change_pct",       "Longest dry-spell change (%)",           "dry_spell_max"),
    ("dry_spell_mean_change_pct",      "Mean dry-spell length change (%)",       "dry_spell_mean"),
]

# Concise, reader-facing display names for each index/metric slug. The raw
# slugs (e.g. "qstd", "peakflow") are internal keys, not suitable for figure
# tick labels; this dictionary maps every DIFF_INDICES slug to its proper name.
# Names verified against the metric definitions in apply_deltas_utils.py:
#   runoff=delta_rel_qmean, peakflow=delta_rel_mean_annual_max,
#   annual_min=delta_rel_mean_annual_min, qstd=std(Q),
#   flashiness_rb=Richards-Baker index, flashiness_event=mean SFD/FRT rising
#   slope, FRT=flood rising time (T_peak-T_rp), SFD=streamflow difference
#   (S_peak-S_rp), dry_spell_*=run length of days below the fixed low-flow
#   threshold (Zhu et al. 2026 for the event-flashiness family).
# The four PERCENTILE indices (q99, q95, q50, q05) are deliberately ABSENT
# here: figstyle.PERCENTILE_LABELS owns them project-wide, and pretty_index()
# reads that table first. They used to sit here as "Very high flow (Q99)" etc.,
# which is the ambiguous form -- "Q95" reads either as the 95th percentile of
# daily flow (a HIGH flow, our convention) or as the flow exceeded on 95 % of
# days (a LOW flow, the flow-duration convention). Listing them in both places
# would be two sources of truth for one label, which is how the shorthand
# drifted in the first place.
INDEX_DISPLAY_NAMES = {
    "runoff":               "Mean annual runoff",
    "peakflow":             "Mean annual maximum",
    "annual_min":           "Mean annual minimum",
    "qstd":                 "Flow std. dev. (σ)",
    # En dash, not a hyphen: Richards and Baker are two people, and the joint
    # eponym is set the way Köppen–Geiger already is elsewhere in these figures
    # and the way Supplementary Table S6 already writes it ("Richards--Baker").
    "flashiness_rb":        "Richards–Baker flashiness",
    "flashiness_event":     "Event flashiness",
    "flashiness_event_frt": "Flood rising time (FRT)",
    "flashiness_event_sfd": "Streamflow rise (SFD)",
    "dry_spell_max":        "Longest dry spell",
    "dry_spell_mean":       "Mean dry-spell length",
}


# ---------------------------------------------------------------------------
#  The FULL metric set (flow_metrics.FLOW_METRICS -> cache columns)
# ---------------------------------------------------------------------------
# The Stage-1 cache carries EVERY streamflow change metric stats.parquet holds
# (29), not the 14 that the earlier hand-typed extraction picked up. Three of
# them were cached under a legacy name before this registry existed, and the
# figure lists above still use those names, so keep them:
#     delta_rel_qmean            -> runoff_change_pct
#     delta_rel_mean_annual_max  -> peakflow_change_pct
#     delta_rel_mean_annual_min  -> annual_min_change_pct
# Everything else is <base>_change_pct, matching what DIFF_INDICES already used.
# The half-flow date is the one absolute metric (a date has no meaningful
# relative change), so it is cached in DAYS, not per cent.
#
# The 12 PRECIPITATION indices in stats.parquet (PRCPTOT, RX1day, R95pTOT, CDD,
# ...) are deliberately NOT cached here: they describe the bias-corrected
# forcing, not the flow, and mixing them into a hydrological-index figure would
# be a category error. They are carried by stats_change.parquet and analysed by
# `analyze_full_vs_reduced_significance.py pindex`.
_CACHE_NAME_OVERRIDES = {"qmean":           "runoff_change_pct",
                         "mean_annual_max": "peakflow_change_pct",
                         "mean_annual_min": "annual_min_change_pct",
                         "hfd_mean":        "hfd_mean_change_days"}
_SLUG_OVERRIDES       = {"qmean":           "runoff",
                         "mean_annual_max": "peakflow",
                         "mean_annual_min": "annual_min"}


def _cache_col(m):
    return _CACHE_NAME_OVERRIDES.get(m.base, f"{m.base}_change_pct")


def _slug_of(m):
    return _SLUG_OVERRIDES.get(m.base, m.base)


def _index_label(m):
    unit = "%" if m.units == "%" else m.units
    verb = "change" if m.units == "%" else "shift"
    return f"{m.name} {verb} ({unit})"


# (stats.parquet delta column, cache column, x100 scale?, Metric) for the cache
# builder. delta_rel_* are FRACTIONS in stats.parquet; the cache stores PER CENT.
CACHE_METRICS = [(m.column, _cache_col(m), m.units == "%", m) for m in FLOW_METRICS]
CACHE_COLUMNS = [c for _, c, _, _ in CACHE_METRICS]

# Every metric, for the cross-index bias-ranking figure and the cache schema
# check. Same tuple shape as DIFF_INDICES: (cache column, axis label, slug).
ALL_INDICES = [(_cache_col(m), _index_label(m), _slug_of(m)) for m in FLOW_METRICS]

# Metrics held out of the cross-index bias-ranking figure only (they stay in the
# cache, the significance spectrum and every other consumer of the registry).
# Keyed on registry base names and resolved to cache columns below, so a name
# that no longer exists in FLOW_METRICS raises instead of silently not excluding.
_RANK_EXCLUDE_BASES = {
    # Persistence and event-TIMING metrics. Quantile mapping transfers the delta
    # of a magnitude distribution, not day-to-day sequencing, so a too-hot bias
    # in a flashiness, event-count or run-length statistic is not interpretable
    # on this axis. Both dry-spell run lengths are excluded for the same reason:
    # the ranking panel plots a PROJECTED-CHANGE bias, which is exactly the
    # temporal-transfer claim the delta method cannot support. DrySpellMax is
    # kept in the cache and in the baseline figures, where the contrast carries
    # the observed sequencing rather than a projected change.
    #
    # Richards-Baker flashiness is the ONE persistence metric the ranking panel
    # keeps: main4.tex quotes its S/N and Supplementary Table S1 carries a
    # Flashiness (RB) row, so it has to be visible and defined. It is labelled
    # supporting evidence only in Table S6 and in the main text.
    "dry_spell_max",            # Longest dry spell
    "flashiness_event",         # Event flashiness
    "flashiness_event_frt",     # Flood rising time (FRT)
    "flashiness_event_sfd",     # Streamflow rise / storm-flow depth (SFD)
    "high_q_freq",              # High-flow event frequency
    "high_q_dur",               # High-flow event duration
    "dry_spell_mean",           # Mean dry-spell length
    # FDC ratio metrics.
    "q5_q95_ratio", "q10_q90_ratio",
    # Threshold-counting low-flow metrics.
    "low_q_freq", "low_q_deficit",
    # Low-flow extremes already represented by the annual minimum flow.
    "q7min_min", "mam7", "q7_10",
    #"flood_q10",
}
_unknown_rank_excl = _RANK_EXCLUDE_BASES - {m.base for m in FLOW_METRICS}
if _unknown_rank_excl:
    raise ValueError("_RANK_EXCLUDE_BASES names metrics absent from "
                     "flow_metrics.FLOW_METRICS: "
                     + ", ".join(sorted(_unknown_rank_excl)))

# The half-flow date joins them: the bottom panel is a bias SHARE of the
# full-ensemble signal, which needs both quantities in the same units, and HFD
# is a shift in days while every other metric is a per-cent change.
BIAS_RANKING_EXCLUDE = {_cache_col(m) for m in FLOW_METRICS
                        if m.base in _RANK_EXCLUDE_BASES} | {"hfd_mean_change_days"}

# every registry metric gets a reader-facing name (the 14 curated ones above
# keep the names already used in the paper's figures)
for _m in FLOW_METRICS:
    INDEX_DISPLAY_NAMES.setdefault(_slug_of(_m), _m.name)


def pretty_index(slug: str) -> str:
    """Display name for an index/metric slug.

    TWO sources, in this order, so each index has exactly one owner and the
    two cannot drift apart:

      1. figstyle.percentile_label(slug) -- the project-wide table for the
         PERCENTILE indices ($Q_{p99}$, $Q_{p95}$, $Q_{p50}$, $Q_{p05}$).
         Those carried a real ambiguity that the p subscript removes; see the
         note above INDEX_DISPLAY_NAMES. percentile_label returns None for
         everything it does not cover, which is why this falls through rather
         than raising.
      2. INDEX_DISPLAY_NAMES, for the indices that never carried that
         ambiguity, then a title-cased slug so an unmapped index is still
         readable rather than raising a KeyError.

    Any caption that names a percentile index should carry
    figstyle.PERCENTILE_CONVENTION_NOTE: the convention is stated, not
    assumed."""
    lab = percentile_label(slug)
    if lab is not None:
        return lab.long
    return INDEX_DISPLAY_NAMES.get(slug, slug.replace("_", " ").title())


# ===========================================================================
#  Index helpers
# ===========================================================================


def _max_consecutive_true(mask: np.ndarray) -> int:
    """Longest run of True in a 1-D boolean array."""
    if mask.size == 0 or not mask.any():
        return 0
    # reset the running count to 0 at every False, keep the running max
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best


def _runoff(q: pd.Series) -> float:
    """Mean daily flow (proportional to mean annual runoff)."""
    return float(q.mean())


def _peakflow(q: pd.Series) -> float:
    """Mean of the annual maxima (peak-flow magnitude)."""
    am = q.groupby(q.index.year).max()
    return float(am.mean()) if len(am) else np.nan


def _q05(q: pd.Series) -> float:
    """5th-percentile daily flow (low-flow / baseflow magnitude)."""
    return float(q.quantile(0.05))


def _dryspell(q: pd.Series, thr: float) -> float:
    """Mean over years of the longest consecutive run with q <= thr (days)."""
    durations = [
        _max_consecutive_true(g.to_numpy() <= thr)
        for _, g in q.groupby(q.index.year)
    ]
    return float(np.mean(durations)) if durations else np.nan


def _pct(fut: float, base: float) -> float:
    """Relative change in %, guarded against a zero/non-finite baseline."""
    if base is None or not np.isfinite(base) or np.isclose(base, 0.0):
        return np.nan
    if fut is None or not np.isfinite(fut):
        return np.nan
    return (fut - base) / base * 100.0


# ===========================================================================
#  Stage 1 — per-realization % changes -> long cache
# ===========================================================================


def _qsim_series(qvar, **sel) -> pd.Series:
    """Lazily select one realization from qsim.nc and return a clean,
    datetime-indexed Series (NaN fill from combine_by_coords dropped)."""
    s = qvar.sel(**sel).to_series().dropna()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


_F05_REF = {}


def _f05_worker_init(gscad_path):
    """Per-worker init (fork Pool): load the static table once so every station
    this worker handles can read lon/lat without re-opening it."""
    _F05_REF["static"] = pd.read_csv(
        os.path.join(gscad_path, "dataframes", "static.csv"),
        comment="#", index_col=0)


def _f05_process_station(args):
    """Worker entry point — process ONE (station, engine). Returns (frame, acc):
    `frame` is this station's per-realization DataFrame (float32 metrics), or
    None when the station yields nothing; `acc` holds the skip-reason counts to
    fold into the global tally. Reference static table comes from _F05_REF."""
    stn, engine = args
    acc = {"not_in_static": 0, "no_stats": 0, "no_qsim": 0,
           "no_baseline": 0, "no_matching_realizations": 0}
    records = []
    static = _F05_REF["static"]
    qvar_name = f"{engine}_qsim"

    sdir = os.path.join(CONFIG.out_path, CONFIG.delta_method, stn)
    if stn not in static.index:
        acc["not_in_static"] += 1
        return None, acc
    lon = float(static.loc[stn, "long"])
    lat = float(static.loc[stn, "lat"])

    # ---- stats.parquet: runoff & peak-flow relative changes ----------
    stats = None
    spq = os.path.join(sdir, "stats.parquet")
    try:
        if os.path.isfile(spq):
            stats = pd.read_parquet(spq)
    except Exception as e:
        logger.warning(f"  {stn}: stats read failed ({e})")
    if stats is not None:
        stats = stats[(stats["engine"] == engine)
                      & (stats["scenario"].isin(SCENARIOS))
                      & (stats["period"].astype(str).isin(ALL_PERIODS))].copy()
        if not stats.empty:
            stats["period"] = stats["period"].astype(str)
    if (stats is None or stats.empty) and INDEX_SOURCE == "stats":
        # fall back to qsim for runoff/peak too
        pass
    if stats is None or stats.empty:
        acc["no_stats"] += 1
        stats = pd.DataFrame()

    # ---- qsim.nc is opened ONLY for the fallback path (stats missing or
    #      INDEX_SOURCE='qsim'). With the default INDEX_SOURCE='stats' and
    #      stats present, every cached metric comes from delta_rel_* columns
    #      in stats and qsim.nc is never touched — that's the big Stage-1
    #      speed-up versus earlier versions.
    use_qsim = (INDEX_SOURCE == "qsim") or stats.empty
    ds = None
    qvar = None
    base_runoff = base_peak = base_q05 = np.nan

    if use_qsim:
        qf = os.path.join(sdir, "qsim.nc")
        if not os.path.isfile(qf):
            acc["no_qsim"] += 1
            return None, acc
        try:
            ds = xr.open_dataset(qf)
            qvar = ds[qvar_name]
        except Exception as e:
            logger.warning(f"  {stn}: qsim open failed ({e})")
            acc["no_qsim"] += 1
            return None, acc
        try:
            qb = _qsim_series(qvar, scenario="baseline", model="baseline",
                              member="baseline", period="historical")
        except Exception:
            qb = pd.Series(dtype="float32")
        if len(qb) == 0:
            acc["no_baseline"] += 1
            ds.close()
            return None, acc
        base_runoff = _runoff(qb)
        base_peak   = _peakflow(qb)
        base_q05    = _q05(qb)

    # realization list as (scenario, period, model, member): prefer what
    # the stats table actually lists (exactly the runs performed); else
    # scan qsim coords over SCENARIOS x ALL_PERIODS.
    if not stats.empty:
        reals = list(stats[["scenario", "period", "model", "member"]]
                     .drop_duplicates()
                     .itertuples(index=False, name=None))
        stats_idx = stats.set_index(
            ["scenario", "period", "model", "member"])
    else:
        ds_scn = set(np.atleast_1d(ds["scenario"].values).astype(str))
        ds_per = set(np.atleast_1d(ds["period"].values).astype(str))
        ds_mdl = set(np.atleast_1d(ds["model"].values).astype(str))
        members = np.atleast_1d(ds["member"].values).astype(str)
        reals = [(s, p, m, e)
                 for s in SCENARIOS if s in ds_scn
                 for p in ALL_PERIODS if p in ds_per
                 for m in ds_mdl if m in FULL_MODELS
                 for e in members]
        stats_idx = None

    n_ok = 0
    for scn, per, model, member in reals:
        if model not in FULL_MODELS:
            continue

        # every registry metric, defaulting to NaN. The fast path copies each
        # one straight out of the stats row (delta_rel_* x100 -> per cent; the
        # half-flow date stays in days); the qsim fallback can only rebuild the
        # three indices its helpers know, so the rest stay NaN there.
        vals = {cache_col: np.nan for _, cache_col, _, _ in CACHE_METRICS}

        key = (scn, per, model, member)
        if stats_idx is not None and key in stats_idx.index:
            row = stats_idx.loc[key]
            if isinstance(row, pd.DataFrame):   # >1 row -> take first
                row = row.iloc[0]
            for stats_col, cache_col, to_pct, _m in CACHE_METRICS:
                v = float(row.get(stats_col, np.nan))
                vals[cache_col] = v * 100.0 if to_pct else v
        elif use_qsim:
            # rare fallback (no stats row): compute only what _runoff /
            # _peakflow / _q05 can give us from the future qsim series;
            # the remaining metrics stay NaN.
            try:
                qfut = _qsim_series(qvar, scenario=scn, model=model,
                                    member=member, period=per)
            except Exception:
                qfut = pd.Series(dtype="float32")
            if len(qfut):
                vals["runoff_change_pct"]   = _pct(_runoff(qfut),   base_runoff)
                vals["peakflow_change_pct"] = _pct(_peakflow(qfut), base_peak)
                vals["q05_change_pct"]      = _pct(_q05(qfut),      base_q05)

        # keep the realization if ANY of the three reference indices is finite —
        # unchanged from before, so the cached station/realization set does not
        # move just because more metrics are now carried alongside them.
        if not np.any(np.isfinite([vals["runoff_change_pct"],
                                   vals["peakflow_change_pct"],
                                   vals["q05_change_pct"]])):
            continue

        rec = {"station": stn, "lon": lon, "lat": lat,
               "scenario": scn, "model": model, "member": member,
               "period": per}
        rec.update(vals)
        records.append(rec)
        n_ok += 1

    if ds is not None:
        ds.close()
    if n_ok == 0:
        acc["no_matching_realizations"] += 1
    # Return a per-station FRAME, not a list of dicts. The parent used to hold
    # every realization of every station as a Python dict before the single
    # from_records() call; at 29 metrics x ~20.6k stations that dict overhead
    # alone is tens of GB. A float32 frame per station keeps the parent's peak
    # to roughly the size of the final cache.
    if not records:
        return None, acc
    sdf = pd.DataFrame.from_records(records)
    metric_cols = [c for c in CACHE_COLUMNS if c in sdf.columns]
    sdf[metric_cols] = sdf[metric_cols].astype("float32")
    return sdf, acc


def compute_cache(engine: str) -> pd.DataFrame:
    """Stage 1 for one engine, covering ALL SCENARIOS x ALL_PERIODS. Stations
    are processed in parallel over a fork Pool of NUM_WORKERS workers (each
    opens its own qsim.nc handles); per-station results are folded back here."""
    cache_fpath = _cache_fpath(engine)

    stat_files = sorted(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "stats.parquet")))
    qsim_only = sorted(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "qsim.nc")))
    # union of stations that have at least one of the two product files
    stations = sorted({os.path.basename(os.path.dirname(p))
                       for p in (stat_files + qsim_only)})
    if MAX_STATIONS is not None:
        stations = stations[:MAX_STATIONS]
    n_workers = (NUM_WORKERS if NUM_WORKERS is not None
                 else max(1, (os.cpu_count() or 1) - 1))
    logger.info(f"{len(stations)} station dirs to scan "
                f"(engine={engine}, source={INDEX_SOURCE}, "
                f"scenarios={SCENARIOS}, periods={ALL_PERIODS}"
                f"{'' if MAX_STATIONS is None else f', capped at {MAX_STATIONS}'}"
                f"; {n_workers} worker process{'es' if n_workers > 1 else ''})")

    skip = {"not_in_static": 0, "no_stats": 0, "no_qsim": 0,
            "no_baseline": 0, "no_matching_realizations": 0}
    frames = []
    tasks = [(stn, engine) for stn in stations]

    if n_workers <= 1:
        _f05_worker_init(CONFIG.gscad_path)
        results_iter = (_f05_process_station(t) for t in tasks)
        pool = None
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=n_workers,
                        initializer=_f05_worker_init,
                        initargs=(CONFIG.gscad_path,))
        results_iter = pool.imap_unordered(_f05_process_station, tasks,
                                           chunksize=4)

    try:
        n_rows = 0
        for j, (sdf, acc) in enumerate(results_iter):
            if sdf is not None:
                frames.append(sdf)
                n_rows += len(sdf)
            for k, v in acc.items():
                skip[k] += v
            if (j + 1) % 500 == 0:
                logger.info(f"  [{j+1}/{len(stations)}] stations processed; "
                            f"{n_rows} rows so far")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if not frames:
        raise RuntimeError(
            f"No records produced for engine={engine} "
            f"(scenarios={SCENARIOS}, periods={ALL_PERIODS}) — check "
            "ENGINES/SCENARIOS/FULL_MODELS and CONFIG.out_path.")
    df = pd.concat(frames, ignore_index=True)
    del frames
    logger.info("Stage 1 station accounting:")
    logger.info(f"  scanned : {len(stations)}")
    for reason, n in skip.items():
        logger.info(f"  skipped [{reason}] : {n}")
    logger.info(f"  USED (>=1 valid realization) : {df['station'].nunique()}")
    logger.info(f"  -> {len(df)} (station, realization) rows cached")

    # metric audit: every registry metric must have made it into the cache with
    # finite values. An all-NaN column here means the metric never reached
    # stats.parquet, not that the figure code lost it.
    logger.info(f"  metric audit ({len(CACHE_METRICS)} streamflow metrics):")
    for _stats_col, cache_col, _to_pct, m in CACHE_METRICS:
        if cache_col not in df.columns:
            logger.warning(f"    MISSING  {cache_col:32s} ({m.name})")
            continue
        frac = float(df[cache_col].notna().sum()) / len(df)
        logger.info(f"    {cache_col:32s} {frac:6.1%} non-null"
                    + ("  ALL-NaN" if frac == 0.0 else ""))

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_fpath)
    # the .csv twin is not read by anything (the figures and the significance
    # analysis both read the parquet) and at ~20.6k stations x 29 metrics it is
    # a multi-GB write on every rebuild, so it is no longer produced.
    logger.info(f"Cached -> {cache_fpath}")
    return df


# ===========================================================================
#  Stage 2 — aggregate to FULL / REDUCED ensembles and plot
# ===========================================================================


def aggregate(df: pd.DataFrame, models, col: str) -> pd.DataFrame:
    """Per-station ensemble mean of `col` over `models` (drops NaNs first)."""
    sub = df[df["model"].isin(models)].dropna(subset=[col]).copy()
    if AGGREGATION == "model_mean":
        # members -> model, then models -> ensemble (one model = one vote)
        per_model = (sub.groupby(["station", "lon", "lat", "model"],
                                 as_index=False)[col].mean())
        out = (per_model.groupby(["station", "lon", "lat"],
                                 as_index=False)[col].mean())
    elif AGGREGATION == "realization_mean":
        out = sub.groupby(["station", "lon", "lat"], as_index=False)[col].mean()
    else:
        raise ValueError(f"Unknown AGGREGATION={AGGREGATION!r}")
    return out


def ensemble_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-station full / reduced / diff for value column `col`.
    Under method=weighting the "reduced" column holds the WEIGHTED ensemble
    mean (internal name kept so downstream plotting works unchanged)."""
    full = aggregate(df, FULL_MODELS, col).rename(columns={col: "full"})
    if em.is_weighting():
        red = aggregate_weighted(df, col).rename(columns={col: "reduced"})
    else:
        red = aggregate(df, REDUCED_MODELS, col).rename(columns={col: "reduced"})
    m = full.merge(red, on=["station", "lon", "lat"], how="inner")
    m["diff"] = m["full"] - m["reduced"]
    return m


def _weighted_quantile(v, w, qs):
    """Weighted quantiles (linear interpolation of the weighted ECDF at the
    midpoints, Harrell type). Used only under method=weighting."""
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    if v.size == 0 or w.sum() <= 0:
        return np.full(len(np.atleast_1d(qs)), np.nan)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.atleast_1d(qs), cw, v)


def _station_means(tbl: pd.DataFrame, col: str):
    """Per-station ensemble mean of a model-level frame; WEIGHTED when the
    frame carries the '_w' column (method=weighting), plain otherwise."""
    if "_w" in tbl.columns:
        num = (tbl[col] * tbl["_w"]).groupby(tbl["station"]).sum()
        den = tbl["_w"].groupby(tbl["station"]).sum()
        return (num / den).to_numpy()
    return tbl.groupby("station")[col].mean().to_numpy()


def _station_band(tbl: pd.DataFrame, col: str, blo: float, bhi: float):
    """Per-station (med, lo, hi) of a model-level frame, sorted by med;
    weighted quantiles when the frame carries '_w' (method=weighting)."""
    if "_w" in tbl.columns:
        def _wq(g):
            lo, med, hi = _weighted_quantile(g[col].to_numpy(float),
                                             g["_w"].to_numpy(float),
                                             (blo, 0.5, bhi))
            return pd.Series({"med": med, "lo": lo, "hi": hi})
        a = tbl.groupby("station")[[col, "_w"]].apply(_wq)
    else:
        g = tbl.groupby("station")[col]
        a = pd.DataFrame({"med": g.median(), "lo": g.quantile(blo),
                          "hi": g.quantile(bhi)})
    return a.dropna().sort_values("med")


def _diff_ticks(dnorm):
    """Explicit colorbar ticks for the difference panels.

    TwoSlopeNorm gives each sign half the physical bar regardless of
    magnitude; the default tick locator works in data units over
    [vmin, vmax] and so drops round numbers from the small-magnitude
    (usually negative) half — that half then shows colour but no labels.
    Build round ticks INDEPENDENTLY for the negative and positive sides so
    both halves are labelled. Returns None for the one-signed (Normalize,
    vmin=0) branch, where the default ticks are already correct.
    """
    from matplotlib.ticker import MaxNLocator
    vmin = float(dnorm.vmin)
    vmax = float(dnorm.vmax)
    if vmin >= 0:
        return None
    loc = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10])
    neg = [t for t in loc.tick_values(vmin, 0.0) if vmin <= t < 0.0]
    pos = [t for t in loc.tick_values(0.0, vmax) if 0.0 < t <= vmax]
    return np.unique(np.array(neg + [0.0] + pos))


def _greyify(cmap, mode="center", grey=None, halfwidth=None):
    """Return a copy of `cmap` whose value~0 region is recoloured to grey.

    Only positions within `halfwidth` of the relevant anchor are blended
    toward `grey`; everything else is the original colormap, so the strong
    end colours (and the colormap's identity) are preserved.

    mode="center" : diverging maps used with a 0-centred norm — the centre
                    (normalized position 0.5, i.e. data value 0) goes grey.
    mode="low"    : sequential maps used for one-signed difference panels
                    (Normalize(vmin=0,...)) — the low end (position 0,
                    i.e. data value 0) goes grey.
    """
    grey = ZERO_GREY if grey is None else grey
    halfwidth = GREY_HALFWIDTH if halfwidth is None else halfwidth
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    x = np.linspace(0.0, 1.0, 256)
    cols = base(x).copy()                       # (256, 4) RGBA
    g = np.asarray(to_rgb(grey))
    anchor = 0.5 if mode == "center" else 0.0
    d = np.abs(x - anchor) / max(halfwidth, 1e-6)
    w = np.clip(1.0 - d, 0.0, 1.0)              # 1 at anchor -> 0 at band edge
    w = w * w * (3.0 - 2.0 * w)                 # smoothstep (soft transition)
    cols[:, :3] = (1.0 - w)[:, None] * cols[:, :3] + w[:, None] * g
    return ListedColormap(cols)


def _scatter_outlined(ax, lon, lat, val, cmap, title, **kw):
    """fig02's _scatter, then add a thin neutral edge + bump the marker size
    so value~0 (white) catchments stay visible on the light background.
    Post-processes the returned PathCollection to avoid editing the shared
    fig02 helper."""
    sc = _scatter(ax, lon, lat, val, cmap, title, **kw)
    sc.set_sizes([MARKER_SIZE])
    if MARKER_EDGE is not None:
        sc.set_edgecolor(MARKER_EDGE)
        sc.set_linewidth(MARKER_EDGE_LW)
    return sc


def _row(axes, m, lim, cmap, labels, cbar_label, dcbar_label):
    """Draw one index row: full, reduced (shared symmetric scale) + diff.

    Uses `RdYlBu_r` for the diverging panels (light yellow ≈ 0, bright red
    for positive, bright blue for negative — no white centre, no near-black
    extremes) and `cmc.batlow` for the sequential one-signed diff branch
    (already bright; dark at 0 which here represents "no bias").
    """
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    s = _scatter_outlined(axes[0], m.lon, m.lat, m["full"], cmap, labels[0],
                          norm=norm)
    _scatter_outlined(axes[1], m.lon, m.lat, m["reduced"], cmap, labels[1],
                      norm=norm)
    dcmap, dnorm = _diff_norm_cmap(m["diff"])
    # `_diff_norm_cmap` returns sequential "Reds" (one-signed, vmin=0) or
    # diverging "RdBu_r" (0-centred). Swap each for a brighter equivalent.
    dcmap = "cmc.batlow" if dcmap == "Reds" else "RdYlBu_r"
    sd = _scatter_outlined(axes[2], m.lon, m.lat, m["diff"], dcmap, labels[2],
                           norm=dnorm)
    fig = axes[0].figure
    fig.colorbar(s, ax=axes[:2], orientation="horizontal", fraction=0.045,
                 pad=0.03, aspect=45, label=cbar_label)
    cb = fig.colorbar(sd, ax=axes[2], orientation="horizontal", fraction=0.045,
                      pad=0.03, aspect=22, label=dcbar_label,
                      ticks=_diff_ticks(dnorm))
    cb.ax.tick_params(labelsize=8)


def load_cache(engine: str) -> pd.DataFrame:
    """Load (or build) the per-engine Stage-1 cache that covers ALL SCENARIOS
    x ALL_PERIODS. Rebuilds only on a genuine schema change (a missing INDICES
    column) or RECOMPUTE. Missing scenarios/periods are NOT force-recomputed
    (re-running cannot add data that was never simulated — use RECOMPUTE for
    that); they are reported and skipped gracefully downstream."""
    cache_fpath = _cache_fpath(engine)
    if os.path.exists(cache_fpath) and not RECOMPUTE:
        logger.info(f"Loading cached per-realization table {cache_fpath} "
                    "(set RECOMPUTE=True to rebuild)")
        df = pd.read_parquet(cache_fpath)
        # auto-rebuild on schema drift. The trigger is now the FULL registry
        # (ALL_INDICES), not just the columns the maps happen to draw: a cache
        # written before a metric existed would otherwise be reused and that
        # metric would silently never reach the bias-ranking figure.
        wanted_cols = ({c for c, _, _ in INDICES}
                       | {c for c, _, _ in DIFF_INDICES}
                       | {c for c, _, _ in ALL_INDICES})
        missing = sorted(c for c in wanted_cols if c not in df.columns)
        if missing:
            logger.info(f"  cache lacks {len(missing)} registry column(s) "
                        f"{missing} -> recomputing")
            df = compute_cache(engine)
    else:
        df = compute_cache(engine)

    # drop blocklisted realizations (corrupt / degenerate qqmap_day deltas);
    # cleans an EXISTING per-engine cache at plot time without a recompute.
    df = apply_blocklist(df, label=f"fig05 {engine} cache")

    have_s = sorted(set(df["scenario"].astype(str)))
    have_p = sorted(set(df["period"].astype(str)))
    miss_s = [s for s in SCENARIOS if s not in have_s]
    miss_p = [p for p in ALL_PERIODS if p not in have_p]
    if miss_s:
        logger.warning(f"[{engine}] cache has no rows for scenarios {miss_s} "
                        "(not simulated?) -> absent from the boxplot")
    if miss_p:
        logger.warning(f"[{engine}] cache has no rows for periods {miss_p}")
    logger.info(f"[{engine}] cache: {df['station'].nunique()} stations, "
                f"scenarios={have_s}, periods={have_p}, {len(df)} rows")
    logger.info(f"FULL models ({len(FULL_MODELS)}): {FULL_MODELS}")
    logger.info(f"REDUCED models ({len(REDUCED_MODELS)}): {REDUCED_MODELS}")
    logger.info(f"Excluded as {_XHOT}: "
                f"{sorted(set(FULL_MODELS) - set(REDUCED_MODELS))}")
    return df


def plot_maps(df: pd.DataFrame, engine: str, scenario: str, period: str):
    """9-panel MAP figure for one (engine, scenario, period)."""
    dmap = df[(df["scenario"].astype(str) == scenario)
              & (df["period"].astype(str) == period)]
    if dmap.empty:
        logger.warning(f"[{engine}] no cached rows for scenario="
                        f"{scenario} / period={period}; skipping maps")
        return None

    # how many distinct models in FULL_MODELS / REDUCED_MODELS actually have
    # cached data for this (scenario, period) — gets stamped on every panel
    # title because the count varies across scenarios & periods (not every
    # model has every (scenario, period) realization).
    present = set(dmap["model"])
    n_full = sum(1 for m in FULL_MODELS    if m in present)
    n_red  = sum(1 for m in REDUCED_MODELS if m in present)

    tables = {col: ensemble_table(dmap, col) for col, _, _ in INDICES}
    for col, label, _ in INDICES:
        t = tables[col]
        logger.info(f"  [{engine} {scenario} {period}] {label}: "
                    f"n={len(t)}  full={t['full'].mean():.2f}  "
                    f"reduced={t['reduced'].mean():.2f}  "
                    f"diff={t['diff'].mean():.2f}")
    logger.info(f"  [{engine} {scenario} {period}] models present: "
                f"full={n_full}, reduced={n_red}")

    try:
        import cartopy.crs as ccrs
        proj = dict(subplot_kw={"projection": ccrs.Robinson()})
    except Exception:
        proj = {}

    fig, axes = plt.subplots(3, 3, figsize=(19, 13.5),
                             gridspec_kw=dict(wspace=WSPACE, hspace=HSPACE),
                             **proj)

    panel = "abcdefghi"
    for r, (col, label, cmap) in enumerate(INDICES):
        m = tables[col]
        # shared, robust, symmetric scale for the full/reduced columns
        lim = float(np.nanpercentile(
            np.abs(np.r_[m["full"], m["reduced"]]), 98))
        if not (np.isfinite(lim) and lim > 0):
            lim = 1.0
        b = r * 3
        _row(axes[r], m, lim, cmap,
             labels=[f"({panel[b]}) {label}: full ensemble (n={n_full})",
                     f"({panel[b+1]}) {label}: reduced ensemble (n={n_red})",
                     f"({panel[b+2]}) {label}: difference "
                     f"(full − reduced; {n_full}/{n_red})"],
             cbar_label=label,
             dcbar_label=f"Δ {label}  (full − reduced)")

    n_sta = int(dmap["station"].nunique())
    out_png = _out_png(engine, scenario, period, n_sta)
    os.makedirs(MAP_9PANEL_DIR, exist_ok=True)
    add_panel_letters(axes)
    out_png = em.tag(out_png)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_png}")
    return out_png


def _whisker_lims(arrays, whis: float = 1.5, pad_frac: float = 0.05):
    """(ymin, ymax) covering the ACTUAL box/whisker extent across `arrays`.

    The boxes are drawn with showfliers=False, so the visible extent is the
    whisker caps (matplotlib default: the most extreme point within
    Q1-1.5*IQR .. Q3+1.5*IQR), NOT the data's 1-99th percentile. Heavy tails
    (esp. Q05, and the skewed global-mean markers) make the percentile range
    far wider than the whiskers, which is why the axis dwarfed the boxes.
    Returns None if nothing finite is present.
    """
    lo, hi = np.inf, -np.inf
    for v in arrays:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        inside_lo = v[v >= q1 - whis * iqr]
        inside_hi = v[v <= q3 + whis * iqr]
        wlo = inside_lo.min() if inside_lo.size else v.min()
        whi = inside_hi.max() if inside_hi.size else v.max()
        lo, hi = min(lo, wlo), max(hi, whi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        return None
    pad = pad_frac * (hi - lo)
    return (lo - pad, hi + pad)


def boxplots_by_scenario(df: pd.DataFrame, engine: str):
    """Per-engine summary: distribution ACROSS CATCHMENTS of the per-catchment
    ensemble-mean change, FULL vs the second ensemble -- the WEIGHTED ensemble
    under method=weighting, the reduced (too-hot-removed) one under
    method=removal; ensemble_table decides which, and the legend follows it via
    em.is_weighting() rather than assuming removal -- with the global
    mean (mean over catchments) overlaid. Grid = len(CORE_BOX_INDICES) rows
    (the 6 core indices, shared with boxplots_full_reduced_by_engine) ×
    len(ALL_PERIODS) columns; each panel has one full+reduced box pair per
    scenario in SCENARIOS (reddish = full, bluish = reduced). This is the
    SCENARIO-GRADIENT view (one engine per file, x-axis = scenarios); the
    engine-comparison companion is boxplots_full_reduced_by_engine (one
    scenario per file, x-axis = engines). Same model-democracy aggregation
    as the maps (via ensemble_table). Mirrors fig02.boxplots_by_scenario."""
    scenarios = [s for s in SCENARIOS if s in set(df["scenario"].astype(str))]
    periods   = [p for p in ALL_PERIODS if p in set(df["period"].astype(str))]
    if not scenarios or not periods:
        logger.warning(f"[{engine}] boxplots_by_scenario: no scenarios/periods "
                        "in cache, skipping")
        return None

    # per (col, period, scenario): per-station FULL & REDUCED ensemble means
    data = {}
    for p in periods:
        for s in scenarios:
            sub = df[(df["scenario"].astype(str) == s)
                     & (df["period"].astype(str) == p)]
            if sub.empty:
                continue
            for col, _ in CORE_BOX_INDICES:
                et = ensemble_table(sub, col)
                data[(col, p, s)] = (et["full"].to_numpy(dtype=float),
                                     et["reduced"].to_numpy(dtype=float))

    nsta = df["station"].nunique()
    dx = 0.19
    col_w = max(5.5, 0.9 * len(scenarios) + 1.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Split the core indices across several figures (BY_SCENARIO_ROWS_PER_FIG
    # rows each); 6 rows in one figure was too cramped. Each chunk -> its own
    # PNG, tagged _set{k} when there is more than one chunk.
    chunks = [CORE_BOX_INDICES[i:i + BY_SCENARIO_ROWS_PER_FIG]
              for i in range(0, len(CORE_BOX_INDICES),
                             BY_SCENARIO_ROWS_PER_FIG)]
    outs = []
    for ci, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(len(chunk), len(periods),
                                 figsize=(col_w * len(periods),
                                          2.5 * len(chunk) + 0.8),
                                 sharex=True, sharey="row", squeeze=False)

        for r, (col, label) in enumerate(chunk):
            # shared y-limits across the row = the actual drawn box/whisker
            # extent (showfliers=False), so heavy tails / skewed global means
            # no longer inflate the axis far past the whiskers.
            row_arrays = [a for (cc, _, _), pair in data.items() if cc == col
                          for a in pair]
            ylim = _whisker_lims(row_arrays)

            for c, per in enumerate(periods):
                ax = axes[r][c]
                for k, s in enumerate(scenarios, start=1):
                    pair = data.get((col, per, s))
                    if pair is None:
                        continue
                    for vals, xoff, color in ((pair[0], -dx, FULL_BOX_C),
                                              (pair[1], +dx, REDUCED_BOX_C)):
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:    # boxplot errors on empty input
                            continue
                        bp = ax.boxplot(vals, positions=[k + xoff],
                                        widths=0.32, showfliers=False,
                                        patch_artist=True)
                        bp["boxes"][0].set(facecolor=color, alpha=0.7,
                                           linewidth=0.6)
                        bp["medians"][0].set(color="black", linewidth=1.0)
                        for w in bp["whiskers"] + bp["caps"]:
                            w.set(linewidth=0.5)
                        ax.scatter([k + xoff], [np.nanmean(vals)], marker="D",
                                   s=22, facecolor="white", edgecolor="black",
                                   linewidth=0.7, zorder=5)         # global mean
                ax.axhline(0.0, color="0.5", lw=0.7, zorder=0)      # sign ref
                ax.set_xlim(0.5, len(scenarios) + 0.5)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.grid(axis="y", color="0.92", lw=0.5)
                ax.set_axisbelow(True)
                ax.tick_params(axis="y", labelsize=10)
                if r == 0:
                    ax.set_title(plabel(per), fontsize=13)
                if c == 0:
                    ax.set_ylabel(label, fontsize=11)

        for c in range(len(periods)):
            axes[-1][c].set_xticks(range(1, len(scenarios) + 1))
            axes[-1][c].set_xticklabels([s.upper() for s in scenarios],
                                        fontsize=10)

        axes[0][0].legend(
            handles=[Patch(facecolor=FULL_BOX_C, alpha=0.7,
                           label="full ensemble"),
                     # Under method=weighting the second box is the WEIGHTED
                     # ensemble, not a removal subset: ensemble_table routes it
                     # through aggregate_weighted and keeps the internal column
                     # name "reduced" only so downstream code works unchanged.
                     # Naming it "reduced (too-hot removed)" therefore described
                     # a treatment that was not applied. Same conditional as
                     # spread_projection below, so the two figures agree.
                     Patch(facecolor=REDUCED_BOX_C, alpha=0.7,
                           label=("weighted" if em.is_weighting()
                                  else f"reduced ({_XHOT} removed)")),
                     Line2D([0], [0], marker="D", linestyle="None",
                            markerfacecolor="white", markeredgecolor="black",
                            label="global mean")],
            loc="best", fontsize=11, framealpha=0.7)

        part = ci if len(chunks) > 1 else None
        out = _box_png(engine, nsta, part=part)
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        add_panel_letters(axes)
        fig.tight_layout()
        out = em.tag(out)
        fig.savefig(out, dpi=250, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out} ({len(scenarios)} scenarios × {len(periods)} "
                    f"periods × {len(chunk)} indices, full vs reduced, "
                    f"{nsta} catchments, {engine})")
        outs.append(out)

    return outs


def boxplots_full_reduced_by_engine(df_hbv: pd.DataFrame,
                                    df_lstm: pd.DataFrame,
                                    scenario: str,
                                    indices=None):
    """ENGINE-COMPARISON boxplot (HBV vs LSTM) in one figure, for a single
    focus `scenario`.

    Grid = `indices` (rows; default CORE_BOX_INDICES) × ALL_PERIODS (cols).
    The x-axis groups the two ENSEMBLES (full, reduced); within each group the
    HBV and LSTM boxes sit side by side (colour = engine: HBV orange, LSTM
    blue), so the difference BETWEEN ENGINES is read directly within each
    ensemble. Distribution = the across-catchment spread of the per-catchment
    ensemble-mean change (same model-democracy aggregation as the maps, via
    ensemble_table), with the global-mean diamond overlaid. Shared y-limits
    per row from the actual box/whisker extent (showfliers=False). The
    scenario-gradient / full-vs-reduced companion is boxplots_by_scenario.

    Output: hydro_boxplots_engines_{scenario}_n{N}sta.png.
    """
    indices = CORE_BOX_INDICES if indices is None else indices
    engines = [("HBV", df_hbv), ("LSTM", df_lstm)]
    periods = [p for p in ALL_PERIODS
               if all(p in set(d["period"].astype(str)) for _, d in engines)]
    if not periods:
        logger.warning(f"boxplots_full_reduced_by_engine({scenario}): no "
                        "period common to both engines; skipping")
        return None

    # per (col, period, engine): per-station full & reduced arrays; also
    # accumulate the contributing stations for the filename count.
    data = {}
    drawn = set()
    for col, _ in indices:
        for p in periods:
            for ename, d in engines:
                if col not in d.columns:
                    continue
                sub = d[(d["scenario"].astype(str) == scenario)
                        & (d["period"].astype(str) == p)]
                if sub.empty:
                    continue
                et = ensemble_table(sub, col)
                if et.empty:
                    continue
                data[(col, p, ename)] = (et["full"].to_numpy(float),
                                         et["reduced"].to_numpy(float))
                drawn.update(et["station"])

    if not data:
        logger.warning(f"boxplots_full_reduced_by_engine({scenario}): no "
                        "usable data; skipping")
        return None

    dx = 0.19
    # Compact size: the 6×2 grid was 10.4×19.8 in, so the default-pt fonts
    # rendered tiny when the tall figure was scaled to fit. Smaller panels +
    # larger fonts below make the labels readable at view/print scale.
    fig, axes = plt.subplots(len(indices), len(periods),
                             figsize=(4.1 * len(periods),
                                      2.05 * len(indices) + 0.9),
                             sharex=True, sharey="row", squeeze=False)

    for r, (col, label) in enumerate(indices):
        # shared y-limits across the row = actual drawn box/whisker extent
        row_arrays = [a for (cc, _, _), pair in data.items() if cc == col
                      for a in pair]
        ylim = _whisker_lims(row_arrays)
        for c, per in enumerate(periods):
            ax = axes[r][c]
            # x-groups = ensembles (full at 1, reduced at 2); within each group
            # HBV (left) and LSTM (right) sit side by side, coloured by engine,
            # so the engine difference is read directly. pair = (full, reduced).
            for ens, base, idx in (("full", 1, 0), ("reduced", 2, 1)):
                for ename, xoff, color in (("HBV", -dx, ENGINE_HBV_C),
                                           ("LSTM", +dx, ENGINE_LSTM_C)):
                    pair = data.get((col, per, ename))
                    if pair is None:
                        continue
                    vals = pair[idx]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    bp = ax.boxplot(vals, positions=[base + xoff],
                                    widths=0.32, showfliers=False,
                                    patch_artist=True)
                    bp["boxes"][0].set(facecolor=color, alpha=0.75,
                                       linewidth=0.6)
                    bp["medians"][0].set(color="black", linewidth=1.0)
                    for w in bp["whiskers"] + bp["caps"]:
                        w.set(linewidth=0.5)
                    ax.scatter([base + xoff], [np.nanmean(vals)], marker="D",
                               s=22, facecolor="white", edgecolor="black",
                               linewidth=0.7, zorder=5)            # global mean
            ax.axhline(0.0, color="0.5", lw=0.7, zorder=0)         # sign ref
            ax.set_xlim(0.5, 2.5)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", color="0.92", lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=11)
            if r == 0:
                ax.set_title(plabel(per), fontsize=14)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)

    for c in range(len(periods)):
        axes[-1][c].set_xticks([1, 2])
        axes[-1][c].set_xticklabels(["full", "reduced"], fontsize=13)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axes[0][0].legend(
        handles=[Patch(facecolor=ENGINE_HBV_C, alpha=0.75, label="HBV"),
                 Patch(facecolor=ENGINE_LSTM_C, alpha=0.75, label="LSTM"),
                 Line2D([0], [0], marker="D", linestyle="None",
                        markerfacecolor="white", markeredgecolor="black",
                        label="global mean")],
        loc="best", fontsize=10, framealpha=0.7)

    n_sta = len(drawn)
    fig.suptitle(f"HBV vs LSTM by ensemble: {scenario.upper()}",
                 fontsize=15, y=0.995)
    out = os.path.join(
        SUMMARY_DIR,
        f"hydro_boxplots_engines_{scenario}_n{n_sta}sta.png")
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    add_panel_letters(axes)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = em.tag(out)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(indices)} indices × {len(periods)} "
                f"periods, HBV vs LSTM by ensemble, {scenario}, "
                f"{n_sta} catchments)")
    return out


def diff_maps_per_index(df: pd.DataFrame, engine: str,
                        col: str, label: str, slug: str):
    """One figure per (index, engine) showing ONLY the full−reduced
    difference (the bias from keeping too-hot models), arranged as
    MAP_SCENARIOS (rows) × ALL_PERIODS (cols) — i.e. 4 × 2 panels with the
    default MAP_SCENARIOS = [ssp126, ssp245, ssp370, ssp585]. The 8 panels
    share a single robust, symmetric, 0-centred RdYlBu_r scale so panel
    magnitudes are directly comparable; one shared horizontal colour bar
    underneath. Catchments without a finite diff value for that index are
    skipped (e.g. flashiness/dry-spell-max can be NaN where the upstream
    calculation failed).
    """
    if col not in df.columns:
        logger.warning(f"[{engine}] diff_maps_per_index({slug}): "
                        f"column {col!r} not in cache; skipping")
        return None
    scenarios = [s for s in MAP_SCENARIOS if s in set(df["scenario"].astype(str))]
    periods   = [p for p in ALL_PERIODS  if p in set(df["period"].astype(str))]
    if not scenarios or not periods:
        logger.warning(f"[{engine}] diff_maps_per_index({slug}): no scenarios/"
                        "periods in cache; skipping")
        return None

    # per-(scenario, period) ensemble_table -> per-station diff; collect all
    # finite diff values to derive a SHARED symmetric robust scale.
    tables = {}
    diff_all = []
    for s in scenarios:
        for p in periods:
            sub = df[(df["scenario"].astype(str) == s)
                     & (df["period"].astype(str) == p)]
            if sub.empty:
                tables[(s, p)] = None
                continue
            et = ensemble_table(sub, col)
            if et.empty:
                tables[(s, p)] = None
                continue
            tables[(s, p)] = et
            diff_all.append(et["diff"].to_numpy(float))

    diff_all = np.concatenate(diff_all) if diff_all else np.array([])
    diff_all = diff_all[np.isfinite(diff_all)]
    if diff_all.size == 0:
        logger.warning(f"[{engine}] diff_maps_per_index({slug}): no finite "
                        "diff values; skipping")
        return None

    lim = float(np.nanpercentile(np.abs(diff_all), 98))
    if not (np.isfinite(lim) and lim > 0):
        lim = 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = "RdYlBu_r"

    try:
        import cartopy.crs as ccrs
        proj = dict(subplot_kw={"projection": ccrs.Robinson()})
    except Exception:
        proj = {}

    fig, axes = plt.subplots(
        len(scenarios), len(periods),
        figsize=(14, 3.3 * len(scenarios) + 0.6),
        gridspec_kw=dict(wspace=WSPACE, hspace=HSPACE),
        squeeze=False, **proj)

    last_sc = None
    panel = "abcdefghij"
    pi = 0
    for r, s in enumerate(scenarios):
        for c, p in enumerate(periods):
            ax = axes[r][c]
            # per-panel model counts (FULL_MODELS / REDUCED_MODELS that have
            # cached rows for this exact (scenario, period)). Varies panel-
            # to-panel because not every model has every realization.
            sub_sp = df[(df["scenario"].astype(str) == s)
                        & (df["period"].astype(str) == p)]
            present_sp = set(sub_sp["model"])
            n_full_sp = sum(1 for mm in FULL_MODELS    if mm in present_sp)
            n_red_sp  = sum(1 for mm in REDUCED_MODELS if mm in present_sp)
            title = f"({panel[pi]}) {s.upper()}, {p}  ({n_full_sp}/{n_red_sp})"
            pi += 1
            et = tables.get((s, p))
            if et is None:
                ax.set_title(title + " (no data)", fontsize=9)
                continue
            sc = _scatter_outlined(ax, et["lon"], et["lat"], et["diff"],
                                   cmap, title, norm=norm)
            last_sc = sc

    if last_sc is not None:
        cb = fig.colorbar(last_sc,
                          ax=axes.ravel().tolist(),
                          orientation="horizontal",
                          fraction=0.04, pad=0.04, aspect=50,
                          label=f"Δ {label}  (full − reduced)",
                          ticks=_diff_ticks(norm))
        cb.ax.tick_params(labelsize=8)

    # n_sta = unique stations that contributed ≥1 panel
    drawn = set()
    for et in tables.values():
        if et is not None:
            drawn.update(et["station"])
    n_sta = len(drawn) if drawn else int(df["station"].nunique())

    out = os.path.join(MAP_DIFF_DIR,
                       f"diff_{slug}_{engine}_n{n_sta}sta.png")
    os.makedirs(MAP_DIFF_DIR, exist_ok=True)
    add_panel_letters(axes)
    out = em.tag(out)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(scenarios)} scenarios × {len(periods)} "
                f"periods, diff full − reduced, {n_sta} catchments, {engine})")
    return out


def value_maps_per_index(df: pd.DataFrame, engine: str,
                         col: str, label: str, slug: str,
                         kind: str):
    """One 4×2 (MAP_SCENARIOS × ALL_PERIODS) figure per (engine, index, kind)
    showing the per-station % change of the chosen ensemble subset (NOT the
    diff — `diff_maps_per_index` covers that).

      kind="full"    -> per-station % change of the FULL CMIP6 ensemble
      kind="reduced" -> same after removing "too-hot" models

    The full and reduced figures for one (engine, index) **share** a single
    robust symmetric RdYlBu_r colour scale derived from `|full ∪ reduced|`
    over all panels, so the two PNGs can be flipped between and visually
    compared at a glance. Catchments without a finite value are skipped per
    panel; `(n_full / n_reduced)` model counts are stamped in every title.

    Output: `paper/figures/hydro_indices/maps/per_index_{kind}/{kind}_{slug}_{engine}_n{N}sta.png`.
    """
    if kind not in ("full", "reduced"):
        raise ValueError(
            f"value_maps_per_index kind must be 'full' or 'reduced', "
            f"got {kind!r}")
    if col not in df.columns:
        logger.warning(f"[{engine}] value_maps_per_index({slug},{kind}): "
                        f"column {col!r} not in cache; skipping")
        return None
    scenarios = [s for s in MAP_SCENARIOS if s in set(df["scenario"].astype(str))]
    periods   = [p for p in ALL_PERIODS  if p in set(df["period"].astype(str))]
    if not scenarios or not periods:
        logger.warning(f"[{engine}] value_maps_per_index({slug},{kind}): no "
                        "scenarios/periods in cache; skipping")
        return None

    # Per-(scenario, period) ensemble_table; pool full ∪ reduced for the
    # SHARED colour scale across the full and reduced figures.
    tables = {}
    pool = []
    for s in scenarios:
        for p in periods:
            sub = df[(df["scenario"].astype(str) == s)
                     & (df["period"].astype(str) == p)]
            if sub.empty:
                tables[(s, p)] = None
                continue
            et = ensemble_table(sub, col)
            if et.empty:
                tables[(s, p)] = None
                continue
            tables[(s, p)] = et
            pool.append(et["full"].to_numpy(float))
            pool.append(et["reduced"].to_numpy(float))

    pool = np.concatenate(pool) if pool else np.array([])
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        logger.warning(f"[{engine}] value_maps_per_index({slug},{kind}): "
                        "no finite full/reduced values; skipping")
        return None

    lim = float(np.nanpercentile(np.abs(pool), 98))
    if not (np.isfinite(lim) and lim > 0):
        lim = 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = "RdYlBu_r"

    try:
        import cartopy.crs as ccrs
        proj = dict(subplot_kw={"projection": ccrs.Robinson()})
    except Exception:
        proj = {}

    fig, axes = plt.subplots(
        len(scenarios), len(periods),
        figsize=(14, 3.3 * len(scenarios) + 0.6),
        gridspec_kw=dict(wspace=WSPACE, hspace=HSPACE),
        squeeze=False, **proj)

    last_sc = None
    panel = "abcdefghij"
    pi = 0
    for r, s in enumerate(scenarios):
        for c, p in enumerate(periods):
            ax = axes[r][c]
            sub_sp = df[(df["scenario"].astype(str) == s)
                        & (df["period"].astype(str) == p)]
            present_sp = set(sub_sp["model"])
            n_full_sp = sum(1 for mm in FULL_MODELS    if mm in present_sp)
            n_red_sp  = sum(1 for mm in REDUCED_MODELS if mm in present_sp)
            title = f"({panel[pi]}) {s.upper()}, {p}  ({n_full_sp}/{n_red_sp})"
            pi += 1
            et = tables.get((s, p))
            if et is None:
                ax.set_title(title + " (no data)", fontsize=9)
                continue
            sc = _scatter_outlined(ax, et["lon"], et["lat"], et[kind],
                                   cmap, title, norm=norm)
            last_sc = sc

    if last_sc is not None:
        cb = fig.colorbar(last_sc,
                          ax=axes.ravel().tolist(),
                          orientation="horizontal",
                          fraction=0.04, pad=0.04, aspect=50,
                          label=f"{label}  ({kind} ensemble)",
                          ticks=_diff_ticks(norm))
        cb.ax.tick_params(labelsize=8)

    drawn = set()
    for et in tables.values():
        if et is not None:
            drawn.update(et["station"])
    n_sta = len(drawn) if drawn else int(df["station"].nunique())

    target_dir = MAP_FULL_DIR if kind == "full" else MAP_REDUCED_DIR
    out = os.path.join(target_dir,
                       f"{slug}_{engine}_n{n_sta}sta.png")
    os.makedirs(target_dir, exist_ok=True)
    add_panel_letters(axes)
    out = em.tag(out)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(scenarios)} scenarios × {len(periods)} "
                f"periods, {kind} ensemble, {n_sta} catchments, {engine})")
    return out


def engine_disagreement_maps(df_a: pd.DataFrame, df_b: pd.DataFrame,
                             col: str, label: str, slug: str,
                             engine_a: str = "lstm", engine_b: str = "hbv"):
    """For ONE index, plot where the two hydrological engines DISAGREE about
    the too-hot bias. Per station and per (scenario, period):

        engine_diff =  (engine_a full−reduced)  −  (engine_b full−reduced)

    With the default `engine_a="lstm"`, `engine_b="hbv"`: positive (red)
    means LSTM's full-vs-reduced bias is larger than HBV's at that catchment;
    negative (blue) means HBV's bias is larger. Near zero = the engines agree
    on the too-hot bias. Layout = MAP_SCENARIOS (rows) × ALL_PERIODS (cols),
    shared diverging RdYlBu_r scale across all 8 panels for comparability.
    """
    if col not in df_a.columns or col not in df_b.columns:
        logger.warning(f"engine_disagreement_maps({slug}): column {col!r} "
                        "missing in one of the engine caches; skipping")
        return None

    scen_set_a = set(df_a["scenario"].astype(str))
    scen_set_b = set(df_b["scenario"].astype(str))
    per_set_a  = set(df_a["period"].astype(str))
    per_set_b  = set(df_b["period"].astype(str))
    scenarios = [s for s in MAP_SCENARIOS if s in scen_set_a and s in scen_set_b]
    periods   = [p for p in ALL_PERIODS   if p in per_set_a  and p in per_set_b]
    if not scenarios or not periods:
        logger.warning(f"engine_disagreement_maps({slug}): no scenarios/"
                        "periods common to both engines; skipping")
        return None

    # per-(scenario, period) per-station engine_diff
    tables = {}
    diff_all = []
    for s in scenarios:
        for p in periods:
            sub_a = df_a[(df_a["scenario"].astype(str) == s)
                         & (df_a["period"].astype(str) == p)]
            sub_b = df_b[(df_b["scenario"].astype(str) == s)
                         & (df_b["period"].astype(str) == p)]
            if sub_a.empty or sub_b.empty:
                tables[(s, p)] = None
                continue
            et_a = ensemble_table(sub_a, col)
            et_b = ensemble_table(sub_b, col)
            if et_a.empty or et_b.empty:
                tables[(s, p)] = None
                continue
            merged = (et_a[["station", "lon", "lat", "diff"]]
                      .rename(columns={"diff": "diff_a"})
                      .merge(et_b[["station", "lon", "lat", "diff"]]
                             .rename(columns={"diff": "diff_b"}),
                             on=["station", "lon", "lat"], how="inner"))
            if merged.empty:
                tables[(s, p)] = None
                continue
            merged["engine_diff"] = merged["diff_a"] - merged["diff_b"]
            tables[(s, p)] = merged
            diff_all.append(merged["engine_diff"].to_numpy(float))

    diff_all = np.concatenate(diff_all) if diff_all else np.array([])
    diff_all = diff_all[np.isfinite(diff_all)]
    if diff_all.size == 0:
        logger.warning(f"engine_disagreement_maps({slug}): no finite "
                        "engine-difference values; skipping")
        return None

    lim = float(np.nanpercentile(np.abs(diff_all), 98))
    if not (np.isfinite(lim) and lim > 0):
        lim = 1.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = "RdYlBu_r"

    try:
        import cartopy.crs as ccrs
        proj = dict(subplot_kw={"projection": ccrs.Robinson()})
    except Exception:
        proj = {}

    fig, axes = plt.subplots(
        len(scenarios), len(periods),
        figsize=(14, 3.3 * len(scenarios) + 0.6),
        gridspec_kw=dict(wspace=WSPACE, hspace=HSPACE),
        squeeze=False, **proj)

    last_sc = None
    panel = "abcdefghij"
    pi = 0
    for r, s in enumerate(scenarios):
        for c, p in enumerate(periods):
            ax = axes[r][c]
            title_base = f"({panel[pi]}) {s.upper()}, {p}"
            pi += 1
            et = tables.get((s, p))
            if et is None:
                ax.set_title(title_base + " (no data)", fontsize=9)
                continue
            sc = _scatter_outlined(ax, et["lon"], et["lat"],
                                   et["engine_diff"], cmap, title_base,
                                   norm=norm)
            last_sc = sc

    if last_sc is not None:
        cb_label = (f"{engine_a.upper()} − {engine_b.upper()}   of   "
                    f"Δ {label}  (full − reduced)")
        cb = fig.colorbar(last_sc,
                          ax=axes.ravel().tolist(),
                          orientation="horizontal",
                          fraction=0.04, pad=0.04, aspect=50,
                          label=cb_label,
                          ticks=_diff_ticks(norm))
        cb.ax.tick_params(labelsize=8)

    drawn = set()
    for et in tables.values():
        if et is not None:
            drawn.update(et["station"])
    n_sta = len(drawn)

    out = os.path.join(MAP_ENGINEDIFF_DIR,
                       f"{slug}_enginediff_n{n_sta}sta.png")
    os.makedirs(MAP_ENGINEDIFF_DIR, exist_ok=True)
    add_panel_letters(axes)
    out = em.tag(out)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({len(scenarios)} scenarios × {len(periods)} "
                f"periods, ({engine_a}−{engine_b}) of (full − reduced), "
                f"{n_sta} catchments)")
    return out


def _bias_ranking_rows(df_hbv: pd.DataFrame, df_lstm: pd.DataFrame,
                       scenario: str, period: str):
    """Per-index data the bias-ranking panels draw, for one (scenario, period).

    Returns ``(rows, indices_present)``, where each row is
    ``(slug, label, diff_hbv, diff_lstm, full_hbv, full_lstm,
    stations_hbv, stations_lstm)``. Split out of bias_ranking_per_period so
    the standalone bias-ranking figure and the combined Fig. 3 share exactly
    one data path and can never diverge.
    """
    indices_present = []
    rows = []          # one tuple per included index
    # ALL_INDICES, not DIFF_INDICES: this is the one cross-index summary in the
    # figure set, so it spans the streamflow metrics stats.parquet records, while
    # the per-index map figures stay on the curated DIFF_INDICES subset. The
    # timing, ratio and redundant low-flow metrics named in BIAS_RANKING_EXCLUDE
    # are held out here (and only here); see that constant for why.
    ranking_indices = [(c, l, s) for (c, l, s) in ALL_INDICES
                       if c not in BIAS_RANKING_EXCLUDE]
    for col, label, slug in ranking_indices:
        if col not in df_hbv.columns or col not in df_lstm.columns:
            continue
        sub_h = df_hbv[(df_hbv["scenario"].astype(str) == scenario)
                       & (df_hbv["period"].astype(str) == period)]
        sub_l = df_lstm[(df_lstm["scenario"].astype(str) == scenario)
                        & (df_lstm["period"].astype(str) == period)]
        if sub_h.empty or sub_l.empty:
            continue
        et_h = ensemble_table(sub_h, col)
        et_l = ensemble_table(sub_l, col)
        if et_h.empty or et_l.empty:
            continue
        diff_h = et_h["diff"].to_numpy(float)
        diff_l = et_l["diff"].to_numpy(float)
        full_h = et_h["full"].to_numpy(float)
        full_l = et_l["full"].to_numpy(float)
        indices_present.append(slug)
        rows.append((slug, label, diff_h, diff_l, full_h, full_l,
                     et_h["station"], et_l["station"]))

    return rows, indices_present


# ===========================================================================
#  Regime-stratified artefact share  (section: regime_strata)
# ===========================================================================
#  Recomputes the published artefact share within climate-regime strata. Runs
#  off the SAME `loaded` caches every other section uses, so the blocklist has
#  already been applied by load_cache() and the GLOBAL stratum reproduces
#  report.tex exactly -- a standalone reader of the parquet does NOT, because
#  every blocklisted realization belongs to a too-hot model and so contaminates
#  the full ensemble while leaving the reduced ensemble untouched.
# ===========================================================================
def _load_regime_attrs() -> pd.DataFrame:
    """station -> (aridity, snowfrac, kgz) from static.csv.

    Imported lazily from the significance module so BOTH analyses read the same
    static.csv columns and can never drift apart. The import is deferred (rather
    than done at module scope) because that module reads sys.argv[1] when it is
    imported; neutralising argv here keeps this script's own CLI arguments from
    being misread as its mode selector.
    """
    _argv_bak = sys.argv
    sys.argv = sys.argv[:1]
    try:
        from analyze_full_vs_reduced_significance import (
            _load_static_attrs, TRANSMISSION_ATTRS,
        )
    finally:
        sys.argv = _argv_bak
    logger.info(f"regime attributes from static.csv: {TRANSMISSION_ATTRS}")
    st = _load_static_attrs()
    st["station"] = st["station"].astype(str)
    return st


def _regime_share_row(sub: pd.DataFrame, label: str, group: str) -> dict:
    """Artefact-share statistics for one stratum.

    `sub` carries one row per station with that station's full- and
    reduced-ensemble projected change (the output of `ensemble_table`).

    Two share definitions are reported because the project uses two:
      share_signed = (mean_full - mean_reduced) / mean_full * 100
          the MANUSCRIPT definition (ratio of stratum-mean projections); the
          global stratum reproduces report.tex P15 / P15b.
      share_abs    = mean(full - reduced) / mean(|full|) * 100
          the `bias_share_pct` definition drawn in the bias-ranking share panel.
    Both are ratios, so both are unstable as the denominator approaches zero.
    mean_bias_pp / median_bias_pp are always bounded and are the statistics to
    quote wherever a share is suppressed.
    """
    n = len(sub)
    full = sub["full"].to_numpy(float)
    diff = sub["diff"].to_numpy(float)

    mean_full = float(np.mean(full))
    mean_bias = float(np.mean(diff))
    mean_abs_full = float(np.mean(np.abs(full)))

    # The share means "the fraction of the projected INTENSIFICATION that is a
    # too-hot artefact", so it is defined only where the full ensemble actually
    # projects an intensification. Guarding on |mean_full| would admit strata
    # with a projected DECLINE, where a positive bias over a negative
    # denominator prints a negative share that reads as the opposite of its
    # meaning -- which is exactly the case in the snow-affected strata under
    # HBV. Those strata are reported through mean_bias_pp alone.
    usable = (n >= REGIME_MIN_N) and (mean_full >= REGIME_MIN_DENOM_PCT)
    share_signed = (mean_bias / mean_full * 100.0) if usable else np.nan
    share_abs = ((mean_bias / mean_abs_full * 100.0)
                 if (n >= REGIME_MIN_N
                     and mean_abs_full >= REGIME_MIN_DENOM_PCT) else np.nan)

    return dict(group=group, stratum=label, n=n,
                mean_full_pct=mean_full,
                mean_reduced_pct=float(np.mean(sub["reduced"].to_numpy(float))),
                mean_bias_pp=mean_bias,
                median_bias_pp=float(np.median(diff)),
                mean_abs_full_pct=mean_abs_full,
                share_signed_pct=share_signed, share_abs_pct=share_abs,
                denom_ok=bool(usable))


def _regime_strata_rows(m: pd.DataFrame) -> list:
    """Global + Koeppen + aridity-decile + snow-bin rows for one merged table."""
    rows = [_regime_share_row(m, "GLOBAL (all catchments)", "global")]

    for kgz in sorted(k for k in m["kgz"].dropna().unique() if k in KGZ_LABELS):
        rows.append(_regime_share_row(m[m["kgz"] == kgz],
                                      KGZ_LABELS[kgz], "koppen"))

    # Deciles are cut on THIS table's own aridity distribution (n differs
    # slightly between engines and indices), so each label carries its edges.
    if np.isfinite(m["aridity"].to_numpy(float)).sum() >= (ARIDITY_N_DECILES
                                                           * REGIME_MIN_N):
        dec = pd.qcut(m["aridity"], ARIDITY_N_DECILES, labels=False,
                      duplicates="drop")
        for d in sorted(pd.unique(dec.dropna())):
            sub = m[dec == d]
            rows.append(_regime_share_row(
                sub,
                f"aridity D{int(d) + 1} (PET/P {sub['aridity'].min():.2f}-"
                f"{sub['aridity'].max():.2f})",
                "aridity"))
    else:
        logger.warning("too few catchments with a finite aridity index -> "
                       "aridity deciles skipped")

    sf = pd.cut(m["snowfrac"], bins=SNOW_BINS, labels=SNOW_LABELS)
    for lab in SNOW_LABELS:
        sub = m[sf == lab]
        if len(sub):
            rows.append(_regime_share_row(sub, lab, "snow"))

    return rows


def regime_stratified_bias_share(loaded: dict, scenario: str,
                                 period: str) -> pd.DataFrame:
    """Artefact share by Koeppen zone, aridity decile and snow fraction.

    `loaded` maps engine -> the Stage-1 cache returned by load_cache(), so the
    realization blocklist is already applied. Writes
    ``stratified_artefact_share_<scenario>_<period>.csv`` to CACHE_DIR
    (fig_data) and returns the same frame. Draws no figure.
    """
    static = _load_regime_attrs()
    logger.info(f"static.csv: {len(static)} stations")

    slug_to_col = {slug: col for col, _lab, slug in ALL_INDICES}
    slug_to_lab = {slug: lab for col, lab, slug in ALL_INDICES}

    out = []
    for engine, df in loaded.items():
        sub = df[(df["scenario"].astype(str) == scenario)
                 & (df["period"].astype(str) == period)]
        logger.info(f"[{engine}] {len(df)} cache rows -> {len(sub)} at "
                    f"{scenario}/{period}")
        if sub.empty:
            logger.warning(f"[{engine}] no rows at {scenario}/{period} -> "
                           "skipped")
            continue

        for slug in REGIME_STRATA_SLUGS:
            col = slug_to_col.get(slug)
            if col is None or col not in sub.columns:
                logger.warning(f"[{engine}] index {slug!r} (column {col!r}) "
                               "absent from the cache -> skipped")
                continue
            et = ensemble_table(sub, col)
            if et.empty:
                logger.warning(f"[{engine}] {slug}: empty ensemble table")
                continue
            et["station"] = et["station"].astype(str)
            m = et.merge(static, on="station", how="inner")
            logger.info(f"[{engine}] {slug}: {len(et)} stations, {len(m)} "
                        "after the static-attribute join")
            if m.empty:
                logger.warning(f"[{engine}] {slug}: no station survived the "
                               "static-attribute join -> skipped")
                continue

            for row in _regime_strata_rows(m):
                row.update(engine=engine, index=slug,
                           index_label=slug_to_lab.get(slug, slug),
                           scenario=scenario, period=period)
                out.append(row)

    if not out:
        logger.error("regime_strata: no rows produced -- nothing written")
        return pd.DataFrame()

    res = pd.DataFrame(out)[[
        "index", "index_label", "engine", "group", "stratum", "n",
        "mean_full_pct", "mean_reduced_pct", "mean_bias_pp", "median_bias_pp",
        "mean_abs_full_pct", "share_signed_pct", "share_abs_pct", "denom_ok",
        "scenario", "period"]]
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_csv = em.tag(os.path.join(
        CACHE_DIR, f"stratified_artefact_share_{scenario}_{period}.csv"))
    res.to_csv(out_csv, index=False)
    logger.info(f"regime_strata -> {out_csv}  ({len(res)} rows)")

    with pd.option_context("display.width", 200, "display.max_rows", 500):
        for slug in REGIME_STRATA_SLUGS:
            blk = res[res["index"] == slug]
            if blk.empty:
                continue
            print(f"\n{'=' * 100}\n{slug_to_lab.get(slug, slug)}  [{slug}]  "
                  f"{scenario} {period}\n{'=' * 100}")
            for grp in ["global", "koppen", "snow", "aridity"]:
                g = blk[blk["group"] == grp]
                if g.empty:
                    continue
                print(f"\n-- {grp} --")
                print(g.pivot_table(index="stratum", columns="engine",
                                    values=["n", "mean_full_pct",
                                            "mean_reduced_pct", "mean_bias_pp",
                                            "median_bias_pp",
                                            "share_signed_pct"],
                                    sort=False).round(2).to_string())
    return res


def _draw_bias_ranking_legend(ax_box, fs_legend: float,
                              loc: str = "upper right") -> None:
    """HBV/LSTM/global-mean key, drawn INSIDE the box panel.

    Inside rather than pinned above the axes: on the combined Fig. 3 canvas an
    external key sits in the outer margin and is scaled down with it at
    typesetting, so it printed smaller than the panel it explains. Placed on the
    axes it scales with the data area instead, which is why it also carries a
    larger font than the tick labels here.

    `loc` is a fixed corner and NOT "best": "best" is chosen from the data, so
    it drifted between scenarios (top-right at ssp585, mid-right at ssp245,
    where it sat on the event-flashiness boxes). A figure pair that gets read
    side by side needs the key in the same place. It is exposed as an argument
    because the empty corner moves with the index set — the flood boxes now
    reach highest on the right — so a future index change can relocate it
    without editing this function. The semi-transparent frame keeps the labels
    readable if a whisker does reach under it.

    Split out of _draw_bias_ranking so the with- and without-share-panel paths
    share one definition.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax_box.legend(
        handles=[Patch(facecolor=ENGINE_HBV_C, alpha=0.75, label="HBV"),
                 Patch(facecolor=ENGINE_LSTM_C, alpha=0.75, label="LSTM"),
                 Line2D([0], [0], marker="D", linestyle="None",
                        markerfacecolor="white", markeredgecolor="black",
                        label="global mean")],
        loc=loc, ncol=3, fontsize=fs_legend, frameon=True,
        framealpha=0.85, edgecolor="none", borderpad=0.6,
        handletextpad=0.6, columnspacing=1.4)


def _draw_bias_ranking(ax_box, ax_bar, rows, indices_present,
                       fs_label: float = 12, fs_tick: float = 11,
                       fs_legend: float = 10,
                       ylabel_box: str = f"{_XHOTT} bias (reduced − full)  [pp]",
                       ylabel_bar: str = "Bias share of signal [%]",
                       share_stat: str = "median",
                       fs_rank_legend: float = None,
                       legend_loc: str = "upper right") -> int:
    """Draw the two bias-ranking panels onto caller-supplied axes.

      ax_box – per-station box-and-whisker of the too-hot bias
               (full − reduced, in percentage-points), one box per
               (index, engine) with a global-mean diamond.
      ax_bar – bars of  stat(bias) / stat(|full|)  expressed as a %, where
               `stat` is the median (default) or the mean. This is the *share
               of the full-ensemble signal* that the screening would remove.

    `share_stat` selects that statistic:

      "median" (default) – ratio of medians. A relative change is
          (future - baseline)/baseline, so a near-zero baseline makes the
          ratio unbounded and its MEAN is then set by a handful of
          catchments. The exposure differs by index -- 40.8 % of HBV
          catchments are effectively zero at Q05 -- so mean-based shares are
          not comparable across indices. Measured on the ssp585 2081-2100
          cache, mean(|full|)/median(|full|) is 1.2-1.7 for every high-flow
          index but 93 (Q50), 226 (annual min) and 3204 (Q05) in HBV, where
          the mean relative change reaches +153,774 %. The ratio of medians
          has no such failure mode. Note it is a ratio OF medians, not a
          median OF ratios; the caption must say which.
      "mean" – the original mean(bias)/mean(|full|). Kept so the earlier
          figure can be reproduced, but it is unsafe for the low-flow
          indices; see diagnostics/dump_bias_ranking_stats.py.

    HBV = orange, LSTM = blue. The two axes are expected to share an x-axis.
    The font sizes are arguments so the combined Fig. 3, drawn on a larger
    canvas, can match the spread panels below it without changing the
    standalone figure. `ylabel_box` / `ylabel_bar` are arguments for the same
    reason: a rotated y-label runs along its panel's HEIGHT, so the defaults
    (sized for the 7.0 in standalone canvas) overrun the shorter panels of the
    combined figure and the two labels then collide at the shared edge; Fig. 3
    passes short forms and carries the definitions in its caption instead.
    Returns the number of catchments drawn.
    """
    n = len(rows)
    # The key sits ON the data area now, so it is sized independently of the
    # spread panels' legends (which stay in their own corners at fs_legend).
    _fs_leg = fs_legend if fs_rank_legend is None else fs_rank_legend

    # ---- TOP: per-station bias boxplots ------------------------------
    # BIAS_SIGN flips the PLOTTED orientation only (reduced − full when −1);
    # the share bars below keep the raw full − reduced arrays so the share of
    # the full-ensemble signal stays a positive removal share.
    dx = 0.19
    box_arrays = []      # for y-limit
    for k, (slug, label, dh, dl, fh, fl, _, _) in enumerate(rows, start=1):
        for vals, xoff, color in ((dh, -dx, ENGINE_HBV_C),
                                  (dl, +dx, ENGINE_LSTM_C)):
            vals = BIAS_SIGN * vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            box_arrays.append(vals)
            bp = ax_box.boxplot(vals, positions=[k + xoff], widths=0.32,
                                showfliers=False, patch_artist=True)
            bp["boxes"][0].set(facecolor=color, alpha=0.75, linewidth=0.6)
            bp["medians"][0].set(color="black", linewidth=1.0)
            for w in bp["whiskers"] + bp["caps"]:
                w.set(linewidth=0.5)
            ax_box.scatter([k + xoff], [np.nanmean(vals)], marker="D",
                           s=22, facecolor="white", edgecolor="black",
                           linewidth=0.7, zorder=5)
    ax_box.axhline(0.0, color="0.5", lw=0.7, zorder=0)
    ax_box.set_xlim(0.5, n + 0.5)
    ylim = _whisker_lims(box_arrays)
    if ylim is not None:
        ax_box.set_ylim(*ylim)
    ax_box.set_ylabel(ylabel_box, fontsize=fs_label)
    ax_box.grid(axis="y", color="0.92", lw=0.5)
    ax_box.set_axisbelow(True)
    ax_box.tick_params(axis="y", labelsize=fs_tick)
    # Whichever panel is LOWEST carries the index names. With the share panel
    # present that is ax_bar, so hide them here: plt.subplots(sharex=True) hides
    # the upper panel's tick labels itself, but a manually shared axis (the
    # combined Fig. 3 gridspec) does not, so hide them for both call sites.
    xs = np.arange(1, n + 1, dtype=float)
    if ax_bar is None:
        ax_box.set_xticks(xs)
        ax_box.set_xticklabels([pretty_index(s) for s in indices_present],
                               rotation=40, ha="right", fontsize=fs_tick)
    else:
        ax_box.tick_params(axis="x", labelbottom=False)

    # ---- BOTTOM: bias as % of full-ensemble signal -------------------
    # ax_bar is None when the caller draws only the box panel (the main-text
    # fig3_combined); everything above still applies. The standalone
    # bias_ranking figure passes a real ax_bar and gets the share bars below.
    if ax_bar is None:
        _draw_bias_ranking_legend(ax_box, _fs_leg, loc=legend_loc)
        drawn = set()
        for _, _, _, _, _, _, sh, sl in rows:
            drawn.update(sh)
            drawn.update(sl)
        return len(drawn)

    if share_stat not in ("median", "mean"):
        raise ValueError(f"share_stat must be 'median' or 'mean', got "
                         f"{share_stat!r}")
    _agg = np.nanmedian if share_stat == "median" else np.nanmean

    def _frac(diff, full):
        d_stat = _agg(diff)
        f_stat_abs = _agg(np.abs(full))
        if not (np.isfinite(d_stat) and np.isfinite(f_stat_abs)
                and f_stat_abs > 0):
            return np.nan
        return 100.0 * d_stat / f_stat_abs

    h_pct, l_pct = [], []
    for slug, label, dh, dl, fh, fl, _, _ in rows:
        h_pct.append(_frac(dh, fh))
        l_pct.append(_frac(dl, fl))
    bar_w = 0.36
    ax_bar.bar(xs - bar_w / 2, h_pct, bar_w, color=ENGINE_HBV_C,
               alpha=0.85, label="HBV", edgecolor="0.2", linewidth=0.4)
    ax_bar.bar(xs + bar_w / 2, l_pct, bar_w, color=ENGINE_LSTM_C,
               alpha=0.85, label="LSTM", edgecolor="0.2", linewidth=0.4)
    ax_bar.axhline(0.0, color="0.5", lw=0.7, zorder=0)
    ax_bar.set_xlim(0.5, n + 0.5)
    # One short line, not two: rotated y-label text runs along the panel HEIGHT,
    # and the two-line "[mean(diff) / mean(|full|), %]" version was longer than
    # this (deliberately short) panel, so it overran into the box panel's label.
    # The mean(diff)/mean(|full|) definition belongs in the caption.
    ax_bar.set_ylabel(ylabel_bar, fontsize=fs_label)
    ax_bar.grid(axis="y", color="0.92", lw=0.5)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis="y", labelsize=fs_tick)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([pretty_index(s) for s in indices_present],
                            rotation=40, ha="right", fontsize=fs_tick)

    # ---- legends ------------------------------------------------------
    _draw_bias_ranking_legend(ax_box, _fs_leg, loc=legend_loc)

    drawn = set()
    for _, _, _, _, _, _, sh, sl in rows:
        drawn.update(sh)
        drawn.update(sl)
    return len(drawn)


def bias_ranking_per_period(df_hbv: pd.DataFrame, df_lstm: pd.DataFrame,
                            scenario: str, period: str,
                            share_stat: str = "median"):
    """Standalone cross-index bias-ranking figure for one (scenario, period):
    the two panels of `_draw_bias_ranking` on their own canvas.

    Output filename:
        hydro_index_bias_ranking_{scenario}_{period}_n{N}sta.png
    """
    rows, indices_present = _bias_ranking_rows(df_hbv, df_lstm,
                                               scenario, period)
    if not rows:
        logger.warning(f"bias_ranking_per_period({scenario},{period}): "
                        "no usable indices in either cache; skipping")
        return None

    n = len(rows)
    # Draw close to the printed size so the fonts below survive typesetting: a
    # Nature Water double-column figure is 180 mm (7.1 in), so a 13 in canvas is
    # scaled to ~0.55 and 12 pt type lands at ~6.6 pt. At ~0.46 in per index the
    # canvas stays near 10 in (n=18), a ~0.72 scale that keeps 12 pt above 8 pt.
    width_fig = max(7.5, 0.46 * n + 1.6)
    # Height 7.0, not 6.6: a rotated y-label runs along its panel's HEIGHT, and
    # both labels are centred on their own panel, so short panels let the two
    # meet at the shared edge and read as one run-on string. At 7.0 in the
    # measured clearance between them is 0.31 in; at 6.4 in it is 0.01 in.
    fig, (ax_box, ax_bar) = plt.subplots(
        2, 1, figsize=(width_fig, 7.0),
        gridspec_kw=dict(height_ratios=[2.05, 1.45]), sharex=True)

    n_sta = _draw_bias_ranking(ax_box, ax_bar, rows, indices_present,
                               share_stat=share_stat)

    # ---- panel letters ------------------------------------------------
    # Same bold top-left letters as before, now via the shared helper so every
    # figure in the paper is stamped from one place (identical style/position).
    add_panel_letters([ax_box, ax_bar])

    # Pin both y-labels to a common x so they read as one column.
    fig.align_ylabels([ax_box, ax_bar])

    # ---- save ---------------------------------------------------------
    out = os.path.join(SUMMARY_DIR,
                       f"hydro_index_bias_ranking_{scenario}_{period}"
                       f"_n{n_sta}sta.png")
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    fig.tight_layout()
    out = em.tag(out)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out} ({n} indices ranked, HBV vs LSTM, "
                f"{scenario} {period}, {n_sta} catchments)")
    return out


def _spread_model_level(df: pd.DataFrame, col: str,
                        scenario: str = None, period: str = None):
    """Per-(station, model) projection for `col` at (scenario, period)
    (members averaged within a model, so one value per model per catchment).
    Defaults to SPREAD_SCENARIO/SPREAD_PERIOD.
    Returns (full, reduced) long frames with columns [station, model, col]."""
    scenario = SPREAD_SCENARIO if scenario is None else scenario
    period = SPREAD_PERIOD if period is None else period
    sub = df[(df["scenario"].astype(str) == scenario)
             & (df["period"].astype(str) == period)].dropna(subset=[col])
    if sub.empty:
        return sub, sub
    ml = sub.groupby(["station", "model"], as_index=False)[col].mean()
    full = ml[ml["model"].isin(FULL_MODELS)]
    if em.is_weighting():
        # weighted mode: the second frame is the FULL model set carrying the
        # frozen weights in '_w'; _station_means/_station_band renormalise
        # per station via the sum ratio.
        reduced = full.copy()
        reduced["_w"] = reduced["model"].map(em.get_weights())
    else:
        reduced = ml[ml["model"].isin(REDUCED_MODELS)]
    return full, reduced


def _spread_density(ax, arrays, col, label, title, xlim=None,
                    fs_title: float = 10, fs_label: float = None):
    """Two filled densities (full behind, reduced in front) over a shared,
    robust x-range; dashed vertical lines at each ensemble global mean.

    `xlim` forces one range across engines. Panels (b) and (d) plot the SAME
    quantity for the two hydrological models, so per-panel autoscaling made
    one distribution look wider than the other purely through the axis
    (co-author review, 10 Jul). When None the range falls back to the
    per-engine full-ensemble percentile."""
    if xlim is not None:
        lo, hi = xlim
    else:
        clo, chi = SPREAD_CLIP_PCT
        ref = arrays[0][1]                   # full values set the x-range
        lo, hi = np.percentile(ref, [clo, chi])
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            lo, hi = float(np.min(ref)), float(np.max(ref) + 1e-6)
    bins = np.linspace(lo, hi, 80)
    for (lbl, vals, c, z) in arrays:
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        # histogram over the robust [lo, hi] range only: values outside are
        # dropped by np.histogram (no clip-and-pile-up at the last bin), so the
        # tails are not artificially spiked.
        h, edg = np.histogram(v, bins=bins, range=(lo, hi), density=True)
        x = 0.5 * (edg[:-1] + edg[1:])
        ax.fill_between(x, 0, h, color=c, alpha=0.55, zorder=z, label=lbl)
        ax.plot(x, h, color=c, lw=1.0, zorder=z)
        ax.axvline(float(np.mean(v)), color=c, ls="--", lw=1.0, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(label, **({} if fs_label is None else {"fontsize": fs_label}))
    ax.set_title(title, fontsize=fs_title)


def _spread_data(loaded: dict, col: str, scenario: str, period: str):
    """Per-engine (full, reduced) model-level frames for the spread panels.
    Returns ``(engines, data, n_sta)`` with `data` keyed by engine; `engines`
    is empty when no engine has usable rows. Split out of spread_projection so
    the standalone spread figure and the combined Fig. 3 share one data path."""
    engines = [e for e in ENGINES if e in loaded]
    data = {}
    n_sta = 0
    for e in engines:
        full, reduced = _spread_model_level(loaded[e], col, scenario, period)
        if full.empty or reduced.empty:
            logger.warning(f"[{e}] spread({col}): no data for "
                            f"{scenario}/{period}; skipping engine")
            continue
        data[e] = (full, reduced)
        n_sta = max(n_sta, full["station"].nunique())
    return [e for e in engines if e in data], data, n_sta


def _draw_spread_rows(axes, data: dict, engines, col: str, label: str,
                      scenario: str, period: str,
                      fs_ylabel: float = 11, fs_title: float = 10,
                      fs_legend: float = 10, fs_xlabel: float = None,
                      bands: bool = True):
    """Draw the per-engine spread panels onto a caller-supplied
    (len(engines), 2) axes grid: column 0 = the ensemble spread band across
    catchments sorted by median, column 1 = the across-catchment density of the
    per-catchment ensemble mean. Full is drawn behind, reduced in front, and
    the y-ranges (and the density x-range) are unified across engine rows so
    the two hydrological models are directly comparable.

    With ``bands=False`` the spread-band column is not drawn at all and each
    entry of `axes` needs ONE axes only, which then carries the density panel.
    The engine name moves onto that panel's y-label, because the band panel it
    used to sit on is gone (main-text Fig. 3, co-author review 9 Sep 2026).

    Font sizes are arguments so the combined Fig. 3 can match the ranking
    panels above it without changing the standalone figure."""
    blo, bhi = SPREAD_BAND_PCT
    band_lbl = f"{int(blo * 100)}–{int(bhi * 100)}%"

    def _smooth(y):
        """Centred rolling median over SPREAD_BAND_SMOOTH already-sorted points
        (legibility only; 0/oversized window -> returned unchanged)."""
        w = int(SPREAD_BAND_SMOOTH)
        if w < 2 or w >= len(y):
            return y
        return (pd.Series(y).rolling(w, center=True, min_periods=1)
                .median().to_numpy())

    # Per-station ensemble means, computed once per engine and reused by the
    # density panels below (they also set the common x-range).
    stat_means = {}
    for e in engines:
        full, reduced = data[e]
        stat_means[e] = (full.groupby("station")[col].mean().to_numpy(),
                         _station_means(reduced, col))

    # COMMON density x-range: the union of the per-engine robust ranges, so
    # panels (b) and (d) are on one axis and are visually comparable. Union
    # rather than intersection, so no engine loses data it previously showed.
    clo, chi = SPREAD_CLIP_PCT
    _los, _his = [], []
    for e in engines:
        ref = stat_means[e][0]               # full ensemble sets the range
        ref = ref[np.isfinite(ref)]
        if ref.size:
            _lo, _hi = np.percentile(ref, [clo, chi])
            if np.isfinite(_lo) and np.isfinite(_hi) and _lo < _hi:
                _los.append(_lo)
                _his.append(_hi)
    dens_xlim = (min(_los), max(_his)) if _los else None

    band_axes, band_ylims = [], []           # panels (a)/(c): shared y below
    dens_axes = []                           # panels (b)/(d): shared y below

    for r, e in enumerate(engines):
        full, reduced = data[e]
        gm_f = full.groupby("station")[col].mean().mean()
        gm_r = float(np.nanmean(_station_means(reduced, col)))
        logger.info(f"[{e} {col}] {scenario}/{period}: "
                    f"full={gm_f:.2f} reduced={gm_r:.2f} "
                    f"bias={gm_f - gm_r:.2f} pp (n={full['station'].nunique()})")

        # (a) — ensemble spread band across catchments sorted by median
        if bands:
            ax = axes[r][0]
            band_edges = []      # for a robust y-limit (a few catchments have
                                 # extreme p90 that would otherwise dwarf the bulk)
            for tbl, c, z, lbl in ((full, FULL_BOX_C, 1, f"full {band_lbl}"),
                                   (reduced, REDUCED_BOX_C, 2,
                                    f"{em.second_label().lower()} {band_lbl}")):
                a = _station_band(tbl, col, blo, bhi)
                x = np.linspace(0.0, 100.0, len(a))
                lo_s, hi_s = _smooth(a["lo"].to_numpy()), _smooth(a["hi"].to_numpy())
                ax.fill_between(x, lo_s, hi_s, color=c, alpha=0.40, zorder=z,
                                label=lbl, linewidth=0)
                ax.plot(x, _smooth(a["med"].to_numpy()), color=c, lw=1.4,
                        zorder=z + 0.1)
                band_edges.append(a["lo"].to_numpy())
                band_edges.append(a["hi"].to_numpy())
            ax.axhline(0.0, color="0.5", lw=0.7, zorder=0)
            # robust y-range: clip the 1st/99th pct of the band edges so the bulk
            # of the two bands is visible instead of a sliver under one outlier
            # spike. Collected, not applied here: panels (a) and (c) also plot the
            # same quantity per hydrological model, so the y-range is unified
            # after both rows are drawn (same review point as the density panels).
            allv = np.concatenate(band_edges)
            allv = allv[np.isfinite(allv)]
            if allv.size:
                ylo, yhi = np.percentile(allv, [1.0, 99.0])
                if np.isfinite(ylo) and np.isfinite(yhi) and ylo < yhi:
                    pad = 0.05 * (yhi - ylo)
                    band_ylims.append((ylo - pad, yhi + pad))
            band_axes.append(ax)
            ax.set_xlabel("Catchment percentile (sorted by median)",
                          **({} if fs_xlabel is None else {"fontsize": fs_xlabel}))
            ax.set_ylabel(f"{e.upper()}\n\n{label}", fontsize=fs_ylabel)
            # No "(a)"/"(b)" prefix here: this title is set once per ENGINE ROW,
            # so the old prefixes repeated (a,b on the HBV row and a,b again on
            # LSTM). add_panel_letters stamps the bold top-left letter across the
            # whole grid instead, giving the correct a,b / c,d over the four
            # panels.
            ax.set_title("Ensemble spread band across catchments",
                         fontsize=fs_title)

        # density panel — per-catchment ensemble mean. Right-hand column when
        # the band column is drawn, the only column when it is not.
        dens_ax = axes[r][1] if bands else axes[r][0]
        _spread_density(
            dens_ax,
            # "weighted (BMA on TCR)" named Massoud's scheme on EVERY
            # weighting run, so the gaussjoint kernel went out under the name
            # of the one scheme main4.tex contrasts it against. The variant is
            # already carried by the tag() suffix in the file name.
            [("full ensemble", stat_means[e][0], FULL_BOX_C, 1),
             (("weighted" if em.is_weighting()
               else f"reduced ({_XHOT} removed)"),
              stat_means[e][1], REDUCED_BOX_C, 2)],
            col, label, "Per-catchment ensemble-mean distribution",
            xlim=dens_xlim, fs_title=fs_title, fs_label=fs_xlabel)
        dens_axes.append(dens_ax)
        if not bands:
            # The engine name lived on the band panel's y-label, which is gone.
            # Same two-blank-line style, so the row identity still reads as a
            # row label rather than as part of the axis quantity.
            dens_ax.set_ylabel(f"{e.upper()}\n\nDensity", fontsize=fs_ylabel)

        if r == 0:
            if bands:
                axes[r][0].legend(loc="upper left", fontsize=fs_legend,
                                  framealpha=0.7, edgecolor="none")
            dens_ax.legend(loc="upper right", fontsize=fs_legend,
                           framealpha=0.7, edgecolor="none")

    # ---- unify the axes across engine rows -------------------------------
    # (a)/(c) share the y-range and (b)/(d) share both x and y, so a reader
    # can compare the two hydrological models directly instead of comparing
    # two differently-scaled pictures.
    if band_ylims:
        _lo = min(v[0] for v in band_ylims)
        _hi = max(v[1] for v in band_ylims)
        for _ax in band_axes:
            _ax.set_ylim(_lo, _hi)
    if dens_axes:
        _top = max(_ax.get_ylim()[1] for _ax in dens_axes)
        for _ax in dens_axes:
            _ax.set_ylim(0.0, _top)

    present = set()
    for e in engines:
        present |= set(data[e][0]["model"])
    n_full = sum(1 for m in FULL_MODELS if m in present)
    if em.is_weighting():
        n_red = em.n_eff_present(present & set(FULL_MODELS))
        logger.info(f"  spread({col}) {scenario}/{period}: models present "
                    f"full={n_full}, N_eff={n_red:.1f}")
    else:
        n_red = sum(1 for m in REDUCED_MODELS if m in present)
        logger.info(f"  spread({col}) {scenario}/{period}: models present "
                    f"full={n_full}, reduced={n_red}")
    return n_full, n_red


def spread_projection(loaded: dict, col: str, label: str):
    """Standalone full-vs-reduced ensemble-spread figure for one index at
    SPREAD_SCENARIO/SPREAD_PERIOD: `_draw_spread_rows` on its own canvas, two
    panels per engine row. Reads the per-engine caches already held in
    `loaded` (no recompute)."""
    engines, data, n_sta = _spread_data(loaded, col,
                                        SPREAD_SCENARIO, SPREAD_PERIOD)
    if not engines:
        logger.warning(f"spread_projection({col}): no usable engine; skipping")
        return None

    fig, axes = plt.subplots(len(engines), 2,
                             figsize=(11, 4.4 * len(engines)), squeeze=False)
    _draw_spread_rows(axes, data, engines, col, label,
                      SPREAD_SCENARIO, SPREAD_PERIOD)

    add_panel_letters(axes)
    slug = col.replace("_change_pct", "")
    out = os.path.join(
        SUMMARY_DIR,
        f"spread_{slug}_{SPREAD_SCENARIO}_{SPREAD_PERIOD}_n{n_sta}sta.png")
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = em.tag(out)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out}")
    return out


def fig3_combined(loaded: dict, scenario: str = None, period: str = None,
                  spread_index=None):
    """Main-text Fig. 3 on ONE canvas: the cross-index too-hot bias ranking
    above the peak-flow ensemble-spread anatomy.

    Replaces the two-PNG \\includegraphics stack in main4.tex, which showed two
    independent a),b) letter sequences and two different type scales. THREE
    panels (FIG3_SPREAD_BANDS = False, co-author review 9 Sep 2026):

        a) per-catchment too-hot bias across the streamflow indices (HBV, LSTM)
        b) HBV  per-catchment ensemble-mean distribution
        c) LSTM per-catchment ensemble-mean distribution

    The two ensemble-spread BAND panels that used to be (b) and (d) are
    dropped: the weighted band sits inside the full band by less than 10% of
    the band width, which a reader cannot resolve at print size. Set
    FIG3_SPREAD_BANDS = True to restore the five-panel layout (a; HBV band,
    HBV density; LSTM band, LSTM density). The band panels remain available in
    full in the standalone `spread` section figures.

    The bias-SHARE bar panel that the standalone `bias_ranking` figure carries
    is deliberately omitted here: the main text quotes the shares in prose and
    the extra panel duplicated information already in panel (a). The share
    panel exists only in the standalone `hydro_index_bias_ranking_*.png`
    figures (see `bias_ranking_per_period`).

    Both halves are drawn by the same helpers the standalone `bias_ranking`
    and `spread` sections use, so the three figures can never disagree. Needs
    BOTH engine caches in `loaded`. Returns the output path.
    """
    scenario = FIG3_SCENARIO if scenario is None else scenario
    period = FIG3_PERIOD if period is None else period
    col, label = FIG3_SPREAD_INDEX if spread_index is None else spread_index

    if "hbv" not in loaded or "lstm" not in loaded:
        logger.warning("fig3_combined: needs both hbv and lstm caches; have "
                       f"{sorted(loaded.keys())}; skipping")
        return None

    rows, indices_present = _bias_ranking_rows(loaded["hbv"], loaded["lstm"],
                                               scenario, period)
    if not rows:
        logger.warning(f"fig3_combined({scenario},{period}): no usable "
                        "indices for the ranking panels; skipping")
        return None
    engines, data, n_spread_sta = _spread_data(loaded, col, scenario, period)
    if not engines:
        logger.warning(f"fig3_combined({scenario},{period}): no usable engine "
                        f"for the {col} spread panels; skipping")
        return None

    # One canvas, two blocks. The blocks need very different internal spacing —
    # the ranking box needs room for its own rotated index labels, the spread
    # grid needs room for its own axis labels — so they get their own
    # subgridspecs instead of one uniform hspace.
    _bands = bool(FIG3_SPREAD_BANDS)
    fig = plt.figure(figsize=FIG3_FIGSIZE if _bands else FIG3_FIGSIZE_NOBANDS)
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=list(FIG3_HEIGHT_RATIOS if _bands
                           else FIG3_HEIGHT_RATIOS_NOBANDS),
        hspace=0.40 if _bands else FIG3_HSPACE_NOBANDS)
    # The ranking block is a single axes: Fig. 3 shows only the too-hot bias
    # box panel (a), not the bias-share bars. That box panel therefore takes
    # the whole ranking-block height and carries the rotated index labels.
    gs_rank = outer[0].subgridspec(1, 1)
    ax_box = fig.add_subplot(gs_rank[0])
    if _bands:
        gs_spread = outer[1].subgridspec(len(engines), 2,
                                         hspace=0.34, wspace=0.20)
        spread_axes = [[fig.add_subplot(gs_spread[r, c]) for c in range(2)]
                       for r in range(len(engines))]
    else:
        # Without the band column the two engines sit SIDE BY SIDE in one row,
        # so each density panel keeps the width it had in the two-column grid
        # and the block loses exactly one row of height. wspace is wider than
        # 0.20 because the right-hand panel now carries its own y-label, which
        # sits in the gap between the two panels.
        gs_spread = outer[1].subgridspec(1, len(engines), wspace=0.32)
        spread_axes = [[fig.add_subplot(gs_spread[0, r])]
                       for r in range(len(engines))]

    # Single type scale across all five panels. The two halves used to be
    # typeset at different \includegraphics scales, so 12 pt in one PNG and
    # 11 pt in the other landed at visibly different printed sizes; on one
    # canvas the same numbers mean the same printed size everywhere.
    fs_label, fs_tick, fs_legend, fs_title = 12, 11, 12, 11
    # Every key on this canvas is drawn INSIDE its own axes, so all five shrink
    # by the same factor at typesetting. They therefore carry ONE size (12 pt,
    # ~6.6 pt printed at the Nature Water double-column width, matching the axis
    # labels); the panel (a) key is no longer sized apart from the spread keys.
    fs_rank_legend = fs_legend

    # ax_bar=None -> _draw_bias_ranking draws only the box panel and puts the
    # index names on it. Short y-label: the standalone wording is longer than
    # this panel is tall; the quantity is defined in the main4.tex caption.
    n_rank_sta = _draw_bias_ranking(ax_box, None, rows, indices_present,
                                    fs_label=fs_label, fs_tick=fs_tick,
                                    fs_legend=fs_legend,
                                    ylabel_box=f"{_XHOTT} bias (pp)",
                                    fs_rank_legend=fs_rank_legend)
    # Title set HERE, not inside _draw_bias_ranking, so the standalone
    # hydro_index_bias_ranking_*.png figures stay untitled: on this canvas the
    # four spread panels carry titles, and an untitled (a) beside them read as
    # an omission. No index count in the wording -- the panel draws whatever
    # indices the cache holds, so a hard-coded number would go stale silently.
    ax_box.set_title(f"Per-catchment {_XHOT} bias by streamflow index",
                     fontsize=fs_title)
    _draw_spread_rows(spread_axes, data, engines, col, label,
                      scenario, period,
                      fs_ylabel=fs_label, fs_title=fs_title,
                      fs_legend=fs_legend, fs_xlabel=fs_label,
                      bands=_bands)
    for row in spread_axes:
        for ax in row:
            ax.tick_params(axis="both", labelsize=fs_tick)

    # Letters run in reading order over every panel in one call, so the
    # sequence is continuous across the two blocks — the defect that made the
    # earlier stacked-PNG version show a,b twice.
    add_panel_letters([ax_box] + [ax for row in spread_axes for ax in row])
    fig.align_ylabels([ax_box])

    n_sta = max(n_rank_sta, n_spread_sta)
    out = os.path.join(
        SUMMARY_DIR,
        f"fig3_biasrank_spread_{scenario}_{period}_n{n_sta}sta.png")
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    out = em.tag(out)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    _npanel = 1 + sum(len(row) for row in spread_axes)
    logger.info(f"Saved {out} ({len(rows)} indices ranked, {_npanel}-panel "
                f"[no share panel; spread bands {'on' if _bands else 'off'}]; "
                f"{col} spread for {engines}; "
                f"ranking n={n_rank_sta}, spread n={n_spread_sta} catchments)")
    return out


if __name__ == "__main__":
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment(logger)
    import argparse
    parser = argparse.ArgumentParser(
        prog="hydro_indices_change.py",
        description="Render the fig05 hydrological-index figures. With no "
                    "section named, every section is drawn.")
    parser.add_argument("sections", nargs="*", default=[], metavar="SECTION",
                        help="sections to draw (default: all). Choices: "
                             + ", ".join(SECTIONS) + ", all")
    parser.add_argument("-n", "--max-stations", type=int, default=None,
                        help="use only the first N stations (quick test)")
    # Bias-share statistic for the STANDALONE hydro_index_bias_ranking_*.png
    # figures (the `bias_ranking` section). Main-text Fig. 3 never shows the
    # share panel, so this flag does not touch it. The share is a ratio of
    # medians by default: a relative change is unbounded on a near-zero
    # baseline, and the exposure differs by index, so mean-based shares are
    # not comparable across indices (see _draw_bias_ranking).
    parser.add_argument("--share-stat", choices=("median", "mean"),
                        default="median",
                        help="standalone bias-ranking share statistic "
                             "(default: median); does not affect Fig. 3")
    # method=removal|weighting and weights=bma|kde are consumed before argparse
    _args = parser.parse_args(em.reject_cli_tokens(sys.argv[1:]))
    logger.info(f"ensemble method = {em.method()}"
                + (f" [{em.variant()}], suffix '{em.suffix()}'"
                   if em.is_weighting() else ""))
    MAX_STATIONS = _args.max_stations          # module global, read by compute_cache
    if not _args.sections or "all" in _args.sections:
        selected = set(SECTIONS)
    else:
        _bad = [s for s in _args.sections if s not in SECTIONS]
        if _bad:
            parser.error("unknown section(s) " + ", ".join(map(repr, _bad))
                         + "; choose from " + ", ".join(SECTIONS) + ", all")
        selected = set(_args.sections)

    # Per engine: build/load the cache ONCE (covers all scenarios & periods),
    # then draw whichever per-engine sections were selected. AFTER both engines
    # are loaded, draw the cross-engine sections (engine_disagreement,
    # bias_ranking, engine_boxplots, spread). Failures are logged and the
    # remaining selected work still runs.
    loaded = {}
    for _engine in ENGINES:
        logger.info(f"================ engine={_engine} ================")
        try:
            loaded[_engine] = load_cache(_engine)
        except Exception as exc:
            logger.exception(f"FAILED to build cache for engine={_engine}: {exc}")
            continue
        _df = loaded[_engine]
        if "maps" in selected:
            for _scn in MAP_SCENARIOS:
                for _period in PERIODS:
                    try:
                        plot_maps(_df, _engine, _scn, _period)
                    except Exception as exc:
                        logger.exception(
                            f"FAILED maps engine={_engine} scenario={_scn} "
                            f"period={_period}: {exc}")
        if "scenario_boxplots" in selected:
            try:
                boxplots_by_scenario(_df, _engine)
            except Exception as exc:
                logger.exception(f"FAILED boxplots engine={_engine}: {exc}")
        # Per-index 4×2 diff and full/reduced value figures (one PNG per index
        # per engine). Both walk DIFF_INDICES; each is gated independently.
        if "diff_maps" in selected or "value_maps" in selected:
            for _col, _label, _slug in DIFF_INDICES:
                if "diff_maps" in selected:
                    try:
                        diff_maps_per_index(_df, _engine, _col, _label, _slug)
                    except Exception as exc:
                        logger.exception(
                            f"FAILED diff_maps_per_index engine={_engine} "
                            f"index={_slug}: {exc}")
                # Sibling full + reduced 4×2 maps per (engine, index). Full and
                # reduced share a colour scale so they're visually comparable.
                if "value_maps" in selected:
                    for _kind in ("full", "reduced"):
                        try:
                            value_maps_per_index(_df, _engine, _col, _label,
                                                 _slug, _kind)
                        except Exception as exc:
                            logger.exception(
                                f"FAILED value_maps_per_index engine={_engine} "
                                f"index={_slug} kind={_kind}: {exc}")

    # Cross-engine disagreement maps: (LSTM full−reduced) − (HBV full−reduced).
    # One PNG per DIFF_INDICES entry; needs BOTH engines loaded.
    if "engine_disagreement" in selected:
        if "hbv" in loaded and "lstm" in loaded:
            logger.info("================ engine-disagreement (LSTM − HBV) "
                        "================")
            for _col, _label, _slug in DIFF_INDICES:
                try:
                    engine_disagreement_maps(loaded["lstm"], loaded["hbv"],
                                             _col, _label, _slug,
                                             engine_a="lstm", engine_b="hbv")
                except Exception as exc:
                    logger.exception(
                        f"FAILED engine_disagreement_maps index={_slug}: {exc}")
        else:
            logger.warning("engine-disagreement skipped — need both hbv and "
                           f"lstm caches; have {sorted(loaded.keys())}")

    # Cross-index bias-ranking summary, one figure per BIAS_RANKING_TARGETS
    # entry. Needs BOTH engines loaded (HBV vs LSTM appear side-by-side).
    if ("bias_ranking" in selected and "hbv" in loaded and "lstm" in loaded
            and BIAS_RANKING_TARGETS):
        logger.info("================ bias-ranking summary ================")
        for _scn, _per in BIAS_RANKING_TARGETS:
            try:
                bias_ranking_per_period(loaded["hbv"], loaded["lstm"],
                                         _scn, _per,
                                         share_stat=_args.share_stat)
            except Exception as exc:
                logger.exception(
                    f"FAILED bias_ranking_per_period scenario={_scn} "
                    f"period={_per}: {exc}")

    # Combined full-vs-reduced boxplot, BOTH engines paired, one figure per
    # focus scenario in BOX_ENGINES_SCENARIOS. Needs both engine caches.
    if ("engine_boxplots" in selected
            and "hbv" in loaded and "lstm" in loaded):
        logger.info("================ full-vs-reduced engine boxplot "
                    "================")
        for _scn in BOX_ENGINES_SCENARIOS:
            try:
                boxplots_full_reduced_by_engine(loaded["hbv"], loaded["lstm"],
                                                _scn)
            except Exception as exc:
                logger.exception(
                    "FAILED boxplots_full_reduced_by_engine "
                    f"scenario={_scn}: {exc}")

    # Spread-of-projections figures (full vs reduced ensemble spread), one PNG
    # per index in SPREAD_INDICES, drawn three ways per engine. Uses the caches
    # already in `loaded`; no recompute.
    if "spread" in selected and loaded:
        logger.info("================ spread-of-projections ================")
        for _col, _label in SPREAD_INDICES:
            try:
                spread_projection(loaded, _col, _label)
            except Exception as exc:
                logger.exception(
                    f"FAILED spread_projection index={_col}: {exc}")

    # Main-text Fig. 3: the ranking box and the spread panels on ONE canvas
    # (five panels, a-e). Needs both engine caches; drawn last so it reuses
    # everything above.
    if "fig3_combined" in selected and loaded:
        logger.info("================ Fig. 3 combined ================")
        try:
            fig3_combined(loaded)
        except Exception as exc:
            logger.exception(f"FAILED fig3_combined: {exc}")

    # Regime-stratified artefact share -> CSV in fig_data, no figure. Needs both
    # engine caches: the whole point of the table is that HBV and LSTM disagree
    # in the snow-affected strata, so a one-engine run would be misleading.
    if "regime_strata" in selected:
        if "hbv" in loaded and "lstm" in loaded:
            logger.info("================ regime-stratified artefact share "
                        "================")
            for _scn, _per in REGIME_STRATA_TARGETS:
                try:
                    regime_stratified_bias_share(loaded, _scn, _per)
                except Exception as exc:
                    logger.exception(
                        f"FAILED regime_stratified_bias_share scenario={_scn} "
                        f"period={_per}: {exc}")
        else:
            logger.warning("regime_strata skipped — need both hbv and lstm "
                           f"caches; have {sorted(loaded.keys())}")
