"""
analyze_full_vs_reduced_significance.py
=======================================

Quantify whether the FULL - REDUCED (too-hot-removed) ensemble difference in
precipitation / temperature change is a real, systematic signal or is buried
in model + internal-variability noise. Runs the SAME three tests below for the
MEAN change and for every higher-percentile change present in the cache, so the
too-hot bias can be tracked from the mean out into the tails (peak/low flow).

Uses ONLY the per-realization cache produced by fig02_precip_change.py:

    paper/figures/fig_data/fig02_change.parquet
    (cols: station, lon, lat, scenario, model, member, period,
           precip_change_pct,  temp_change_degC,            # the MEAN change
           precip_q{90,95,99}_change_pct,                   # precip percentiles
           temp_q{5,50,90,95,99}_change_degC)               # temp   percentiles
    The percentile columns are AUTO-DETECTED, so whatever fig02 wrote is used.

For every (scenario, period) it reports three things:

  Q2  VARIANCE PARTITION  -- of the spread across realizations, how much is
      model uncertainty (between models) vs internal climate variability
      (members within a model)?  Answers "mainly model or realization choice".

  Q1  SIGNIFICANCE        -- per catchment, is (hot_mean - retained_mean)
      distinguishable from zero given the inter-model spread (Welch t-test)?
      Because diff = (h/N)*(hot_mean - retained_mean), this is exactly the
      test of "full != reduced".  We report the fraction of catchments that
      pass at p<0.05 with a consistent sign.

  Q3  SIGNAL-TO-NOISE     -- |full - reduced| / inter-model_std, i.e. the
      (h/N)-scaled ensemble-mean shift from screening (the plotted "bias")
      relative to the model spread. Shows whether the spread "masks" that shift
      at the single-model level (S/N<1) even when the mean difference is
      significant. (NOT the raw hot-vs-retained contrast, which is ~N/h larger.)

-------------------------------------------------------------------------------
MEAN vs PERCENTILES
-------------------------------------------------------------------------------
`precip_change_pct` / `temp_change_degC` are the change in the *MEAN* (~ mean
annual runoff forcing). The `*_q{N}_*` columns are the change at the Nth
percentile of the DAILY distribution: high precip percentiles drive PEAK FLOW,
low ones LOW FLOW. The tails are noisier and need more models, so expect the
significance to weaken and the spread (sigma) to grow as you move toward the
tails -- read that straight off the per-column rows of the summary table.

The FULL vs TOO_HOT split is NOT hardcoded here: it is derived exactly the way
fig02_precip_change.py derives it -- FULL_MODELS from models_in_results() and
EXCLUDED_MODELS = TCR > 2.2 K via METRICS (fig01_model_screening.py) -- so this
test always partitions the ensemble the same way as the figure it analyses.

Two datasets are supported, auto-selected from the CLI arg:
  * CLIMATE    (figures/fig_data/fig02_change.parquet, from fig02_precip_change.py)
               precip / temp mean + percentile change columns.
  * STREAMFLOW (figures/fig_data/stats_change.parquet, from build_streamflow_cache.py)
               per-realization flow-change metrics (delta_rel_*). The two
               hydrological engines (hbv, lstm) are analysed SEPARATELY, each
               getting its own '_<engine>' figures / CSVs (engines are never
               pooled, else HBV-vs-LSTM differences would contaminate the
               internal-vs-model variance partition).

Run:
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py          # precip: mean + all percentiles
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py temp     # temp:   mean + all percentiles
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py all      # both climate variables
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py flow     # streamflow, per engine (hbv & lstm)
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py delta_rel_q95          # one flow column
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py precip_q99_change_pct  # one climate column
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py transmission           # T->Q transmission vs catchment character (reads cached focus CSVs; run climate 'all' + 'flow' first)
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py snr_cascade            # Beat-2 T->P->Q S/N cascade figure (reads cached focus CSVs)
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py tcr_sensitivity        # TCR-threshold sweep robustness figure (reads the parquet caches)
    python paper/figures/scripts/analyze_full_vs_reduced_significance.py beats                  # both of the above in one go
"""

import os
import re
import sys
import glob
import math
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:                       # normal-approx fallback
    HAVE_SCIPY = False

# This file now lives in paper/figures/scripts/, the SAME dir as
# fig01_model_screening.py (imported below). config.py and the shared libs are at
# the repo root, three dirs up; add that plus this dir to sys.path.
_THIS = os.path.dirname(os.path.abspath(__file__))           # paper/figures/scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))  # repo root (config)
sys.path.insert(0, _THIS)                                    # fig01_model_screening.py is here
from fig01_model_screening import METRICS, models_in_results
from config import CONFIG
# same realization exclusion list fig02/fig05/build_streamflow use
from realization_blocklist import apply_blocklist
# registry of EVERY change metric in stats.parquet (29 streamflow + 12 precip
# indices) -- the single source of truth shared with hydro_indices_change.py and
# build_streamflow_cache.py, so no metric is silently left out of the analysis
from flow_metrics import (FLOW_METRICS, PRECIP_INDEX_METRICS, BY_COLUMN,
                          present as metrics_present,
                          missing as metrics_missing)
# Nature-style bold top-left panel letters (multi-panel figures only)
from figstyle import (add_panel_letters, mfmt, plabel, PANEL_LETTER_FONTSIZE,
                      PANEL_LETTER_X, PANEL_LETTER_Y, percentile_label)
# method switch: removal (default; behaviour unchanged) vs weighting — the
# per-station test becomes a linear-contrast test of the weighted-vs-equal
# ensemble mean and every derived output gains the _wt/_wtkde tag.
import ensemble_method as em

# Screen-aware label text: "too-hot" under SCREEN="hot", but the excluded set
# is not too-hot under SCREEN="both", so every figure/label string uses these
# instead of hardcoding the word.
_XHOT  = em.excluded_label()
_XHOTT = em.excluded_label(title=True)

HERE  = os.path.dirname(os.path.abspath(__file__))
# Stage-1 caches AND all DATA outputs (the per-catchment signif_*.csv files)
# live under figures/fig_data/. CLIMATE = precip/temp change from
# fig02_precip_change.py; FLOW = streamflow change from build_streamflow_cache.py.
CACHE_DIR     = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")  # per-method cache on /ibex (must match fig02/build_streamflow_cache)
CLIMATE_CACHE = os.path.join(CACHE_DIR, "fig02_change.parquet")
FLOW_CACHE    = os.path.join(CACHE_DIR, "stats_change.parquet")
# FIGURE outputs (PNG maps + the summary plots) go here; the per-catchment CSV
# DATA files go to CACHE_DIR (fig_data), NOT here — keep significance/ csv-free.
SIG_DIR = os.path.join(os.path.dirname(HERE), "significance")  # paper/figures/significance
# The per-catchment focus MAPS (Robinson scatter PNGs from export_focus) go in a
# maps/ subfolder so the summary plots (unc_decomp / signal_vs_noise / spectrum_*)
# at the top level of significance/ are easy to find.
MAPS_DIR = os.path.join(SIG_DIR, "maps")  # paper/figures/significance/maps

# ---- T -> Q transmission analysis (catchment-character attribution) --------
# Per-catchment attributes for relating the screening signal to catchment
# character. static.csv lives beside the forcing dataframes (same file
# apply_deltas_stn.py reads for lat/area). Keyed by 'stations'; public networks
# (e.g. Arcticnet__49052) match the signif_*.csv 'station' key directly.
STATIC_CSV = os.path.join(CONFIG.gscad_path, "dataframes", "static.csv")
TRANSMISSION_ATTRS = {
    "aridity":  "aridity_index_mswx_mswepv3",     # PET/P: HIGHER = drier / water-limited
    "snowfrac": "WorldClim_V2_snowfall_fraction_P.mat",  # snow fraction of precip (0-1)
    "kgz":      "KGZ_major",                       # Koeppen-Geiger major zone (A-E)
}

# Streamflow metric set: EVERY change metric stats.parquet records, taken from
# the flow_metrics registry (29 columns), ordered drought/low-flow -> central ->
# high-flow/flood -> FDC shape -> flashiness -> timing. That order is the x-axis
# of the significance / signal-to-noise spectrum. The label "Mean" (qmean) is
# reused by the uncertainty-decomposition figure. Only columns actually present
# in stats_change.parquet are analysed (see resolve_flow_metrics), so a cache
# built before a metric existed degrades to a warning, not a crash.
FLOW_METRIC_DEFS   = [(m.column, m.label) for m in FLOW_METRICS]
# The 12 ETCCDI-style PRECIPITATION indices stats.parquet also records. They
# describe the bias-corrected FORCING, not the flow, so they are a SEPARATE pass
# ('pindex') and never share an axis with a streamflow metric.
PINDEX_METRIC_DEFS = [(m.column, m.label) for m in PRECIP_INDEX_METRICS]
FLOW_ORDER    = {lab: i for i, (_, lab)
                 in enumerate(FLOW_METRIC_DEFS + PINDEX_METRIC_DEFS)}
FLOW_KEYWORDS   = {"flow", "streamflow", "q", "qsim", "hydro", "runoff"}
PINDEX_KEYWORDS = {"pindex", "precip_index", "precip_indices", "etccdi"}
# Metrics whose units are NOT the common "%" of the relative changes (only the
# half-flow date, whose shift is in days -- a relative change of a date is
# meaningless). Significance and S/N are scale-invariant, so these metrics stay
# in the tables, the focus maps and the spectrum figure; they are dropped ONLY
# from the signal_vs_noise figure, whose bars are absolute sigma and would
# otherwise put days and per-cent on one axis.
MIXED_UNIT_LABELS = {m.label for m in FLOW_METRICS + PRECIP_INDEX_METRICS
                     if m.units != "%"}

# Tick-label, legend, panel-title and y-label point size of the signal_vs_noise
# panels. The figures go into the supplement at ~0.9 linewidth, where the
# previous 8/7 pt was unreadable in print.
SN_TICK_FONTSIZE   = 13
SN_LEGEND_FONTSIZE = 13
SN_YLABEL_FONTSIZE = 13
# Panel-title point size, by variable. The title reads
# "SSP585  (full=32, reduced=17)" — ~29 characters — and has to fit inside ONE
# panel width, so it is tied to SN_PANEL_SIZE below: the wide streamflow and
# precipitation-index panels take the larger size, the narrow climate panels a
# smaller one.
SN_TITLE_FONTSIZE_DEFAULT = 13
SN_TITLE_FONTSIZE = {"precip": 11, "temp": 11}
# Per-panel canvas size (inches) of the signal_vs_noise figures, by variable.
# The climate panels carry 4 (precip) to 6 (temp) metric labels, the streamflow
# and precipitation-index panels 15; at one common width the climate panels were
# mostly white space and their text read small. Shrinking only the climate
# panels enlarges their text relative to the canvas without crowding the dense
# flow panels, which keep the default.
SN_PANEL_SIZE_DEFAULT = (3.8, 3.3)
# 2.8 x 2.6 was a shade too tight once the fonts went up — the tick and title
# text touched — so the climate panels get a small part of that back.
SN_PANEL_SIZE = {"precip": (3.1, 2.9), "temp": (3.1, 2.9)}
# Legend point size shared by the significance / bias-to-spread spectrum
# figures: spectrum_forcing*.png (make_summary_plots) and
# spectrum_hydrindices*.png (make_combined_flow_spectrum, whose single legend
# now serves all four panels). Both carry the same per-scenario model-count
# entries in a ~5.7-inch-wide panel, so one size fits both.
SPECTRUM_LEGEND_FONTSIZE = 10
# Tick-label, axis-label and panel-title point size of the spectrum_forcing*
# figures. The panels are ~5.7 inches wide and carry 6 (climate) to 12
# (precipitation-index) x labels, so the matplotlib defaults of 10/10/12 left
# the text small for the supplement.
SPECTRUM_TICK_FONTSIZE  = 11
SPECTRUM_LABEL_FONTSIZE = 12
SPECTRUM_TITLE_FONTSIZE = 13
# spectrum_forcing panel titles restate the y-axis label — "Precip_index:
# significance vs metric" sits above "Precip_index / % catchments significant" —
# so they are dropped for the passes named here, keyed by the filename suffix.
# The bold panel letters are unaffected: add_panel_letters places them in axes
# coordinates, not relative to the title. Add "" to this set to drop the titles
# from the climate spectrum_forcing figure as well.
SPECTRUM_NO_PANEL_TITLE = {"_pindex"}

# The 15 streamflow indices reported in Supplementary Table S6
# (paper/tables/streamflow_indices_table.tex). The summary FIGURES that carry a
# metric axis -- the combined HBV/LSTM spectrum and the per-engine
# signal_vs_noise panels -- are restricted to this set so a reader matching a
# figure against the table finds the same indices in both. The wider
# flow_metrics registry (29 columns) is still computed and still reaches the
# printed tables, the focus CSVs and the maps; only these two figures subset it.
# Labels must match the Metric(...) `label` field in flow_metrics.py exactly.
TABLE_S6_LABELS = frozenset({
    "Mean", "Q50", "Q/P",                    # average conditions
    "Q95", "Q99", "AnnMax", "RL10", "RL20", "RL50",   # high flows and floods
    "Q05", "AnnMin", "7Q10",                 # low flows and drought
    "SD", "FDCslope",                        # distribution shape
    "FlashRB",                               # hydrograph flashiness
})
# The complement within the 29-column streamflow registry (14 labels: the extra
# low-flow, high-flow-event, FDC-ratio, flashiness and timing indices). Derived,
# never hardcoded, so adding a Metric to flow_metrics.py cannot leave an index
# out of BOTH spectrum figures. Drawn as the companion 'other' spectrum from the
# same summary frames; significance and bias-to-spread are scale-invariant, so
# the days-unit HFD is legitimate on that shared axis.
OFF_TABLE_S6_LABELS = frozenset(lab for (_, lab) in FLOW_METRIC_DEFS
                                if lab not in TABLE_S6_LABELS)

# Per-column CAP (in %, the rescaled delta_rel_* units) on the symmetric focus-map
# colour limit. The default scale is the 98th pctile of |full-reduced|, but a few
# RELATIVE flow metrics have a heavy tail — at catchments whose baseline low flow
# is ~0 the relative change explodes (|Δ| reaches 10^2–10^8 %), so the p98 scale
# is dragged out to ±125…±7400 and the whole map collapses to one flat colour.
# For these columns we cap the limit near the BULK of the distribution (~p95 of
# the well-behaved values) so the spatial gradient is visible; the handful of
# blow-up catchments saturate to the end colour (shown as the colourbar arrows).
# Applied as min(p98, cap): it only TIGHTENS the scale, so engines/columns whose
# real range is already below the cap (e.g. flashiness_event_frt for HBV,
# mean_annual_min / q05 for LSTM) are left unchanged. Keyed by the bare column
# (no engine suffix); add/remove freely.
COLORBAR_LIM_CAP = {
    "delta_rel_flashiness_event_frt":  20.0,   # bulk |Δ| p95≈10 %  (was ±125 for LSTM)
    "delta_rel_flashiness_event":      50.0,   # bulk |Δ| p90≈25–45 % (was ±115/205)
    "delta_rel_mean_annual_min":       50.0,   # low-flow rel-Δ bulk p90≈30 % (was ±2000 HBV)
    "delta_rel_q05":                   40.0,   # low-flow rel-Δ bulk p90≈17 % (was ±7400 HBV)
}
# Fallback cap for every OTHER relative metric. The low-flow family added by the
# registry (7Q10, MAM7, Q7min, low-flow deficit/frequency) and the FDC ratios
# share the same near-zero-baseline blow-up as the four columns above, so an
# uncapped p98 would again collapse the map to one flat colour. ±100 % keeps the
# scale on the bulk; the colourbar arrows mark the catchments that clip. Applied
# only where no explicit cap exists, and only as min(p98, cap), so it never
# LOOSENS a scale that is already tighter. A plotting choice: it changes no
# statistic (the CSVs carry the uncapped values).
DEFAULT_REL_LIM_CAP = 100.0

# ---- model universe & the active screen (single source of truth) -----------
# Both come from ensemble_method.py, so this significance test splits the
# ensemble identically to the figures it analyses. Edit SCREEN / METHOD /
# VARIANT at the top of that file to change the treatment; there is no
# command-line switch.
#   * FULL_MODELS is auto-derived from one station's mean_deltas.nc (the same
#     model universe Fig. 1 / Fig. 2 use).
#   * EXCLUDED_MODELS are the models the active screen drops: metric above the
#     upper bound under SCREEN='hot' (TCR > 2.2 K by default), plus metric below
#     the lower bound under SCREEN='both' (TCR < 1.4 K by default).
#   * A FULL_MODELS entry missing from METRICS fails loudly inside
#     em.full_models(), so it gets fixed in fig01 rather than mislabelled.
FULL_MODELS_STATION   = em.FULL_MODELS_STATION
TOO_HOT_TCR_THRESHOLD = em.TCR_UPPER       # K; ALWAYS the TCR bound (imported
                                           # by local_vs_aggregate_null.py)
FULL_MODELS     = list(em.full_models())
EXCLUDED_MODELS = em.excluded_models()
RETAINED_MODELS = em.retained_models()
_tcr            = em.tcr                   # TCR (K) of one model, from METRICS

# The screening metric actually in force (em.METRIC). Under the default
# METRIC='tcr' these are exactly the TCR names above, so the sweep is unchanged;
# under METRIC='ecs' they carry ECS and its AR6 bounds. Anything that
# reimplements the screen (the threshold sweep) must read THESE, not _tcr.
_metric_value   = em.metric_value          # value of the ACTIVE metric
SCREEN_LABEL    = em.metric_label()        # 'TCR' | 'ECS'
SCREEN_CUT      = em.upper_bound()         # the chosen cut of the active metric
SCREEN_LO_BOUND = em.lower_bound()         # two-sided lower bound, active metric

# CLI arg selects the dataset AND which column(s) to analyse:
#   (none) / "precip"  -> CLIMATE: precip mean + every precip_q{N}_change_pct
#   "temp"             -> CLIMATE: temp   mean + every temp_q{N}_change_degC
#   "all"              -> CLIMATE: both variables
#   "flow"/"streamflow"-> STREAMFLOW: the FLOW_METRIC_DEFS set, per engine
#   "<column name>"    -> just that one column (precip_/temp_ = climate,
#                         delta_ = streamflow)
# method=removal|weighting / weights=bma|kde are consumed from argv BEFORE the
# positional ARG is read (ARG is a module-level read, mirrored by the replot
# drivers, so the consumption must happen here rather than in __main__).
sys.argv = sys.argv[:1] + em.reject_cli_tokens(sys.argv[1:])
ARG        = sys.argv[1] if len(sys.argv) > 1 else "precip"
FOCUS_SCEN = "ssp585"          # detailed per-station export + map use this
FOCUS_PER  = "2081-2100"
ALPHA      = 0.05

# SIGN CONVENTION (2 Aug 2026, matching gev_fdr_dipole.py / fig02). The
# manuscript reports every ensemble difference as reduced − full. The STORED
# outputs of this script (signif_*.csv focus files, summary CSVs, the
# tcr_sensitivity CSV) deliberately KEEP the full_minus_reduced column and
# orientation — downstream consumers read that column by name and every
# statistic here (two-sided Welch p, |bias|/sigma S/N, sign consistency) is
# invariant under the flip. BIAS_SIGN is applied only where a SIGNED bias is
# drawn (the tcr_sensitivity inflation panel).
BIAS_SIGN = -1.0
SIGNCONV  = "full − reduced" if BIAS_SIGN > 0 else "reduced − full"
# For EACH analysed column, always write the per-catchment focus CSV at
# FOCUS_SCEN/FOCUS_PER; also write the stippled map when MAKE_FOCUS_MAPS is True.
MAKE_FOCUS_MAPS = True
# Render the summary figures (uncertainty split, signal-vs-noise, significance/
# S-N spectrum) from the printed table — see make_summary_plots().
MAKE_SUMMARY_PLOTS = True

# Parallelism: the ~100 (column, scenario, period) "cells" are independent, so a
# fork Pool runs them concurrently. Workers INHERIT the loaded DataFrame
# copy-on-write (no pickling of the big frame) — same pattern as fig02/04/05.
# Worker count is derived from the SLURM allocation (SLURM_CPUS_PER_TASK, set by
# --cpus-per-task) so it never drifts from the job script; off SLURM it falls
# back to os.cpu_count()-1. Each worker is pinned to 1 math thread by the env.
NUM_WORKERS = (int(os.environ["SLURM_CPUS_PER_TASK"])
               if os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit()
               else max(1, (os.cpu_count() or 1) - 1))
# Set by analyze_pass before forking the Pool; workers read their slice from it.
_WORK_DF = None


# ---------------------------------------------------------------------------
#  small stats helpers
# ---------------------------------------------------------------------------
def _p_from_t(t, dof):
    if not np.isfinite(t) or not np.isfinite(dof) or dof <= 0:
        return np.nan
    if HAVE_SCIPY:
        return float(2.0 * sps.t.sf(abs(t), dof))
    # normal approximation (good enough for dof >~ 20; conservative otherwise)
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))


def welch(a, b):
    """Welch t-test of mean(a) vs mean(b). Returns (diff, se, t, dof, p)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    diff = a.mean() - b.mean()
    if se == 0:
        return (diff, 0.0, np.nan, np.nan, np.nan)
    t = diff / se
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return (diff, se, t, dof, _p_from_t(t, dof))


# ---------------------------------------------------------------------------
#  Q2 - variance partition (per station, then summarised)
# ---------------------------------------------------------------------------
def variance_partition(df_sp, col):
    """For one (scenario, period) slice, return a DataFrame with per-station
    internal vs model variance components of `col`."""
    rows = []
    for stn, sub in df_sp.groupby("station"):
        sub = sub.dropna(subset=[col])                    # percentile cols can be NaN
        if sub.empty:
            continue
        g_mean  = sub.groupby("model")[col].mean()        # model means
        g_count = sub.groupby("model")[col].count()
        if len(g_mean) < 2:
            continue
        # internal variance = pooled within-model variance (multi-member only)
        multi = g_count[g_count >= 2].index
        if len(multi):
            ss = df = 0.0
            for m in multi:
                v = sub.loc[sub["model"] == m, col].to_numpy(float)
                v = v[np.isfinite(v)]
                ss += (len(v) - 1) * v.var(ddof=1)
                df += (len(v) - 1)
            s2_internal = ss / df if df > 0 else np.nan
        else:
            s2_internal = np.nan
        # between-model variance of the model means, corrected for internal
        # noise that inflates each mean: Var(xbar_m) = s2_model + s2_internal/M_m
        s2_between = float(g_mean.var(ddof=1))
        if np.isfinite(s2_internal):
            corr = s2_internal * float(np.mean(1.0 / g_count.to_numpy(float)))
            s2_model = max(0.0, s2_between - corr)
        else:
            s2_model = s2_between
        rows.append({
            "station": stn, "n_models": int(len(g_mean)),
            "n_members": int(g_count.sum()),
            "s2_internal": s2_internal, "s2_model": s2_model,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Q1 + Q3 - per-station hot-vs-retained test and signal-to-noise
# ---------------------------------------------------------------------------
def per_station_test_weighted(df_sp, col):
    """Weighted counterpart of per_station_test. The tested 'bias' is the
    weighted-minus-equal ensemble-mean difference, a linear CONTRAST in the
    per-model means: Delta = sum_m (w~_m - 1/N) xbar_m, with w~ the frozen
    weights renormalised over the N models present. Under H0 (exchangeable
    model means, variance sigma^2) E[Delta]=0 and Var(Delta) = sigma^2 *
    sum(c_m^2), giving a t statistic with dof N-1 (homoscedastic analogue of
    the removal branch's Welch test; the aggregate weight-permutation null
    lives in bootstrap_random_removal.py's weighting mode).

    The stored full_minus_reduced column keeps the removal file orientation
    (full - weighted) so every downstream consumer works unchanged; the file
    additionally carries full_mean / weighted_mean / n_eff.
    """
    W = em.get_weights()
    rows = []
    for stn, sub in df_sp.groupby("station"):
        sub = sub.dropna(subset=[col])
        if sub.empty:
            continue
        mm = sub.groupby("model")[col].mean()      # one value per model
        N = len(mm)
        if N < 4:
            continue
        w = np.array([W.get(m, np.nan) for m in mm.index], float)
        if not np.all(np.isfinite(w)):
            raise KeyError(f"model(s) without weight at {stn}: "
                           f"{[m for m, wi in zip(mm.index, w) if not np.isfinite(wi)]}")
        w = w / w.sum()
        c = w - 1.0 / N
        x = mm.to_numpy(float)
        full_mean = float(x.mean())
        weighted_mean = float(w @ x)
        fmr = full_mean - weighted_mean            # full − weighted
        s = float(mm.var(ddof=1)) ** 0.5
        se = s * math.sqrt(float((c * c).sum()))
        t = (fmr / se) if se > 0 else np.nan
        p = _p_from_t(t, N - 1) if np.isfinite(t) else np.nan
        sn = abs(fmr) / s if s > 0 else np.nan
        rows.append({
            "station": stn,
            "lon": float(sub["lon"].iloc[0]), "lat": float(sub["lat"].iloc[0]),
            "n_models": N, "n_eff": float(1.0 / (w * w).sum()),
            "full_mean": full_mean, "weighted_mean": weighted_mean,
            "full_minus_reduced": fmr,
            "se": se, "t": t, "p": p,
            "inter_model_std": s, "signal_to_noise": sn,
            "significant": bool(np.isfinite(p) and p < ALPHA),
        })
    return pd.DataFrame(rows)


def per_station_test(df_sp, col):
    if em.is_weighting():
        return per_station_test_weighted(df_sp, col)
    rows = []
    for stn, sub in df_sp.groupby("station"):
        sub = sub.dropna(subset=[col])                     # percentile cols can be NaN
        if sub.empty:
            continue
        mm = sub.groupby("model")[col].mean()              # one value per model
        present = set(mm.index)
        H = mm[[m for m in EXCLUDED_MODELS  if m in present]].to_numpy(float)
        R = mm[[m for m in RETAINED_MODELS if m in present]].to_numpy(float)
        n_all = len(present)
        if len(H) < 2 or len(R) < 2:
            continue
        diff_groups, se, t, dof, p = welch(H, R)
        frac = len(H) / n_all                              # h/N
        full_minus_reduced = frac * diff_groups
        inter_model_std = float(mm.var(ddof=1)) ** 0.5      # spread of ALL models
        # S/N convention: |full - reduced| / inter-model sigma — i.e. the
        # screening-induced shift in the ENSEMBLE MEAN (the quantity the maps,
        # boxplots and the paper's "bias" use) relative to the model spread. This
        # is the (h/N)-scaled full_minus_reduced, NOT the raw hot-vs-retained
        # group contrast, so S/N stays consistent with the plotted bias
        # everywhere (temperature median ~0.64, not 1.37). The Welch p-value
        # (significance) is scale-invariant, so it is unaffected by this choice.
        sn = abs(full_minus_reduced) / inter_model_std if inter_model_std > 0 else np.nan
        rows.append({
            "station": stn,
            "lon": float(sub["lon"].iloc[0]), "lat": float(sub["lat"].iloc[0]),
            "n_hot": len(H), "n_retained": len(R),
            "hot_mean": H.mean(), "retained_mean": R.mean(),
            "hot_minus_retained": diff_groups,
            "full_minus_reduced": full_minus_reduced,
            "se": se, "t": t, "p": p,
            "inter_model_std": inter_model_std, "signal_to_noise": sn,
            "significant": bool(np.isfinite(p) and p < ALPHA),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  metric resolution (mean + auto-detected percentile columns)
# ---------------------------------------------------------------------------
def resolve_metrics(arg, columns):
    """Map the CLI `arg` to an ordered list of (variable, label, column) to
    analyse. Percentile columns are AUTO-DETECTED from the cache, so whatever
    fig02_precip_change.py wrote is picked up. Returns [] if nothing matches."""
    cols = set(columns)
    pr = sorted(int(re.match(r"precip_q(\d+)_change_pct$", c).group(1))
                for c in cols if re.match(r"precip_q\d+_change_pct$", c))
    tp = sorted(int(re.match(r"temp_q(\d+)_change_degC$", c).group(1))
                for c in cols if re.match(r"temp_q\d+_change_degC$", c))
    precip = ([("precip", "Mean", "precip_change_pct")]
              if "precip_change_pct" in cols else []) \
        + [("precip", f"Q{q}", f"precip_q{q}_change_pct") for q in pr]
    temp = ([("temp", "Mean", "temp_change_degC")]
            if "temp_change_degC" in cols else []) \
        + [("temp", f"Q{q}", f"temp_q{q}_change_degC") for q in tp]
    a = str(arg).lower()
    if a in ("precip", "pr", "precipitation"):
        return precip
    if a in ("temp", "tas", "temperature"):
        return temp
    if a == "all":
        return precip + temp
    if arg in cols:                                    # explicit single column
        var = ("precip" if arg.startswith("precip")
               else "temp" if arg.startswith("temp") else "?")
        label = "Mean" if arg in ("precip_change_pct",
                                  "temp_change_degC") else arg
        return [(var, label, arg)]
    return []


def resolve_flow_metrics(arg, columns):
    """Map `arg` to (variable, label, column) metrics from the streamflow cache.

    A bare flow keyword returns EVERY streamflow metric in the registry that the
    cache actually holds (variable='flow'); 'pindex' returns every precipitation
    index (variable='precip_index'); an explicit delta_ column returns just it,
    with the variable taken from the registry. Metrics the registry knows but
    the cache lacks are NAMED on stdout rather than dropped in silence.
    """
    cols = set(columns)
    a = str(arg).lower()

    def _resolve(metric_defs, variable, group):
        have = [(c, l) for (c, l) in metric_defs if c in cols]
        gone = [c for (c, _) in metric_defs if c not in cols]
        if gone:
            print(f"WARNING: {len(gone)} {group} metric(s) are in the registry "
                  f"but NOT in the cache — rebuild build_streamflow_cache.py "
                  f"if this is unexpected: {gone}")
        return [(variable, l, c) for (c, l) in have]

    if a in FLOW_KEYWORDS:
        return _resolve(FLOW_METRIC_DEFS, "flow", "streamflow")
    if a in PINDEX_KEYWORDS:
        return _resolve(PINDEX_METRIC_DEFS, "precip_index", "precipitation-index")
    if arg in cols:                                    # explicit single column
        m = BY_COLUMN.get(arg)
        variable = ("precip_index"
                    if m is not None and m in PRECIP_INDEX_METRICS else "flow")
        label = m.label if m is not None else arg
        return [(variable, label, arg)]
    return []


def is_pindex_arg(arg):
    """True when `arg` selects the precipitation indices rather than the flow
    metrics — both live in the streamflow cache but describe different things."""
    if str(arg).lower() in PINDEX_KEYWORDS:
        return True
    m = BY_COLUMN.get(arg)
    return m is not None and m in PRECIP_INDEX_METRICS


def pick_dataset(arg):
    """'flow' if `arg` is a streamflow/precip-index keyword or a delta_ column,
    else 'climate'. Decides which cache to load before resolving metrics. The
    precipitation indices live in the STREAMFLOW cache (apply_deltas_stn.py
    records them in the same stats.parquet rows), so they load 'flow' too."""
    a = str(arg).lower()
    if (a in FLOW_KEYWORDS or a in PINDEX_KEYWORDS
            or str(arg).startswith("delta_")):
        return "flow"
    return "climate"


# ---------------------------------------------------------------------------
#  per-CELL worker: Q2 + Q1 + Q3 for ONE (variable, label, column, scenario,
#  period) unit. Reads the forked global _WORK_DF so the big frame is never
#  pickled to workers. Returns (row_dict, focus_or_None), or None for an empty
#  slice. Top-level so a fork Pool can dispatch it.
# ---------------------------------------------------------------------------
def _run_unit(unit):
    var, label, col, scen, per = unit
    df = _WORK_DF
    # Slice to the (scenario, period) rows AND to the five columns this unit
    # actually reads: station + lon/lat (focus export), model (variance
    # partition / hot-vs-retained grouping) and the metric itself. The old code
    # took a FULL-WIDTH copy, so each of the 64 forked workers duplicated all
    # metric columns of its slice — with the registry's 41 metrics over ~20.6k
    # stations that peaks far past the 190 G cgroup limit. Column-slimming the
    # slice makes the worker footprint independent of how many metrics the pass
    # analyses, which is what lets the full metric set run at 64 workers/190 G.
    keep = ["station", "lon", "lat", "model", col]
    sl = df.loc[(df["scenario"] == scen) & (df["period"] == per), keep]
    if sl.empty:
        return None
    # model counts present for THIS scenario/period (varies by scenario).
    # Under weighting the "reduced" count becomes the rounded Kish effective
    # ensemble size of the renormalised weights over the present models.
    present_models = set(sl["model"].unique())
    n_full = len(present_models)
    if em.is_weighting():
        n_reduced = int(round(em.n_eff_present(present_models)))
    else:
        n_reduced = len(present_models & set(RETAINED_MODELS))
    vp = variance_partition(sl, col)
    ts = per_station_test(sl, col)

    # internal share of variance (median over catchments)
    if not vp.empty:
        tot = vp["s2_internal"].fillna(0) + vp["s2_model"]
        share = (vp["s2_internal"] / tot).replace([np.inf, -np.inf], np.nan)
        internal_share = float(np.nanmedian(share))
        med_sig_int = float(np.nanmedian(np.sqrt(vp["s2_internal"])))
        med_sig_mod = float(np.nanmedian(np.sqrt(vp["s2_model"])))
    else:
        internal_share = med_sig_int = med_sig_mod = np.nan

    if not ts.empty:
        frac_sig = float(ts["significant"].mean())
        # of the significant ones, how many share the dominant sign
        sig = ts[ts["significant"]]
        sign_consistency = (float((np.sign(sig["full_minus_reduced"])
                            == np.sign(sig["full_minus_reduced"].median())).mean())
                            if not sig.empty else np.nan)
        med_diff = float(ts["full_minus_reduced"].median())
        med_sn = float(np.nanmedian(ts["signal_to_noise"]))
        n_sta = len(ts)
    else:
        frac_sig = sign_consistency = med_diff = med_sn = np.nan
        n_sta = 0

    row = {
        "variable": var, "pctile": label,
        "scenario": scen, "period": per,
        "n_full": n_full, "n_reduced": n_reduced, "n_sta": n_sta,
        "internal_share": internal_share,
        "sigma_internal": med_sig_int, "sigma_model": med_sig_mod,
        "med_full_minus_reduced": med_diff,
        "frac_catch_significant": frac_sig,
        "sign_consistency": sign_consistency,
        "median_S/N": med_sn,
    }
    focus = ((col, ts) if scen == FOCUS_SCEN and per == FOCUS_PER
             and not ts.empty else None)
    return (row, focus)


# ---------------------------------------------------------------------------
#  per-catchment focus maps + CSV for one column
# ---------------------------------------------------------------------------
def _make_map(plt, df_pts, col, lim, title, out_png, stipple_pts=None):
    """Robinson scatter of `df_pts` coloured by full_minus_reduced on a fixed
    symmetric ±`lim` RdYlBu_r scale (Blue/-, Yellow/~0, Red/+; matches the
    hydro_indices maps). If `stipple_pts` is given, overplot those as small
    black dots (significance stippling)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(11, 5.6))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.5")
        tk = dict(transform=ccrs.PlateCarree())
    except Exception:
        fig, ax = plt.subplots(figsize=(11, 5.6))
        ax.set_xlim(-180, 180); ax.set_ylim(-60, 85); tk = {}
    sc = ax.scatter(df_pts["lon"], df_pts["lat"],
                    c=df_pts["full_minus_reduced"], s=6, cmap="RdYlBu_r",
                    vmin=-lim, vmax=lim, linewidths=0, rasterized=True, **tk)
    if stipple_pts is not None and len(stipple_pts):
        ax.scatter(stipple_pts["lon"], stipple_pts["lat"], s=2.5, marker=".",
                   c="k", linewidths=0, alpha=0.55, **tk)   # stippling
    # extend="both": the scale is the (capped) 98th pctile, so a few catchments
    # always sit beyond ±lim — the arrow caps signal that they clip to the ends.
    fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045,
                 pad=0.04, aspect=45, extend="both",
                 label=f"full - {em.second_label().lower()} ({col})")
    ax.set_title(title, fontsize=10)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_focus(focus_test, col, suffix=""):
    os.makedirs(CACHE_DIR, exist_ok=True)
    # nsta = catchments actually in THIS map (varies by metric: e.g. q05/q99
    # drop NaN-baseline catchments). Appended as '_n{nsta}sta'.
    base = (f"signif_{col}{suffix}_{FOCUS_SCEN}_{FOCUS_PER}"
            f"_n{len(focus_test)}sta")
    out_csv = em.tag(os.path.join(CACHE_DIR, base + ".csv"))  # DATA -> cache dir
    focus_test.to_csv(out_csv, index=False)
    sig = focus_test[focus_test["significant"]]
    n_sig = len(sig)
    print(f"  [{col}] {os.path.basename(out_csv)} | "
          f"catchments={len(focus_test)} | "
          f"significant={focus_test['significant'].mean():.1%} | "
          f"median S/N={np.nanmedian(focus_test['signal_to_noise']):.2f} | "
          f"median full-reduced={focus_test['full_minus_reduced'].median():.3f}")
    if not MAKE_FOCUS_MAPS:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(MAPS_DIR, exist_ok=True)
        # one symmetric colour scale (from ALL catchments) so both maps below
        # are directly comparable. For heavy-tailed relative metrics the p98 is
        # dragged out by a few near-zero-baseline blow-ups, so cap it near the
        # bulk (COLORBAR_LIM_CAP) — min() only tightens, never loosens.
        lim = float(np.nanpercentile(
            np.abs(focus_test["full_minus_reduced"]), 98)) or 1.0
        cap = COLORBAR_LIM_CAP.get(col)
        if cap is None and col.startswith("delta_rel_"):
            cap = DEFAULT_REL_LIM_CAP
        if cap is not None:
            lim = min(lim, cap)

        # (1) ALL catchments coloured, significant ones stippled black.
        # remove "signif_" from base
        base = base.replace("signif_", "", 1)
        out_all = em.tag(os.path.join(MAPS_DIR, base + ".png"))
        _make_map(plt, focus_test, col, lim,
                  f"{FOCUS_SCEN} {FOCUS_PER} [{col}]: full-reduced difference; "
                  f"dots = significant (p<{ALPHA})",
                  out_all, stipple_pts=sig)
        print(f"  [{col}] {os.path.basename(out_all)}")

        # (2) SIGNIFICANT catchments only (no stippling -> easier to read).
        if n_sig:
            sig_base = (f"{col}{suffix}_{FOCUS_SCEN}_{FOCUS_PER}"
                        f"_sigonly_n{n_sig}sta")
            out_sig = em.tag(os.path.join(MAPS_DIR, sig_base + ".png"))
            _make_map(plt, sig, col, lim,
                      f"{FOCUS_SCEN} {FOCUS_PER} [{col}]: full-reduced at "
                      f"SIGNIFICANT catchments only (p<{ALPHA}, n={n_sig})",
                      out_sig)
            print(f"  [{col}] {os.path.basename(out_sig)}")
        else:
            print(f"  [{col}] (0 significant catchments -> sig-only map skipped)")
    except Exception as e:
        print(f"  [{col}] (map skipped: {e})")


# ---------------------------------------------------------------------------
#  summary plots — answer Prof. Beck's questions straight from the table
# ---------------------------------------------------------------------------
def _pctile_key(lbl):
    """Sort key so 'Mean' comes first, then Q5 < Q50 < Q90 < ... ."""
    if lbl == "Mean":
        return -1
    m = re.match(r"Q(\d+)$", str(lbl))
    return int(m.group(1)) if m else 9999


def _pctile_key_q5first(lbl):
    """Like _pctile_key but 'Q5' sorts before 'Mean' (Q5, Mean, Q50, Q90, ...)."""
    if lbl == "Q5":
        return -2
    if lbl == "Mean":
        return -1
    m = re.match(r"Q(\d+)$", str(lbl))
    return int(m.group(1)) if m else 9999


# Reverse map from the short axis label a summary row carries ('Q05', 'Mean',
# 'AnnMax', 'R99pTOT') to the delta column it was computed from, so a streamflow
# or precipitation-index label can be looked up in figstyle's percentile table
# the same way a climate label is. Built from the registry, never hand-typed.
_LABEL_TO_COLUMN = {lab: col for (col, lab)
                    in FLOW_METRIC_DEFS + PINDEX_METRIC_DEFS}


def _metric_symbol(var, lbl):
    """Axis tick label for the summary-table metric label `lbl` of variable
    `var`, taken from figstyle's project-wide percentile table.

    The summary frame stores the terse internal labels resolve_metrics() builds
    ('Mean', 'Q5', 'Q50', ...). Those are ambiguous on a drawn axis: a bare
    'Q5' does not say which variable it belongs to, and it does not say whether
    it is the 5th percentile of the daily distribution (our convention) or the
    value exceeded on 5 % of days. figstyle.PERCENTILE_LABELS is the single
    project-wide table that resolves both, so 'Q5' on the temperature panel is
    drawn as $T_{p05}$ and 'Mean' as $T_{\\mathrm{mean}}$.

    Indices that never carried the ambiguity (AnnMax, RL10, FlashRB, the ETCCDI
    names) are absent from that table on purpose and keep the label they have
    in flow_metrics.py — the fallback below, not an error.
    """
    key = None
    if var in ("precip", "temp"):
        if lbl == "Mean":
            key = "precip_change_pct" if var == "precip" else "temp_change_degC"
        else:
            m = re.match(r"Q(\d+)$", str(lbl))
            if m:
                key = (f"precip_q{m.group(1)}_change_pct" if var == "precip"
                       else f"temp_q{m.group(1)}_change_degC")
    else:
        key = _LABEL_TO_COLUMN.get(lbl)
    lab = percentile_label(key) if key else None
    return lab.short if lab is not None else lbl


def _flow_key(lbl):
    """Sort key for streamflow indicators (drought/low -> high flow)."""
    return FLOW_ORDER.get(lbl, 9999)


def _count_lbl(nf, nr):
    """Legend fragment for the ensemble sizes: 'full=N, reduced=M' under
    removal; 'full=N, N_eff=M' under weighting (M = rounded Kish size, the
    n_reduced column of the summary frame)."""
    if em.is_weighting():
        return f"full={nf}, N_eff={nr}"
    return f"full={nf}, reduced={nr}"


def _cap(s):
    """Upper-case the first LETTER of an axis title / panel title, leaving the
    rest untouched (str.capitalize would lower-case acronyms such as TCR, and
    str[0].upper() would do nothing to a label that opens with '#' or a digit).
    House style: every panel title and axis label starts with a capital."""
    s = str(s)
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
    return s


def make_summary_plots(sm, order_key=_pctile_key, units=None,
                       xlabel="daily-distribution percentile", suffix="",
                       nsta=0, make_uncertainty=True, make_spectrum=True):
    """Render up to three figures from the per-(variable, pctile, scenario,
    period) summary table `sm`, saved to SIG_DIR. `suffix` (e.g. '_hbv') is
    appended to every filename so engines don't collide; `nsta` (# catchments in
    this analysis) is appended as '_n{nsta}sta'; `order_key` orders the metric
    labels along x (percentiles for climate, flow indicators for streamflow).
    Only testable cells (n_sta>0) are drawn:

      unc_decomp{suffix}_n{nsta}sta.png      : internal vs model variance share
            for the reference 'Mean' metric -- Prof. Beck Q2. Skipped when
            make_uncertainty=False (e.g. flow, where the per-engine decomps are
            replaced by the combined make_combined_flow_uncertainty figure).
      signal_vs_noise_<var>{suffix}_n{nsta}sta.png     : sigma_internal,
            sigma_model and the |full-reduced| signal at every metric label,
            one panel per scenario x period -- Q1/Q3.
      spectrum_forcing_{suffix}_n{nsta}sta.png         : % catchments
            significant and median S/N vs the metric label, per scenario -- Q4.
            Skipped when make_spectrum=False (e.g. flow, where the per-engine
            spectra are replaced by the combined make_combined_flow_spectrum
            figure).
    """
    if sm is None or sm.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(summary plots skipped: {e})")
        return
    os.makedirs(SIG_DIR, exist_ok=True)
    units = units or {"precip": "%", "temp": "°C", "flow": "%"}

    d = sm[sm["n_sta"] > 0].copy()
    if d.empty:
        print("(summary plots skipped: no testable scenario/period rows)")
        return
    variables = [v for v in ("precip", "temp", "flow") if v in set(d["variable"])]
    if not variables:
        variables = sorted(set(d["variable"]))
    C_INT, C_MOD, C_SIG = "#4C9F70", "#C24B4B", "#3B6FB6"

    def _xlabels(dd):
        return (dd["scenario"].str.upper() + "\n" + dd["period"]).tolist()

    # ---- Figure 1: uncertainty decomposition (reference 'Mean' metric) -----
    # Skipped for flow (make_uncertainty=False): the per-engine decomposition is
    # replaced by the combined make_combined_flow_uncertainty figure.
    f1 = None
    if make_uncertainty:
        fig, axes = plt.subplots(1, len(variables),
                                 figsize=(6.0 * len(variables), 4.6),
                                 squeeze=False)
        for j, var in enumerate(variables):
            ax = axes[0][j]
            dm = (d[(d["variable"] == var) & (d["pctile"] == "Mean")]
                  .sort_values(["scenario", "period"]))
            if dm.empty:
                continue
            x = np.arange(len(dm))
            ish = dm["internal_share"].to_numpy(float)
            ax.bar(x, ish, color=C_INT, label="internal variability")
            ax.bar(x, 1 - ish, bottom=ish, color=C_MOD, label="model uncertainty")
            ax.axhline(0.5, color="0.4", lw=0.6, ls="--")
            ax.set_xticks(x); ax.set_xticklabels(_xlabels(dm), fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("share of realization variance")
            ax.set_title(f"{var}: mean change")
            if j == 0:
                ax.legend(fontsize=8, loc="lower center")
        add_panel_letters(axes)
        fig.tight_layout()
        f1 = em.tag(os.path.join(SIG_DIR, f"unc_decomp{suffix}_n{nsta}sta.png"))
        fig.savefig(f1, dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 2: signal vs noise across metric labels, one fig per var ----
    # rows = period, cols = scenario; each panel = grouped bars of
    # sigma_internal, sigma_model and |full-reduced| at every metric label.
    sig_noise_files = []
    for var in variables:
        dvar = d[d["variable"] == var]
        pers = sorted(set(dvar["period"]))
        scen_list = [s for s in ("ssp126", "ssp245", "ssp370", "ssp585")
                     if s in set(dvar["scenario"])]
        # sigma_internal / sigma_model / |full-reduced| are ABSOLUTE, in the
        # units of the metric, so a days-valued metric (the half-flow date)
        # cannot share this y-axis with the per-cent ones. Drop those labels
        # here only — they keep their row in the printed table, their focus CSV,
        # their map and their place in the spectrum figure (whose quantities,
        # % significant and S/N, are scale-invariant).
        order = [lbl for lbl in sorted(set(dvar["pctile"]), key=order_key)
                 if lbl not in MIXED_UNIT_LABELS]
        dropped = sorted(set(dvar["pctile"]) & MIXED_UNIT_LABELS)
        if dropped:
            print(f"  (signal_vs_noise[{var}]: {', '.join(dropped)} omitted — "
                  f"not in {units.get(var, '?')}; see the spectrum figure)")
        # Streamflow only: keep the axis to the indices defined in Supplementary
        # Table S6. The precipitation-index and climate passes carry their own
        # label sets, which that table does not describe, so they are untouched.
        if var == "flow":
            off_table = [lbl for lbl in order if lbl not in TABLE_S6_LABELS]
            order = [lbl for lbl in order if lbl in TABLE_S6_LABELS]
            if off_table:
                print(f"  (signal_vs_noise[{var}]: {len(off_table)} metrics not "
                      f"in Supplementary Table S6 omitted — "
                      f"{', '.join(off_table)})")
        if not pers or not scen_list or not order:
            continue
        nrow, ncol = len(pers), len(scen_list)
        pw, ph = SN_PANEL_SIZE.get(var, SN_PANEL_SIZE_DEFAULT)
        fs_title = SN_TITLE_FONTSIZE.get(var, SN_TITLE_FONTSIZE_DEFAULT)
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(pw * ncol + 1.0, ph * nrow + 0.6),
                                 squeeze=False, sharey="row")
        # Every panel uses the SAME metric order (all labels in `order`) so the
        # period rows share one x-axis and align column-for-column; a metric
        # absent from a given (period, scenario) cell leaves a NaN gap.
        for r, per in enumerate(pers):
            for cc, s in enumerate(scen_list):
                ax = axes[r][cc]
                dps = (dvar[(dvar["period"] == per) & (dvar["scenario"] == s)]
                       .set_index("pctile"))
                if dps.empty:
                    ax.set_visible(False)
                    continue
                x = np.arange(len(order)); w = 0.27
                si = [float(dps.loc[o, "sigma_internal"]) if o in dps.index
                      else np.nan for o in order]
                sm_ = [float(dps.loc[o, "sigma_model"]) if o in dps.index
                       else np.nan for o in order]
                sg = [abs(float(dps.loc[o, "med_full_minus_reduced"]))
                      if o in dps.index else np.nan for o in order]
                ax.bar(x - w, si, w, color=C_INT, label="σ internal")
                ax.bar(x, sm_, w, color=C_MOD, label="σ model")
                ax.bar(x + w, sg, w, color=C_SIG,
                       label=f"|full − {em.second_label().lower()}| bias")
                ax.set_xticks(x)
                # x-axis is shared down each column, so only the bottom row is
                # labelled (rotated 90°); upper rows hide their tick labels.
                if r == nrow - 1:
                    ax.set_xticklabels(
                        [_metric_symbol(var, o) for o in order],
                        fontsize=SN_TICK_FONTSIZE, rotation=90)
                else:
                    ax.set_xticklabels([])
                ax.tick_params(axis="y", labelsize=SN_TICK_FONTSIZE)
                ax.grid(axis="y", color="0.92", lw=0.5); ax.set_axisbelow(True)
                if r == 0:
                    nf = int(dps["n_full"].iloc[0])
                    nr = int(dps["n_reduced"].iloc[0])
                    ax.set_title(f"{s.upper()}  ({_count_lbl(nf, nr)})",
                                 fontsize=fs_title)
                if cc == 0:
                    ax.set_ylabel(f"{plabel(per)}\n{units.get(var, '')}",
                                  fontsize=SN_YLABEL_FONTSIZE)
                if r == 0 and cc == 0:
                    ax.legend(fontsize=SN_LEGEND_FONTSIZE)
        add_panel_letters(axes)
        fig.tight_layout()
        fout = em.tag(os.path.join(
            SIG_DIR, f"signal_vs_noise_{var}{suffix}_n{nsta}sta.png"))
        fig.savefig(fout, dpi=200, bbox_inches="tight"); plt.close(fig)
        sig_noise_files.append(fout)

    # ---- Figure 3: significance + S/N spectrum vs metric label (focus per) -
    # legend shows per-scenario FULL / REDUCED model counts (they vary).
    # Skipped for flow (make_spectrum=False): the per-engine spectra are replaced
    # by the combined make_combined_flow_spectrum figure.
    f3 = None
    dp = d[d["period"] == FOCUS_PER] if make_spectrum else d.iloc[0:0]
    if make_spectrum and dp.empty:
        f3 = "(skipped: no FOCUS_PER rows)"
    if not dp.empty:
        fig, axes = plt.subplots(len(variables), 2,
                                 figsize=(11.5, 4.2 * len(variables)),
                                 squeeze=False)
        row_labels = []          # legend entries per variable row (see below)
        for i, var in enumerate(variables):
            dv = dp[dp["variable"] == var]
            # Second row only: put Q5 before Mean (Q5, Mean, Q50, Q90, ...).
            row_key = (_pctile_key_q5first
                       if (order_key is _pctile_key and i == 1) else order_key)
            order = sorted(set(dv["pctile"]), key=row_key)
            x = np.arange(len(order))
            scen_list = [s for s in ("ssp126", "ssp245", "ssp370", "ssp585")
                         if s in set(dv["scenario"])]
            colors = plt.get_cmap("viridis")(
                np.linspace(0, 0.85, max(1, len(scen_list))))
            row_labels.append([])
            for s, c in zip(scen_list, colors):
                ds = dv[dv["scenario"] == s].set_index("pctile")
                nf = int(ds["n_full"].iloc[0]) if len(ds) else 0
                nr = int(ds["n_reduced"].iloc[0]) if len(ds) else 0
                lbl = f"{s.upper()} ({_count_lbl(nf, nr)})"
                row_labels[i].append(lbl)
                sig = [ds.loc[o, "frac_catch_significant"] * 100
                       if o in ds.index else np.nan for o in order]
                sn = [ds.loc[o, "median_S/N"]
                      if o in ds.index else np.nan for o in order]
                axes[i][0].plot(x, sig, marker="o", color=c, label=lbl)
                axes[i][1].plot(x, sn, marker="o", color=c, label=lbl)
            for ax in axes[i]:
                ax.set_xticks(x)
                # Same project-wide symbol table the signal_vs_noise panels
                # use, so the two figures name an index identically. Labels the
                # table does not cover (the ETCCDI names of the _pindex pass)
                # come back unchanged.
                ax.set_xticklabels([_metric_symbol(var, o) for o in order],
                                   rotation=45,
                                   fontsize=SPECTRUM_TICK_FONTSIZE)
                ax.tick_params(axis="y", labelsize=SPECTRUM_TICK_FONTSIZE)
                ax.set_xlabel(_cap(xlabel), fontsize=SPECTRUM_LABEL_FONTSIZE)
                ax.grid(axis="y", color="0.92", lw=0.5); ax.set_axisbelow(True)
            axes[i][0].set_ylabel(_cap(f"{var}\n% catchments significant"),
                                  fontsize=SPECTRUM_LABEL_FONTSIZE)
            axes[i][0].set_ylim(0, 100)
            axes[i][1].set_ylabel(_cap("median bias-to-spread ratio\n(|full−reduced| / climate-model σ)"),
                                  fontsize=SPECTRUM_LABEL_FONTSIZE)
            axes[i][1].axhline(1.0, color="0.4", ls="--", lw=0.8)
            if suffix not in SPECTRUM_NO_PANEL_TITLE:
                axes[i][0].set_title(_cap(f"{var}: significance vs metric"),
                                     fontsize=SPECTRUM_TITLE_FONTSIZE)
                axes[i][1].set_title(
                    _cap(f"{var}: bias-to-spread ratio vs metric"),
                    fontsize=SPECTRUM_TITLE_FONTSIZE)
        # ONE legend, on panel (a). Every variable row draws the same scenarios
        # in the same colours with the same per-scenario model counts (the
        # ensemble is screened once, not per variable), so the panel-(c) copy
        # was a duplicate. A row whose entries genuinely differ keeps its own
        # legend, so a future cache with unequal per-variable model coverage
        # cannot lose that information silently.
        axes[0][0].legend(fontsize=SPECTRUM_LEGEND_FONTSIZE, title=FOCUS_PER)
        for i in range(1, len(variables)):
            if row_labels[i] != row_labels[0]:
                print(f"  (spectrum_forcing{suffix}: {variables[i]} legend "
                      f"entries differ from {variables[0]} — keeping its own)")
                axes[i][0].legend(fontsize=SPECTRUM_LEGEND_FONTSIZE,
                                  title=FOCUS_PER)
        add_panel_letters(axes)
        fig.tight_layout()
        f3 = em.tag(os.path.join(
            SIG_DIR, f"spectrum_forcing{suffix}_n{nsta}sta.png"))
        fig.savefig(f3, dpi=200, bbox_inches="tight"); plt.close(fig)

    print(f"\nSummary plots -> {SIG_DIR}:")
    for f in [f1] + sig_noise_files + [f3]:
        if f is None:
            continue
        print(f"  {os.path.basename(f) if str(f).endswith('.png') else f}")


# ---------------------------------------------------------------------------
#  combined (HBV + LSTM) streamflow significance / S-N spectrum — one figure,
#  one row per engine. Mirrors Figure 3 of make_summary_plots but stacks the
#  engines instead of writing a separate per-engine file, so the paper can show
#  both engines together. Draws from the per-engine summary frames returned by
#  analyze_pass (no recompute). FOCUS_PER only, columns ordered by order_key.
# ---------------------------------------------------------------------------
def make_combined_flow_spectrum(sm_by_engine, order_key=_flow_key,
                                xlabel="flow metric", nsta=0,
                                labels=None, out_stem="spectrum_hydrindices"):
    """`labels` restricts the metric axis; default = Supplementary Table S6.
    Pass the complement (and a different `out_stem`) to draw the off-table
    indices as a companion figure. Both variants read the SAME summary frames,
    so neither recomputes anything."""
    if not sm_by_engine:
        return
    keep = TABLE_S6_LABELS if labels is None else frozenset(labels)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(combined spectrum skipped: {e})")
        return
    os.makedirs(SIG_DIR, exist_ok=True)
    engines = [e for e in ("hbv", "lstm") if e in sm_by_engine]
    if not engines:
        return

    def _eng_focus(eng):
        sm = sm_by_engine[eng]
        return sm[(sm["n_sta"] > 0) & (sm["period"] == FOCUS_PER)
                  & (sm["variable"] == "flow")]

    # Single metric order shared by BOTH engine rows (union of the FOCUS_PER
    # flow metrics present in either engine, ordered by order_key) so HBV and
    # LSTM sit on an identical x-axis and are directly comparable. Restricted to
    # `keep` (by default the indices defined in Supplementary Table S6, so the
    # figure and the table describe the same set).
    present = {o for eng in engines for o in set(_eng_focus(eng)["pctile"])}
    order = sorted(present & keep, key=order_key)
    dropped = sorted(present - keep, key=order_key)
    if dropped:
        print(f"  ({out_stem}: {len(dropped)} metrics outside this figure's "
              f"index set omitted — {', '.join(dropped)})")
    if not order:
        print(f"({out_stem} skipped: none of its metrics are present)")
        return
    x = np.arange(len(order))
    fig, axes = plt.subplots(len(engines), 2,
                             figsize=(11.5, 4.2 * len(engines)), squeeze=False)
    row_labels = []          # legend entries per engine row (see below)
    for i, eng in enumerate(engines):
        d = _eng_focus(eng)
        scen_list = [s for s in ("ssp126", "ssp245", "ssp370", "ssp585")
                     if s in set(d["scenario"])]
        colors = plt.get_cmap("viridis")(
            np.linspace(0, 0.85, max(1, len(scen_list))))
        row_labels.append([])
        for s, c in zip(scen_list, colors):
            ds = d[d["scenario"] == s].set_index("pctile")
            nf = int(ds["n_full"].iloc[0]) if len(ds) else 0
            nr = int(ds["n_reduced"].iloc[0]) if len(ds) else 0
            lbl = f"{s.upper()} ({_count_lbl(nf, nr)})"
            row_labels[i].append(lbl)
            sig = [ds.loc[o, "frac_catch_significant"] * 100
                   if o in ds.index else np.nan for o in order]
            sn = [ds.loc[o, "median_S/N"]
                  if o in ds.index else np.nan for o in order]
            axes[i][0].plot(x, sig, marker="o", color=c, label=lbl)
            axes[i][1].plot(x, sn, marker="o", color=c, label=lbl)
        bottom = (i == len(engines) - 1)   # shared x-axis: label bottom row only
        for ax in axes[i]:
            ax.set_xticks(x)
            if bottom:
                # Only the five percentile/mean indices carry the Q-symbol
                # ambiguity and are rewritten (Mean, Q50, Q95, Q99, Q05);
                # AnnMax, RL10/20/50, 7Q10, SD, FDCslope, FlashRB and Q/P are
                # unambiguous, are absent from the table on purpose, and keep
                # the label flow_metrics.py gives them.
                ax.set_xticklabels([_metric_symbol("flow", o) for o in order],
                                   rotation=90)
            else:
                ax.set_xticklabels([])
            ax.grid(axis="y", color="0.92", lw=0.5); ax.set_axisbelow(True)
        axes[i][0].set_ylabel(f"{eng.upper()}\n% catchments significant")
        axes[i][0].set_ylim(0, 100)
        axes[i][1].set_ylabel("median bias-to-spread ratio\n(|full−reduced| / climate-model σ)")
        axes[i][1].axhline(1.0, color="0.4", ls="--", lw=0.8)
        axes[i][0].set_title(f"{eng.upper()}: significance vs metric")
        axes[i][1].set_title(f"{eng.upper()}: bias-to-spread ratio vs metric")
    # ONE legend, on panel (a). Every row draws the same four scenarios with the
    # same colours and the same per-scenario model counts (the too-hot screening
    # is engine-independent), so the panel-(c) copy was a duplicate. A row whose
    # entries genuinely differ still gets its own legend, so a future cache with
    # unequal per-engine model coverage cannot lose that information silently.
    axes[0][0].legend(fontsize=SPECTRUM_LEGEND_FONTSIZE, title=FOCUS_PER)
    for i in range(1, len(engines)):
        if row_labels[i] != row_labels[0]:
            print(f"  ({out_stem}: {engines[i].upper()} legend entries differ "
                  f"from {engines[0].upper()} — keeping its own legend)")
            axes[i][0].legend(fontsize=SPECTRUM_LEGEND_FONTSIZE,
                              title=FOCUS_PER)
    add_panel_letters(axes)
    fig.tight_layout()
    out = em.tag(os.path.join(SIG_DIR, f"{out_stem}_n{nsta}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nCombined flow significance spectrum ({len(order)} indices) "
          f"-> {os.path.basename(out)}")


# ---------------------------------------------------------------------------
#  combined (HBV + LSTM) streamflow uncertainty decomposition — one figure,
#  one panel per engine (HBV left, LSTM right), each the internal-vs-model
#  variance share of the reference 'Mean' flow metric across scenario x period.
#  Mirrors Figure 1 of make_summary_plots but stacks the engines side by side.
# ---------------------------------------------------------------------------
def make_combined_flow_uncertainty(sm_by_engine, nsta=0):
    if not sm_by_engine:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(combined uncertainty skipped: {e})")
        return
    os.makedirs(SIG_DIR, exist_ok=True)
    engines = [e for e in ("hbv", "lstm") if e in sm_by_engine]
    if not engines:
        return
    C_INT, C_MOD = "#4C9F70", "#C24B4B"
    fig, axes = plt.subplots(1, len(engines),
                             figsize=(6.0 * len(engines), 4.6), squeeze=False)
    for j, eng in enumerate(engines):
        ax = axes[0][j]
        sm = sm_by_engine[eng]
        dm = (sm[(sm["n_sta"] > 0) & (sm["variable"] == "flow")
                 & (sm["pctile"] == "Mean")]
              .sort_values(["scenario", "period"]))
        if dm.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(dm))
        ish = dm["internal_share"].to_numpy(float)
        ax.bar(x, ish, color=C_INT, label="internal variability")
        ax.bar(x, 1 - ish, bottom=ish, color=C_MOD, label="model uncertainty")
        ax.axhline(0.5, color="0.4", lw=0.6, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(
            (dm["scenario"].str.upper() + "\n" + dm["period"]).tolist(),
            fontsize=8)
        ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel("share of realization variance")
            ax.legend(fontsize=8, loc="lower center")
        ax.set_title(f"{eng.upper()}: mean runoff change")
    add_panel_letters(axes)
    fig.tight_layout()
    out = em.tag(os.path.join(SIG_DIR, f"unc_decomp_flow_n{nsta}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"Combined flow uncertainty decomposition -> {os.path.basename(out)}")


# ---------------------------------------------------------------------------
#  combined FORCING + FLOW uncertainty decomposition — ONE 2x2 figure
#  (row 1 = forcing: precip, temp; row 2 = flow: HBV, LSTM), each panel the
#  internal-vs-model variance share of the reference 'Mean' metric across
#  scenario x period. Supersedes the separate unc_decomp_*.png (forcing) and
#  unc_decomp_flow_*.png (flow) figures, which are NO LONGER written by the
#  normal passes. This is a self-contained figure path: it reads ONLY the
#  reference 'Mean' column from the existing parquet caches (no cache rebuild),
#  runs the variance partition for that one column (no Welch battery, no focus
#  exports, no other figures), and writes the single combined PNG.
# ---------------------------------------------------------------------------
def _unc_share_unit(unit):
    """Worker: median internal-variance share of `col` for one (scenario,
    period) slice of the forked global _WORK_DF. Identical to _run_unit's
    internal_share (variance_partition -> median of s2_internal/(s2_internal +
    s2_model) over catchments), but WITHOUT the per-station Welch test."""
    col, scen, per = unit
    df = _WORK_DF
    # column-slimmed like _run_unit: variance_partition reads only station,
    # model and the metric, so the worker never copies the other columns.
    sl = df.loc[(df["scenario"] == scen) & (df["period"] == per),
                ["station", "model", col]]
    if sl.empty:
        return None
    vp = variance_partition(sl, col)
    if vp.empty:
        return None
    tot = vp["s2_internal"].fillna(0) + vp["s2_model"]
    share = (vp["s2_internal"] / tot).replace([np.inf, -np.inf], np.nan)
    return (scen, per, float(np.nanmedian(share)))


def _internal_share_cells(d, col):
    """Return [(scenario, period, internal_share), ...] for `col`, ordered by
    (scenario, period) exactly as the old figures' .sort_values did. The cells
    are independent, so they run through the same fork Pool as analyze_pass."""
    global _WORK_DF
    scen_order = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
    scens = [s for s in scen_order if s in set(d["scenario"])]
    pers = sorted(set(d["period"]))
    work = [(col, s, p) for s in scens for p in pers]
    if not work:
        return []
    _WORK_DF = d
    n_workers = min(NUM_WORKERS, len(work))
    if n_workers > 1 and len(work) > 1:
        with mp.get_context("fork").Pool(processes=n_workers) as pool:
            res = pool.map(_unc_share_unit, work)
    else:
        res = [_unc_share_unit(u) for u in work]
    _WORK_DF = None
    cells = [r for r in res if r is not None]
    cells.sort(key=lambda r: (r[0], r[1]))   # match old .sort_values order
    return cells


def _load_unc_frame(cache, dataset, metric_cols, engine=None):
    """Load ONLY the id + `metric_cols` columns for one engine, apply the
    blocklist + FULL_MODELS filter, cast metrics to float32 (delta_rel_* ->
    percent for flow), NaN-out infs on the metric columns. Mirrors main()'s
    _load closure exactly so the internal shares match the per-pass figures."""
    import pyarrow.parquet as pq
    schema_cols = list(pq.ParquetFile(cache).schema.names)
    ID_COLS = ["station", "lon", "lat", "scenario", "model", "member",
               "period", "engine"]
    need_cols = ([c for c in ID_COLS if c in schema_cols]
                 + [c for c in metric_cols if c not in ID_COLS])
    filters = [("engine", "==", engine)] if engine is not None else None
    d = pd.read_parquet(cache, columns=need_cols, filters=filters)
    d = apply_blocklist(
        d, label=f"{dataset} significance cache"
                 f"{'' if engine is None else f' [{engine}]'}")
    d = d[d["model"].isin(FULL_MODELS)].copy()
    d["scenario"] = d["scenario"].astype(str)
    d["period"]   = d["period"].astype(str)
    scale = np.float32(100.0)
    for c in metric_cols:
        col = d[c].astype("float32")
        if dataset == "flow" and c.startswith("delta_rel_"):
            col = col * scale
        d[c] = col
    d[metric_cols] = d[metric_cols].replace([np.inf, -np.inf], np.nan)
    return d


def make_combined_uncertainty():
    """Build the single 2x2 uncertainty-decomposition figure (row 1 = forcing
    precip/temp, row 2 = flow HBV/LSTM). Reads only the reference 'Mean' column
    from the climate and flow caches; no cache rebuild, no other outputs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        sys.exit(f"(combined uncertainty skipped: {e})")
    for cache, builder in ((CLIMATE_CACHE, "fig02_precip_change.py"),
                           (FLOW_CACHE, "build_streamflow_cache.py")):
        if not os.path.exists(cache):
            sys.exit(f"cache not found: {cache} (run {builder} first)")

    # panels: (row, col_idx, label, dataset, cache, metric_col, engine)
    FORCING = [("precip", "precip_change_pct", None),
               ("temp",   "temp_change_degC",  None)]
    FLOW    = [("HBV",  "delta_rel_qmean", "hbv"),
               ("LSTM", "delta_rel_qmean", "lstm")]
    C_INT, C_MOD = "#4C9F70", "#C24B4B"

    print("=== combined uncertainty decomposition (forcing + flow, one figure) ===")
    # --- forcing row: precip + temp share of the 'Mean' change ------------
    dfc = _load_unc_frame(CLIMATE_CACHE, "climate",
                          [c for _, c, _ in FORCING], engine=None)
    forcing_cells, nf = {}, int(dfc["station"].nunique()) if not dfc.empty else 0
    for lab, col, _ in FORCING:
        forcing_cells[lab] = _internal_share_cells(dfc, col) if col in dfc else []
    del dfc

    # --- flow row: HBV + LSTM share of the 'Mean' runoff change -----------
    flow_cells, flow_sta = {}, set()
    for lab, col, eng in FLOW:
        de = _load_unc_frame(FLOW_CACHE, "flow", [col], engine=eng)
        if de.empty:
            flow_cells[lab] = []
            continue
        flow_sta |= set(de["station"].unique())
        flow_cells[lab] = _internal_share_cells(de, col)
        del de
    nq = len(flow_sta)

    def _panel(ax, cells, title, show_ylabel, show_legend,
               show_xticklabels=True):
        if not cells:
            ax.set_visible(False)
            return
        x = np.arange(len(cells))
        ish = np.array([c[2] for c in cells], dtype=float)
        ax.bar(x, ish, color=C_INT, label="internal variability")
        ax.bar(x, 1 - ish, bottom=ish, color=C_MOD, label="model uncertainty")
        ax.axhline(0.5, color="0.4", lw=0.6, ls="--")
        ax.set_xticks(x)
        # The forcing (row 1) and flow (row 2) panels share one scenario x period
        # x-axis, so the top row's tick labels are hidden and only the bottom row
        # is labelled. Scenario and period stack on two rows, which keeps the
        # label horizontal but makes its width ('2041-2060', ~5.5 em) the binding
        # constraint on how narrow the figure can be: eight of them must fit
        # across each panel, so the width below is the smallest that clears them.
        if show_xticklabels:
            ax.set_xticklabels([f"{s.upper()}\n{plabel(p)}" for s, p, _ in cells],
                               fontsize=7)
        else:
            ax.set_xticklabels([])
        ax.set_ylim(0, 1)
        if show_ylabel:
            ax.set_ylabel("Share of realization variance", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        if show_legend:
            ax.legend(fontsize=11, loc="lower center")
        ax.set_title(title, fontsize=11)

    def _mtitle(col, engine=None):
        """Panel title from figstyle's project-wide label table: plain-language
        name plus symbol, so the panel states WHICH statistic of the variable it
        shows. The old titles ("Precip: mean change") named the variable but not
        the statistic, and a bare "Q" on a temperature or precipitation panel is
        the collision PERCENTILE_LABELS exists to remove. Raises rather than
        falling back, exactly as _psym does for the cascade figure."""
        lab = percentile_label(col)
        if lab is None:
            raise KeyError(f"{col!r} is not in figstyle.PERCENTILE_LABELS; add "
                           "it there rather than typing a bare metric name here")
        head = f"{lab.long} ({lab.short})"
        return head if engine is None else f"{head}, {engine}"

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), squeeze=False)
    _panel(axes[0][0], forcing_cells.get("precip", []),
           _mtitle("precip_change_pct"), True, True, show_xticklabels=False)
    _panel(axes[0][1], forcing_cells.get("temp", []),
           _mtitle("temp_change_degC"), False, False, show_xticklabels=False)
    _panel(axes[1][0], flow_cells.get("HBV", []),
           _mtitle("delta_rel_qmean", "HBV"), True, False)
    _panel(axes[1][1], flow_cells.get("LSTM", []),
           _mtitle("delta_rel_qmean", "LSTM"), False, False)
    add_panel_letters(axes)
    fig.tight_layout()
    os.makedirs(SIG_DIR, exist_ok=True)
    out = os.path.join(SIG_DIR,
                       f"unc_decomp_forc{nf}_flow{nq}sta.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"forcing catchments n={nf} | flow catchments n={nq}")
    print(f"Combined uncertainty decomposition -> {out}")


# ---------------------------------------------------------------------------
#  one analysis pass (one engine for flow, or the single climate pass)
# ---------------------------------------------------------------------------
def analyze_pass(df, metrics, scenarios, periods, order_key, units, xlabel,
                 suffix="", engine_label=None,
                 make_uncertainty=True, make_spectrum=True):
    """Run every (column, scenario, period) cell through the Q1/Q2/Q3 battery --
    in PARALLEL via a fork Pool over the ~100 independent cells (workers inherit
    `df` copy-on-write) -- then print the summary table, render the plots, and
    write the focus exports. `suffix` (e.g. '_hbv') is appended to all
    figure/CSV filenames so engines don't collide. `make_uncertainty` /
    `make_spectrum` (both False for per-engine flow passes) gate the unc_decomp
    and spectrum_forcing summary figures, which for flow are superseded by the
    combined two-engine figures. Returns the summary frame."""
    global _WORK_DF
    _WORK_DF = df
    work = [(var, label, col, scen, per)
            for (var, label, col) in metrics
            for scen in scenarios for per in periods]
    n_workers = min(NUM_WORKERS, len(work))
    if n_workers > 1 and len(work) > 1:
        with mp.get_context("fork").Pool(processes=n_workers) as pool:
            results = pool.map(_run_unit, work)        # input order preserved
    else:
        results = [_run_unit(u) for u in work]
    _WORK_DF = None                                    # drop parent's reference

    all_summary = []
    focus_jobs = []
    for res in results:
        if res is None:
            continue
        row, focus = res
        all_summary.append(row)
        if focus is not None:
            focus_jobs.append(focus)

    sm = pd.DataFrame(all_summary)
    pd.set_option("display.width", 200, "display.max_columns", 30,
                  "display.max_rows", 300)
    hdr = "" if engine_label is None else f"  [engine = {engine_label}]"
    print(f"PER (variable, pctile, scenario, period) SUMMARY{hdr}")
    print("  pctile         : Mean, or the metric label (percentile / flow indicator)")
    print("  n_full/n_reduced : # models in the FULL / REDUCED ensemble "
          "present for that scenario (varies by scenario)")
    print("  internal_share : median fraction of realization variance that is "
          "internal variability (rest is model uncertainty)")
    print("  sigma_*        : median per-catchment std (units of the value col)")
    print("  frac_catch_significant : fraction of catchments where hot!=retained "
          f"at p<{ALPHA} (Welch)")
    print("  sign_consistency       : of those, fraction sharing the dominant sign")
    print("  median_S/N     : |full-reduced| / inter-model std  (single-model masking; <1 = masked)\n")
    print(sm.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if MAKE_SUMMARY_PLOTS:
        nsta = int(df["station"].nunique())   # catchment universe for this pass
        make_summary_plots(sm, order_key=order_key, units=units,
                           xlabel=xlabel, suffix=suffix, nsta=nsta,
                           make_uncertainty=make_uncertainty,
                           make_spectrum=make_spectrum)

    if focus_jobs:
        print(f"\nFocus per-catchment export [{FOCUS_SCEN} {FOCUS_PER}]{hdr}:")
        for col, focus_test in focus_jobs:
            export_focus(focus_test, col, suffix=suffix)
    return sm


# ---------------------------------------------------------------------------
#  T -> Q transmission of the too-hot screening vs catchment character
# ---------------------------------------------------------------------------
#  Hypothesis: the (robust, S/N>1) TEMPERATURE screening signal propagates into
#  runoff in energy-limited / snow-dominated basins (flow tracks temperature)
#  but drowns in water-limited basins (flow tracks noisy precipitation). We
#  quantify it per catchment with a TRANSMISSION ratio
#
#      tau = S/N(mean-runoff correction) / S/N(mean-temperature correction)
#
#  both read from the per-catchment focus CSVs already written by export_focus
#  (no recompute). Dividing by the temperature S/N removes the polar-
#  amplification confound (big dT at high latitude cancels top and bottom), so
#  tau is a property of the catchment's hydrology, not of where it sits. tau~1:
#  the river inherits the temperature-correction detectability; tau~0: it
#  drowns. tau is then related to aridity index / snow fraction / Koeppen zone
#  (static.csv). Flow target = MEAN ANNUAL RUNOFF (delta_rel_qmean); HBV and
#  LSTM are kept separate so agreement is a robustness check. The formal
#  confound test is a Spearman PARTIAL correlation controlling for |lat|.
# ---------------------------------------------------------------------------
def _latest_focus_csv(pattern):
    """Newest (by mtime) focus CSV in CACHE_DIR matching `pattern`, else None."""
    hits = glob.glob(os.path.join(CACHE_DIR, pattern))
    return max(hits, key=os.path.getmtime) if hits else None


def _load_static_attrs():
    """station -> (aridity, snowfrac, kgz) from static.csv."""
    st = pd.read_csv(STATIC_CSV, comment="#", index_col=0).rename_axis("station")
    st = st.reset_index()
    out = pd.DataFrame({"station": st["station"].astype(str)})
    out["aridity"]  = pd.to_numeric(st[TRANSMISSION_ATTRS["aridity"]],  errors="coerce")
    out["snowfrac"] = pd.to_numeric(st[TRANSMISSION_ATTRS["snowfrac"]], errors="coerce")
    out["kgz"]      = st[TRANSMISSION_ATTRS["kgz"]].astype(str).str.strip()
    return out


def _partial_spearman(x, y, z):
    """Spearman partial correlation of x and y controlling for z (correlation of
    the rank residuals after regressing each on z). Returns (r, p, n); (nan, nan,
    n) if scipy is missing or n<10."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    n = int(len(x))
    if n < 10 or not HAVE_SCIPY:
        return (np.nan, np.nan, n)
    rx, ry, rz = sps.rankdata(x), sps.rankdata(y), sps.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = sps.pearsonr(ex, ey)
    return (float(r), float(p), n)


def make_transmission_figure(flow_col="delta_rel_qmean", flow_label="qmean"):
    """Build the T->Q transmission-vs-catchment-character figure from the cached
    focus CSVs (temperature + per-engine mean-runoff) joined to static.csv. No
    delta / hydrology recompute. Writes one PNG to SIG_DIR and a per-catchment
    table to CACHE_DIR."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Patch
    except Exception as e:
        print(f"(transmission figure skipped: {e})")
        return

    tpath = _latest_focus_csv(
        f"signif_temp_change_degC_{FOCUS_SCEN}_{FOCUS_PER}"
        f"_n*sta{em.suffix()}.csv")
    if tpath is None:
        print("transmission: no temperature focus CSV found "
              "(run the climate 'all' pass first). Abort.")
        return
    temp = pd.read_csv(tpath)[["station", "signal_to_noise"]].rename(
        columns={"signal_to_noise": "sn_temp"})
    temp["station"] = temp["station"].astype(str)
    print(f"transmission: temperature S/N from {os.path.basename(tpath)} "
          f"({len(temp)} catchments)")

    attrs = _load_static_attrs()
    print(f"transmission: static attributes for {len(attrs)} stations "
          f"from {os.path.basename(STATIC_CSV)}")

    ECOL = {"hbv": "#3B6FB6", "lstm": "#C24B4B"}
    per_eng, stats_txt = {}, {}
    for eng in ("hbv", "lstm"):
        fpath = _latest_focus_csv(
            f"signif_{flow_col}_{eng}_{FOCUS_SCEN}_{FOCUS_PER}"
            f"_n*sta{em.suffix()}.csv")
        if fpath is None:
            print(f"transmission: no {eng} {flow_col} focus CSV "
                  "(run the flow pass first); skipping engine.")
            continue
        fl = pd.read_csv(fpath)[["station", "lon", "lat",
                                 "signal_to_noise", "significant"]].rename(
            columns={"signal_to_noise": "sn_flow", "significant": "sig_flow"})
        fl["station"] = fl["station"].astype(str)
        m = fl.merge(temp, on="station").merge(attrs, on="station")
        # temperature must be detectable to normalise by it (it is ~everywhere)
        m = m[np.isfinite(m["sn_temp"]) & (m["sn_temp"] > 0.1)
              & np.isfinite(m["sn_flow"])].copy()
        m["tau"] = m["sn_flow"] / m["sn_temp"]
        m["abslat"] = m["lat"].abs()
        per_eng[eng] = m
        print(f"transmission: {eng} {os.path.basename(fpath)} -> "
              f"{len(m)} catchments joined")

    if not per_eng:
        print("transmission: no engine data joined; abort.")
        return

    # ---- confound check: partial Spearman controlling |lat| ---------------
    print("\nPartial Spearman (control |lat|)   tau vs attribute:")
    for eng, m in per_eng.items():
        ra, pa, na = _partial_spearman(m["aridity"],  m["tau"], m["abslat"])
        rs, ps, ns = _partial_spearman(m["snowfrac"], m["tau"], m["abslat"])
        stats_txt[eng] = (ra, pa, rs, ps)
        print(f"  {eng:4s}: tau~aridity  r={ra:+.3f} p={pa:.1e} (n={na}) | "
              f"tau~snowfrac r={rs:+.3f} p={ps:.1e} (n={ns})")

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          hspace=0.33, wspace=0.24)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- Panel A: tau vs aridity (decile-binned medians) ------------------
    for eng, m in per_eng.items():
        d = m[np.isfinite(m["aridity"])]
        if len(d) < 60:
            continue
        edges = np.unique(np.nanpercentile(d["aridity"], np.arange(0, 101, 10)))
        if len(edges) < 3:
            continue
        b = np.clip(np.digitize(d["aridity"].to_numpy(float), edges[1:-1]),
                    0, len(edges) - 2)
        xs, ys = [], []
        for k in range(len(edges) - 1):
            sel = d[b == k]
            if len(sel) < 20:
                continue
            xs.append(float(np.median(sel["aridity"])))
            ys.append(float(np.median(sel["tau"])))
        # mfmt, not f"{...:+.2f}": the partial correlation is negative for both
        # engines, and a plain format would draw a hyphen next to the minus
        # signs matplotlib puts on this panel's own axes.
        axA.plot(xs, ys, marker="o", color=ECOL[eng],
                 label=f"{eng.upper()} (partial r={mfmt(stats_txt[eng][0], '+.2f')})")
    axA.set_xscale("log")
    axA.set_xlabel("aridity index (PET/P; higher = drier / water-limited)")
    axA.set_ylabel("transmission τ\n(flow bias-to-spread ÷ temperature bias-to-spread)")
    axA.grid(color="0.92", lw=0.5); axA.set_axisbelow(True)
    axA.legend(fontsize=8, title="control |lat|")
    axA.set_title("τ vs aridity (decile medians)")

    # ---- Panel B: tau by Koeppen major zone ------------------------------
    zones = [z for z in ("A", "B", "C", "D", "E")
             if any(z in set(m["kgz"]) for m in per_eng.values())]
    width = 0.36
    for j, eng in enumerate(per_eng):
        m = per_eng[eng]
        data, pos = [], []
        for zi, z in enumerate(zones):
            v = m.loc[m["kgz"] == z, "tau"].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) < 5:                       # skip empty/tiny (zone, engine)
                continue
            data.append(v); pos.append(zi + (j - 0.5) * width)
        if not data:
            continue
        bp = axB.boxplot(data, positions=pos, widths=width * 0.9,
                         showfliers=False, patch_artist=True,
                         medianprops=dict(color="k"))
        for box in bp["boxes"]:
            box.set(facecolor=ECOL[eng], alpha=0.6)
    axB.set_xticks(np.arange(len(zones)))
    axB.set_xticklabels(list(zones))
    # wrapped onto two lines: on one line the label overruns the axes by ~25 px
    # and bbox_inches="tight" sets the right boundary from the axes spine, not
    # from the label, so the trailing "w)" is clipped out of the saved PNG.
    axB.set_xlabel("Köppen–Geiger major zone\n(B arid, D cold, E polar/snow)")
    axB.set_ylabel("transmission τ")
    allt = np.concatenate([m["tau"].to_numpy(float) for m in per_eng.values()])
    allt = allt[np.isfinite(allt)]
    if len(allt):
        axB.set_ylim(0, float(np.nanpercentile(allt, 95)))
    axB.legend(handles=[Patch(facecolor=ECOL[e], alpha=0.6, label=e.upper())
                        for e in per_eng], fontsize=8)
    axB.grid(axis="y", color="0.92", lw=0.5); axB.set_axisbelow(True)
    axB.set_title("τ by climate zone")

    # ---- Panel C: map of two-engine-mean tau -----------------------------
    # keep lon/lat for every station (coalesce across engines) so stations
    # present in only one engine still map.
    coords = pd.concat([m[["station", "lon", "lat"]] for m in per_eng.values()],
                       ignore_index=True).drop_duplicates("station")
    merged = None
    for eng, m in per_eng.items():
        sub = m[["station", "tau"]].rename(columns={"tau": f"tau_{eng}"})
        merged = sub if merged is None else merged.merge(sub, on="station",
                                                         how="outer")
    tcols = [c for c in merged.columns if c.startswith("tau_")]
    merged["tau_mean"] = merged[tcols].mean(axis=1)   # skipna -> single engine ok
    merged = merged.merge(coords, on="station", how="left")
    mm = merged[np.isfinite(merged["tau_mean"])]
    vmax = float(np.nanpercentile(mm["tau_mean"], 90)) or 1.0

    have_cartopy = True
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception:
        have_cartopy = False
    if have_cartopy:
        axC = fig.add_subplot(gs[1, :], projection=ccrs.Robinson())
        axC.set_global()
        axC.add_feature(cfeature.LAND, facecolor="0.93", zorder=0)
        axC.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.5")
        tk = dict(transform=ccrs.PlateCarree())
    else:
        axC = fig.add_subplot(gs[1, :])
        axC.set_xlim(-180, 180); axC.set_ylim(-60, 85); tk = {}
    sc = axC.scatter(mm["lon"], mm["lat"], c=mm["tau_mean"], s=6,
                     cmap="viridis", vmin=0, vmax=vmax, linewidths=0,
                     rasterized=True, **tk)
    fig.colorbar(sc, ax=axC, orientation="horizontal", fraction=0.045,
                 pad=0.04, aspect=45, extend="max",
                 label="transmission τ (HBV–LSTM mean)")
    _flow_titles = {"qmean":  "mean annual runoff",
                    "annmax": "mean annual maximum daily flow"}
    axC.set_title("Where the temperature screening reaches "
                  f"{_flow_titles.get(flow_label, flow_label)} "
                  f"({FOCUS_SCEN} {plabel(FOCUS_PER)})")

    n_map = int(len(mm))
    fig.suptitle(f"T→Q transmission of the {_XHOT} screening vs catchment "
                 "character", fontsize=12, y=0.98)
    # axC is the wide map spanning the bottom row (gs[1, :]); it is created in
    # both the cartopy and the fallback branch, so it always exists here.
    # It is lettered separately: cartopy shrinks the GeoAxes box to the Robinson
    # aspect ratio, so the axes' own left edge sits well inside the (wider,
    # centred) panel title and a letter placed there lands on top of the title
    # text. Anchor "c" to the left edge of the bottom gridspec CELL (figure x)
    # while keeping the usual height above the map box (axes y), which also
    # aligns it with the "a" of the panel above.
    add_panel_letters([axA, axB])
    _cell = gs[1, :].get_position(fig)
    _tr = mtransforms.blended_transform_factory(fig.transFigure, axC.transAxes)
    fig.text(_cell.x0 + PANEL_LETTER_X, PANEL_LETTER_Y, "c", transform=_tr,
             fontsize=PANEL_LETTER_FONTSIZE, fontweight="bold",
             va="bottom", ha="left")
    out = em.tag(os.path.join(
        SIG_DIR, f"transmission_{flow_label}_{FOCUS_SCEN}_{FOCUS_PER}"
                 f"_n{n_map}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nTransmission figure -> {os.path.basename(out)}")

    # ---- per-catchment table (data file) ---------------------------------
    first = per_eng[list(per_eng)[0]]
    keep = merged.merge(first[["station", "aridity", "snowfrac", "kgz"]],
                        on="station", how="left")
    os.makedirs(CACHE_DIR, exist_ok=True)
    out_csv = em.tag(os.path.join(
        CACHE_DIR, f"transmission_{flow_label}_{FOCUS_SCEN}_{FOCUS_PER}"
                   f"_n{n_map}sta.csv"))
    keep.to_csv(out_csv, index=False)
    print(f"Transmission table  -> {os.path.basename(out_csv)}")


# ---------------------------------------------------------------------------
#  Beat-2: T -> P -> Q signal-to-noise CASCADE  (folded in from fig_snr_cascade)
# ---------------------------------------------------------------------------
#  Tracks the SAME too-hot correction as it propagates from the climate forcing
#  into streamflow, showing its per-catchment detectability (S/N = |full -
#  reduced| / inter-model sigma) collapse along the physical chain
#  Temperature -> Precipitation -> Runoff / high flow / peak flow. Reads the
#  focus CSVs written by export_focus (no recompute). ONLY magnitude / FDC
#  indices — flashiness and other persistence metrics are excluded (qqmap_day2
#  does not faithfully transfer day-to-day sequencing).
def _psym(col):
    """Short percentile symbol for `col` from figstyle's project-wide table.

    The four percentile stages used to carry hand-typed labels ("Temp\\nQ99",
    "High flow\\nQ95"). "Q95" reads either as the 95th percentile of daily flow
    (a HIGH flow, our convention) or as the flow exceeded on 95 % of days (a
    LOW flow, the flow-duration convention), and the cascade is the one figure
    that puts a temperature, a precipitation and two streamflow percentiles on
    the same axis, where the bare symbols are least distinguishable. Raising
    on an unknown column is deliberate: a typo must not fall back silently to
    a bare "Q99"."""
    lab = percentile_label(col)
    if lab is None:
        raise KeyError(f"{col!r} is not in figstyle.PERCENTILE_LABELS; add it "
                       "there rather than typing a bare percentile symbol into "
                       "CASCADE_STAGES")
    return lab.short


# ORDER IS BY DESCENDING MEDIAN SIGNAL-TO-NOISE, not by variable -- that
# descent is the cascade this figure exists to show, which is why temperature
# runs mean -> p99 while precipitation runs p99 -> mean. Do not regroup.
#
# Every tick is ONE symbol from figstyle.PERCENTILE_LABELS. The labels used to
# mix two systems: four carried a symbol and four carried words only ("Temp\n
# (mean)"), and the two that carried both named the variable twice ("Temp
# $T_{p99}$" = temperature, temperature). Drawing all eight from one family
# removes both problems and matches the notation in the main text and in
# Supplementary Tables S2, S3 and S6. The T_mean and P_mean entries exist in
# figstyle for exactly this axis; see the comment above them there.
#
# AnnMax keeps its name: it is deliberately outside PERCENTILE_LABELS because
# it never carried the percentile ambiguity, and it is the abbreviation the
# manuscript and flow_metrics.py already use.
CASCADE_STAGES = [
    dict(label=_psym("temp_change_degC"),
         col="temp_change_degC",            group="forcing", engines=[None]),
    dict(label=_psym("temp_q99_change_degC"),
         col="temp_q99_change_degC",        group="forcing", engines=[None]),
    dict(label=_psym("precip_q99_change_pct"),
         col="precip_q99_change_pct",       group="forcing", engines=[None]),
    dict(label=_psym("precip_change_pct"),
         col="precip_change_pct",           group="forcing", engines=[None]),
    dict(label="AnnMax",
         col="delta_rel_mean_annual_max",   group="impact",  engines=["hbv", "lstm"]),
    dict(label=_psym("delta_rel_q99"),
         col="delta_rel_q99",               group="impact",  engines=["hbv", "lstm"]),
    dict(label=_psym("delta_rel_q95"),
         col="delta_rel_q95",               group="impact",  engines=["hbv", "lstm"]),
    dict(label=_psym("delta_rel_qmean"),
         col="delta_rel_qmean",             group="impact",  engines=["hbv", "lstm"]),
]
CASCADE_COL_FORCING = "#5B6B73"
CASCADE_COL_HBV     = "#3B6FB6"
CASCADE_COL_LSTM    = "#C24B4B"
CASCADE_ENG_COLOR   = {None: CASCADE_COL_FORCING, "hbv": CASCADE_COL_HBV, "lstm": CASCADE_COL_LSTM}
CASCADE_SN_CLIP     = 1.5
# Panel-a axis cap, kept SEPARATE from the data clip above. The clip guards the
# violin kernel against an outlier; the cap sets how much empty head room the
# panel carries above the reference line at 1.0. The largest per-catchment ratio
# anywhere in the twelve cascade series is 0.53, so CASCADE_SN_CLIP is inert and
# the old cap of 1.5 spent a third of the panel on white space. 1.1 still shows
# the whole ratio-of-one line, which is the reference the panel exists to make.
# Raise this (never the clip) if a future treatment pushes a violin near 1.0.
CASCADE_SN_YMAX     = 1.1


def _cascade_load_stage(col, engine):
    """(sn_array, med, frac_sig, n, file, engine) for one (col, engine) focus
    CSV, or None if absent."""
    suffix = "" if engine is None else f"_{engine}"
    path = _latest_focus_csv(f"signif_{col}{suffix}_{FOCUS_SCEN}_{FOCUS_PER}"
                             f"_n*sta{em.suffix()}.csv")
    if path is None:
        return None
    df = pd.read_csv(path)
    sn = pd.to_numeric(df["signal_to_noise"], errors="coerce").to_numpy(float)
    sn = sn[np.isfinite(sn)]
    if sn.size == 0:
        return None
    frac_sig = float(pd.to_numeric(df["significant"].astype(float),
                                   errors="coerce").fillna(0).mean())
    return dict(sn=sn, med=float(np.median(sn)), frac_sig=frac_sig,
                n=int(sn.size), file=os.path.basename(path), engine=engine)


def make_snr_cascade():
    """Render the T->P->Q S/N cascade figure to SIG_DIR from the cached focus
    CSVs (run the climate 'all' and 'flow' passes first). No recompute."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as e:
        print(f"(snr cascade skipped: {e})")
        return

    cols = []
    print(f"S/N cascade | {FOCUS_SCEN} {FOCUS_PER} | reading focus CSVs from\n  {CACHE_DIR}")
    for si, st in enumerate(CASCADE_STAGES):
        for eng in st["engines"]:
            rec = _cascade_load_stage(st["col"], eng)
            if rec is None:
                print(f"  [skip] {st['col']} engine={eng}: no focus CSV")
                continue
            rec.update(stage=si, label=st["label"], group=st["group"])
            cols.append(rec)
            print(f"  {st['col']:32s} {str(eng):4s}  n={rec['n']:5d}  "
                  f"medianS/N={rec['med']:.2f}  %sig={rec['frac_sig']*100:4.1f}")
    if not cols:
        print("snr cascade: no focus CSVs found — run 'all' and 'flow' first.")
        return

    stage_ids = sorted({c["stage"] for c in cols})
    stage_x   = {s: i for i, s in enumerate(stage_ids)}
    n_stage   = len(stage_ids)
    width     = 0.34
    for c in cols:
        sib = [k for k in cols if k["stage"] == c["stage"]]
        k = len(sib); j = sib.index(c)
        c["x"] = stage_x[c["stage"]] + (j - (k - 1) / 2.0) * (width if k > 1 else 0)

    n_union = int(max(c["n"] for c in cols))
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(1.45 * n_stage + 2.5, 7.2),
        gridspec_kw=dict(height_ratios=[2.2, 1.0], hspace=0.28))

    forcing_slots = [stage_x[c["stage"]] for c in cols if c["group"] == "forcing"]
    if forcing_slots:
        xb = max(forcing_slots) + 0.5
        for ax in (axA, axB):
            ax.axvspan(-0.6, xb, color="#F2C94C", alpha=0.10, zorder=0)
            ax.axvspan(xb, n_stage - 0.4, color="#2D9CDB", alpha=0.08, zorder=0)

    for c in cols:
        v = np.clip(c["sn"], 0, CASCADE_SN_CLIP)
        parts = axA.violinplot([v], positions=[c["x"]], widths=width * 0.92,
                               showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(CASCADE_ENG_COLOR[c["engine"]]); b.set_alpha(0.55)
            b.set_edgecolor(CASCADE_ENG_COLOR[c["engine"]]); b.set_linewidth(0.6)
        axA.plot([c["x"]], [c["med"]], "o", ms=5.5, color="k", zorder=5)
        axA.annotate(f"{c['med']:.2f}", (c["x"], c["med"]),
                     textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=10.5)
    axA.axhline(1.0, color="0.25", ls="--", lw=1.1)
    # The label sits BELOW the reference line and at the LEFT edge. Below,
    # because the head room above 1.0 is now 0.1 and no longer holds a line of
    # text; left, because the upper-right corner of the panel belongs to the
    # legend. No violin reaches this corner (largest ratio 0.53), so the label
    # overlaps neither the legend nor the data.
    axA.text(-0.5, 0.97, "Ratio = 1  (bias = climate-model spread)",
             ha="left", va="top", fontsize=11, color="0.25")
    axA.set_ylim(0, CASCADE_SN_YMAX)
    axA.set_ylabel("Per-catchment bias-to-spread ratio\n"
                   f"|{em.second_label().lower()} − full| ÷ climate-model σ",
                   fontsize=13)
    axA.set_xticks(range(n_stage)); axA.set_xticklabels([])
    axA.set_xlim(-0.6, n_stage - 0.4)
    axA.tick_params(axis="y", labelsize=11)
    axA.grid(axis="y", color="0.92", lw=0.5); axA.set_axisbelow(True)
    axA.legend(handles=[Patch(facecolor=CASCADE_COL_FORCING, alpha=0.55, label="Forcing (T, P)"),
                        Patch(facecolor=CASCADE_COL_HBV, alpha=0.55, label="HBV"),
                        Patch(facecolor=CASCADE_COL_LSTM, alpha=0.55, label="LSTM")],
               fontsize=11, loc="upper right", framealpha=0.9, edgecolor="none")

    for c in cols:
        axB.bar(c["x"], c["frac_sig"] * 100, width=width * 0.92,
                color=CASCADE_ENG_COLOR[c["engine"]], alpha=0.75)
        axB.annotate(f"{c['frac_sig']*100:.0f}", (c["x"], c["frac_sig"] * 100),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=10.5)
    axB.set_ylabel("Catchments significant\n(p<0.05) (%)", fontsize=13)
    axB.set_ylim(0, 100)
    axB.set_xticks(range(n_stage))
    # The x ticks are the reader's entry point into both panels (axA hides its
    # own), so they are set larger than the y ticks (11) and the in-panel
    # annotations (10.5). 14 became comfortable once the labels shrank to
    # single symbols; the previous 11.5 was sized for the old two-line words.
    axB.set_xticklabels([CASCADE_STAGES[s]["label"] for s in stage_ids], fontsize=14)
    axB.set_xlim(-0.6, n_stage - 0.4)
    axB.tick_params(axis="y", labelsize=11)
    axB.grid(axis="y", color="0.92", lw=0.5); axB.set_axisbelow(True)

    #fig.suptitle(f"Signal-to-noise cascade of the too-hot screening  "
    #             f"({FOCUS_SCEN.upper()} {FOCUS_PER})", y=0.995, fontsize=14.5)
    add_panel_letters([axA, axB])
    os.makedirs(SIG_DIR, exist_ok=True)
    out = em.tag(os.path.join(
        SIG_DIR, f"snr_cascade_{FOCUS_SCEN}_{FOCUS_PER}_n{n_union}sta.png"))
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\ncascade figure -> {os.path.basename(out)}")


# ---------------------------------------------------------------------------
#  TCR-threshold sensitivity sweep  (folded in from fig_tcr_sensitivity)
# ---------------------------------------------------------------------------
#  Sweep the too-hot TCR cut across the AR6-plausible range and show the T->Q
#  result is threshold-robust: (a) the DETECTABILITY ordering (temperature
#  highest, hydrology lowest, nothing above S/N=1) and (b) the inflation
#  MAGNITUDE. S/N here uses the UNIFIED |full - reduced| / sigma convention (the
#  same as the rest of the paper), so as the cut rises and fewer models are
#  dropped BOTH the S/N and the inflation shrink; the robust statement is the
#  ORDERING and the below-1 masking, not a fixed value. Heavy step (station x
#  model table) is threshold-independent, built once per variable; the sweep is
#  vectorised numpy.
TCR_THRESHOLDS = np.round(np.arange(1.8, 3.001, 0.1), 2)   # K, AR6-plausible range
# ECS counterpart, used only under em.METRIC='ecs'. It runs from the AR6 best
# estimate (3.0 degC) past the *likely* upper bound (4.0, the chosen cut) to the
# *very likely* upper bound (5.0), so the cut sits mid-grid and the sweep spans
# both bounds the assessment provides. Beyond 5.0 only six models remain
# excluded and the curves are flat.
ECS_THRESHOLDS = np.round(np.arange(3.0, 5.001, 0.1), 2)   # degC
SWEEP_THRESHOLDS = ECS_THRESHOLDS if em.is_ecs() else TCR_THRESHOLDS
# figure/CSV stem: keep the historical name under TCR (report.tex \includegraphics
# points at significance/tcr_sensitivity_*.png), switch it under ECS
SWEEP_STEM = "ecs_sensitivity" if em.is_ecs() else "tcr_sensitivity"
# (label, dataset, column, engine, color, linestyle, marker)
TCR_TARGETS = [
    ("Temperature (mean)",   "climate", "temp_change_degC",          None,   "#000000", "-",  "o"),
    ("Precipitation (mean)", "climate", "precip_change_pct",         None,   "#2D9CDB", "-",  "s"),
    ("Runoff HBV",           "flow",    "delta_rel_qmean",           "hbv",  "#3B6FB6", "--", "^"),
    ("Runoff LSTM",          "flow",    "delta_rel_qmean",           "lstm", "#C24B4B", "--", "^"),
    ("Mean annual max HBV",  "flow",    "delta_rel_mean_annual_max", "hbv",  "#3B6FB6", "-",  "v"),
    ("Mean annual max LSTM", "flow",    "delta_rel_mean_annual_max", "lstm", "#C24B4B", "-",  "v"),
]


def _tcr_load(dataset):
    """Load + prepare a cache exactly as main() does, sliced to FOCUS scen/per.

    Memory-lean, like main()'s _load: the sweep needs only the TCR_TARGETS
    columns of this dataset and only the FOCUS scenario/period, so both are
    pushed down into the parquet read. The old version read EVERY column of the
    whole cache, upcast all of them to float64 and then ran a frame-wide
    .replace() — three full copies of a table that, with the registry's 41
    metrics over ~20.6k stations, does not fit the 190 G limit.
    """
    import pyarrow.parquet as pq
    cache = FLOW_CACHE if dataset == "flow" else CLIMATE_CACHE
    if not os.path.exists(cache):
        print(f"tcr sweep: cache not found: {cache}")
        return None
    schema_cols = set(pq.ParquetFile(cache).schema.names)
    ID_COLS = ["station", "lon", "lat", "scenario", "model", "member",
               "period", "engine"]
    # dict.fromkeys de-duplicates while preserving order: TCR_TARGETS lists the
    # SAME flow column once per engine (delta_rel_qmean for HBV and for LSTM),
    # so a plain comprehension yields duplicate names — and `df[want] = ...` with
    # duplicate keys raises "Columns must be same length as key". The engine
    # split happens downstream via the `engine` id column, not by column name.
    want = list(dict.fromkeys(
        c for (_, ds, c, _, _, _, _) in TCR_TARGETS
        if ds == dataset and c in schema_cols))
    need_cols = [c for c in ID_COLS if c in schema_cols] + want
    df = pd.read_parquet(
        cache, columns=need_cols,
        filters=[("scenario", "==", FOCUS_SCEN), ("period", "==", FOCUS_PER)])
    df = apply_blocklist(df, label=f"{dataset} tcr-sweep")
    df = df[df["model"].isin(FULL_MODELS)].copy()
    df["scenario"] = df["scenario"].astype(str)
    df["period"]   = df["period"].astype(str)
    for c in want:
        col = df[c].astype("float32")
        if dataset == "flow" and c.startswith("delta_rel_"):
            col = col * np.float32(100.0)          # fractions -> %
        df[c] = col
    df[want] = df[want].replace([np.inf, -np.inf], np.nan)
    return df


def _tcr_sweep_one(M, thresholds):
    """Vectorised Welch hot-vs-retained over every station (rows of M) for each
    threshold of the ACTIVE screening metric (em.METRIC). S/N and inflation both
    use |full - reduced| (the unified convention). Returns a DataFrame per
    threshold."""
    models = list(M.columns)
    # the ACTIVE metric, not _tcr: under METRIC='ecs' the sweep must cut on ECS
    # or it would sweep a threshold the screen never applies
    tcr = np.array([_metric_value(m) for m in models], float)
    Mall = M.to_numpy(float)
    sigma_all = np.nanstd(Mall, axis=1, ddof=1)          # inter-model spread (all models)
    # Under SCREEN='both' the sweep varies the UPPER bound only; the too-cold
    # models (metric < lower bound) stay excluded at every step, so the anchor
    # at thr = upper bound reproduces the two-sided reduced ensemble of the run.
    cold = (tcr < SCREEN_LO_BOUND) if em.is_two_sided() else np.zeros(len(tcr), bool)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # all-NaN / 0-dof rows
        for thr in thresholds:
            hot = (tcr > thr) | cold; ret = ~hot
            if hot.sum() < 2 or ret.sum() < 2:
                rows.append(dict(threshold=thr, n_removed=int(hot.sum()),
                                 n_valid=0, median_SN=np.nan, frac_sig=np.nan,
                                 mean_bias=np.nan))
                continue
            Mh, Mr = Mall[:, hot], Mall[:, ret]
            nh = np.sum(np.isfinite(Mh), axis=1)
            nr = np.sum(np.isfinite(Mr), axis=1)
            mh, mr = np.nanmean(Mh, axis=1), np.nanmean(Mr, axis=1)
            vh, vr = np.nanvar(Mh, axis=1, ddof=1), np.nanvar(Mr, axis=1, ddof=1)
            diff = mh - mr
            se = np.sqrt(vh / nh + vr / nr)
            frac = nh / (nh + nr)
            fmr = frac * diff                            # full - reduced = (h/N)(hot-ret)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = diff / se
                dof = (vh / nh + vr / nr) ** 2 / (
                    (vh / nh) ** 2 / (nh - 1) + (vr / nr) ** 2 / (nr - 1))
                sn = np.abs(fmr) / sigma_all             # UNIFIED: |full-reduced|/sigma
            if HAVE_SCIPY:
                p = 2.0 * sps.t.sf(np.abs(t), dof)
            else:
                p = np.array([2.0 * (1 - 0.5 * (1 + math.erf(abs(ti) / math.sqrt(2))))
                              if np.isfinite(ti) else np.nan for ti in t])
            valid = ((nh >= 2) & (nr >= 2) & np.isfinite(sigma_all)
                     & (sigma_all > 0) & np.isfinite(sn))
            rows.append(dict(
                threshold=thr, n_removed=int(hot.sum()), n_valid=int(valid.sum()),
                median_SN=float(np.nanmedian(sn[valid])) if valid.any() else np.nan,
                frac_sig=float(np.nanmean((p[valid] < ALPHA).astype(float)))
                if valid.any() else np.nan,
                mean_bias=float(np.nanmean(fmr[valid])) if valid.any() else np.nan))
    return pd.DataFrame(rows)


def make_tcr_sensitivity():
    """Render the TCR-threshold sweep. The data TABLE (CSV) goes to CACHE_DIR
    (fig_data, beside the parquet caches it reads); the FIGURE (PNG) goes to
    SIG_DIR (paper/figures/significance), like every other significance figure.
    Uses the unified |full - reduced| / sigma S/N convention."""
    if em.is_weighting():
        print("tcr_sensitivity is a removal-mode analysis (threshold sweep); "
              "skipped under method=weighting. The weighting counterpart "
              "(target-width sweep) is not implemented yet.")
        return
    if em.is_two_sided():
        # _tcr_sweep_one keeps the too-cold models (metric < lower bound)
        # excluded at every step and sweeps the UPPER bound, so the anchor at
        # the upper bound reproduces the two-sided reduced ensemble of this run.
        print(f"{SWEEP_STEM} under SCREEN='both': the lower bound is held "
              f"at {SCREEN_LO_BOUND} K while the upper bound is swept.")
    # The "chosen cut" anchor is located with np.isclose against the grid. If a
    # bound is edited to a value off the 0.1 grid (ensemble_method.py invites
    # exactly that: AR6 very-likely 2.0-5.0, or Asenjan's ECS 4.1), the anchor
    # lookup silently returns an empty frame, every "@ cut" number prints as nan
    # and the figure still draws a cut line at a threshold never evaluated.
    if not np.any(np.isclose(SWEEP_THRESHOLDS, SCREEN_CUT)):
        raise SystemExit(
            f"{SWEEP_STEM}: the chosen cut {SCREEN_CUT} is not on the sweep "
            f"grid {SWEEP_THRESHOLDS[0]}-{SWEEP_THRESHOLDS[-1]} step 0.1. "
            f"Adjust {'ECS_THRESHOLDS' if em.is_ecs() else 'TCR_THRESHOLDS'} "
            f"so the cut is one of its points.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(tcr sensitivity skipped: {e})")
        return

    print(f"{SCREEN_LABEL} threshold sweep | {FOCUS_SCEN} {FOCUS_PER} | "
          f"FULL={len(FULL_MODELS)} models | chosen cut={SCREEN_CUT} K")
    caches, results = {}, {}
    for label, dataset, col, engine, *_ in TCR_TARGETS:
        if dataset not in caches:
            caches[dataset] = _tcr_load(dataset)
        sl = caches[dataset]
        if sl is None:
            continue
        if engine is not None:
            sl = sl[sl["engine"].astype(str) == engine]
        if col not in sl.columns:
            print(f"  [skip] {label}: column {col} absent"); continue
        sl = sl.dropna(subset=[col])
        M = sl.pivot_table(index="station", columns="model", values=col, aggfunc="mean")
        if M.shape[0] < 50 or M.shape[1] < 4:
            print(f"  [skip] {label}: too few stations/models ({M.shape})"); continue
        res = _tcr_sweep_one(M, SWEEP_THRESHOLDS)
        results[label] = res
        at = res[np.isclose(res["threshold"], SCREEN_CUT)]
        sn_at = float(at["median_SN"].iloc[0]) if len(at) else np.nan
        bias_at = float(at["mean_bias"].iloc[0]) if len(at) else np.nan
        unit = "degC" if dataset == "climate" and col.startswith("temp") else "%"
        print(f"  {label:22s} stations={M.shape[0]:5d} models={M.shape[1]:2d} | "
              f"median S/N @ {SCREEN_CUT}K = {sn_at:.2f} | "
              f"inflation @ {SCREEN_CUT}K = {bias_at:+.2f} {unit}")

    if not results:
        print(f"{SCREEN_LABEL} sweep: no targets produced a sweep "
              "— check caches / columns.")
        return

    tidy = pd.concat([r.assign(target=lbl) for lbl, r in results.items()],
                     ignore_index=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SIG_DIR, exist_ok=True)
    out_csv = em.tag(os.path.join(
        CACHE_DIR, f"{SWEEP_STEM}_{FOCUS_SCEN}_{FOCUS_PER}.csv"))
    tidy.to_csv(out_csv, index=False)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.4, 9.6), sharex=True,
                                   gridspec_kw=dict(hspace=0.16))
    for label, dataset, col, engine, color, ls, mk in TCR_TARGETS:
        if label not in results:
            continue
        r = results[label]
        axA.plot(r["threshold"], r["median_SN"], marker=mk, color=color,
                 ls=ls, lw=1.8, ms=5, label=label)
    axA.axhline(1.0, color="0.25", ls="--", lw=1.2)
    axA.text(SWEEP_THRESHOLDS[-1], 1.02, "ratio = 1  (bias = climate-model spread)",
             ha="right", va="bottom", fontsize=9, color="0.25")
    axA.axvline(SCREEN_CUT, color="0.55", lw=1.0, ls=":")
    # panel letter moved out of the y-label into the bold top-left stamp below
    axA.set_ylabel(_cap("median per-catchment bias-to-spread ratio\n"
                        f"(|{SIGNCONV}| ÷ climate-model σ)"))
    axA.set_title(f"Sensitivity of the {_XHOT} result to the screening threshold\n"
                  f"({FOCUS_SCEN.upper()} {plabel(FOCUS_PER)})",
                  fontsize=10.5)
    axA.grid(color="0.92", lw=0.5); axA.set_axisbelow(True)
    axA.legend(fontsize=8.5, ncol=2, loc="upper right", framealpha=0.9)
    ref = next(iter(results.values()))
    axt = axA.secondary_xaxis("top")
    ticks = SWEEP_THRESHOLDS[::2]
    axt.set_xticks(ticks)
    axt.set_xticklabels([int(ref.loc[np.isclose(ref["threshold"], tv),
                                     "n_removed"].iloc[0]) for tv in ticks])
    axt.set_xlabel(_cap("# models removed"), fontsize=9)

    axB2 = axB.twinx()
    for label, dataset, col, engine, color, ls, mk in TCR_TARGETS:
        if label not in results:
            continue
        r = results[label]
        tgt = axB if (dataset == "climate" and col.startswith("temp")) else axB2
        # mean_bias is stored as full − reduced in the CSV; BIAS_SIGN flips
        # the DRAWN orientation to the manuscript convention.
        tgt.plot(r["threshold"], BIAS_SIGN * r["mean_bias"], marker=mk,
                 color=color, ls=ls, lw=1.8, ms=5)
    axB.axhline(0.0, color="0.6", lw=0.8)
    axB.axvline(SCREEN_CUT, color="0.55", lw=1.0, ls=":")
    axB.text(SCREEN_CUT, axB.get_ylim()[1],
             f"  chosen cut {SCREEN_CUT} K",
             va="top", ha="left", fontsize=9, color="0.4", rotation=90)
    axB.set_xlabel(_cap(f"upper {SCREEN_LABEL} threshold (K); models with "
                        f"{SCREEN_LABEL} above this "
                        + (f"or below {SCREEN_LO_BOUND} K "
                           if em.is_two_sided() else "") + "are dropped"))
    axB.set_ylabel(_cap(f"temperature bias\n{SIGNCONV} (°C)"))
    axB2.set_ylabel(_cap(f"precip / flow bias\n{SIGNCONV} (%)"), color="#3B6FB6")
    axB.grid(color="0.92", lw=0.5); axB.set_axisbelow(True)

    # axB2 is a twinx of axB (same panel), so it is deliberately NOT lettered.
    add_panel_letters([axA, axB])
    out_png = em.tag(os.path.join(
        SIG_DIR, f"{SWEEP_STEM}_{FOCUS_SCEN}_{FOCUS_PER}.png"))
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\n{SCREEN_LABEL} sensitivity figure -> {os.path.basename(out_png)}")
    print(f"{SCREEN_LABEL} sensitivity table  -> {os.path.basename(out_csv)}")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    if str(ARG).lower() in ("transmission", "attribution", "attr", "tq"):
        make_transmission_figure()
        make_transmission_figure(flow_col="delta_rel_mean_annual_max",
                                 flow_label="annmax")
        return
    if str(ARG).lower() in ("snr_cascade", "cascade"):
        make_snr_cascade()
        return
    # the ecs_* aliases run the SAME sweep; which metric it cuts on is set by
    # em.METRIC, not by the verb. They exist so an ECS run can be invoked by the
    # name it writes (ecs_sensitivity_*.csv) instead of the TCR one.
    if str(ARG).lower() in ("tcr_sensitivity", "tcr_sweep", "tcr",
                            "ecs_sensitivity", "ecs_sweep"):
        make_tcr_sensitivity()
        return
    if str(ARG).lower() in ("beats", "extra_figs"):
        make_snr_cascade()
        make_tcr_sensitivity()
        return
    if str(ARG).lower() in ("unc_decomp", "unc_combined", "uncertainty", "unc"):
        make_combined_uncertainty()
        return

    dataset = pick_dataset(ARG)
    cache = FLOW_CACHE if dataset == "flow" else CLIMATE_CACHE
    if not os.path.exists(cache):
        builder = ("build_streamflow_cache.py" if dataset == "flow"
                   else "fig02_precip_change.py")
        sys.exit(f"cache not found: {cache} (run {builder} first)")

    # --- Memory-lean load (2026-07-07). The old path read the ENTIRE cache
    #     (all ~82 columns incl. every unused delta_abs_* twin, float64, both
    #     engines resident at once) then forked 64 workers over it; CPython
    #     refcount churn defeated the copy-on-write assumption and the flow pass
    #     peaked ~200 G and OOM-hung. Three fixes:
    #       #1 read only the columns this pass needs (resolve metrics from the
    #          parquet SCHEMA first — footer metadata, no data load);
    #       #3 cast the metric columns to float32 (Welch/variance don't need f64);
    #       #4 load ONE engine at a time via a pyarrow pushdown filter and free
    #          it before the next, instead of holding the full frame + slice.
    import pyarrow.parquet as pq
    schema_cols = list(pq.ParquetFile(cache).schema.names)

    pindex = is_pindex_arg(ARG)
    if dataset == "flow":
        metrics   = resolve_flow_metrics(ARG, schema_cols)
        order_key = _flow_key
        units     = {"flow": "%", "precip_index": "%"}
        # No 'timing' group: the half-flow date (HFD) is the only timing metric
        # and Supplementary Table S6 does not list it, so the restricted figures
        # never show it.
        xlabel    = ("precipitation index" if pindex else
                     "flow metric (drought/low → central → high flow → "
                     "FDC shape → flashiness)")
        # The precipitation indices describe the bias-corrected FORCING, which
        # is identical under both hydrological engines — stats.parquet simply
        # repeats them on the HBV and the LSTM rows. Reading ONE engine keeps
        # each realization counted once; the streamflow metrics ARE engine-
        # specific, so those still get one pass per engine.
        engines   = ["hbv"] if pindex else ["hbv", "lstm"]
    else:
        metrics   = resolve_metrics(ARG, schema_cols)
        order_key = _pctile_key
        units     = {"precip": "%", "temp": "°C"}
        xlabel    = "daily-distribution percentile"
        engines   = [None]

    if not metrics:
        avail = sorted(c for c in schema_cols
                       if "change" in c or c.startswith("delta_"))
        sys.exit(f"arg {ARG!r} matches no metric in {os.path.basename(cache)}. "
                 f"Use 'precip'/'temp'/'all' (climate) or 'flow' (streamflow), "
                 f"or one of these columns: {avail}")

    metric_cols = [c for (_, _, c) in metrics]
    # id columns the compute/export/blocklist paths need: station+lon/lat (focus
    # maps), model/member/scenario/period (blocklist + variance partition),
    # engine (per-engine filter). member is unused by the stats but REQUIRED by
    # apply_blocklist, so it stays.
    ID_COLS  = ["station", "lon", "lat", "scenario", "model", "member",
                "period", "engine"]
    need_cols = ([c for c in ID_COLS if c in schema_cols]
                 + [c for c in metric_cols if c not in ID_COLS])

    def _load(engine=None):
        """Load only need_cols for one engine (or all, climate), apply the
        blocklist + FULL_MODELS filter, cast metrics to float32 (delta_rel_* ->
        percent), and NaN-out infs on the metric columns only."""
        filters = [("engine", "==", engine)] if engine is not None else None
        d = pd.read_parquet(cache, columns=need_cols, filters=filters)
        d = apply_blocklist(
            d, label=f"{dataset} significance cache"
                     f"{'' if engine is None else f' [{engine}]'}")
        d = d[d["model"].isin(FULL_MODELS)].copy()
        d["scenario"] = d["scenario"].astype(str)
        d["period"]   = d["period"].astype(str)
        # delta_rel_* are FRACTIONS ((fut-base)/base); rescale to PERCENT so flow
        # magnitudes/units match the climate %-change columns. Significance and
        # S/N are scale-invariant, so only sigma_* / med_full_minus_reduced are
        # affected. Cast every metric column to float32 (#3), in place.
        scale = np.float32(100.0)
        for c in metric_cols:
            col = d[c].astype("float32")
            if dataset == "flow" and c.startswith("delta_rel_"):
                col = col * scale
            d[c] = col
        # inf (stray relative change on a ~0 baseline) survives dropna(); NaN it
        # so the per-station tests drop it cleanly — metric columns ONLY (the id
        # columns are strings/coords and copying them whole was pure waste).
        d[metric_cols] = d[metric_cols].replace([np.inf, -np.inf], np.nan)
        return d

    print(f"\n=== Full vs {em.second_label()} significance | "
          f"dataset = {dataset} | arg = {ARG} | method = {em.method()}"
          + (f" [{em.variant()}]" if em.is_weighting() else "")
          + f" | scipy={'yes' if HAVE_SCIPY else 'no (normal approx)'} ===")
    if em.is_weighting():
        # preload the frozen weights once in the parent so the fork Pool
        # inherits them instead of re-reading the CSV in every worker
        _w = em.get_weights()
        print(f"frozen weights: {len(_w)} models from model_weights_wt.csv "
              f"(column {em.variant()}), N_eff = "
              f"{em.n_eff_present(_w.keys()):.2f}")
    print(f"cache: {cache}")
    print(f"loading {len(need_cols)} of {len(schema_cols)} columns "
          f"(float32 metrics; one engine at a time)")
    print(f"FULL={len(FULL_MODELS)} models  "
          f"EXCLUDED={len(EXCLUDED_MODELS)} ({em.screen_expr()})  "
          f"RETAINED={len(RETAINED_MODELS)}")
    print(f"excluded models ({em.screen()} screen): {sorted(EXCLUDED_MODELS)}")
    print("analysing columns: "
          + ", ".join(f"{v}/{lab}={c}" for v, lab, c in metrics))
    if dataset == "flow":
        print(f"engines (separate figures/CSVs each): {engines}")
    print()

    sm_by_engine  = {}
    station_union = set()
    for engine in engines:
        de = _load(engine)
        if de.empty:
            print(f"(engine {engine}: no rows after filter/blocklist — skipped)")
            continue
        # Derive scenarios/periods from THIS engine's own coverage. The old code
        # passed the both-engine union to each pass, but analyze_pass skips any
        # (col, scen, per) cell with no rows — so a per-engine list yields the
        # identical set of non-empty cells (hence identical results), without
        # holding a combined frame.
        scenarios = [s for s in
                     ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]
                     if s in set(de["scenario"])]
        periods = sorted(set(de["period"]))
        station_union |= set(de["station"].unique())
        if engine is None:
            # make_uncertainty=False: the forcing unc_decomp_*.png is retired,
            # superseded by the combined 2x2 figure (run `... unc_decomp`).
            analyze_pass(de, metrics, scenarios, periods, order_key, units,
                         xlabel, suffix="", engine_label=None,
                         make_uncertainty=False)
        elif pindex:
            # forcing indices: ONE pass, its own suffix so the files are not
            # mistaken for streamflow output, and its own spectrum figure (there
            # is no second engine to combine with).
            print(f"\n========== PRECIPITATION INDICES "
                  f"({de['station'].nunique()} stations, read from the "
                  f"{engine} rows) ==========")
            analyze_pass(de, metrics, scenarios, periods, order_key, units,
                         xlabel, suffix="_pindex", engine_label="forcing",
                         make_uncertainty=False, make_spectrum=True)
        else:
            print(f"\n========== ENGINE: {engine} "
                  f"({de['station'].nunique()} stations) ==========")
            # per-engine unc_decomp / spectrum_forcing are NOT written for flow
            # — they're superseded by the combined two-engine figures below.
            sm_by_engine[engine] = analyze_pass(
                de, metrics, scenarios, periods, order_key, units,
                xlabel, suffix=f"_{engine}", engine_label=engine,
                make_uncertainty=False, make_spectrum=False)
        del de   # free this engine before loading the next (#4)

    # flow: stack the per-engine significance/S-N spectra into ONE 2-row figure
    # (HBV top, LSTM bottom) and the per-engine uncertainty decompositions into
    # ONE 2-panel figure (HBV left, LSTM right), for the paper — on top of the
    # per-engine files above. ONLY for the full 'flow' set: an explicit
    # single-column run (e.g. `... delta_rel_flood_q20`) must not overwrite the
    # paper's all-metric spectrum/uncertainty figures with a one-metric version.
    if (dataset == "flow" and len(sm_by_engine) >= 2
            and str(ARG).lower() in FLOW_KEYWORDS):
        nsta_comb = len(station_union)   # union across engines (no full frame held)
        make_combined_flow_spectrum(sm_by_engine, order_key=order_key,
                                    xlabel=xlabel, nsta=nsta_comb)
        # companion figure: the 14 registry indices Table S6 does not list
        make_combined_flow_spectrum(sm_by_engine, order_key=order_key,
                                    xlabel=xlabel, nsta=nsta_comb,
                                    labels=OFF_TABLE_S6_LABELS,
                                    out_stem="spectrum_hydrindices_other")
        # unc_decomp_flow_*.png retired: superseded by the combined 2x2 figure
        # (row 2 = flow). Run `... unc_decomp` to build it.
        # make_combined_flow_uncertainty(sm_by_engine, nsta=nsta_comb)


if __name__ == "__main__":
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment()
    main()
