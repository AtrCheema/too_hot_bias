"""
Figure 1 + Table 1 — CMIP6 model screening.

Produces, from a single station's mean_deltas.nc:
  * figures/fig01_model_screening.png — the SCREENED sensitivity metric
    (em.METRIC, TCR by default) vs the 1980-2014 warming trend, coloured by
    the other metric (in the style of tcr_vs_atr.webp). Under em.METRIC='ecs'
    the y-axis becomes ECS and the colour becomes TCR, so the cut being
    applied is always the visible axis; the output is tagged '_ecs'.
  * too_hot_models_table.tex          — LaTeX table of models with TCR, ECS,
    and ensemble-member counts per scenario (set WRITE_CSV = True below to
    also emit the .csv version).

Reproduces the style of tcr_vs_atr.webp but:
  * the model set is whatever is present in ONE station's mean_deltas.nc
    (`realization` coord = "{scenario}_{model}_{member}"). The realization
    count is the same across stations, so a single file is enough. This is
    the full list of models in the study (CONFIG.models is only a wishlist;
    many of its entries never had downloadable data).
  * TCR / ECS / 1980-2014 trend are LITERATURE values (they cannot be computed
    from the delta files). Edit the METRICS table below to match the exact
    source you cite for a pixel-exact reproduction.

Optionally pass a station name or a path to a specific mean_deltas.nc:
    python paper/fig01_model_screening.py
    python paper/fig01_model_screening.py Arcticnet__49052
    python paper/fig01_model_screening.py /path/to/mean_deltas.nc

Sources for the embedded numbers:
  ECS, TCR : Meehl et al. (2020) Sci. Adv. 6, eaba1981 ;
             Zelinka et al. (2020) GRL 47, e2019GL085782 ; IPCC AR6 WG1.
  1980-2014 near-surface air-temperature trend : Tokarska et al. (2020)
             Sci. Adv. 6, eaaz9549 (Table S1) / IPCC AR6 WG1 consistent.

Run (python only, as you require):
    python paper/fig01_model_screening.py
"""

import os
import sys
import csv
import glob
import logging

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# this file now lives in paper/figures/scripts/; the repo root (config.py and the
# shared libs) is three dirs up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import CONFIG

# House font style (single typeface + sizes shared by every figure in the paper).
# Importing figstyle applies it; apply_style() is called explicitly so the import
# does not look unused and get stripped. This figure is single-panel, so it takes
# no panel letter -- Nature letters multi-panel figures only.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import apply_style
apply_style()
# Treatment switch (SCREEN/METHOD/VARIANT) and the AR6 TCR bounds. No import
# cycle: ensemble_method imports THIS module lazily, inside its functions.
import ensemble_method as em

# ---------------------------------------------------------------------------
# Assessed ranges drawn as shaded bands (edit to match your cited assessment)
# ---------------------------------------------------------------------------
# Observed 1980-2014 global-mean near-surface air-temperature trend, from Beck
# et al. (2023, Sci. Data 10:724), section "Constraining CMIP6 projections":
# the ensemble-mean trend over the 200 HadCRUT5 (v5.0.1.0) members is
# 0.1851 degC/decade, and their Eq. (1) combines the observational structural
# uncertainty with internal variability into a *likely* range (68% CI) of
# 0.130-0.241 degC/decade. Do NOT edit these without re-reading that section.
OBS_TREND      = 0.185      # best estimate of observed 1980-2014 trend (degC/decade)
OBS_TREND_LO   = 0.130      # lower bound of the likely observed-trend range (68% CI)
OBS_TREND_HI   = 0.241      # upper bound of the likely observed-trend range (68% CI)

# These three keep their TCR meaning whatever em.METRIC is active: they are
# imported by fig_tcr_attribution.py, whose panels are TCR-intrinsic.
TCR_BEST       = 1.80          # IPCC AR6 central TCR estimate (degC)
TCR_LIKELY_LO  = em.TCR_LOWER  # AR6 *likely* range lower bound (66% CI)
TCR_LIKELY_HI  = em.TCR_UPPER  # AR6 *likely* range upper bound (66% CI)

ECS_BEST       = 3.00          # IPCC AR6 central ECS estimate (degC)
ECS_LIKELY_LO  = em.ECS_LOWER  # AR6 *likely* range lower bound (66% CI)
ECS_LIKELY_HI  = em.ECS_UPPER  # AR6 *likely* range upper bound (66% CI)

CMAP           = "RdYlGn_r" # green (low sensitivity) -> red (high), as in reference
# vmin must sit at or below the SMALLEST value plotted: Normalize has no clip
# handling here, so a model below vmin is drawn in the same colour as one at
# vmin. INM-CM4-8 (ECS 1.89) was previously rendered identically to an ECS=2.00
# model because vmin was 2.0; 1.8 fixes that. vmax stays 6.0: the plotted models
# top out at ECS 5.79, so raising it would only recolour the figure for no gain.
# (METRICS itself reaches 6.66 at CESM2-FV2, which has no daily data and so is
# never plotted; if that ever changes, raise ECS_VMAX to 6.7.)
ECS_VMIN, ECS_VMAX = 1.8, 6.0
TCR_VMIN, TCR_VMAX = 1.2, 3.0

# ---------------------------------------------------------------------------
# The ACTIVE screening metric (em.METRIC)
# ---------------------------------------------------------------------------
# The panel puts the SCREENED metric on y and colours the markers by the other
# one, so that the cut being applied is always the visible axis. Under
# em.METRIC='tcr' (the default) every value resolved below is exactly the TCR
# constant it replaces, so the figure and the table are unchanged.
_AXIS = {
    "tcr": dict(idx=1, best=TCR_BEST, lo=TCR_LIKELY_LO, hi=TCR_LIKELY_HI,
                axis_label="Transient Climate Response (TCR; °C)",
                cbar_label="Transient Climate Response (TCR; °C)",
                ylim=(1.18, 3.05), vmin=TCR_VMIN, vmax=TCR_VMAX,
                thr_pad=0.025),
    "ecs": dict(idx=0, best=ECS_BEST, lo=ECS_LIKELY_LO, hi=ECS_LIKELY_HI,
                axis_label="Equilibrium Climate Sensitivity (ECS; °C)",
                cbar_label="Equilibrium Climate Sensitivity (ECS; °C)",
                # upper limit covers the whole METRICS table (max 6.66,
                # CESM2-FV2), not just the 1.89-5.79 spanned by the models that
                # currently have daily data, so a newly downloaded model cannot
                # be plotted outside the panel
                ylim=(1.70, 6.80), vmin=ECS_VMIN, vmax=ECS_VMAX,
                thr_pad=0.06),
}
SCREEN_METRIC = em.metric()             # 'tcr' | 'ecs'
SCREEN_LABEL  = em.metric_label()       # 'TCR' | 'ECS'
SCREEN_LO, SCREEN_HI = em.bounds()      # bounds of the ACTIVE metric
Y_AXIS = _AXIS[SCREEN_METRIC]                                  # screened -> y
C_AXIS = _AXIS["ecs" if SCREEN_METRIC == "tcr" else "tcr"]     # other -> colour

# screen_tag, not tag: Fig. 1 and Table 1 describe the model universe and the
# screen, and are byte-identical under removal and weighting. A "_wt" copy
# would leave supplement.tex \input{}ing a stale untagged table.
OUT_PNG = em.screen_tag(os.path.join(os.getcwd(), "paper", "figures",
                                     "fig01_model_screening.png"))

# ---------------------------------------------------------------------------
# Model-label placement
# ---------------------------------------------------------------------------
# 44 model names have to sit in one crowded scatter. The placement is done by
# textalloc, a candidate-based allocator: for every label it builds a few
# hundred candidate boxes AROUND ITS OWN MARKER and keeps the first that
# collides with no other label, no marker and no other leader line.
#
# The two properties that matter here, and that the force-based relaxation used
# before (adjustText) does not offer, are
#   * a hard bound on how far a label may travel from its marker
#     (LABEL_MAX_DIST, as a fraction of the axes span), and
#   * a constraint that leader lines may not cross each other.
# Without them labels drifted to the opposite side of the panel and their
# leader lines crossed both other leaders and other markers.
#
# LABEL_MAX_DIST is deliberately generous (0.22, was 0.12): candidates are
# searched nearest-first, so isolated markers keep their label adjacent; only
# the congested mid-panel cluster uses the extra travel to spill into the
# surrounding empty space instead of piling every label onto the data cloud.
FIGSIZE          = (12.0, 8.0)
LABEL_FONTSIZE   = 10
LABEL_MAX_DIST   = 0.22    # max marker-to-label distance, fraction of axes span
LABEL_MIN_DIST   = 0.012   # min marker-to-label gap, fraction of axes span
LABEL_MARGIN     = 0.006   # extra clearance kept around every box
LABEL_CANDIDATES = 800     # candidate boxes searched per label

# ---------------------------------------------------------------------------
# Table 1 output paths and column configuration
# ---------------------------------------------------------------------------
# write into paper/tables/ — the copy that main.tex and supplement.tex \input
CSV_PATH = em.screen_tag(os.path.join(os.getcwd(), "paper", "tables",
                                      "too_hot_models_table.csv"))
TEX_PATH = em.screen_tag(os.path.join(os.getcwd(), "paper", "tables",
                                      "too_hot_models_table.tex"))

SCENARIO_COLS = ["ssp126", "ssp245", "ssp370", "ssp585"]
HIST_COL      = None           # set to None to drop the historical column
SORT_BY       = "model"        # "model" (A->Z) or "tcr" (high TCR first)
ADD_TOTAL     = True           # append a per-column TOTAL row
ADD_COUNTS_ROW = True          # append a "Models available (full/reduced)" model-count row
# K; kept as a name because model_weights.py imports it. The reduced ensemble
# keeps TCR <= this, and additionally TCR >= em.TCR_LOWER when em.SCREEN='both'.
TOO_HOT_TCR_THRESHOLD = em.TCR_UPPER
MISSING       = "--"           # placeholder when TCR/ECS is not in METRICS
WRITE_CSV     = False          # also write the CSV alongside the LaTeX table
MAX_MEMBERS_DISPLAY = 20       # cap per-cell member counts (None to disable);
                               # apply_deltas_stn.py has an off-by-one that
                               # lets 21 members through when CONFIG.max_members_per_model=20

# pretty SSP labels for the LaTeX header
SCEN_LABEL = {
    "historical": "Historical",
    "ssp119": "SSP1-1.9", "ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0", "ssp434": "SSP4-3.4", "ssp460": "SSP4-6.0",
    "ssp585": "SSP5-8.5",
}

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Literature climate-sensitivity table:  model -> (ECS degC, TCR degC, trend degC/dec)
# This is the ONLY place to edit for an exact match to your cited source.
# Models not present here are plotted at the median and flagged in the log.
# ---------------------------------------------------------------------------
check_tcr = {
    'CNRM-CM6-1': {"beck": 2.25, "tokarska": 2.13},
    'KACE-1-0-G': {"beck": 2.56, "meehl": 1.4}
}

METRICS = {               # ECS, TCR, 1980-2014 trend (degC/decade)
    "ACCESS-CM2":        (5.51, 1.97, 0.279),  # Beck, Beck, Beck
    "ACCESS-ESM1-5":     (4.91, 2.04, 0.258),  # Beck, Beck, Beck
    "AWI-CM-1-1-MR":     (3.2, 2.0, 0.239),  # Meehl, Meehl, Beck
    "AWI-ESM-1-1-LR":    (3.16, 2.03, 0.273),  # %,    Beck, Beck
    "BCC-CSM2-MR":       (3.58, 1.58, 0.213),  # Beck, Beck, %
    "BCC-ESM1":          (3.23, 1.73, 0.286),  # tokarska, tokarska, %
    "CAMS-CSM1-0":       (2.29, 1.75, 0.152),  # tokarska, tokarska, %
    "CanESM5":           (5.79, 2.70, 0.381),  # Beck, Beck, Beck
    "CanESM5-1":         (4.98, 2.38, 0.357),  # Beck, Beck, Beck
    "CanESM5-CanOE":     (5.62, 2.63, 0.358),  #   %  Beck,  Beck
    "CAS-ESM2-0":        (3.67, 2.23, 0.252),  # Beck, Beck, Beck
    "CESM2":             (5.19, 2.06, 0.252),  # tokarska, tokarska, %
    "CESM2-FV2":         (6.66, 2.03, 0.268),  # Beck, Beck, Beck
    "CESM2-WACCM":       (4.70, 1.98, 0.301),  # tokarska, tokarska, %
    "CESM2-WACCM-FV2":   (5.60, 2.08, 0.285),  # Beck, Beck, Beck
    "CIESM":             (5.26, 2.41, 0.261),  # Beck, Beck, Beck
    "CMCC-CM2-HR4":      (3.88, 2.05, 0.208),  #  %,  %,   Beck
    "CMCC-CM2-SR5":      (3.50, 2.20, 0.239),  # Beck, Beck, Beck
    "CMCC-ESM2":         (3.53, 1.99, 0.222),  # Beck, Beck, Beck
    "CNRM-CM6-1":        (4.68, 2.25, 0.212),  # Beck, Beck, Beck
    "CNRM-CM6-1-HR":     (4.10, 2.49, 0.211),  # Beck, Beck, Beck
    "CNRM-ESM2-1":       (4.87, 1.86, 0.189),  # Beck, Beck, Beck
    "E3SM-1-0":          (5.32, 2.99, 0.322),  # tokarska, tokarska, %
    "E3SM-1-1":          (5.32, 2.90, 0.309),  #   %,   %,   Beck
    "E3SM-1-1-ECA":      (5.32, 2.90, 0.295),  #  %,   %,   Beck
    "E3SM-2-0":          (4.12, 2.50, 0.248),  # Beck, Beck, Beck
    "EC-Earth3":         (4.21, 2.66, 0.261),  # Beck, Beck, Beck
    "EC-Earth3-AerChem": (3.89, 2.15, 0.310),  # Beck, Beck, Beck
    "EC-Earth3-CC":      (4.19, 2.71, 0.279),  # Beck, Beck, Beck
    "EC-Earth3-Veg":     (4.43, 2.70, 0.269),  # Beck, Beck, Beck
    "EC-Earth3-Veg-LR":  (4.37, 2.45, 0.260),  # Beck, Beck, Beck
    "FGOALS-f3-L":       (2.99, 2.09, 0.205),  # tokarska, tokarska, %
    "FGOALS-g3":         (2.84, 1.40, 0.199),  # Beck, Beck, Beck
    "FIO-ESM-2-0":       (4.71, 2.21, 0.249),  # Beck, Beck, Beck
    "GFDL-CM4":          (4.36, 2.09, 0.286),  # Beck, Beck, Beck
    "GFDL-ESM4":         (2.62, 1.61, 0.211),  # tokarska, tokarska, %
    "GISS-E2-1-G":       (2.72, 1.76, 0.231),   # tokarska, tokarska, tokarska
    "GISS-E2-1-G-CC":    (2.72, 1.80, 0.235),  #  %,   %,   Beck
    "GISS-E2-1-H":       (3.11, 1.92, 0.252),   # tokarska, tokarska, %
    "GISS-E2-2-G":       (2.4, 1.7, 0.205),    # Meehl, Meehl, %
    "GISS-E2-2-H":       (3.00, 1.87, 0.220),  #  %,  Beck,  Beck
    "HadGEM3-GC31-LL":   (5.65, 2.44, 0.326),  # Beck, Beck, Beck
    "HadGEM3-GC31-MM":   (5.30, 2.66, 0.279),  # Beck, Beck, Beck
    "IITM-ESM":          (2.37, 1.7, 0.150),   #  %  Meehl, %
    "INM-CM4-8":         (1.89, 1.33, 0.212),  # Beck, Beck, Beck
    "INM-CM5-0":         (2.05, 1.39, 0.210),  # Beck, Beck, Beck
    "IPSL-CM5A2-INCA":   (4.09, 2.0, 0.265),   # Beck, Beck, Beck
    "IPSL-CM6A-LR":      (4.90, 2.36, 0.250),  # Beck, Beck, Beck
    "IPSL-CM6A-LR-INCA": (4.56, 2.32, 0.247),  #  %,  %,  Beck
    "KACE-1-0-G":        (5.55, 2.56, 0.284),   # Beck, Beck, Beck
    "KIOST-ESM":         (4.63, 1.78, 0.271),  # Beck,  %,  Beck
    "MCM-UA-1-0":        (3.7, 1.9, 0.205),   # Meehl, Meehl, %
    "MIROC6":            (2.53, 1.58, 0.177),  # Beck, Beck, Beck
    "MIROC-ES2H":        (3.10, 1.80, 0.187),  #  %,  %,  Beck
    "MIROC-ES2L":        (2.55, 1.52, 0.201),   # Beck, Beck, Beck
    "MPI-ESM-1-2-HAM":   (3.15, 1.71, 0.227),  # Beck, Beck, Beck
    "MPI-ESM1-2-HR":     (3.30, 1.67, 0.206),  # Beck, Beck, Beck
    "MPI-ESM1-2-LR":     (3.06, 1.86, 0.203),  # Beck, Beck, Beck
    "MRI-ESM2-0":        (3.45, 1.67, 0.214),  # Beck, Beck, Beck
    "NESM3":             (4.61, 2.58, 0.302),  # Beck, Beck, Beck
    "NorCPM1":           (3.05, 1.56, 0.196),  # %, tokarska, tokarska
    "NorESM2-LM":        (2.89, 1.54, 0.189),  # Beck, Beck, Beck
    "NorESM2-MM":        (2.78, 1.25, 0.167),  # Beck, Beck, Beck
    "SAM0-UNICON":       (3.72, 2.26, 0.252),  # tokarska, tokarska, %
    "TaiESM1":           (4.65, 2.29, 0.283),  # Beck, Beck, Beck
    "UKESM1-0-LL":       (5.54, 2.86, 0.340),  # Beck, Beck, Beck
    "UKESM1-1-LL":       (5.36, 2.60, 0.285),
}


def _resolve_mean_deltas_file(arg=None):
    """Pick ONE mean_deltas.nc: explicit path, station name, or first found."""
    if arg:
        if os.path.isfile(arg):
            return arg
        cand = os.path.join(CONFIG.out_path, CONFIG.delta_method, arg, "mean_deltas.nc")
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"No mean_deltas.nc for argument {arg!r}")
    files = sorted(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "mean_deltas.nc")))
    if not files:
        raise FileNotFoundError(f"No mean_deltas.nc found under {CONFIG.out_path}/{CONFIG.delta_method}/")
    return files[0]


def collect_members(arg=None):
    """Read ONE mean_deltas.nc; return ({model: {scenario: set(members)}}, n_real)."""
    f = _resolve_mean_deltas_file(arg)
    ds = xr.open_dataset(f, engine="netcdf4")
    if "realization" not in ds.coords and "realization" not in ds.dims:
        ds.close()
        raise KeyError(f"{f} has no 'realization' coord/dim")
    reals = [str(r) for r in np.atleast_1d(ds["realization"].values)]
    ds.close()

    table, other_scen = {}, set()
    for r in reals:
        parts = r.split("_", 2)              # scenario _ model _ member
        if len(parts) != 3:
            logger.warning(f"skipping malformed realization {r!r}")
            continue
        scen, model, member = parts
        table.setdefault(model, {}).setdefault(scen, set()).add(member)
        if scen not in SCENARIO_COLS:
            other_scen.add(scen)

    logger.info(f"Using {f}")
    logger.info(f"{len(reals)} realizations, {len(table)} models")
    if other_scen:
        logger.warning("realizations present for scenarios NOT shown as "
                       f"columns (excluded from the table): {sorted(other_scen)}")
    for s in SCENARIO_COLS:
        if not any(s in v for v in table.values()):
            logger.warning(f"no realizations found for requested scenario {s!r} "
                           "(column will be all zeros)")
    return table, len(reals)


def models_in_results(arg=None):
    """Convenience: sorted unique model names in one station's mean_deltas.nc."""
    table, _ = collect_members(arg)
    return sorted(table.keys())


# ---------------------------------------------------------------------------
# Table 1 builders / writers (formerly in paper/create_table1.py)
# ---------------------------------------------------------------------------
def _count_cols():
    return ([HIST_COL] if HIST_COL else []) + SCENARIO_COLS


def _fnum(x):
    return MISSING if x is None else f"{x:.2f}"


def _tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def build_rows(table):
    rows, missing_metrics = [], []
    for model in sorted(table):
        scen_members = table[model]
        if model in METRICS:
            ecs, tcr, _trend = METRICS[model]
        else:
            ecs = tcr = None
            missing_metrics.append(model)
        def _cap(n):
            return min(n, MAX_MEMBERS_DISPLAY) if MAX_MEMBERS_DISPLAY else n

        row = {"model": model, "tcr": tcr, "ecs": ecs}
        if HIST_COL:
            # distinct members across ALL of this model's SSP realizations
            row[HIST_COL] = _cap(len(set().union(*scen_members.values()))
                                 if scen_members else 0)
        for s in SCENARIO_COLS:
            row[s] = _cap(len(scen_members.get(s, set())))
        rows.append(row)

    if SORT_BY == "tcr":
        rows.sort(key=lambda d: (-(d["tcr"] if d["tcr"] is not None else -1.0),
                                 d["model"]))

    if ADD_TOTAL and rows:
        total = {"model": "Total", "tcr": None, "ecs": None}
        for c in ([HIST_COL] if HIST_COL else []) + SCENARIO_COLS:
            total[c] = sum(r[c] for r in rows)
        rows.append(total)
    if ADD_COUNTS_ROW and table:
        # per scenario: number of models available (full ensemble) and the number
        # surviving the ACTIVE screen (em.METRIC/em.SCREEN), as "full/reduced"
        counts = {"model": "Models available (full/reduced)",
                  "tcr": None, "ecs": None}
        if HIST_COL:
            counts[HIST_COL] = MISSING
        for s in SCENARIO_COLS:
            present = [m for m in table if len(table[m].get(s, set())) > 0]
            # follows em.SCREEN: hot-only, or hot+cold under SCREEN='both'
            reduced = sum(1 for m in present
                          if m in METRICS and not em.is_excluded(m))
            counts[s] = f"{len(present)}/{reduced}"
        rows.append(counts)
    return rows, missing_metrics


def _screen_caption():
    """TeX description of the ACTIVE screen, for the Table 1 caption."""
    if em.is_two_sided():
        return (rf"{SCREEN_LABEL}\,$\le$\,{SCREEN_HI}\,$^{{\circ}}$C and "
                rf"{SCREEN_LABEL}\,$\ge$\,{SCREEN_LO}\,$^{{\circ}}$C")
    return rf"{SCREEN_LABEL}\,$\le$\,{SCREEN_HI}\,$^{{\circ}}$C"


def _screen_annotation():
    """Plain-text description of the ACTIVE screen, for the Fig. 1 panel."""
    if em.is_two_sided():
        return (f"Screening thresholds ({SCREEN_LABEL} = {SCREEN_LO:.1f} "
                f"and {SCREEN_HI:.1f} °C)")
    return "AR6 likely upper bound"


def write_csv(rows, path):
    cols = _count_cols()
    header = ["model", "TCR_K", "ECS_K"] + cols
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([r["model"], _fnum(r["tcr"]), _fnum(r["ecs"])]
                       + [r[c] for c in cols])
    logger.info(f"wrote {path}")


def write_tex(rows, path, src_basename):
    cols = _count_cols()
    colspec = "l r r " + " ".join("r" for _ in cols)
    head = ["Model", r"TCR ($^{\circ}$C)", r"ECS ($^{\circ}$C)"] + \
           [SCEN_LABEL.get(c, c) for c in cols]

    lines = [
        f"% Auto-generated by fig01_model_screening.py from {src_basename}.",
        "% Do not edit by hand; re-run the script instead.",
        r"\begin{table}[!ht]",
        r"    \centering",
        r"    \small",
        r"    \caption{CMIP6 models used in this study with their transient "
        r"climate response (TCR) and equilibrium climate sensitivity (ECS) "
        r"compiled by Beck et al.\ (2023, their Table~3), and the number of "
        r"ensemble members per scenario. The final row gives, per scenario, "
        r"the number of models available (full ensemble) and the number "
        rf"retained after the {_screen_caption()} screen (reduced ensemble).}}",
        r"    \label{tab:model_screening}",
        rf"    \begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        # first three headers are literal TeX (degree symbol); only the
        # scenario labels come from data and need escaping
        "    " + " & ".join(head[:3] + [_tex_escape(h) for h in head[3:]])
        + r" \\",
        r"    \midrule",
    ]
    for r in rows:
        if r["model"] == "Total":
            lines.append(r"    \midrule")
        cells = ([r"\textbf{Total}" if r["model"] == "Total"
                  else _tex_escape(r["model"]),
                  _fnum(r["tcr"]), _fnum(r["ecs"])]
                 + [str(r[c]) for c in cols])
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"    \end{tabular}", r"\end{table}", ""]

    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    logger.info(f"wrote {path}")


def print_table_stdout(rows):
    cols = _count_cols()
    head = ["Model", "TCR", "ECS"] + [SCEN_LABEL.get(c, c) for c in cols]
    body = [[r["model"], _fnum(r["tcr"]), _fnum(r["ecs"])]
            + [str(r[c]) for c in cols] for r in rows]
    widths = [max(len(head[i]), *(len(b[i]) for b in body))
              for i in range(len(head))]
    fmt = lambda parts: "  ".join(  # noqa: E731
        p.ljust(widths[i]) if i == 0 else p.rjust(widths[i])
        for i, p in enumerate(parts))
    print()
    print(fmt(head))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for b in body:
        print(fmt(b))
    print()


def place_labels(ax, xs, ys, names, scatter=None,
                 x_obstacles=None, y_obstacles=None,
                 fontsize=LABEL_FONTSIZE,
                 max_distance=LABEL_MAX_DIST,
                 min_distance=LABEL_MIN_DIST,
                 margin=LABEL_MARGIN,
                 nbr_candidates=LABEL_CANDIDATES):
    """Draw `names` at (xs, ys) without overlaps, with short leader lines.

    Uses textalloc (candidate-based allocation, see the note beside
    LABEL_MAX_DIST). `scatter` is the PathCollection returned by ax.scatter;
    passing it lets the allocator use the true marker bounding boxes as
    obstacles instead of the bare point coordinates. `x_obstacles`/`y_obstacles`
    are extra keep-out points in data coordinates (the sampled boxes of the
    in-panel range captions); the adjustText fallback ignores them.

    The axes must already carry its final limits, colourbar and layout when
    this is called: the allocator measures every box in display coordinates,
    so a later tight_layout() would invalidate the solution.

    Returns the list of model names that could not be placed without an
    overlap (drawn on top of their own marker as a last resort).
    """
    try:
        import textalloc
    except ImportError:
        return _place_labels_fallback(ax, xs, ys, names, fontsize)

    _, lines, _, _ = textalloc.allocate(
        ax, list(xs), list(ys), list(names),
        scatter_plot=scatter,
        x_scatter=x_obstacles, y_scatter=y_obstacles,
        textsize=fontsize, textcolor="0.15",
        margin=margin, min_distance=min_distance, max_distance=max_distance,
        nbr_candidates=nbr_candidates,
        draw_lines=True, linecolor="0.55", linewidth=0.6,
        draw_all=True,                   # never silently drop a model
        avoid_label_lines_overlap=True,  # a leader may not run through a label
        avoid_crossing_label_lines=True, # leaders may not cross each other
        priority_strategy="largest",     # longest names get first pick
        plot_kwargs=dict(zorder=2),
        zorder=4,
    )
    # allocate() leaves result_lines[i] = None for a label it had to fall back
    # on, i.e. one that is sitting on its own marker.
    return [n for n, ln in zip(names, lines) if ln is None]


def _place_labels_fallback(ax, xs, ys, names, fontsize):
    """adjustText fallback for an environment without textalloc.

    Force-based relaxation: weaker than the candidate search (it bounds neither
    the travel distance nor the leader-line crossings), so the parameters here
    keep labels deliberately close to their markers rather than letting them
    spread across the panel.
    """
    texts = [ax.text(x, y, n, fontsize=fontsize, color="0.15", zorder=4)
             for x, y, n in zip(xs, ys, names)]
    try:
        from adjustText import adjust_text
    except ImportError:
        logger.warning("neither textalloc nor adjustText is installed; "
                       "labels will overlap")
        return list(names)
    adjust_text(
        texts, x=list(xs), y=list(ys), ax=ax,
        expand=(1.15, 1.3), force_text=(0.3, 0.5),
        force_static=(0.2, 0.4), force_pull=(0.05, 0.05),
        min_arrow_len=4,
        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6,
                        shrinkA=4, shrinkB=4),
    )
    return []


def _annotate_ranges(ax):
    """Caption the shaded assessed ranges and mark the screening threshold of
    the ACTIVE metric (em.METRIC).

    The trend-sensitivity correlation leaves the top-left and bottom-right of
    the panel without markers, so the captions live there and the shading is
    readable without the figure caption. Must run AFTER the axes limits are
    final (positions derive from them) and BEFORE place_labels (the returned
    Text objects are handed to the allocator as obstacles).
    """
    right = ax.get_xlim()[1] - 0.006
    top   = ax.get_ylim()[1] - 0.03

    # the too-hot threshold coincides with the upper edge of the AR6 band; a
    # two-sided screen (em.SCREEN='both') also cuts at the lower edge
    ax.axhline(SCREEN_HI, ls=(0, (5, 3)), color="0.25", lw=1.2, zorder=1.5)
    if em.is_two_sided():
        ax.axhline(SCREEN_LO, ls=(0, (5, 3)), color="0.25", lw=1.2,
                   zorder=1.5)
    return [
        ax.text(OBS_TREND, top, "Observed trend\n(68% CI)",
                ha="center", va="top", fontsize=12, color="#8a7d00",
                linespacing=1.3, zorder=5),
        ax.text(right, (Y_AXIS["lo"] + Y_AXIS["best"]) / 2.0,
                f"IPCC AR6 likely\n{SCREEN_LABEL} range",
                ha="right", va="center", fontsize=12, color="0.35",
                linespacing=1.3, zorder=5),
        ax.text(right, SCREEN_HI + Y_AXIS["thr_pad"],
                _screen_annotation(),
                ha="right", va="bottom", fontsize=12, color="0.25", zorder=5),
    ]


def _text_obstacle_points(fig, ax, texts, step_px=8):
    """Sample each Text's display box into data-coordinate points.

    textalloc takes point obstacles (x_scatter/y_scatter), not text boxes, so
    each caption's bounding box is rasterised into a grid of points at most
    `step_px` display-pixels apart -- dense enough that no candidate label box
    can slip between two samples. Requires a drawn canvas (fig.canvas.draw()).
    """
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    pts = []
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        nx = int(np.ceil(bb.width / step_px)) + 2
        ny = int(np.ceil(bb.height / step_px)) + 2
        gx, gy = np.meshgrid(np.linspace(bb.x0, bb.x1, nx),
                             np.linspace(bb.y0, bb.y1, ny))
        pts.append(inv.transform(np.column_stack([gx.ravel(), gy.ravel()])))
    xy = np.vstack(pts)
    return xy[:, 0].tolist(), xy[:, 1].tolist()


def draw_figure(models, out_png=OUT_PNG, figsize=FIGSIZE, **label_kwargs):
    """Screened metric (em.METRIC: TCR by default) against the 1980-2014
    warming trend, coloured by the OTHER sensitivity metric. Returns missing
    models."""
    fig, ax = plt.subplots(figsize=figsize)

    # ---- assessed-range shading (drawn first, behind points) --------------
    ax.axhspan(Y_AXIS["lo"], Y_AXIS["hi"], color="0.55", alpha=0.20, lw=0)
    ax.axvspan(OBS_TREND_LO, OBS_TREND_HI, color="#d9c84a", alpha=0.18, lw=0)
    ax.axhline(Y_AXIS["best"], ls=":", color="navy", lw=1.2, alpha=0.8)
    # ymax keeps the dotted best-estimate line from striking through the
    # in-band caption that _annotate_ranges puts at the top of the band
    ax.axvline(OBS_TREND, ymax=0.90, ls=":", color="#8a7d00", lw=1.2, alpha=0.8)

    norm = Normalize(vmin=C_AXIS["vmin"], vmax=C_AXIS["vmax"])
    cmap = mpl.colormaps[CMAP]

    plotted = [m for m in models if m in METRICS]
    missing = [m for m in models if m not in METRICS]

    # ---- all study models: coloured by the non-screened metric ------------
    # ONE scatter call (not one per model) so the label placer can be handed
    # the PathCollection and treat every marker as an obstacle.
    xs = [METRICS[m][2] for m in plotted]
    ys = [METRICS[m][Y_AXIS["idx"]] for m in plotted]
    cols = [cmap(norm(METRICS[m][C_AXIS["idx"]])) for m in plotted]
    sc = ax.scatter(xs, ys, s=70, c=cols,
                    edgecolors="0.25", linewidths=0.5, zorder=3)

    # ---- colourbar --------------------------------------------------------
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(C_AXIS["cbar_label"], fontsize=14)
    cb.ax.tick_params(labelsize=13)

    # ---- axes cosmetics ---------------------------------------------------
    ax.set_xlabel("Air temperature trend 1980–2014 (°C decade$^{-1}$)",
                  fontsize=15)
    ax.set_ylabel(Y_AXIS["axis_label"], fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=13)
    # limits hug the data (trends 0.150-0.381, TCR 1.25-2.99, ECS 1.89-5.79 in
    # METRICS) while keeping the observed-band lower edge (0.130) inside the
    # panel; the old 0.10-0.40 / 1.15-3.20 limits left dead margins that
    # compressed the cloud
    ax.set_xlim(0.12, 0.39)
    ax.set_ylim(*Y_AXIS["ylim"])
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("0.3")

    # ---- range captions, then labels (in that order: the captions become
    # ---- obstacles for the label allocator), after the layout is final ----
    fig.tight_layout()
    notes = _annotate_ranges(ax)
    fig.canvas.draw()          # realises text extents for the obstacle sampler
    obst_x, obst_y = _text_obstacle_points(fig, ax, notes)
    unplaced = place_labels(ax, xs, ys, plotted, scatter=sc,
                            x_obstacles=obst_x, y_obstacles=obst_y,
                            **label_kwargs)
    if unplaced:
        logger.warning(f"{len(unplaced)} label(s) could not be placed clear of "
                       f"everything else: {sorted(unplaced)}")

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")
    return missing


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "Arcticnet__49052"
    table, n_real = collect_members(arg)
    models = sorted(table.keys())

    # ====================== FIGURE 1 =======================================
    missing = draw_figure(models)

    if missing:
        print("\nModels with NO literature ECS/TCR/trend entry "
              "(add them to METRICS):")
        for m in sorted(set(missing)):
            print(f"  - {m}")

    # ====================== TABLE 1 ========================================
    rows, missing_metrics = build_rows(table)
    print_table_stdout(rows)
    src = os.path.basename(_resolve_mean_deltas_file(arg))
    write_tex(rows, TEX_PATH, src)
    if WRITE_CSV:
        write_csv(rows, CSV_PATH)
    logger.info(f"{len(table)} models, {n_real} realizations summarised.")
    if missing_metrics:
        logger.warning("models with NO ECS/TCR in METRICS "
                       f"(shown as {MISSING!r}; add them above): "
                       f"{sorted(set(missing_metrics))}")


if __name__ == "__main__":
    em.reject_cli_tokens(sys.argv[1:])
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment(logger)
    main()
