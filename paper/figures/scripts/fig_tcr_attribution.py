"""
fig_tcr_attribution.py
======================

Per-model TCR vs the three steps of the causal chain, one point per CMIP6
model (co-author request, 10 Aug 2026): does the excess warming of the
too-hot models actually reach (a) catchment air temperature, (b) extreme
precipitation and (c) the annual-maximum flood response?

Panels (x-axis = literature TCR from fig01_model_screening.METRICS):
  a  catchment-averaged warming              temp_change_degC          (degC)
  b  catchment-averaged extreme-precipitation precip_q99_change_pct    (%)
     change (daily q99)
  c  catchment-averaged AnnMax change        delta_rel_mean_annual_max (%)
     HBV (circles) and LSTM (triangles)

Points are coloured by ECS (same RdYlGn_r scale as Fig. 1) so the TCR-ECS
covariation stays visible; the dashed vertical line is the TCR = 2.2 degC
screen. Pearson r / Spearman rho (two-sided) are annotated per panel. The
console + CSV output additionally reports ECS correlations, OLS slopes,
Rx1day as an alternative extreme-precipitation metric, and the two-predictor
check proposed in the co-author chat: the partial correlation of AnnMax with
TCR after controlling for the extreme-precipitation change (does TCR add
anything once precipitation is in the regression?).

Aggregation: per (model, member) the change is first averaged over all
catchments, then members are averaged within a model (one model = one vote,
the same "model democracy" fig02 uses). The error bar is +-1 SD across
members, drawn only for models with >= 2 members. Catchment medians are
cached alongside the means (set PLOT_STAT = "median" to plot them).

Caveat carried from the chat, printed with the stats: the models are not
independent samples (4 EC-Earth3 variants, CMCC/CESM siblings, ...), so the
effective degrees of freedom are fewer than the point count and the p-values
are optimistic. The figure is a consistency check of the causal chain, not a
formal attribution test.

Companion map ("corrmap" mode, added 10 Aug 2026)
-------------------------------------------------
Panel c is an across-catchment aggregate: one number per model, averaged over
all catchments. `corr_map_main()` resolves that same relationship in space. At
EACH catchment it correlates the per-model AnnMax change (members averaged
first, so one model = one vote, exactly as in panel c) against the literature
TCR of those models, and maps the resulting Pearson r -- one map per
hydrological model. The per-catchment sample is the model count (<= 32), not
the catchment count, so a single catchment's r is a noisy estimate; the maps
answer where the TCR->flood link survives at a single site, not how large it is.

Inputs (read-only; built by fig02_precip_change.py / build_streamflow_cache.py):
    <fig_data>/fig02_change.parquet     climate forcing change per realization
    <fig_data>/stats_change.parquet     flow + precip-index change per realization
Outputs:
    <fig_data>/tcr_attr_permember_<scen>_<per>.csv     audit trail (one row per member)
    <fig_data>/tcr_attr_permodel_<scen>_<per>.csv      the per-model points
    <fig_data>/tcr_attr_correlations_<scen>_<per>.csv  every correlation/regression stat
    paper/figures/fig_tcr_attribution_<scen>_<per>.png
  corrmap mode only:
    <fig_data>/tcr_annmax_corrmap_<scen>_<per>.csv     per-catchment r / rho / slope
    paper/figures/fig_tcr_annmax_corrmap_<scen>_<per>.png

Run (via slurm -- stage 1 reads an 11 GB parquet):
    sbatch slurm/fig_tcr_attribution.sh
    sbatch slurm/fig_tcr_corrmap.sh                    # the per-catchment map only
    python paper/figures/scripts/fig_tcr_attribution.py            # stage 2 only, if CSVs exist
    python paper/figures/scripts/fig_tcr_attribution.py recompute  # force stage 1
    python paper/figures/scripts/fig_tcr_attribution.py corrmap    # map only (needs its CSV)
    python paper/figures/scripts/fig_tcr_attribution.py corrmap recompute
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st

# this file lives in paper/figures/scripts/; config.py is at the repo root,
# three dirs up (same pattern as every other script here).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
# single source of truth for literature ECS/TCR and for the text-obstacle
# sampler the label allocator needs
from fig01_model_screening import (METRICS, SCEN_LABEL, _text_obstacle_points,
                                   TCR_LIKELY_LO, TCR_LIKELY_HI)
from realization_blocklist import apply_blocklist
from figstyle import apply_style, add_panel_letters
import ensemble_method as em

apply_style()

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
#  USER SETTINGS
# ===========================================================================
FOCUS_SCEN = "ssp585"
FOCUS_PER  = "2081-2100"

TEMP_COL   = "temp_change_degC"          # panel a  (climate cache)
PRECIP_COL = "precip_q99_change_pct"     # panel b  (climate cache)
FLOW_COL   = "delta_rel_mean_annual_max" # panel c  (flow cache, fraction -> %)
RX1DAY_COL = "delta_rel_rx1day"          # CSV-only alternative for panel b

TOO_HOT_TCR_THRESHOLD = em.TCR_UPPER     # degC; from em (single source)

# The x-axis of the scatter, the correlation of the map and the screening
# line(s) follow the ACTIVE metric of ensemble_method.py (19 Aug 2026):
#   em.METRIC='tcr' (the historical figure): TCR on x, coloured by ECS, band and
#       dashed line at the AR6 likely TCR bounds; UNTAGGED file names, so the
#       outputs are byte-identical to the pre-switch figure.
#   em.METRIC='ecs': ECS on x, coloured by TCR, band and dashed line(s) at the
#       AR6 likely ECS bounds (both bounds under SCREEN='both'); the PNGs and the
#       corrmap CSV carry the screen tag (e.g. '_ecs_2s') and 'ecs' replaces
#       'tcr' in their stems, so no TCR file is overwritten.
# The Stage-1 per-model / per-member CSVs and the correlation table are
# metric-INDEPENDENT (they carry both tcr and ecs columns) and stay untagged.
# mod["too_hot"] stays the TCR > 2.2 K flag whatever the metric (documented
# in the CSV as such); the excluded set of the active screen is em.is_excluded.
XM        = em.metric()                  # 'tcr' | 'ecs'  -> x-axis / map metric
CM        = "ecs" if XM == "tcr" else "tcr"   # the OTHER metric -> marker colour
X_LABEL   = {"tcr": "Transient Climate Response (TCR; °C)",
             "ecs": "Equilibrium Climate Sensitivity (ECS; °C)"}
X_SHORT   = {"tcr": "TCR", "ecs": "ECS"}
X_LO, X_HI = em.bounds()                 # AR6 likely bounds of the ACTIVE metric
X_TWO_SIDED = em.is_two_sided()          # draw the lower cut too
_MSFX     = em.screen_suffix() if XM == "ecs" else ""   # '' | '_ecs' | '_ecs_2s'

PLOT_STAT    = "mean"    # "mean" = catchment-averaged (as requested) | "median"
LABEL_POINTS = True      # model names via textalloc (skipped if not installed)
ECS_VMIN, ECS_VMAX = 2.0, 6.0            # colour scale, as in Fig. 1
TCR_VMIN, TCR_VMAX = 1.2, 3.0            # colour scale when TCR is the colour
C_VMIN, C_VMAX = ((ECS_VMIN, ECS_VMAX) if CM == "ecs" else (TCR_VMIN, TCR_VMAX))
CMAP = "RdYlGn_r"

# Font sizes that depart from the figstyle house sizes (axes.labelsize 12,
# legend.fontsize 10, panel letter 13). Set here rather than in figstyle so no
# other figure in the manuscript changes.
AXLABEL_FS      = 14     # x- and y-axis labels
CBLABEL_FS      = 14     # ECS colorbar label
LEGEND_FS       = 12     # HBV / LSTM legend entries
PANEL_LETTER_FS = 16     # bold a, b, c

# --- corrmap mode (per-catchment TCR vs delta-AnnMax correlation map) ------
# Minimum number of CMIP6 models with a finite AnnMax change at a catchment
# before its correlation is kept. 32 models are available for SSP5-8.5, so 20
# keeps only catchments where the correlation rests on most of the ensemble.
CORRMAP_MIN_MODELS = 20
CORRMAP_ALPHA      = 0.05        # for the "significant fraction" summary only
# Diverging map centred on r = 0. RdYlBu reversed: r = 0 is pale yellow
# (#FFFFBF) rather than white (CLAUDE.md) or the neutral gray of coolwarm, and
# the reversal keeps the warm end on positive r, as in the fig02 difference
# maps. Anything replacing it must not put white or gray at zero.
CORRMAP_CMAP       = "RdYlBu_r"
CORRMAP_VLIM       = 1.0         # colour scale runs -1 .. +1
# Canvas. savefig.bbox='tight' (figstyle) trims this, and the two
# equal-aspect Robinson maps are height-constrained here, so CORRMAP_FIG_H
# sets the size of the trimmed output: 6.6 in gives ~4.6 x 5.5 in of
# content, meant to be included at width=0.72\linewidth in main4.tex
# (6.61 in textwidth), i.e. at ~1:1 so the point sizes below print as set.
CORRMAP_FIG_W, CORRMAP_FIG_H = 7.2, 6.6
CORRMAP_MARKER_S   = 1.8         # scatter marker area. Smaller than the
# fig02 maps' 4: the map is ~40% narrower here, and marker area has to
# follow the square of that or the ~20k catchments merge into blobs.
CORRMAP_LAND_FC    = "0.96"      # land polygon fill, as in the fig02 maps
# inset histogram of r, in axes fractions (x0, y0, w, h), placed in the empty
# south-east Pacific of the Robinson map. Three placement constraints:
#   - the whole inset, including the "Pearson r" label and the -1.0 tick label,
#     sits inside the map boundary. The binding constraint is the boundary at
#     the inset's left edge: in Robinson, x = 0.11 meets the outline at
#     y = 0.12, and x = 0.085 (the left end of the -1.0 tick label) at y = 0.15,
#     so y0 = 0.20 leaves the labels clear of the outline;
#   - x0 + w stays left of x = 0.30, where the southern-Chile catchments begin;
#   - the top of the inset stays below y = 0.49, short of the Peruvian and
#     Ecuadorian catchments near the equator.
# The inset background is transparent, which is only safe because the rectangle
# above holds no catchment; over a populated region the bars would be unreadable.
CORRMAP_HIST_RECT  = [0.11, 0.20, 0.175, 0.27]

RECOMPUTE = "recompute" in sys.argv[1:]
CORRMAP_ONLY = "corrmap" in sys.argv[1:]

# ---------------------------------------------------------------------------
#  paths
# ---------------------------------------------------------------------------
FIG_DATA      = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")
CLIMATE_CACHE = os.path.join(FIG_DATA, "fig02_change.parquet")
FLOW_CACHE    = os.path.join(FIG_DATA, "stats_change.parquet")

TAG            = f"{FOCUS_SCEN}_{FOCUS_PER}"
PER_MEMBER_CSV = os.path.join(FIG_DATA, f"tcr_attr_permember_{TAG}.csv")
PER_MODEL_CSV  = os.path.join(FIG_DATA, f"tcr_attr_permodel_{TAG}.csv")
STATS_CSV      = os.path.join(FIG_DATA, f"tcr_attr_correlations_{TAG}.csv")
# metric-DEPENDENT outputs: 'tcr' stems, untagged, under METRIC='tcr' (as
# before); 'ecs' stems + screen tag under METRIC='ecs'
OUT_PNG        = os.path.join(os.getcwd(), "paper", "figures",
                              f"fig_{XM}_attribution_{TAG}{_MSFX}.png")
CORRMAP_CSV    = os.path.join(FIG_DATA, f"{XM}_annmax_corrmap_{TAG}{_MSFX}.csv")
CORRMAP_PNG    = os.path.join(os.getcwd(), "paper", "figures",
                              f"fig_{XM}_annmax_corrmap_{TAG}{_MSFX}.png")


# ===========================================================================
#  Stage 1 — aggregate the two realization-level caches to one row per model
# ===========================================================================
def _read_cache(cache, metric_cols, extra_filters=None, label="cache",
                extra_id_cols=()):
    """Read only the id + metric columns for the focus scenario/period,
    blocklist-filtered, infs NaN-ed. Mirrors the significance script's
    _load_unc_frame so the numbers cannot drift from the other analyses.

    `extra_id_cols` adds further non-metric columns (the corrmap needs the
    station lon/lat); they are dropped silently if the cache lacks them."""
    import pyarrow.parquet as pq
    schema = set(pq.ParquetFile(cache).schema.names)
    missing = [c for c in metric_cols if c not in schema]
    if missing:
        raise KeyError(f"{label} is missing column(s) {missing}; "
                       f"available metric columns: "
                       f"{sorted(c for c in schema if 'change' in c or c.startswith('delta_'))}")
    id_cols = [c for c in (("station", "scenario", "model", "member",
                            "period", "engine") + tuple(extra_id_cols))
               if c in schema]
    filters = [("scenario", "==", FOCUS_SCEN), ("period", "==", FOCUS_PER)]
    if extra_filters:
        filters += extra_filters
    d = pd.read_parquet(cache, columns=id_cols + metric_cols, filters=filters)
    logger.info(f"{label}: {len(d)} rows for {FOCUS_SCEN}/{FOCUS_PER}")
    d = apply_blocklist(d, label=label)
    d[metric_cols] = (d[metric_cols].astype("float64")
                      .replace([np.inf, -np.inf], np.nan))
    unknown = sorted(set(d["model"].astype(str)) - set(METRICS))
    if unknown:
        logger.warning(f"{label}: dropping {len(unknown)} model(s) without a "
                       f"METRICS (TCR/ECS) entry: {unknown}")
        d = d[~d["model"].isin(unknown)]
    return d


def _member_then_model(d, cols, prefix_map):
    """Catchment mean AND median per (model, member), then member mean / SD /
    count per model. Returns (per_member_df, per_model_df)."""
    aggs = {"n_stations": ("station", "nunique")}
    for c in cols:
        aggs[f"{prefix_map[c]}_mean"]   = (c, "mean")
        aggs[f"{prefix_map[c]}_median"] = (c, "median")
    mem = (d.groupby(["model", "member"], observed=True)
             .agg(**aggs).reset_index())

    aggs2 = {"n_members": ("member", "nunique"),
             "n_stations": ("n_stations", "median")}
    for c in cols:
        p = prefix_map[c]
        aggs2[f"{p}_mean"]      = (f"{p}_mean", "mean")
        aggs2[f"{p}_mean_sd"]   = (f"{p}_mean", "std")
        aggs2[f"{p}_median"]    = (f"{p}_median", "mean")
        aggs2[f"{p}_median_sd"] = (f"{p}_median", "std")
    mod = mem.groupby("model", observed=True).agg(**aggs2).reset_index()
    return mem, mod


def compute_stage1():
    for cache, builder in ((CLIMATE_CACHE, "fig02_precip_change.py"),
                           (FLOW_CACHE, "build_streamflow_cache.py")):
        if not os.path.exists(cache):
            sys.exit(f"cache not found: {cache} (run {builder} first)")

    # ---- climate: warming + extreme precip --------------------------------
    dc = _read_cache(CLIMATE_CACHE, [TEMP_COL, PRECIP_COL],
                     label="climate cache")
    mem_c, mod_c = _member_then_model(
        dc, [TEMP_COL, PRECIP_COL],
        {TEMP_COL: "temp", PRECIP_COL: "pq99"})
    n_sta_c = dc["station"].nunique()
    del dc

    # ---- flow: AnnMax per engine ------------------------------------------
    frames_mem, mod_f = [], None
    for eng in ("hbv", "lstm"):
        df = _read_cache(FLOW_CACHE, [FLOW_COL],
                         extra_filters=[("engine", "==", eng)],
                         label=f"flow cache [{eng}]")
        df[FLOW_COL] = df[FLOW_COL] * 100.0          # fraction -> %
        mem_e, mod_e = _member_then_model(
            df, [FLOW_COL], {FLOW_COL: f"annmax_{eng}"})
        mem_e.insert(2, "engine", eng)
        frames_mem.append(mem_e)
        mod_e = mod_e.rename(columns={"n_members": f"n_members_{eng}",
                                      "n_stations": f"n_stations_{eng}"})
        mod_f = mod_e if mod_f is None else mod_f.merge(mod_e, on="model",
                                                        how="outer")
        del df

    # ---- Rx1day (precip index; identical across engines, read hbv rows) ---
    dr = _read_cache(FLOW_CACHE, [RX1DAY_COL],
                     extra_filters=[("engine", "==", "hbv")],
                     label="flow cache [rx1day]")
    dr[RX1DAY_COL] = dr[RX1DAY_COL] * 100.0
    _, mod_r = _member_then_model(dr, [RX1DAY_COL], {RX1DAY_COL: "rx1day"})
    del dr

    # ---- merge + literature TCR/ECS ---------------------------------------
    mod = (mod_c.rename(columns={"n_members": "n_members_clim",
                                 "n_stations": "n_stations_clim"})
                .merge(mod_f, on="model", how="outer")
                .merge(mod_r.drop(columns=["n_members", "n_stations"]),
                       on="model", how="left"))
    mod.insert(1, "tcr", [METRICS[m][1] for m in mod["model"]])
    mod.insert(2, "ecs", [METRICS[m][0] for m in mod["model"]])
    mod.insert(3, "too_hot", mod["tcr"] > TOO_HOT_TCR_THRESHOLD)
    mod = mod.sort_values("tcr").reset_index(drop=True)

    os.makedirs(FIG_DATA, exist_ok=True)
    pd.concat(frames_mem + [mem_c], axis=0).to_csv(PER_MEMBER_CSV, index=False)
    mod.to_csv(PER_MODEL_CSV, index=False)
    logger.info(f"wrote {PER_MODEL_CSV}  ({len(mod)} models, "
                f"{int(mod['too_hot'].sum())} too-hot, {n_sta_c} catchments)")
    logger.info(f"wrote {PER_MEMBER_CSV}")

    # cross-check against the manuscript's model-democracy numbers
    for eng in ("hbv", "lstm"):
        col = f"annmax_{eng}_mean"
        if col in mod:
            full = mod[col].mean()
            # follows the ACTIVE screen; mod["too_hot"] stays the hot-only
            # flag because the scatter legend is about the too-hot subset
            keep = ~mod["model"].map(em.is_excluded).astype(bool)
            red  = mod.loc[keep, col].mean()
            logger.info(f"model-democracy catchment-mean AnnMax change "
                        f"[{eng}]: full = {full:+.2f}%, reduced = {red:+.2f}% "
                        f"(bias {full - red:+.2f} pp) -- compare main4.tex")
    return mod


# ===========================================================================
#  correlation / regression table
# ===========================================================================
def _pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _partial_corr(y, x, z):
    """Pearson correlation of y with x after removing the OLS fit on z from
    both (first-order partial correlation); p two-sided with n-3 df."""
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    r = float(np.corrcoef(ry, rx)[0, 1])
    n = len(y)
    t = r * np.sqrt(max(n - 3, 1) / max(1.0 - r * r, 1e-12))
    p = 2.0 * st.t.sf(abs(t), max(n - 3, 1))
    return r, p, n


def correlation_table(mod, stat=PLOT_STAT):
    rows = []
    responses = [("temp",        "warming (degC)"),
                 ("pq99",        "extreme precip q99 (%)"),
                 ("rx1day",      "Rx1day (%)"),
                 ("annmax_hbv",  "AnnMax HBV (%)"),
                 ("annmax_lstm", "AnnMax LSTM (%)")]
    for pred in ("tcr", "ecs"):
        for key, label in responses:
            col = f"{key}_{stat}"
            if col not in mod:
                continue
            sub = mod[[pred, col]].dropna()
            if len(sub) < 3:
                continue
            x, y = sub[pred].values, sub[col].values
            pr, pp = st.pearsonr(x, y)
            sr, sp = st.spearmanr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            rows.append(dict(predictor=pred.upper(), response=label,
                             stat=stat, n=len(sub),
                             pearson_r=pr, pearson_p=pp,
                             spearman_rho=sr, spearman_p=sp,
                             ols_slope=slope, ols_intercept=intercept))
    # the two-predictor check: does TCR still correlate with AnnMax once the
    # extreme-precipitation change is controlled for?
    for eng in ("hbv", "lstm"):
        col = f"annmax_{eng}_{stat}"
        if col not in mod:
            continue
        sub = mod[["tcr", f"pq99_{stat}", col]].dropna()
        if len(sub) < 4:
            continue
        r, p, n = _partial_corr(sub[col].values, sub["tcr"].values,
                                sub[f"pq99_{stat}"].values)
        rows.append(dict(predictor="TCR | Pq99",
                         response=f"AnnMax {eng.upper()} (%)", stat=stat,
                         n=n, pearson_r=r, pearson_p=p,
                         spearman_rho=np.nan, spearman_p=np.nan,
                         ols_slope=np.nan, ols_intercept=np.nan))
    tab = pd.DataFrame(rows)
    tab.to_csv(STATS_CSV, index=False)
    logger.info(f"wrote {STATS_CSV}")
    return tab


# ===========================================================================
#  Stage 2 — the 3-panel figure
# ===========================================================================
def _label_points(fig, ax, xs, ys, names, sc, extra_texts=()):
    """Model-name labels via textalloc (as in fig01); silently skipped if the
    package is unavailable. `extra_texts` (stats box, legend-free corners) are
    sampled into point obstacles so no label lands on them."""
    try:
        import textalloc
    except ImportError:
        logger.warning("textalloc not installed; model names not drawn")
        return
    ox, oy = ([], [])
    if extra_texts:
        ox, oy = _text_obstacle_points(fig, ax, list(extra_texts))
    textalloc.allocate(
        ax, list(xs), list(ys), list(names),
        scatter_plot=sc, x_scatter=ox or None, y_scatter=oy or None,
        textsize=7.5, textcolor="0.45",
        margin=0.005, min_distance=0.01, max_distance=0.35,
        nbr_candidates=1500,
        draw_lines=True, linecolor="0.6", linewidth=0.5,
        draw_all=True, avoid_label_lines_overlap=True,
        avoid_crossing_label_lines=True,
        priority_strategy="largest",
        plot_kwargs=dict(zorder=2), zorder=4)


def _stats_text(mod, key, stat=PLOT_STAT):
    col = f"{key}_{stat}"
    sub = mod[[XM, col]].dropna()
    pr, pp = st.pearsonr(sub[XM], sub[col])
    sr, sp = st.spearmanr(sub[XM], sub[col])
    return (f"r = {pr:.2f} ({_pfmt(pp)})\n"
            f"ρ = {sr:.2f} ({_pfmt(sp)})")


def _fit_line(ax, x, y, **kw):
    sl, ic = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, sl * xx + ic, color=kw.pop("color", "0.4"),
            lw=kw.pop("lw", 1.2), ls=kw.pop("ls", "-"), zorder=1.8, **kw)


def draw_figure(mod, stat=PLOT_STAT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D

    norm = Normalize(vmin=C_VMIN, vmax=C_VMAX)
    cmap = matplotlib.colormaps[CMAP]
    colors = [cmap(norm(e)) for e in mod[CM]]

    # Layout: three square panels in one row, with a slim horizontal colorbar
    # centred underneath. The row is given a height equal to the panel WIDTH in
    # inches, so the square box aspect (set below) fills its cell exactly and
    # leaves no empty band between the panels and the colorbar.
    FIG_W, FIG_H = 15.2, 5.9
    L, R, TOP, WS = 0.040, 0.995, 0.955, 0.16
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    panel_w = (R - L) / (3 + 2 * WS)              # figure fraction
    panel_h = panel_w * FIG_W / FIG_H             # same in inches -> square
    gs = fig.add_gridspec(1, 3, left=L, right=R, top=TOP,
                          bottom=TOP - panel_h, wspace=WS)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    # colorbar: 46% of the figure width (it spanned one panel until 10 Aug 2026),
    # centred on the middle panel
    cb_w = 0.46
    cax  = fig.add_axes([L + panel_w * (1.5 + WS) - cb_w / 2,
                         0.075, cb_w, 0.028])
    panels = [ax_a, ax_b, ax_c]

    x = mod[XM].values

    # ---- panels a, b: climate (no titles: the description lives in the
    # ---- y-label, so a long title cannot collide with the panel letter) ---
    specs = [(ax_a, "temp", "Catchment-averaged warming (°C)"),
             (ax_b, "pq99", "Catchment-averaged Δ precipitation q99 (%)")]
    scatters = {}
    for ax, key, ylab in specs:
        y  = mod[f"{key}_{stat}"].values
        sd = mod[f"{key}_{stat}_sd"].values
        # SD across members is NaN for single-member models -> no bar drawn
        ax.errorbar(x, y, yerr=np.where(np.nan_to_num(sd) > 0, sd, np.nan),
                    fmt="none", ecolor="0.78", elinewidth=0.8, zorder=1.5)
        _fit_line(ax, x, y)
        scatters[key] = ax.scatter(x, y, s=55, c=colors,
                                   edgecolors="0.25", linewidths=0.5, zorder=3)
        ax.set_ylabel(ylab, fontsize=AXLABEL_FS)

    # ---- panel c: AnnMax, both engines ------------------------------------
    yh  = mod[f"annmax_hbv_{stat}"].values
    sdh = mod[f"annmax_hbv_{stat}_sd"].values
    ax_c.errorbar(x, yh, yerr=np.where(np.nan_to_num(sdh) > 0, sdh, np.nan),
                  fmt="none", ecolor="0.78", elinewidth=0.8, zorder=1.5)
    _fit_line(ax_c, x, yh, ls="-")
    sc_h = ax_c.scatter(x, yh, s=55, c=colors, marker="o",
                        edgecolors="0.25", linewidths=0.5, zorder=3)
    yl = mod[f"annmax_lstm_{stat}"].values
    have_lstm = np.isfinite(yl).sum() >= 3
    if have_lstm:
        m = np.isfinite(yl)
        _fit_line(ax_c, x[m], yl[m], ls="--")
        ax_c.scatter(x[m], yl[m], s=45, c=[c for c, k in zip(colors, m) if k],
                     marker="^", edgecolors="0.25", linewidths=0.5, zorder=3)
    ax_c.set_ylabel("Catchment-averaged Δ AnnMax (%)", fontsize=AXLABEL_FS)

    # ---- shared cosmetics --------------------------------------------------
    for ax in panels:
        # IPCC AR6 likely TCR range, shaded as in fig01 (there it is an
        # axhspan because TCR sits on the y-axis; here TCR is the x-axis)
        ax.axvspan(X_LO, X_HI, color="0.55", alpha=0.16,
                   lw=0, zorder=0.5)
        ax.axvline(X_HI, ls=(0, (5, 3)), color="0.25",
                   lw=1.1, zorder=1.2)
        if X_TWO_SIDED:   # two-sided screen: the lower AR6 bound is a cut too
            ax.axvline(X_LO, ls=(0, (5, 3)), color="0.25", lw=1.1, zorder=1.2)
        ax.set_xlabel(X_LABEL[XM],
                      fontsize=AXLABEL_FS)
        ax.grid(True, color="0.93", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_box_aspect(1.0)
        # square axes are centre-anchored by default; anchor to the top so any
        # residual rounding in the cell height opens no gap above the panels
        ax.set_anchor("N")
    # short horizontal note beside the line (the bottom-right of panel a is
    # empty: every TCR > 2.2 model warms by more than 4.5 degC)
    txt_screen = ax_a.text(
        X_HI + 0.03, ax_a.get_ylim()[0]
        + 0.02 * np.diff(ax_a.get_ylim())[0],
        f"{X_HI:.1f} °C screen", va="bottom", ha="left",
        fontsize=8.5, color="0.25")
    # left-aligned just inside the band, below the panel-a stats box (the
    # top-centre position collided with the stats text)
    txt_band = ax_a.text(
        X_LO + 0.03, ax_a.get_ylim()[1]
        - 0.10 * np.diff(ax_a.get_ylim())[0],
        f"AR6 likely {X_SHORT[XM]} range", va="top", ha="left",
        fontsize=8.5, color="0.35")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(X_LABEL[CM],
                 fontsize=CBLABEL_FS)

    # No suptitle (removed 10 Aug 2026): the scenario / period / model count
    # belong in the manuscript caption, not on the figure. Logged here so the
    # caption can be written from the run rather than from memory.
    base = "-".join(p[:4] for p in CONFIG.historical_period)
    logger.info(f"caption facts: {SCEN_LABEL.get(FOCUS_SCEN, FOCUS_SCEN)}, "
                f"{FOCUS_PER} minus baseline ({base}); "
                f"n = {len(mod)} CMIP6 models")

    # ---- stats boxes (drawn before the label allocator runs so the labels
    # ---- treat them as obstacles) -----------------------------------------
    notes = {}
    for ax, key in ((ax_a, "temp"), (ax_b, "pq99")):
        notes[key] = ax.text(0.03, 0.97, _stats_text(mod, key),
                             transform=ax.transAxes, va="top", ha="left",
                             fontsize=9, linespacing=1.35, zorder=5)
    def _short(key):
        col = f"{key}_{stat}"
        sub = mod[[XM, col]].dropna()
        pr, pp = st.pearsonr(sub[XM], sub[col])
        sr, sp = st.spearmanr(sub[XM], sub[col])
        return (f"r = {pr:.2f}, ρ = {sr:.2f} "
                f"({_pfmt(max(pp, sp)).replace('p = ', 'p ≤ ')})")
    txt_c = "HBV:  " + _short("annmax_hbv")
    if have_lstm:
        txt_c += "\nLSTM: " + _short("annmax_lstm")
    notes["annmax"] = ax_c.text(0.03, 0.97, txt_c, transform=ax_c.transAxes,
                                va="top", ha="left", fontsize=8.5,
                                linespacing=1.4, zorder=5)
    if have_lstm:
        # marker + matching fit-line style in one legend entry, so the reader
        # can tie both the points and the lines to their hydrological model
        handles = [Line2D([], [], ls="-", lw=1.2, color="0.4",
                          marker="o", markersize=7,
                          markerfacecolor="0.7", markeredgecolor="0.25",
                          label="HBV (solid fit)"),
                   Line2D([], [], ls="--", lw=1.2, color="0.4",
                          marker="^", markersize=7,
                          markerfacecolor="0.7", markeredgecolor="0.25",
                          label="LSTM (dashed fit)")]
        # sits just above the corner; the TaiESM1 LSTM outlier triangle is
        # far to the left of it, so the legend can hug the axis
        ax_c.legend(handles=handles, loc="lower right",
                    bbox_to_anchor=(1.0, 0.02), frameon=False,
                    fontsize=LEGEND_FS)

    fig.canvas.draw()
    if LABEL_POINTS:
        names = mod["model"].tolist()
        _label_points(fig, ax_a, x, mod[f"temp_{stat}"], names,
                      scatters["temp"],
                      extra_texts=[notes["temp"], txt_screen, txt_band])
        _label_points(fig, ax_b, x, mod[f"pq99_{stat}"], names,
                      scatters["pq99"], extra_texts=[notes["pq99"]])
        _label_points(fig, ax_c, x, yh, names, sc_h,
                      extra_texts=[notes["annmax"]])

    add_panel_letters(panels, fontsize=PANEL_LETTER_FS)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


# ===========================================================================
#  corrmap — panel c resolved per catchment
# ===========================================================================
def _rowwise_corr(Y, X):
    """Row-wise Pearson r and OLS slope dY/dX for two equally shaped matrices.

    Each row is one catchment and each column one climate model. Only the
    columns finite in BOTH matrices enter that row's statistics, so a model
    missing at one catchment does not remove it everywhere. Returns
    (r, slope, n_used) with one element per row."""
    M = np.isfinite(Y) & np.isfinite(X)
    n = M.sum(axis=1).astype("float64")
    safe_n = np.where(n > 0, n, 1.0)
    # Two-pass (centred) sums. The one-pass form sum(y^2) - sum(y)^2/n cancels
    # catastrophically when a catchment's AnnMax changes are large relative to
    # their spread, which these relative changes can be; the round-off then
    # surfaces as a spurious |r| = 1 with p = 0.
    ybar = np.where(M, Y, 0.0).sum(axis=1) / safe_n
    xbar = np.where(M, X, 0.0).sum(axis=1) / safe_n
    Yc = np.where(M, Y - ybar[:, None], 0.0)
    Xc = np.where(M, X - xbar[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (Yc * Xc).sum(axis=1)
        vy  = (Yc * Yc).sum(axis=1)
        vx  = (Xc * Xc).sum(axis=1)
        ok_x = (n >= 3) & (vx > 0)
        ok = ok_x & (vy > 0)
        r = np.where(ok, cov / np.sqrt(np.where(ok, vx * vy, 1.0)), np.nan)
        slope = np.where(ok_x, cov / np.where(ok_x, vx, 1.0), np.nan)
    return np.clip(r, -1.0, 1.0), slope, n


def _rowwise_rank(A):
    """Average ranks along axis 1 over the finite entries of each row; NaN
    entries stay NaN. Ties take the mean rank, as scipy's default does."""
    if np.isfinite(A).all():
        return st.rankdata(A, axis=1).astype("float64")
    out = np.full(A.shape, np.nan)
    fin = np.isfinite(A)
    for i in np.flatnonzero(fin.any(axis=1)):
        m = fin[i]
        out[i, m] = st.rankdata(A[i, m])
    return out


def _corr_pvalue(r, n):
    """Two-sided p for a Pearson r on n paired values (t test, n - 2 df).

    The models are not independent draws (four EC-Earth3 variants, CMCC/CESM
    siblings), so the effective sample is smaller than n and these p-values are
    optimistic. They are reported as a descriptive summary, not a formal test."""
    r = np.asarray(r, dtype="float64")
    n = np.asarray(n, dtype="float64")
    p = np.full(r.shape, np.nan)
    usable = np.isfinite(r) & (n > 2)
    perfect = usable & (np.abs(r) >= 1.0)
    ok = usable & ~perfect
    with np.errstate(invalid="ignore", divide="ignore"):
        t = r[ok] * np.sqrt((n[ok] - 2.0) / (1.0 - r[ok] ** 2))
    p[ok] = 2.0 * st.t.sf(np.abs(t), n[ok] - 2.0)
    p[perfect] = 0.0
    return p


def _station_model_matrix(engine):
    """(coords, model list, station x model matrix of AnnMax change in %).

    Members are averaged within a model first, so every model contributes one
    value per catchment (one model = one vote, as in panel c). Panel c then
    averages over catchments; here the catchment axis is kept. Note the orders
    differ: panel c averages catchments within a member before averaging
    members, so the two aggregates coincide only where the catchment-by-member
    panel of finite values is complete."""
    d = _read_cache(FLOW_CACHE, [FLOW_COL],
                    extra_filters=[("engine", "==", engine)],
                    label=f"flow cache [{engine}]",
                    extra_id_cols=("lon", "lat"))
    d[FLOW_COL] = d[FLOW_COL] * 100.0                     # fraction -> %
    for c in ("station", "model"):
        if str(d[c].dtype) == "category":
            d[c] = d[c].cat.remove_unused_categories()

    coords = (d.groupby("station", observed=True)[["lon", "lat"]]
                .first().astype("float64"))
    wide = (d.groupby(["station", "model"], observed=True)[FLOW_COL]
              .mean().unstack("model"))
    del d

    models = [m for m in wide.columns.astype(str)
              if m in METRICS and wide[m].notna().any()]
    if not models:
        sys.exit(f"[{engine}] no climate model with a TCR entry has data in "
                 f"{FLOW_CACHE} for {FOCUS_SCEN}/{FOCUS_PER}")
    wide = wide.reindex(columns=models)
    coords = coords.reindex(wide.index)
    logger.info(f"[{engine}] {len(wide)} catchments x {len(models)} models "
                f"(TCR {min(METRICS[m][1] for m in models):.2f}-"
                f"{max(METRICS[m][1] for m in models):.2f} degC)")
    return coords, models, wide


def compute_corr_map():
    """Per-catchment correlation of the per-model AnnMax change with TCR, for
    both hydrological models. Writes CORRMAP_CSV and returns it."""
    if not os.path.exists(FLOW_CACHE):
        sys.exit(f"cache not found: {FLOW_CACHE} "
                 "(run build_streamflow_cache.py first)")

    out = []
    for eng in ("hbv", "lstm"):
        coords, models, wide = _station_model_matrix(eng)
        Y = wide.to_numpy(dtype="float64")
        # value of the ACTIVE metric (TCR by default, ECS under METRIC="ecs")
        tcr = np.array([em.metric_value(m) for m in models], dtype="float64")
        X = np.where(np.isfinite(Y), tcr[None, :], np.nan)

        r, slope, n = _rowwise_corr(Y, X)
        rho, _, _ = _rowwise_corr(_rowwise_rank(Y), _rowwise_rank(X))

        # Across-catchment aggregate of the SAME matrix, as a cross-check on
        # Fig. 1c. Panel c averages catchments within a member and then members
        # within a model; this averages members first and then catchments. The
        # two orders coincide only where the (catchment x member) panel of
        # finite values is complete, so a small difference is expected rather
        # than a defect.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y_agg = np.nanmean(Y, axis=0)
            y_sta = np.nanmean(Y, axis=1)
        agg_ok = np.isfinite(y_agg)
        r_agg = float(np.corrcoef(tcr[agg_ok], y_agg[agg_ok])[0, 1])
        slope_agg = float(np.polyfit(tcr[agg_ok], y_agg[agg_ok], 1)[0])
        logger.info(f"[{eng}] across-catchment aggregate: r with {X_SHORT[XM]} = "
                    f"{r_agg:.3f}, OLS slope = {slope_agg:.2f} % per degC "
                    f"(n = {int(agg_ok.sum())} models) -- cross-check against "
                    "Fig. 1c and against tcr_attr_correlations_*.csv")

        out.append(pd.DataFrame({
            "station": wide.index.astype(str),
            "lon": coords["lon"].to_numpy(),
            "lat": coords["lat"].to_numpy(),
            "engine": eng,
            "n_models": n.astype("int64"),
            "pearson_r": r,
            "pearson_p": _corr_pvalue(r, n),
            "spearman_rho": rho,
            "spearman_p": _corr_pvalue(rho, n),
            f"slope_pct_per_degC_{XM}": slope,
            "annmax_change_pct_modelmean": y_sta,
            "aggregate_r": r_agg,
        }))
        del wide, Y, X

    cm = pd.concat(out, ignore_index=True)
    os.makedirs(FIG_DATA, exist_ok=True)
    cm.to_csv(CORRMAP_CSV, index=False)
    logger.info(f"wrote {CORRMAP_CSV} ({len(cm)} rows)")
    return cm


def _corrmap_subset(cm, engine):
    """Rows of `cm` for one hydrological model, restricted to catchments whose
    correlation rests on at least CORRMAP_MIN_MODELS climate models."""
    s = cm[(cm["engine"] == engine)
           & (cm["n_models"] >= CORRMAP_MIN_MODELS)
           & np.isfinite(cm["pearson_r"])
           & np.isfinite(cm["lon"]) & np.isfinite(cm["lat"])]
    return s


def summarize_corr_map(cm):
    """Log the numbers a caption or a Results sentence would quote."""
    zser = None
    static_f = os.path.join(CONFIG.gscad_path, "dataframes", "static.csv")
    if os.path.exists(static_f):
        static = pd.read_csv(static_f, comment="#", index_col=0)
        if "KGZ_major" in static.columns:
            zser = static["KGZ_major"].astype(str).str.strip()

    for eng in ("hbv", "lstm"):
        s = _corrmap_subset(cm, eng)
        if s.empty:
            logger.warning(f"[{eng}] no catchment passes the "
                           f">= {CORRMAP_MIN_MODELS} model filter")
            continue
        r = s["pearson_r"].to_numpy()
        q25, q50, q75 = np.percentile(r, [25, 50, 75])
        logger.info(
            f"[{eng}] n = {len(s)} catchments | median r = {q50:.2f} "
            f"(IQR {q25:.2f} to {q75:.2f}) | r > 0 at {100*np.mean(r > 0):.1f}% "
            f"| p < {CORRMAP_ALPHA} at "
            f"{100*np.mean(s['pearson_p'].to_numpy() < CORRMAP_ALPHA):.1f}% "
            f"| median rho = {np.nanmedian(s['spearman_rho']):.2f} "
            f"| slope: median "
            f"{np.nanmedian(s[f'slope_pct_per_degC_{XM}']):.1f}, mean "
            f"{np.nanmean(s[f'slope_pct_per_degC_{XM}']):.1f} % per degC {X_SHORT[XM]} "
            f"(the MEAN, not the median, is what the aggregate slope equals)")
        logger.info(f"    n_models per catchment: "
                    f"min {int(s['n_models'].min())}, "
                    f"median {int(s['n_models'].median())}, "
                    f"max {int(s['n_models'].max())}")
        if zser is not None:
            z = s["station"].map(zser)
            for zone in ["A", "B", "C", "D", "E"]:
                v = r[(z == zone).to_numpy()]
                if v.size:
                    logger.info(f"    Koppen {zone}: n = {v.size}, "
                                f"median r = {np.median(v):.2f}")
    logger.info("CAVEAT: the per-catchment sample is the climate-model count "
                "(<= 32), the models are not independent, and no "
                "field-significance control is applied to these p-values.")


def _corrmap_axes(ax, lon, lat, val, cmap, norm, title, geo):
    """One Robinson map of the per-catchment correlation; falls back to a plain
    lon/lat axes if cartopy is unavailable (same pattern as fig02's _scatter).

    `geo` says whether `ax` really is a cartopy GeoAxes. It is decided ONCE in
    draw_corr_map: deciding it here from a bare try/except would let a failure
    inside add_feature drop the PlateCarree transform on a GeoAxes, which
    silently collapses every catchment onto (0, 0)."""
    if geo:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor=CORRMAP_LAND_FC, zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=0.3, edgecolor="0.5", zorder=1)
        kw = dict(transform=ccrs.PlateCarree())
    else:
        ax.set_facecolor("0.97")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        kw = {}
    sc = ax.scatter(lon, lat, c=val, s=CORRMAP_MARKER_S, cmap=cmap, norm=norm,
                    linewidths=0, rasterized=True, zorder=2, **kw)
    ax.set_title(title, fontsize=12)
    return sc


def _corrmap_hist(ax, values, color_edge="0.25"):
    """Inset histogram of the mapped correlations, with the median marked."""
    inset = ax.inset_axes(CORRMAP_HIST_RECT)
    inset.hist(values, bins=np.linspace(-1.0, 1.0, 41),
               color="0.62", edgecolor=color_edge, linewidth=0.3)
    med = float(np.median(values))
    inset.axvline(0.0, color="0.45", lw=0.7)
    inset.axvline(med, color="black", lw=1.1)
    # above the axes, not inside it: at x0 = 0.03 the label would sit on the
    # left tail of the histogram
    inset.text(0.03, 1.04, f"Median r = {med:.2f}", transform=inset.transAxes,
               ha="left", va="bottom", fontsize=7.5)
    inset.set_xlim(-1.0, 1.0)
    inset.set_xticks([-1, -0.5, 0, 0.5, 1])
    inset.set_yticks([])
    inset.set_xlabel("Pearson r", fontsize=7.5, labelpad=1)
    inset.tick_params(axis="x", labelsize=6.5, length=2, pad=1)
    # transparent: CORRMAP_HIST_RECT keeps the inset over open ocean, clear of
    # every catchment, so there is nothing for a filled background to mask
    inset.set_facecolor("none")
    for side, sp in inset.spines.items():
        sp.set_visible(side == "bottom")
        if side == "bottom":
            sp.set_linewidth(0.4)
    return inset


def draw_corr_map(cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    try:
        import cartopy.crs as ccrs
        proj = dict(subplot_kw={"projection": ccrs.Robinson()})
        geo = True
    except ImportError:
        logger.warning("cartopy unavailable; drawing plain lon/lat axes")
        proj, geo = {}, False

    norm = Normalize(vmin=-CORRMAP_VLIM, vmax=CORRMAP_VLIM)
    cmap = matplotlib.colormaps[CORRMAP_CMAP]

    fig, axes = plt.subplots(2, 1, figsize=(CORRMAP_FIG_W, CORRMAP_FIG_H),
                             gridspec_kw=dict(hspace=0.16), **proj)
    titles = {"hbv": "HBV hydrological model",
              "lstm": "LSTM hydrological model"}
    sc = None
    drawn = []
    for ax, eng in zip(np.ravel(axes), ("hbv", "lstm")):
        s = _corrmap_subset(cm, eng)
        if s.empty:
            ax.set_visible(False)
            continue
        # draw the weakest correlations first: with ~20k markers at s = 1.8 the
        # plotting order would otherwise decide which catchment is visible in
        # the dense regions, and station order is arbitrary
        s = s.reindex(s["pearson_r"].abs().sort_values().index)
        sc = _corrmap_axes(ax, s["lon"].to_numpy(), s["lat"].to_numpy(),
                           s["pearson_r"].to_numpy(), cmap, norm, titles[eng],
                           geo)
        _corrmap_hist(ax, s["pearson_r"].to_numpy())
        frac = 100.0 * np.mean(s["pearson_p"].to_numpy() < CORRMAP_ALPHA)
        # The catchment count belongs in the caption, not on the panel. The
        # text sits over the empty Southern Ocean / Antarctic margin, clear of
        # the inset histogram (which ends at x = 0.285), and short enough at
        # 8 pt to stay inside the map outline on the right.
        # two lines: on one line the string crosses the Robinson outline on
        # the right at this height. va="bottom" pins the block's bottom, so
        # the wrap grows upward into the empty Southern Ocean. The anchor sits
        # at y = 0.11 because the Antarctic coast reaches y = 0.08, and the
        # block is ~0.10 axes fractions tall, so it clears the continent below
        # and the southernmost South African catchments (y = 0.26) above.
        ax.text(0.42, 0.11,
                f"p < {CORRMAP_ALPHA:g} at {frac:.1f}%\n"
                f"({100 * CORRMAP_ALPHA:g}% expected by chance)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                linespacing=1.35, zorder=5)
        drawn.append(ax)

    if not drawn:
        plt.close(fig)
        sys.exit("no catchment passed the "
                 f"n_models >= {CORRMAP_MIN_MODELS} filter; nothing to draw")

    cb = fig.colorbar(sc, ax=drawn, location="bottom",
                      fraction=0.045, pad=0.03, aspect=30, shrink=0.55)
    # kept short on purpose: at ~4.6 in of map width a longer label runs off
    # the canvas. "across climate models" is stated in the caption.
    cb.set_label(f"Correlation of Δ AnnMax with {X_SHORT[XM]} (Pearson r)",
                 fontsize=10)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    # 13 (the figstyle house size), not the 16 the scatter panels use: the
    # corrmap prints at ~1:1, so 16 pt would tower over the 12 pt titles
    add_panel_letters(drawn, fontsize=13, force=len(drawn) == 1)
    fig.savefig(CORRMAP_PNG, dpi=300)
    plt.close(fig)
    print(f"Saved {CORRMAP_PNG}")


def corr_map_main():
    if RECOMPUTE or not os.path.exists(CORRMAP_CSV):
        cm = compute_corr_map()
    else:
        cm = pd.read_csv(CORRMAP_CSV)
        logger.info(f"loaded {CORRMAP_CSV} ({len(cm)} rows); "
                    "pass 'recompute' to rebuild")
    summarize_corr_map(cm)
    draw_corr_map(cm)


# ===========================================================================
def main():
    if CORRMAP_ONLY:
        corr_map_main()
        return

    if RECOMPUTE or not os.path.exists(PER_MODEL_CSV):
        mod = compute_stage1()
    else:
        mod = pd.read_csv(PER_MODEL_CSV)
        logger.info(f"loaded {PER_MODEL_CSV} ({len(mod)} models); "
                    "pass 'recompute' to rebuild")

    tab = correlation_table(mod)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\nTable T1 - per-model values (sorted by TCR):")
    show = [c for c in ("model", "tcr", "ecs", "too_hot", "n_members_clim",
                        f"temp_{PLOT_STAT}", f"pq99_{PLOT_STAT}",
                        f"rx1day_{PLOT_STAT}", f"annmax_hbv_{PLOT_STAT}",
                        f"annmax_lstm_{PLOT_STAT}") if c in mod.columns]
    print(mod[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\nTable T2 - correlations with the per-model responses "
          f"(catchment {PLOT_STAT}):")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\nNOTE: models are not independent samples (EC-Earth3 variants, "
          "shared-component siblings); p-values are optimistic. "
          "Consistency check, not a formal attribution test.")

    draw_figure(mod)


if __name__ == "__main__":
    em.reject_cli_tokens(sys.argv[1:])
    main()
