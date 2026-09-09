"""
bootstrap_random_removal.py  (outputs are named random_removal_null_*)
=====================================================================

Random-removal null for the too-hot screening effect (analysis 2 of the
Nature-Water gap list). Answers the objection "removing 15 of 33 models
mechanically shifts the mean and shrinks the spread — the TCR screen proves
nothing": for every (column, scenario, period[, engine]) cell it compares the
OBSERVED effect of removing the TOO_HOT (TCR > 2.2 K) models against a null
distribution obtained by removing the SAME NUMBER of models at random, B times.
Method reference: Rahimpour Asenjan et al. (2023), HESS 27, 4355-4367, Sect. 3
(their null removed the same number of random models and compared ensemble-
spread reduction against it).

Two nulls are computed per cell:
  * family-block (PRIMARY) — removes whole model genealogies at random (the last
    one trimmed to hit exactly h models), so the null contains genealogically
    clustered removals comparable to the too-hot set, which is itself only ~6
    lineages (EC-Earth3, E3SM, CanESM5, HadGEM3/UKESM, CNRM, IPSL, ...). This
    controls the CMIP6 non-exchangeability confound (Knutti et al. 2013;
    Brunner et al. 2020): removing 15 *correlated* models shifts more than
    removing 15 *dispersed* ones, so a uniform null overstates the screen's
    specificity. p_* columns are this family-block null.
  * uniform (REFERENCE) — the classic Asenjan draw of h exchangeable models;
    kept for transparency and shown as a step outline on the figures. p_*_unif.

Three statistics per cell, each with a one-sided empirical p-value:

  bias      global MEDIAN over catchments of (full_mean - reduced_mean) per
            station, model democracy (members -> model -> ensemble). The median
            (not the mean) is used because relative-change indices blow up on
            arid/intermittent basins with a baseline quantile ~ 0 (HBV Q05 mean
            bias was ~+88,500 %); the median is robust and the same estimator is
            used for observed and both nulls, so the p-value stays valid.
            p = fraction of null draws with bias >= observed.
  spread    median over catchments of sigma_reduced / sigma_full, where sigma is
            the std of the per-model mean changes at that station (Asenjan's
            spread-ratio metric). p = fraction of draws with ratio <= observed.
  tailtrim  p90 across catchments of the per-station ensemble-mean change, full
            minus reduced ("upper-tail trimming"). p = fraction >= observed.

Because every statistic is computed on the per-(station, model) MEAN change
(model democracy, exactly as per_station_test in
analyze_full_vs_reduced_significance.py builds its groups), the differing
member counts of the models cannot enter: the null is a draw over MODELS,
which is the quantity the screen selects on.

Uses ONLY the existing Stage-1 caches (no new simulations):
    <CONFIG.out_path>/<delta_method>/fig_data/fig02_change.parquet   (climate)
    <CONFIG.out_path>/<delta_method>/fig_data/stats_change.parquet   (flow)

Outputs (the directory name conveys the method, so the PNG names are just the
cell identity):
    <fig_data>/random_removal_null_<dataset>.csv                  summary table
    <fig_data>/random_removal_null_<dataset>_results.pkl          results cache
    paper/figures/significance/random_removal_null/
        <dataset>[_<engine>]_<scenario>_<period>_n<n>sta.png      combined null figure

For flow this yields TWO figures (one per engine, HBV and LSTM), each stacking
the four selected magnitude metrics as rows; climate yields one four-metric
figure. Columns bootstrapped only for the Supplementary table (flow Q99 and
flashiness) are not drawn.

The CSV keeps only per-statistic SUMMARIES (null mean and 2.5/97.5 percentiles),
so it cannot redraw the histograms, which need the full B-length null-draw
arrays. Those arrays (for the FOCUS cells) plus the summary rows are pickled to
the *_results.pkl cache, keyed by a signature of everything the bootstrap
depends on (B, seed, model universe, and the input parquet mtime/size). A later
run reuses the pickle and only redraws the figures, making layout edits cheap.
The bootstrap reruns automatically when the signature changes; force a rerun by
deleting the pickle or passing --recompute.

Run (via slurm only):
    python paper/figures/scripts/bootstrap_random_removal.py climate
    python paper/figures/scripts/bootstrap_random_removal.py flow
    python paper/figures/scripts/bootstrap_random_removal.py all
    python paper/figures/scripts/bootstrap_random_removal.py flow 2000   # custom B
    python paper/figures/scripts/bootstrap_random_removal.py all --recompute
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))            # paper/figures/scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))  # repo root
sys.path.insert(0, _THIS)
from fig01_model_screening import METRICS, models_in_results
import ensemble_method as em
from config import CONFIG
from realization_blocklist import apply_blocklist
# Nature-style bold top-left panel letters (multi-panel figures only)
from figstyle import add_panel_letters

# ---- caches (same layout as analyze_full_vs_reduced_significance.py) -------
CACHE_DIR     = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")
CLIMATE_CACHE = os.path.join(CACHE_DIR, "fig02_change.parquet")
FLOW_CACHE    = os.path.join(CACHE_DIR, "stats_change.parquet")
FIG_DIR       = os.path.join(os.path.dirname(_THIS), "significance",
                             "random_removal_null")

# SIGN CONVENTION (2 Aug 2026, matching gev_fdr_dipole.py / fig02). The
# manuscript reports every ensemble difference as reduced − full. The stored
# results (pkl cache, CSV) keep the original orientation (bias =
# full − reduced, tail-trim = p90(full) − p90(reduced)) so the cache stays
# valid and the one-sided p-values are untouched; BIAS_SIGN mirrors ONLY the
# drawn histograms of the two difference statistics (bias, tail-trim). The
# spread ratio is orientation-free.
BIAS_SIGN = -1.0
SIGNCONV  = "full − reduced" if BIAS_SIGN > 0 else "reduced − full"

# ---- model universe & the active screen ------------------------------------
# From ensemble_method.py, the single source of truth for the treatment, so
# this null splits the ensemble identically to the figures it tests. Edit
# SCREEN / METHOD / VARIANT at the top of that file; there is no CLI switch.
# EXCLUDED_MODELS is the too-hot set under SCREEN='hot' and additionally the
# too-cold set under SCREEN='both', so the null below automatically draws
# len(EXCLUDED_MODELS) models per draw, matching whichever screen is active.
FULL_MODELS_STATION   = em.FULL_MODELS_STATION
TOO_HOT_TCR_THRESHOLD = em.TCR_UPPER   # K; ALWAYS the TCR bound. It is imported
                                       # by local_vs_aggregate_null.py and is in
                                       # the pickle signature, so it must not
                                       # follow em.METRIC. Figure labels use
                                       # _screen_label(), which does.
FULL_MODELS     = list(em.full_models())
EXCLUDED_MODELS = em.excluded_models()
RETAINED_MODELS = em.retained_models()

# ---- CMIP6 genealogy: model -> family, for the family-block null ------------
# Models that share a base model / major components are grouped so the null can
# remove whole lineages together (they are NOT independent draws). Grouping is
# by documented shared-component genealogy (Knutti et al. 2013, GRL 40, 1194;
# Brunner et al. 2020, ESD 11, 995) plus obvious same-base version families.
# Everything not matched is its own singleton family. UKESM1 is built on
# HadGEM3-GC3.1 (Sellar et al. 2019) so the two are merged; borderline
# cross-institution links that are NOT merged (e.g. ACCESS-CM2 also uses the
# UM/HadGEM3 atmosphere; TaiESM1/SAM0-UNICON/CIESM share CAM physics) are left
# separate — the conservative choice, since merging more would only widen the
# null and make the screen look *less* special.
_FAMILY_OVERRIDES = {
    "HadGEM3-GC31-LL": "HadGEM3-UKESM", "HadGEM3-GC31-MM": "HadGEM3-UKESM",
    "UKESM1-0-LL":     "HadGEM3-UKESM", "UKESM1-1-LL":     "HadGEM3-UKESM",
}
_FAMILY_PREFIXES = [                    # (prefix, family label); first match wins
    ("CanESM5", "CanESM5"), ("EC-Earth3", "EC-Earth3"), ("E3SM", "E3SM"),
    ("CESM2", "CESM2"), ("CNRM", "CNRM"), ("IPSL", "IPSL"), ("GISS", "GISS"),
    ("MIROC", "MIROC"), ("MPI-ESM", "MPI-ESM"), ("AWI", "AWI"), ("BCC", "BCC"),
    ("CMCC", "CMCC"), ("FGOALS", "FGOALS"), ("GFDL", "GFDL"), ("INM", "INM"),
    ("Nor", "NorESM"), ("ACCESS", "ACCESS"),
]


def family_of(model):
    """Map a CMIP6 model name to its genealogy label (singleton if unmatched)."""
    if model in _FAMILY_OVERRIDES:
        return _FAMILY_OVERRIDES[model]
    for pref, fam in _FAMILY_PREFIXES:
        if model.startswith(pref):
            return fam
    return model

# ---- columns analysed by default -------------------------------------------
CLIMATE_COLS = ["precip_change_pct", "temp_change_degC",
                "precip_q99_change_pct", "temp_q99_change_degC"]
FLOW_COLS    = ["delta_rel_mean_annual_max",   # peak flow (headline)
                "delta_rel_flood_q20",         # 20-yr design flood (headline)
                "delta_rel_qmean",             # mean annual runoff (null contrast)
                "delta_rel_q99", "delta_rel_q95",
                "delta_rel_q05",
                "delta_rel_flashiness_rb"]     # reference only (method caveat)

# ---- combined (multi-row) figure layout ------------------------------------
# The FOCUS-cell null panels are stacked into ONE figure per engine, one metric
# per row. For flow we plot FOUR magnitude indices spanning the flow-duration
# curve from the flood extreme to drought low flow: peak flow (annual maximum),
# high flow (Q95), mean annual runoff, and low flow (Q05). All four are
# magnitude (not timing/persistence) indices, so they transfer faithfully under
# qqmap_day2; sequencing-based indices such as flashiness do not (day-to-day
# ordering is not preserved by quantile mapping) and are excluded. Q99 and
# flashiness are still bootstrapped (they populate the Supplementary
# random-removal table) but are NOT plotted.
FLOW_COMBINED_ROWS    = ["delta_rel_mean_annual_max", "delta_rel_q95",
                         "delta_rel_qmean", "delta_rel_q05"]
CLIMATE_COMBINED_ROWS = ["precip_change_pct", "temp_change_degC",
                         "precip_q99_change_pct", "temp_q99_change_degC"]
ROW_LABELS = {
    "delta_rel_mean_annual_max": "mean annual maximum",
    "delta_rel_q05":             "low flow Q05",
    "delta_rel_q95":             "high flow Q95",
    "delta_rel_q99":             "very-high flow Q99",
    "delta_rel_qmean":           "runoff (mean)",
    "precip_change_pct":         "precip (mean)",
    "temp_change_degC":          "temp (mean)",
    "precip_q99_change_pct":     "precip Q99",
    "temp_q99_change_degC":      "temp Q99",
}

FOCUS = ("ssp585", "2081-2100")               # cell that gets the histogram figure
SEED  = 42
# B = 10,000 so the add-one p-resolution is 1/(B+1) ~ 1e-4: an observed value
# beyond every draw is honestly reportable as p < 1e-4, not pinned at "0.001"
# as it was at B = 1000. Displayed p-strings use "<" at the limit.
B_DEFAULT = 10000


def per_station_model_means(df, col):
    """(station x model) matrix of member-mean changes; model democracy step 1."""
    mm = (df.dropna(subset=[col])
            .groupby(["station", "model"], observed=True)[col].mean()
            .unstack("model"))
    # keep only stations with >= 2 hot and >= 2 retained models present, the
    # same eligibility rule per_station_test applies before the Welch test
    hot_cols = [m for m in EXCLUDED_MODELS  if m in mm.columns]
    ret_cols = [m for m in RETAINED_MODELS if m in mm.columns]
    ok = (mm[hot_cols].notna().sum(axis=1) >= 2) & (mm[ret_cols].notna().sum(axis=1) >= 2)
    return mm.loc[ok]


def masked_stats(M, keep):
    """Ensemble mean and std per station over the kept model columns.

    M    : (S x K) float matrix with NaN where a model is absent
    keep : (K x B) 0/1 matrix, one column per draw
    returns mean (S x B), std (S x B) via the E[x^2] - E[x]^2 identity.
    """
    val   = np.where(np.isnan(M), 0.0, M)
    val2  = val * val
    has   = (~np.isnan(M)).astype(float)
    n     = has @ keep                             # models present AND kept
    n     = np.where(n > 0, n, np.nan)
    mean  = (val @ keep) / n
    ex2   = (val2 @ keep) / n
    var   = np.maximum(ex2 - mean * mean, 0.0) * n / np.where(n > 1, n - 1, np.nan)
    return mean, np.sqrt(var)


def family_block_keep(fam, h, B, rng):
    """(K x B) keep matrix; each column removes EXACTLY h models drawn at the
    family level: whole genealogies are removed together (permuted at random),
    the last family trimmed so the removed count is exactly h. This gives a null
    of genealogically clustered removals comparable to the too-hot set."""
    K       = len(fam)
    fam_arr = np.asarray(fam, dtype=object)
    uniq    = list(dict.fromkeys(fam))                  # families present
    members = [np.where(fam_arr == f)[0] for f in uniq]
    n_fam   = len(uniq)
    keep    = np.ones((K, B))
    for b in range(B):
        removed = []
        for j in rng.permutation(n_fam):
            room = h - len(removed)
            if room <= 0:
                break
            mem = members[j]
            if len(mem) <= room:
                removed.extend(mem.tolist())
            else:
                removed.extend(rng.choice(mem, size=room, replace=False).tolist())
                break
        keep[removed, b] = 0.0
    return keep


def cell_bootstrap(mm, rng, B):
    """Observed screen effect vs random-removal nulls for one cell.

    mm: (station x model) DataFrame of member-mean changes. Returns the observed
    statistics plus the PRIMARY (family-block, genealogy-aware) null and the
    REFERENCE (uniform, exchangeable) null, each with a one-sided p-value.
    """
    models = list(mm.columns)
    K      = len(models)
    hot_ix = np.array([i for i, m in enumerate(models) if m in EXCLUDED_MODELS])
    h      = len(hot_ix)
    if h < 2 or K - h < 2:
        return None
    M   = mm.to_numpy(float)
    fam = [family_of(m) for m in models]
    n_fam_obs = len({fam[i] for i in hot_ix})           # lineages in the screen

    keep_full = np.ones((K, 1))
    keep_obs  = np.ones((K, 1)); keep_obs[hot_ix, 0] = 0.0
    # uniform null (reference): remove h exchangeable models at random
    keep_unif = np.ones((K, B))
    for b in range(B):
        keep_unif[rng.choice(K, size=h, replace=False), b] = 0.0
    # family-block null (primary): remove h models drawn whole-genealogy-wise
    keep_fam = family_block_keep(fam, h, B, rng)

    full_mean, full_std = masked_stats(M, keep_full)
    red_mean,  red_std  = masked_stats(M, keep_obs)
    uni_mean,  uni_std  = masked_stats(M, keep_unif)
    fam_mean,  fam_std  = masked_stats(M, keep_fam)

    def stats_for(red_m, red_s):
        # MEDIAN (not mean) over catchments: robust to the relative-change
        # blow-up on arid/intermittent basins with baseline quantile ~ 0 (HBV
        # Q05 mean bias was ~+88,500 %). Same estimator for observed and both
        # nulls, so the empirical p-value stays valid (flaw-1 fix).
        bias  = np.nanmedian(full_mean - red_m, axis=0)                  # (B,)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.nanmedian(red_s / full_std, axis=0)
        tail  = (np.nanpercentile(full_mean, 90, axis=0)
                 - np.nanpercentile(red_m, 90, axis=0))
        return bias, ratio, tail

    ob, orat, otail = (s[0] for s in stats_for(red_mean, red_std))
    ub, urat, utail = stats_for(uni_mean, uni_std)      # uniform null (ref.)
    fb, frat, ftail = stats_for(fam_mean, fam_std)      # family-block (primary)

    def p_ge(null, obs):  # one-sided, add-one correction
        return (int(np.sum(null >= obs)) + 1) / (B + 1)

    def p_le(null, obs):
        return (int(np.sum(null <= obs)) + 1) / (B + 1)

    return {
        "n_sta": M.shape[0], "n_models": K, "h_removed": h,
        "n_fam_obs": n_fam_obs, "B": B,
        "obs_bias": ob, "obs_spread_ratio": orat, "obs_tailtrim": otail,
        # PRIMARY: genealogy-aware family-block null
        "null_bias": fb, "null_spread_ratio": frat, "null_tailtrim": ftail,
        "p_bias":   p_ge(fb,   ob),
        "p_spread": p_le(frat, orat),
        "p_tail":   p_ge(ftail, otail),
        # REFERENCE: uniform exchangeable null
        "null_bias_unif": ub, "null_spread_ratio_unif": urat,
        "null_tailtrim_unif": utail,
        "p_bias_unif":   p_ge(ub,   ob),
        "p_spread_unif": p_le(urat, orat),
        "p_tail_unif":   p_ge(utail, otail),
    }


def null_summary(row, key):
    arr = row[f"null_{key}"]
    return {f"null_{key}_mean":  float(np.nanmean(arr)),
            f"null_{key}_p2.5":  float(np.nanpercentile(arr, 2.5)),
            f"null_{key}_p97.5": float(np.nanpercentile(arr, 97.5))}


def fmt_p(p, B):
    """Honest p-string: at the add-one resolution limit (no null draw as extreme
    as observed) report 'p < 1/B' rather than the pinned '= 1/(B+1)' (flaw-4)."""
    if p <= 1.0 / (B + 1) + 1e-12:
        return f"p < {1.0 / B:.1g}"
    if p < 1e-3:                       # avoid the misleading '= 0.000'
        return f"p = {p:.1g}"
    return f"p = {p:.3f}"


def histogram_figure(res, title, fname, units):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    B = res["B"]
    panels = [("bias",         "bias",   f"median bias ({SIGNCONV}, {units})"),
              ("spread_ratio", "spread", "median spread ratio (reduced/full)"),
              ("tailtrim",     "tail",   f"p90 upper-tail shift ({SIGNCONV}, {units})")]
    for ax, (key, pk, xlabel) in zip(axes, panels):
        # difference statistics are mirrored to the manuscript orientation;
        # the stored values and their one-sided p remain untouched.
        flip = BIAS_SIGN if key in ("bias", "tailtrim") else 1.0
        fam_draws  = flip * np.asarray(res[f"null_{key}"], float)
        unif_draws = flip * np.asarray(res[f"null_{key}_unif"], float)
        obs = flip * res[f"obs_{key}"]
        both = np.concatenate([fam_draws, unif_draws, [obs]])
        bins = np.linspace(np.nanmin(both), np.nanmax(both), 41)
        ax.hist(fam_draws, bins=bins, color="0.72",
                label=f"family-block null (B={B})")
        ax.hist(unif_draws, bins=bins, histtype="step", color="0.35", lw=1.1,
                label="uniform null")
        ax.axvline(obs, color="firebrick", lw=2,
                   label=_screen_label())
        ax.set_xlabel(xlabel, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_title(f"{fmt_p(res['p_' + pk], B)}"
                     f"   (uniform {fmt_p(res['p_' + pk + '_unif'], B)})",
                     fontsize=11)
    axes[0].set_ylabel("draws", fontsize=11)
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=12)
    add_panel_letters(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  wrote {fname}")


def combined_histogram_figure(rows, suptitle, fname):
    """Stack per-metric random-removal null panels into ONE figure: one metric
    per ROW, three columns (bias / spread ratio / tail trimming), identical to
    histogram_figure. `rows` is an ordered list of dicts with keys 'res',
    'label' and 'units'."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11.0, 2.9 * n), squeeze=False)
    panels = [("bias",         "bias",   f"median bias ({SIGNCONV})"),
              ("spread_ratio", "spread", "median spread ratio (reduced/full)"),
              ("tailtrim",     "tail",   f"p90 upper-tail shift ({SIGNCONV})")]
    for r, row in enumerate(rows):
        res, units, B = row["res"], row["units"], row["res"]["B"]
        for cc, (key, pk, base) in enumerate(panels):
            ax = axes[r][cc]
            # mirror the difference statistics to the manuscript orientation
            flip = BIAS_SIGN if key in ("bias", "tailtrim") else 1.0
            fam_draws  = flip * np.asarray(res[f"null_{key}"], float)
            unif_draws = flip * np.asarray(res[f"null_{key}_unif"], float)
            obs = flip * res[f"obs_{key}"]
            both = np.concatenate([fam_draws, unif_draws, [obs]])
            bins = np.linspace(np.nanmin(both), np.nanmax(both), 41)
            ax.hist(fam_draws, bins=bins, color="0.72",
                    label=f"family-block null (B={B})")
            ax.hist(unif_draws, bins=bins, histtype="step", color="0.35", lw=1.1,
                    label="uniform null")
            ax.axvline(obs, color="firebrick", lw=2,
                       label=_screen_label())
            ax.set_xlabel(base if key == "spread_ratio" else f"{base} ({units})",
                          fontsize=11)
            ax.tick_params(labelsize=10)
            ax.set_title(f"{fmt_p(res['p_' + pk], B)}"
                         f"   (uniform {fmt_p(res['p_' + pk + '_unif'], B)})",
                         fontsize=11)
        axes[r][0].set_ylabel(f"{row['label']}\ndraws", fontsize=11)
    axes[0][0].legend(fontsize=9, frameon=False)
    #fig.suptitle(suptitle, fontsize=13)
    add_panel_letters(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  wrote {fname}")


# ---- results cache: split the costly bootstrap from the cheap plotting ------
# cell_bootstrap (B=10,000 draws per cell) is the only expensive step; the
# figures are redrawn often while tuning layout. The CSV keeps only per-statistic
# summaries, so it cannot redraw the histograms (they need the full B-length
# null-draw arrays). We pickle the FOCUS-cell results (arrays included) plus the
# summary rows, keyed by a signature of everything the bootstrap depends on. A
# later run reuses the pickle and only redraws. To force a recompute, delete the
# pickle or pass --recompute.
def _screen_label():
    """Name the ACTIVE screen for figure legends and titles. Hardcoding
    "TCR > 2.2 K" here would mislabel a two-sided or an ECS run. The spacing is
    kept as it was so the TCR labels are unchanged."""
    lab, lo, hi = em.metric_label(), em.lower_bound(), em.upper_bound()
    if em.is_two_sided():
        return (f"{lab} > {hi} K or {lab} < {lo} K screen")
    return f"{lab} > {hi} K screen"


def _results_cache_path(dataset):
    return em.tag(os.path.join(CACHE_DIR,
                               f"random_removal_null_{dataset}_results.pkl"))


def _signature(dataset, B, cache):
    """Everything the bootstrap output depends on; a mismatch forces a rerun."""
    sig = {"dataset": dataset, "B": B, "seed": SEED,
           "too_hot_threshold": TOO_HOT_TCR_THRESHOLD,
           "full_models": sorted(FULL_MODELS),
           "too_hot_models": sorted(EXCLUDED_MODELS),
           # The analysed column list, added 27 Aug 2026 when RL20 joined
           # FLOW_COLS. Without it a pickle written before a column was added
           # still validates, so the run silently serves a result that never
           # contained the new index. This DOES invalidate every pickle
           # written before that date; the cached CSVs and figures on disk are
           # unaffected, only the next run of an affected treatment pays the
           # recompute.
           "cols": list(FLOW_COLS if dataset == "flow" else CLIMATE_COLS),
           "input_cache": os.path.abspath(cache),
           "input_mtime": os.path.getmtime(cache),
           "input_size": os.path.getsize(cache)}
    if em.is_ecs():
        # Added ONLY under ECS so the TCR signature stays byte-identical and
        # every pickle already on disk keeps validating. The ECS pickle lands
        # under a different (tagged) filename anyway; these keys just make the
        # blob self-describing.
        sig["screen_metric"] = em.metric()
        sig["screen_cut"] = em.upper_bound()
    return sig


def _load_cached_results(dataset, B, cache):
    """Return the results blob if the pickle exists and its signature matches the
    current B, seed, model universe and input parquet; else None (-> recompute)."""
    path = _results_cache_path(dataset)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
    except Exception as exc:                       # corrupt/legacy pickle
        print(f"  results cache unreadable ({exc}); recomputing")
        return None
    want, got = _signature(dataset, B, cache), blob.get("signature", {})
    stale = [k for k in want if want[k] != got.get(k)]
    if stale:
        print(f"  results cache stale on {stale}; recomputing")
        return None
    print(f"  reusing cached bootstrap results: {path}")
    return blob


def _save_cached_results(blob):
    path = _results_cache_path(blob["dataset"])
    with open(path, "wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote bootstrap results cache: {path}")


def compute_results(dataset, B, cache, cols, idcols):
    """Run the (costly) random-removal bootstrap over every cell and return a
    picklable results blob: summary rows plus the FOCUS-cell null-draw arrays."""
    # read only the needed columns (schema check first, so a missing column
    # never forces a full read of the 5.3 GB flow cache)
    import pyarrow.parquet as pq
    schema_names = set(pq.ParquetFile(cache).schema_arrow.names)
    cols = [c for c in cols if c in schema_names]
    missing_id = [c for c in idcols if c not in schema_names]
    if missing_id:
        sys.exit(f"{cache} lacks required id column(s) {missing_id}")
    if not cols:
        sys.exit(f"none of the requested metric columns exist in {cache}")
    df = pd.read_parquet(cache, columns=idcols + cols)
    df = apply_blocklist(df, label=f"{dataset} random-removal-null cache")
    df = df[df["model"].isin(FULL_MODELS)]
    df = df[df["scenario"].astype(str) != "baseline"]
    if dataset == "flow":
        df[cols] = df[cols].astype(float) * 100.0     # fractions -> %
    df = df.replace([np.inf, -np.inf], np.nan)

    engines = sorted(df["engine"].unique()) if dataset == "flow" else [""]
    rng  = np.random.default_rng(SEED)
    rows = []
    focus_cells = {}       # (engine, column) -> {"res", "units"} for the FOCUS cell
    for eng in engines:
        sub_e = df[df["engine"] == eng] if eng else df
        for (scen, per), sub in sub_e.groupby(["scenario", "period"], observed=True):
            for col in cols:
                if col not in sub.columns:
                    continue
                mm = per_station_model_means(sub, col)
                if mm.empty:
                    continue
                res = cell_bootstrap(mm, rng, B)
                if res is None:
                    continue
                out = {"dataset": dataset, "engine": eng, "column": col,
                       "scenario": scen, "period": per,
                       "n_sta": res["n_sta"], "n_models": res["n_models"],
                       "h_removed": res["h_removed"], "n_fam_obs": res["n_fam_obs"],
                       "B": B,
                       "obs_bias": res["obs_bias"], "p_bias": res["p_bias"],
                       "p_bias_unif": res["p_bias_unif"],
                       "obs_spread_ratio": res["obs_spread_ratio"],
                       "p_spread": res["p_spread"], "p_spread_unif": res["p_spread_unif"],
                       "obs_tailtrim": res["obs_tailtrim"], "p_tail": res["p_tail"],
                       "p_tail_unif": res["p_tail_unif"]}
                for key in ("bias", "spread_ratio", "tailtrim"):
                    out.update(null_summary(res, key))
                rows.append(out)
                print(f"[{dataset}{'/' + eng if eng else ''}] {col} {scen} {per}: "
                      f"bias={res['obs_bias']:.3f} ({fmt_p(res['p_bias'], B)})  "
                      f"spread={res['obs_spread_ratio']:.3f} ({fmt_p(res['p_spread'], B)})  "
                      f"tail={res['obs_tailtrim']:.3f} ({fmt_p(res['p_tail'], B)})  "
                      f"[{res['n_fam_obs']} fam]")
                if (scen, per) == FOCUS:
                    focus_cells[(eng, col)] = {
                        "res": res,
                        "units": "%" if "degC" not in col else "degC"}
    return {"signature": _signature(dataset, B, cache), "dataset": dataset,
            "B": B, "engines": engines, "cols": cols,
            "rows": rows, "focus_cells": focus_cells}


def make_figures(blob):
    """Draw the FOCUS-cell null figures and write the summary CSV from a results
    blob (freshly computed or loaded from the pickle cache)."""
    dataset     = blob["dataset"]
    B           = blob["B"]
    engines     = blob["engines"]
    cols        = blob["cols"]
    rows        = blob["rows"]
    focus_cells = blob["focus_cells"]
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- combined FOCUS-cell figures: one metric per row --------------------
    # flow -> one figure per engine (4 magnitude metrics); climate -> one figure
    # (4 metrics). Columns bootstrapped only for the Supplementary table (flow
    # Q99, flashiness) are NOT drawn.
    combined_order = (CLIMATE_COMBINED_ROWS if dataset == "climate"
                      else FLOW_COMBINED_ROWS)
    scen, per = FOCUS
    for eng in engines:
        entries = [{"res":   focus_cells[(eng, c)]["res"],
                    "units": focus_cells[(eng, c)]["units"],
                    "label": ROW_LABELS.get(c, c)}
                   for c in combined_order if (eng, c) in focus_cells]
        if entries:
            n_comb  = max(e["res"]["n_sta"] for e in entries)
            eng_tag = f"{dataset}_{eng}" if eng else dataset
            title_e = f"({eng}) " if eng else ""
            fname   = em.tag(os.path.join(
                FIG_DIR, f"{eng_tag}_{scen}_{per}_n{n_comb}sta.png"))
            combined_histogram_figure(
                entries,
                f"Random-removal null  {title_e}{scen} {per}:  "
                f"{_screen_label()} vs {B} genealogy-aware removals",
                fname)

    out_csv = em.tag(os.path.join(CACHE_DIR,
                                  f"random_removal_null_{dataset}.csv"))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(rows)} cells)")


def run_dataset(dataset, B, recompute=False):
    cache = FLOW_CACHE if dataset == "flow" else CLIMATE_CACHE
    if not os.path.exists(cache):
        sys.exit(f"cache not found: {cache}")
    cols   = FLOW_COLS if dataset == "flow" else CLIMATE_COLS
    idcols = ["station", "scenario", "period", "model", "member"]
    if dataset == "flow":
        idcols.append("engine")

    blob = None if recompute else _load_cached_results(dataset, B, cache)
    if blob is None:
        blob = compute_results(dataset, B, cache, cols, idcols)
        _save_cached_results(blob)
    make_figures(blob)


if __name__ == "__main__":
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment()
    # This null (random REMOVAL of h models) is meaningful only for the
    # exclusion method. The weighting analogue (random permutation of the
    # weight vector across models / family blocks) is not implemented yet;
    # fail loudly rather than produce a mislabelled null.
    _rest = em.reject_cli_tokens(sys.argv[1:])
    if em.is_weighting():
        sys.exit("bootstrap_random_removal.py is a removal-mode null; the "
                 "weight-permutation null for method=weighting is not "
                 "implemented yet.")
    sys.argv = sys.argv[:1] + _rest
    # --recompute / --force / --no-cache: ignore any existing pickle and rerun
    # the bootstrap (also refreshes the cache). Positional args keep their order.
    argv = sys.argv[1:]
    recompute = any(f in argv for f in ("--recompute", "--force", "--no-cache"))
    argv = [a for a in argv if not a.startswith("--")]
    arg = argv[0] if argv else "all"
    B   = int(argv[1]) if len(argv) > 1 else B_DEFAULT
    print(f"FULL={len(FULL_MODELS)}  EXCLUDED={len(EXCLUDED_MODELS)} "
          f"({em.screen_expr()})  RETAINED={len(RETAINED_MODELS)}  B={B}"
          f"{'  [recompute]' if recompute else ''}")
    for ds in (["climate", "flow"] if arg == "all" else [arg]):
        run_dataset(ds, B, recompute=recompute)
