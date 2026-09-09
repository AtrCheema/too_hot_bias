"""
weight_permutation_null.py  (outputs are named weight_permutation_null_*)
=========================================================================

The WEIGHTING-mode counterpart of bootstrap_random_removal.py: the aggregate
test of whether the projected change produced by a weighted ensemble is
distinguishable from what an arbitrary assignment of the SAME weights would
have produced.

bootstrap_random_removal.py exits under METHOD="weighting" and says so
explicitly ("the weight-permutation null for method=weighting is not
implemented yet"). This script is that null. It is a SEPARATE file on purpose:
the removal script is the number authority for every published removal result,
so it is not edited here.

The two nulls are exact analogues:

    removal      observed = drop the h screened models
                 null     = drop h models at random, B times
    weighting    observed = weight every model by w_m
                 null     = reassign the SAME weight VALUES to models at
                            random, B times

Because the null is a PERMUTATION, the multiset of weights, the effective
ensemble size, the weight span and the degree of compression are all held
fixed; only the model -> weight assignment is destroyed. The test therefore
isolates the one thing the scheme claims: that weighting by climate
sensitivity is different from weighting arbitrarily. It is the aggregate
version of the per-catchment `p_perm` column already written by
weight_spread_ratio.py.

Three statistics per (column, scenario, period[, engine]) cell, each with a
one-sided empirical p-value, matching bootstrap_random_removal.py so the two
tables can be read side by side:

  bias      median over catchments of (full_mean - weighted_mean) per station,
            aggregated member -> model -> ensemble. The MEDIAN, not the mean,
            because relative-change indices blow up on arid/intermittent basins
            with a baseline quantile near zero; the same estimator is used for
            the observed value and for every null draw, so the p-value stays
            valid. p = fraction of draws with bias >= observed.
  spread    median over catchments of sigma_weighted / sigma_full, where
            sigma_full is the ordinary unbiased standard deviation of the
            per-model mean changes and sigma_weighted its reliability-weighted
            counterpart, sum(w_i (x_i - xbar_w)^2) / (1 - sum w_i^2). That
            estimator reduces EXACTLY to the unbiased sample variance when the
            weights are equal, so numerator and denominator are commensurate;
            _check_equal_weight_identity() asserts this at run time against
            bootstrap_random_removal.masked_stats, testing the bulk of the
            network at 1e-9 and capping the ill-conditioned tail at 1e-5 (see
            that function for why a bare max() is the wrong comparison).
            p = fraction of draws with ratio <= observed.
  tailtrim  p90 across catchments of the per-station ensemble-mean change,
            full minus weighted ("upper-tail trimming").
            p = fraction of draws with tail trim >= observed.

Two permutation nulls per cell
------------------------------
  * family-block (REFERENCE) -- whole genealogies keep their weight BLOCK
    together, and the blocks are re-assigned among families of the SAME SIZE.
    This preserves the weight multiset exactly while keeping the null draws as
    genealogically coherent as the observed weighting is: under an ECS-fitted
    scheme, whole lineages (EC-Earth3, HadGEM3/UKESM, CNRM, ...) carry similar
    weights, so a null that scatters those weights across unrelated models
    would understate how easily a coherent assignment arises. Same reasoning as
    the family-block removal null of bootstrap_random_removal.py, and the same
    reference (Knutti et al. 2013; Brunner et al. 2020). p_* columns are this
    null.
    LIMITATION, state it wherever the p-value is quoted: blocks can only swap
    with families of equal size, so a family with a unique size cannot move at
    all. n_perm_distinct in the CSV reports how many distinct permutations the
    construction admits; read the p-value against that number, not against B.
  * uniform (PRIMARY) -- a plain random permutation of the K weights. This is
    the null weight_spread_ratio.py already applies per catchment, so the
    aggregate and per-catchment weighting verdicts come from the same
    construction. p_*_unif columns.

Uses ONLY the existing Stage-1 caches (no new simulations):
    <CONFIG.out_path>/<delta_method>/fig_data/fig02_change.parquet   (climate)
    <CONFIG.out_path>/<delta_method>/fig_data/stats_change.parquet   (flow)

Outputs (every name carries the active weighting tag, e.g. _ecs_wt):
    <fig_data>/weight_permutation_null_<dataset>.csv              summary table
    <fig_data>/weight_permutation_null_<dataset>_results.pkl      results cache
    paper/figures/significance/weight_permutation_null/
        <dataset>[_<engine>]_<scenario>_<period>_n<n>sta.png      null figure

Run (via slurm only). The treatment must already be a weighting one; the
script sets METHOD="weighting" itself but NOT the variant, so select the
variant either by editing ensemble_method.VARIANT or through
run_ecs_weighting.py:

    python paper/figures/scripts/weight_permutation_null.py climate
    python paper/figures/scripts/weight_permutation_null.py flow
    python paper/figures/scripts/weight_permutation_null.py all
    python paper/figures/scripts/weight_permutation_null.py all 2000
    python paper/figures/scripts/weight_permutation_null.py all --recompute
    python paper/figures/scripts/weight_permutation_null.py all --figures-only
"""

import os
import sys
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))            # paper/figures/scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
sys.path.insert(0, _THIS)

import ensemble_method as em
from config import CONFIG
from realization_blocklist import apply_blocklist
from figstyle import add_panel_letters, percentile_label

# The whole point of this file is to reuse the removal null's machinery rather
# than re-derive it: same caches, same eligibility rule, same genealogy map,
# same p-string formatting, same metric lists and row labels. Importing the
# module does not run its bootstrap (that lives under __main__).
import bootstrap_random_removal as brr

CACHE_DIR     = brr.CACHE_DIR
CLIMATE_CACHE = brr.CLIMATE_CACHE
FLOW_CACHE    = brr.FLOW_CACHE
FIG_DIR       = os.path.join(os.path.dirname(_THIS), "significance",
                             "weight_permutation_null")

# Eligibility: keep a station only if at least MIN_PER_SIDE models sit on each
# side of the ACTIVE metric's AR6 upper bound. This is the same rule
# bootstrap_random_removal.per_station_model_means applies, but resolved from
# em at CALL time rather than from module-level constants frozen at import, so
# the catchment set cannot depend on whether this file was imported before or
# after the treatment was switched to weighting.
MIN_PER_SIDE = 2

CLIMATE_COLS = brr.CLIMATE_COLS
FLOW_COLS    = brr.FLOW_COLS
FLOW_COMBINED_ROWS    = brr.FLOW_COMBINED_ROWS
CLIMATE_COMBINED_ROWS = brr.CLIMATE_COMBINED_ROWS
# Row labels follow the project's percentile-symbol convention
# (figstyle.PERCENTILE_LABELS, added 2026-08-17): P / T / Q with an italic `p`
# subscript for percentiles and an upright `mean` subscript for means. The bare
# "Q95" that bootstrap_random_removal.ROW_LABELS carries is exactly the string
# that table was written to retire -- it reads as a HIGH flow here and as a LOW
# flow under the flow-duration-curve convention. brr.ROW_LABELS itself is left
# untouched: it is the removal figures' label set and this file must not edit
# the removal authority. figstyle deliberately covers no AnnMax / RL20 key
# (those were never ambiguous), so those two keep a descriptive label.
_UNCOVERED_LABELS = {"delta_rel_mean_annual_max": "AnnMax",
                     "delta_rel_flood_q20":       "Flood RL20"}


def _row_label(col):
    lab = percentile_label(col)
    if lab is not None:
        return lab.short
    return _UNCOVERED_LABELS.get(col, brr.ROW_LABELS.get(col, col))


ROW_LABELS   = {c: _row_label(c) for c in
                (set(brr.ROW_LABELS) | set(_UNCOVERED_LABELS))}
FOCUS        = brr.FOCUS
SEED         = brr.SEED
B_DEFAULT    = brr.B_DEFAULT

# SIGN CONVENTION, matching bootstrap_random_removal.py: the stored results keep
# the original orientation (bias = full - weighted) so the one-sided p-values
# are untouched; BIAS_SIGN mirrors ONLY the drawn histograms of the two
# difference statistics. The spread ratio is orientation-free.
BIAS_SIGN = brr.BIAS_SIGN
SIGNCONV  = "full - weighted" if BIAS_SIGN > 0 else "weighted - full"

# Draws are processed in chunks so the (stations x draws) arrays stay small:
# 20,600 x 10,000 float64 is 1.6 GB per array and there are several of them.
CHUNK = 500


# ---------------------------------------------------------------------------
#  station eligibility
# ---------------------------------------------------------------------------
def per_station_model_means(df, col):
    """(station x model) matrix of member-mean changes; model democracy step 1.

    Eligibility follows MIN_PER_SIDE above."""
    mm = (df.dropna(subset=[col])
            .groupby(["station", "model"], observed=True)[col].mean()
            .unstack("model"))
    cut = em.upper_bound()
    above = [m for m in mm.columns if em.metric_value(m) > cut]
    below = [m for m in mm.columns if em.metric_value(m) <= cut]
    if not above or not below:
        raise RuntimeError(
            f"no models on one side of {em.metric_label()} {cut}; the "
            f"eligibility rule cannot be applied")
    ok = ((mm[above].notna().sum(axis=1) >= MIN_PER_SIDE)
          & (mm[below].notna().sum(axis=1) >= MIN_PER_SIDE))
    return mm.loc[ok]


# ---------------------------------------------------------------------------
#  weighted statistics
# ---------------------------------------------------------------------------
def weighted_stats(M, W):
    """Weighted ensemble mean and standard deviation per station.

    M : (S x K) float matrix of per-model mean changes, NaN where absent
    W : (K x B) matrix of weight vectors, one column per draw (any positive
        normalisation; it cancels)
    returns mean (S x B), std (S x B)

    The variance is the reliability-weighted unbiased estimator

        var = sum_i p_i (x_i - xbar)^2 / (1 - sum_i p_i^2),

    with p the weights renormalised over the models PRESENT at that station.
    With equal weights sum p^2 = 1/n, so the denominator is (n-1)/n and the
    expression collapses to the ordinary unbiased sample variance -- which is
    what bootstrap_random_removal.masked_stats computes. That identity is what
    makes sigma_weighted/sigma_full a meaningful ratio, and it is asserted at
    run time by _check_equal_weight_identity()."""
    val = np.where(np.isnan(M), 0.0, M)
    has = (~np.isnan(M)).astype(float)
    den = has @ W                                  # weight present per station
    den = np.where(den > 0, den, np.nan)
    mean = (val @ W) / den
    ex2 = ((val * val) @ W) / den
    sum_p2 = (has @ (W * W)) / (den * den)
    corr = 1.0 - sum_p2                            # 0 when one model carries all
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(ex2 - mean * mean, 0.0) / np.where(corr > 0, corr,
                                                            np.nan)
    return mean, np.sqrt(var)


_IDENTITY_TAIL_WARNED = set()


def _check_equal_weight_identity(M, rtol=1e-9, tail_tol=1e-5, tail_q=99.9,
                                 label=""):
    """weighted_stats() with equal weights must reproduce masked_stats().

    WHY THIS IS NOT A PLAIN max() COMPARISON. Both routines take the variance
    from the one-pass identity E[x^2] - E[x]^2. That identity loses relative
    precision as the mean grows against the spread: its condition number is
    kappa = 1 + mean^2 / var, and a float64 evaluation carries a relative error
    of order eps * kappa in the variance. The two routines apply the SAME
    algebra in a different operation order (masked_stats divides by the model
    count, weighted_stats multiplies by 1/K first), so their disagreement is
    bounded by kappa and is not evidence of a different estimator.

    On the flow indices a handful of catchments have models that agree closely
    on a very large relative change -- Q05 and the mean annual minimum reach
    median changes of order 1e5 % -- so mean/sigma runs to 1e3-1e4 there,
    kappa to 1e7-1e8, and the relative difference in sigma reaches ~1e-9 at
    those catchments while sitting at machine precision across the other
    20,000. A single max() over the network therefore fails on arithmetic
    noise, which is what killed the flow pass of job 50790519.

    A genuine estimator mismatch (a wrong Bessel-type correction, a different
    normalisation, weights renormalised over the wrong set) shifts EVERY
    catchment and so moves the bulk of the distribution. Cancellation noise
    moves only the tail. The check tests the bulk at `rtol`, keeps a loose cap
    `tail_tol` on the tail, and reports the conditioning when it fires so the
    two causes can be told apart from the error message alone."""
    K = M.shape[1]
    ref_mean, ref_std = brr.masked_stats(M, np.ones((K, 1)))
    got_mean, got_std = weighted_stats(M, np.full((K, 1), 1.0 / K))
    for name, a, b in (("mean", ref_mean, got_mean), ("std", ref_std, got_std)):
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            continue
        scale = np.maximum(np.maximum(np.abs(a[ok]), np.abs(b[ok])), 1e-12)
        err = np.abs(a[ok] - b[ok]) / scale
        bulk = float(np.percentile(err, tail_q))
        worst = float(err.max())
        if bulk <= rtol and worst <= tail_tol:
            if worst > rtol and label not in _IDENTITY_TAIL_WARNED:
                # report once per cell label: the identity holds, but say so
                # rather than passing an ill-conditioned cell silently
                _IDENTITY_TAIL_WARNED.add(label)
                print(f"  note[{label or 'cell'}]: equal-weight identity holds "
                      f"(p{tail_q} rel. diff {bulk:.2e} <= {rtol:.0e}); worst "
                      f"catchment {worst:.2e} reflects the conditioning of the "
                      f"E[x^2]-E[x]^2 identity, not a different estimator")
            continue
        # ---- failure: report the conditioning so the cause is identifiable ---
        with np.errstate(invalid="ignore", divide="ignore"):
            kappa = 1.0 + (ref_mean[ok] / np.where(ref_std[ok] > 0,
                                                   ref_std[ok], np.nan)) ** 2
        kmax = float(np.nanmax(kappa)) if np.isfinite(kappa).any() else np.nan
        verdict = ("the bulk of the network disagrees, so this is a genuine "
                   "estimator mismatch, not arithmetic noise"
                   if bulk > rtol else
                   "the bulk agrees but the tail exceeds the cap; inspect the "
                   "worst catchments before loosening tail_tol")
        raise RuntimeError(
            f"weighted_stats does not reduce to masked_stats for the {name} "
            f"[{label or 'cell'}]: median rel. diff "
            f"{float(np.median(err)):.3e}, p{tail_q} {bulk:.3e}, max {worst:.3e} "
            f"(bulk tol {rtol:.0e}, tail cap {tail_tol:.0e}); worst one-pass "
            f"condition number 1+mean^2/var = {kmax:.3e}. {verdict}. The spread "
            f"ratio would compare two different estimators")


# ---------------------------------------------------------------------------
#  the two permutation nulls
# ---------------------------------------------------------------------------
def uniform_permute(w, B, rng):
    """(K x B) plain random permutations of the weight vector `w` -- the
    REFERENCE null, and the same construction weight_spread_ratio.py applies
    per catchment."""
    w = np.asarray(w, float)
    return np.stack([rng.permutation(w) for _ in range(B)], axis=1)


def family_block_permute(fam, w, B, rng):
    """(K x B) genealogy-coherent permutations of `w` -- the PRIMARY null.

    Whole families keep their weight BLOCK together and the blocks are
    re-assigned among families of the SAME SIZE, so the multiset of weights is
    preserved exactly (it is a true permutation) while the null draws stay as
    genealogically coherent as the observed weighting.

    Returns (W, n_distinct) where n_distinct is the number of distinct
    permutations the construction admits, product over sizes of (number of
    families of that size)!. A family whose size is unique cannot move, so
    n_distinct must be reported next to any p-value from this null."""
    w = np.asarray(w, float)
    fam_arr = np.asarray(fam, dtype=object)
    uniq = list(dict.fromkeys(fam))
    members = [np.where(fam_arr == f)[0] for f in uniq]
    by_size = {}
    for j, mem in enumerate(members):
        by_size.setdefault(len(mem), []).append(j)

    n_distinct = 1
    for fam_idx in by_size.values():
        n_distinct *= math.factorial(len(fam_idx))

    W = np.empty((len(w), B))
    for b in range(B):
        col = np.empty(len(w))
        for fam_idx in by_size.values():
            order = rng.permutation(len(fam_idx))
            for src, dst in zip(fam_idx, [fam_idx[o] for o in order]):
                col[members[dst]] = w[members[src]]   # block src -> family dst
        W[:, b] = col
    return W, n_distinct


# ---------------------------------------------------------------------------
#  per-cell test
# ---------------------------------------------------------------------------
def _cell_statistics(M, full_mean, full_std, W):
    """bias / spread ratio / tail trim for every column of W, chunked over
    draws so the (stations x draws) arrays stay bounded."""
    bias, ratio, tail = [], [], []
    for lo in range(0, W.shape[1], CHUNK):
        wm, ws = weighted_stats(M, W[:, lo:lo + CHUNK])
        bias.append(np.nanmedian(full_mean - wm, axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio.append(np.nanmedian(ws / full_std, axis=0))
        tail.append(np.nanpercentile(full_mean, 90, axis=0)
                    - np.nanpercentile(wm, 90, axis=0))
    return (np.concatenate(bias), np.concatenate(ratio), np.concatenate(tail))


def cell_permutation_test(mm, rng, B, label=""):
    """Observed weighting effect vs the two permutation nulls for one cell.

    mm: (station x model) DataFrame of member-mean changes. Returns the same
    keys bootstrap_random_removal.cell_bootstrap returns, so the two summary
    tables have the same shape, plus n_perm_distinct.
    label: cell identifier used only in the equal-weight identity diagnostic."""
    models = list(mm.columns)
    K = len(models)
    if K < 4:                       # a permutation over <4 models is degenerate
        return None
    M = mm.to_numpy(float)
    fam = [brr.family_of(m) for m in models]

    _check_equal_weight_identity(M, label=label)
    w = em.weight_array(models)                 # frozen weights, renormalised
    full_mean, full_std = weighted_stats(M, np.full((K, 1), 1.0 / K))
    full_mean, full_std = full_mean[:, 0], full_std[:, 0]

    W_obs = w.reshape(K, 1)
    W_unif = uniform_permute(w, B, rng)
    W_fam, n_distinct = family_block_permute(fam, w, B, rng)

    ob, orat, otail = (s[0] for s in
                       _cell_statistics(M, full_mean[:, None],
                                        full_std[:, None], W_obs))
    ub, urat, utail = _cell_statistics(M, full_mean[:, None],
                                       full_std[:, None], W_unif)
    fb, frat, ftail = _cell_statistics(M, full_mean[:, None],
                                       full_std[:, None], W_fam)

    def p_ge(null, obs):            # one-sided, add-one correction
        return (int(np.sum(null >= obs)) + 1) / (B + 1)

    def p_le(null, obs):
        return (int(np.sum(null <= obs)) + 1) / (B + 1)

    from model_weights import n_eff, renormalize
    return {
        "n_sta": M.shape[0], "n_models": K, "h_removed": 0,
        "n_fam_obs": len(set(fam)), "n_perm_distinct": n_distinct,
        "n_eff_weights": n_eff(renormalize(em.get_weights(), models)),
        "B": B,
        "obs_bias": ob, "obs_spread_ratio": orat, "obs_tailtrim": otail,
        # PRIMARY: genealogy-coherent block permutation
        "null_bias": fb, "null_spread_ratio": frat, "null_tailtrim": ftail,
        "p_bias":   p_ge(fb,   ob),
        "p_spread": p_le(frat, orat),
        "p_tail":   p_ge(ftail, otail),
        # REFERENCE: plain permutation (weight_spread_ratio.py's null)
        "null_bias_unif": ub, "null_spread_ratio_unif": urat,
        "null_tailtrim_unif": utail,
        "p_bias_unif":   p_ge(ub,   ob),
        "p_spread_unif": p_le(urat, orat),
        "p_tail_unif":   p_ge(utail, otail),
    }


# ---------------------------------------------------------------------------
#  figures
# ---------------------------------------------------------------------------
def _weight_label():
    """Name the ACTIVE weighting for figure legends and titles."""
    return f"{em.variant()} weights"


def fmt_p(p, B):
    """As bootstrap_random_removal.fmt_p, but TWO SIGNIFICANT FIGURES below the
    resolution limit instead of three decimals.

    brr's "%.3f" renders 0.0016 as "0.002" and 0.0011 as "0.001", while
    Supplementary Table S2 quotes both to two significant figures. The number
    is the same, but a reader comparing the panel with the table sees two
    different strings -- the exact complaint that prompted the 30 Aug 2026
    re-basing of that table. Two significant figures makes panel and table
    agree digit for digit. brr.fmt_p itself is left alone: it belongs to the
    removal figures and this file must not edit the removal authority.
    """
    if p <= 1.0 / (B + 1) + 1e-12:
        return f"p < {1.0 / B:.1g}"
    # two significant figures WITH trailing zeros ("p = 0.30", not "p = 0.3"),
    # capped at four decimals so the smallest resolvable values read 0.0002
    # rather than 0.00020 -- the convention Supplementary Table S2 uses.
    d = min(4, max(2, 1 - int(math.floor(math.log10(p)))))
    return f"p = {p:.{d}f}"


def combined_histogram_figure(rows, fname, show_family_block=False):
    """One metric per ROW, three columns (bias / spread ratio / tail trim);
    the same layout as bootstrap_random_removal.combined_histogram_figure, with
    the permutation nulls in place of the removal ones.

    show_family_block=False (DEFAULT) draws ONLY the uniform permutation null,
    as the filled histogram, and annotates only its p-value. The family-block
    null collapses the model set into same-size exchangeable blocks and has
    little power at that number of blocks, so it is reported as a conservative
    bound rather than as the manuscript's null; drawing it filled while
    quoting the uniform p-value in the text would put a distribution on the
    page that no quoted number comes from.
    show_family_block=True restores the two-null figure (family-block filled,
    uniform outlined, both p-values annotated) for the sensitivity analysis.
    """
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11.0, 2.9 * n), squeeze=False)
    panels = [("bias",         "bias",   f"Median bias ({SIGNCONV})"),
              ("spread_ratio", "spread", "Median spread ratio (weighted/full)"),
              ("tailtrim",     "tail",   f"P90 upper-tail shift ({SIGNCONV})")]
    for r, row in enumerate(rows):
        res, units, B = row["res"], row["units"], row["res"]["B"]
        for cc, (key, pk, base) in enumerate(panels):
            ax = axes[r][cc]
            flip = BIAS_SIGN if key in ("bias", "tailtrim") else 1.0
            fam_draws = flip * np.asarray(res[f"null_{key}"], float)
            unif_draws = flip * np.asarray(res[f"null_{key}_unif"], float)
            obs = flip * res[f"obs_{key}"]
            drawn = ([fam_draws, unif_draws] if show_family_block
                     else [unif_draws])
            both = np.concatenate(drawn + [[obs]])
            bins = np.linspace(np.nanmin(both), np.nanmax(both), 41)
            if show_family_block:
                ax.hist(fam_draws, bins=bins, color="0.72",
                        label=f"Family-block permutation (B={B})")
                ax.hist(unif_draws, bins=bins, histtype="step", color="0.35",
                        lw=1.1, label="Uniform permutation")
                title = (f"{fmt_p(res['p_' + pk], B)}   "
                         f"(uniform {fmt_p(res['p_' + pk + '_unif'], B)})")
            else:
                # the uniform null is the only one drawn, so it takes the
                # filled style and its p-value is the only one annotated
                ax.hist(unif_draws, bins=bins, color="0.72",
                        label=f"Uniform permutation (B={B})")
                title = fmt_p(res["p_" + pk + "_unif"], B)
            ax.axvline(obs, color="firebrick", lw=2, label=_weight_label())
            ax.set_xlabel(base if key == "spread_ratio"
                          else f"{base} ({units})", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.set_title(title, fontsize=11)
        axes[r][0].set_ylabel(f"{row['label']}\ndraws", fontsize=11)
    axes[0][0].legend(fontsize=9, frameon=False)
    add_panel_letters(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  wrote {fname}")


# ---------------------------------------------------------------------------
#  results cache (same split as bootstrap_random_removal: costly draws once,
#  cheap redraws afterwards)
# ---------------------------------------------------------------------------
def _results_cache_path(dataset):
    return em.tag(os.path.join(
        CACHE_DIR, f"weight_permutation_null_{dataset}_results.pkl"))


def _signature(dataset, B, cache):
    """Everything the permutation test depends on; a mismatch forces a rerun."""
    weights = em.get_weights()
    return {"dataset": dataset, "B": B, "seed": SEED,
            "variant": em.variant(),
            "weight_column": em.weight_column(),
            # the analysed column list: without it, adding an index to
            # brr.FLOW_COLS silently returns a cached result computed before
            # that index existed (found 27 Aug 2026 when RL20 was added)
            "cols": list(brr.FLOW_COLS if dataset == "flow"
                         else brr.CLIMATE_COLS),
            "full_models": sorted(brr.FULL_MODELS),
            # the weight VALUES themselves, so a refit of the target gamma
            # invalidates the cache even though the column name is unchanged
            "weights": [round(float(weights[m]), 12)
                        for m in sorted(brr.FULL_MODELS)],
            "input_cache": os.path.abspath(cache),
            "input_mtime": os.path.getmtime(cache),
            "input_size": os.path.getsize(cache)}


def _load_cached_results(dataset, B, cache):
    path = _results_cache_path(dataset)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
    except Exception as exc:
        print(f"  results cache unreadable ({exc}); recomputing")
        return None
    want, got = _signature(dataset, B, cache), blob.get("signature", {})
    stale = [k for k in want if want[k] != got.get(k)]
    if stale:
        print(f"  results cache stale on {stale}; recomputing")
        return None
    print(f"  reusing cached permutation results: {path}")
    return blob


def _save_cached_results(blob):
    path = _results_cache_path(blob["dataset"])
    with open(path, "wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote permutation results cache: {path}")


def compute_results(dataset, B, cache, cols, idcols):
    """Run the permutation test over every cell and return a picklable blob."""
    import pyarrow.parquet as pq
    schema_names = set(pq.ParquetFile(cache).schema_arrow.names)
    cols = [c for c in cols if c in schema_names]
    missing_id = [c for c in idcols if c not in schema_names]
    if missing_id:
        sys.exit(f"{cache} lacks required id column(s) {missing_id}")
    if not cols:
        sys.exit(f"none of the requested metric columns exist in {cache}")
    df = pd.read_parquet(cache, columns=idcols + cols)
    df = apply_blocklist(df, label=f"{dataset} weight-permutation-null cache")
    df = df[df["model"].isin(brr.FULL_MODELS)]
    df = df[df["scenario"].astype(str) != "baseline"]
    if dataset == "flow":
        df[cols] = df[cols].astype(float) * 100.0     # fractions -> %
    df = df.replace([np.inf, -np.inf], np.nan)

    engines = sorted(df["engine"].unique()) if dataset == "flow" else [""]
    rng = np.random.default_rng(SEED)
    rows = []
    focus_cells = {}
    for eng in engines:
        sub_e = df[df["engine"] == eng] if eng else df
        for (scen, per), sub in sub_e.groupby(["scenario", "period"],
                                              observed=True):
            for col in cols:
                if col not in sub.columns:
                    continue
                mm = per_station_model_means(sub, col)
                if mm.empty:
                    continue
                cell_label = "/".join(x for x in (dataset, eng, col, scen, per)
                                      if x)
                res = cell_permutation_test(mm, rng, B, label=cell_label)
                if res is None:
                    continue
                out = {"dataset": dataset, "engine": eng, "column": col,
                       "scenario": scen, "period": per,
                       "variant": em.variant(),
                       "weight_column": em.weight_column(),
                       "n_sta": res["n_sta"], "n_models": res["n_models"],
                       "n_fam_obs": res["n_fam_obs"],
                       "n_perm_distinct": res["n_perm_distinct"],
                       "n_eff_weights": res["n_eff_weights"], "B": B,
                       "obs_bias": res["obs_bias"], "p_bias": res["p_bias"],
                       "p_bias_unif": res["p_bias_unif"],
                       "obs_spread_ratio": res["obs_spread_ratio"],
                       "p_spread": res["p_spread"],
                       "p_spread_unif": res["p_spread_unif"],
                       "obs_tailtrim": res["obs_tailtrim"],
                       "p_tail": res["p_tail"],
                       "p_tail_unif": res["p_tail_unif"]}
                for key in ("bias", "spread_ratio", "tailtrim"):
                    out.update(brr.null_summary(res, key))
                rows.append(out)
                print(f"[{dataset}{'/' + eng if eng else ''}] {col} {scen} "
                      f"{per}: bias={res['obs_bias']:.3f} "
                      f"({brr.fmt_p(res['p_bias'], B)})  "
                      f"spread={res['obs_spread_ratio']:.3f} "
                      f"({brr.fmt_p(res['p_spread'], B)})  "
                      f"tail={res['obs_tailtrim']:.3f} "
                      f"({brr.fmt_p(res['p_tail'], B)})  "
                      f"[{res['n_fam_obs']} fam, N_eff "
                      f"{res['n_eff_weights']:.1f}]")
                if (scen, per) == FOCUS:
                    focus_cells[(eng, col)] = {
                        "res": res,
                        "units": "%" if "degC" not in col else "degC"}
    return {"signature": _signature(dataset, B, cache), "dataset": dataset,
            "B": B, "engines": engines, "cols": cols,
            "rows": rows, "focus_cells": focus_cells}


def make_figures(blob, show_family_block=False):
    dataset = blob["dataset"]
    engines = blob["engines"]
    rows = blob["rows"]
    focus_cells = blob["focus_cells"]
    os.makedirs(FIG_DIR, exist_ok=True)

    combined_order = (CLIMATE_COMBINED_ROWS if dataset == "climate"
                      else FLOW_COMBINED_ROWS)
    scen, per = FOCUS
    for eng in engines:
        entries = [{"res":   focus_cells[(eng, c)]["res"],
                    "units": focus_cells[(eng, c)]["units"],
                    "label": ROW_LABELS.get(c, c)}
                   for c in combined_order if (eng, c) in focus_cells]
        if entries:
            n_comb = max(e["res"]["n_sta"] for e in entries)
            eng_tag = f"{dataset}_{eng}" if eng else dataset
            fname = em.tag(os.path.join(
                FIG_DIR, f"{eng_tag}_{scen}_{per}_n{n_comb}sta.png"))
            combined_histogram_figure(entries, fname,
                                      show_family_block=show_family_block)

    out_csv = em.tag(os.path.join(
        CACHE_DIR, f"weight_permutation_null_{dataset}.csv"))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(rows)} cells)")


def run_dataset(dataset, B, recompute=False, show_family_block=False,
                figures_only=False):
    cache = FLOW_CACHE if dataset == "flow" else CLIMATE_CACHE
    if figures_only:
        # Redraw from the stored draws and SKIP _load_cached_results' signature
        # check. A signature key added after a pickle was written (e.g. "cols",
        # added 27 Aug 2026) makes that pickle look stale, so a plain rerun
        # would spend ~1.5 h drawing a fresh 10,000 and silently move every
        # published p-value. Use this whenever only the drawing changes.
        path = _results_cache_path(dataset)
        if not os.path.exists(path):
            sys.exit(f"--figures-only needs the results cache: {path}")
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        print(f"  --figures-only: redrawing stored draws from {path} "
              f"(B={blob['B']}, no new draws)")
        make_figures(blob, show_family_block=show_family_block)
        return
    if not os.path.exists(cache):
        sys.exit(f"cache not found: {cache}")
    cols = FLOW_COLS if dataset == "flow" else CLIMATE_COLS
    idcols = ["station", "scenario", "period", "model", "member"]
    if dataset == "flow":
        idcols.append("engine")

    blob = None if recompute else _load_cached_results(dataset, B, cache)
    if blob is None:
        blob = compute_results(dataset, B, cache, cols, idcols)
        _save_cached_results(blob)
    make_figures(blob, show_family_block=show_family_block)


def main(argv):
    argv = em.reject_cli_tokens(argv)
    if not em.is_weighting():
        em.set_method("weighting")      # the analysis is weighting-only
    em.log_treatment()
    recompute = any(f in argv for f in ("--recompute", "--force", "--no-cache"))
    figs_only = any(f in argv for f in ("--figures-only", "--replot"))
    if figs_only and recompute:
        sys.exit("--figures-only and --recompute are contradictory")
    # figures show the uniform permutation null only, unless asked otherwise
    show_fam = any(f in argv for f in ("--family-block", "--show-family-block"))
    argv = [a for a in argv if not a.startswith("--")]
    arg = argv[0] if argv else "all"
    B = int(argv[1]) if len(argv) > 1 else B_DEFAULT
    _, n_distinct = family_block_permute(
        [brr.family_of(m) for m in brr.FULL_MODELS],
        np.ones(len(brr.FULL_MODELS)), 1, np.random.default_rng(0))
    print(f"FULL={len(brr.FULL_MODELS)} models  variant={em.variant()} "
          f"({em.weight_column()})  tag={em.suffix()!r}  B={B}"
          f"{'  [recompute]' if recompute else ''}")
    print(f"family-block permutation admits {n_distinct:,} distinct "
          f"assignments; quote p-values against that number, not against B")
    print(f"figures: {'family-block + uniform' if show_fam else 'uniform only'} "
          f"(pass --family-block for the two-null version)")
    for ds in (["climate", "flow"] if arg == "all" else [arg]):
        run_dataset(ds, B, recompute=recompute, show_family_block=show_fam,
                    figures_only=figs_only)


if __name__ == "__main__":
    main(sys.argv[1:])
