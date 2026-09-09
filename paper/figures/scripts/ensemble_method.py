"""
ensemble_method.py
==================

SINGLE SOURCE OF TRUTH for how this project treats the hot-model problem.

Everything is configured by editing the four literals in the CONFIGURATION
block below. There is no command-line interface: a treatment is selected by
editing this file, then running the downstream analysis. Running a second
treatment means editing this file again and re-running. Outputs of different
treatments never collide, because every method-dependent file name carries the
tag() suffix of the treatment that produced it.

Three independent axes
----------------------
METRIC  — the climate-sensitivity metric the SCREEN is applied to
          (method="removal"):
    "tcr"   (default) transient climate response, bounds TCR_LOWER/TCR_UPPER.
            The metric Hausfather et al. (2022) recommend, because the AR6
            assessed-warming constraints correlate with TCR, and because this
            project projects a TRANSIENT 2081-2100 response.
    "ecs"   equilibrium climate sensitivity, bounds ECS_LOWER/ECS_UPPER. The
            alternative Hausfather et al. (2022) offer in the same paragraph:
            "using a 'likely' 2.5-4 degC ECS screen also reproduces AR6 results
            well, although at the expense of discarding 60% of the models".
            PRIMARY since 19 Aug 2026 — see KNOWN LIMITS OF METRIC="ecs" below.

SCREEN  — which climate models the REDUCED ensemble drops (method="removal"):
    "hot"   drop <metric> > <metric>_UPPER only. An attribution of projected
            change to the too-hot models. NOTE: this is NOT the Hausfather et
            al. (2022) recommendation; that paper recommends "screening out
            models with a TCR that lies outside the likely 1.4-2.2 degC range",
            i.e. "both".
    "both"  (default since 17 Aug 2026) additionally drop
            <metric> < <metric>_LOWER, i.e. keep only models inside the IPCC
            AR6 *likely* range for the active metric, the Hausfather et al.
            (2022) screen. Note this is not a pure too-hot attribution: the
            difference reduced-full nets a downward (hot-side) against an
            upward (cold-side) correction.

METHOD  — removal versus weighting:
    "removal"   (default) the SCREEN above; "reduced" = the retained models.
    "weighting" per-model weights from model_weights.load_frozen(); "reduced"
                everywhere becomes the WEIGHTED ensemble (weights renormalised
                over the models present). SCREEN is not used in this mode, and
                METRIC is DERIVED from the variant rather than set by hand
                (a weight column is fitted against one metric's assessed
                distribution, so the two cannot be chosen independently).
        VARIANT   fitted on TCR
                "bma"  (default) w_bma_tcr, faithful Massoud et al. (2023) BMA
                "kde"  w_kde_tcr, the density-ratio variant
                "like" w_like_ar6_tcr, direct assessed-likelihood weights against
                       the IPCC AR6 likely TCR range (1.4-2.2 degC at 66%),
                       fitted in model_weights.fit_gamma_to_interval
                "gaussjoint" w_gaussjoint_ar6_tcr, PRIMARY TCR weighting
                       (23 Aug 2026): Gaussian kernel whose centre and width
                       are solved jointly so 66% of the kernel mass sits in
                       1.4-2.2 degC AND the weighted-ensemble mean TCR is
                       exactly 1.80 degC. Our construction (the ensemble mean
                       is IMPOSED, and the kernel is ensemble-dependent) —
                       see the provenance warning at
                       model_weights.fit_gauss_joint_tcr.
                  fitted on ECS (added 23 Aug 2026)
                "ecslike"  w_likemean_ar6_ecs, PRIMARY ECS scheme: assessed-
                       likelihood weights against the gamma whose own mean is
                       the AR6 best estimate 3.0 degC and which puts 66% of its
                       mass in the assessed likely range 2.5-4.0 degC. Both
                       facts are PDF-verified (massoud2023.txt p. 3 quoting
                       AR6), which is why the mean-constrained fit is primary
                       here and only a sensitivity for TCR.
                "ecsequal" w_like_ar6_ecs, the equal-tail ECS target (mean is an
                       output of the fit, no best estimate asserted).
                "ecstilt"  w_tilt_ar6_ecs, the mean-pinned target exponentially
                       tilted until the WEIGHTED ENSEMBLE mean ECS is exactly
                       3.0. Our construction, not a published weighting scheme
                       (see model_weights.exponential_tilt); sensitivity only.

KNOWN LIMITS OF SCREEN="both" (read before using it for a result)
---------------------------------------------------------------
1. It is not a too-hot attribution. reduced-full then nets a downward hot-side
   against an upward cold-side correction, so it answers "what does keeping
   only the assessed likely range do", not "what do the too-hot models add".
2. The Welch test then compares a BIMODAL excluded group (15 hot + 3 cold)
   against the retained set. Its variance is inflated and its mean difference
   partly cancels, so significant fractions fall for two reasons at once. A
   drop must NOT be read as "the correction is smaller".
3. The genealogy-aware family-block null in bootstrap_random_removal.py draws
   whole lineages and can split at most one per draw. The two-sided removal set
   splits TWO families (NorESM2-MM removed while NorESM2-LM is retained; the
   CNRM family is already split by the hot-only screen), so the observed
   configuration sits in a near-zero-probability region of that null. p_spread
   is then optimistic and p_bias/p_tail conservative.
4. The eligible catchment set shifts, because the >=2-models-per-side guard
   binds differently at 18/15 than at 15/18. Report n per screen; do not
   compare significant FRACTIONS across screens without it.
5. The TCR-threshold sweep varies the UPPER bound only; under SCREEN="both"
   the lower bound stays fixed at TCR_LOWER while the upper bound is swept
   (see analyze_full_vs_reduced_significance.make_tcr_sensitivity).
6. paper/main4.tex Methods currently argues FOR the one-sided screen. Any
   two-sided output is a sensitivity analysis until that text is revised.

KNOWN LIMITS OF METRIC="ecs" (read before using it for a result)
---------------------------------------------------------------
1. It is a NESTED test, not an independent one. Over this project's 33 models
   every model with TCR > 2.20 also has ECS > 4.00, so the ECS-hot set strictly
   CONTAINS the TCR-hot set and adds five more (ACCESS-CM2, ACCESS-ESM1-5,
   CNRM-ESM2-1, GFDL-CM4, KIOST-ESM). An ECS run can therefore only make the
   correction larger; it cannot falsify the TCR result.
2. It over-corrects the transient response. ECS 2.5-4.0 leaves a retained-mean
   TCR of 1.66 degC against the AR6 best estimate of 1.80, whereas TCR 1.4-2.2
   leaves 1.78. Symmetrically the TCR screen leaves a retained-mean ECS of 3.70
   against an assessed 3.00. Neither screen is consistent on both metrics.
3. It is expensive in sample size: 13 models retained (ECS<=4.0) or 11
   (ECS 2.5-4.0) out of 33, against 18 and 15 for TCR. SSP5-8.5 thins to
   12/58 and 10/56 (models/realisations). The >=2-models-per-side guard
   therefore binds on a different catchment set again; report n per treatment
   and do not compare significant FRACTIONS across metrics without it.
4. Five models sit within 0.4 degC of the 4.00 UPPER cut (CNRM-CM6-1-HR 4.10,
   EC-Earth3-CC 4.19, EC-Earth3 4.21, GFDL-CM4 4.36, EC-Earth3-Veg-LR 4.37) and
   published ECS compilations disagree by that much (Beck 2023 gives
   IPSL-CM6A-LR 4.90; Tokarska 2020, as quoted by Asenjan 2023, gives 4.52).
   The upper-bound sweep in analyze_full_vs_reduced_significance is mandatory
   before any ECS number is reported.
5. The 2.50 LOWER bound is a knife edge that the sweep CANNOT test, because
   _tcr_sweep_one holds the lower bound fixed and varies only the upper one.
   MIROC6 (2.53) and MIROC-ES2L (2.55) sit 0.03 and 0.05 degC above it and are
   the same family, so they leave together: +0.06 degC on the lower bound takes
   the retained ensemble from 11 models to 9, and +0.35 (past FGOALS-g3 2.84 and
   the NorESM pair 2.78/2.89) takes it to 6. That margin is far smaller than the
   0.3-1.0 degC disagreement between the ECS compilations cited in limit 4. Any
   ecs_2s result needs its own lower-bound sweep, or an explicit statement that
   the retained set is two models from collapsing.
6. The genealogy caveat in KNOWN LIMITS OF SCREEN="both" item 3 is INVERTED
   here, so do not carry it over. Under ECS the screen removes WHOLE families
   and splits none: hot-only drops 11 complete families (ACCESS, CNRM, CanESM5,
   EC-Earth3, GFDL, HadGEM3-UKESM, IPSL, KACE, KIOST, NESM3, TaiESM1) = 20
   models, and two-sided adds the whole INM family = 22. The observed
   configuration is therefore MORE genealogically coherent than a typical
   family_block_keep draw, not less, so the direction of the p-value bias is
   opposite to the two-sided TCR case. The null itself stays well defined:
   K - h = 13 (hot-only) or 11 (two-sided), both above the >=2-per-side guard.
7. An ECS SCREEN and an ECS WEIGHTING are different treatments and carry
   different tags ("_ecs_2s" vs "_ecs_wt"). Since 23 Aug 2026 METHOD="weighting"
   does have ECS variants, but METRIC is then derived from the variant, not
   chosen: _validate rejects a VARIANT/METRIC mismatch rather than silently
   relabelling a TCR-fitted weighting as an ECS treatment.

Output tags produced by suffix() / tag()
----------------------------------------
    removal   + tcr + hot     ""        (hot-only outputs, kept for the decomposition)
    removal   + tcr + both    "_2s"     (primary 17-19 Aug 2026; now sensitivity)
    removal   + ecs + hot     "_ecs"    (supplementary robustness only)
    removal   + ecs + both    "_ecs_2s" (the manuscript's primary outputs since 19 Aug 2026)
    weighting + bma           "_wt"
    weighting + kde           "_wtkde"
    weighting + like          "_wtlike"
    weighting + gaussjoint    "_tcr_wt"      (primary TCR weighting, 23 Aug 2026)
    weighting + ecslike       "_ecs_wt"      (primary ECS weighting, 23 Aug 2026)
    weighting + ecsequal      "_ecs_wteq"
    weighting + ecstilt       "_ecs_wttilt"

A glob built for one tag cannot match another, because the literal that
precedes the tag in every file name ("...n20596sta") is part of the pattern:
"..._n*sta_wt.csv" requires the name to end "sta_wt.csv" and so does not match
"..._n20596sta_ecs_wt.csv", which ends "ecs_wt.csv".

Stage-1 per-realization caches (fig02_change.parquet, fig05_hydro_change_*.parquet,
stats_change.parquet, ...) are treatment-INDEPENDENT and must NEVER be tagged —
they are keyed by (station, scenario, model, member) and are reused unchanged
across every treatment. Tag only aggregated/derived outputs: figures, signif
CSVs, summary tables.

Usage in a figure script
------------------------
    import ensemble_method as em

    FULL_MODELS     = em.full_models()        # every model in the results
    EXCLUDED_MODELS = em.excluded_models()    # dropped by the active SCREEN
    REDUCED_MODELS  = em.retained_models()    # FULL minus EXCLUDED

    if em.is_weighting():
        second = em.weighted_model_mean(model_means)   # else: mean over REDUCED
    out_png = em.tag(OUT_PNG)                 # inserts the treatment suffix

    em.log_treatment(logger)                  # record what actually ran

Because the treatment now lives in an edited file rather than on the command
line, the slurm log is the only record of what a given job ran. Every entry
point must therefore call log_treatment() before doing any work.
"""

import os
import sys
from functools import lru_cache

import numpy as np

# ===========================================================================
#  CONFIGURATION — edit these four literals to change the treatment
# ===========================================================================

METRIC  = "tcr"        # "tcr" | "ecs"      (screens when removal; under
                       #                     weighting it is derived from
                       #                     VARIANT and must match it)
SCREEN  = "hot"         # "hot" | "both"      (used only when removal)
METHOD  = "weighting"    # "removal" | "weighting"
VARIANT = "vlike"        # used only when weighting. Fitted on TCR: "bma" |
                       # "kde" | "like".  Fitted on ECS: "ecslike" (primary) |
                       # "ecsequal" | "ecstilt".  Prefer selecting an ECS
                       # weighting through run_ecs_weighting.py, which sets it
                       # in process and leaves these literals alone.

# TCR bounds of the IPCC AR6 *likely* range (at least 66% probability),
# Forster et al. (2021), AR6 WG1 Chapter 7. The upper bound is the conventional
# too-hot screening threshold (Hausfather et al. 2022, Nature 605:26).
# Both comparisons are STRICT, so a model sitting exactly on a bound is kept:
# CMCC-CM2-SR5 (TCR = 2.20) and FGOALS-g3 (TCR = 1.40) are both retained.
TCR_UPPER = 2.20       # K; models ABOVE this are "too hot"
TCR_LOWER = 1.40       # K; models BELOW this are "too cold" (SCREEN="both" only)

# ECS bounds of the IPCC AR6 *likely* range (2.5-4.0 degC at least 66%
# probability), Forster et al. (2021), AR6 WG1 Chapter 7, offered as the
# alternative screen by Hausfather et al. (2022, Nature 605:26). The AR6 *very
# likely* range (90%) is 2.0-5.0 degC; set the bounds to those instead to screen
# only the hot tail. Comparisons are STRICT, as for TCR, so a model sitting
# exactly on a bound is kept.
# NOTE Asenjan et al. (2023, HESS) use ECS > 4.1, which they attribute to
# Sherwood et al. (2020). Hausfather et al. (2022) report Sherwood's *likely*
# range as 2.6-3.9. The two disagree about the same source and Sherwood (2020)
# is not in paper/literature/, so 4.1 is NOT used as the default here.
ECS_UPPER = 4.00       # degC; models ABOVE this are "too hot"
ECS_LOWER = 2.50       # degC; models BELOW this are "too cold" (SCREEN="both" only)

# The one station whose mean_deltas.nc defines the model universe. Every script
# must use the SAME station or the ensembles would differ between figures.
FULL_MODELS_STATION = "Arcticnet__49052"

# ===========================================================================
#  End of configuration — nothing below here should need editing
# ===========================================================================

SCREEN_HOT       = "hot"
SCREEN_BOTH      = "both"
METHOD_REMOVAL   = "removal"
METHOD_WEIGHTING = "weighting"
METRIC_TCR       = "tcr"
METRIC_ECS       = "ecs"

_VARIANT_COLUMN  = {"bma": "w_bma_tcr", "kde": "w_kde_tcr",
                    "like": "w_like_ar6_tcr",
                    # jointly solved Gaussian TCR kernel (23 Aug 2026), the
                    # PRIMARY TCR weighting; see model_weights
                    # .add_gaussjoint_columns()
                    "gaussjoint": "w_gaussjoint_ar6_tcr",
                    # very-likely-range Gaussian TCR kernel (27 Aug 2026),
                    # co-author proposal: mu 1.80, sigma 0.3648 fitted to 90%
                    # in 1.2-2.4 degC; neither the 66% likely-range mass nor
                    # the weighted-mean TCR is imposed. See model_weights
                    # .add_vlikenorm_columns()
                    "vlike": "w_vlikenorm_ar6_tcr",
                    # ECS-fitted schemes (23 Aug 2026); see model_weights
                    # .add_ecs_likelihood_columns()
                    "ecslike":  "w_likemean_ar6_ecs",
                    "ecsequal": "w_like_ar6_ecs",
                    "ecstilt":  "w_tilt_ar6_ecs"}
_VARIANT_SUFFIX  = {"bma": "_wt",       "kde": "_wtkde",
                    "like": "_wtlike",
                    "gaussjoint": "_tcr_wt",
                    "vlike":      "_tcr_vl",
                    "ecslike":  "_ecs_wt",
                    "ecsequal": "_ecs_wteq",
                    "ecstilt":  "_ecs_wttilt"}
# Which climate-sensitivity metric each weight column is FITTED on. Under
# METHOD="weighting" the METRIC axis is not free: it is derived from the
# variant, because a weight column fitted against the assessed TCR
# distribution is not an ECS treatment however the run is labelled.
_VARIANT_METRIC  = {"bma": "tcr", "kde": "tcr", "like": "tcr",
                    "gaussjoint": "tcr", "vlike": "tcr",
                    "ecslike": "ecs", "ecsequal": "ecs", "ecstilt": "ecs"}
_SCREEN_SUFFIX   = {SCREEN_HOT: "",     SCREEN_BOTH: "_2s"}
_METRIC_SUFFIX   = {METRIC_TCR: "",     METRIC_ECS: "_ecs"}
_METRIC_LABEL    = {METRIC_TCR: "TCR",  METRIC_ECS: "ECS"}
# position of the metric inside a fig01_model_screening.METRICS value, which is
# the tuple (ECS, TCR, 1980-2014 trend)
_METRIC_INDEX    = {METRIC_TCR: 1,      METRIC_ECS: 0}
_METRIC_BOUNDS   = {METRIC_TCR: (TCR_LOWER, TCR_UPPER),
                    METRIC_ECS: (ECS_LOWER, ECS_UPPER)}

if not (set(_VARIANT_COLUMN) == set(_VARIANT_SUFFIX) == set(_VARIANT_METRIC)):
    raise SystemExit("ensemble_method: the variant tables disagree; every "
                     "variant needs a column, a suffix and a fitted metric")
if len(set(_VARIANT_SUFFIX.values())) != len(_VARIANT_SUFFIX):
    raise SystemExit("ensemble_method: two variants share an output suffix; "
                     "their results would overwrite each other")

_state = {"metric": METRIC, "screen": SCREEN, "method": METHOD,
          "variant": VARIANT, "weights": None}

# Tokens that used to configure this module from the command line. The CLI was
# removed deliberately (the treatment is a file edit now), so a stale slurm
# script still passing them would otherwise run the DEFAULT treatment and write
# UNTAGGED file names straight over the primary results. reject_cli_tokens()
# turns that silent overwrite into a loud failure.
_DEAD_CLI_AXES     = ("method", "weights", "screen", "variant")
_DEAD_CLI_PREFIXES = tuple(f"{d}{a}=" for a in _DEAD_CLI_AXES
                           for d in ("", "-", "--"))
# bare space-separated forms too ("--method weighting"), which carry no "="
_DEAD_CLI_EXACT    = tuple(f"{d}{a}" for a in _DEAD_CLI_AXES
                           for d in ("-", "--"))


def _validate(metric, screen, method, variant):
    """Reject an impossible configuration as early and as loudly as possible."""
    if metric not in _METRIC_SUFFIX:
        raise SystemExit(f"ensemble_method: unknown METRIC={metric!r} "
                         f"(expected {METRIC_TCR!r} or {METRIC_ECS!r})")
    if screen not in _SCREEN_SUFFIX:
        raise SystemExit(f"ensemble_method: unknown SCREEN={screen!r} "
                         f"(expected {SCREEN_HOT!r} or {SCREEN_BOTH!r})")
    if method not in (METHOD_REMOVAL, METHOD_WEIGHTING):
        raise SystemExit(f"ensemble_method: unknown METHOD={method!r} "
                         f"(expected {METHOD_REMOVAL!r} or {METHOD_WEIGHTING!r})")
    if variant not in _VARIANT_COLUMN:
        raise SystemExit(f"ensemble_method: unknown VARIANT={variant!r} "
                         f"(expected {'|'.join(_VARIANT_COLUMN)})")
    if method == METHOD_WEIGHTING and screen == SCREEN_BOTH:
        # Weighting removes nothing, so SCREEN is unused in that mode. Allowing
        # the combination would let an output be read as a "two-sided weighted"
        # analysis, which is not a thing this code computes.
        raise SystemExit(
            "ensemble_method: METHOD='weighting' does not use SCREEN, so "
            "SCREEN='both' is meaningless. Set SCREEN='hot' when weighting.")
    if method == METHOD_WEIGHTING and metric != _VARIANT_METRIC[variant]:
        # A weight column is fitted against ONE metric's assessed distribution,
        # so METRIC is not free under weighting: it must name the metric the
        # active variant was fitted on. Allowing a mismatch would tag a
        # TCR-weighted result as an ECS treatment, or vice versa.
        raise SystemExit(
            f"ensemble_method: VARIANT={variant!r} is fitted on "
            f"{_VARIANT_METRIC[variant].upper()} "
            f"(column {_VARIANT_COLUMN[variant]!r}), so METHOD='weighting' "
            f"needs METRIC={_VARIANT_METRIC[variant]!r}, not {metric!r}. "
            f"Set METRIC={_VARIANT_METRIC[variant]!r}; under weighting it only "
            f"names the fit, it does not screen anything.")


_validate(METRIC, SCREEN, METHOD, VARIANT)


def reject_cli_tokens(argv):
    """Fail loudly if `argv` still carries a dead method=/weights=/screen=
    token, otherwise return it unchanged.

    The module used to consume these tokens. It no longer does, so a token left
    behind in a slurm script would silently select the file's default treatment
    and write UNTAGGED output over the primary results. Returns `argv` so the
    former `consume_cli` call sites keep working as pass-throughs."""
    dead = [t for t in argv
            if str(t).lower().startswith(_DEAD_CLI_PREFIXES)
            or str(t).lower() in _DEAD_CLI_EXACT]
    if dead:
        raise SystemExit(
            f"ensemble_method: the command-line treatment switch was removed, "
            f"but this run still passes {dead}. The treatment is now set by "
            f"editing SCREEN / METHOD / VARIANT at the top of "
            f"{os.path.basename(__file__)}. Delete these tokens from the "
            f"calling script, set the literals, and re-run.")
    return argv


# ---------------------------------------------------------------------------
#  Active treatment
# ---------------------------------------------------------------------------
def metric():
    """The active climate-sensitivity metric key, 'tcr' or 'ecs'."""
    return _state["metric"]


def metric_label():
    """Display name of the active metric, 'TCR' or 'ECS'."""
    return _METRIC_LABEL[_state["metric"]]


def is_ecs():
    """True when the SCREEN is applied to ECS rather than TCR."""
    return not is_weighting() and _state["metric"] == METRIC_ECS


def bounds():
    """(lower, upper) screening bounds of the ACTIVE metric, in degC."""
    return _METRIC_BOUNDS[_state["metric"]]


def lower_bound():
    return _METRIC_BOUNDS[_state["metric"]][0]


def upper_bound():
    return _METRIC_BOUNDS[_state["metric"]][1]


def screen():
    return _state["screen"]


def method():
    return _state["method"]


def is_weighting():
    return _state["method"] == METHOD_WEIGHTING


def is_two_sided():
    """True when the REDUCED ensemble also drops the too-cold models."""
    return not is_weighting() and _state["screen"] == SCREEN_BOTH


def variant():
    return _state["variant"]


def weight_column():
    """Name of the frozen-CSV column the active variant reads. Public so a
    consumer can record the provenance of a weighted run without reaching into
    the private _VARIANT_COLUMN table."""
    return _VARIANT_COLUMN[_state["variant"]]


def weight_metric():
    """The climate-sensitivity metric the ACTIVE variant is fitted on, 'tcr' or
    'ecs'. Unlike metric(), this is defined under METHOD='removal' too, so a
    caller can ask what a weighting variant WOULD use without switching mode."""
    return _VARIANT_METRIC[_state["variant"]]


def _set_state(**kw):
    new = dict(_state)
    new.update(kw)
    if new["method"] == METHOD_WEIGHTING:
        # Under weighting the METRIC is DERIVED from the variant (each weight
        # column is fitted against one metric's assessed distribution) and the
        # SCREEN is unused (weighting removes nothing). Pinning both here is
        # what lets set_method()/set_variant() be called in either order from a
        # weighting-only script without tripping the cross-checks in
        # _validate(). The file LITERALS are still checked strictly at import,
        # so an inconsistent edit fails loudly rather than being coerced.
        new["metric"] = _VARIANT_METRIC[new["variant"]]
        new["screen"] = SCREEN_HOT
    _validate(new["metric"], new["screen"], new["method"], new["variant"])
    if new["method"] != _state["method"] or new["variant"] != _state["variant"]:
        new["weights"] = None          # force a reload of the frozen weights
    _state.update(new)


def set_metric(value):
    """Override METRIC in code (analysis scripts that are metric-specific).
    Prefer editing the METRIC literal; this exists for scripts that legitimately
    fix the treatment themselves, and for the ECS robustness runs."""
    _set_state(metric=value)


def set_screen(value):
    """Override SCREEN in code (analysis scripts that are screen-specific).
    Prefer editing the SCREEN literal; this exists for scripts that legitimately
    fix the treatment themselves."""
    _set_state(screen=value)


def set_method(value):
    """Override METHOD in code (e.g. weight_spread_ratio.py, whose analysis is
    weighting-only). Prefer editing the METHOD literal.

    Switching to weighting also drops SCREEN back to 'hot' and sets METRIC to
    the metric the ACTIVE VARIANT is fitted on, because weighting removes
    nothing (so the screen is unused) and the metric is a property of the
    weight column, not a free axis. Without this a valid file configuration
    (METHOD='removal', SCREEN='both', METRIC='ecs') would make every
    weighting-only script abort on the _validate cross-checks."""
    _set_state(method=value)


def set_variant(value):
    """Override VARIANT in code. Under METHOD='weighting' this also moves
    METRIC to the metric that variant is fitted on (see _set_state), so
    set_method('weighting') and set_variant('ecslike') may be called in either
    order."""
    _set_state(variant=value)


# ---------------------------------------------------------------------------
#  Model universe and the screen
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def full_models():
    """Every climate model present in the results, from ONE station's
    mean_deltas.nc — the same universe Fig. 1 uses.

    Cached because it opens a NetCDF file: a fork Pool then inherits the list
    instead of every worker repeating the read. Fails loudly if any model is
    missing from METRICS, so an unclassified model is fixed in fig01 rather
    than silently mislabelled as retained."""
    fig01 = _fig01_module()
    models = fig01.models_in_results(FULL_MODELS_STATION)
    METRICS = fig01.METRICS
    missing = [m for m in models if m not in METRICS]
    if missing:
        raise RuntimeError(
            f"{len(missing)} model(s) in the results have no METRICS entry in "
            f"fig01_model_screening.py: {sorted(missing)}. Add them to METRICS "
            "(the single source of truth for ECS/TCR/trend).")
    return tuple(models)


def _fig01_module():
    """The fig01_model_screening module object, WITHOUT re-executing it.

    fig01 is both an entry point and this module's data source. When it runs as
    a script its module name is "__main__", so a plain
    `import fig01_model_screening` would execute the whole file a SECOND time
    under a different name: apply_style() re-runs, METRICS is rebuilt, the
    tagged output paths are recomputed, and mean_deltas.nc is opened again.
    Reuse the already-loaded __main__ object instead."""
    main = sys.modules.get("__main__")
    if str(getattr(main, "__file__", "")).endswith("fig01_model_screening.py"):
        return main
    import fig01_model_screening
    return fig01_model_screening


def tcr(model):
    """TCR (K) of `model` from METRICS; KeyError if absent (intended).
    Always TCR, whatever METRIC is active — use it where TCR itself is the
    object of study (the weighting scheme, the TCR scatter panels)."""
    return _fig01_module().METRICS[model][1]


def ecs(model):
    """ECS (K) of `model` from METRICS; KeyError if absent (intended).
    Always ECS, whatever METRIC is active."""
    return _fig01_module().METRICS[model][0]


def metric_value(model):
    """Value of the ACTIVE metric for `model`. This is what the screen tests,
    so any code that reimplements the screen (threshold sweeps, nulls) must
    read the models through here rather than through tcr()."""
    return _fig01_module().METRICS[model][_METRIC_INDEX[_state["metric"]]]


def is_too_hot(model):
    return metric_value(model) > upper_bound()


def is_too_cold(model):
    return metric_value(model) < lower_bound()


def is_excluded(model):
    """True when the active SCREEN drops `model` from the reduced ensemble."""
    if is_too_hot(model):
        return True
    return is_two_sided() and is_too_cold(model)


def excluded_models():
    """Models the active SCREEN drops. Under SCREEN='hot' this is the too-hot
    set; under SCREEN='both' it also carries the too-cold models, which is why
    the name is `excluded` and not `too_hot`."""
    return [m for m in full_models() if is_excluded(m)]


def too_hot_models():
    """The too-hot models specifically, whatever SCREEN is active. Use this
    only where the too-hot subset itself is the object of study (e.g. a TCR
    scatter); use excluded_models() for the reduced-ensemble split."""
    return [m for m in full_models() if is_too_hot(m)]


def too_cold_models():
    return [m for m in full_models() if is_too_cold(m)]


def retained_models():
    """Models forming the REDUCED ensemble under the active SCREEN."""
    excluded = set(excluded_models())
    return [m for m in full_models() if m not in excluded]


# ---------------------------------------------------------------------------
#  Output naming
# ---------------------------------------------------------------------------
def suffix():
    """Treatment tag: '' (removal+tcr+hot), '_2s', '_ecs', '_ecs_2s', or the
    weighting variant tag ('_wt', '_wtkde', '_wtlike').

    The metric tag comes FIRST so that a glob built for one treatment can never
    match another: '..._n*sta.csv' does not match '..._n123sta_ecs_2s.csv', and
    '..._n*sta_2s.csv' does not match it either."""
    if is_weighting():
        return _VARIANT_SUFFIX[_state["variant"]]
    return _METRIC_SUFFIX[_state["metric"]] + _SCREEN_SUFFIX[_state["screen"]]


def screen_suffix():
    """Tag for outputs that depend on the METRIC and SCREEN but NOT on the
    weighting variant -- Fig. 1 and Table 1 describe the model universe and the
    screen, and are identical under removal and weighting. Tagging them with
    '_wt' would orphan them from supplement.tex, which \\input{}s the plain
    name."""
    return _METRIC_SUFFIX[_state["metric"]] + _SCREEN_SUFFIX[_state["screen"]]


def screen_tag(path):
    """tag() for screen-dependent, weighting-independent outputs."""
    s = screen_suffix()
    if not s:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{s}{ext}"


def tag(path):
    """Insert suffix() before the file extension (no-op for removal+hot)."""
    s = suffix()
    if not s:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{s}{ext}"


def second_label():
    """Display name of the non-full ensemble."""
    if is_weighting():
        return "Weighted"
    return "Reduced (two-sided)" if is_two_sided() else "Reduced"


def excluded_label(title=False):
    """Adjective phrase for the models the ACTIVE screen drops, for figure
    labels and printed summaries. Under SCREEN='both' the excluded set is not
    "too-hot" any more, so hardcoding that word mislabels the run."""
    # "off-range" (the active metric outside its AR6 likely range) is short and
    # grammatical in every slot these labels occupy: "off-range bias",
    # "from off-range models", "reduced (off-range removed)".
    txt = "off-range" if is_two_sided() else "too-hot"
    return txt[0].upper() + txt[1:] if title else txt


def screen_expr():
    """Compact expression of the ACTIVE screen, e.g. 'TCR>2.2 K',
    'TCR>2.2 K or TCR<1.4 K', or 'ECS>4.0 K or ECS<2.5 K'."""
    lab, (lo, hi) = metric_label(), bounds()
    if is_two_sided():
        return f"{lab}>{hi} K or {lab}<{lo} K"
    return f"{lab}>{hi} K"


def describe():
    """One-line description of the active treatment, for logs and CSV columns."""
    if is_weighting():
        return (f"method=weighting variant={_state['variant']} "
            f"tag={suffix()!r}")
    lab, (lo, hi) = metric_label(), bounds()
    return (f"method=removal metric={_state['metric']} "
            f"screen={_state['screen']} "
            f"({lab}>{hi}"
            + (f" or {lab}<{lo}" if is_two_sided() else "")
            + f") tag={suffix()!r}")


def log_treatment(logger=None):
    """Record the active treatment. With the switch now living in an edited
    file, the slurm log is the only evidence of what a job actually ran, so
    every entry point should call this before doing any work."""
    msg = f"TREATMENT: {describe()}"
    if not is_weighting():
        msg += (f" | full={len(full_models())} "
                f"excluded={len(excluded_models())} "
                f"retained={len(retained_models())}")
    (logger.info if logger is not None else print)(msg)
    return msg


# ---------------------------------------------------------------------------
#  Weighting
# ---------------------------------------------------------------------------
def get_weights():
    """{model: weight} for the active variant, from the frozen CSV (cached).
    Raises under METHOD='removal' — callers must branch on is_weighting()."""
    if not is_weighting():
        raise RuntimeError("get_weights() called under METHOD='removal'")
    if _state["weights"] is None:
        from model_weights import load_frozen
        _state["weights"] = load_frozen(_VARIANT_COLUMN[_state["variant"]])
    return _state["weights"]


def weighted_model_mean(model_means):
    """Weighted ensemble mean of a pandas Series indexed by model name
    (weights renormalised over the models present with finite values).
    Returns np.nan if nothing overlaps."""
    from model_weights import renormalize
    mm = model_means.dropna()
    if mm.empty:
        return np.nan
    w = renormalize(get_weights(), list(mm.index))
    return float(sum(w[m] * mm[m] for m in w))


def weight_array(models):
    """Renormalised weight vector aligned to `models` (list) — for vectorised
    consumers. NaN-free; raises on unknown models."""
    from model_weights import renormalize
    w = renormalize(get_weights(), list(models))
    return np.array([w[m] for m in models], float)


def n_eff_present(models):
    """Kish effective ensemble size of the renormalised weights over
    `models` (under removal: the count of models, unchanged semantics)."""
    if not is_weighting():
        return float(len(list(models)))
    from model_weights import n_eff, renormalize
    return n_eff(renormalize(get_weights(), list(models)))
