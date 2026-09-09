"""
run_tcr_weighting.py
====================

Driver for the TCR-based weighting treatment (jointly solved Gaussian kernel,
chosen 23 Aug 2026) — the TCR counterpart of run_ecs_weighting.py.

WHY A DRIVER AT ALL. ensemble_method.py selects the treatment through four
file literals, and the manuscript's primary treatment currently occupies them
(METRIC="ecs", SCREEN="both", METHOD="removal", tag _ecs_2s). This driver sets
the weighting treatment IN PROCESS through the public ensemble_method
.set_method() / set_variant() API, before importing any analysis module, so
the literals are never edited and no other job is affected.

The treatment it selects:

    METHOD  = "weighting"
    VARIANT = "gaussjoint" ->  column w_gaussjoint_ar6_tcr, output tag "_tcr_wt"
    METRIC  = "tcr"        (derived from the variant, not chosen)
    SCREEN  = "hot"        (unused under weighting; pinned by ensemble_method)

VARIANT can be overridden with variant=bma / variant=kde / variant=like, which
write to "_wt" / "_wtkde" / "_wtlike" instead, so the TCR schemes never
collide. ECS variants are rejected — use run_ecs_weighting.py for those.

Steps (each may be run on its own; "core" runs figs+headline+spread+null):

    figs      the gaussjoint weight figure               -> paper/figures/weighting/
    headline  full vs screened vs every weighted column  -> phase0_headline_v2<tag>.csv
    spread    weighted vs shuffled-weight 10-90% spread  -> annmax_spread_ratio_*<tag>
              (also writes the per-catchment permutation p_perm column)
    null      aggregate weight-permutation null          -> weight_permutation_null_*<tag>
              (the weighting-mode analogue of the random-removal null)
    signif    per-catchment significance battery (analyze_full_vs_reduced
              _significance.py, weighted linear-contrast t test). ONE battery
              argument per process (the module reads it at import time), so
              select it with sigarg=all|flow|pindex|snr_cascade|transmission
              (default all) and submit one job per argument.
    gev       GEV/FDR/dipole battery (gev_fdr_dipole.py: fdr, atlas,
              atlas_extra, change_maps, dipole)          -> *_<tag> outputs
              (fdr and the atlases read the signif focus CSVs, so run the
              signif steps first; dipole rebuilds the gev caches if absent)
    localnull PER-CATCHMENT weight-permutation null, B = 50,000
              (local_vs_aggregate_null.py test1)
              -> percatchment_removal_null_*<tag> + _summary_*<tag>
              This is Supplementary Table S3 (tab:si_localnull), the local
              counterpart of the aggregate null that "null" writes for Table
              S2. TEST 2 is skipped under weighting by that script itself.

Manuscript-figure steps (the weighted counterparts of every treatment-tagged
figure in main4.tex / supplement.tex):

    fig02     fig02_precip_change.py, full run -> fig02/*_<tag>.png
              (main Fig. 2 precip_temp_change + boxplots_by_scenario,
              spectrum, dryspell_spectrum, boxplots_by_model)
    hydro     hydro_indices_change.py all -> hydro_indices/*_<tag>.png
              (main Fig. 3 fig3_biasrank_spread + hydro boxplots,
              spread_runoff, summary figures)
    gevfig    gev_fdr_dipole.py steps gev, gev_map, fig4
              -> main Fig. 4 gev_translation_fig4_*_<tag>.png + caches
    detect    fig_detectability.py -> significance/detectability_*<tag>.png
              (needs the signif focus CSVs of THIS tag: run after signif)
    table     tables/model_weights_table<tag>.tex — the weighting counterpart
              of the model list table (which lists the screen; under weighting
              the model set is not screened, it is weighted)
    scengrad  full vs weighted means AND medians per (scenario, period,
              engine) (scenario_gradient.py)
              -> fig_data/scenario_gradient_v1<tag>.csv
              Backs the scenario-gradient sentences of main4.tex.

"figs" needs only the frozen weights CSV (run `python model_weights.py
addgauss` first, via slurm/tcr_weights_and_figs.sh); the other steps read the
Stage-1 per-realization caches, which are treatment-independent and are never
written.

Run (via slurm only):
    python paper/figures/scripts/run_tcr_weighting.py figs
    python paper/figures/scripts/run_tcr_weighting.py headline
    python paper/figures/scripts/run_tcr_weighting.py spread
    python paper/figures/scripts/run_tcr_weighting.py null
    python paper/figures/scripts/run_tcr_weighting.py core
    python paper/figures/scripts/run_tcr_weighting.py signif sigarg=flow
    python paper/figures/scripts/run_tcr_weighting.py gev
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
sys.path.insert(0, _THIS)

import ensemble_method as em                                   # noqa: E402

DEFAULT_VARIANT = "vlike"
CORE_STEPS = ("figs", "headline", "spread", "null")
STEPS = CORE_STEPS + ("signif", "gev", "localnull",
                      "fig02", "hydro", "gevfig", "detect", "table", "scengrad")
FIG_SCHEMES = ("vlike",)
SHUFFLES = 1000          # weight_spread_ratio.py default, kept explicit here
B_NULL = 10000           # bootstrap_random_removal.B_DEFAULT, kept explicit
# local_vs_aggregate_null.B_DEFAULT. Larger than B_NULL on purpose: the
# per-catchment p-values are BH-controlled across ~20,600 catchments, where the
# smallest threshold is ~5e-7 while an empirical p can never fall below
# 1/(B+1), so a bigger B makes the BH step-up less conservative.
B_LOCALNULL = 50000
SIGARGS = ("all", "flow", "pindex", "snr_cascade", "transmission")
GEV_STEPS = ("fdr", "atlas", "atlas_extra", "change_maps", "dipole")


def _select(variant):
    """Put the process into the TCR weighting treatment BEFORE any analysis
    module is imported, so every module-level constant those modules compute at
    import time is already the weighted one."""
    em.set_method("weighting")
    em.set_variant(variant)
    if not em.is_weighting() or em.weight_metric() != "tcr":
        raise SystemExit(f"variant {variant!r} is not a TCR weighting scheme "
                         f"(it is fitted on {em.weight_metric().upper()}); "
                         "use run_ecs_weighting.py for the ECS schemes")
    print("=" * 72)
    em.log_treatment()
    print(f"weight column = {em.weight_column()} | output tag = "
          f"{em.suffix()!r} | derived metric = {em.metric()}")
    print("=" * 72)


def step_figs(_opt):
    import fig_model_weights as fmw
    for scheme in FIG_SCHEMES:
        print(f"\n--- weight figure: {scheme} ---")
        fmw.plot(fmw.load(None, scheme), scheme)


def step_headline(_opt):
    import phase0_headline as ph
    print(f"\n--- headline table -> {ph.DEFAULT_OUT} ---")
    df = ph.build(ph.DEFAULT_OUT)
    ph.validate(df)
    ph.report(df)


def step_spread(_opt):
    import weight_spread_ratio as wsr
    print(f"\n--- weighted vs shuffled-weight spread ratio "
          f"({SHUFFLES} shuffles) ---")
    wsr.main([f"shuffles={SHUFFLES}"])


def step_null(_opt):
    import weight_permutation_null as wpn
    print(f"\n--- aggregate weight-permutation null (B={B_NULL}) ---")
    wpn.main(["all", str(B_NULL)])


def step_signif(opt):
    # The battery module reads its positional ARG from sys.argv AT IMPORT TIME
    # (module-level), so exactly one battery argument fits one process. Point
    # sys.argv at it before the import; nothing else may sit in argv.
    sigarg = opt.get("sigarg", "all")
    if sigarg not in SIGARGS:
        raise SystemExit(f"sigarg={sigarg!r} not in {SIGARGS}")
    print(f"\n--- significance battery: {sigarg} (weighted contrast test) ---")
    sys.argv = [sys.argv[0], sigarg]
    import analyze_full_vs_reduced_significance as sig
    sig.main()


def step_gev(_opt):
    # gev_fdr_dipole reads sys.argv only under __main__, so its step functions
    # can be called directly. Same step list as slurm/weighting_postbattery.sh.
    sys.argv = sys.argv[:1]
    import gev_fdr_dipole as gev
    for name in GEV_STEPS:
        print(f"\n--- gev battery: {name} ---")
        {"fdr":         gev.fdr_field_significance,
         "atlas":       gev.bias_atlas,
         "atlas_extra": gev.bias_atlas_extra,
         "change_maps": gev.projected_change_maps,
         "dipole":      gev.quantify_peakflow_dipole}[name]()


def step_localnull(_opt):
    # local_vs_aggregate_null.py has no callable main either, so runpy it with
    # the treatment already selected in this process.
    #
    # Its TEST 1 is the PER-CATCHMENT null: at every catchment the observed
    # shift is compared against B reassignments of the weight vector, and the
    # per-catchment p-values are then BH-controlled across catchments. That is
    # what fills Supplementary Table S3 (tab:si_localnull), the local-scale
    # counterpart of the aggregate weight-permutation null in Table S2.
    #
    # Only "test1" is requested. That script skips TEST 2 under weighting by
    # itself -- aggregate_cell() contrasts a too-hot group against a retained
    # group, a split a weighted ensemble does not have -- so asking for "all"
    # would only print the skip notice. Run it under METHOD="removal" if the
    # TEST 2 numbers are ever wanted.
    #
    # Before this step existed the script was launched by slurm/
    # local_vs_aggregate_null.sh, which reads the treatment from the
    # ensemble_method.py file literals. Going through the driver instead means
    # the variant is selected explicitly, so a stale literal cannot silently
    # answer the question for a different weighting scheme.
    import runpy
    print(f"\n--- local_vs_aggregate_null test1 (B={B_LOCALNULL}) ---")
    script = os.path.join(_THIS, "local_vs_aggregate_null.py")
    sys.argv = [script, "test1", str(B_LOCALNULL)]
    runpy.run_path(script, run_name="__main__")


def step_fig02(_opt):
    # fig02's CLI parsing lives under __main__; a plain import leaves every
    # module-level flag at its default, which is exactly the no-argument run
    # weighting_figs.sh used for the _wt figures.
    print("\n--- fig02_precip_change (full run) ---")
    import fig02_precip_change as f02
    f02.main()


def step_hydro(opt):
    # hydro_indices_change has no callable main: its whole run logic sits
    # under __main__, so it is executed as a script via runpy with the
    # treatment already selected in this process (module state is shared).
    #
    # sections= narrows the run to a subset of hydro_indices_change.SECTIONS.
    # "all" draws 125 figures and takes ~2 h 40 min on 32 cpu, but only the 11
    # under hydro_indices/summary/ are cited in main4.tex or supplement.tex;
    # the other 114 are per-index atlas maps (per_index_{full,reduced,diff,
    # enginediff}, change_9panel) that no manuscript file references. The
    # sections are independently guarded in that script's __main__, so any
    # subset is safe to select. Default stays "all" so existing callers are
    # unaffected.
    import runpy
    sections = opt.get("sections", "all").split(",")
    print(f"\n--- hydro_indices_change {' '.join(sections)} ---")
    script = os.path.join(_THIS, "hydro_indices_change.py")
    sys.argv = [script] + sections
    runpy.run_path(script, run_name="__main__")


def step_gevfig(_opt):
    sys.argv = sys.argv[:1]
    import gev_fdr_dipole as gev
    for name, fn in (("gev",     gev.gev_return_period_translation),
                     ("gev_map", gev.gev_translation_maps),
                     ("fig4",    gev.gev_translation_figure4)):
        print(f"\n--- gev figure step: {name} ---")
        fn()


def step_detect(_opt):
    # fig_detectability runs at import (no __main__ block) and anchors its
    # focus-CSV glob on the ACTIVE tag, so it must run after the signif steps
    # of this treatment have written their _tcr_wt focus CSVs.
    print("\n--- detectability curve ---")
    sys.argv = sys.argv[:1]
    import fig_detectability                                   # noqa: F401


# Per-variant provenance for the table caption. Three things differ between
# the TCR kernels and every one of them is a factual claim, so they are
# tabulated rather than written into the caption string: the two parameter
# columns in the frozen CSV, how the kernel is defined, and -- the one that
# matters most -- whether the weighted-ensemble mean TCR is a CONDITION of the
# fit or an OUTCOME of it. gaussjoint solves the kernel against the archive so
# that the weighted mean lands on the assessed best estimate; vlike fixes the
# kernel from the assessed interval alone, so its weighted mean is a result
# (see model_weights.add_vlikenorm_columns). Printing the gaussjoint sentence
# under vlike would state the opposite of what the scheme does.
_TABLE_KERNEL = {
    "gaussjoint": {
        "mu_col": "gaussjoint_mu",
        "sigma_col": "gaussjoint_sigma",
        "kernel": "jointly solved Gaussian TCR kernel",
        "wmean": ("the weighted-ensemble mean TCR of {wmean:.2f}"
                  "\\,$^{{\\circ}}$C is imposed by the construction of the "
                  "kernel, not a result"),
        "rebuild": "addgauss",
    },
    "vlike": {
        "mu_col": "vlikenorm_mu",
        "sigma_col": "vlikenorm_sigma",
        "kernel": ("Gaussian TCR kernel whose central 90\\% interval is the "
                   "assessed \\emph{very likely} TCR range of "
                   "1.2--2.4\\,$^{\\circ}$C"),
        "wmean": ("the weighted-ensemble mean TCR of {wmean:.2f}"
                  "\\,$^{{\\circ}}$C is an outcome of the kernel, which is "
                  "fixed by the assessed range alone, not a condition "
                  "imposed on the fit"),
        "rebuild": "addvlike",
    },
}


def step_table(_opt):
    """Weighting counterpart of Table S3: one row per model with TCR, ECS and
    the active weight column, plus N_eff and the weighted-mean TCR in the
    caption. Same skeleton as tables/too_hot_models_table*.tex."""
    import pandas as pd
    from model_weights import frozen_csv_path, n_eff
    variant = em.variant()
    if variant not in _TABLE_KERNEL:
        raise SystemExit(
            f"step 'table' has no caption provenance for variant "
            f"{variant!r} (known: {sorted(_TABLE_KERNEL)}). Add its kernel "
            "parameters and its imposed-versus-outcome wording to "
            "_TABLE_KERNEL rather than letting it print another scheme's "
            "caption.")
    kern = _TABLE_KERNEL[variant]
    df = pd.read_csv(frozen_csv_path())
    col = em.weight_column()
    missing = [c for c in (col, kern["mu_col"], kern["sigma_col"])
               if c not in df.columns]
    if missing:
        raise SystemExit(f"{missing} not in {frozen_csv_path()} — run "
                         f"`python model_weights.py {kern['rebuild']}` first")
    df = df.sort_values("model", kind="mergesort")
    w = df[col].astype(float)
    k = len(df)
    neff = n_eff(dict(zip(df["model"], w)))
    wmean = float((w * df["tcr"].astype(float)).sum() / w.sum())
    out = em.tag(os.path.join(os.path.dirname(os.path.dirname(_THIS)),
                              "tables", "model_weights_table.tex"))
    lines = [
        "% Auto-generated by run_tcr_weighting.py (step 'table') from the",
        "% frozen weights CSV. Do not edit by hand; re-run the step instead.",
        "\\begin{table}[!ht]",
        "    \\centering",
        "    \\small",
        "    \\caption{CMIP6 models with their transient climate response "
        "(TCR) and equilibrium climate sensitivity (ECS) compiled by Beck "
        "et al.\\ (2023, their Table~3), and the weight assigned by the "
        f"{kern['kernel']} ($\\mu$ = "
        f"{float(df[kern['mu_col']].iloc[0]):.3f}\\,$^{{\\circ}}$C, "
        f"$\\sigma$ = {float(df[kern['sigma_col']].iloc[0]):.3f}"
        "\\,$^{\\circ}$C), normalised to sum to one. The rightmost column "
        f"expresses the weight as a multiple of one equal vote (1/{k}). "
        f"The effective ensemble size is {neff:.1f} of {k} models; "
        + kern["wmean"].format(wmean=wmean) + ".}",
        "    \\label{tab:model_weights_tcr}",
        "    \\begin{tabular}{l r r r r}",
        "    \\toprule",
        "    Model & TCR ($^{\\circ}$C) & ECS ($^{\\circ}$C) & Weight & "
        "Weight $\\times$ " + str(k) + " \\\\",
        "    \\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"    {r['model']} & {r['tcr']:.2f} & {r['ecs']:.2f} & "
                     f"{r[col]:.4f} & {r[col] * k:.2f} \\\\")
    lines += ["    \\bottomrule", "    \\end{tabular}", "\\end{table}", ""]
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"weights table -> {out}")


def step_scengrad(_opt):
    # Full vs weighted ensemble MEANS AND MEDIANS for every (scenario, period,
    # engine) cell, on the phase0_headline aggregation. Backs the scenario-
    # gradient sentences in main4.tex (the AnnMax bias from SSP1-2.6 to
    # SSP5-8.5, the mid-century values, and the Discussion sentence on the
    # mid-century Q05 median bias) -> fig_data/scenario_gradient_v1<tag>.csv.
    #
    # scenario_gradient computes OUT_CSV with em.tag() at MODULE level, so
    # the treatment has to be selected before it is imported. It is: _select()
    # runs before any step's import, which is the whole point of this driver.
    # Reads stats_change.parquet with row-group filters, one cell at a time.
    print("\n--- scenario_gradient (scenario-gradient means/medians) ---")
    import scenario_gradient as sg
    sg.main()


RUNNERS = {"figs": step_figs, "headline": step_headline,
           "spread": step_spread, "null": step_null,
           "signif": step_signif, "gev": step_gev,
           "localnull": step_localnull,
           "fig02": step_fig02, "hydro": step_hydro,
           "gevfig": step_gevfig, "detect": step_detect,
           "table": step_table, "scengrad": step_scengrad}


def main(argv):
    # NOTE: this driver's own variant= option is parsed BEFORE
    # em.reject_cli_tokens(), because "variant" is also a dead CLI axis of the
    # old ensemble_method interface and the rejector would refuse the token.
    opt = {"variant": DEFAULT_VARIANT}
    steps, rest = [], []
    for tok in argv:
        if "=" in tok and tok.split("=", 1)[0] in ("variant", "sigarg",
                                                   "sections"):
            k, v = tok.split("=", 1)
            opt[k] = v
        elif tok in STEPS:
            steps.append(tok)
        elif tok == "core":
            steps.extend(CORE_STEPS)
        else:
            rest.append(tok)
    em.reject_cli_tokens(rest)          # loud failure on stale method=/screen=
    if rest:
        raise SystemExit(f"unknown argument(s) {rest} (expected one of "
                         f"{STEPS + ('core',)}, variant=<name> or "
                         f"sigarg=<battery arg>)")
    if not steps:
        steps = list(CORE_STEPS)
    if steps.count("signif") == 0 and "sigarg" in opt:
        raise SystemExit("sigarg= given but the signif step is not selected")
    if steps.count("hydro") == 0 and "sections" in opt:
        raise SystemExit("sections= given but the hydro step is not selected")
    if steps.count("signif") > 0 and len(steps) > 1:
        # signif consumes sys.argv at import; anything after it in the same
        # process would inherit a rewritten argv. One battery run per process.
        raise SystemExit("run 'signif' on its own (one process per battery "
                         "argument); the module reads argv at import time")

    _select(opt["variant"])
    for step in steps:
        print(f"\n{'#' * 24}  {step}  {'#' * 24}")
        RUNNERS[step](opt)


if __name__ == "__main__":
    main(sys.argv[1:])
