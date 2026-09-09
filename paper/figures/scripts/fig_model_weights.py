"""
fig_model_weights.py
====================

Presentation of the frozen model weights: the climate-sensitivity metric the
scheme is fitted on, the AR6 likely-range flag for that metric, and the weight
column of the chosen scheme, for every model in the 33-model universe.

  panel a  per-model weight, models ordered by the metric (hottest at the top),
           coloured by the screening flag, against the equal-weight line 1/K.
  panel b  weight against the metric, with the AR6 likely upper bound as the
           screening threshold and the equal-weight line; the right-hand axis
           reads the same weights as a multiple of one equal vote.

Schemes (see model_weights.py for the definitions). Each scheme names the
metric it is fitted on, and both panels follow that metric:

  fitted on TCR
    bma      (default)  w_bma_tcr       Massoud et al. (2023) BMA posterior mean
    like                w_like_ar6_tcr  assessed-likelihood weights against the
                                        IPCC AR6 likely TCR range (1.4-2.2 degC
                                        at 66%)
    gaussjoint          w_gaussjoint_ar6_tcr  PRIMARY TCR weighting: Gaussian
                                        kernel with 66% of its mass in
                                        1.4-2.2 degC AND the weighted-ensemble
                                        mean TCR pinned to 1.80 degC (our
                                        construction; both parameters solved,
                                        read back from the frozen CSV)
  fitted on ECS
    ecslike             w_likemean_ar6_ecs  PRIMARY ECS scheme: target gamma
                                        with mean 3.0 degC (the AR6 best
                                        estimate) and 66% of its mass in
                                        2.5-4.0 degC
    ecsequal            w_like_ar6_ecs  equal-tail ECS target (no best estimate
                                        asserted)
    ecstilt             w_tilt_ar6_ecs  mean-pinned target tilted until the
                                        WEIGHTED ENSEMBLE mean ECS is exactly
                                        3.0; our construction, sensitivity only

For every likelihood scheme the weights ARE the target density renormalised
over the K models, so panel b also carries the target curve and it passes
exactly through every model point.

Reads ONLY the frozen CSV written by `python model_weights.py weights` /
`... addlike` / `... addecs` (fig_data/model_weights_wt.csv) -- it never
recomputes weights, so the figure cannot drift from the numbers used by the
weighted figure runs.

    python paper/figures/scripts/fig_model_weights.py
    python paper/figures/scripts/fig_model_weights.py like
    python paper/figures/scripts/fig_model_weights.py ecslike
    python paper/figures/scripts/fig_model_weights.py like /path/to/weights.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
sys.path.insert(0, _THIS)

from figstyle import add_panel_letters                       # noqa: E402
import ensemble_method as em                                 # noqa: E402
from model_weights import (TCR_TARGET_AR6_GAMMA, TCR_TARGET_GAMMA,  # noqa: E402
                           ECS_TARGET_AR6_GAMMA, ECS_TARGET_AR6_MEANFIT,
                           AR6_ECS_BEST_ESTIMATE, AR6_ECS_LIKELY,
                           AR6_ECS_LIKELY_PROB, AR6_TCR_BEST_ESTIMATE,
                           AR6_TCR_LIKELY, AR6_TCR_LIKELY_PROB,
                           _gamma_interval_mass, _normal_interval_mass,
                           frozen_csv_path, gamma_pdf, n_eff, normal_logpdf)

_FIGDIR = os.path.dirname(_THIS)
# The weighting figures are collected in paper/figures/weighting/ (the TCR ones
# were moved there on 18 Aug 2026). New schemes are written straight into that
# folder; the two TCR schemes keep their original _FIGDIR paths so a re-run
# cannot overwrite the archived versions.
_WEIGHTDIR = os.path.join(_FIGDIR, "weighting")

# scheme -> how to read it, name it and where to write it. `target` is the
# gamma the scheme is calibrated against, reported in the panel-b note; `pdf`
# says whether the weights are literally that density (likelihood scheme), in
# which case panel b can carry the curve itself.
SCHEMES = {
    "bma": {
        "column": "w_bma_tcr",
        "label": "BMA weight",
        "target": TCR_TARGET_GAMMA,
        "pdf": False,
        "out": os.path.join(_FIGDIR, "fig_model_weights_bma.png"),
    },
    "like": {
        "column": "w_like_ar6_tcr",
        "label": "AR6 likelihood weight",
        "target": TCR_TARGET_AR6_GAMMA,
        "pdf": True,
        "out": os.path.join(_FIGDIR, "fig_model_weights_ar6like.png"),
    },
    # ---- ECS-fitted schemes (23 Aug 2026) ---------------------------------
    # Same two panels, but the model axis is ECS and the screening line is the
    # AR6 likely ECS upper bound of 4.0 degC. Separate output names, so no TCR
    # figure is overwritten.
    "ecslike": {
        "column": "w_likemean_ar6_ecs",
        "metric": "ecs",
        "label": "AR6 likelihood weight",
        "target": ECS_TARGET_AR6_MEANFIT,
        "pdf": True,
        "pdf_label": "Assessed ECS density (renormalised)",
        "note_wrap": True,
        "note_pos": (0.97, "top"),
        "legend_loc": "center right",
        "headroom": 0.85,
        "assessed": AR6_ECS_BEST_ESTIMATE,
        "assessed_range": AR6_ECS_LIKELY,
        "assessed_prob": AR6_ECS_LIKELY_PROB,
        "out": os.path.join(_WEIGHTDIR, "fig_model_weights_ar6like_ecs.png"),
    },
    "ecsequal": {
        "column": "w_like_ar6_ecs",
        "metric": "ecs",
        "label": "AR6 likelihood weight",
        "target": ECS_TARGET_AR6_GAMMA,
        "pdf": True,
        "pdf_label": "Assessed ECS density (renormalised)",
        "note_wrap": True,
        "note_pos": (0.97, "top"),
        "legend_loc": "center right",
        "headroom": 0.85,
        "assessed": AR6_ECS_BEST_ESTIMATE,
        "assessed_range": AR6_ECS_LIKELY,
        "assessed_prob": AR6_ECS_LIKELY_PROB,
        "out": os.path.join(_WEIGHTDIR,
                            "fig_model_weights_ar6like_ecs_equaltail.png"),
    },
    # ---- jointly solved Gaussian TCR kernel (23 Aug 2026) -----------------
    # PRIMARY TCR weighting. The target is a NORMAL kernel, not a gamma, and
    # its parameters depend on the archive (the ensemble mean is imposed), so
    # they are read back from the frozen CSV like the tilt's.
    "gaussjoint": {
        "column": "w_gaussjoint_ar6_tcr",
        "metric": "tcr",
        "family": "normal",
        "label": "Gaussian kernel weight",
        "target_cols": ("gaussjoint_mu", "gaussjoint_sigma"),
        "pdf": True,
        "pdf_label": "Solved Gaussian kernel (renormalised)",
        "note_wrap": True,
        "note_pos": (0.97, "top"),
        "legend_loc": "center right",
        "headroom": 0.85,
        "assessed": AR6_TCR_BEST_ESTIMATE,
        "assessed_range": AR6_TCR_LIKELY,
        "assessed_prob": AR6_TCR_LIKELY_PROB,
        "out": os.path.join(_WEIGHTDIR, "fig_model_weights_gaussjoint_tcr.png"),
    },
    # ---- very-likely-range Gaussian TCR kernel (27 Aug 2026) --------------
    # Co-author proposal: mu = 1.80 (the assessed best estimate) and sigma
    # fitted so 90% of the kernel mass sits in the AR6 VERY LIKELY range
    # 1.2-2.4 degC. Nothing is solved against the archive, so the note's
    # likely-range coverage (72.7%, against the assessed 66%) and the weighted
    # mean TCR (1.846, against 1.80) are OUTCOMES, and the figure prints both
    # rather than asserting the scheme reproduces the assessment.
    "vlike": {
        "column": "w_vlikenorm_ar6_tcr",
        "metric": "tcr",
        "family": "normal",
        "label": "Gaussian kernel weight",
        "target_cols": ("vlikenorm_mu", "vlikenorm_sigma"),
        "pdf": True,
        "pdf_label": "Very-likely-range Gaussian kernel (renormalised)",
        "note_wrap": True,
        "note_pos": (0.97, "top"),
        "legend_loc": "center right",
        "headroom": 0.85,
        "assessed": AR6_TCR_BEST_ESTIMATE,
        "assessed_range": AR6_TCR_LIKELY,
        "assessed_prob": AR6_TCR_LIKELY_PROB,
        # what this kernel WAS fitted to, so the reader can see that the
        # likely-range coverage above is a consequence, not the target.
        # No backslash before the %: this string is not mathtext, and
        # matplotlib would draw the escape literally.
        "note_extra": "Fitted to the very likely range, not the likely one:\n"
                      "  90% of the kernel mass in 1.2 to 2.4 "
                      "$^{\\circ}$C",
        # two steps down from the default: this scheme's note carries an extra
        # line and would otherwise run into the right-hand axis
        "note_fs": 8.5,
        "out": os.path.join(_WEIGHTDIR, "fig_model_weights_vlike_tcr.png"),
    },
    "ecstilt": {
        "column": "w_tilt_ar6_ecs",
        "metric": "ecs",
        "label": "Mean-constrained weight",
        # the tilted target is not a module constant (it depends on the models
        # in the archive), so it is read back from the frozen CSV
        "target_cols": ("ecs_tilt_gamma_a", "ecs_tilt_gamma_scale"),
        "pdf": True,
        "pdf_label": "Assessed density, tilted to the best estimate",
        "note_wrap": True,
        "note_pos": (0.97, "top"),
        "legend_loc": "center right",
        "headroom": 0.85,
        "assessed": AR6_ECS_BEST_ESTIMATE,
        "assessed_range": AR6_ECS_LIKELY,
        "assessed_prob": AR6_ECS_LIKELY_PROB,
        "out": os.path.join(_WEIGHTDIR, "fig_model_weights_tilt_ar6_ecs.png"),
    },
}
DEFAULT_SCHEME = "bma"
OUT_PNG = SCHEMES[DEFAULT_SCHEME]["out"]      # kept for importers

TOO_HOT_TCR_THRESHOLD = em.TCR_UPPER   # AR6 likely upper bound, from em
# per-metric presentation: (CSV column, short name, axis name, screening cut)
_METRIC_SPEC = {
    "tcr": ("tcr", "TCR", "Transient climate response", em.TCR_UPPER),
    "ecs": ("ecs", "ECS", "Equilibrium climate sensitivity", em.ECS_UPPER),
}
COL_HOT  = "#c0392b"      # metric above the AR6 likely upper bound
COL_KEEP = "#2c7fb8"      # metric at or below it
COL_REF  = "#4d4d4d"      # reference lines


def load(path=None, scheme=DEFAULT_SCHEME):
    path = path or frozen_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run `python model_weights.py weights` first.")
    df = pd.read_csv(path)
    spec = SCHEMES[scheme]
    column = spec["column"]
    mcol, _, _, cut = _METRIC_SPEC[spec.get("metric", "tcr")]
    need = {"model", "tcr", "too_hot", mcol, column} | set(spec.get("target_cols", ()))
    missing = need - set(df.columns)
    if missing:
        if scheme == "gaussjoint":
            cmd = "addgauss"
        else:
            cmd = "addecs" if spec.get("metric") == "ecs" else "addlike"
        extra = (f"; run `python model_weights.py {cmd}` to append the "
                 "likelihood columns"
                 if missing - {"model", "tcr", "too_hot", mcol} else "")
        raise KeyError(f"{sorted(missing)} missing from {path}{extra}")
    df["too_hot"] = df["too_hot"].astype(str).str.lower().eq("true")
    # The screening flag of the ACTIVE metric, recomputed from the metric column
    # so an ECS scheme is not coloured by the TCR flag. Under metric='tcr' this
    # must reproduce the stored column exactly; assert it rather than assume it,
    # so the TCR figures cannot move.
    df["_hot"] = df[mcol].astype(float) > cut
    if spec.get("metric", "tcr") == "tcr" and not df["_hot"].equals(df["too_hot"]):
        raise RuntimeError("the recomputed TCR>%.2f flag disagrees with the "
                           "frozen too_hot column" % cut)
    # hottest first, so the y axis of panel a reads top-down by the metric
    return df.sort_values([mcol, "model"], ascending=[False, True],
                          kind="mergesort").reset_index(drop=True)


def _target_family(spec):
    """'gamma' (default) or 'normal' — which density family the target is."""
    return spec.get("family", "gamma")


def _target_pdf(spec, target, v):
    """Target density at `v` for either family."""
    if _target_family(spec) == "normal":
        return np.exp(normal_logpdf(v, *target))
    return gamma_pdf(v, *target)


def _target_own_mean(spec, target):
    """The target density's OWN mean (mu for normal, a*scale for gamma)."""
    if _target_family(spec) == "normal":
        return target[0]
    return target[0] * target[1]


def _target_interval_mass(spec, target, lo, hi):
    if _target_family(spec) == "normal":
        return _normal_interval_mass(*target, lo, hi)
    return _gamma_interval_mass(*target, lo, hi)


def build_assessed_note(spec, target, w, x, mshort, k, wm_metric, model_names):
    """Panel-b statistics box for a scheme with an assessed reference.

    Names every quantity, so "target" cannot be read as two different things
    across panels: the target density's OWN mean and interval coverage, the
    WEIGHTED ensemble mean it produces, and the ASSESSED values it is
    answering to."""
    lo, hi = spec["assessed_range"]
    prob = spec["assessed_prob"]
    cov = _target_interval_mass(spec, target, lo, hi)
    fam = ("Target gamma" if _target_family(spec) == "gamma"
           else "Gaussian kernel")
    # optional line naming what the kernel was actually fitted to, for a scheme
    # whose target is NOT the interval reported on the lines above
    extra = spec.get("note_extra")
    # The coverage line carries the assessed value inline, in brackets. The
    # earlier layout put the kernel's coverage and the assessed probability on
    # separate lines that both ended in the same interval, and a reader
    # (correctly cautious) took the assessed 66% for a property of the kernel.
    # Naming the subject of every line is what stops that: the kernel's own
    # coverage and the number the assessment asks for now sit side by side.
    return (f"Effective ensemble size 1/$\\Sigma w^2$ = "
            f"{n_eff(dict(zip(model_names, w))):.1f} of {k}\n"
            f"{fam}: own mean {_target_own_mean(spec, target):.3f} "
            f"$^{{\\circ}}$C\n"
            f"{fam} mass in {lo:g} to {hi:g} $^{{\\circ}}$C: {cov:.1%} "
            f"(assessment asks {prob:.0%})\n"
            f"Weighted mean {mshort} = {wm_metric:.3f} $^{{\\circ}}$C "
            f"(equal weights {x.mean():.3f})\n"
            f"Assessed best estimate: {spec['assessed']:g} "
            f"$^{{\\circ}}$C\n"
            + (f"{extra}\n" if extra else "")
            + f"Weight span {w.min():.4f} to {w.max():.4f} "
              f"(ratio {w.max() / w.min():.1f})")


def plot(df, scheme=DEFAULT_SCHEME, out_png=None):
    spec = SCHEMES[scheme]
    out_png = out_png or spec["out"]
    wlabel = spec["label"]
    mkey = spec.get("metric", "tcr")
    mcol, mshort, mlong, mcut = _METRIC_SPEC[mkey]
    # a scheme whose target gamma depends on the archive (the tilt) carries its
    # parameters in the frozen CSV rather than as a module constant
    if "target_cols" in spec:
        ca, cs = spec["target_cols"]
        target = (float(df[ca].iloc[0]), float(df[cs].iloc[0]))
    else:
        target = spec["target"]
    k = len(df)
    equal = 1.0 / k
    w = df[spec["column"]].to_numpy(float)
    tcr = df[mcol].to_numpy(float)          # values of the ACTIVE metric
    hot = df["_hot"].to_numpy(bool)
    colours = np.where(hot, COL_HOT, COL_KEEP)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(13.2, 9.4), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---------------------------------------------------------------- panel a
    y = np.arange(k)[::-1]                      # first row of df at the top
    axa.barh(y, w, height=0.72, color=colours, edgecolor="none")
    axa.axvline(equal, color=COL_REF, linestyle="--", linewidth=1.2)
    axa.set_yticks(y)
    axa.set_yticklabels([f"{m}  ({t:.2f})" for m, t in zip(df["model"], tcr)])
    # headroom above the first bar for the equal-weight label, so nothing is
    # drawn over a bar
    axa.set_ylim(-0.8, k + 0.9)
    axa.set_xlim(0, max(w.max(), equal) * 1.12)
    axa.set_xlabel(wlabel)
    axa.set_title(f"Weight per model, ordered by {mshort} "
                  f"({mshort} in brackets)")
    axa.annotate(f"Equal weight (1/{k} = {equal:.4f})", xy=(equal, k + 0.15),
                 xytext=(5, 0), textcoords="offset points", ha="left",
                 va="center", color=COL_REF, fontsize=10)
    axa.grid(axis="x", color="0.88", linewidth=0.6)
    axa.set_axisbelow(True)
    for side in ("top", "right"):
        axa.spines[side].set_visible(False)

    # ---------------------------------------------------------------- panel b
    axb.scatter(tcr[~hot], w[~hot], s=64, color=COL_KEEP, zorder=3,
                label=f"Retained, {mshort} $\\leq$ {mcut:g} $^{{\\circ}}$C "
                      f"(n = {(~hot).sum()})")
    axb.scatter(tcr[hot], w[hot], s=64, color=COL_HOT, zorder=3,
                label=f"Too hot, {mshort} > {mcut:g} $^{{\\circ}}$C "
                      f"(n = {hot.sum()})")
    if spec["pdf"]:
        # the weights ARE the target density renormalised over the K models, so
        # this curve passes exactly through every point rather than fitting it
        grid = np.linspace(tcr.min() - 0.15, tcr.max() + 0.15, 400)
        dens = _target_pdf(spec, target, grid)
        axb.plot(grid, dens / _target_pdf(spec, target, tcr).sum(),
                 color=COL_REF, linewidth=1.2, alpha=0.75, zorder=1,
                 label=spec.get("pdf_label",
                                "Assessed TCR density (renormalised)"))
    axb.axhline(equal, color=COL_REF, linestyle="--", linewidth=1.2, zorder=2)
    axb.axvline(mcut, color=COL_REF, linestyle=":",
                linewidth=1.4, zorder=2)
    axb.annotate("Equal weight", xy=(tcr.max(), equal), xytext=(-4, 4),
                 textcoords="offset points", ha="right", va="bottom",
                 color=COL_REF, fontsize=10)
    # label sits in the empty lower-left region, left of the threshold line
    lo, hi = w.min(), w.max()
    axb.annotate("Screening threshold",
                 xy=(mcut, lo + 0.10 * (hi - lo)),
                 xytext=(-6, 0), textcoords="offset points", rotation=90,
                 ha="right", va="bottom", color=COL_REF, fontsize=10)
    # room under the lowest-weight model for the diagnostics note, and, where
    # the density curve is drawn, headroom above its peak so the legend cannot
    # sit on it. Both bounds are no-ops for the BMA scheme, whose weight span
    # is narrow and well away from zero.
    span = hi - lo
    axb.set_ylim(max(0.0, lo - 0.30 * span),        # a weight is never negative
                 hi + spec.get("headroom",
                               0.30 if spec["pdf"] else 0.06) * span)
    axb.set_xlabel(f"{mlong} ($^{{\\circ}}$C)")
    axb.set_ylabel(wlabel)
    axb.set_title(f"Weight against {mshort}")
    axb.legend(loc=spec.get("legend_loc", "upper right"), framealpha=0.9)
    axb.grid(color="0.88", linewidth=0.6)
    axb.set_axisbelow(True)
    for side in ("top",):
        axb.spines[side].set_visible(False)

    # right-hand reading of the same axis: multiples of one equal vote
    axr = axb.twinx()
    axr.set_ylim(np.array(axb.get_ylim()) / equal)
    axr.set_ylabel(f"Weight / equal weight")
    axr.grid(False)
    axr.spines["top"].set_visible(False)

    wm_metric = float(np.average(tcr, weights=w))
    # a scheme whose GOAL differs from the mean of the curve it draws (the
    # mean-constrained tilt) states its goal here instead; defined for both
    # note styles because the diagnostics printed below use it either way
    target_mean = spec.get("target_mean", _target_own_mean(spec, target))
    if spec.get("assessed") is None:
        # a wide note box can cover the lowest-weight models; `note_wrap` breaks
        # the long middle line so the box stays clear of them (default: unchanged)
        sep = "\n" if spec.get("note_wrap") else " "
        note = (f"Effective ensemble size 1/$\\Sigma w^2$ = {n_eff(dict(zip(df['model'], w))):.1f} "
                f"of {k}\n"
                f"Weighted mean {mshort} = {wm_metric:.3f} $^{{\\circ}}$C{sep}"
                f"(equal weights {tcr.mean():.3f}, target {target_mean:.3f})\n"
                f"Weight span {w.min():.4f} to {w.max():.4f} "
                f"(ratio {w.max() / w.min():.2f})")
    else:
        # The word "target" was doing two jobs at once: the gamma's own mean in
        # the likelihood schemes and the constraint being solved for in the
        # tilted one. Here each quantity is named, and the target's interval
        # coverage is printed so a scheme that does NOT reproduce the assessed
        # range says so on the figure itself.
        note = build_assessed_note(spec, target, w, tcr, mshort, k, wm_metric,
                                   df["model"])
    # bottom-left by default; a scheme whose lowest-weight models sit in that
    # corner moves the box to the top instead so it cannot cover them
    note_y, note_va = spec.get("note_pos", (0.03, "bottom"))
    axb.text(0.03, note_y, note, transform=axb.transAxes,
             fontsize=spec.get("note_fs", 10),
             ha="left", va=note_va,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="0.85", alpha=0.92))

    add_panel_letters([axa, axb])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"figure -> {out_png}")

    print(f"\nscheme={scheme} ({spec['column']}, metric={mkey}) | K={k} | "
          f"equal weight={equal:.4f} | N_eff="
          f"{n_eff(dict(zip(df['model'], w))):.2f} | weighted-mean {mshort}="
          f"{wm_metric:.4f} | equal-weight mean {mshort}={tcr.mean():.4f} | "
          f"target={target_mean:.4f} | target {_target_family(spec)}="
          f"({target[0]:.4f}, {target[1]:.6f})")
    print(f"share of weight above {mshort} {mcut:g} = {w[hot].sum():.4f} "
          f"(equal-weight share {hot.sum() / k:.4f}) -> hot influence removed "
          f"= {1 - w[hot].sum() / (hot.sum() / k):.1%}")


if __name__ == "__main__":
    em.reject_cli_tokens(sys.argv[1:])
    scheme, csv = DEFAULT_SCHEME, None
    for tok in sys.argv[1:]:                    # scheme and path, either order
        if tok in SCHEMES:
            scheme = tok
        else:
            csv = tok
    plot(load(csv, scheme), scheme)
