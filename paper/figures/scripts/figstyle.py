"""
figstyle.py
===========

Single source of truth for figure presentation: the house font style and the
Nature-style panel letters. Both are presentation-only concerns, so they live
together and are imported ONLY by plotting scripts.

Deliberately NOT merged with realization_blocklist.py / flow_metrics.py: those
two are data-layer modules imported by build_streamflow_cache.py, which runs
headless under slurm and must not pull in matplotlib.

This module imports `matplotlib` but NOT `matplotlib.pyplot`, so importing it
does not lock in a backend -- scripts calling `matplotlib.use("Agg")` later
(several do, from inside their plotting functions) keep working.

The style is applied at import time, so a plotting script only needs:

    from figstyle import add_panel_letters

and every figure it draws picks up the house fonts. Call `apply_style()`
explicitly to re-apply after something else has overwritten rcParams.

Font house style
----------------
One typeface everywhere: DejaVu Sans (matplotlib's default sans-serif, pinned
here by name so the same figure renders identically on any machine). The sizes
below set what an unannotated Axes gets; a per-call `fontsize=` at a call site
still wins, which is intended for the few labels that genuinely need to differ.

`axes.titleweight` / `axes.labelweight` are pinned to 'normal' to enforce the
CLAUDE.md rule that the panel letter is the only bold text in a figure.

Panel letters
-------------
House style (Nature panel style):

    bare bold lower-case letter -- "a", "b", ... -- placed at the TOP-LEFT of
    each panel, in axes coordinates at (0.0, 1.02), i.e. just OUTSIDE the axes
    so the letter can never land on a data point, box, whisker or map.
    fontsize 13, va="bottom", ha="left".

The letters carried a closing parenthesis ("a)", "b)") until 2026-07-30. The
`suffix` argument still exists, so pass suffix=")" to reproduce a figure exactly
as it was rendered before that date; every script now takes the bare default.

Legends
-------
`legend.edgecolor` is pinned to 'none', so no legend carries the matplotlib
default gray frame line (CLAUDE.md). A call site that passes `edgecolor=` still
wins; `frameon=False` at a call site removes the white patch as well.

Two rules from CLAUDE.md are enforced here rather than at ~25 call sites:

  * Letters go on MULTI-PANEL figures only. A single-panel figure gets nothing,
    so `add_panel_letters` is a no-op below 2 visible panels (override with
    force=True if a one-panel figure ever genuinely needs an "a)").
  * The letter is the ONLY bold text in the figure. This module never bolds
    anything else; it is not a general annotation helper.

Usage -- call it immediately before `fig.savefig(...)`, passing the axes that
`plt.subplots` returned (NOT `fig.axes`, which by then also holds the colorbar
axes and would letter them):

    from figstyle import add_panel_letters
    ...
    add_panel_letters(axes)          # ndarray from plt.subplots(nrow, ncol)
    add_panel_letters([axA, axB])    # explicit tuple-unpacked axes
    fig.savefig(out)

Calling it just before the save matters: panels hidden with
`ax.set_visible(False)` (several figures blank unused cells) are skipped, so the
lettering stays contiguous over the panels a reader actually sees.
"""

import re
import string

import matplotlib
import numpy as np

# ---------------------------------------------------------------------------
# house font style -- keep in sync with the docstring above
# ---------------------------------------------------------------------------
HOUSE_STYLE = {
    # one typeface everywhere; pinned by name so it does not drift by machine
    'font.family':          'sans-serif',
    'font.sans-serif':      ['DejaVu Sans', 'Bitstream Vera Sans', 'Arial'],
    'mathtext.fontset':     'dejavusans',   # math matches the surrounding text

    'font.size':            11,             # base: anything not set explicitly
    'axes.titlesize':       12,
    'axes.labelsize':       12,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'legend.fontsize':      10,
    'legend.title_fontsize': 11,
    'figure.titlesize':     13,

    # the panel letter is the only bold text in a figure (CLAUDE.md)
    'axes.titleweight':     'normal',
    'axes.labelweight':     'normal',

    # no gray edge line around a legend (CLAUDE.md). The semi-opaque white
    # patch is kept: several legends sit over data and need to stay readable.
    'legend.edgecolor':     'none',

    # output: 300 dpi rasters, and text embedded as editable TrueType (not
    # Type-3 / outlined paths) so a vector export stays journal-compliant
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
    'svg.fonttype':         'none',
}


def apply_style(overrides=None):
    """Apply the house font style to the global rcParams.

    Runs once at import. Call it again to re-assert the style after something
    else has overwritten rcParams, optionally with a dict of `overrides` for a
    figure that genuinely needs to depart from the house sizes.

    Returns the dict that was applied.
    """
    style = dict(HOUSE_STYLE)
    if overrides:
        style.update(overrides)
    matplotlib.rcParams.update(style)
    return style


apply_style()


# ---------------------------------------------------------------------------
# panel letters -- keep in sync with the docstring above
# ---------------------------------------------------------------------------
PANEL_LETTER_FONTSIZE = 13
PANEL_LETTER_X = 0.0      # axes coords: left edge
PANEL_LETTER_Y = 1.02     # axes coords: just above the top edge


def panel_letter_sequence(n):
    """['a', 'b', ... ] of length `n`; continues 'aa', 'ab', ... past 26 so a
    very large grid still gets unique, ordered labels instead of an IndexError."""
    ab = string.ascii_lowercase
    out = []
    for i in range(n):
        if i < 26:
            out.append(ab[i])
        else:
            out.append(ab[i // 26 - 1] + ab[i % 26])
    return out


def flatten_axes(axes):
    """Axes in reading order (row-major) as a flat list.

    Accepts what the call sites actually hold: the ndarray `plt.subplots`
    returns (any shape, incl. squeeze=False's 2-D), a list/tuple of axes, or a
    single Axes. `np.ravel` on the subplots ndarray already yields row-major
    order, which is the order a reader scans, so no re-sorting by position.
    """
    if axes is None:
        return []
    if isinstance(axes, np.ndarray):
        return [a for a in axes.ravel().tolist() if a is not None]
    if isinstance(axes, (list, tuple)):
        flat = []
        for a in axes:
            flat.extend(flatten_axes(a))
        return flat
    return [axes]


def add_panel_letters(axes, fontsize=PANEL_LETTER_FONTSIZE,
                      x=PANEL_LETTER_X, y=PANEL_LETTER_Y,
                      skip_invisible=True, force=False, start=0,
                      suffix=""):
    """Draw "a", "b", ... at the top-left of every visible panel in `axes`.

    axes           ndarray / list / tuple / single Axes -- the PANEL axes (pass
                   what plt.subplots returned, not fig.axes: the latter includes
                   colorbar axes by save time and they must not be lettered).
    skip_invisible skip axes blanked with set_visible(False) (unused grid cells),
                   so the sequence has no gaps over the panels a reader sees.
    force          letter even a single-panel figure (default: no-op below 2).
    start          index into the letter sequence, for a figure whose panels are
                   drawn by more than one call.
    suffix         string appended to each letter. Default "" gives the bare
                   "a", "b" of the Nature panel style, which is what every
                   figure in the manuscript now uses. Pass ")" to reproduce a
                   figure as it was rendered before 2026-07-30.

    Returns the list of axes that were lettered.
    """
    axs = flatten_axes(axes)
    if skip_invisible:
        axs = [a for a in axs if a.get_visible()]
    if len(axs) < 2 and not force:
        return []                      # single-panel figure: Nature adds no letter
    letters = panel_letter_sequence(start + len(axs))[start:]
    for ax, letter in zip(axs, letters):
        ax.text(x, y, f"{letter}{suffix}", transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold",
                va="bottom", ha="left")
    return axs


# ===========================================================================
#  Minus sign in hand-formatted numbers  (added 2026-09-07)
# ===========================================================================
# matplotlib's own tick formatters render a negative value with U+2212 MINUS
# SIGN (rcParam axes.unicode_minus, left at its default True everywhere in this
# project). Python's own number formatting does NOT: f"{-2.8:+.1f}" emits
# U+002D HYPHEN-MINUS, which is shorter and sits lower than the glyph on the
# neighbouring axis. Wherever a script writes a number into a legend entry, a
# panel title or a hand-set tick label, the two conventions therefore end up
# side by side in one figure -- in one case ("median -2.8 pp; 3% < −30") inside
# a single string.
#
# `mfmt` formats a number the way matplotlib would label it. Use it at EVERY
# site that turns a possibly-negative value into figure text; leave the
# automatic tick labels alone, they are already correct.
MINUS = "−"


def mfmt(x, spec=".2f"):
    """format(x, spec) with the hyphen replaced by a true minus sign (U+2212).

    `spec` is an ordinary format specification, so mfmt(v, "+.1f") and
    mfmt(v, ".2g") stand in directly for f"{v:+.1f}" and f"{v:.2g}".
    """
    return format(float(x), spec).replace("-", MINUS)


# ===========================================================================
#  Year ranges in figure text  (added 2026-09-07)
# ===========================================================================
# The period strings ('2041-2060', '2081-2100', '2001-2020') are DATA KEYS:
# they index the caches, name the output files and are matched against the
# 'period' column. They are also printed straight into panel titles, row
# labels and tick labels, which is where the ASCII hyphen becomes wrong -- a
# span of years takes an en dash (Nature Portfolio journals print 1850–1900,
# 2036–2065; Garner 2016 pp. 373-374 gives "as an equivalent of 'to'" as the
# first use of the en dash).
#
# `plabel` converts ONLY for display. Never feed its output back into a cache
# lookup, a glob or a file name: the key stays hyphenated everywhere else.
EN_DASH = "–"

# Deliberately narrow: FOUR digits, hyphen, FOUR digits. An ISO date
# ('2001-01-01', 4-2-2) does not match, and neither does a scenario name
# ('SSP5-8.5') or a model name ('CNRM-CM6-1'), so plabel is safe to apply to a
# title that mixes a period with either of those.
_YEAR_RANGE = re.compile(r"(?<!\d)(\d{4})-(\d{4})(?!\d)")


def plabel(period):
    """Display form of a period string: '2081-2100' -> '2081–2100'.

    Converts every YYYY-YYYY span in the string and nothing else, so it can be
    handed a bare period or a longer label that contains one. Non-string input
    is returned unchanged, which keeps a NaN or a None from raising inside a
    title f-string.
    """
    if not isinstance(period, str):
        return period
    return _YEAR_RANGE.sub(rf"\1{EN_DASH}\2", period)


# ===========================================================================
#  Percentile-index labels  (added 2026-08-17)
# ===========================================================================
# Removes three naming collisions that ran through the figures and the
# manuscript:
#
#   1. "Q99" was used for the 99th percentile of TEMPERATURE and of
#      PRECIPITATION as well as of FLOW. Q is the discharge symbol, so a
#      discharge symbol was carrying a temperature.
#
#   2. "Q95"/"Q05" collide with the flow-duration-curve convention of the
#      low-flow literature (Smakhtin 2001, J. Hydrol. 240, 147-186; Gustard &
#      Demuth 2009, WMO Manual on Low-flow Estimation and Prediction), in which
#      Q95 is the flow equalled or exceeded 95% of the time -- a LOW flow.
#      Here q95 = numpy.quantile(q, 0.95), a HIGH flow. Opposite meaning, same
#      symbol.
#
#   3. The two conventions were both live in the code base and inverted with
#      respect to each other: apply_deltas_utils.py:1769-1772 builds
#      q5_q95_ratio from q05e = quantile(0.95) ("high flow") and
#      q95e = quantile(0.05) ("low flow"), so inside that ratio "Q95" is the
#      LOW flow while the standalone q95 index is labelled "High flow (Q95)".
#      The arithmetic is right; only the naming was inverted.
#
# The convention adopted here:
#
#      * Q is reserved for discharge. Temperature and precipitation percentiles
#        use their own symbol (T, P).
#      * Every percentile index carries a `p` subscript and is a percentile of
#        the DAILY-VALUE distribution, ascending. So Q_p95 is a high flow and
#        Q_p05 is a low flow. This is the complement of the flow-duration
#        convention; state that explicitly at first use in the Methods.
#      * Ratio indices are written high-over-low in the new notation, so
#        Q_p95/Q_p05 replaces the old "Q5/Q95" and can only be read one way.
#
# SCOPE: this table covers only the indices that carried the ambiguity. It is
# deliberately NOT a complete index dictionary -- AnnMax, RL20, the ETCCDI
# precipitation indices and so on are unambiguous already and keep the labels
# they have in flow_metrics.py. Look a key up with `percentile_label()`, which
# returns None for anything not covered, so a call site can write:
#
#      lab = percentile_label(col)
#      title = lab.long if lab else EXISTING_LABEL[col]
#
# NOTE ON UNITS: this table carries NO units field, deliberately. A unit is a
# property of the quantity a panel plots, not of the index. The same Q_p95
# appears as a baseline magnitude (mm day-1, baseline.py), as a projected
# change (%, hydro_indices_change.py) and as a reduced-minus-full bias
# (percentage points, "pp", screen_effect_maps.py). Any single string stored
# here would be wrong in two of those three figures, and wrong silently. Take
# the unit from what the panel plots.

from collections import namedtuple

PercentileLabel = namedtuple(
    "PercentileLabel", "long short plain tex definition")

# Keys are the canonical data-layer names actually used in the caches: the raw
# statistic name for flow indices (`q95`), and the full column name for the
# climate columns (`temp_q99_change_degC`). `percentile_label()` also accepts
# the `delta_rel_` / `delta_abs_` prefixed flow columns.
PERCENTILE_LABELS = {
    # ---- climate forcing ---------------------------------------------------
    "temp_q99_change_degC": PercentileLabel(
        long="Temperature, 99th percentile",
        short=r"$T_{p99}$",
        plain="Tp99",
        tex=r"$T_{p99}$",
        definition="99th percentile of daily mean near-surface air temperature",
    ),
    "temp_q5_change_degC": PercentileLabel(
        long="Temperature, 5th percentile",
        short=r"$T_{p05}$",
        plain="Tp05",
        tex=r"$T_{p05}$",
        definition="5th percentile of daily mean near-surface air temperature",
    ),
    # The three temperature percentiles below and the two precipitation ones
    # further down complete the climate-forcing set (added 2026-08-30). They
    # carry exactly the collision the table was written for -- a bare "Q95" on
    # a temperature axis -- and were simply missing, so the percentile
    # spectrum figure had to hand-type its own symbols.
    "temp_q95_change_degC": PercentileLabel(
        long="Temperature, 95th percentile",
        short=r"$T_{p95}$",
        plain="Tp95",
        tex=r"$T_{p95}$",
        definition="95th percentile of daily mean near-surface air temperature",
    ),
    "temp_q90_change_degC": PercentileLabel(
        long="Temperature, 90th percentile",
        short=r"$T_{p90}$",
        plain="Tp90",
        tex=r"$T_{p90}$",
        definition="90th percentile of daily mean near-surface air temperature",
    ),
    "temp_q50_change_degC": PercentileLabel(
        long="Temperature, 50th percentile",
        short=r"$T_{p50}$",
        plain="Tp50",
        tex=r"$T_{p50}$",
        definition=("50th percentile (median) of daily mean near-surface air "
                    "temperature"),
    ),
    "precip_q99_change_pct": PercentileLabel(
        long="Precipitation, 99th percentile",
        short=r"$P_{p99}$",
        plain="Pp99",
        tex=r"$P_{p99}$",
        definition="99th percentile of daily precipitation",
    ),
    "precip_q95_change_pct": PercentileLabel(
        long="Precipitation, 95th percentile",
        short=r"$P_{p95}$",
        plain="Pp95",
        tex=r"$P_{p95}$",
        definition="95th percentile of daily precipitation",
    ),
    "precip_q90_change_pct": PercentileLabel(
        long="Precipitation, 90th percentile",
        short=r"$P_{p90}$",
        plain="Pp90",
        tex=r"$P_{p90}$",
        definition="90th percentile of daily precipitation",
    ),

    # The two MEAN columns (added 2026-08-30). A mean is not a percentile, so
    # these stretch the table's name, but an axis that puts the mean next to
    # the percentiles of the same variable has to draw both from one symbol
    # family -- a bare "Mean" beside T_p05 leaves the reader to guess which
    # variable it belongs to. The multi-letter subscript is upright while the
    # percentile p stays italic, following the usual typesetting rule for a
    # descriptive subscript. `long` reproduces screen_effect_maps.MAP_LABEL
    # verbatim so that figure's titles are unchanged by these two keys.
    "temp_change_degC": PercentileLabel(
        long="Temperature, mean",
        short=r"$T_{\mathrm{mean}}$",
        plain="Tmean",
        tex=r"$T_{\mathrm{mean}}$",
        definition="mean of daily mean near-surface air temperature",
    ),
    "precip_change_pct": PercentileLabel(
        long="Precipitation, mean",
        short=r"$P_{\mathrm{mean}}$",
        plain="Pmean",
        tex=r"$P_{\mathrm{mean}}$",
        definition="mean of daily precipitation",
    ),
    # ---- streamflow percentiles -------------------------------------------
    "q99": PercentileLabel(
        long=r"Very high flow ($Q_{p99}$)",
        short=r"$Q_{p99}$",
        plain="Qp99",
        tex=r"$Q_{p99}$",
        definition="99th percentile of daily flow",
    ),
    "q95": PercentileLabel(
        long=r"High flow ($Q_{p95}$)",
        short=r"$Q_{p95}$",
        plain="Qp95",
        tex=r"$Q_{p95}$",
        definition="95th percentile of daily flow",
    ),
    "q50": PercentileLabel(
        long=r"Median flow ($Q_{p50}$)",
        short=r"$Q_{p50}$",
        plain="Qp50",
        tex=r"$Q_{p50}$",
        definition="50th percentile (median) of daily flow",
    ),
    "q05": PercentileLabel(
        long=r"Low flow ($Q_{p05}$)",
        short=r"$Q_{p05}$",
        plain="Qp05",
        tex=r"$Q_{p05}$",
        definition="5th percentile of daily flow",
    ),

    # Mean runoff (added 2026-08-30), the discharge companion to the two mean
    # climate columns above. A mean is not a percentile, but a panel that puts
    # mean runoff beside Q_p95 needs both drawn from one symbol family. `long`
    # reproduces screen_effect_maps.MAP_LABEL verbatim so that figure's titles
    # are unchanged by this key.
    "qmean": PercentileLabel(
        long="Mean runoff",
        short=r"$Q_{\mathrm{mean}}$",
        plain="Qmean",
        tex=r"$Q_{\mathrm{mean}}$",
        definition="mean of daily flow, reported as mean annual runoff",
    ),
    # ---- ratio indices -----------------------------------------------------
    # Old labels "Q5/Q95" and "Q10/Q90" were written in the EXCEEDANCE
    # convention while every neighbouring index was written in the percentile
    # convention, so the same symbol Q95 meant a high flow in one panel and a
    # low flow in the next. Written high-over-low below, they cannot be
    # misread; the computed values are unchanged.
    "q5_q95_ratio": PercentileLabel(
        long=r"$Q_{p95}/Q_{p05}$ ratio",
        short=r"$Q_{p95}/Q_{p05}$",
        plain="Qp95/Qp05",
        tex=r"$Q_{p95}/Q_{p05}$",
        definition="ratio of the 95th to the 5th percentile of daily flow",
    ),
    "q10_q90_ratio": PercentileLabel(
        long=r"$Q_{p90}/Q_{p10}$ ratio",
        short=r"$Q_{p90}/Q_{p10}$",
        plain="Qp90/Qp10",
        tex=r"$Q_{p90}/Q_{p10}$",
        definition="ratio of the 90th to the 10th percentile of daily flow",
    ),
}

# One sentence to paste at first use in the Methods and in any caption that
# names a percentile index, so the convention is stated rather than assumed.
PERCENTILE_CONVENTION_NOTE = (
    "Percentile indices carry a $p$ subscript and are percentiles of the "
    "daily-value distribution, so $Q_{p95}$ is a high flow. This is the "
    "complement of the flow-duration convention, in which $Q_{95}$ denotes "
    "the flow exceeded on 95% of days."
)

_DELTA_PREFIXES = ("delta_rel_", "delta_abs_", "delta_")


def percentile_label(key):
    """Return the PercentileLabel for `key`, or None if it is not a percentile
    index covered by this table.

    Accepts the raw statistic name (``q95``), the delta column
    (``delta_rel_q95``) and the climate columns (``temp_q99_change_degC``).
    Returning None rather than raising is deliberate: a call site keeps its
    existing label for every index that never carried the ambiguity.
    """
    if key in PERCENTILE_LABELS:
        return PERCENTILE_LABELS[key]
    for pre in _DELTA_PREFIXES:
        if key.startswith(pre):
            return PERCENTILE_LABELS.get(key[len(pre):])
    return None
