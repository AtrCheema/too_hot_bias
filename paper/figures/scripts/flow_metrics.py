"""
flow_metrics.py
===============

Single source of truth for the change metrics that ``apply_deltas_stn.py``
writes into every per-station ``stats.parquet``.

Why this file exists
--------------------
``stats.parquet`` carries 41 future-vs-baseline change metrics (a ``delta_abs_``
and a ``delta_rel_`` twin per base name; ``hfd_mean`` has only the absolute
twin). Before this module, each downstream script re-declared its own hand-typed
subset -- ``build_streamflow_cache.py`` took whatever started with ``delta_``,
``analyze_full_vs_reduced_significance.py`` listed 12 flow columns and
``hydro_indices_change.py`` listed 14 -- so a metric added upstream silently
never reached the figures. Every consumer now imports the registry below, and a
new metric in ``apply_deltas_utils.add_change_vs_baseline`` needs one entry here.

Layout
------
``FLOW_METRICS``   -- the 29 STREAMFLOW change metrics, ordered
                      drought/low-flow -> median -> high-flow/flood -> shape ->
                      flashiness -> timing. That order is the x-axis of the
                      significance spectrum and the bias-ranking figures.
``PRECIP_INDEX_METRICS`` -- the 12 ETCCDI-style PRECIPITATION indices that
                      apply_deltas_stn.py also records in stats.parquet (they
                      describe the bias-corrected FORCING, not the flow, so they
                      are kept in a separate group and never mixed into a
                      streamflow figure).

Each entry is a ``Metric``:
    base      stats.parquet base name (e.g. "q95")
    column    the delta column actually analysed ("delta_rel_q95")
    label     short axis/tick label ("Q95")
    name      reader-facing name ("High flow (Q95)")
    family    grouping key (see FAMILY_ORDER)
    units     units of `column` AFTER the standard x100 rescale of delta_rel_*
              ("%" for every relative metric; "days" for the absolute HFD shift)

Definitions come from ``apply_deltas_utils.py``: ``compute_flow_stats``,
``flood_return_levels``, ``fdc_signatures``, ``low_flow_signatures``,
``high_flow_frequency``, ``dry_spell_stats``, ``flashiness_stats`` and
``compute_precip_stats``.
"""

from collections import namedtuple

Metric = namedtuple("Metric", "base column label name family units")

# family order = the left-to-right order of every cross-metric figure
FAMILY_ORDER = ["lowflow", "central", "highflow", "shape", "flashiness",
                "timing"]

# ---------------------------------------------------------------------------
#  STREAMFLOW change metrics (29)
# ---------------------------------------------------------------------------
# Sign conventions worth remembering when reading the figures:
#   * dry_spell_* / low_q_freq / low_q_deficit / slope_fdc / q*_q*_ratio rise
#     when the regime gets FLASHIER or DRIER, i.e. their sign is not the same
#     "wetter = positive" as the magnitude metrics.
#   * flashiness_event_frt is a rising TIME: a NEGATIVE change = a faster flood
#     peak, the opposite sense of the magnitude metrics.
#   * hfd_mean is a DATE (day of water year), so only its absolute shift is
#     meaningful; a relative change of a date is not. apply_deltas_stn.py wraps
#     it to +/- half a year (circular difference).
FLOW_METRICS = [
    # ---- drought / low flow ------------------------------------------------
    Metric("dry_spell_max",     "delta_rel_dry_spell_max",     "DrySpellMax", "Longest dry spell",              "lowflow",  "%"),
    Metric("dry_spell_mean",    "delta_rel_dry_spell_mean",    "DrySpellMean","Mean dry-spell length",          "lowflow",  "%"),
    Metric("mean_annual_min",   "delta_rel_mean_annual_min",   "AnnMin",      "Mean annual minimum",            "lowflow",  "%"),
    Metric("q7min_min",         "delta_rel_q7min_min",         "Q7min",       "Minimum 7-day flow",             "lowflow",  "%"),
    Metric("mam7",              "delta_rel_mam7",              "MAM7",        "Mean annual 7-day minimum",      "lowflow",  "%"),
    Metric("q7_10",             "delta_rel_q7_10",             "7Q10",        "7-day 10-year low flow",         "lowflow",  "%"),
    Metric("low_q_freq",        "delta_rel_low_q_freq",        "LowQFreq",    "Low-flow day frequency",         "lowflow",  "%"),
    Metric("low_q_deficit",     "delta_rel_low_q_deficit",     "LowQDef",     "Low-flow deficit volume",        "lowflow",  "%"),
    Metric("q05",               "delta_rel_q05",               "Q05",         "Low flow (Q05)",                 "lowflow",  "%"),
    # ---- central tendency --------------------------------------------------
    Metric("q50",               "delta_rel_q50",               "Q50",         "Median flow (Q50)",              "central",  "%"),
    Metric("qmean",             "delta_rel_qmean",             "Mean",        "Mean annual runoff",             "central",  "%"),
    Metric("runoff_ratio",      "delta_rel_runoff_ratio",      "Q/P",         "Runoff ratio (Q/P)",             "central",  "%"),
    # ---- high flow / flood -------------------------------------------------
    Metric("q95",               "delta_rel_q95",               "Q95",         "High flow (Q95)",                "highflow", "%"),
    Metric("q99",               "delta_rel_q99",               "Q99",         "Very high flow (Q99)",           "highflow", "%"),
    Metric("mean_annual_max",   "delta_rel_mean_annual_max",   "AnnMax",      "Mean annual maximum",            "highflow", "%"),
    Metric("flood_q10",         "delta_rel_flood_q10",         "RL10",        "10-year flood return level",     "highflow", "%"),
    Metric("flood_q20",         "delta_rel_flood_q20",         "RL20",        "20-year flood return level",     "highflow", "%"),
    Metric("flood_q50",         "delta_rel_flood_q50",         "RL50",        "50-year flood return level",     "highflow", "%"),
    Metric("high_q_freq",       "delta_rel_high_q_freq",       "HighQFreq",   "High-flow event frequency",      "highflow", "%"),
    Metric("high_q_dur",        "delta_rel_high_q_dur",        "HighQDur",    "High-flow event duration",       "highflow", "%"),
    # ---- variability / FDC shape ------------------------------------------
    Metric("qstd",              "delta_rel_qstd",              "SD",          "Flow standard deviation",        "shape",    "%"),
    Metric("slope_fdc",         "delta_rel_slope_fdc",         "FDCslope",    "Flow-duration-curve slope",      "shape",    "%"),
    Metric("q5_q95_ratio",      "delta_rel_q5_q95_ratio",      "Q5/Q95",      "Q5:Q95 ratio",                   "shape",    "%"),
    Metric("q10_q90_ratio",     "delta_rel_q10_q90_ratio",     "Q10/Q90",     "Q10:Q90 ratio",                  "shape",    "%"),
    # ---- flashiness --------------------------------------------------------
    Metric("flashiness_rb",          "delta_rel_flashiness_rb",          "FlashRB",  "Richards-Baker flashiness", "flashiness", "%"),
    Metric("flashiness_event",       "delta_rel_flashiness_event",       "FlashEvt", "Event flashiness",          "flashiness", "%"),
    Metric("flashiness_event_frt",   "delta_rel_flashiness_event_frt",   "RiseTime", "Flood rising time (FRT)",   "flashiness", "%"),
    Metric("flashiness_event_sfd",   "delta_rel_flashiness_event_sfd",   "StormDep", "Storm-flow depth (SFD)",    "flashiness", "%"),
    # ---- timing (absolute only: a date has no meaningful relative change) --
    Metric("hfd_mean",          "delta_abs_hfd_mean",          "HFD",         "Half-flow date",                 "timing",   "days"),
]

# ---------------------------------------------------------------------------
#  PRECIPITATION indices (12) -- describe the bias-corrected FORCING
# ---------------------------------------------------------------------------
# These are recorded in the same stats.parquet rows as the flow metrics, so they
# are duplicated across the two hydrological engines (the forcing does not
# depend on the engine). Any analysis of them must therefore read ONE engine.
PRECIP_INDEX_METRICS = [
    Metric("prcptot",              "delta_rel_prcptot",              "PRCPTOT", "Total wet-day precipitation", "central",  "%"),
    Metric("rx1day",               "delta_rel_rx1day",               "RX1day",  "Maximum 1-day precipitation", "highflow", "%"),
    Metric("rx5day",               "delta_rel_rx5day",               "RX5day",  "Maximum 5-day precipitation", "highflow", "%"),
    Metric("r95ptot",              "delta_rel_r95ptot",              "R95pTOT", "Very wet-day precipitation",  "highflow", "%"),
    Metric("r99ptot",              "delta_rel_r99ptot",              "R99pTOT", "Extremely wet-day precipitation", "highflow", "%"),
    Metric("r10mm",                "delta_rel_r10mm",                "R10mm",   "Heavy precipitation days",    "highflow", "%"),
    Metric("r20mm",                "delta_rel_r20mm",                "R20mm",   "Very heavy precipitation days", "highflow", "%"),
    Metric("cdd_annual_max_mean",  "delta_rel_cdd_annual_max_mean",  "CDDmean", "Consecutive dry days (annual mean)", "lowflow", "%"),
    Metric("cdd_annual_max_max",   "delta_rel_cdd_annual_max_max",   "CDDmax",  "Consecutive dry days (maximum)",     "lowflow", "%"),
    Metric("precip_dry_spell_mean","delta_rel_precip_dry_spell_mean","PDrySpMean", "Mean precipitation dry spell",    "lowflow", "%"),
    Metric("precip_dry_spell_max", "delta_rel_precip_dry_spell_max", "PDrySpMax",  "Longest precipitation dry spell", "lowflow", "%"),
    Metric("n_precip_dry_spells",  "delta_rel_n_precip_dry_spells",  "PDrySpN",    "Number of precipitation dry spells", "lowflow", "%"),
]

ALL_METRICS = FLOW_METRICS + PRECIP_INDEX_METRICS

FLOW_COLUMNS         = [m.column for m in FLOW_METRICS]
PRECIP_INDEX_COLUMNS = [m.column for m in PRECIP_INDEX_METRICS]
ALL_COLUMNS          = FLOW_COLUMNS + PRECIP_INDEX_COLUMNS

BY_COLUMN = {m.column: m for m in ALL_METRICS}
# x-order for every cross-metric figure: registry order
ORDER = {m.label: i for i, m in enumerate(ALL_METRICS)}


def present(metrics, columns):
    """Subset of `metrics` whose `column` exists in `columns` (a cache schema).

    Nothing is silently dropped elsewhere: every consumer filters through this,
    so a metric missing from a cache is visible in that script's own log.
    """
    have = set(columns)
    return [m for m in metrics if m.column in have]


def missing(metrics, columns):
    """The `metrics` whose `column` is absent from `columns` -- log these."""
    have = set(columns)
    return [m for m in metrics if m.column not in have]
