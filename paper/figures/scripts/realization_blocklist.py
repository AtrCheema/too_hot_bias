"""
realization_blocklist.py — single source of truth for the CMIP6 realizations
excluded from the too-hot figures because the underlying delta is corrupt or a
degenerate qqmap_day artefact (verified June 2026; see diagnostics/ and the
project notes).

Imported by fig02_precip_change.py, build_streamflow_cache.py,
hydro_indices_change.py and analyze_full_vs_reduced_significance.py so the
SAME exclusion is applied to both the precip/temp cache and the streamflow
cache — keeping the figures and the significance test consistent and impossible
to drift apart.

Entries are (model, member, scenario, period); None is a wildcard.
  * NESM3 r2i1p1f1 (all scen/periods): qqmap_day DRY-TAIL ARTEFACT. The raw
    CMIP6 change is a sane ~+6%, but utils.quantile_mapping divides the FULL
    sorted daily distribution (incl. the ~2/3 near-zero dry tail), so ~10.5% of
    ranks hit the +10x clip -> a forcing-insensitive ~+52% catchment-mean
    wetting in EVERY scenario/period (and contaminated qsim/stats downstream).
    Root cause is in quantile_mapping; the durable fix is wet-day-only intensity
    + a recompute. Until then, exclude the tripped realization.
  * EC-Earth3 r103i1p1f1 / r104i1p1f1, ssp585 (all periods): CORRUPT DOWNLOAD —
    the ssp585 2081-2100 future pr file is all-NaN; only stale 1-station
    leftovers (~-84%) remain in the caches. Their ssp245 data is healthy and
    is intentionally NOT blocked.

`model`/`member` strings here match the CMIP6 forms used in both stats.csv
(model="NESM3", member="r2i1p1f1") and the fig02 cache, so the match is exact.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# (model, member, scenario, period) — None = wildcard (matches anything)
BLOCKLIST = [
    ("NESM3",     "r2i1p1f1",   None,     None),
    ("EC-Earth3", "r103i1p1f1", "ssp585", None),
    ("EC-Earth3", "r104i1p1f1", "ssp585", None),
]


def is_blocked(model, member, scenario, period):
    """True if (model, member, scenario, period) matches any BLOCKLIST entry;
    a None field in an entry is a wildcard."""
    for bm, bmem, bscn, bper in BLOCKLIST:
        if (bm in (None, model) and bmem in (None, member)
                and bscn in (None, scenario) and bper in (None, period)):
            return True
    return False


def apply_blocklist(df, label="cache"):
    """Vectorised removal of BLOCKLIST realizations from a long-format frame
    that carries model/member/scenario/period columns (the fig02, fig05 and
    streamflow caches all do). Logs how many rows were dropped. No-op (with a
    warning) if the frame is empty or is missing the key columns, so it is safe
    to call defensively on any cache."""
    if df is None or len(df) == 0 or not BLOCKLIST:
        return df
    needed = {"model", "member", "scenario", "period"}
    missing = needed - set(df.columns)
    if missing:
        logger.warning(f"apply_blocklist: {label} missing column(s) {missing}; "
                       "blocklist NOT applied")
        return df
    blocked = pd.Series(False, index=df.index)
    for bm, bmem, bscn, bper in BLOCKLIST:
        m = pd.Series(True, index=df.index)
        if bm is not None:
            m &= df["model"].astype(str) == bm
        if bmem is not None:
            m &= df["member"].astype(str) == bmem
        if bscn is not None:
            m &= df["scenario"].astype(str) == bscn
        if bper is not None:
            m &= df["period"].astype(str) == bper
        blocked |= m
    n = int(blocked.sum())
    if n:
        nsta = (df.loc[~blocked, "station"].nunique()
                if "station" in df.columns else "?")
        logger.info(f"Blocklist removed {n} of {len(df)} rows from {label}; "
                    f"{nsta} stations remain. Entries: {BLOCKLIST}")
    else:
        logger.info(f"Blocklist matched 0 rows in {label} (already clean).")
    return df.loc[~blocked].copy()
