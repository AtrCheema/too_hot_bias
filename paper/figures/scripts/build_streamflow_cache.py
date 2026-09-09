"""
build_streamflow_cache.py
=========================

Stage-1 cache builder for the STREAMFLOW version of the full-vs-reduced
(too-hot-removed) significance analysis. It plays the same role that
fig02_precip_change.py's compute_cache() plays for precipitation/temperature,
but the expensive hydrology is already done: the per-station stats.csv files
written by apply_deltas_stn.py already hold, for every realization, the
change-vs-baseline flow statistics (delta_abs_* / delta_rel_*). So this script
just READS, FILTERS, tidies and CONCATENATES them into one long-format parquet
that analyze_full_vs_reduced_significance.py can consume column-by-column.

Output -> paper/figures/fig_data/stats_change.parquet
    cols: station, lon, lat, scenario, model, member, period, engine,
          <every delta_* flow-change column found in stats.csv>

Notes
-----
* One stats.csv per station folder under CONFIG.out_path; a station with no
  stats.csv (e.g. HBV was never calibrated there) is simply skipped.
* Only FUTURE rows are kept (scenario != 'baseline'); the change is already
  relative to that station's own historical baseline run.
* TWO hydrological engines are present (engine = 'hbv' / 'lstm'). They are kept
  as a column, NOT pooled -- the downstream significance test must be run on one
  engine at a time, otherwise HBV-vs-LSTM (hydrological-model) differences would
  leak into the climate-model / internal-variability variance partition.
* The recommended cross-catchment metrics are the RELATIVE ones (delta_rel_*,
  in %), analogous to precip_change_pct: e.g.
      delta_rel_qmean              -> mean annual runoff
      delta_rel_q95 / _mean_annual_max  -> peak flow (floods)
      delta_rel_q05 / _mean_annual_min / _dry_spell_max -> low flow / drought
  delta_abs_* are kept too, in case an additive metric is preferred.

Run:
    python paper/figures/scripts/build_streamflow_cache.py            # all stations
    python paper/figures/scripts/build_streamflow_cache.py 200        # first 200 (quick test)
"""

import os
import sys
import glob
import logging

import numpy as np
import pandas as pd

# this file lives in paper/figures/scripts/; config.py is at the repo root, three dirs up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # flow_metrics.py is here
from config import CONFIG
# same exclusion list fig02/fig05/significance use
from realization_blocklist import apply_blocklist
# the registry of EVERY change metric apply_deltas_stn.py writes into
# stats.parquet -- used here only to audit that the cache really carries them
from flow_metrics import ALL_METRICS, FLOW_METRICS, PRECIP_INDEX_METRICS

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HERE     = os.path.dirname(os.path.abspath(__file__))
FIG_DATA = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")  # per-method cache on /ibex (avoids cross-method contamination)
OUT_PARQUET = os.path.join(FIG_DATA, "stats_change.parquet")

# realization identifier columns carried over verbatim from stats.parquet
ID_COLS = ["scenario", "model", "member", "period", "engine"]

# First-N stations only (int) for a quick test; None = all. Also overridable
# from the command line: `python build_streamflow_cache.py 200`.
MAX_STATIONS = None
if len(sys.argv) > 1:
    MAX_STATIONS = int(sys.argv[1])


def build():
    stat_files = sorted(glob.glob(os.path.join(CONFIG.out_path, CONFIG.delta_method, "*", "stats.parquet")))
    if MAX_STATIONS is not None:
        stat_files = stat_files[:MAX_STATIONS]
    logger.info(f"{len(stat_files)} station(s) have stats.parquet under "
                f"{os.path.join(CONFIG.out_path, CONFIG.delta_method)}"
                f"{'' if MAX_STATIONS is None else f' (capped at {MAX_STATIONS})'}")

    # station lon/lat for the maps, from the same static.csv fig02 uses
    static = pd.read_csv(os.path.join(CONFIG.gscad_path, "dataframes", "static.csv"),
                         comment="#", index_col=0)
    lon_s = static["long"].astype(float)
    lat_s = static["lat"].astype(float)

    skip = {"read_fail": 0, "no_future": 0, "no_static": 0}
    frames = []
    for i, f in enumerate(stat_files):
        stn = os.path.basename(os.path.dirname(f))
        try:
            sdf = pd.read_parquet(f)
        except Exception as e:
            logger.warning(f"  {stn}: cannot read stats.parquet ({e})")
            skip["read_fail"] += 1
            continue

        # keep only future rows (drop the historical_baseline rows)
        sdf = sdf[sdf["scenario"].astype(str) != "baseline"]
        if sdf.empty:
            skip["no_future"] += 1
            continue

        ids  = [c for c in ID_COLS if c in sdf.columns]
        # EVERY change metric, not a hand-picked subset: both the delta_abs_ and
        # the delta_rel_ twin of all 41 bases (29 streamflow + 12 precipitation
        # indices). The downstream metric choice is made in flow_metrics.py, so
        # a metric added upstream reaches the analysis without touching this file.
        dels = [c for c in sdf.columns if c.startswith("delta_")]
        out = sdf[ids + dels].copy()
        # float32: the analysis (significance, bootstrap) casts to float32 anyway
        # -- doing it here halves both the concat peak and the parquet, which
        # matters now that the cache carries every metric over ~20.6k stations.
        out[dels] = out[dels].astype("float32")
        out.insert(0, "station", stn)
        if stn in lon_s.index:
            out.insert(1, "lon", float(lon_s.loc[stn]))
            out.insert(2, "lat", float(lat_s.loc[stn]))
        else:
            out.insert(1, "lon", np.nan)
            out.insert(2, "lat", np.nan)
            skip["no_static"] += 1
        frames.append(out)

        if (i + 1) % 200 == 0:
            logger.info(f"  [{i+1}/{len(stat_files)}] read; "
                        f"{sum(len(fr) for fr in frames)} rows so far")

    if not frames:
        raise RuntimeError("No stats.parquet produced any future rows -- check "
                           "CONFIG.out_path/CONFIG.delta_method and that apply_deltas_stn.py has run.")

    df = pd.concat(frames, ignore_index=True)
    for c in ("station", "scenario", "model", "member", "period", "engine"):
        if c in df.columns:
            df[c] = df[c].astype(str)

    # drop blocklisted realizations (corrupt / degenerate qqmap_day deltas) so
    # the streamflow Stage-1 cache is clean for fig05 + the significance test.
    df = apply_blocklist(df, label="streamflow stats_change cache")

    os.makedirs(FIG_DATA, exist_ok=True)
    # parquet only: the downstream consumers (analyze_full_vs_reduced_significance
    # .py, bootstrap_random_removal.py) read stats_change.parquet via read_parquet;
    # nothing reads a .csv twin, which for this cache is tens of GB to write.
    df.to_parquet(OUT_PARQUET)

    # accounting so the run is auditable
    n_used = df["station"].nunique()
    engines = sorted(df["engine"].unique()) if "engine" in df else []
    scens   = sorted(df["scenario"].unique())
    pers    = sorted(df["period"].unique())
    dels = [c for c in df.columns if c.startswith("delta_")]
    logger.info("Stage-1 streamflow cache accounting:")
    logger.info(f"  scanned (had stats.parquet) : {len(stat_files)}")
    for reason, n in skip.items():
        logger.info(f"  skipped [{reason}] : {n}")
    logger.info(f"  USED stations           : {n_used}")
    logger.info(f"  rows cached             : {len(df)}")
    logger.info(f"  engines                 : {engines}")
    logger.info(f"  scenarios               : {scens}")
    logger.info(f"  periods                 : {pers}")
    logger.info(f"  delta_* columns cached  : {len(dels)}")

    # METRIC AUDIT -- the point of the rebuild. Every metric the registry knows
    # about must be in the cache AND have finite values; anything absent or
    # all-NaN is named here rather than silently vanishing from the figures.
    logger.info("Metric audit (registry -> cache), non-null share of rows:")
    n_rows = float(len(df))
    for group, metrics in (("streamflow", FLOW_METRICS),
                           ("precip-index", PRECIP_INDEX_METRICS)):
        logger.info(f"  --- {group} ({len(metrics)} metrics) ---")
        for m in metrics:
            if m.column not in df.columns:
                logger.warning(f"    MISSING  {m.column:36s} ({m.name})")
                continue
            frac = float(df[m.column].notna().sum()) / n_rows if n_rows else 0.0
            flag = "  ALL-NaN" if frac == 0.0 else ""
            logger.info(f"    {m.column:36s} {frac:6.1%}{flag}")
    extra = sorted(set(dels) - {m.column for m in ALL_METRICS})
    if extra:
        logger.info(f"  cached but not in the registry ({len(extra)}): {extra}")
    logger.info(f"Cached -> {OUT_PARQUET}")
    return df


if __name__ == "__main__":
    build()
