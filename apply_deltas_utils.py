

import os
import glob
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union, Dict, Tuple, Optional, List

import xarray as xr

import numpy as np
import pandas as pd
import geopandas as gpd

import shapefile

import geojson
from geojson import Polygon, MultiPolygon

import rasterio

from models.hbv import hbv_nb

from utils import to_minus180_180_coords


logger = logging.getLogger(__name__)


def apply_mon_dist_delta_pr(
    obs: pd.Series,
    intensity_mq: np.ndarray,   # shape (12, 10), multiplicative factors on wet-day quantiles
    freq_delta_m: np.ndarray,   # shape (12,),    additive change in wet-day fraction in [-1, 1]
    wet_thr: float = 1.0,       # mm/day; must match the unit of obs
    q_levels: np.ndarray | None = None,
) -> pd.Series:
    """
    Apply monthly distributional deltas to daily observed precipitation.

    Combines:
      (1) Multiplicative wet-day intensity scaling per within-month rank quantile bin.
      (2) Rank-based wet-day frequency adjustment: promote driest dry days / demote
          smallest wet days to match target wet-day count for each calendar month.
    """
    intensity_mq = np.asarray(intensity_mq, dtype=np.float32)
    freq_delta_m = np.asarray(freq_delta_m, dtype=np.float32)
    if intensity_mq.shape != (12, 10):
        raise ValueError(f"intensity_mq must have shape (12, 10), got {intensity_mq.shape}")
    if freq_delta_m.shape != (12,):
        raise ValueError(f"freq_delta_m must have shape (12,), got {freq_delta_m.shape}")

    if q_levels is None:
        q_levels = np.linspace(0.05, 0.95, 10, dtype=np.float32)
    q_levels = np.asarray(q_levels, dtype=np.float32)

    out = obs.astype("float32").copy()

    for m in range(1, 13):
        mask = (obs.index.month == m)
        if not mask.any():
            continue

        sm = obs.loc[mask].to_numpy(dtype=np.float32)
        n_days = len(sm)
        is_wet = sm > wet_thr
        n_wet = int(is_wet.sum())

        sm_new = sm.copy()

        # --- (1) intensity scaling on wet days (rank within wet-day subset) ---
        if n_wet > 0:
            wet_vals = sm[is_wet]
            pct = pd.Series(wet_vals).rank(pct=True, method="average").to_numpy(dtype=np.float32)
            qi = np.clip(np.searchsorted(q_levels, pct, side="right") - 1, 0, 9)
            sm_new[is_wet] = wet_vals * intensity_mq[m - 1, qi]

        # --- (2) frequency adjustment (rank-based, deterministic) ---
        n_wet_target = int(np.rint(n_wet + freq_delta_m[m - 1] * n_days))
        n_wet_target = max(0, min(n_days, n_wet_target))
        diff = n_wet_target - n_wet

        if diff < 0:
            # demote the driest |diff| wet days to zero
            wet_idx = np.where(is_wet)[0]
            demote = wet_idx[np.argsort(sm_new[wet_idx])[: -diff]]
            sm_new[demote] = 0.0
        elif diff > 0:
            # promote the wettest `diff` dry days to a genuine light-wet value.
            # A promoted day MUST end up > wet_thr or the wet-day count never
            # actually increases. The old approach (wet_thr * intensity_mq[m-1,0])
            # silently produced sub-threshold "wet" days whenever the lowest
            # wet-quantile factor was <= 1, so a wetter future never gained wet
            # days. Instead draw the value from the low end of THIS month's
            # perturbed wet-day distribution, with a hard floor just above wet_thr.
            dry_idx = np.where(~is_wet)[0]
            n_promote = min(diff, len(dry_idx))
            if n_promote > 0:
                promote = dry_idx[np.argsort(-sm[dry_idx])[:n_promote]]
                wet_now = sm_new[sm_new > wet_thr]
                promo_val = float(np.quantile(wet_now, 0.05)) if wet_now.size else wet_thr * 1.1
                promo_val = max(promo_val, wet_thr * (1.0 + 1e-3))
                sm_new[promote] = np.float32(promo_val)

        out.loc[mask] = sm_new

    return out


def apply_qqmap_day_delta_pr(
    obs: pd.Series,
    intensity_by_month: Dict[int, np.ndarray],   # {month: array of rank-indexed multiplicative factors}
    freq_delta_m: np.ndarray,                    # shape (12,); additive wet-day-fraction change
    wet_thr: float = 1.0,                        # mm/day; must match obs units
) -> pd.Series:
    """
    Apply qqmap_day (CDF-mapping) deltas to daily observed precipitation.

    Works in two stages, mirroring ``apply_mon_dist_delta_pr``:
      (1) Rank-based multiplicative intensity scaling. For each calendar
          month, each observed day is located at its rank-within-month; the
          delta at that rank is looked up and applied. If the CMIP6 record
          length per month differs from the obs record length, the delta
          curve is linearly interpolated onto the obs rank axis (a warning
          is emitted so the caller knows the pairing wasn't exact).
      (2) Rank-based wet-day frequency adjustment. Using ``freq_delta_m``,
          the target wet-day count is computed and the driest wet days are
          demoted to zero, or the wettest dry days are promoted to a small
          wet value, to match the target.

    Parameters
    ----------
    obs : pd.Series
        Daily observed precipitation indexed by datetime.
    intensity_by_month : dict {1..12 -> np.ndarray}
        Per-month rank-indexed multiplicative factors (output of
        ``catch_avg_for_qqmap_day_deltas``). Array length N_m equals the
        number of CMIP6 daily samples for that month.
    freq_delta_m : np.ndarray, shape (12,)
        Additive change in wet-day fraction per calendar month, in [-1, 1].
    wet_thr : float
        Wet-day threshold in the same units as ``obs`` (mm/day for MSWEP).

    Returns
    -------
    pd.Series
        Future-perturbed daily precipitation on the same index as ``obs``.
    """
    freq_delta_m = np.asarray(freq_delta_m, dtype=np.float32)
    if freq_delta_m.shape != (12,):
        raise ValueError(f"freq_delta_m must have shape (12,), got {freq_delta_m.shape}")

    out = obs.astype("float32").copy()

    for m in range(1, 13):
        mask = (obs.index.month == m)
        if not mask.any():
            continue
        if m not in intensity_by_month:
            # No delta for this month (e.g. CMIP6 gap) -> leave obs unchanged.
            continue

        sm = obs.loc[mask].to_numpy(dtype=np.float32)
        n_days = len(sm)
        delta_m = np.asarray(intensity_by_month[m], dtype=np.float32)
        n_delta = len(delta_m)

        # --- (1) rank-based intensity scaling on all days ----------------
        # rank(method="first") gives 1..n_days dense integer ranks and
        # breaks ties in index order; subtract 1 for 0-based indexing.
        # Looking up delta[rank] on every day is equivalent to "sort obs,
        # multiply by delta pairwise, unsort" but avoids the explicit
        # sort/argsort round-trip.
        ranks = pd.Series(sm).rank(method="first").to_numpy(dtype=np.int64) - 1

        if n_days == n_delta:
            per_day = delta_m[ranks]
        else:
            # CMIP6 and obs have different per-month sample counts. Treat
            # delta as a function of normalized rank in [0, 1] and linearly
            # interpolate onto obs's rank positions. Warn so the caller
            # knows rank pairing wasn't exact.
            # Demoted to debug: the obs/CMIP6 per-month sample counts almost
            # always differ, so at INFO this fires for every month × realization
            # × station and bloats the .out logs. The interpolation it warns
            # about is the intended fallback, not an error.
            logger.debug(
                f"apply_qqmap_day_delta_pr: month={m} sample-count mismatch "
                f"(obs={n_days}, delta={n_delta}); interpolating delta onto "
                "the obs rank axis."
            )
            x_src = np.linspace(0.0, 1.0, n_delta, dtype=np.float32)
            x_tgt = ranks.astype(np.float32) / max(n_days - 1, 1)
            per_day = np.interp(x_tgt, x_src, delta_m).astype(np.float32)

        sm_new = sm * per_day

        # --- (2) wet-day frequency adjustment (rank-based) ---------------
        # Multiplicative factors alone cannot turn a dry day wet or vice
        # versa, so shift the wet-day count toward its target explicitly.
        is_wet = sm > wet_thr
        n_wet = int(is_wet.sum())
        n_wet_target = int(np.rint(n_wet + freq_delta_m[m - 1] * n_days))
        n_wet_target = max(0, min(n_days, n_wet_target))
        diff = n_wet_target - n_wet

        if diff < 0:
            # demote the |diff| driest wet days to zero
            wet_idx = np.where(is_wet)[0]
            demote = wet_idx[np.argsort(sm_new[wet_idx])[: -diff]]
            sm_new[demote] = 0.0
        elif diff > 0:
            # promote the `diff` wettest dry days to a genuine light-wet value.
            # A promoted day MUST end up > wet_thr or the wet-day count never
            # actually increases. The old approach (wet_thr * min(delta_m))
            # silently produced sub-threshold "wet" days whenever the smallest
            # intensity factor was <= 1 (the normal case once dry ranks are 1.0),
            # so a wetter future never gained wet days. Instead draw the value
            # from the low end of THIS month's perturbed wet-day distribution
            # (a realistic light wet day), with a hard floor just above wet_thr.
            dry_idx = np.where(~is_wet)[0]
            n_promote = min(diff, len(dry_idx))
            if n_promote > 0:
                promote = dry_idx[np.argsort(-sm[dry_idx])[:n_promote]]
                wet_now = sm_new[sm_new > wet_thr]
                promo_val = float(np.quantile(wet_now, 0.05)) if wet_now.size else wet_thr * 1.1
                promo_val = max(promo_val, wet_thr * (1.0 + 1e-3))
                sm_new[promote] = np.float32(promo_val)

        out.loc[mask] = sm_new

    return out


def apply_qqmap_day_delta_temp(
    obs: pd.Series,
    delta_by_month: Dict[int, np.ndarray],   # {month: array of rank-indexed additive deltas}
) -> pd.Series:
    """
    Apply qqmap_day (CDF-mapping) additive deltas to daily observed
    temperature (tmin / tmax / tas).

    Mirrors ``apply_qqmap_day_delta_pr`` but uses additive deltas and has
    no wet-day-frequency step (temperature has no point mass). Emits a
    warning if the per-month obs and delta lengths differ and falls back
    to linear interpolation of the delta onto the obs rank axis.

    Parameters
    ----------
    obs : pd.Series
        Daily observed temperature indexed by datetime.
    delta_by_month : dict {1..12 -> np.ndarray}
        Per-month rank-indexed additive deltas.

    Returns
    -------
    pd.Series
        Future-perturbed daily temperature on the same index as ``obs``.
    """
    out = obs.astype("float32").copy()

    for m in range(1, 13):
        mask = (obs.index.month == m)
        if not mask.any():
            continue
        if m not in delta_by_month:
            continue

        sm = obs.loc[mask].to_numpy(dtype=np.float32)
        n_days = len(sm)
        delta_m = np.asarray(delta_by_month[m], dtype=np.float32)
        n_delta = len(delta_m)

        # rank-based lookup: the day at rank r in obs gets the delta stored
        # at rank r in CMIP6 (see apply_qqmap_day_delta_pr for the reasoning).
        ranks = pd.Series(sm).rank(method="first").to_numpy(dtype=np.int64) - 1

        if n_days == n_delta:
            per_day = delta_m[ranks]
        else:
            # Demoted to debug for the same reason as in
            # apply_qqmap_day_delta_pr (avoids huge .out logs).
            logger.debug(
                f"apply_qqmap_day_delta_temp: month={m} sample-count mismatch "
                f"(obs={n_days}, delta={n_delta}); interpolating delta onto "
                "the obs rank axis."
            )
            x_src = np.linspace(0.0, 1.0, n_delta, dtype=np.float32)
            x_tgt = ranks.astype(np.float32) / max(n_days - 1, 1)
            per_day = np.interp(x_tgt, x_src, delta_m).astype(np.float32)

        out.loc[mask] = sm + per_day

    return out


def apply_mon_dist_delta(
    obs: pd.Series,
    delta_mq: np.ndarray,
    op: str,
    q_levels: np.ndarray | None = None,
    keep_zeros: bool = False,
) -> pd.Series:
    """
    Apply monthly distributional deltas to a daily observed series.

    obs: daily series with DatetimeIndex
    delta_mq: array shape (12, 10) [month, quantile]
    op: "mul" for multiplicative (e.g., pr), "add" for additive (e.g., tas/tmin/tmax)
    q_levels: quantile probabilities corresponding to the 10 bins (default: 0.05..0.95)
    keep_zeros: if True and op=="mul", keep obs==0 as 0 (useful for precipitation)
    """
    delta_mq = np.asarray(delta_mq, dtype=np.float32)
    if delta_mq.shape != (12, 10):
        raise ValueError(f"delta_mq must have shape (12, 10), got {delta_mq.shape}")

    if q_levels is None:
        q_levels = np.linspace(0.05, 0.95, 10, dtype=np.float32)
    q_levels = np.asarray(q_levels, dtype=np.float32)

    out = pd.Series(index=obs.index, dtype="float32")

    for m in range(1, 13):
        mask = obs.index.month == m
        sm = obs.loc[mask]
        if sm.empty:
            continue

        # within-month percentile rank of each daily value (0..1]
        pct = sm.rank(pct=True, method="average").to_numpy(dtype=np.float32)

        # map percentile -> quantile bin 0..9
        qi = np.searchsorted(q_levels, pct, side="right") - 1
        qi = np.clip(qi, 0, 9)

        d = delta_mq[m - 1, qi]  # (n_days_in_month,)

        x = sm.to_numpy(dtype=np.float32)
        if op == "mul":
            y = x * d
            if keep_zeros:
                y = np.where(x == 0.0, 0.0, y)
        elif op == "add":
            y = x + d
        else:
            raise ValueError(f"op must be 'mul' or 'add', got {op}")

        out.loc[mask] = y

    return out


def build_future_data(
        deltas,
        fut_per: Tuple[str, str],
        obs_pr: pd.Series,
        obs_tmin: pd.Series,
        obs_tmax: pd.Series,
        obs_temp: pd.Series,
        lat_val: float,
        CONFIG,
):

    st_yr, en_yr = fut_per[0].split('-')[0], fut_per[1].split('-')[0]
    per_key = f"{st_yr}-{en_yr}"

    delta_pr         = deltas[f"{per_key}_pr"]
    delta_pr_wetfreq = deltas.get(f"{per_key}_pr_wetfreq", None)
    delta_tmin       = deltas[f"{per_key}_tmin"]
    delta_tmax       = deltas[f"{per_key}_tmax"]

    # Restrict obs to the historical window so rank/sample counts match the deltas.
    hist_start = pd.Timestamp(CONFIG.historical_period[0])
    hist_end   = pd.Timestamp(CONFIG.historical_period[1])
    obs_pr_h   = obs_pr.loc[hist_start:hist_end]
    obs_tmin_h = obs_tmin.loc[hist_start:hist_end]
    obs_tmax_h = obs_tmax.loc[hist_start:hist_end]
    obs_temp_h = obs_temp.loc[hist_start:hist_end]

    if CONFIG.delta_method == "scalar":
        fut_pr_h   = obs_pr_h   * delta_pr
        fut_tmin_h = obs_tmin_h + delta_tmin
        fut_tmax_h = obs_tmax_h + delta_tmax
    else:
        if delta_pr_wetfreq is None:
            raise KeyError(
                f"Missing '{per_key}_pr_wetfreq' in deltas. "
                "Re-run calc_deltas_2d.py so pr deltas include the wetfreq variable."
            )
        if CONFIG.delta_method == "qqmap_bin_day":
            fut_pr_h   = apply_mon_dist_delta_pr(
                obs_pr_h, intensity_mq=delta_pr, freq_delta_m=delta_pr_wetfreq, wet_thr=1.0,
            )
            fut_tmin_h = apply_mon_dist_delta(obs_tmin_h, delta_tmin, op="add")
            fut_tmax_h = apply_mon_dist_delta(obs_tmax_h, delta_tmax, op="add")
        elif CONFIG.delta_method in ("qqmap_day", "qqmap_day2"):
            fut_pr_h   = apply_qqmap_day_delta_pr(obs_pr_h, delta_pr, delta_pr_wetfreq)
            fut_tmin_h = apply_qqmap_day_delta_temp(obs_tmin_h, delta_tmin)
            fut_tmax_h = apply_qqmap_day_delta_temp(obs_tmax_h, delta_tmax)
        else:
            raise ValueError(f"Unsupported delta_method {CONFIG.delta_method}")

    # Temperature: CMIP6 gives only tasmin/tasmax deltas here (no tas), so the
    # future mean-temperature CHANGE is taken as the average of the additive
    # tmin/tmax deltas actually applied, and added to the observed mean-temp
    # product. This keeps baseline and future 'temp' on the SAME definition (the
    # MSWX mean-temp product) so the projected change is purely climatic, and it
    # keeps the LSTM 'temp' feature in the distribution it was trained on --
    # unlike redefining temp as (tmin+tmax)/2, which injected a spurious
    # non-climatic offset between baseline and future.
    temp_delta_h = ((fut_tmin_h - obs_tmin_h) + (fut_tmax_h - obs_tmax_h)) / 2.0
    fut_temp_h   = (obs_temp_h + temp_delta_h).rename("temp")

    # Prepend the warmup year with raw (non-perturbed) obs.
    warmup_start = hist_start - pd.Timedelta(days=365)
    warmup_end   = hist_start - pd.Timedelta(days=1)

    fut_pr   = pd.concat([obs_pr.loc[warmup_start:warmup_end],   fut_pr_h  ]).rename("pcp")
    fut_tmin = pd.concat([obs_tmin.loc[warmup_start:warmup_end], fut_tmin_h]).rename("tmin")
    fut_tmax = pd.concat([obs_tmax.loc[warmup_start:warmup_end], fut_tmax_h]).rename("tmax")
    fut_tas  = pd.concat([obs_temp.loc[warmup_start:warmup_end], fut_temp_h ]).rename("temp")

    fut_data = build_forcing_dataframe(
        pcp=fut_pr,
        tmin=fut_tmin,
        tmax=fut_tmax,
        temp=fut_tas,
        START_DATE=warmup_start,
        END_DATE=hist_end,
        qobs=None,
    )

    fut_pet = calculate_pet(fut_data, lat=lat_val)
    fut_data.insert(0, "pet", fut_pet, True)

    return fut_data


def save_mean_deltas_as_nc(
    mean_deltas: Dict[str, Dict[str, np.ndarray]],
    fpath: Union[str, os.PathLike],
):

    realizations = sorted(mean_deltas.keys())

    # collect periods from inner dict keys (strip trailing _pr_wetfreq, _pr, _tmin, _tmax, _tas)
    periods_set = set()
    for r in realizations:
        for k in mean_deltas[r].keys():
            if k.endswith("_pr_wetfreq"):
                per = k[: -len("_pr_wetfreq")]
            else:
                per, _v = k.rsplit("_", 1)
            periods_set.add(per)
    periods = sorted(periods_set)

    vars_mq = ["pr", "tmin", "tmax", "tas"]
    vars_m  = ["pr_wetfreq"]

    n_r, n_p = len(realizations), len(periods)
    n_m, n_q = 12, 10

    data_mq = {v: np.full((n_r, n_p, n_m, n_q), np.nan, dtype=np.float32) for v in vars_mq}
    data_m  = {v: np.full((n_r, n_p, n_m),      np.nan, dtype=np.float32) for v in vars_m}

    per_to_i  = {p: i for i, p in enumerate(periods)}
    real_to_i = {r: i for i, r in enumerate(realizations)}

    for r in realizations:
        ri = real_to_i[r]
        inner = mean_deltas[r]
        for k, arr in inner.items():
            a = np.asarray(arr, dtype=np.float32)
            if k.endswith("_pr_wetfreq"):
                per = k[: -len("_pr_wetfreq")]
                if a.shape != (12,):
                    raise ValueError(f"{r} {k} has shape {a.shape}, expected (12,)")
                data_m["pr_wetfreq"][ri, per_to_i[per], :] = a
            else:
                per, v = k.rsplit("_", 1)
                if v not in data_mq:
                    continue
                if a.shape != (12, 10):
                    raise ValueError(f"{r} {k} has shape {a.shape}, expected (12, 10)")
                data_mq[v][ri, per_to_i[per], :, :] = a

    ds_vars = {v: (("realization", "period", "month", "quantile"), data_mq[v]) for v in vars_mq}
    ds_vars.update({v: (("realization", "period", "month"), data_m[v]) for v in vars_m})

    ds_mean_deltas = xr.Dataset(
        ds_vars,
        coords={
            "realization": realizations,
            "period":      periods,
            "month":       np.arange(1, 13, dtype=np.int16),
            "quantile":    np.arange(1, 11, dtype=np.int16),
        },
        attrs={
            "description": "Catchment-mean monthly distributional deltas",
            "shape_intensity": "(12 months, 10 quantiles)",
            "shape_wetfreq":   "(12 months,)",
        },
    )

    encoding = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
                for v in (vars_mq + vars_m)}
    ds_mean_deltas.to_netcdf(fpath, encoding=encoding)
    return


def save_mean_deltas_qqmap_day_as_nc(
    mean_deltas: Dict[str, Dict],
    fpath: Union[str, os.PathLike],
):
    """
    Serialise qqmap_day catchment-mean deltas to netCDF.

    Input layout (produced by ``catch_avg_deltas_qqmap_day``):
        mean_deltas[realization] = {
            "{period}_pr":         {1: np.ndarray(N_m1,), ..., 12: np.ndarray(N_m12,)},
            "{period}_pr_wetfreq": np.ndarray(12,),
            "{period}_tmin":       {1: np.ndarray(N_m1,), ..., 12: ...},
            "{period}_tmax":       {1: ..., ..., 12: ...},
            ...
        }

    Per-month sample counts (N_m) differ between months (Jan has ~20*31,
    Feb ~20*28, etc.) but are the same across realizations because they
    all span the same CMIP6 historical window. We therefore flatten the
    12 monthly arrays into one ``sample`` axis of length N_total =
    sum_m N_m, and carry ``sample_month`` (1..12) and ``sample_rank``
    (rank-within-month, 0..N_m-1) as auxiliary coords so that reload
    code can split by month and look up by rank.

    Output Dataset
    --------------
    ``pr``, ``tmin``, ``tmax`` : (realization, period, sample)
        Rank-indexed catchment-mean deltas. Multiplicative for pr,
        additive for temperature.
    ``pr_wetfreq`` : (realization, period, month)
        Additive change in wet-day fraction.
    Coords:
        realization, period         -- labels for the outer dims
        month         (1..12)       -- dim coord of pr_wetfreq
        sample_month  (1..12)       -- aux coord on 'sample'
        sample_rank   (0..N_m-1)    -- aux coord on 'sample'

    NOTE: aux coords are named ``sample_month`` / ``sample_rank`` (not
    ``month`` / ``rank``) to avoid collision with ``month`` as a dim
    coord and with xarray's built-in ``rank()`` method, matching the
    convention used by ``quantile_mapping``.
    """
    realizations = sorted(mean_deltas.keys())

    # ---- 1. collect the set of periods across all realizations -----------
    periods_set = set()
    for r in realizations:
        for k in mean_deltas[r].keys():
            if k.endswith("_pr_wetfreq"):
                per = k[: -len("_pr_wetfreq")]
            else:
                per, _v = k.rsplit("_", 1)
            periods_set.add(per)
    periods = sorted(periods_set)

    # ---- 2. infer per-month sample counts from the first dict entry ------
    # All realizations are assumed to share the same 20-year historical
    # window and the same calendar, so N_m per month is constant. If a
    # realization disagrees, we fail loudly in step 4.
    n_per_month: Optional[Dict[int, int]] = None
    for r in realizations:
        for k, v in mean_deltas[r].items():
            if isinstance(v, dict):
                n_per_month = {int(m): int(len(arr)) for m, arr in v.items()}
                break
        if n_per_month is not None:
            break
    if n_per_month is None:
        raise ValueError(
            "save_mean_deltas_qqmap_day_as_nc: no per-month delta dicts "
            "found in mean_deltas; nothing to save."
        )

    # ---- 3. build the flat 'sample' axis with month + rank aux coords ----
    # Layout: Jan ranks 0..N_jan-1, then Feb ranks 0..N_feb-1, etc.
    months_in_order = sorted(n_per_month.keys())
    sample_month_list: List[int] = []
    sample_rank_list: List[int] = []
    for m in months_in_order:
        N_m = n_per_month[m]
        sample_month_list.extend([m] * N_m)
        sample_rank_list.extend(range(N_m))
    sample_month_arr = np.asarray(sample_month_list, dtype=np.int16)
    sample_rank_arr = np.asarray(sample_rank_list, dtype=np.int32)
    n_total = len(sample_month_arr)

    # offset of each month's block along the sample axis
    mo_offset: Dict[int, int] = {}
    cum = 0
    for m in months_in_order:
        mo_offset[m] = cum
        cum += n_per_month[m]

    vars_sample = ["pr", "tmin", "tmax"]
    vars_m = ["pr_wetfreq"]

    n_r, n_p = len(realizations), len(periods)
    n_m = 12

    data_sample = {v: np.full((n_r, n_p, n_total), np.nan, dtype=np.float32) for v in vars_sample}
    data_m = {v: np.full((n_r, n_p, n_m),    np.nan, dtype=np.float32) for v in vars_m}

    per_to_i = {p: i for i, p in enumerate(periods)}
    real_to_i = {r: i for i, r in enumerate(realizations)}

    # ---- 4. fill the arrays from the nested dicts ------------------------
    bad_realizations = set()
    
    for r in realizations:
        ri = real_to_i[r]
        inner = mean_deltas[r]

        bad_r = False

        for k, arr in inner.items():

            if bad_r:
                break

            if k.endswith("_pr_wetfreq"):
                per = k[: -len("_pr_wetfreq")]
                a = np.asarray(arr, dtype=np.float32)
                if a.shape != (12,):
                    raise ValueError(f"{r} {k} has shape {a.shape}, expected (12,)")
                data_m["pr_wetfreq"][ri, per_to_i[per], :] = a
                continue

            # intensity/additive rank-indexed delta: dict {month: array(N_m,)}
            per, v = k.rsplit("_", 1)
            if v not in data_sample:
                # unknown trailing token (e.g. 'tas' if not generated); skip quietly.
                continue
            if not isinstance(arr, dict):
                raise ValueError(
                    f"{r} {k} expected dict {{month: array}}, got {type(arr).__name__}"
                )

            for m, mon_arr in arr.items():
                mon_arr = np.asarray(mon_arr, dtype=np.float32)
                if int(m) not in n_per_month:
                    raise ValueError(f"{r} {k} month {m} not in expected months {months_in_order}")
                expected_N = n_per_month[int(m)]
                if len(mon_arr) != expected_N:
                    # Per-month N_m drifted between realizations — bias-correction
                    # would be ill-defined, so fail loudly instead of padding.
                    logger.warning(
                        f"{r} {k} month {m} has length {len(mon_arr)}, "
                        f"expected {expected_N}. All realizations must share "
                        "the same per-month sample count."
                    )

                    bad_realizations.add(r)
                    bad_r = True
                    break

                off = mo_offset[int(m)]
                data_sample[v][ri, per_to_i[per], off:off + expected_N] = mon_arr

    # ---- 5. assemble the Dataset -----------------------------------------
    ds_vars = {v: (("realization", "period", "sample"), data_sample[v]) for v in vars_sample}
    ds_vars.update({v: (("realization", "period", "month"), data_m[v]) for v in vars_m})

    ds_mean_deltas = xr.Dataset(
        ds_vars,
        coords={
            "realization": realizations,
            "period":      periods,
            # dim coord for pr_wetfreq -- integer month labels 1..12
            "month":       np.arange(1, 13, dtype=np.int16),
            # aux coords on the flat 'sample' axis -- let consumers rebuild
            # the per-month blocks by filtering on sample_month.
            "sample_month": ("sample", sample_month_arr),
            "sample_rank":  ("sample", sample_rank_arr),
        },
        attrs={
            "description":     "Catchment-mean qqmap_day (CDF-mapping) deltas",
            "shape_intensity": "(realization, period, sample); split by sample_month / sample_rank",
            "shape_wetfreq":   "(realization, period, month)",
            "n_per_month":     ",".join(f"{m}={n_per_month[m]}" for m in months_in_order),
        },
    )

    if bad_realizations:
        keep_realizations = [r for r in realizations if r not in bad_realizations]
        ds_mean_deltas = ds_mean_deltas.sel(realization=keep_realizations)

    encoding = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
                for v in (vars_sample + vars_m)}
    ds_mean_deltas.to_netcdf(fpath, encoding=encoding)
    return


def all_files_have_all_vars(
        files:List[str],
        vars:List[str],
        verbosity=0,
        )->bool:
    
    for file in files:
        ds = xr.open_dataset(file, engine='netcdf4')
        for var in vars:
            if var not in ds:
                if verbosity:
                    logger.warning(f"{var} not found in {file}")
                ds.close()
                return False
        ds.close()
    return True


def outputs_exist(out_dir: str, key: str) -> bool:
    return os.path.isfile(os.path.join(out_dir, f"{key}_metrics.json"))


def to_rasterio_dataset(
        grid,
        raster_idx: Union[int, str],
        savepath:str = None,
        dtype: Optional[Union[str, np.dtype]] = None,
        verbosity:int = 0
) -> Tuple[str, str, tuple]:
    """saves the data as raster/.tif file so that it can be loaded as rasterio
     dataset"""

    if savepath is None:
        savepath = os.path.join(os.getcwd(), "results")
        if not os.path.exists(savepath):
            os.mkdir(savepath)

    if isinstance(grid, str):
        raise ValueError(f"Invalid grid of {grid}")

    elif isinstance(grid, Path):
        raise ValueError(f"Invalid grid {grid} of type {type(grid)}")

    elif isinstance(grid, np.ndarray):
        raise ValueError("Cannot convert numpy array to raster, please save it as .tif or .mat file and provide the path")

    elif isinstance(grid, xr.DataArray):

        # todo : maybe we should rename only if the names are not already x/y
        grid = grid.rename({'lon': 'x', 'lat': 'y'})

        raster_name = raster_idx
        shape = grid.shape
        raster = os.path.join(savepath, f"{raster_name}.tif")
        grid.rio.to_raster(raster, dtype=dtype)

    else:
        raise ValueError(f"""
Invalid raster provided at position {raster_idx} which has type {type(grid)}
""")

    return raster, raster_name, shape


def extract_static(
        boundary: Union[Polygon, MultiPolygon],
        raster: str,
        log_msgs: str = '',
        ):
    """
    calcuate average by intersecting raster with boundary"""
    with rasterio.open(raster) as src:

    # if the src.dtypes are int/uint, then nodata=np.nan returns error
    # therefore using filled=False, see comparison between filled vs without filled

        out_image, _ = rasterio.mask.mask(
            src,
            [boundary],
            all_touched=True,
            # nodata=np.nan, it is ignored when filled is False
            filled=False,
            crop=True,
        )

    if np.isnan(out_image).all():
        val = -9991  # case 1, when all values inside boundary are nans
    else:
        val = np.nanmean(out_image)

        log_msgs += f"{out_image.size} cells found"

    del out_image
    del src
    del _

    return val, log_msgs


def remove_tmp_files(tmp_path, prefix:str):
    files = glob.glob(os.path.join(tmp_path, f"{prefix}_*.tif"))
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass
    return


def get_catch_avg_deltas(
        path:str,
        key:str,
        catchment,
        tmp_path,
        jobid:str,
        delta_method:str = "scalar"
):
    """
    Calculates the catchment average value of deltas.
    """ 
    scenario, model, member = key
    deltaP = xr.open_dataset(os.path.join(path, f'{scenario}_{model}_{member}_pr.nc'))
    deltaTmin = xr.open_dataset(os.path.join(path, f'{scenario}_{model}_{member}_tasmin.nc'))
    deltaTmax = xr.open_dataset(os.path.join(path, f'{scenario}_{model}_{member}_tasmax.nc'))
    # deltaTavg = xr.open_dataset(os.path.join(path, f'{scenario}_{model}_{member}_tas.nc'))

    if delta_method == "scalar":
        mean_vars = catch_avg_deltas_scalar(deltaP, deltaTmin, deltaTmax, f"{jobid}_{key}", tmp_path, catchment)
    elif delta_method == "qqmap_bin_day":
        mean_vars = catch_avg_deltas_qqmap_bin_day(deltaP, deltaTmin, deltaTmax, f"{jobid}_{key}", tmp_path, catchment)
    elif delta_method in ("qqmap_day", "qqmap_day2"):
        mean_vars = catch_avg_deltas_qqmap_day(deltaP, deltaTmin, deltaTmax, f"{jobid}_{key}", tmp_path, catchment)

    deltaP.close()
    deltaTmin.close()
    deltaTmax.close()

    remove_tmp_files(tmp_path, prefix=jobid)
    remove_tmp_files(tmp_path, prefix=f"{jobid}_{key}")
    return mean_vars


def catch_avg_deltas_scalar(
        deltaP:xr.Dataset,
        deltaTmin:xr.Dataset,
        deltaTmax:xr.Dataset,
        jobid:str,
        tmp_path:str,
        catchment,
):
    
    mean_vars = {}

    for scen, ds in deltaP.data_vars.items():

        da = deltaP[scen]
        da = to_minus180_180_coords(da)
        raster, grid_name, shape = to_rasterio_dataset(da, f"{jobid}_{scen}", savepath=tmp_path, verbosity=0)                  
        feature_, _= extract_static(catchment, raster)

        mean_vars[f"{scen}_pr"] = float(feature_)

    for scen, ds in deltaTmin.data_vars.items():

        da = deltaTmin[scen]
        da = to_minus180_180_coords(da)
        raster, grid_name, shape = to_rasterio_dataset(da, f"{jobid}_{scen}", savepath=tmp_path, verbosity=0)                  
        feature_, _= extract_static(catchment, raster)

        mean_vars[f"{scen}_tmin"] = float(feature_)

    for scen, ds in deltaTmax.data_vars.items():

        da = deltaTmax[scen]
        da = to_minus180_180_coords(da)
        raster, grid_name, shape = to_rasterio_dataset(da, f"{jobid}_{scen}", savepath=tmp_path, verbosity=0)                  
        feature_, _= extract_static(catchment, raster)

        mean_vars[f"{scen}_tmax"] = float(feature_)
    return mean_vars    


def catch_avg_deltas_qqmap_bin_day(
        deltaP:xr.Dataset,
        deltaTmin:xr.Dataset,
        deltaTmax:xr.Dataset,
        jobid:str,
        tmp_path:str,
        catchment,
):

    mean_vars = {}

    for period_var, ds in deltaP.data_vars.items():

        da = deltaP[period_var]

        if period_var.endswith("_wetfreq"):
            # wet-day frequency delta: dims (month, lat, lon); catchment-mean -> shape (12,)
            period = period_var[: -len("_wetfreq")]
            month_dim = "month" if "month" in da.dims else da.dims[0]
            n_month = da.sizes[month_dim]

            mean_m = np.full((n_month,), np.nan, dtype=np.float32)

            for mi in range(n_month):
                da2 = da.isel({month_dim: mi})
                da2 = to_minus180_180_coords(da2)
                raster_name = f"{jobid}_{period}_pr_wetfreq_m{mi+1:02d}"
                raster, _, _ = to_rasterio_dataset(da2, raster_name, savepath=tmp_path, verbosity=0)
                feature_, _ = extract_static(catchment, raster)
                mean_m[mi] = float(feature_)

            mean_vars[f"{period}_pr_wetfreq"] = mean_m
            continue

        # intensity delta: dims (month, quantile, lat, lon) -> shape (12, 10)
        month_dim = "month" if "month" in da.dims else da.dims[0]
        quant_dim = "quantile" if "quantile" in da.dims else da.dims[1]

        n_month = da.sizes[month_dim]
        n_quant = da.sizes[quant_dim]

        mean_mq = np.full((n_month, n_quant), np.nan, dtype=np.float32)

        for mi in range(n_month):
            for qi in range(n_quant):

                da2 = da.isel({month_dim: mi, quant_dim: qi})
                da2 = to_minus180_180_coords(da2)

                raster_name = f"{jobid}_{period_var}_pr_m{mi+1:02d}_q{qi+1:02d}"
                raster, _, _ = to_rasterio_dataset(da2, raster_name, savepath=tmp_path, verbosity=0)
                feature_, _ = extract_static(catchment, raster)

                mean_mq[mi, qi] = float(feature_)

        mean_vars[f"{period_var}_pr"] = mean_mq

    for period, ds in deltaTmin.data_vars.items():

        da = deltaTmin[period]

        month_dim = "month" if "month" in da.dims else da.dims[0]
        quant_dim = "quantile" if "quantile" in da.dims else da.dims[1]

        n_month = da.sizes[month_dim]
        n_quant = da.sizes[quant_dim]

        mean_mq = np.full((n_month, n_quant), np.nan, dtype=np.float32)

        for mi in range(n_month):
            for qi in range(n_quant):

                da2 = da.isel({month_dim: mi, quant_dim: qi})
                da2 = to_minus180_180_coords(da2)

                raster_name = f"{jobid}_{period}_tmin_m{mi+1:02d}_q{qi+1:02d}"
                raster, _, _ = to_rasterio_dataset(da2, raster_name, savepath=tmp_path, verbosity=0)
                feature_, _ = extract_static(catchment, raster)

                mean_mq[mi, qi] = float(feature_)

        mean_vars[f"{period}_tmin"] = mean_mq

    for period, ds in deltaTmax.data_vars.items():

        da = deltaTmax[period]

        month_dim = "month" if "month" in da.dims else da.dims[0]
        quant_dim = "quantile" if "quantile" in da.dims else da.dims[1]

        n_month = da.sizes[month_dim]
        n_quant = da.sizes[quant_dim]

        mean_mq = np.full((n_month, n_quant), np.nan, dtype=np.float32)

        for mi in range(n_month):
            for qi in range(n_quant):

                da2 = da.isel({month_dim: mi, quant_dim: qi})
                da2 = to_minus180_180_coords(da2)

                raster_name = f"{jobid}_{period}_tmax_m{mi+1:02d}_q{qi+1:02d}"
                raster, _, _ = to_rasterio_dataset(da2, raster_name, savepath=tmp_path, verbosity=0)
                feature_, _ = extract_static(catchment, raster)

                mean_mq[mi, qi] = float(feature_)

        mean_vars[f"{period}_tmax"] = mean_mq

    # for period, ds in deltaTavg.data_vars.items():

    #     da = deltaTavg[period]

    #     month_dim = "month" if "month" in da.dims else da.dims[0]
    #     quant_dim = "quantile" if "quantile" in da.dims else da.dims[1]

    #     n_month = da.sizes[month_dim]
    #     n_quant = da.sizes[quant_dim]

    #     mean_mq = np.full((n_month, n_quant), np.nan, dtype=np.float32)

    #     for mi in range(n_month):
    #         for qi in range(n_quant):

    #             da2 = da.isel({month_dim: mi, quant_dim: qi})
    #             da2 = to_minus180_180_coords(da2)

    #             raster_name = f"{jobid}_{period}_tas_m{mi+1:02d}_q{qi+1:02d}"
    #             raster, _, _ = to_rasterio_dataset(da2, raster_name, savepath=tmp_path, verbosity=0)
    #             feature_, _ = extract_static(catchment, raster)

    #             mean_mq[mi, qi] = float(feature_)

    #     mean_vars[f"{period}_tas"] = mean_mq

    return mean_vars


def _catch_mean_per_sample(
        da: xr.DataArray,
        catchment,
        ) -> Dict[int, np.ndarray]:
    """
    Vectorised catchment-mean reduction over the ``sample`` dim.

    Replaces the earlier per-sample rasterize-to-disk + rasterio.mask loop
    (which dominated runtime for daily qqmap_day deltas: one TIF write and
    one ``rasterio.mask.mask`` call per sample, times ~7300 samples per
    variable-period). The new version:

      (1) Normalises lon to -180..180 and renames lat/lon -> y/x so
          rioxarray auto-detects the spatial dims.
      (2) Calls ``da.rio.clip(..., drop=True)`` **once** for the whole
          DataArray. Internally rioxarray builds the geometry mask for the
          (y, x) grid using ``rasterio.features.geometry_mask``, crops to
          the catchment bounding box, and broadcasts the 2D mask across
          the leading ``sample`` dim -- so every sample is processed in a
          single pass with no per-sample rasterio overhead.
      (3) Takes a vectorised ``nanmean`` over (y, x). Cells outside the
          geometry are NaN and are excluded by ``skipna=True`` (matching
          the old ``extract_static`` semantics). All-NaN samples fall
          back to ``-9991.0``, the same sentinel ``extract_static`` used.
      (4) Splits the 1D (N_total,) result into per-month blocks ordered
          by ascending rank, so apply-time rank lookup works directly.

    This is typically 100-1000x faster than the previous approach and
    needs no multiprocessing: the dominant cost was disk I/O and the
    geometry-mask rebuild per slice, both of which are now done once.

    Reduces a DataArray with dims (sample, lat, lon) and auxiliary coords
    ``sample_month`` (1..12) and ``rang`` (0..N_m-1) along ``sample`` to a
    per-month dict of catchment-mean arrays.
    """
    # Stash the aux coords before the rename/clip round-trip; they sit on
    # the 'sample' dim and are preserved through rio.clip, but reading
    # them off the original DataArray avoids depending on that contract.
    month_coord = da['sample_month'].values
    rank_coord  = da['rang'].values

    # (1) normalise lon and expose spatial dims as x/y for rioxarray.
    da = to_minus180_180_coords(da)
    da = da.rename({'lon': 'x', 'lat': 'y'})
    if da.rio.crs is None:
        # Delta maps are on a regular CMIP6 lat/lon grid; assume WGS84.
        da = da.rio.write_crs('EPSG:4326')

    # (2) clip once. drop=True crops to the catchment bounding box so the
    # subsequent reduction only touches the handful of cells that can
    # possibly be inside the catchment. all_touched=True matches the
    # behaviour of the old extract_static call (rasterio.mask with
    # all_touched=True).
    clipped = da.rio.clip([catchment], all_touched=True, drop=True, from_disk=False)

    # (3) per-sample catchment mean. skipna=True ignores both the NaN
    # cells introduced by clip (outside the geometry) and any inside-but-
    # NaN data, which matches nanmean semantics in extract_static.
    means_all = clipped.mean(dim=('y', 'x'), skipna=True).values
    means_all = np.where(np.isnan(means_all), -9991.0, means_all).astype(np.float32)

    # (4) split by calendar month; within each month sort by ascending
    # rank so that output position i corresponds to the i-th-lowest rank.
    by_month: Dict[int, np.ndarray] = {}
    for m in range(1, 13):
        sel = (month_coord == m)
        if not np.any(sel):
            continue
        idx = np.where(sel)[0]
        idx = idx[np.argsort(rank_coord[idx])]
        by_month[m] = means_all[idx].copy()

    return by_month


def catch_avg_deltas_qqmap_day(
        deltaP: xr.Dataset,
        deltaTmin: xr.Dataset,
        deltaTmax: xr.Dataset,
        jobid: str,
        tmp_path: str,
        catchment,
        ) -> Dict:
    """
    Catchment-mean reduction of qqmap_day deltas.

    Input Datasets are produced by ``quantile_mapping`` + the save step in
    calc_deltas_2d.py. For each future period (e.g. "2041-2060") the pr
    dataset carries two variables:
        "{period}"          : dims (sample, lat, lon) -- rank-indexed
                              multiplicative intensity factors, with
                              auxiliary coords month (1..12) and rank.
        "{period}_wetfreq"  : dims (month, lat, lon)  -- additive wet-day
                              frequency delta.
    Temperature datasets (tasmin, tasmax) only carry "{period}" with the
    (sample, lat, lon) layout (additive deltas).

    Returns
    -------
    dict
        Keys mirror the mon_dist convention:
          "{period}_pr"          -> dict {month: np.ndarray(N_m,)}  (multiplicative)
          "{period}_pr_wetfreq"  -> np.ndarray(12,)                 (additive)
          "{period}_tmin"        -> dict {month: np.ndarray(N_m,)}  (additive)
          "{period}_tmax"        -> dict {month: np.ndarray(N_m,)}  (additive)
    """
    mean_vars: Dict = {}

    # ---- pr ---------------------------------------------------------------
    # Two variable types live side-by-side in deltaP: "{period}" for the
    # rank-indexed intensity delta and "{period}_wetfreq" for the monthly
    # wet-day frequency delta. Dispatch on the suffix.
    for period_var, _ in deltaP.data_vars.items():
        da = deltaP[period_var]

        if period_var.endswith("_wetfreq"):
            # wet-day frequency delta: identical shape to mon_dist's wetfreq
            # (12, lat, lon), so reuse that per-month rasterize loop.
            period = period_var[: -len("_wetfreq")]
            month_dim = "month" if "month" in da.dims else da.dims[0]
            n_month = da.sizes[month_dim]

            mean_m = np.full((n_month,), np.nan, dtype=np.float32)
            for mi in range(n_month):
                da2 = da.isel({month_dim: mi})
                da2 = to_minus180_180_coords(da2)
                raster_name = f"{jobid}_{period}_pr_wetfreq_m{mi+1:02d}"
                raster, _, _ = to_rasterio_dataset(da2, raster_name, savepath=tmp_path, verbosity=0)
                feature_, _ = extract_static(catchment, raster)
                mean_m[mi] = float(feature_)

            mean_vars[f"{period}_pr_wetfreq"] = mean_m
            remove_tmp_files(tmp_path, prefix=jobid)
            continue

        # intensity delta: (sample, lat, lon) with month/rank coords.
        by_month = _catch_mean_per_sample(da, catchment=catchment)
        mean_vars[f"{period_var}_pr"] = by_month

    # ---- tmin -------------------------------------------------------------
    for period, _ in deltaTmin.data_vars.items():
        da = deltaTmin[period]
        by_month = _catch_mean_per_sample(da, catchment=catchment)
        mean_vars[f"{period}_tmin"] = by_month

    # ---- tmax -------------------------------------------------------------
    for period, _ in deltaTmax.data_vars.items():
        da = deltaTmax[period]
        by_month = _catch_mean_per_sample(da, catchment=catchment)
        mean_vars[f"{period}_tmax"] = by_month

    return mean_vars


def build_forcing_dataframe(
    pcp: pd.Series,
    tmin: pd.Series,
    tmax: pd.Series,
    temp: pd.Series,
    START_DATE: Union[str, pd.Timestamp],
    END_DATE: Union[str, pd.Timestamp],
    qobs: Optional[pd.Series] = None
) -> pd.DataFrame:
    items = [pcp.rename("pcp"), tmin.rename("tmin"), tmax.rename("tmax"), temp.rename("temp")]
    if qobs is not None:

        assert isinstance(qobs, pd.Series), f"qobs should be a pandas Series but it is {type(qobs)}"

        items.append(qobs.rename("qobs"))

    df = pd.concat(items, axis=1).sort_index().loc[START_DATE:END_DATE].copy()
    return df


def hargreaves(tmin, tmax, tmean, lat, day_of_year):
    
    # Loop to reduce memory usage
    pet = np.zeros(tmean.shape,dtype=np.float32)*np.nan
    for ii in np.arange(len(pet[:,0])):
        trange = tmax[ii,:] - tmin[ii,:]
        trange[trange<0] = 0
        
        latitude = np.deg2rad(lat[ii,:])
        
        SOLAR_CONSTANT = 0.0820
        
        sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year[ii,:] - 1.39))
        
        sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec),-1,1))
    
        ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year[ii,:]))
        
        tmp1 = (24.0 * 60.0) / np.pi
        tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
        tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
        et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)
        
        pet[ii,:] = 0.0023 * (tmean[ii,:] + 17.8) * trange ** 0.5 * 0.408 * et_rad
        
    pet[pet<0] = 0
    
    return pet


def calculate_pet(df: pd.DataFrame, lat: float) -> pd.Series:
    pet = hargreaves(
        tmin=df['tmin'].values.reshape(-1, 1),
        tmax=df['tmax'].values.reshape(-1, 1),
        tmean=df['temp'].values.reshape(-1, 1),
        lat=np.full((len(df), 1), fill_value=float(lat)),
        day_of_year=df.index.day_of_year.values.reshape(-1, 1)
    )
    return pd.Series(pet.reshape(-1,), index=df.index, name='pet')


def safe_rel_change(future_val: float, baseline_val: float) -> Optional[float]:
    if baseline_val is None:
        return None
    if not np.isfinite(baseline_val):
        return None
    if np.isclose(baseline_val, 0.0):
        return None
    return float((future_val - baseline_val) / baseline_val)


def add_change_vs_baseline(stats: Dict, baseline_stats: Dict,
                           compare_keys: Optional[list] = None) -> Dict:
    """
    Add absolute and relative change against baseline stats. ``compare_keys``
    defaults to the streamflow stats; pass an explicit list to reuse this for
    other metrics (e.g. precipitation dry-spell).
    """
    if compare_keys is None:
        compare_keys = [
            "qmean", "qstd", "q05", "q50", "q95", "q99",
            "mean_annual_max", "mean_annual_min",
            "dry_spell_mean", "dry_spell_max",
            "flashiness_rb",
            "flashiness_event", "flashiness_event_frt", "flashiness_event_sfd",
            # marginal distributional signatures
            "flood_q10", "flood_q20", "flood_q50",
            "slope_fdc", "q5_q95_ratio", "q10_q90_ratio",
            "mam7", "q7min_min", "q7_10",
            "high_q_freq", "high_q_dur",
            "low_q_freq", "low_q_deficit",
            "runoff_ratio",
        ]

    for k in compare_keys:
        fut = stats.get(k)
        base = baseline_stats.get(k)

        if fut is None or base is None:
            stats[f"delta_abs_{k}"] = None
            stats[f"delta_rel_{k}"] = None
        else:
            stats[f"delta_abs_{k}"] = float(fut - base)
            stats[f"delta_rel_{k}"] = safe_rel_change(fut, base)

    # hfd_mean is a date, not a magnitude: delta_abs_ only, wrapped to +/-
    # half a year (circular difference); a relative change of a day-of-year
    # is meaningless, so no delta_rel_hfd_mean is ever written (matches the
    # backfilled stats.parquet schema).
    if "hfd_mean" in stats and "hfd_mean" in baseline_stats:
        fut, base = stats.get("hfd_mean"), baseline_stats.get("hfd_mean")
        if fut is None or base is None:
            stats["delta_abs_hfd_mean"] = None
        else:
            d = float(fut) - float(base)
            d -= 365.25 * round(d / 365.25)
            stats["delta_abs_hfd_mean"] = d

    return stats


def _spell_lengths(mask: np.ndarray) -> np.ndarray:
    """
    Lengths (in array positions) of consecutive ``True`` runs in a boolean
    mask. Counts positions, not calendar days -- callers must ensure the
    series has no gaps if calendar-day spells are required.
    """
    mask = np.asarray(mask, dtype=np.int8)
    if mask.sum() == 0:
        return np.array([], dtype=int)
    padded = np.concatenate(([0], mask, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts


def dry_spell_stats(qsim: pd.Series, threshold: float) -> Dict:
    """
    Dry-spell statistics: a "dry" day is one where ``qsim < threshold``;
    a dry spell is a run of consecutive dry days. ``threshold`` is meant
    to be a *fixed* low-flow value (e.g. the baseline q05) shared across
    baseline and future runs so the metric carries the climate signal.
    """
    q = qsim.dropna()
    spells = _spell_lengths((q < threshold).to_numpy())
    if len(spells) == 0:
        return {"dry_spell_mean": 0.0, "dry_spell_max": 0, "n_dry_spells": 0}
    return {
        "dry_spell_mean": float(spells.mean()),
        "dry_spell_max": int(spells.max()),
        "n_dry_spells": int(len(spells)),
    }


def precip_dry_spell_stats(pcp: pd.Series, wet_thr: float = 1.0) -> Dict:
    """
    Precipitation dry-spell (consecutive-dry-day) statistics. A day is "dry"
    when ``pcp < wet_thr``; a dry spell is a run of consecutive dry days.

    ``wet_thr`` defaults to 1.0 mm/day to match the WMO/ETCCDI wet/dry split
    (wet day RR >= 1 mm, dry day RR < 1 mm), so ``precip_dry_spell_max`` is
    exactly the ETCCDI Consecutive-Dry-Days (CDD) index and the wet/dry
    boundary is consistent with prcptot/percentiles in ``compute_precip_stats``.
    Note the delta-application step (``apply_*_delta_pr``) uses a strict wet
    threshold ``> wet_thr``; that differs from this only at exactly 1.0 mm,
    which is numerically negligible.

    This is a forcing (climate) metric, identical for HBV and LSTM. Compute it
    over the same post-warmup window as the flow stats so the two are comparable.
    """
    p = pd.Series(pcp).dropna()
    if len(p) == 0:
        return {"precip_dry_spell_mean": None, "precip_dry_spell_max": None,
                "n_precip_dry_spells": None}
    spells = _spell_lengths((p.to_numpy() < wet_thr))
    if len(spells) == 0:
        return {"precip_dry_spell_mean": 0.0, "precip_dry_spell_max": 0,
                "n_precip_dry_spells": 0}
    return {
        "precip_dry_spell_mean": float(spells.mean()),
        "precip_dry_spell_max": int(spells.max()),
        "n_precip_dry_spells": int(len(spells)),
    }


def annual_max_cdd(
        pcp: pd.Series,
        wet_thr: float = 1.0,
        min_days_per_year: int = 300,
) -> Dict:
    """
    Mean annual-maximum Consecutive Dry Days (ETCCDI CDD, climatological mean).

    Per calendar year the longest run of dry days (RR < ``wet_thr``) is taken
    (spells reset at the year boundary), then averaged over adequately-covered
    years. This is the standard, robust CDD summary (Zhang et al. 2011, WIREs
    Clim. Change 2, 851-870; Sillmann et al. 2013, JGR-Atmos. 118, 1716-1733;
    climdex.pcic::climdex.cdd), i.e. the period mean of the ANNUAL index -- not
    the single whole-record maximum reported by ``precip_dry_spell_max``, which
    is one extreme order statistic per period and far noisier as a change signal.

    NOTE: CDD is a temporal/sequencing metric. Under the delta/quantile-mapping
    method the dry/wet run structure is inherited from the observed record and
    only crudely perturbed by the wet-frequency adjustment, so this carries a
    weak, incomplete climate signal -- treat it as a caveated forcing diagnostic,
    not a headline result.

    Returns
    -------
    dict with
        cdd_annual_max_mean : mean over years of the per-year longest dry run.
        cdd_annual_max_max  : largest per-year longest dry run over the record.
        n_cdd_years         : number of adequately-covered years used.
    """
    p = pd.Series(pcp).dropna()
    out = {"cdd_annual_max_mean": None, "cdd_annual_max_max": None, "n_cdd_years": None}
    if len(p) == 0:
        return out
    dry = p.to_numpy() < wet_thr
    yr = p.index.year.to_numpy()
    per_year: List[int] = []
    for y in np.unique(yr):
        sel = yr == y
        if sel.sum() < min_days_per_year:
            continue
        spells = _spell_lengths(dry[sel])
        per_year.append(int(spells.max()) if len(spells) else 0)
    if not per_year:
        return out
    per_year = np.asarray(per_year)
    out["cdd_annual_max_mean"] = float(per_year.mean())
    out["cdd_annual_max_max"] = int(per_year.max())
    out["n_cdd_years"] = int(len(per_year))
    return out


def compute_precip_stats(
        pcp: pd.Series,
        wet_thr: float = 1.0,
        p95_ref: Optional[float] = None,
        p99_ref: Optional[float] = None,
) -> Dict:
    """
    Precipitation (forcing) statistics: the ETCCDI heavy-precipitation indices
    plus the consecutive-dry-day metrics (``precip_dry_spell_stats`` merged in).

    This is a climate/forcing metric -- identical for HBV and LSTM -- so it is
    computed once per realization+period on the daily precipitation series, not
    per hydrological engine. Compute it over the same post-warmup window as the
    flow stats so the two are comparable. ``pcp`` in mm/day.

    ETCCDI indices (reported as mean-annual values over the record):
      prcptot  annual total precip on wet days (RR >= wet_thr)
      r10mm / r20mm  annual count of days with RR >= 10 / 20 mm
      r95ptot / r99ptot  annual total precip on days with RR > the wet-day
                         95th / 99th percentile of the REFERENCE period
      rx1day / rx5day  annual maximum 1-day / consecutive-5-day precip
    Definitions: Zhang et al. (2011, WIREs Clim. Change 2, 851-870); Sillmann
    et al. (2013, JGR-Atmos. 118, 1716-1733). Reference software: the ETCCDI /
    climdex family (https://github.com/pacificclimate/climdex.pcic) and xclim
    (https://github.com/Ouranosinc/xclim).

    IMPORTANT (Zhang et al. 2011): R95p/R99p use a FIXED base-period wet-day
    percentile so baseline and future are comparable. Pass the baseline's
    ``p95_ref`` / ``p99_ref`` when scoring a future run; when None they are
    computed from THIS series' wet days and returned (keys ``p95_ref`` /
    ``p99_ref``) so the caller can forward them -- the same fixed-threshold
    pattern used for ``dry_threshold``.
    """
    p = pd.Series(pcp).dropna()
    out = {"prcptot": None, "r10mm": None, "r20mm": None,
           "r95ptot": None, "r99ptot": None, "rx1day": None, "rx5day": None,
           "p95_ref": p95_ref, "p99_ref": p99_ref}

    if len(p) > 0:
        wet = p[p >= wet_thr]
        if p95_ref is None:
            p95_ref = float(wet.quantile(0.95)) if len(wet) else None
        if p99_ref is None:
            p99_ref = float(wet.quantile(0.99)) if len(wet) else None
        out["p95_ref"], out["p99_ref"] = p95_ref, p99_ref

        yr = p.index.year
        _annual_mean = lambda s: float(s.mean()) if len(s) else None  # noqa: E731

        out["prcptot"] = _annual_mean(p.where(p >= wet_thr, 0.0).groupby(yr).sum())
        out["r10mm"] = _annual_mean((p >= 10.0).groupby(yr).sum())
        out["r20mm"] = _annual_mean((p >= 20.0).groupby(yr).sum())
        out["rx1day"] = _annual_mean(p.groupby(yr).max())
        out["rx5day"] = _annual_mean(
            p.rolling(window=5, min_periods=5).sum().groupby(yr).max()
        )
        if p95_ref is not None:
            out["r95ptot"] = _annual_mean(p.where(p > p95_ref, 0.0).groupby(yr).sum())
        if p99_ref is not None:
            out["r99ptot"] = _annual_mean(p.where(p > p99_ref, 0.0).groupby(yr).sum())

    # consecutive-dry-day (CDD) metrics on the same series, merged in.
    # precip_dry_spell_* is the whole-record dry-spell summary; cdd_annual_max_*
    # is the ETCCDI-aligned mean annual-maximum CDD. Both kept for now so the
    # robust vs whole-record variants can be compared before choosing one.
    out.update(precip_dry_spell_stats(pcp, wet_thr=wet_thr))
    out.update(annual_max_cdd(pcp, wet_thr=wet_thr))
    return out


def flashiness_index(qsim: pd.Series) -> Optional[float]:
    """
    Richards-Baker Flashiness Index (RBI): the sum of absolute day-to-day
    flow changes divided by the sum of flows,

        RBI = sum_i |q_i - q_{i-1}| / sum_i q_i

    Dimensionless, so it is unit-agnostic (mm/day here). Returns None when
    there are fewer than 2 days or the total flow is non-positive.
    """
    q = qsim.dropna().to_numpy(dtype=float)
    if len(q) < 2:
        return None
    denom = q.sum()
    if denom <= 0:
        return None
    return float(np.abs(np.diff(q)).sum() / denom)


# square kilometres per square mile (MTI in Zhu et al. uses area in mi^2)
_KM2_PER_MI2 = 2.589988110336


def _flood_event_windows(s: np.ndarray, thd: float, mti: int) -> List[List[int]]:
    """
    Identify independent flood episodes as [start, end] index windows.

    Episodes are runs of days where the (smoothed) streamflow ``s`` is at or
    above the flood threshold ``thd``. Two adjacent episodes are merged into
    one event when they are not independent, following the inter-event
    criterion of Zhu et al.:
      * their above-threshold runs are separated by <= ``mti`` days, OR
      * the recession minimum between their peaks exceeds 75% of the smaller
        of the two peaks (the "Q-check").
    """
    n = len(s)
    above = s >= thd

    episodes: List[List[int]] = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            episodes.append([i, j])
            i = j + 1
        else:
            i += 1

    if not episodes:
        return []

    merged = [episodes[0]]
    for first, end in episodes[1:]:
        pfirst, pend = merged[-1]
        gap = first - pend
        peak_prev = s[pfirst:pend + 1].max()
        peak_cur = s[first:end + 1].max()
        trough = s[pend:first + 1].min()
        if gap <= mti or trough > 0.75 * min(peak_prev, peak_cur):
            merged[-1][1] = end          # not independent -> merge
        else:
            merged.append([first, end])
    return merged


def flashiness_stats(
        qsim: pd.Series,
        area_km2: float,
        K: float = 0.5,
        smooth_window: int = 3,
) -> Dict:
    """
    Event-based hydrograph flashiness after Zhu et al. (2026, Commun. Earth
    Environ., "Widespread intensification of global river hydrograph
    flashiness under climate change").

    For each independent flood event the flashiness is the flood rising
    slope (Eqs. 5-10):

        Flashiness = (1/A) * (S_peak - S_rp) / (T_peak - T_rp)

    where (S_rp, T_rp) is the rising point and (S_peak, T_peak) the peak.
    The paper's ``S`` is discharge (scales with area), hence the explicit
    ``1/A``. Here ``qsim`` is already an area-normalised depth (mm/day), so
    the area term cancels and flashiness reduces to SFD/FRT in mm/day**2;
    area is still used for the MTI event-independence term (Eq. 2). The
    returned ``flashiness_event`` is the mean over all detected events, plus
    the mean rising time (FRT, days), mean streamflow difference (SFD,
    mm/day) and the number of events (to allow the SFD/FRT decomposition of
    Eqs. 11-13 downstream).

    Pipeline: 3-day moving-average smoothing -> median (P50) flood threshold
    (Eq. 1) -> above-threshold episodes -> MTI merge for event independence
    -> per-event FPT buffer (Eqs. 3-4) -> slope-based rising point via the
    modified difference D_i = S_i - K*(S_{i+1}-S_i) (Eqs. 6-7), K=0.5.
    """
    empty = {"flashiness_event": None, "flashiness_event_frt": None,
             "flashiness_event_sfd": None, "n_flood_events": 0}

    q = qsim.dropna()
    if len(q) < smooth_window + 2 or not area_km2 or area_km2 <= 0:
        return empty

    raw = q.to_numpy(dtype=float)
    # 3-day centred moving average to suppress spurious threshold crossings
    s = pd.Series(raw).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()

    thd = float(np.median(raw))            # Eq. 1: P50 of daily streamflow

    # MTI (Eq. 2): 5 + ln(A), A in square miles
    area_mi2 = area_km2 / _KM2_PER_MI2
    mti = int(max(round(5.0 + np.log(area_mi2)), 1)) if area_mi2 > 0 else 5

    n = len(s)
    events = _flood_event_windows(s, thd, mti)

    fls_list, sfd_list, frt_list = [], [], []
    for first, end in events:
        # Flood Potential Time window: extend the episode by MTI either side
        # so the rising/recession limbs sit inside the window (Eqs. 3-4).
        w0 = max(first - mti, 0)
        w1 = min(end + mti, n - 1)
        seg = s[w0:w1 + 1]
        if len(seg) < 3:
            continue

        tpeak = int(np.argmax(seg))        # Eq. 5: time of peak in window
        if tpeak == 0:
            continue

        # modified streamflow difference D_i = S_i - K*(S_{i+1}-S_i) (Eq. 6)
        D = seg.copy()
        D[:-1] = seg[:-1] - K * np.diff(seg)

        # RPT (Eq. 7): last occurrence of min(D) before the peak
        pre = D[:tpeak + 1]
        rpt = int(np.where(pre == pre.min())[0][-1])

        frt = tpeak - rpt                  # Eq. 9 (days)
        sfd = seg[tpeak] - seg[rpt]        # peak-to-rising-point difference
        if frt <= 0 or sfd <= 0:
            continue

        fls_list.append(sfd / frt)         # Eq. 10 (area cancels, see docstring)
        sfd_list.append(sfd)
        frt_list.append(frt)

    if not fls_list:
        return empty

    return {
        "flashiness_event": float(np.mean(fls_list)),
        "flashiness_event_frt": float(np.mean(frt_list)),
        "flashiness_event_sfd": float(np.mean(sfd_list)),
        "n_flood_events": int(len(fls_list)),
    }


# ---------------------------------------------------------------------------
# Marginal streamflow signatures (flood return levels, FDC shape, high/low
# flow). All are functions of the flow *distribution* only (no day-to-day
# sequencing), so the delta/quantile-mapping method transfers them faithfully.
# ---------------------------------------------------------------------------
def _annual_block_maxima(qsim: pd.Series, min_days_per_year: int = 300) -> np.ndarray:
    """Calendar-year maxima, keeping only years with adequate day coverage."""
    q = qsim.dropna()
    if len(q) == 0:
        return np.array([], dtype=float)
    by_year = q.groupby(q.index.year)
    counts, maxima = by_year.count(), by_year.max()
    good = counts[counts >= min_days_per_year].index
    return maxima.loc[maxima.index.intersection(good)].to_numpy(dtype=float)


def _gev_lmom_fit(x: np.ndarray):
    """
    Fit a GEV distribution by L-moments (Hosking & Wallis 1997, "Regional
    Frequency Analysis", CUP, App. A.9; Hosking 1990, JRSS-B 52, 105-124).
    Returns (loc, scale, k) in Hosking's parameterisation, with quantile
    ``x(F) = loc + (scale/k)*(1 - (-ln F)**k)``; Hosking's ``k`` equals ``-xi``
    in the Coles (2001) convention. L-moments are preferred over MLE for the
    short (~20-year) annual-maxima samples here: far less shape bias at small n.
    Returns None if the sample is too small or the fit is degenerate.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 10:
        return None
    # probability-weighted moments b0, b1, b2 (unbiased estimators)
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
    if l2 <= 0:
        return None
    t3 = l3 / l2                                            # L-skewness
    c = 2.0 / (3.0 + t3) - math.log(2.0) / math.log(3.0)   # Hosking approx.
    k = 7.8590 * c + 2.9554 * c * c
    if abs(k) < 1e-6:                                       # Gumbel limit
        scale = l2 / math.log(2.0)
        return (l1 - 0.5772156649 * scale, scale, 0.0)
    gk = math.gamma(1.0 + k)
    scale = l2 * k / ((1.0 - 2.0 ** (-k)) * gk)
    loc = l1 - scale * (1.0 - gk) / k
    if not (np.isfinite(scale) and scale > 0 and np.isfinite(loc)):
        return None
    return (loc, scale, k)


def _gev_return_level(params, T: float) -> float:
    """Return level for return period T (annual maxima): quantile at 1 - 1/T."""
    loc, scale, k = params
    F = 1.0 - 1.0 / T
    if k == 0.0:
        return loc - scale * math.log(-math.log(F))
    return loc + (scale / k) * (1.0 - (-math.log(F)) ** k)


def flood_return_levels(
        qsim: pd.Series,
        return_periods=(20, 50),
        min_years: int = 15,
        min_days_per_year: int = 300,
) -> Dict:
    """
    Flood return levels Q_T (Q20/Q50) from a GEV fit to annual maxima of daily
    streamflow (block-maxima method). Q_T is the flow with non-exceedance
    probability 1 - 1/T. References: Coles (2001, "An Introduction to
    Statistical Modeling of Extreme Values", Springer, Ch. 3, Eq. 3.4); Katz,
    Parlange & Naveau (2002, Adv. Water Resour. 25, 1287-1304). Code refs:
    scipy.stats.genextreme; lmoments3 (https://github.com/OpenHydrology/lmoments3).
    Units follow qsim (mm/day here).

    CAVEAT: with ~20-year windows even Q50 is an extrapolation (~2.5x the
    record), so Q20 is the most defensible; Q100 is deliberately not computed
    (5x extrapolation). Reported as None when fewer than ``min_years``
    adequately-covered years exist or the fit fails.
    """
    keys = {int(T): f"flood_q{int(T)}" for T in return_periods}
    out = {v: None for v in keys.values()}
    maxima = _annual_block_maxima(qsim, min_days_per_year)
    if maxima.size < min_years or np.all(maxima <= 0) or np.std(maxima) == 0:
        return out
    params = _gev_lmom_fit(maxima)
    if params is None:
        return out
    for T in return_periods:
        try:
            rl = _gev_return_level(params, float(T))
            out[keys[int(T)]] = float(rl) if np.isfinite(rl) else None
        except (ValueError, OverflowError):
            out[keys[int(T)]] = None
    return out


def fdc_signatures(qsim: pd.Series) -> Dict:
    """
    Flow-duration-curve marginal-variability signatures.

    slope_fdc : slope of the FDC between the 33% and 66% exceedance flows in
        natural-log space, ``[ln Q_33 - ln Q_66] / (0.66 - 0.33)`` where Q_p is
        the flow exceeded p of the time (Q_33 > Q_66). Larger = steeper FDC =
        flashier/more variable regime. Yilmaz, Gupta & Wagener (2008, WRR 44,
        W09417); Sawicz et al. (2011, HESS 15, 2895-2911); CAMELS 'slope_fdc'
        (Addor et al. 2017, HESS 21, 5293-5313). Code refs:
        https://github.com/naddor/camels ; https://github.com/TOSSHtoolbox/TOSSH.

    q5_q95_ratio, q10_q90_ratio : ratios of a high-flow to a low-flow exceedance
        percentile -- marginal 'range/flashiness' surrogates that complement the
        temporal Richards-Baker index without the sequencing dependence (Olden &
        Poff 2003, River Res. Appl. 19, 101-121; Addor et al. 2018, WRR 54, 8792).

    Q_p (exceedance) = quantile(q, 1 - p). Returns None for a slope/ratio when
    the required percentile flow is <= 0 (log / division undefined).
    """
    q = qsim.dropna().to_numpy(dtype=float)
    out = {"slope_fdc": None, "q5_q95_ratio": None, "q10_q90_ratio": None}
    if q.size < 10:
        return out
    q05e = np.quantile(q, 0.95)   # exceeded  5% of the time (high flow)
    q10e = np.quantile(q, 0.90)   # exceeded 10%
    q33e = np.quantile(q, 0.67)   # exceeded 33%
    q66e = np.quantile(q, 0.34)   # exceeded 66%
    q90e = np.quantile(q, 0.10)   # exceeded 90%
    q95e = np.quantile(q, 0.05)   # exceeded 95% of the time (low flow)
    if q33e > 0 and q66e > 0:
        out["slope_fdc"] = float((np.log(q33e) - np.log(q66e)) / (0.66 - 0.33))
    if q95e > 0:
        out["q5_q95_ratio"] = float(q05e / q95e)
    if q90e > 0:
        out["q10_q90_ratio"] = float(q10e / q90e)
    return out


def high_flow_frequency(qsim: pd.Series, threshold: Optional[float] = None) -> Dict:
    """
    High-flow frequency and duration. A high-flow day has Q > 9 x median daily
    flow (Clausen & Biggs 2000, J. Hydrol. 237, 184-197; Olden & Poff 2003;
    CAMELS 'high_q_freq', Addor et al. 2017, HESS 21, 5293-5313). Code refs:
    https://github.com/naddor/camels ; https://github.com/TOSSHtoolbox/TOSSH.
    Reports days per year (high_q_freq), mean high-flow spell length in days
    (high_q_dur) and the event count.

    ``threshold``: if given, this FIXED flow value is used instead of 9 x this
    series' median. For a baseline-vs-future climate signal pass the baseline's
    9 x median so both periods share one threshold (mirrors ``dry_threshold``);
    None makes the metric self-referential (CAMELS convention), which is what
    the baseline run itself uses.
    """
    q = qsim.dropna()
    out = {"high_q_freq": None, "high_q_dur": None,
           "n_high_q_events": None, "high_q_threshold": None}
    if len(q) == 0:
        return out
    thr = float(9.0 * q.median()) if threshold is None else float(threshold)
    n_years = q.index.year.nunique()
    above = q.to_numpy() > thr
    spells = _spell_lengths(above)
    out["high_q_threshold"] = thr
    out["high_q_freq"] = float(above.sum() / n_years) if n_years else None
    out["n_high_q_events"] = int(len(spells))
    out["high_q_dur"] = float(spells.mean()) if len(spells) else 0.0
    return out


def runoff_ratio(qsim: pd.Series, pcp: Optional[pd.Series]) -> Optional[float]:
    """
    Runoff ratio RR = sum(Q) / sum(P) over the common dates -- a water-balance
    signature (Sawicz et al. 2011, HESS 15, 2895-2911; CAMELS 'runoff_ratio',
    Addor et al. 2017). Q and P MUST share units (mm/day here) and are aligned
    on their common dates before summing. None if P is absent or total P <= 0.
    """
    if pcp is None:
        return None
    q, p = qsim.dropna(), pcp.dropna()
    idx = q.index.intersection(p.index)
    if len(idx) == 0:
        return None
    psum = float(p.loc[idx].sum())
    if psum <= 0:
        return None
    return float(q.loc[idx].sum() / psum)


def low_flow_signatures(
        qsim: pd.Series,
        min_years: int = 10,
        min_days_per_year: int = 300,
) -> Dict:
    """
    Low-flow signatures from the annual minimum 7-day mean flow (MAM7 family).
    Smakhtin (2001, J. Hydrol. 240, 147-186, low-flow review). Code ref: xclim
    (https://github.com/Ouranosinc/xclim), 7-day low flow.

    mam7      : mean annual minimum 7-day mean flow (the '7-day low flow').
    q7min_min : lowest annual 7-day mean over the record.
    q7_10     : 7Q10 -- annual-minimum 7-day flow with a 10-year recurrence
                (non-exceedance probability 0.1 of the annual-minima
                distribution), via a log-normal fit (US-standard alternative:
                log-Pearson III). Defined only when all annual 7-day minima are
                > 0; None for intermittent (zero-low-flow) streams, while mam7
                (robust to zeros) is still reported.
    """
    out = {"mam7": None, "q7min_min": None, "q7_10": None}
    q = qsim.dropna()
    if len(q) == 0:
        return out
    q7 = q.rolling(window=7, min_periods=7).mean()
    ann_min = q7.groupby(q7.index.year).min()
    counts = q.groupby(q.index.year).count()
    good = counts[counts >= min_days_per_year].index
    ann_min = ann_min.loc[ann_min.index.intersection(good)].dropna()
    if len(ann_min) < min_years:
        return out
    am = ann_min.to_numpy(dtype=float)
    out["mam7"], out["q7min_min"] = float(am.mean()), float(am.min())
    if np.all(am > 0):                                      # log-normal 7Q10
        logs = np.log(am)
        sd = logs.std(ddof=1)
        out["q7_10"] = (float(np.exp(logs.mean() - 1.2815515594 * sd))  # z_0.1
                        if sd > 0 and np.isfinite(sd) else float(am.mean()))
    return out


def low_flow_freq_deficit(qsim: pd.Series, threshold: Optional[float]) -> Dict:
    """
    Marginal low-flow frequency / severity below a FIXED threshold (the
    engine's baseline q05 -- the same value and the same strict
    ``Q < threshold`` comparison as the dry-spell family). Order-independent,
    so it does not inherit the day-to-day sequencing caveat of the delta
    method.

    low_q_freq    : days per year with Q < threshold; order-independent
        counterpart of the dry-spell run lengths (convention after CAMELS
        'low_q_freq', Addor et al. 2017, HESS 21, 5293-5313; Olden & Poff
        2003, River Res. Appl. 19, 101-121).
    low_q_deficit : mean annual deficit volume below the threshold,
        sum(threshold - Q | Q < threshold) / n_years, mm yr-1
        (threshold-level drought method: Yevjevich 1967; Hisdal et al. 2004;
        Fleig et al. 2006, HESS 10, 535-552; Van Loon 2015, WIREs Water 2).

    Both None when ``threshold`` is None/non-finite or the series is empty.
    """
    out = {"low_q_freq": None, "low_q_deficit": None}
    if threshold is None or not np.isfinite(threshold):
        return out
    q = qsim.dropna()
    if len(q) == 0:
        return out
    n_years = q.index.year.nunique()
    if not n_years:
        return out
    v = q.to_numpy(dtype=float)
    thr = float(threshold)
    below = v < thr
    out["low_q_freq"] = float(below.sum() / n_years)
    out["low_q_deficit"] = float((thr - v[below]).sum() / n_years)
    return out


def half_flow_date(qsim: pd.Series, lat: Optional[float],
                   min_years: int = 10, min_days_per_year: int = 300):
    """
    Mean half-flow date: day of the water year on which cumulative daily flow
    first reaches half of that water year's total, averaged over water years
    with at least ``min_days_per_year`` covered days (Court 1962, J. Geophys.
    Res. 67; CAMELS 'hfd_mean', Addor et al. 2017, HESS 21, 5293-5313).
    Water year starts 1 October for lat >= 0 and 1 April for lat < 0,
    following the hemisphere-adapted CAMELS variants (CAMELS-CL,
    Alvarez-Garreton et al. 2018, HESS 22; CAMELS-AUS, Fowler et al. 2021,
    ESSD 13).

    Because the deltas are applied per calendar month, monthly flow
    redistribution is a transferred climate signal, so this seasonal-timing
    index stays within the delta method's fidelity (unlike the day-to-day
    persistence indices). None when ``lat`` is unknown or fewer than
    ``min_years`` adequately covered water years exist.
    """
    if lat is None or not np.isfinite(lat):
        return None
    q = qsim.dropna()
    if len(q) == 0:
        return None
    start_month = 10 if lat >= 0 else 4
    # water-year label = calendar year in which the water year ENDS
    wy = np.where(q.index.month >= start_month, q.index.year + 1,
                  q.index.year)
    vals = q.to_numpy(dtype=float)
    hfds = []
    for y in np.unique(wy):
        sel = wy == y
        if int(sel.sum()) < min_days_per_year:
            continue
        v = vals[sel]
        tot = v.sum()
        if tot <= 0:
            continue
        i = int(np.searchsorted(np.cumsum(v), 0.5 * tot))
        wy_start = pd.Timestamp(int(y) - 1, start_month, 1)
        hfds.append((q.index[sel][i] - wy_start).days + 1)
    if len(hfds) < min_years:
        return None
    return float(np.mean(hfds))


def compute_flow_stats(
        qsim: pd.Series,
        dry_threshold: Optional[float] = None,
        area_km2: Optional[float] = None,
        precip: Optional[pd.Series] = None,
        high_flow_threshold: Optional[float] = None,
        lat: Optional[float] = None,
) -> Dict:
    """
    Compute statistics from simulated streamflow.
    Assumes qsim is daily and indexed by datetime.

    Always reported: basic moments/quantiles, mean annual max/min, monthly
    means, the Richards-Baker flashiness index, and the marginal
    distributional signatures -- flood return levels Q10/Q20/Q50
    (``flood_return_levels``), FDC slope + percentile ratios
    (``fdc_signatures``), the 7-day low-flow / 7Q10 family
    (``low_flow_signatures``), and the mean half-flow date
    (``half_flow_date``; needs ``lat`` for the water-year start, else
    ``hfd_mean`` is None). High-flow frequency/duration
    (``high_flow_frequency``) is always computed; pass ``high_flow_threshold``
    (the baseline's 9 x median) on future runs so both periods share one
    threshold, else each period uses its own 9 x median.

    ``precip`` (mm/day, same window/units as qsim) enables the runoff ratio
    Q/P; when None ``runoff_ratio`` is None.

    ``dry_threshold``: when given, dry-spell statistics AND the marginal
    low-flow frequency/severity pair (``low_flow_freq_deficit``:
    low_q_freq, low_q_deficit) are added using that fixed flow value
    (typically the baseline q05); when None those keys are omitted (e.g. the
    baseline run, before its own q05 is known -- the caller then patches the
    baseline stats afterwards, as apply_deltas_stn.py does).

    ``flashiness_event`` and its components (event-based Zhu et al. hydrograph
    flashiness) are added only when ``area_km2`` is given.
    """
    qsim = qsim.dropna().copy()

    if len(qsim) == 0:
        stats = {
            "n_days": 0,
            "qmean": None,
            "qstd": None,
            "q05": None,
            "q50": None,
            "q95": None,
            "q99": None,
            "mean_annual_max": None,
            "mean_annual_min": None,
            "flashiness_rb": None,
            "monthly_mean_qsim": {}
        }
    else:
        annual_max = qsim.groupby(qsim.index.year).max()
        annual_min = qsim.groupby(qsim.index.year).min()
        monthly_mean = qsim.groupby(qsim.index.month).mean()

        stats = {
            "n_days": int(len(qsim)),
            "qmean": float(qsim.mean()),
            "qstd": float(qsim.std()),
            "q05": float(qsim.quantile(0.05)),
            "q50": float(qsim.quantile(0.50)),
            "q95": float(qsim.quantile(0.95)),
            "q99": float(qsim.quantile(0.99)),
            "mean_annual_max": float(annual_max.mean()) if len(annual_max) > 0 else None,
            "mean_annual_min": float(annual_min.mean()) if len(annual_min) > 0 else None,
            "flashiness_rb": flashiness_index(qsim),
            "monthly_mean_qsim": {
                f"{m:02d}": float(v) for m, v in monthly_mean.items()
            }
        }

    # marginal distributional signatures (each returns a None-filled dict on an
    # empty series, so the stats schema is stable across all runs).
    stats.update(flood_return_levels(qsim, return_periods=(10, 20, 50)))
    stats.update(fdc_signatures(qsim))
    stats.update(low_flow_signatures(qsim))
    stats.update(high_flow_frequency(qsim, threshold=high_flow_threshold))
    stats["runoff_ratio"] = runoff_ratio(qsim, precip)
    stats["hfd_mean"] = half_flow_date(qsim, lat)

    if dry_threshold is not None:
        if len(qsim) > 0:
            stats.update(dry_spell_stats(qsim, dry_threshold))
        else:
            stats.update({"dry_spell_mean": None, "dry_spell_max": None, "n_dry_spells": None})
        # marginal low-flow frequency/severity below the SAME fixed threshold
        # (handles empty/degenerate series itself with None values)
        stats.update(low_flow_freq_deficit(qsim, dry_threshold))

    if area_km2 is not None:
        stats.update(flashiness_stats(qsim, area_km2))

    return stats


def run_hbv_and_save(
    data: pd.DataFrame,
    parameters: Dict[str, float],
    warmup_days: int = 365,
    baseline_stats: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    dry_threshold: Optional[float] = None,
    area_km2: Optional[float] = None,
    high_flow_threshold: Optional[float] = None,
    lat: Optional[float] = None,
) -> Tuple[pd.Series, Dict]:
    """
    Runs HBV using data['pcp'], data['temp'], data['pet'] and saves:
    - <key>_data.csv
    - <key>_metrics.json
    """

    # Run HBV
    qsim = hbv_nb(
        data['pcp'],
        data['temp'],
        data['pet'],
        parameters=parameters,
        initialize=True
    )

    if len(qsim) != len(data):
        raise ValueError(f"HBV output length {len(qsim)} != input length {len(data)}")

    out = data.copy()
    out.insert(0, "qsim", np.asarray(qsim).reshape(-1), True)

    # Apply warmup after simulation
    out = out.iloc[warmup_days:].copy()

    # Stats from simulated q; out["pcp"] is post-warmup, aligned with qsim, so
    # the runoff ratio is computed over the same window as the flow stats.
    stats = compute_flow_stats(
        out["qsim"],
        dry_threshold=dry_threshold,
        area_km2=area_km2,
        precip=out["pcp"],
        high_flow_threshold=high_flow_threshold,
        lat=lat,
    )

    if baseline_stats is not None:
        stats = add_change_vs_baseline(stats, baseline_stats)

    if metadata is not None:
        stats["metadata"] = metadata

    # with open(os.path.join(out_dir, f"{key}_metrics.json"), "w") as f:
    #     json.dump(stats, f, indent=4, sort_keys=True)

    return out['qsim'], stats


def read_obs_q(
        stn:str,
        stations_path:Union[str, os.PathLike],
        area:float
        )->pd.Series:

    stn_fpath = os.path.join(stations_path, stn)

    q_file = os.path.join(stn_fpath, 'daily_q.csv')

    q_stn = pd.read_csv(
        q_file, index_col='date', na_values=["-999.0"],
        dtype={f"{stn.split('__')[1]}": np.float32},
        )
    try:
        q_stn.index = pd.to_datetime(q_stn.index, format='%Y-%m-%d')
    except ValueError:
        q_stn.index = pd.to_datetime(q_stn.index, format='%Y-%m-%d %H:%M:%S')
        # todo : following 3 lines should not have been required!
        q_stn.index = q_stn.index.normalize()
        q_stn = q_stn.sort_index()
        q_stn = q_stn.loc[~q_stn.index.duplicated(keep='first')]

    # for Portugal, the index is not sorted yet
    if stn.startswith('Portugal__'):
        q_stn = q_stn.sort_index()

    q_stn.columns = ['Qobs']

    assert len(q_stn) > 0

    # converting from cms to mm/day, considering area is in km2 and q is in cms
    q_stn = 24*3600*1000*q_stn / (area * 1000 * 1000)

    return q_stn['Qobs']


def load_shapefile(
        path:Union[str, os.PathLike],
        out_type:str = "geojson"
):
    assert os.path.exists(path), f"{path} does not exist"

    if out_type == "geojson":
        return load_shapefile_as_geojson(path)

    if out_type == "gpd_series":
        gdf = gpd.read_file(path)

    return gdf.geometry


def load_shapefile_as_geojson(
        path:Union[str, os.PathLike]
):
    """This function loads the shapefile and returns a geojson object
    i.e. either as Polygon or MultiPolygon depending on the number of rings in the shapefile.
    """
    r = shapefile.Reader(path)
    geoms = []

    for shp in r.shapes():
        g = shp.__geo_interface__  # warning can occur here
        if g["type"] == "Polygon":
            geoms.append(geojson.Polygon(g["coordinates"]))
        elif g["type"] == "MultiPolygon":
            geoms.append(geojson.MultiPolygon(g["coordinates"]))
        else:
            continue

    if not geoms:
        raise ValueError("No Polygon/MultiPolygon geometries found.")

    # If multiple features, return a MultiPolygon (flattening Polygons into it)
    if len(geoms) == 1:
        return geoms[0]

    mp_coords = []
    for geom in geoms:
        if isinstance(geom, geojson.Polygon):
            mp_coords.append(list(geom.coordinates))
        else:  # MultiPolygon
            mp_coords.extend(list(geom.coordinates))

    return geojson.MultiPolygon(mp_coords)


def _decode_mean_deltas_qqmap_day_ds(ds: xr.Dataset) -> dict:
    # precompute sample indices per month (keeps rank order)
    sample_month = ds["sample_month"].values.astype(int)
    sample_rank = ds["sample_rank"].values if "sample_rank" in ds.coords else None

    idx_by_month = {}
    for m in range(1, 13):
        idx = np.where(sample_month == m)[0]
        if idx.size == 0:
            continue
        if sample_rank is not None:
            idx = idx[np.argsort(sample_rank[idx])]
        idx_by_month[m] = idx

    out = {}
    for r in ds["realization"].values:
        r = str(r)
        out[r] = {}
        for per in ds["period"].values:
            per = str(per)

            out[r][f"{per}_pr_wetfreq"] = (
                ds["pr_wetfreq"].sel(realization=r, period=per).values.astype(np.float32)
            )

            for v in ["pr", "tmin", "tmax"]:
                vec = ds[v].sel(realization=r, period=per).values.astype(np.float32)  # (sample,)
                out[r][f"{per}_{v}"] = {m: vec[idx_by_month[m]].copy() for m in idx_by_month}

    return out


def load_mean_deltas(fpath: str, delta_method: str):
    if fpath.endswith(".nc"):
        ds = xr.open_dataset(fpath, engine="netcdf4")
        try:
            if delta_method in ("qqmap_day", "qqmap_day2"):
                return _decode_mean_deltas_qqmap_day_ds(ds)
            else:
                raise ValueError(f"load_mean_deltas: unsupported delta_method={delta_method} for .nc")
        finally:
            ds.close()
    raise ValueError("load_mean_deltas: only .nc supported here")
