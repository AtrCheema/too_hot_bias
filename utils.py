
import os
import csv
import glob
import time
import signal
import logging
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Union, List, Tuple, Optional, Sequence, Iterable

import cftime

import numpy as np

import pandas as pd

import xarray as xr
from xarray.coding.times import CFDatetimeCoder

from netCDF4 import Dataset

logger = logging.getLogger(__name__)


#===================================
#            Helper functions
#===================================

def load_historical_data(files:list, var:str, CONFIG)->xr.DataArray:
    
    if CONFIG.timestep == 'day':
        hist_da = load_historical_data_day(files, var, CONFIG)
    else:
        hist_da = load_historical_data_Amon(files, var, CONFIG)

    return hist_da


def _slice_str(dt: datetime) -> str:
    # Works for both numpy datetime and cftime-indexed xarray objects
    return dt.strftime("%Y-%m-%d")


def cf_bound(date, calendar_cls, *, end: bool = False, fallback=None):
    """Build a cftime datetime in a given CF calendar, clamping the day so
    that out-of-range day numbers (e.g. Dec 31 on a 360_day calendar) are
    quietly rounded down to the largest day the calendar accepts.

    This is needed because xarray's default string-based time slicing
    (``ds.sel(time=slice("2020-01-01", "2020-12-31"))``) tries to construct
    the calendar's own datetime class from the string, which raises for
    ``Datetime360Day(2020, 12, 31)`` (only 30 days/month) and similar
    mismatches. By building the bound explicitly with this helper, callers
    can use the same start/end date across datasets whose files use
    different CF calendars (``gregorian``, ``noleap``, ``360_day``, ...).

    Parameters
    ----------
    date : str or datetime-like
        Either a 'YYYY-MM-DD' string, or any object exposing integer
        ``year``, ``month`` and ``day`` attributes (``datetime.datetime``,
        ``pandas.Timestamp``, ``cftime.datetime`` subclasses, etc.). Only
        Y/M/D are read; any time-of-day on the input is ignored (the
        returned bound uses 00:00:00 or 23:59:59 depending on ``end``).
    calendar_cls : type
        The concrete cftime datetime class for the target calendar, e.g.
        ``cftime.Datetime360Day``, ``cftime.DatetimeNoLeap``,
        ``cftime.DatetimeGregorian``. Typically obtained from an
        xarray CFTimeIndex via ``ds.indexes['time'].date_type`` or from
        an existing cftime object via ``type(t)``.
    end : bool, keyword-only, default False
        If False (default) return the start-of-day instant (00:00:00),
        suitable as an inclusive *lower* slice bound. If True return the
        end-of-day instant (23:59:59), suitable as an inclusive *upper*
        slice bound so that e.g. a 2020-12-30 timestamp on a 360_day
        calendar is still included when the requested end is 2020-12-31.
    fallback : Any, optional
        Value returned if none of the tried day numbers (the original
        ``day`` followed by 30, 29, 28) is valid for the target calendar
        at this year/month. Callers typically pass the first/last entry of
        the dataset's time index so slicing simply clamps to the available
        range instead of failing. Defaults to None.

    Returns
    -------
    cftime.datetime or type(fallback)
        An instance of ``calendar_cls`` representing the requested bound
        (day-clamped as needed), or ``fallback`` if no valid day was
        found. The returned value is safe to pass directly into
        ``DataArray.sel(time=slice(lo, hi))`` alongside other bounds built
        against the same calendar.
    """
    if isinstance(date, str):
        y, m, d = map(int, date.split("-"))
    else:
        y, m, d = date.year, date.month, date.day
    for day in (d, 30, 29, 28):
        try:
            return calendar_cls(y, m, day, 23, 59, 59) if end else calendar_cls(y, m, day)
        except ValueError:
            continue
    return fallback


def load_temporal_data(
        files:List[str],
        var:str,
        start:datetime,
        end:datetime,
        )->xr.DataArray:
    """Get data for var in [start, end] across files; robust to cftime calendars (incl 360_day)."""
    das = []
    modelname = ""

    start_s = _slice_str(start)
    end_s = _slice_str(end)

    decode_times = CFDatetimeCoder(use_cftime=True)

    for file in files:
        filename = os.path.basename(file)
        modelname = filename.split('_')[1]
        membername = filename.split('_')[2]

        try:
            # FIX: keep cftime when present; avoids 360_day conversion crashes
            ds = xr.open_dataset(file, decode_times=decode_times)
        except (OSError, ValueError):
            logger.error(f'OS/Value Error opening {filename}, trying netCDF4 engine explicitly')
            try:
                ds = xr.open_dataset(file, engine='netcdf4', decode_times=decode_times)
            except (OSError, ValueError):
                logger.error(f'OS/Value Error opening {filename} with netCDF4 engine')
                try:
                    logger.error('Trying to open without decoding times')
                    ds = xr.open_dataset(file, engine='netcdf4', decode_times=False)
                except (OSError, ValueError):
                    logger.error(f'Still cannot open {filename}')
                    continue

        if "time" not in ds or len(ds.time) == 0:
            logger.warning(f'No time values found in {filename}')
            ds.close()
            continue

        if var not in ds:
            logger.error(f'{var} not found in {filename}')
            ds.close()
            continue

        # Drop duplicate time stamps robustly (works for cftime too)
        try:
            tvals = ds["time"].values
            dup = pd.Index(tvals).duplicated()
            if bool(np.any(dup)):
                ds = ds.isel(time=~dup)
                logger.warning(f'{int(np.sum(dup))} duplicate time values found in {filename}')
        except Exception:
            pass

        ds = ds.sortby("time")

        # Build calendar-aware bounds (handles 360_day, noleap, etc.);
        # fall back to string slicing for pandas DatetimeIndex.
        tindex = ds.indexes.get("time")
        if hasattr(tindex, "date_type"):          # CFTimeIndex
            cls = tindex.date_type
            t_lo = cf_bound(start, cls, end=False, fallback=tindex[0])
            t_hi = cf_bound(end, cls, end=True, fallback=tindex[-1])
            da_ = ds[var].sel(time=slice(t_lo, t_hi))
        else:                                      # pandas DatetimeIndex
            da_ = ds[var].sel(time=slice(start_s, end_s))

        if len(getattr(da_, "time", [])) < 1:
            logger.warning(f"No data after slicing {start_s}:{end_s} in {filename}")
            ds.close()
            continue

        if len(da_.shape) not in [3, 4]:
            logger.warning(f"Invalid shape {da_.shape} for {filename}")
            ds.close()
            continue

        das.append(da_)
        ds.close()

    if len(das) == 0:
        logger.warning(f'No data found for period ({start}-{end}) for variable {var} from {len(files)} files')
        return xr.DataArray()

    if var == 'tos':
        if len(set([da.shape[1:] for da in das])) > 1:
            logger.warning(f'Variable shapes: Var: {var} Num Files: {len(files)} Model: {modelname} Ens: {membername} Duration: {start} {end}')
            return xr.DataArray()

    out = xr.concat(das, dim='time')
    return to_noleap_calendar(out)


def get_xy_coordname(datarray:xr.DataArray)->Tuple[str, str]:
    # the name of lat/long dimension in the netcdf file
    x,y = None, None

    if 'lat' in datarray.coords:
        x = 'lat'
    elif 'latitude' in datarray.coords:
        x = 'latitude'
    elif 'nav_lat' in datarray.coords:
        x = 'nav_lat'

    if 'lon' in datarray.coords:
        y = 'lon'
    elif 'longitude' in datarray.coords:
        y = 'longitude'
    elif 'nav_lon' in datarray.coords:
        y = 'nav_lon'
    
    if x is None or y is None:
        raise ValueError(f'Cannot find lat/lon coords in the dataarray {datarray.coords._names}')

    return x, y


def get_cpus()->int:
    if os.name == "nt":
        return os.cpu_count()
    else:
        return len(os.sched_getaffinity(0))


def to_minus180_180_coords(obj):
    """Make lon in [-180, 180) for Dataset or DataArray."""
    if 'lon' in obj.coords:
        lon = obj['lon']
        # detect 0..360
        if float(lon.min()) >= 0 and float(lon.max()) > 180:
            new_lon = ((lon + 180) % 360) - 180
            obj = obj.assign_coords(lon=new_lon).sortby('lon')
    return obj


def to_noleap_calendar(da: xr.DataArray) -> xr.DataArray:
    """Convert a DataArray's time axis to the 'noleap' (365_day) calendar.

    Harmonizes per-month day counts across CMIP6 models so that rank-indexed
    (qqmap_day) deltas share a common sample axis.

    - noleap / 365_day sources: returned unchanged (no-op).
    - standard / gregorian / proleptic_gregorian / julian / all_leap:
      extra days (e.g. Feb 29) are dropped by xarray's convert_calendar.
    - 360_day sources: stretched to 365 days via align_on='year'; the 5
      inserted NaN days per year are filled by linear interpolation along
      the time axis, with a final ffill/bfill to cover boundary gaps.
    """
    tindex = da.indexes.get("time")
    if tindex is None:
        return da

    cal = getattr(tindex, "calendar", None)
    if cal is None and hasattr(tindex, "date_type"):
        name = tindex.date_type.__name__.lower()
        if "noleap" in name or "365" in name:
            cal = "noleap"
        elif "360" in name:
            cal = "360_day"
        else:
            cal = "standard"
    if cal is None:
        cal = "standard"  # pandas DatetimeIndex

    if cal in ("noleap", "365_day"):
        return da

    if cal == "360_day":
        out = da.convert_calendar("noleap", align_on="year", missing=np.nan)
        out = out.interpolate_na(dim="time", method="linear", use_coordinate=False)
        return out.ffill(dim="time").bfill(dim="time")

    return da.convert_calendar("noleap")


def create_skip_file(
        stn_dir:Union[str, os.PathLike],
        jobid,
        fname:str = 'skip',
        verbosity:int=1,
        ):
    fpath = os.path.join(stn_dir, fname)
    if not os.path.exists(fpath):
        with open(fpath, 'w') as f:
            f.write("done!")
        
        if verbosity:
            stn = os.path.basename(stn_dir)
            logger.info(f"{stn} is being undetaken by {jobid} ****")
    return


def remove_skip_files(path:Union[str, os.PathLike]) -> int:
    stns_found = 0
    all_stations = os.listdir(path)
    stns_found = 0
    for _, stn in enumerate(all_stations):

        stn_dir = os.path.join(path, stn)

        if os.path.isdir(stn_dir) and len(os.listdir(stn_dir)) > 0:

            fpath = os.path.join(stn_dir, 'skip')
            
            if os.path.exists(fpath):

                stns_found +=1
                # remove the file
                os.remove(fpath)
                #break
    logger.info(f"{stns_found} skip files found!")
    return stns_found


HISTORICAL_EXPERIMENTS = {"historical", "hist-1950", "highresSST-present"}


def is_historical_exp(exp: str) -> bool:
    """True for CMIP DECK 'historical' and the HighResMIP historical analogs
    ('hist-1950' coupled and 'highresSST-present' SST-forced). Used to route
    these experiments through the historical-period slicing branch instead of
    the future-scenario branch.
    """
    return exp in HISTORICAL_EXPERIMENTS


def get_filenames(exp, model, member, var, grid, CONFIG)->List[str]:

    if CONFIG.timestep == 'day':
        if is_historical_exp(exp):
            fnames = [f"{exp}_{model}_{member}_{var}_{CONFIG.timestep}_{grid}_{CONFIG.historical_period[0].replace('-', '')}_{CONFIG.historical_period[1].replace('-', '')}.nc"]
        else:
            fnames = []
            for future_period in CONFIG.future_periods:
                period_start, period_end = future_period
                period_fname = f"{exp}_{model}_{member}_{var}_{CONFIG.timestep}_{grid}_{period_start.replace('-', '')}_{period_end.replace('-', '')}.nc"
                fnames.append(period_fname)
    else:
        fnames = [f"{exp}_{model}_{member}_{var}_{CONFIG.timestep}_{grid}.nc"]
    return fnames


def get_fpaths(exp:str, model:str, member:str, var:str, grid:str, CONFIG)->List[str]:

    files = get_filenames(exp, model, member, var, grid, CONFIG)
    fpaths = [os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f) for f in files]
    return fpaths


def all_files_exist(exp:str, model:str, member:str, var:str, grid:str, CONFIG)->bool:

    fpaths = get_fpaths(exp, model, member, var, grid, CONFIG)
    for fpath in fpaths:
        if not os.path.exists(fpath):
            return False
    return True


def load_historical_data_Amon(
        files:list, 
        var:str,
        CONFIG,
        ) -> xr.DataArray:
    assert len(files) == 1
    # loop over each file, find if the file is in the historical period (2001-2020), if yes then load the data from the file
    # if the file is not in the historical period, skip it
    date_st = datetime(int(CONFIG.historical_period[0].split('-')[0]), int(CONFIG.historical_period[0].split('-')[1]), 1)
    date_en = datetime(int(CONFIG.historical_period[1].split('-')[0]), int(CONFIG.historical_period[1].split('-')[1]), 31)
    da = load_temporal_data(files, var, date_st, date_en)

    if len(da.shape)==0:
        logger.warning(f'No historical data found for {var}')
        return da

    end = da.time.to_pandas().max()
    end = datetime(end.year, end.month, end.day)

    if end < datetime(2015, 1, 1):
        # we need to get data from SSP5-8.5 scenario
        file = os.path.basename(files[0])
        model = file.split('_')[1]
        member = file.split('_')[2]
        hist_grid = file.split('_')[5].split('.')[0]
        ssp585_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f'ssp585_{model}_{member}_{var}_Amon_{hist_grid}.nc'))
        
        if len(ssp585_files)> 0:
            da_ssp585 = load_temporal_data(ssp585_files, var, datetime(2015,1,1), datetime(2020,12,31))
            if len(da_ssp585.shape)==0:
                logger.warning(f'No SSP585 data found for {model} {member} {var}')
            else:
                da = xr.concat([da, da_ssp585], dim='time')
        else:
            logging.warning(f'No SSP585 data found for {model} {member} {var}')
            #NO_2015_2020_DATA[file] = True

    return da, True


def load_historical_data_day(
        day_hist_files: List[str], 
        var: str,
        CONFIG,
        ) -> Tuple[xr.DataArray, bool]:
    """
    Daily-only historical loader.

    Expects daily files with names like:
      historical_{model}_{member}_{var}_day_YYYYMMDD_YYYYMMDD.nc
    (i.e., period-chunked day files)

    Loads CONFIG.historical_period and, if the loaded historical data ends before
    2015-01-01, appends 2015-2020 from ssp585 daily files for the same model/member.
    """
    if len(day_hist_files) == 0:
        logger.warning(f"No daily historical files provided for {var}")
        return xr.DataArray(), False

    def _parse_date(s: str, *, is_end: bool) -> datetime:
        """
        Accept 'YYYY-MM-DD' or 'YYYY-MM' or 'YYYY'.
        For daily workflow, prefer 'YYYY-MM-DD', but support others to be safe.
        """
        parts = s.split("-")
        if len(parts) == 3:
            y, m, d = map(int, parts)
            return datetime(y, m, d)
        if len(parts) == 2:
            y, m = map(int, parts)
            d = int(pd.Period(f"{y}-{m:02d}").days_in_month) if is_end else 1
            return datetime(y, m, d)
        if len(parts) == 1:
            y = int(parts[0])
            return datetime(y, 12, 31) if is_end else datetime(y, 1, 1)
        raise ValueError(f"Unrecognized date format: {s!r}")

    day_hist_files = sorted(day_hist_files)

    # Infer model/member from filename: {scenario}_{model}_{member}_...
    file0 = os.path.basename(day_hist_files[0])
    model = file0.split("_")[1]
    member = file0.split("_")[2]

    hist_start = _parse_date(CONFIG.historical_period[0], is_end=False)
    hist_end = _parse_date(CONFIG.historical_period[1], is_end=True)

    hist_da = load_temporal_data(day_hist_files, var, hist_start, hist_end)
    if len(getattr(hist_da, "shape", ())) == 0:
        logger.warning(f"No daily historical data found for {var} ({model}, {member})")
        return hist_da, False

    # If historical ends before 2015, append daily ssp585 for 2015-2020
    try:
        # ensure sorted, then take the last time value without .to_pandas()
        hist_da = hist_da.sortby("time")

        # works for both pandas/np datetime and cftime calendars
        if "time" in getattr(hist_da, "indexes", {}):
            last_time = hist_da.indexes["time"][-1]   # pandas.Timestamp OR cftime.datetime
        else:
            last_time = hist_da["time"].values[-1]
            if hasattr(last_time, "item"):  # unwrap 0-d numpy scalar
                last_time = last_time.item()

        # convert to a python datetime-like for your comparison
        # (for cftime this just builds a python datetime with same Y/M/D)
        last_dt = datetime(int(last_time.year), int(last_time.month), int(last_time.day))

    except Exception:
        logger.warning(
            f"Could not determine last timestamp in daily historical for {var} ({model}, {member})"
        )
        return hist_da, False

    # Compare against the requested end date (hist_end) rather than a hard
    # 2015-01-01 boundary, so models whose historical files end somewhere
    # between 2015-01-01 and hist_end (e.g. CESM2 spilling a day into 2015,
    # FGOALS-g3 ending 2016-12-31) also get the ssp585 splice instead of a
    # silently-truncated record. A ~1-month tolerance absorbs calendar
    # artefacts (e.g. 360_day files ending Dec 30 of the final year).
    gap_tol = timedelta(days=30)
    if last_dt < hist_end - gap_tol:
        # Start from later of ssp585's start (2015-01-01) and the historical's
        # last day; the dedup step below removes any one-day overlap.
        gap_start = max(datetime(2015, 1, 1), last_dt)
        gap_end = hist_end

        hist_grid = os.path.basename(day_hist_files[0]).split('_')[5].split('.')[0]
        ssp_files = sorted(
            glob.glob(
                os.path.join(
                    CONFIG.cmip6_data,
                    "day",
                    f"ssp585_{model}_{member}_{var}_day_{hist_grid}_20150101_20201231.nc",
                )
            )
        )

        if len(ssp_files) == 0:
            logger.warning(
                f"No daily ssp585 files found to fill gap "
                f"{gap_start:%Y-%m-%d}..{gap_end:%Y-%m-%d} for {var} ({model}, {member}); "
                f"historical ends {last_dt:%Y-%m-%d}"
            )
            return hist_da, False

        ext_da = load_temporal_data(ssp_files, var, gap_start, gap_end)
        if len(getattr(ext_da, "shape", ())) == 0:
            logger.warning(
                f"Daily ssp585 extension missing/empty for gap "
                f"{gap_start:%Y-%m-%d}..{gap_end:%Y-%m-%d} for {var} ({model}, {member})"
            )
            return hist_da, False

        logger.info(
            f"   Splicing ssp585 {gap_start:%Y-%m-%d}..{gap_end:%Y-%m-%d} onto historical "
            f"   ending {last_dt:%Y-%m-%d} for {var} ({model}, {member})"
        )

        hist_da = xr.concat([hist_da, ext_da], dim="time").sortby("time")

        # Drop any overlapping/duplicate days defensively
        try:
            if hist_da.time.to_pandas().index.duplicated().sum() > 0:
                hist_da = hist_da.drop_duplicates(dim="time")
        except Exception:
            pass

    return hist_da, True


def standardize_and_match_grids(
        hist_da: xr.DataArray, 
        fut_da: xr.DataArray
        ) -> Tuple[xr.DataArray, xr.DataArray, bool]:
    """
    Make lon consistent (-180..180 when possible), sort coords, and ensure fut is on hist grid.
    Returns (hist, fut, ok). If ok=False, caller should skip.
    """
    # normalize lon if named "lon" (common CMIP6); harmless otherwise
    hist_da = to_minus180_180_coords(hist_da)
    fut_da = to_minus180_180_coords(fut_da)

    xname, yname = get_xy_coordname(hist_da)

    # sort for stable alignment/interp
    try:
        hist_da = hist_da.sortby(xname).sortby(yname)
        fut_da = fut_da.sortby(xname).sortby(yname)
    except Exception:
        pass

    # If coords already identical, we're good
    try:
        if hist_da[xname].identical(fut_da[xname]) and hist_da[yname].identical(fut_da[yname]):
            return hist_da, fut_da, True
    except Exception:
        pass

    # Try strict align first (will raise if coords differ)
    try:
        hist2, fut2 = xr.align(hist_da, fut_da, join="exact")
        return hist2, fut2, True
    except Exception:
        pass

    # Last resort: interpolate future onto historical grid
    try:
        fut_i = fut_da.interp({xname: hist_da[xname], yname: hist_da[yname]}, method="linear")
        # align after interp to ensure coords now match
        hist2, fut2 = xr.align(hist_da, fut_i, join="exact")
        return hist2, fut2, True
    except Exception:
        return hist_da, fut_da, False


def is_delta_valid(delta_: xr.DataArray) -> Tuple[bool, int]:
    """
    Check whether delta_ contains any NaN/Inf.
    Returns (valid, bad_count). Works for numpy-backed and dask-backed
    xarray objects.
    """

    assert isinstance(delta_, xr.DataArray)

    # empty array (e.g. a 0-length lat/lon after grid alignment) is not a
    # usable delta; treat as invalid rather than letting the .min()/.max()
    # reduction below raise on size==0, which would abort the whole task.
    if delta_.size == 0:
        return False, "empty delta (size 0)"

    bad = ~np.isfinite(delta_)

    # Count bad values robustly (handles dask arrays)
    bad_sum = bad.sum()
    bad_count = int(bad_sum.compute().item()) if hasattr(bad_sum, "compute") else int(bad_sum.item())

    if bad_count:
        return False, bad_count

    # Layer 2: degenerate (constant) field — a real delta is never globally
    # constant. Caught EC-Earth3 r103/r104 ssp585 2081-2100 (pr intensity ≡ 1.0).
    # Uses min/max REDUCTIONS (no full materialisation of the multi-GB array).
    vmin, vmax = float(delta_.min()), float(delta_.max())
    if (vmax - vmin) < 1e-6:
        return False, f"degenerate constant delta (== {vmin:.3g} everywhere)"
    
    return True, 0


def calc_mon_dist_delta(
        variable: str, 
        hist_data: xr.DataArray, 
        future_data: xr.DataArray
        ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """calcualtes the monthly distributional delta between historical and future data for a given variable.
    For pr, returns (delta_intensity, delta_wetfreq). For tmin/tmax/tas, returns (delta, None).
    
    """
    quantiles = np.linspace(0.05, 0.95, 10, dtype=np.float32)  # 10 levels: 0.05..0.95

    if variable == 'pr':
        wet_thr = 1.0 / 86400.0   # 1 mm/day in kg m^-2 s^-1

        # (a) wet-day frequency per month (shape: month, lat, lon)
        hist_is_wet = (hist_data > wet_thr)
        fut_is_wet  = (future_data > wet_thr)
        hist_freq = hist_is_wet.groupby('time.month').mean('time')
        fut_freq  = fut_is_wet.groupby('time.month').mean('time')

        # frequency delta 
        diff = fut_freq - hist_freq
        freq_delta = diff.where(np.isfinite(diff), 0.0).clip(-1.0, 1.0)

        # (b) intensity delta on wet days only (shape: month, quantile, lat, lon)
        hist_wet = hist_data.where(hist_is_wet)
        fut_wet  = future_data.where(fut_is_wet)
        hist_q = hist_wet.groupby('time.month').quantile(quantiles, dim='time', skipna=True)
        fut_q  = fut_wet.groupby('time.month').quantile(quantiles, dim='time', skipna=True)

        delta_ = (fut_q / hist_q).transpose('month', 'quantile', ...)
        delta_ = delta_.where(np.isfinite(delta_), 1.0).clip(min=0.0, max=5.0)
    else:
        hist_monthly_q = hist_data.groupby('time.month').quantile(quantiles, dim='time', skipna=True)
        future_monthly_q = future_data.groupby('time.month').quantile(quantiles, dim='time', skipna=True)                        
        delta_ = (future_monthly_q - hist_monthly_q).transpose('month', 'quantile', ...)
    
    return delta_, freq_delta if variable == 'pr' else None


def calc_mon_mean_delta(
        variable: str, 
        hist_data: xr.DataArray, 
        future_data: xr.DataArray
        ) -> xr.DataArray:
    """Calculates the monthly mean delta between historical and future data for a given variable."""
    if variable == 'pr':
        hist_monthly = hist_data.groupby('time.month').mean()
        hist_monthly = hist_monthly.where(hist_monthly > 1e-10)
        delta_ = (future_data.groupby('time.month').mean() / hist_monthly)
    else:
        delta_ = future_data.groupby('time.month').mean() - hist_data.groupby('time.month').mean()
    return delta_


def quantile_mapping(
        variable: str,
        hist_data: xr.DataArray,
        future_data: xr.DataArray,
        ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """
    Per-month CDF (quantile) mapping deltas, following the "every sorted
    pair" prescription.

    For each calendar month m, historical (CMIP6 present) and future (CMIP6
    future) daily values are sorted along the time axis. The rank-matched
    delta is then computed:

        pr         : delta = future_sorted / hist_sorted      (multiplicative)
        tas/tmin/tmax : delta = future_sorted - hist_sorted   (additive)

    Sorted-rank blocks for months 1..12 are concatenated into a single
    ``sample`` axis, giving shape ``(N_total, lat, lon)`` where
    ``N_total`` is the number of daily time steps in ``hist_data``
    (e.g. ~20*365 for 20 years of daily data).

    Application: for each month m in the observation series (MSWEP /
    temperature obs), sort the month's observed values, pair them rank-by-rank
    with the N_m delta values for that month, multiply (pr) or add
    (temperature), then un-sort back to calendar order.

    Parameters
    ----------
    variable : {'pr', 'tas', 'tasmin', 'tasmax'}
    hist_data, future_data : xr.DataArray
        Daily CMIP6 arrays with a ``time`` coord and spatial dims. Grids
        must already match. If per-month sample counts differ between
        hist and future, a warning is emitted and future's sorted CDF is
        linearly interpolated onto the historical rank axis.

    Returns
    -------
    delta : xr.DataArray
        Shape ``(sample, lat, lon)``. Multiplicative factor for pr,
        additive offset for temperature. ``sample`` carries auxiliary
        coords ``month`` (1..12) and ``rank`` (rank-within-month,
        0..N_m-1). Attrs include ``method``, ``variable``, ``delta_kind``.
    freq_delta : xr.DataArray or None
        For ``variable == 'pr'``: additive change in wet-day fraction per
        month per cell, shape ``(month, lat, lon)``, clipped to [-1, 1].
        ``None`` for temperature variables. The caller is responsible for
        folding ``freq_delta`` into the final Dataset (mirrors the
        ``calc_mon_dist_delta`` contract).
    """
    # ---- 0. sanity check on inputs ---------------------------------------
    assert variable in ('pr', 'tas', 'tasmin', 'tasmax'), \
        f"variable must be one of pr/tas/tasmin/tasmax, got {variable!r}"

    # todo : set some threshold!
    # Warn early if the two arrays don't cover the same number of days.
    # Pairing by rank is only exact when the two samples have equal length;
    # otherwise we fall back to rank-axis interpolation (see step 3 below).
    if hist_data.sizes['time'] != future_data.sizes['time']:

        # if the difference between hist_data and future_data is > 365, return
        # empty DataArray, since this indicates a whole year is missing
        if abs(hist_data.sizes['time'] - future_data.sizes['time']) > 365:
            logger.warning(
                "quantile_mapping: hist_data and future_data differ by more than "
                f"365 time steps ({hist_data.sizes['time']} vs "
                f"{future_data.sizes['time']}); this likely indicates a whole year "
                "is missing from one of the datasets. Returning empty delta."
            )
            empty = xr.DataArray(
                np.full((0,) + hist_data.shape[1:], dtype=np.float32, fill_value=np.nan),
                dims=('sample',) + hist_data.dims[1:],
                coords={dim: hist_data.coords[dim] for dim in hist_data.dims[1:]},
                attrs={
                    'method': 'quantile_mapping',
                    'variable': variable,
                    'delta_kind': 'intensity' if variable == 'pr' else 'additive',
                    'note': 'empty delta due to large time length mismatch between hist and future data',
                }
            )
            return empty, None
        else:
            logger.warning(
                "quantile_mapping: hist_data and future_data have different "
                f"total time lengths ({hist_data.sizes['time']} vs "
                f"{future_data.sizes['time']}). Per-month sample counts that "
                "differ will be linearly interpolated onto the historical rank "
                "axis; verify this is acceptable for your bias-correction."
            )

    # Identify the spatial coord names (lat/lon, latitude/longitude, etc.)
    xname, yname = get_xy_coordname(hist_data)

    # ---- 1. split both arrays into per-calendar-month groups -------------
    # groupby('time.month') bins every daily timestamp into months 1..12 so
    # each group contains all Jan days across all years, all Feb days, etc.
    # We materialise the groups into dicts to keep the loop below readable.
    hist_grp = dict(iter(hist_data.groupby('time.month')))
    fut_grp  = dict(iter(future_data.groupby('time.month')))

    # Per-month delta blocks will be concatenated along a single 'sample'
    # axis. month_coord / rank_coord record which month + rank each row
    # along that axis belongs to, so the delta can be looked up by month
    # and rank at application time.
    delta_blocks = []
    month_coord = []
    rank_coord = []

    # ---- 2. iterate over the 12 calendar months --------------------------
    for m in range(1, 13):
        if m not in hist_grp or m not in fut_grp:
            continue
        h_m = hist_grp[m]
        f_m = fut_grp[m]

        # Sort along the time axis ascending. After sorting, index 0 holds
        # the lowest value of that month across all years, index -1 the
        # highest -- i.e. we're constructing empirical CDFs per grid cell.
        t_axis_h = h_m.get_axis_num('time')
        t_axis_f = f_m.get_axis_num('time')
        h_sorted = np.sort(h_m.values, axis=t_axis_h)
        f_sorted = np.sort(f_m.values, axis=t_axis_f)

        # Move the (now rank-) axis to position 0 so downstream concat and
        # indexing are straightforward regardless of the original layout.
        h_sorted = np.moveaxis(h_sorted, t_axis_h, 0)
        f_sorted = np.moveaxis(f_sorted, t_axis_f, 0)

        n_h = h_sorted.shape[0]
        n_f = f_sorted.shape[0]

        # ---- 3. align rank axes if sample counts differ ------------------
        # Hist and future should have the same number of days per month
        # (same span of years). If not, interpolate future's sorted curve
        # onto the same number of equally-spaced quantile positions as
        # hist so delta = future / hist is well-defined pairwise.
        if n_f != n_h:
            logger.warning(
                f"quantile_mapping: month={m} sample count mismatch "
                f"(hist={n_h}, future={n_f}); interpolating future's sorted "
                "CDF onto the historical rank axis."
            )
            x_src = np.linspace(0.0, 1.0, n_f, dtype=np.float32)
            x_tgt = np.linspace(0.0, 1.0, n_h, dtype=np.float32)
            flat = f_sorted.reshape(n_f, -1)
            interp = np.empty((n_h, flat.shape[1]), dtype=np.float32)
            for j in range(flat.shape[1]):
                interp[:, j] = np.interp(x_tgt, x_src, flat[:, j])
            f_sorted = interp.reshape((n_h,) + h_sorted.shape[1:])

        # ---- 4. rank-by-rank delta ---------------------------------------
        if variable == 'pr':
            # Multiplicative delta is the natural choice for precipitation
            # (keeps values non-negative, captures relative intensification).
            # Divisions where hist == 0 produce inf/nan; replace with 1.0
            # (no change) and clip extreme ratios so a single near-zero hist
            # day cannot produce an absurd factor.
            with np.errstate(divide='ignore', invalid='ignore'):
                d = f_sorted / h_sorted
            d = np.where(np.isfinite(d), d, 1.0).astype(np.float32)
            d = np.clip(d, 0.0, 10.0)
        else:
            # Additive delta for temperature: preserves units (K or degC)
            # and behaves symmetrically around zero.
            d = (f_sorted - h_sorted).astype(np.float32)

        # Record which month + rank each row in 'd' corresponds to.
        delta_blocks.append(d)
        month_coord.extend([m] * n_h)
        rank_coord.extend(range(n_h))

    if len(delta_blocks) == 0:
        raise ValueError("No overlapping months between hist_data and future_data")

    # ---- 5. stack the 12 monthly blocks along the sample axis ------------
    # Total length is sum of per-month N_m which equals N_total (e.g. ~7300
    # for 20 years of daily data).
    delta_arr = np.concatenate(delta_blocks, axis=0)

    delta_da = xr.DataArray(
        delta_arr,
        dims=('sample', xname, yname),
        coords={
            xname: hist_data[xname],
            yname: hist_data[yname],
            # Auxiliary coords on 'sample' so consumers can select by month
            # (delta.where(delta.sample_month == 1, drop=True)) and then
            # index by rank 0..N_m-1.
            # NOTE: named 'sample_month' (not 'month') because the sibling
            # wetfreq_delta variable has 'month' as a *dim* coord (values
            # 1..12); packing both into one Dataset and serialising to
            # netCDF would otherwise drop the sample-level aux coord on
            # load. 'rang' (not 'rank') to avoid shadowing xarray's
            # built-in rank() method on DataArrays.
            'sample_month': ('sample', np.asarray(month_coord, dtype=np.int16)),
            'rang':         ('sample', np.asarray(rank_coord, dtype=np.int32)),
        },
        name='delta',
        attrs={
            'method': 'quantile_mapping',
            'variable': variable,
            'delta_kind': 'multiplicative' if variable == 'pr' else 'additive',
        },
    )

    # ---- 6. extra wet-day-frequency delta for pr -------------------------
    # Why: daily precipitation has a large point mass at zero (dry days).
    # Pure multiplicative CDF matching can rescale the intensity of wet
    # days but cannot turn a dry day wet or vice versa (0 * anything = 0,
    # and anything / 0 is undefined). If the future climate changes the
    # *fraction* of wet days (e.g. more/fewer rainy days in January), the
    # intensity factors alone will not capture that shift. freq_delta
    # records the additive change in wet-day fraction per month so that
    # the downstream apply step can promote/demote days to match the
    # target wet-day count, independent of the intensity rescaling.
    if variable == 'pr':
        wet_thr = 1.0 / 86400.0  # 1 mm/day expressed in kg m^-2 s^-1
        delta_da.attrs['wet_thr'] = float(wet_thr)
        hist_freq = (hist_data > wet_thr).groupby('time.month').mean('time')
        fut_freq  = (future_data > wet_thr).groupby('time.month').mean('time')
        freq_diff = fut_freq - hist_freq
        # NaNs can appear over always-dry cells; treat as no change.
        freq_delta = freq_diff.where(np.isfinite(freq_diff), 0.0).clip(-1.0, 1.0)
        return delta_da, freq_delta

    # ---- 7. non-pr: return delta DataArray, and None for freq_delta ------
    return delta_da, None


def quantile_mapping1(
        variable: str,
        hist_data: xr.DataArray,
        future_data: xr.DataArray,
        wet_thr: float = 1.0 / 86400.0,
        ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """CORRECTED per-month qqmap_day delta — candidate replacement for
    `quantile_mapping` (the original is intentionally left UNTOUCHED).

    Two changes vs `quantile_mapping`:
      1. PRECIP dry-tail fix. The multiplicative intensity factor
         ``future_sorted / hist_sorted`` is kept ONLY at ranks where BOTH the
         historical and future value are WET (> wet_thr); every other rank
         (the ~2/3 near-zero dry tail) gets factor 1.0. Wet-day OCCURRENCE is
         still carried separately by ``freq_delta``, so the near-zero/near-zero
         ratios that blew up to the +10 clip in the original no longer arise.
      2. A seasonal-consistency QC flag. The fraction of cells whose 12-month
         hist and future climatologies are ANTI-correlated is logged; a large
         fraction is the fingerprint of a corrupt/inverted historical time axis
         (e.g. NESM3 r2) that NO quantile method can repair — it must be fixed
         at the source. Temperature deltas stay additive (unchanged formula).

    Same output structure & signature as `quantile_mapping`, so it is a drop-in
    for `calc_delta` and `apply_qqmap_day_delta_*`.
    """
    assert variable in ('pr', 'tas', 'tasmin', 'tasmax'), \
        f"variable must be one of pr/tas/tasmin/tasmax, got {variable!r}"

    if hist_data.sizes['time'] != future_data.sizes['time']:
        if abs(hist_data.sizes['time'] - future_data.sizes['time']) > 365:
            logger.warning("quantile_mapping1: hist/future differ by >365 steps; "
                           "returning empty delta.")
            empty = xr.DataArray(
                np.full((0,) + hist_data.shape[1:], dtype=np.float32, fill_value=np.nan),
                dims=('sample',) + hist_data.dims[1:],
                coords={d: hist_data.coords[d] for d in hist_data.dims[1:]},
                attrs={'method': 'quantile_mapping1', 'variable': variable})
            return empty, None
        logger.warning("quantile_mapping1: per-month counts differ; future's "
                       "sorted CDF will be interpolated onto the hist rank axis.")

    xname, yname = get_xy_coordname(hist_data)

    # ---- seasonal-consistency QC (per cell: corr of 12-month hist vs fut) ---
    frac_anti = float('nan')
    try:
        hc = hist_data.groupby('time.month').mean('time').transpose('month', ...)
        fc = future_data.groupby('time.month').mean('time').transpose('month', ...)
        hcv = np.asarray(hc.values, dtype=np.float64)
        fcv = np.asarray(fc.values, dtype=np.float64)
        hm = hcv - np.nanmean(hcv, axis=0, keepdims=True)
        fm = fcv - np.nanmean(fcv, axis=0, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = (np.nansum(hm * fm, axis=0)
                    / np.sqrt(np.nansum(hm ** 2, axis=0) * np.nansum(fm ** 2, axis=0)))
        frac_anti = float(np.nanmean(corr < 0))
        msg = (f"quantile_mapping1[{variable}]: seasonal-consistency QC — "
               f"{100 * frac_anti:.1f}% of cells have hist/future monthly cycles "
               f"ANTI-correlated (mean cell corr={np.nanmean(corr):.2f})")
        if frac_anti > 0.20:
            logger.warning(msg + "  >>> LIKELY CORRUPT/INVERTED HISTORICAL TIME "
                           "AXIS: the delta is unreliable, fix at source.")
        else:
            logger.info(msg)
    except Exception as e:
        logger.warning(f"quantile_mapping1: seasonal QC skipped ({e})")

    # ---- per-month rank-matched delta --------------------------------------
    hist_grp = dict(iter(hist_data.groupby('time.month')))
    fut_grp = dict(iter(future_data.groupby('time.month')))
    delta_blocks, month_coord, rank_coord = [], [], []
    for mth in range(1, 13):
        if mth not in hist_grp or mth not in fut_grp:
            continue
        h_m, f_m = hist_grp[mth], fut_grp[mth]
        t_h, t_f = h_m.get_axis_num('time'), f_m.get_axis_num('time')
        h_sorted = np.moveaxis(np.sort(h_m.values, axis=t_h), t_h, 0)
        f_sorted = np.moveaxis(np.sort(f_m.values, axis=t_f), t_f, 0)
        n_h, n_f = h_sorted.shape[0], f_sorted.shape[0]
        if n_f != n_h:
            x_src = np.linspace(0.0, 1.0, n_f, dtype=np.float32)
            x_tgt = np.linspace(0.0, 1.0, n_h, dtype=np.float32)
            flat = f_sorted.reshape(n_f, -1)
            interp = np.empty((n_h, flat.shape[1]), dtype=np.float32)
            for j in range(flat.shape[1]):
                interp[:, j] = np.interp(x_tgt, x_src, flat[:, j])
            f_sorted = interp.reshape((n_h,) + h_sorted.shape[1:])

        if variable == 'pr':
            # multiplicative intensity ONLY where BOTH are wet; else no change.
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = f_sorted / h_sorted
            both_wet = (h_sorted > wet_thr) & (f_sorted > wet_thr)
            d = np.where(both_wet & np.isfinite(ratio), ratio, 1.0).astype(np.float32)
            d = np.clip(d, 0.0, 10.0)
        else:
            d = (f_sorted - h_sorted).astype(np.float32)

        delta_blocks.append(d)
        month_coord.extend([mth] * n_h)
        rank_coord.extend(range(n_h))

    if not delta_blocks:
        raise ValueError("No overlapping months between hist_data and future_data")
    delta_arr = np.concatenate(delta_blocks, axis=0)
    delta_da = xr.DataArray(
        delta_arr, dims=('sample', xname, yname),
        coords={
            xname: hist_data[xname], yname: hist_data[yname],
            'sample_month': ('sample', np.asarray(month_coord, dtype=np.int16)),
            'rang': ('sample', np.asarray(rank_coord, dtype=np.int32))},
        name='delta',
        attrs={'method': 'quantile_mapping1', 'variable': variable,
               'delta_kind': 'multiplicative' if variable == 'pr' else 'additive',
               'seasonal_anticorr_frac': frac_anti})

    if variable == 'pr':
        delta_da.attrs['wet_thr'] = float(wet_thr)
        hist_freq = (hist_data > wet_thr).groupby('time.month').mean('time')
        fut_freq = (future_data > wet_thr).groupby('time.month').mean('time')
        freq_diff = fut_freq - hist_freq
        freq_delta = freq_diff.where(np.isfinite(freq_diff), 0.0).clip(-1.0, 1.0)
        return delta_da, freq_delta
    return delta_da, None


def quantile_mapping2(
        variable: str,
        hist_data: xr.DataArray,
        future_data: xr.DataArray,
        wet_thr: float = 1.0 / 86400.0,
        ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """Second corrected qqmap_day delta. Fixes BOTH known flaws while leaving
    `quantile_mapping` AND `quantile_mapping1` untouched:

      * the DRY-TAIL division blow-up of `quantile_mapping` (near-zero/near-zero
        ratios clipped to x10), and
      * the FREQUENCY/INTENSITY entanglement of `quantile_mapping1` (its
        both-wet gate, applied at the same full-array rank, paired hist's wet
        days against the UPPER part of the future wet distribution whenever the
        wet-day COUNT changed, biasing intensity by ~the frequency change).

    Method (precip): the multiplicative intensity factor is a true
    QUANTILE-TO-QUANTILE map on WET DAYS ONLY. For each cell, the q-th quantile
    of the historical wet-day distribution is paired with the q-th quantile of
    the FUTURE wet-day distribution (future's sorted wet days interpolated onto
    the historical wet-quantile positions). This is INDEPENDENT of how many wet
    days each side has, so a pure change in wet-day FREQUENCY leaves the
    intensity factor at exactly 1.0 (all occurrence change is carried by
    freq_delta — no double-count, no residual too-hot bias). Dry ranks get
    factor 1.0. Temperature is additive (identical to `quantile_mapping`).

    Output structure & signature are identical to `quantile_mapping`, so it is a
    drop-in for `calc_delta` and `apply_qqmap_day_delta_*`.
    """
    assert variable in ('pr', 'tas', 'tasmin', 'tasmax'), \
        f"variable must be one of pr/tas/tasmin/tasmax, got {variable!r}"

    if hist_data.sizes['time'] != future_data.sizes['time']:
        if abs(hist_data.sizes['time'] - future_data.sizes['time']) > 365:
            logger.warning("quantile_mapping2: hist/future differ by >365 steps; "
                           "returning empty delta.")
            empty = xr.DataArray(
                np.full((0,) + hist_data.shape[1:], dtype=np.float32, fill_value=np.nan),
                dims=('sample',) + hist_data.dims[1:],
                coords={d: hist_data.coords[d] for d in hist_data.dims[1:]},
                attrs={'method': 'quantile_mapping2', 'variable': variable})
            return empty, None
        logger.warning("quantile_mapping2: per-month counts differ; future's "
                       "sorted CDF is interpolated onto the hist rank axis first.")

    xname, yname = get_xy_coordname(hist_data)

    # ---- seasonal-consistency QC (same as quantile_mapping1) ----------------
    frac_anti = float('nan')
    try:
        hc = hist_data.groupby('time.month').mean('time').transpose('month', ...)
        fc = future_data.groupby('time.month').mean('time').transpose('month', ...)
        hcv = np.asarray(hc.values, dtype=np.float64)
        fcv = np.asarray(fc.values, dtype=np.float64)
        hm = hcv - np.nanmean(hcv, axis=0, keepdims=True)
        fm = fcv - np.nanmean(fcv, axis=0, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = (np.nansum(hm * fm, axis=0)
                    / np.sqrt(np.nansum(hm ** 2, axis=0) * np.nansum(fm ** 2, axis=0)))
        frac_anti = float(np.nanmean(corr < 0))
        if frac_anti > 0.20:
            logger.warning(f"quantile_mapping2[{variable}]: {100*frac_anti:.1f}% of "
                           "cells have ANTI-correlated hist/future seasonal cycles "
                           ">>> LIKELY INVERTED HISTORICAL TIME AXIS (fix at source).")
    except Exception as e:
        logger.warning(f"quantile_mapping2: seasonal QC skipped ({e})")

    hist_grp = dict(iter(hist_data.groupby('time.month')))
    fut_grp = dict(iter(future_data.groupby('time.month')))
    delta_blocks, month_coord, rank_coord = [], [], []
    for mth in range(1, 13):
        if mth not in hist_grp or mth not in fut_grp:
            continue
        h_m, f_m = hist_grp[mth], fut_grp[mth]
        t_h, t_f = h_m.get_axis_num('time'), f_m.get_axis_num('time')
        h_sorted = np.moveaxis(np.sort(h_m.values, axis=t_h), t_h, 0)
        f_sorted = np.moveaxis(np.sort(f_m.values, axis=t_f), t_f, 0)
        n_h, n_f = h_sorted.shape[0], f_sorted.shape[0]
        if n_f != n_h:                       # put future on the hist rank axis
            x_src = np.linspace(0.0, 1.0, n_f, dtype=np.float32)
            x_tgt = np.linspace(0.0, 1.0, n_h, dtype=np.float32)
            flat = f_sorted.reshape(n_f, -1)
            interp = np.empty((n_h, flat.shape[1]), dtype=np.float32)
            for j in range(flat.shape[1]):
                interp[:, j] = np.interp(x_tgt, x_src, flat[:, j])
            f_sorted = interp.reshape((n_h,) + h_sorted.shape[1:])
        N = n_h

        if variable == 'pr':
            # wet-quantile-to-wet-quantile multiplicative factor (per cell).
            Wh = (h_sorted > wet_thr).sum(axis=0)        # (...spatial) hist wet count
            Wf = (f_sorted > wet_thr).sum(axis=0)        # future wet count
            rank_idx = np.arange(N).reshape((N,) + (1,) * (h_sorted.ndim - 1))
            Wh_safe = np.where(Wh > 0, Wh, 1).astype(np.float64)
            Wf_safe = np.where(Wf > 0, Wf, 1).astype(np.float64)
            # hist wet-quantile of each rank (only meaningful at wet ranks)
            q_h = (rank_idx - (N - Wh) + 0.5) / Wh_safe
            # fractional FUTURE rank holding that same wet-quantile
            fr = (N - Wf) + (q_h * Wf_safe - 0.5)
            fr = np.clip(fr, 0.0, N - 1.0)
            lo = np.floor(fr).astype(np.int64)
            hi = np.minimum(lo + 1, N - 1)
            wgt = (fr - lo).astype(np.float32)
            f_lo = np.take_along_axis(f_sorted, lo, axis=0)
            f_hi = np.take_along_axis(f_sorted, hi, axis=0)
            fut_at_q = f_lo * (1.0 - wgt) + f_hi * wgt
            with np.errstate(divide='ignore', invalid='ignore'):
                d = fut_at_q / h_sorted
            wet_rank = (rank_idx >= (N - Wh)) & (h_sorted > wet_thr) & (Wf > 0)
            d = np.where(wet_rank & np.isfinite(d), d, 1.0).astype(np.float32)
            d = np.clip(d, 0.0, 10.0)         # safety; wet/wet ratios rarely bind
        else:
            d = (f_sorted - h_sorted).astype(np.float32)

        delta_blocks.append(d)
        month_coord.extend([mth] * N)
        rank_coord.extend(range(N))

    if not delta_blocks:
        raise ValueError("No overlapping months between hist_data and future_data")
    delta_arr = np.concatenate(delta_blocks, axis=0)
    delta_da = xr.DataArray(
        delta_arr, dims=('sample', xname, yname),
        coords={
            xname: hist_data[xname], yname: hist_data[yname],
            'sample_month': ('sample', np.asarray(month_coord, dtype=np.int16)),
            'rang': ('sample', np.asarray(rank_coord, dtype=np.int32))},
        name='delta',
        attrs={'method': 'quantile_mapping2', 'variable': variable,
               'delta_kind': 'multiplicative' if variable == 'pr' else 'additive',
               'seasonal_anticorr_frac': frac_anti})

    if variable == 'pr':
        delta_da.attrs['wet_thr'] = float(wet_thr)
        hist_freq = (hist_data > wet_thr).groupby('time.month').mean('time')
        fut_freq = (future_data > wet_thr).groupby('time.month').mean('time')
        freq_diff = fut_freq - hist_freq
        freq_delta = freq_diff.where(np.isfinite(freq_diff), 0.0).clip(-1.0, 1.0)
        return delta_da, freq_delta
    return delta_da, None


def calc_delta(
        variable: str,
        hist_data: xr.DataArray,
        future_data: xr.DataArray,
        delta_method: str,
):
    if delta_method == 'mon_mean':
        delta_ = calc_mon_mean_delta(variable, hist_data, future_data)
    elif delta_method == 'qqmap_bin_day':
        delta_, freq_delta = calc_mon_dist_delta(variable, hist_data, future_data)
    elif delta_method == 'qqmap_day':
        delta_, freq_delta = quantile_mapping(variable, hist_data, future_data)
    elif delta_method == 'qqmap_day2':
        delta_, freq_delta = quantile_mapping2(variable, hist_data, future_data)
    else:
        raise ValueError(f"Invalid delta method: {delta_method}")
    return delta_, freq_delta


def _pick_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    return None


def _time_to_datetime_index(time_values) -> pd.DatetimeIndex:
    """
    Convert time values to pandas DatetimeIndex.

    Raises if values are out of pandas range (common for non-standard calendars).
    """
    arr = np.asarray(time_values)

    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.DatetimeIndex(arr)

    if arr.size == 0:
        return pd.DatetimeIndex([])

    t0 = arr.flat[0]
    if hasattr(t0, "year") and hasattr(t0, "month"):
        # Build from components (no string parsing)
        ts = [pd.Timestamp(int(t.year), int(t.month), int(getattr(t, "day", 1))) for t in arr]
        return pd.DatetimeIndex(ts)

    return pd.DatetimeIndex(pd.to_datetime(arr, errors="raise", utc=False))


def _is_monthly_pandas(dti: pd.DatetimeIndex):
    dti = pd.DatetimeIndex(dti).sort_values()

    if dti.hasnans:
        return False, "time contains NaT"

    if len(dti) < 2:
        return False, "too few time points"

    if dti.has_duplicates:
        return False, "duplicate timestamps"

    if not dti.is_monotonic_increasing:
        return False, "time not increasing"

    month_index = dti.year * 12 + dti.month
    steps = np.diff(month_index)

    if np.all(steps == 1):
        return True, "monthly"

    uniq, cnt = np.unique(steps, return_counts=True)
    return False, f"not monthly; month-step diffs={dict(zip(uniq.tolist(), cnt.tolist()))}"


def _as_cftime_list(time_values):
    arr = np.asarray(time_values)
    if arr.size == 0:
        return []

    t0 = arr.flat[0]
    if isinstance(t0, cftime.datetime):
        return list(arr)

    # If decode_times produced numpy datetime64, convert to python datetimes (still comparable)
    if np.issubdtype(arr.dtype, np.datetime64):
        return [pd.Timestamp(t).to_pydatetime() for t in arr]

    # If it's some other object type, last-ditch: try list() and rely on attributes/comparability
    return list(arr)


def _time_key(t):
    # Works for cftime + python datetime-like objects
    return (
        int(getattr(t, "year", 0)),
        int(getattr(t, "month", 0)),
        int(getattr(t, "day", 1)),
        int(getattr(t, "hour", 0)),
        int(getattr(t, "minute", 0)),
        int(getattr(t, "second", 0)),
    )


def _is_monthly_cftime(time_list):
    # Sort BEFORE checks (matches your earlier requirement)
    time_sorted = sorted(time_list, key=_time_key)

    if len(time_sorted) < 2:
        return False, "too few time points", time_sorted

    if any(t is None for t in time_sorted):
        return False, "time contains None", time_sorted

    # duplicates (compare by components to avoid calendar quirks)
    keys = [_time_key(t) for t in time_sorted]
    if len(set(keys)) != len(keys):
        return False, "duplicate timestamps", time_sorted

    # monotonic increasing (strict)
    if any(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)):
        return False, "time not increasing", time_sorted

    month_index = np.array([k[0] * 12 + k[1] for k in keys], dtype=np.int64)
    steps = np.diff(month_index)

    if np.all(steps == 1):
        return True, "monthly", time_sorted

    uniq, cnt = np.unique(steps, return_counts=True)
    return False, f"not monthly; month-step diffs={dict(zip(uniq.tolist(), cnt.tolist()))}", time_sorted


def _is_daily_cftime(time_list):
    # Sort BEFORE checks (matches the monthly checker's contract)
    time_sorted = sorted(time_list, key=_time_key)

    if len(time_sorted) < 2:
        return False, "too few time points", time_sorted

    if any(t is None for t in time_sorted):
        return False, "time contains None", time_sorted

    # duplicates (compare by components to avoid calendar quirks)
    keys = [_time_key(t) for t in time_sorted]
    if len(set(keys)) != len(keys):
        return False, "duplicate timestamps", time_sorted

    # monotonic increasing (strict)
    if any(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)):
        return False, "time not increasing", time_sorted

    # Calendar-aware day stepping: two cftime datetimes one calendar day
    # apart differ by exactly timedelta(days=1) regardless of the file's
    # calendar (noleap/360_day/gregorian), because cftime arithmetic
    # respects that calendar. So a clean daily series has every
    # consecutive difference equal to one day (e.g. on a 360_day file the
    # day-30 -> next-month day-1 step is still exactly 1 day).
    one_day = timedelta(days=1)
    try:
        diffs = [
            time_sorted[i + 1] - time_sorted[i]
            for i in range(len(time_sorted) - 1)
        ]
    except Exception as e:
        return False, f"daily-step check failed: {type(e).__name__}: {e}", time_sorted

    bad = [d for d in diffs if d != one_day]
    if not bad:
        return True, "daily", time_sorted

    sizes = np.array([round(d.total_seconds() / 86400.0, 6) for d in bad])
    uniq, cnt = np.unique(sizes, return_counts=True)
    return False, f"not daily; day-step diffs={dict(zip(uniq.tolist(), cnt.tolist()))}", time_sorted


def _units_per_day(units: str) -> Optional[float]:
    """How many of the file's time units make one calendar day, or None if
    the unit token isn't understood. CMIP6 daily files are almost always
    'days since ...' (-> 1.0)."""
    tok = str(units).strip().split()[0].lower() if units else ""
    return {
        "day": 1.0, "days": 1.0, "d": 1.0,
        "hour": 24.0, "hours": 24.0, "hr": 24.0, "hrs": 24.0, "h": 24.0,
        "minute": 1440.0, "minutes": 1440.0, "min": 1440.0, "mins": 1440.0,
        "second": 86400.0, "seconds": 86400.0, "sec": 86400.0,
        "secs": 86400.0, "s": 86400.0,
    }.get(tok)


def _is_daily_numeric(raw, units):
    """Vectorised daily-spacing check on the *undecoded* numeric time axis.

    Daily CMIP6 points are exactly one day apart in the numeric encoding
    regardless of calendar (noleap/360_day/gregorian), so we never need to
    build cftime objects: ``np.diff`` on the sorted raw values must be a
    constant equal to "1 day" expressed in the file's time units. This is
    pure vectorised numpy (no per-timestep Python objects), which is why it
    is dramatically faster than decoding ~7300 cftime datetimes per daily
    file and looping over them.

    Returns ``(ok, msg)``. ``msg == "unparseable_units"`` signals the caller
    to fall back to the (correct but slow) cftime check.
    """
    step = _units_per_day(units)
    if step is None:
        return False, "unparseable_units"

    t = np.asarray(raw, dtype="float64").ravel()
    if t.size < 2:
        return False, "too few time points"
    if not np.all(np.isfinite(t)):
        return False, "time contains non-finite values"

    t = np.sort(t)
    d = np.diff(t)

    if np.any(d <= 0):
        return False, "duplicate or non-increasing timestamps"

    # tolerance: 1e-3 of a day (~1.4 min) absorbs float round-off in
    # 'days since' encodings without admitting sub-daily/multi-day steps.
    tol = 1e-3 * step
    if np.all(np.abs(d - step) <= tol):
        return True, "daily"

    bad = d[np.abs(d - step) > tol] / step          # express gaps in days
    uniq, cnt = np.unique(np.round(bad, 6), return_counts=True)
    return False, f"not daily; day-step diffs={dict(zip(uniq.tolist(), cnt.tolist()))}"


def _data_var_name(ds, time_name, lat_name, lon_name):
    """Geophysical variable = the data_var spanning time+lat+lon.

    Skips bounds/aux vars (lat_bnds, lon_bnds, height, ...) which lack a
    time dim. Returns the first match, or None if no such variable exists.
    """
    needed = {time_name, lat_name, lon_name}
    for v in ds.data_vars:
        if needed.issubset(set(ds[v].dims)):
            return v
    return None


def _check_data_slices(
        nc_path, time_name, lat_name, lon_name,
        check_nan=True, check_constant=False, n_samples=3,
        ):
    """Lightweight data sanity checks via a *separate* lazy handle.

    Opens ``nc_path`` independently with ``mask_and_scale=True`` (so
    ``_FillValue``/``missing_value`` sentinels read back as NaN) and
    ``chunks={}`` (lazy), then materialises only a few full lat/lon slices
    sampled along the time axis. A single 2-D slice covers every spatial
    cell, so any valid land/ocean cell at a sampled timestep is seen; the
    whole cube is never read, keeping the check cheap across thousands of
    files. Both checks share this one read.

    Checks (run only when requested):
      * ``check_nan``      -- every sampled value non-finite -> ``"all_nan"``.
      * ``check_constant`` -- every finite sampled value identical ->
        ``"constant"``. Subsumes all-zero; a geophysical field is never
        globally+temporally uniform, so a flat field is corrupt. Compared
        over finite values only, so masked (NaN) cells don't defeat it.

    Returns ``(ok, reason)``:
      * ``(False, "no_data_var")`` -- no time+lat+lon variable present
      * ``(False, "all_nan")``     -- no finite values in the sample
      * ``(False, "constant")``    -- all finite sampled values are identical
      * ``(True,  "ok")``          -- passed every requested check
    """
    try:
        ds = xr.open_dataset(
            nc_path,
            decode_times=False,
            mask_and_scale=True,
            chunks={},
        )
    except Exception as e:
        return False, f"data_open_failed: {type(e).__name__}: {e}"
    try:
        var = _data_var_name(ds, time_name, lat_name, lon_name)
        if var is None:
            return False, "no_data_var"

        da = ds[var]
        nt = int(da.sizes.get(time_name, 0))
        if nt <= n_samples:
            sample = da
        else:
            idx = sorted({0, nt // 2, nt - 1})
            sample = da.isel({time_name: idx})

        arr = np.asarray(sample.values)        # loads only the sampled slices
        finite = np.isfinite(arr)

        if check_nan and not finite.any():
            return False, "all_nan"

        if check_constant:
            vals = arr[finite]
            if vals.size == 0:
                # nothing finite to compare; that's an all-NaN field
                return False, "all_nan"
            if vals.min() == vals.max():
                return False, "constant"

        return True, "ok"
    except Exception as e:
        return False, f"data_check_failed: {type(e).__name__}: {e}"
    finally:
        ds.close()


def validate_nc_file(
        nc_path: Path,
        time_coder: CFDatetimeCoder,
        timestep: str = "mon",
        check_data: bool = False,
        check_constant: bool = False,
        ):
    out = {
        "path": str(nc_path),
        "valid": False,
        "time_coord": None,
        "lat_coord": None,
        "lon_coord": None,
        "n_time": None,
        "time_start": None,
        "time_end": None,
        "expected_freq": None,
        "freq_ok": None,
        "data_ok": None,
        "time_check_mode": "",
        "reason": "",
    }

    # Anything other than 'day' (e.g. 'Amon', 'mon') is treated as monthly.
    is_daily = str(timestep).lower() == "day"
    out["expected_freq"] = "daily" if is_daily else "monthly"

    try:
        # Daily: skip time decoding entirely. Decoding builds a cftime
        # object per timestep (~7300 for a 20-yr daily file) which is the
        # main reason daily validation is slow; the spacing check below
        # works directly on the raw numeric axis.
        ds = xr.open_dataset(
            nc_path,
            decode_times=False if is_daily else time_coder,
            mask_and_scale=not is_daily,
        )
    except Exception as e:
        out["reason"] = f"open_failed: {e}"

        return out

    try:
        time_name = _pick_coord_name(ds, ["time", "Time"])
        lat_name = _pick_coord_name(ds, ["lat", "latitude"])
        lon_name = _pick_coord_name(ds, ["lon", "longitude"])

        out["time_coord"] = time_name
        out["lat_coord"] = lat_name
        out["lon_coord"] = lon_name

        if time_name is None or lat_name is None or lon_name is None:
            missing = [k for k, v in [("time", time_name), ("lat", lat_name), ("lon", lon_name)] if v is None]
            out["reason"] = f"missing_coord: {','.join(missing)}"
            return out

        # Optional: confirm the file actually holds data -- not an all-NaN /
        # all-fill field (check_data) and/or not a flat constant field
        # (check_constant, subsumes all-zero). Both share one separate lazy
        # handle, so the existing open/time logic is left untouched.
        if check_data or check_constant:
            data_ok, data_reason = _check_data_slices(
                nc_path, time_name, lat_name, lon_name,
                check_nan=check_data, check_constant=check_constant,
            )
            out["data_ok"] = bool(data_ok)
            if not data_ok:
                out["reason"] = data_reason
                return out

        if is_daily:
            # Fast path: validate spacing on the raw, undecoded numeric time
            # axis (vectorised numpy, no per-timestep cftime objects). Daily
            # points are exactly 1 day apart numerically in every CMIP6
            # calendar, so this is both correct and ~30x cheaper than
            # decoding the whole axis to cftime and looping in Python.
            tvar = ds[time_name]
            raw = np.asarray(tvar.values).ravel()
            units = tvar.attrs.get("units", "") or tvar.encoding.get("units", "")
            calendar = tvar.attrs.get("calendar", "standard")
            out["time_check_mode"] = "numeric"
            out["n_time"] = int(raw.size)

            ok, msg = _is_daily_numeric(raw, units)

            if msg == "unparseable_units":
                # Rare (CMIP6 uses 'days since ...'): fall back to the
                # correct, calendar-aware but slower cftime check.
                out["time_check_mode"] = "cftime"
                try:
                    decoded = cftime.num2date(raw, units, calendar)
                    time_list = list(np.asarray(decoded).ravel())
                except Exception as e:
                    out["reason"] = f"time_decode_failed: {type(e).__name__}: {e}"
                    return out
                ok, msg, time_sorted = _is_daily_cftime(time_list)
                if len(time_sorted) > 0:
                    out["time_start"] = getattr(time_sorted[0], "isoformat", lambda: str(time_sorted[0]))()
                    out["time_end"] = getattr(time_sorted[-1], "isoformat", lambda: str(time_sorted[-1]))()
            elif raw.size > 0 and np.all(np.isfinite(raw)):
                # Decode only the two endpoints for human-readable reporting.
                try:
                    lo = cftime.num2date(float(np.min(raw)), units, calendar)
                    hi = cftime.num2date(float(np.max(raw)), units, calendar)
                    out["time_start"] = lo.isoformat()
                    out["time_end"] = hi.isoformat()
                except Exception:
                    pass

            out["freq_ok"] = bool(ok)
            out["reason"] = "ok" if ok else msg
            out["valid"] = bool(ok)
            return out

        time_vals = ds[time_name].values

        # ---- monthly: pandas first (nice reporting), cftime fallback -----
        try:
            dti = _time_to_datetime_index(time_vals).sort_values()
            out["time_check_mode"] = "pandas"
            out["n_time"] = int(len(dti))
            if len(dti) > 0 and not dti.hasnans:
                out["time_start"] = dti[0].isoformat()
                out["time_end"] = dti[-1].isoformat()

            ok, msg = _is_monthly_pandas(dti)
            out["freq_ok"] = bool(ok)
            out["reason"] = "ok" if ok else msg
            out["valid"] = bool(ok)
            return out

        except Exception:
            # Fallback: cftime / datetime-like checks without pandas conversion
            out["time_check_mode"] = "cftime"
            time_list = _as_cftime_list(time_vals)
            out["n_time"] = int(len(time_list))

            ok, msg, time_sorted = _is_monthly_cftime(time_list)
            out["freq_ok"] = bool(ok)
            if len(time_sorted) > 0:
                out["time_start"] = getattr(time_sorted[0], "isoformat", lambda: str(time_sorted[0]))()
                out["time_end"] = getattr(time_sorted[-1], "isoformat", lambda: str(time_sorted[-1]))()

            out["reason"] = "ok" if ok else msg
            out["valid"] = bool(ok)
            return out

    finally:
        ds.close()


def validate_cmip6_files(
        root_dir, 
        report_csv_path, 
        pattern="*.nc", 
        recursive=True,
        progress_every=100,
        del_invalid:bool = False,
        timestep: Optional[str] = None,
        check_data: bool = False,
        check_constant: bool = False,
        ):
    """Validate saved CMIP6 files (coords present + expected time frequency).

    If ``check_data`` is True, also verify that the geophysical variable holds
    at least one finite value (a few sampled timesteps); files that are
    entirely NaN/fill, or that carry no time+lat+lon variable, are marked
    invalid (``reason`` = ``all_nan`` / ``no_data_var``).

    If ``check_constant`` is True, also reject files whose sampled data is a
    flat constant (all finite values identical; ``reason`` = ``constant``).
    This subsumes the all-zero case and catches dead/garbage fields for any
    variable, since no real geophysical field is globally+temporally uniform.

    Both checks share one extra (lazy) open per file and read only a handful
    of lat/lon slices, so they stay cheap across many files; both off by
    default.

    The expected time frequency is taken from ``timestep``: ``'day'`` checks
    that the files are daily, anything else (e.g. ``'Amon'``) checks monthly.
    If ``timestep`` is None it defaults to ``CONFIG.timestep`` so that the
    same folder layout (``CONFIG.cmip6_data/CONFIG.timestep``) is validated
    with the matching frequency check.

    import os

    from utils import validate_cmip6_files
    from config import CONFIG

    validate_cmip6_files(
        root_dir=os.path.join(CONFIG.cmip6_data, CONFIG.timestep),
        timestep = CONFIG.timestep,
        report_csv_path=f"{CONFIG.timestep}_validation_report.csv",
        pattern="*.nc",
        recursive=True,
        progress_every=500,
        #del_invalid=True,
    )
    """
    if timestep is None:
        from config import CONFIG
        timestep = CONFIG.timestep

    root_dir = Path(root_dir)
    files = sorted(root_dir.rglob(pattern) if recursive else root_dir.glob(pattern))

    print(
        f"Validating {len(files)} files in {root_dir} | pattern={pattern} | "
        f"timestep={timestep}"
    )

    time_coder = CFDatetimeCoder(use_cftime=True)

    start = time.time()

    rows = []
    total = len(files)
    for i, p in enumerate(files, start=1):
        result = validate_nc_file(p, time_coder=time_coder, timestep=timestep,
                                  check_data=check_data, check_constant=check_constant)
        rows.append(result)
        if progress_every and (i % progress_every == 0):
            print(f"Processed {i}/{total}")

        if result['valid'] is False:
            print(f"Invalid file: {result['path']} | reason: {result['reason']}")

            if del_invalid:
                try:
                    Path(result['path']).unlink()
                except Exception as e:
                    print(f"Failed to delete invalid file: {result['path']} | error: {e}")

    df = pd.DataFrame(rows)
    Path(report_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_csv_path, index=False)

    n_ok = int(df["valid"].sum()) if len(df) else 0
    print(f"Wrote {report_csv_path} | valid {n_ok}/{len(df)}")
    print(f"Validation completed in {time.time() - start:.1f} seconds")
    return df


#===================================
#   Catalog / data-source backends (Pangeo + ESGF)
#===================================

def _search_pangeo(scenarios, common_query) -> pd.DataFrame:
    """Pangeo CMIP6 catalog search (Zarr on Google Cloud)."""
    import intake
    col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(col_url)
    hist_df = col.search(
        activity_id="CMIP",
        experiment_id="historical",
        **common_query,
    ).df
    ssp_df = col.search(
        activity_id="ScenarioMIP",
        experiment_id=scenarios,
        **common_query,
    ).df
    logger.info(
        f"Pangeo search: {len(hist_df)} historical + {len(ssp_df)} scenario rows"
    )
    return pd.concat([hist_df, ssp_df], ignore_index=True)


def _search_esgf(scenarios, common_query, CONFIG) -> pd.DataFrame:
    """ESGF CMIP6 catalog search via intake-esgf.

    Returns a DataFrame whose group-key columns (experiment_id, source_id,
    member_id, variable_id, table_id, grid_label, activity_id) match the
    Pangeo path so the rest of download_cmip_data.py is backend-agnostic.
    Replicas and old versions are filtered out so each
    (exp, model, member, var, grid) appears once.
    """
    from intake_esgf import ESGFCatalog
    try:
        from intake_esgf.exceptions import NoSearchResults
    except ImportError:
        NoSearchResults = ()  # nothing to catch on older versions

    cache_dir = os.path.join(CONFIG.tmp_path, "esgf_raw")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from intake_esgf import conf as _esgf_conf
        _esgf_conf.set(local_cache=[cache_dir])
    except Exception:
        logger.warning(
            "intake_esgf.conf.set unavailable; using library-default cache "
            f"instead of {cache_dir}"
        )

    def _query(activity_id, experiment_id) -> pd.DataFrame:
        try:
            cat = ESGFCatalog().search(
                project="CMIP6",
                activity_id=activity_id,
                experiment_id=experiment_id,
                quiet=True,
                **common_query,
            )
        except NoSearchResults:
            logger.info(
                f"ESGF: no results for activity={activity_id}, "
                f"experiment={experiment_id}"
            )
            return pd.DataFrame()
        df = cat.df.copy()
        if len(df) == 0:
            return df
        # intake-esgf names the activity column 'activity_drs'; align with Pangeo.
        if "activity_drs" in df.columns and "activity_id" not in df.columns:
            df = df.rename(columns={"activity_drs": "activity_id"})
        if not CONFIG.esgf_prefer_replica and "replica" in df.columns:
            primary = df[~df["replica"]]
            df = primary if len(primary) else df
        if CONFIG.esgf_latest_only and "version" in df.columns:
            keep_cols = [
                "source_id", "experiment_id", "member_id",
                "variable_id", "table_id", "grid_label",
            ]
            df = (
                df.sort_values("version")
                  .groupby(keep_cols, as_index=False, dropna=False)
                  .tail(1)
            )
        return df

    hist_df = _query("CMIP", "historical")
    ssp_df = _query("ScenarioMIP", scenarios)

    # HighResMIP: high-resolution models (e.g. ECMWF-IFS-HR, EC-Earth3P-HR,
    # CMCC-CM2-VHR4, MRI-AGCM3-2-S) live under activity_id='HighResMIP' with
    # their own experiment names rather than 'historical'/'sspXXX'.
    #   hist-1950 / highresSST-present : historical analogs (~1950-2014)
    #   highres-future / highresSST-future : future analog (~2015-2050)
    # We query all four so coupled and SST-forced runs are both surfaced.
    hr_experiments = [
        "hist-1950",
        "highresSST-present",
        "highres-future",
        "highresSST-future",
    ]
    hr_df = _query("HighResMIP", hr_experiments)

    logger.info(
        f"ESGF search: {len(hist_df)} historical (CMIP) + "
        f"{len(ssp_df)} scenario (ScenarioMIP) + "
        f"{len(hr_df)} HighResMIP rows"
    )
    #return pd.concat([hist_df, ssp_df, hr_df], ignore_index=True)
    return ssp_df


def search_catalog(scenarios, variables, models, CONFIG) -> pd.DataFrame:
    """Unified CMIP6 catalog search; dispatches on CONFIG.data_source."""
    common_query = dict(
        table_id=CONFIG.timestep,
        variable_id=variables,
        source_id=models,
        grid_label=CONFIG.grid_label,
    )
    if CONFIG.data_source == "pangeo":
        return _search_pangeo(scenarios, common_query)
    if CONFIG.data_source == "esgf":
        return _search_esgf(scenarios, common_query, CONFIG)
    raise ValueError(
        f"Unknown CONFIG.data_source: {CONFIG.data_source!r} "
        "(expected 'pangeo' or 'esgf')"
    )


def _ensure_esgf_files_downloaded(group, CONFIG) -> List[str]:
    """Download (if not already cached) the NetCDF files for one group from
    ESGF and return sorted local paths. The cache lives under
    CONFIG.tmp_path/esgf_raw/ so the caller can delete it later. Caching
    is performed by intake-esgf when its conf API is available; otherwise
    the library's default cache is used.
    """
    from intake_esgf import ESGFCatalog
    from intake_esgf.exceptions import DatasetLoadError

    try:
        from intake_esgf.exceptions import NoSearchResults
    except ImportError:
        NoSearchResults = ()

    cache_dir = os.path.join(CONFIG.tmp_path, "esgf_raw")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from intake_esgf import conf as _esgf_conf
        _esgf_conf.set(local_cache=[cache_dir])
    except Exception:
        logger.warning(
            "intake_esgf.conf.set unavailable; using library-default cache "
            f"instead of {cache_dir}"
        )

    row = group.iloc[0]
    try:
        cat = ESGFCatalog().search(
            project="CMIP6",
            activity_id=row["activity_id"],
            experiment_id=row["experiment_id"],
            source_id=row["source_id"],
            member_id=row["member_id"],
            table_id=row["table_id"],
            variable_id=row["variable_id"],
            grid_label=row["grid_label"],
            quiet=True,
        )
    except NoSearchResults:
        logger.warning(f"ESGF returned no datasets for group {row.to_dict()}")
        return []
    if len(cat.df) == 0:
        logger.warning(f"ESGF returned no datasets for group {row.to_dict()}")
        return []

    if not CONFIG.esgf_prefer_replica and "replica" in cat.df.columns:
        primary = cat.df[~cat.df["replica"]]
        if len(primary):
            cat.df = primary
    if CONFIG.esgf_latest_only and "version" in cat.df.columns:
        cat.df = cat.df.sort_values("version").tail(1)

    try:
        path_dict = cat.to_path_dict(quiet=True)
    except DatasetLoadError as e:
        logger.error(
            f"ESGF could not download files for "
            f"{row['experiment_id']}/{row['source_id']}/{row['member_id']}/"
            f"{row['variable_id']}/{row['grid_label']}: {e}"
        )
        return []

    paths: List[str] = []
    for _, plist in path_dict.items():
        paths.extend(str(p) for p in plist)
    return sorted(paths)


def open_group_dataset(group, decode_times, CONFIG) -> Optional[xr.Dataset]:
    """Open the multi-file CMIP6 dataset for one group; dispatches on
    CONFIG.data_source. Slicing/save logic in download_cmip_data.py is
    therefore identical across backends. Returns None if the ESGF backend
    could not retrieve any files for this group, so the caller can skip
    instead of aborting the whole loop.
    """
    if CONFIG.data_source == "pangeo":
        return xr.open_mfdataset(
            group.zstore.tolist(),
            combine="by_coords",
            engine="zarr",
            chunks={},
            decode_times=decode_times,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            backend_kwargs={
                "consolidated": True,
                "storage_options": {"token": "anon"},
            },
        )
    if CONFIG.data_source == "esgf":
        files = _ensure_esgf_files_downloaded(group, CONFIG)
        if not files:
            row = group.iloc[0]
            logger.warning(
                "ESGF returned no files for "
                f"{row['experiment_id']}/{row['source_id']}/{row['member_id']}/"
                f"{row['variable_id']}/{row['grid_label']}; skipping group"
            )
            return None
        return xr.open_mfdataset(
            files,
            combine="by_coords",
            engine="netcdf4",
            chunks={},
            decode_times=decode_times,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
    raise ValueError(
        f"Unknown CONFIG.data_source: {CONFIG.data_source!r} "
        "(expected 'pangeo' or 'esgf')"
    )


# ----------------------------
# Timeout (keeps scan fast)
# ----------------------------
class TimeoutError(RuntimeError):
    pass


@contextmanager
def time_limit(seconds: int):
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ----------------------------
# Expected schema helpers
# ----------------------------
def future_period_keys_from_config(CONFIG) -> Sequence[str]:
    keys = []
    for start, end in CONFIG.future_periods:
        st_yr = int(start.split("-")[0])
        en_yr = int(end.split("-")[0])
        keys.append(f"{st_yr}-{en_yr}")
    return keys


def parse_variable_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    if not base.endswith(".nc"):
        return None
    stem = base[:-3]
    parts = stem.split("_")
    # expected: scenario_model_member_variable.nc
    if len(parts) < 4:
        return None
    return parts[-1]


def _infer_spatial_dims(nc: Dataset) -> Optional[Tuple[str, ...]]:
    dims = set(nc.dimensions.keys())
    vars_ = nc.variables

    # Rectilinear grids (most common)
    if "lat" in dims and "lon" in dims:
        return ("lat", "lon")
    if "latitude" in dims and "longitude" in dims:
        return ("latitude", "longitude")

    # Curvilinear (coords exist as 2D variables)
    for a, b in (("nav_lat", "nav_lon"), ("lat", "lon"), ("latitude", "longitude")):
        if a in vars_ and b in vars_:
            da = vars_[a].dimensions
            db = vars_[b].dimensions
            if da == db and len(da) >= 2:
                return tuple(da)

    # Fallback common grid dims
    for cand in (("y", "x"), ("j", "i")):
        if cand[0] in dims and cand[1] in dims:
            return cand

    return None


def _dim_sizes(nc: Dataset, dim_names: Iterable[str]) -> dict:
    return {d: int(nc.dimensions[d].size) for d in dim_names if d in nc.dimensions}


@dataclass
class ValidationResult:
    path: str
    ok: bool
    reason: str
    variable: str
    method: str


def validate_delta_file(
    path: str,
    expected_keys: Sequence[str],
    method: str,
    timeout_s: int = 2,
) -> ValidationResult:
    var = parse_variable_from_filename(path) or ""
    required_vars = list(expected_keys)
    if var == "pr":
        required_vars += [f"{k}_wetfreq" for k in expected_keys]

    try:
        with time_limit(timeout_s):
            nc = Dataset(path, mode="r")
    except Exception as e:
        return ValidationResult(path, False, f"open_failed: {type(e).__name__}: {e}", var, method)

    try:
        spatial_dims = _infer_spatial_dims(nc)
        if spatial_dims is None:
            return ValidationResult(path, False, "missing_spatial_dims_or_coords", var, method)

        sizes = _dim_sizes(nc, spatial_dims)
        if any(sizes.get(d, 0) <= 0 for d in spatial_dims):
            return ValidationResult(path, False, f"empty_spatial_dim(s): {sizes}", var, method)

        missing = [v for v in required_vars if v not in nc.variables]
        if missing:
            return ValidationResult(path, False, f"missing_vars: {missing[:10]}{'...' if len(missing)>10 else ''}", var, method)

        # Method-specific dim checks (metadata-only; very fast)
        if method in ("qqmap_day", "qqmap_day2"):
            for k in expected_keys:
                dims = nc.variables[k].dimensions
                if "sample" not in dims:
                    return ValidationResult(path, False, f"{k}: missing_dim=sample; dims={dims}", var, method)
                if not all(d in dims for d in spatial_dims):
                    return ValidationResult(path, False, f"{k}: missing_spatial_dims; dims={dims}", var, method)

            for aux in ("sample_month", "rang"):
                if aux not in nc.variables:
                    return ValidationResult(path, False, f"missing_aux_var: {aux}", var, method)
                if nc.variables[aux].dimensions != ("sample",):
                    return ValidationResult(path, False, f"{aux}: expected_dims=('sample',) got {nc.variables[aux].dimensions}", var, method)

            if var == "pr":
                for k in expected_keys:
                    wk = f"{k}_wetfreq"
                    dims = nc.variables[wk].dimensions
                    if "month" not in dims:
                        return ValidationResult(path, False, f"{wk}: missing_dim=month; dims={dims}", var, method)
                    if not all(d in dims for d in spatial_dims):
                        return ValidationResult(path, False, f"{wk}: missing_spatial_dims; dims={dims}", var, method)

        elif method == "qqmap_bin_day":
            for k in expected_keys:
                dims = nc.variables[k].dimensions
                for needed in ("month", "quantile"):
                    if needed not in dims:
                        return ValidationResult(path, False, f"{k}: missing_dim={needed}; dims={dims}", var, method)
                if not all(d in dims for d in spatial_dims):
                    return ValidationResult(path, False, f"{k}: missing_spatial_dims; dims={dims}", var, method)

        elif method == "mon_mean":
            for k in expected_keys:
                dims = nc.variables[k].dimensions
                if "month" not in dims:
                    return ValidationResult(path, False, f"{k}: missing_dim=month; dims={dims}", var, method)
                if not all(d in dims for d in spatial_dims):
                    return ValidationResult(path, False, f"{k}: missing_spatial_dims; dims={dims}", var, method)

        else:
            return ValidationResult(path, False, f"unknown_method: {method}", var, method)

        return ValidationResult(path, True, "ok", var, method)

    except TimeoutError as e:
        return ValidationResult(path, False, f"timeout: {e}", var, method)
    except Exception as e:
        return ValidationResult(path, False, f"validate_failed: {type(e).__name__}: {e}", var, method)
    finally:
        try:
            nc.close()
        except Exception:
            pass


def validate_delta_maps(
        pattern: Optional[str] = "*.nc",
        progress_every: int = 200,
        delete_invalid: bool = False,
):
    """
    from utils import validate_delta_maps

    validate_delta_maps(
        delete_invalid=True,
    )

    Parameters
    ----------
    delete_invalid : bool, optional
        If True, delete each delta file that fails validation. The CSV is still
        written first (recording every file and its ``ok``/``reason``) so there
        is a record of what was removed. Defaults to False.
    """
    from config import CONFIG

    method = str(CONFIG.delta_method)
    root = os.path.join(CONFIG.cmip6_deltas, method)
    keys = future_period_keys_from_config(CONFIG)

    nc_files = sorted(glob.glob(os.path.join(root, pattern)))
    print(f"Delta method: {method}")
    print(f"Folder: {root}")
    print(f"Found {len(nc_files)} netCDF files")

    out_csv = os.path.join(os.getcwd(), f"delta_validation_{method}.csv")

    t0 = time.time()
    ok = 0
    bad = 0
    deleted = 0

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "method",
                "variable",
                "ok",
                "reason",
                "bytes",
            ],
        )
        w.writeheader()

        for i, path in enumerate(nc_files, 1):
            res = validate_delta_file(path, expected_keys=keys, method=method, timeout_s=2)
            size_b = ""
            try:
                size_b = os.path.getsize(path)
            except Exception:
                pass

            w.writerow(
                {
                    "path": path,
                    "method": res.method,
                    "variable": res.variable,
                    "ok": int(bool(res.ok)),
                    "reason": res.reason,
                    "bytes": size_b,
                }
            )

            if res.ok:
                ok += 1
            else:
                bad += 1
                if delete_invalid:
                    try:
                        os.remove(path)
                        deleted += 1
                        print(f"Deleted invalid delta file ({res.reason}): {path}")
                    except OSError as e:
                        print(f"Could not delete {path}: {e}")

            if i % progress_every == 0:
                elapsed = time.time() - t0
                print(f"[{i}/{len(nc_files)}] ok={ok} bad={bad} elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ok={ok}  bad={bad}")
    if delete_invalid:
        print(f"Deleted {deleted} invalid delta file(s)")
    print(f"CSV written to: {out_csv}")

    return out_csv
