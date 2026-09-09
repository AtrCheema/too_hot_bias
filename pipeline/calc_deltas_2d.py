"""
This script calculates the delta maps for the 2D variables (pr, tas, tasmin, tasmax) 
for all models and members in the CMIP6 dataset. The delta maps are calculated as the 
difference between the future period (2015-2100) and the historical period (2001-2020) 
for each month of the year. The delta maps are saved as netCDF files in the 
cmip6_deltas/{CONFIG.delta_method} directory.

The generated delta files will have following dimensions and variables:
- dimensions: lat, lon
- variables: one variable per future period, e.g. "2041-2060", additionally for pr there will be "2041-2060_wetfreq" which is the change in wet day frequency

This script needs to be run after downloading the CMIP6 data using the download_cmip_data.py script.
This script can be run in parallel for different variables and scenarios using the calc_deltas.sh script.
"""

import gc
import os
import sys
import time
import glob
import logging
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# repo root (one dir up) holds config.py and the shared library modules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG

from utils import (
    calc_delta,
    is_delta_valid,
    standardize_and_match_grids,
    get_xy_coordname,
    load_temporal_data,
    load_historical_data,
)


variable = str(sys.argv[1]) if len(sys.argv)>1 else 'pr'
scenarios = [str(sys.argv[2])] if len(sys.argv)>2 else CONFIG.scenarios
model_filter = [str(sys.argv[3])] if len(sys.argv)>3 else CONFIG.models
jobid = str(sys.argv[4]) if len(sys.argv)>4 else 'delta_job'

logging.basicConfig(#filename=f'delta_{variable}_{jobid}.log', filemode='w',
                    stream=sys.stdout,
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.info(f"Variable: {variable}")
logger.info(f"Scenarios: {scenarios}")
logger.info(f"Model filter: {model_filter}")
logger.info(f"Job ID: {jobid}")

# For some models the resolution changes during the simulation 
# For example, see the following two files:
# pr_Amon_EC-Earth3-LR_piControl_r1i1p1f1_gr_199101-199101.nc
# pr_Amon_EC-Earth3-LR_piControl_r1i1p1f1_gr_221901-221912.nc
# This issue should be addressed at some point

NO_2015_2020_DATA = {}
NO_HIST_FILES = []
NO_FILES_FOUND = []
NO_FUTURE_DATA = []
WRONG_NUM_FILES = []
INVALID_DELTA = []

#==============================================================================
#   Settings
#==============================================================================

variables = ['pr',
             'tasmin', 
             'tasmax', 
             #'tas'
             ]

PREFIX = {
    'tos': "Omon",
}

#==============================================================================
#   Loop over scenarios, models, and members and load data and generate 
#   figures for verication.
#==============================================================================

figures_path = os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method, 'figures')
if os.path.isdir(figures_path)==False:
    logger.info('Creating '+figures_path)
    os.makedirs(figures_path, exist_ok=True)   # exist_ok: avoid race when many array tasks start together

# get the variable
assert variable in variables

# Loop over scenarios     
scen_start = time.time()   
for scenario in scenarios:

    if CONFIG.verbosity: 
        logger.info(' Processing '+variable+' '+scenario)
    
    if CONFIG.timestep == "Amon":
        prefix = PREFIX.get(variable, 'Amon')
    else:
        prefix = CONFIG.timestep

    # Generate list of models
    if CONFIG.timestep == 'day':
        scen_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"{scenario}_*_{variable}_{prefix}_*.nc"))
    else:
        scen_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"{scenario}_*_{variable}_{prefix}.nc"))
    models = sorted(np.unique([os.path.basename(x).split('_')[1] for x in scen_files]).tolist())

    try:
        models.remove('historical')
    except:
        pass

    models = [model for model in models if model in model_filter]
    
    # Loop over models
    m_start = time.time()  # just for the case when len(models)==0
    for mm, model in enumerate(models):
        m_start = time.time()

        if CONFIG.verbosity>1:
            logger.info(f"  Processing {variable} {scenario} model: {model} ({mm}/{len(models)})")
    
        # Generate list of ensemble members
        if CONFIG.timestep == 'day':
            mod_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"{scenario}_{model}_*_{variable}_{prefix}_*.nc"))
        else:
            mod_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"{scenario}_{model}_*_{variable}_{prefix}.nc"))
        members = sorted(np.unique([os.path.basename(x).split('_')[2] for x in mod_files]).tolist())
        # Cap to the first N sorted members, matching the selection used by
        # download_cmip_data.py and catch_avg_deltas_stn.py (the only consumers
        # of these raw delta maps). Members beyond the cap are never read
        # downstream, so computing them is wasted effort (e.g. EC-Earth3 ships
        # 58-98 members but only ~20 are ever used). max_members=None -> all.
        _max_members = getattr(CONFIG, "max_members_per_model", None)
        if _max_members is not None:
            members = members[:_max_members]
        #random.shuffle(members)
        
        # Loop over ensemble members
        ens_start = time.time()  # just for the case when len(members)==0
        for ee, member in enumerate(members):
            ens_start = time.time()

            if CONFIG.verbosity>2: 
                logger.info(f"   Processing {variable} {scenario} model: {model} ({mm}/{len(models)}) member: {member} ({ee}/{len(members)})") 
            
            # HB: CHECK IF DELTA MAP ALREADY PRODUCED
            if os.path.isfile(os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method, f"{scenario}_{model}_{member}_{variable}.nc")):
                if CONFIG.verbosity>3:
                    logger.info(f"    Delta map already produced for {scenario}_{model}_{member}_{variable}.nc")
                continue
                    
            # List of files to be loaded
            if CONFIG.timestep == 'day':
                ens_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"{scenario}_{model}_{member}_{variable}_{prefix}_*.nc"))
                # remove the files which have _20150101_20201231 in the name
                ens_files = [f for f in ens_files if not f.endswith("_20150101_20201231.nc")]
            else:
                ens_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"{scenario}_{model}_{member}_{variable}_{prefix}.nc"))
            ens_files = sorted(ens_files)
            if len(ens_files)==0:
                logger.warning('No ensemble files found for '+variable+' '+scenario+' '+model+' '+member)
                NO_FILES_FOUND.append(f"{scenario}_{model}_{member}_{variable}")
                continue

            # derive grid from future files; all future files for this (scenario, model, member, variable)
            # must share one grid — otherwise something inconsistent was downloaded.
            def _grid_of(p: str) -> str:
                return os.path.basename(p).split('_')[5] if CONFIG.timestep == 'day' \
                    else os.path.basename(p).split('_')[5].split('.')[0]

            grids = {_grid_of(f) for f in ens_files}
            if len(grids) != 1:
                logger.warning(f"Mixed grids {sorted(grids)} for {scenario}_{model}_{member}_{variable}; skipping")
                NO_FILES_FOUND.append(f"{scenario}_{model}_{member}_{variable}")
                continue
            grid = grids.pop()

            if CONFIG.timestep == 'day':
                hist_scen_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"historical_{model}_{member}_{variable}_{prefix}_{grid}_*.nc"))
                # remove the files which have _20150101_20201231 in the name
                hist_scen_files = [f for f in hist_scen_files if not f.endswith("_20150101_20201231.nc")]
            else:
                hist_scen_files = glob.glob(os.path.join(CONFIG.cmip6_data, CONFIG.timestep, f"historical_{model}_{member}_{variable}_{prefix}_{grid}.nc"))
            hist_scen_files = sorted(hist_scen_files)
            if len(hist_scen_files)==0:
                logger.warning('No hist files found for '+variable+' historical '+model+' '+member+' '+grid)
                NO_HIST_FILES.append(f"{scenario}_{model}_{member}_{variable}")
                continue

            # load historical data
            hist_data, data_available = load_historical_data(hist_scen_files, variable, CONFIG)

            if not data_available:
                logger.warning(f"No historical data available for {variable} {scenario} {model} {member} {grid}, skipping delta calculation")
                NO_HIST_FILES.append(f"{scenario}_{model}_{member}_{variable}")
                continue

            # ens_files count is fixed for this (scenario, model, member, variable);
            # check it once before iterating future_periods
            if len(ens_files) != 2:
                logger.warning(f"Expected 2 future files for {scenario}_{model}_{member}_{variable} but found {len(ens_files)}: {ens_files}; skipping")
                WRONG_NUM_FILES.append(f"{scenario}_{model}_{member}_{variable}")
                continue

            deltas = {}
            wetfreq_deltas = {}   # only populated for pr
            for future_period in CONFIG.future_periods:
                st_yr = int(future_period[0].split('-')[0])
                st_mon = int(future_period[0].split('-')[1])
                st_day = int(future_period[0].split('-')[2])
                en_yr = int(future_period[1].split('-')[0])
                en_mon = int(future_period[1].split('-')[1])
                en_day = int(future_period[1].split('-')[2])
                # load future data
                ens_files_period = [f for f in ens_files if f.endswith(f"{st_yr}{st_mon:02d}{st_day:02d}_{en_yr}{en_mon:02d}31.nc")]
                future_data = load_temporal_data(ens_files_period, variable, datetime(st_yr, st_mon, st_day), datetime(en_yr, en_mon, en_day))

                # check for empty data before comparing shapes
                if len(future_data.shape)==0:
                    logger.warning(f'No future data found for {variable}_{scenario}_{model}_{member}')
                    continue

                # NEW: guard against an all-NaN (corrupt/missing) future field. CMIP6 pr is a
                # complete global field; a fully-NaN array means the download/regrid failed
                # (e.g. EC-Earth3 ssp585 2081-2100 r103/r104). Catch it HERE, because
                # quantile_mapping otherwise launders the NaNs into a constant 1.0 intensity
                # (np.where) + wetfreq = -hist_freq, which then slips past is_delta_valid.
                # When all future data is NaN, fin will be 0.0, so fin < 0.01 is True and the guard does fire
                fin = np.isfinite(future_data).mean()
                fin = float(fin.compute()) if hasattr(fin, "compute") else float(fin)
                if fin < 0.01:                      # essentially all NaN
                    logger.warning(f"Future field is {100*(1-fin):.0f}% NaN for "
                                f"{variable}_{scenario}_{model}_{member} ({st_yr}-{en_yr}); skipping")
                    INVALID_DELTA.append(f"{scenario}_{model}_{member}_{variable}_{st_yr}-{en_yr}")
                    continue

                if len(hist_data.shape)==0:
                    logger.warning(f'No historical data found for {variable}_{scenario}_{model}_{member}')
                    continue

                # Ensure grids/coords match (critical for correct xarray math)
                hist_data, future_data, ok = standardize_and_match_grids(hist_data, future_data)
                if not ok:
                    logger.warning(f"Could not match grids for {variable}_{scenario}_{model}_{member} ({st_yr}-{en_yr}); skipping")
                    continue

                # find average monthly difference between future and historical data DataArrays
                key = f'{st_yr}-{en_yr}'

                delta_, freq_delta = calc_delta(variable, hist_data, future_data, CONFIG.delta_method)

                valid, bad_count = is_delta_valid(delta_)
                if not valid:
                    total = int(delta_.size)
                    logger.warning(
                        f"Invalid delta (NaN/Inf): {bad_count}/{total} bad values for "
                        f"var={variable} scen={scenario} model={model} member={member} "
                        f"period={st_yr}-{en_yr} method={CONFIG.delta_method}; skipping period"
                    )
                    INVALID_DELTA.append(f"{scenario}_{model}_{member}_{variable}_{st_yr}-{en_yr}")
                    continue

                xname, yname = get_xy_coordname(delta_)

                if len(getattr(delta_, yname))==0:
                    logger.warning(f'lon 0 found for {variable}_{scenario}_{model}_{member}')
                    continue

                if len(getattr(delta_, xname))==0:
                    logger.warning(f'lat 0 found for {variable}_{scenario}_{model}_{member}')
                    continue

                deltas[key] = delta_
                if variable == 'pr':
                    wetfreq_deltas[key] = freq_delta
            
            if len(deltas)==0:
                logger.warning(f'No future data found for {variable}_{scenario}_{model}_{member}')
                continue

            ds_vars = {k: v for k, v in deltas.items()}
            if variable == 'pr':
                for k, v in wetfreq_deltas.items():
                    ds_vars[f"{k}_wetfreq"] = v      # e.g. "2041-2060_wetfreq"
            delta = xr.Dataset(ds_vars)
            delta.attrs["grid_label"] = grid

            # Save delta maps as netCDF
            if CONFIG.verbosity>1:
                logger.info(f"  Saving delta map for {scenario}_{model}_{member}_{variable}.nc")
            delta.to_netcdf(os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method, f"{scenario}_{model}_{member}_{variable}.nc"),
                            encoding={var: {"dtype": "float32", "zlib": True, "complevel": 4} for var in delta.data_vars})
        
            if (ee==0) and (CONFIG.make_delta_figures==True):
                for key, delta_ in deltas.items():                        
                    # Make a 2D field to plot
                    if CONFIG.delta_method in ["qqmap_bin_day", "qqmap_day", "qqmap_day2"]:
                        # plot January, nearest-to-median quantile
                        da_plot = delta_.sel(month=1).sel(quantile=0.5, method="nearest")
                        fig_name = f"{scenario}_{model}_{member}_{variable}_{key}_m01_q50_map.png"
                    else:
                        # mon_mean: plot month=1 as an example
                        da_plot = delta_.sel(month=1) if "month" in delta_.dims else delta_
                        fig_name = f"{scenario}_{model}_{member}_{variable}_{key}_m01_map.png"

                    da_plot.plot()
                    plt.savefig(os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method, "figures", fig_name), dpi=200, bbox_inches="tight")
                    plt.close("all")

            del deltas, delta, hist_data, future_data, wetfreq_deltas

        gc.collect()

        total = round(time.time() - ens_start, 2)
        if CONFIG.verbosity>2:
            logger.info(f"   All {len(members)} members for model {model} done in {total} secs.")

    total = round((time.time() - m_start) / 60, 2)
    if CONFIG.verbosity>1:
        logger.info(f"  All {len(models)} models for scenario {scenario} done in {total} mins.")

total = round((time.time() - scen_start) / 60, 2)
logger.info(f"All {len(scenarios)} scenarios for {variable} done in {total} mins.")
