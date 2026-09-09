"""
Calculates catchment-averaged deltas (realizations) for each scenario-model-member combination and
saves them in a json or netcdf file for later use in delta application. The deltas are calculated
as the difference between future and historical climate model simulations, averaged over the catchment area

import os
import sys
cmd = f"{sys.executable} catch_avg_deltas_stn.py Arcticnet__49052 test_run"
os.system(cmd)

Following files/directories are expected to be present for this file to run

gscad_path
    stations
        stn
            boundary.json
"""

import warnings
# Filter out UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
import glob
import json
import logging
from collections import defaultdict

import numpy as np

# repo root (one dir up) holds config.py and the shared library modules
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apply_deltas_utils import (
    save_mean_deltas_as_nc,
    save_mean_deltas_qqmap_day_as_nc,
    all_files_have_all_vars,
    get_catch_avg_deltas,
    load_shapefile,
    load_mean_deltas,
)

from config import CONFIG

logging.basicConfig(#filename=f'delta_{variable}_{jobid}.log', filemode='w',
                    stream=sys.stdout,
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# stn = "Arcticnet__49052" #sys.argv[1]
# jobid = "test_job" # sys.argv[2]
stn = sys.argv[1]
jobid = sys.argv[2]


if CONFIG.verbosity > 0:
    logging.info(f"Delta application for {stn} with jobid {jobid}")

# It is possible that HBV/MSWEP_V280_Past was not calibrated for this station
hbv_param_fpath = os.path.join(CONFIG.calib_path, stn, 'MSWEP_V280_Past', 'hbv', 'parameters.json')
if not os.path.exists(hbv_param_fpath):
    logging.info(f"HBV parameters file not found for station {stn} at {hbv_param_fpath}. Skipping ...")
    sys.exit(1)

os.makedirs(CONFIG.tmp_path, exist_ok=True)

dataframes_path = os.path.join(CONFIG.gscad_path, "dataframes")
stns_path = os.path.join(CONFIG.gscad_path, "stations")

stn_path = os.path.join(stns_path, stn)

stn_boundary = os.path.join(stn_path, 'boundary.shp')
if not os.path.exists(stn_boundary):
    logging.error(f"Boundary shapefile not found for station {stn} at expected path {stn_boundary}. Exiting.")
    sys.exit(1)
catchment = load_shapefile(stn_boundary)

out_stn_path = os.path.join(CONFIG.out_path, CONFIG.delta_method, stn)
os.makedirs(out_stn_path, exist_ok=True)

# -----------------------------
# Build catchment-mean deltas
# -----------------------------

mean_deltas = {}
if CONFIG.delta_method in ["mon_mean", "qqmap_bin_day", "qqmap_day", "qqmap_day2"]:
    mean_deltas_fpath = os.path.join(out_stn_path, f'mean_deltas.nc')
else:
    mean_deltas_fpath = os.path.join(out_stn_path, f'mean_deltas.json')

if os.path.exists(mean_deltas_fpath) and not CONFIG.overwrite:
    logger.info(f" Mean deltas file already exists at {mean_deltas_fpath} and overwrite is False, skipping delta calculation.")
    mean_deltas = load_mean_deltas(mean_deltas_fpath, delta_method=CONFIG.delta_method)


existing_per_model = defaultdict(int)
for rk in mean_deltas.keys():
    scen, mod, _ = rk.split("_", 2)
    existing_per_model[(scen, mod)] += 1

new_count = 0

scenarios = [
        'ssp119',
        'ssp126',
        'ssp245', 
        'ssp370',
        #'ssp434', 
        #'ssp460',
        'ssp585'
        ]
#else:
for scenario in scenarios:

    # the files are named as scenario_model_ensemble_variable.nc
    # get all files for the scenario
    scen_files = glob.glob(os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method, f'{scenario}_*.nc'))

    # now get names of all models
    models = sorted(np.unique([os.path.basename(x).split('_')[1] for x in scen_files]).tolist())

    if CONFIG.verbosity>1: logging.info(f"  Processing {scenario} {len(models)} models")

    for mm, model in enumerate(models):

        if model not in CONFIG.models:
            continue

        # get all files for the model
        model_files = glob.glob(os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method, f'{scenario}_{model}_*.nc'))
        members = sorted(np.unique([os.path.basename(x).split('_')[2] for x in model_files]).tolist())

        if CONFIG.verbosity>1: logging.info(f"  model: {model} ({mm}/{len(models)}) {len(members)} members")

        #scen_mod_members = 0
        scen_mod_members = existing_per_model[(scenario, model)]

        for ee, member in enumerate(members):

            real_key = f"{scenario}_{model}_{member}"
            if real_key in mean_deltas:
                if CONFIG.verbosity>2: logging.info(f"   already have deltas for {real_key} from previous run, skipping ...")
                continue

            # get all files for the member
            member_files = glob.glob(os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method, f'{scenario}_{model}_{member}_*.nc'))
            variables = sorted(np.unique([os.path.basename(x).split('_')[3].split('.')[0] for x in member_files]).tolist())

            if all([var in variables for var in ['pr', 'tasmin', 'tasmax']]):

                if all_files_have_all_vars(member_files, ['2041-2060', '2081-2100'], max(0, CONFIG.verbosity-2)):

                    if scen_mod_members > CONFIG.max_members_per_model:
                        if CONFIG.verbosity>2:
                            logging.info(f"   reached max_members_per_model={CONFIG.max_members_per_model} for {scenario} {model}, skipping remaining members")
                        break

                    if CONFIG.verbosity>2: 
                        logging.info(f'   {ee+1}/{len(members)}   calculating catchment-averaged deltas for {scenario} {model} {member}')

                    mean_deltas[f"{scenario}_{model}_{member}"] = get_catch_avg_deltas(
                        os.path.join(CONFIG.cmip6_deltas, CONFIG.delta_method),
                        key=(scenario, model, member), 
                        catchment=catchment, 
                        tmp_path=CONFIG.tmp_path, 
                        jobid=jobid,
                        delta_method=CONFIG.delta_method)
                    scen_mod_members += 1
                    existing_per_model[(scenario, model)] += 1
                    new_count += 1

                else:
                    if CONFIG.verbosity>2:
                        logging.info(f'   {ee+1}/{len(members)}   skipping {member} since file does not have all variables for all future periods')
            else:
                if CONFIG.verbosity>2:
                    missing_vars = [var for var in ['pr', 'tasmin', 'tasmax'] if var not in variables]
                    logging.info(f'   {ee+1}/{len(members)}   skipping {member} since missing variables: {missing_vars}')

if CONFIG.verbosity>1: logging.info(f"  Finished creating mean deltas. Total realizations : {len(mean_deltas)}.")

if len(mean_deltas) == 0:
    logging.info("No mean deltas found, exiting.")
    sys.exit(0)

if new_count > 0:
    tmp = mean_deltas_fpath + f".tmp_{jobid}"
    if CONFIG.delta_method == "qqmap_bin_day":
        save_mean_deltas_as_nc(mean_deltas, mean_deltas_fpath)
    elif CONFIG.delta_method in ("qqmap_day", "qqmap_day2"):
        save_mean_deltas_qqmap_day_as_nc(mean_deltas, tmp)
        os.replace(tmp, mean_deltas_fpath)
    else:
        # save the mean_deltas
        with open(mean_deltas_fpath, 'w') as f:
            json.dump(mean_deltas, f, indent=4, sort_keys=True)

logger.info(f"Finished processing mean deltas for station {stn}. ")
logger.info(f"Total new mean deltas calculated: {new_count}. ")
logger.info(f"Total mean deltas available: {len(mean_deltas)}. Saved at {mean_deltas_fpath}.")
