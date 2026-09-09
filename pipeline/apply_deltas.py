
import os
import sys
import logging

import pandas as pd
import xarray as xr

# repo root (one dir up) holds config.py and the shared library modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG

# which per-station script to run for each station. Maps the job_type passed on
# the command line to (script to run, output file used to decide if a station can
# be skipped because it is already done).
JOB_TYPES = {
    'apply_deltas':     'apply_deltas_stn.py',      # produces qsim.nc
    'catch_avg_deltas': 'catch_avg_deltas_stn.py',  # produces mean_deltas.nc
}

job_type = sys.argv[1]  # 'apply_deltas' or 'catch_avg_deltas'
job_id = sys.argv[2] #'test'
memory = sys.argv[3]
cpus = sys.argv[4]
current_chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 1
total_chunks = int(sys.argv[6]) if len(sys.argv) > 6 else 1

if job_type not in JOB_TYPES:
    raise ValueError(f"Unknown job_type {job_type!r}, expected one of {list(JOB_TYPES)}")
# resolved next to this file so the os.system() launch below works regardless of cwd
stn_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), JOB_TYPES[job_type])


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info(f"{memory} memory, job id: {job_id}, cpus: {cpus}, job type: {job_type} ({stn_script})")
logger.info(f"Processing chunk {current_chunk} of {total_chunks}")

# list of HBV-calibrated ("done") stations to process. (Previously read from
# CONFIG.calib_path/parameters.nc, but that file is not always present and its
# absence kills every array task at startup.) The station list lives in this CSV
# (single 'station' column) in the repo root; resolved relative to this file so
# it works regardless of the job's cwd.
done_stations_fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'data', 'hbv_MSWEP_V280_Past_done_stations.csv')
stations = pd.read_csv(done_stations_fpath)['station'].tolist()

# Split stations into chunks
if total_chunks > 1:
    num_stations = len(stations)
    start_idx = (current_chunk - 1) * num_stations // total_chunks
    end_idx = current_chunk * num_stations // total_chunks
    stations = stations[start_idx:end_idx]
    logger.info(f"Processing stations {start_idx} to {end_idx} out of {num_stations}")
else:
    logger.info(f"Processing all {len(stations)} stations")

for idx, stn in enumerate(stations):

    stn_dir = os.path.join(CONFIG.out_path, CONFIG.delta_method, stn)
    if not os.path.exists(stn_dir):
        try:
            os.makedirs(stn_dir, exist_ok=True)
        except FileExistsError:
            pass

    # the output file that, if already present, means this station is done for
    # the requested job_type and can be skipped.
    if job_type == 'apply_deltas':
        done_file = os.path.join(stn_dir, 'qsim.nc')
    else:  # catch_avg_deltas
        if CONFIG.delta_method in ["mon_mean", "qqmap_bin_day", "qqmap_day", "qqmap_day2"]:
            done_file = os.path.join(stn_dir, 'mean_deltas.nc')
        else:
            done_file = os.path.join(stn_dir, 'mean_deltas.json')

    if os.path.exists(done_file):
        if CONFIG.verbosity>1: logger.info(f"Skipping {stn} ({idx+1}/{len(stations)})")
        continue

    logger.info(f"Processing {stn} ({idx+1}/{len(stations)})")

    #                             1      2
    cmd = f"{sys.executable} {stn_script} {stn} {job_id}"

    os.system(cmd)

    # if idx>1:
    #     break

logger.info("All done!")
