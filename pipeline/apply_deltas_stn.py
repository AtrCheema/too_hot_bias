
"""
Applies delta values to a single catchment, runs HBV for:
1) historical baseline forcing
2) future delta-perturbed forcing for each scenario/model/member/period

and saves:
- *_data.csv  : qsim, forcing, qobs (if available)
- *_metrics.json : flow statistics from simulated q, plus change vs baseline for future runs

import os
import sys
cmd = f"{sys.executable} apply_deltas_stn.py Arcticnet__49052 test_run"
os.system(cmd)

Following files/directories are expected to be present for this file to run

gscad_path
    dataframes
        MSWEP_V280_Past.nc
        MSWX_V100_Past_Temp.nc
        MSWX_V100_Past_Tmin.nc
        MSWX_V100_Past_Tmax.nc
        static.csv
    stations
        stn
            boundary.json
calib_path
    stn
        MSWEP_V280_Past
            parameters.json

"""

import warnings
# Filter out UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys
import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import xarray as xr

# repo root (one dir up) holds config.py and the shared library modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apply_deltas_utils import (
    build_future_data,
    add_change_vs_baseline,
    compute_flow_stats,
    dry_spell_stats,
    low_flow_freq_deficit,
    compute_precip_stats,
    build_forcing_dataframe,
    calculate_pet,
    run_hbv_and_save,
    load_mean_deltas,
)

# precipitation (forcing) change metrics deltaed against the baseline forcing:
# ETCCDI heavy-precip indices + consecutive-dry-day (CDD) metrics.
PRECIP_COMPARE_KEYS = [
    "precip_dry_spell_mean", "precip_dry_spell_max", "n_precip_dry_spells",
    "cdd_annual_max_mean", "cdd_annual_max_max",
    "prcptot", "r10mm", "r20mm", "r95ptot", "r99ptot", "rx1day", "rx5day",
]

from ml_utils import Inference, run_lstm

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

dataframes_path = os.path.join(CONFIG.gscad_path, "dataframes")
out_stn_path = os.path.join(CONFIG.out_path, CONFIG.delta_method, stn)

if CONFIG.delta_method in ["mon_mean", "qqmap_bin_day", "qqmap_day", "qqmap_day2"]:
    mean_deltas_fpath = os.path.join(out_stn_path, f'mean_deltas.nc')
else:
    mean_deltas_fpath = os.path.join(out_stn_path, f'mean_deltas.json')

if not os.path.exists(mean_deltas_fpath):
    logger.info(f"mean deltas file not found exiting...")
    sys.exit(0)
mean_deltas = load_mean_deltas(mean_deltas_fpath, delta_method=CONFIG.delta_method)


qsim_path = os.path.join(out_stn_path, f"qsim.nc")
stats_parquet_fpath = os.path.join(out_stn_path, f"stats.parquet")
# check if the final output files already exist, if yes skip the station
if os.path.exists(qsim_path) and os.path.exists(stats_parquet_fpath) and not CONFIG.overwrite:
    logger.info(f" Output files already exist for {stn} and overwrite is False, skipping model runs and exitting.")
    sys.exit(0)

# get the observed/reference climate data
mswep_path = os.path.join(dataframes_path, 'daily_dataframes', 'MSWEP_V280_Past.nc')
obs_pr = xr.open_dataset(mswep_path)['var1'].sel(stations=stn).to_pandas().sort_index().rename('pcp')
obs_pr.loc[pd.Timestamp('2020-12-31')] = 0.0  # todo
obs_pr = obs_pr.sort_index()
temp_path = os.path.join(dataframes_path, 'daily_dataframes', 'MSWX_V100_Past_Temp.nc')
obs_temp = xr.open_dataset(temp_path)['var1'].sel(stations=stn).to_pandas().sort_index().rename('temp')
obs_temp.index = pd.to_datetime(obs_temp.index, format="%Y%j")
tmin_path = os.path.join(dataframes_path, 'daily_dataframes', 'MSWX_V100_Past_Tmin.nc')
obs_tmin = xr.open_dataset(tmin_path)['var1'].sel(stations=stn).to_pandas().sort_index().rename('tmin')
obs_tmin.index = pd.to_datetime(obs_tmin.index, format="%Y%j")
tmax_path = os.path.join(dataframes_path, 'daily_dataframes', 'MSWX_V100_Past_Tmax.nc')
obs_tmax = xr.open_dataset(tmax_path)['var1'].sel(stations=stn).to_pandas().sort_index().rename('tmax')
obs_tmax.index = pd.to_datetime(obs_tmax.index, format="%Y%j")

# we need area and lat for the station
static = pd.read_csv(os.path.join(dataframes_path, 'static.csv'), comment='#', index_col=0)
lat_val = float(static.loc[stn, 'lat'])
area_val = float(static.loc[stn, 'area'])  # km2; used for event-based flashiness (MTI)

stats_records = []

# -----------------------------
# 2) Historical baseline run
# -----------------------------
if CONFIG.verbosity > 2:
    logging.info(f"   Running historical baseline")

baseline_key = "historical_baseline"

hist_data = build_forcing_dataframe(
    pcp=obs_pr,
    tmin=obs_tmin,
    tmax=obs_tmax,
    temp=obs_temp,
    START_DATE=datetime.strptime(CONFIG.historical_period[0], "%Y-%m-%d") + timedelta(days=-365),  # add warmup period
    END_DATE=CONFIG.historical_period[1],
    qobs=None
)

hist_pet = calculate_pet(hist_data, lat=lat_val)
hist_data.insert(0, "pet", hist_pet, True)

# Prepare static features for LSTM inference
for col in static.columns:
    if static[col].dtype == 'object':
        static[col] = static[col].astype('category').cat.codes
        if CONFIG.verbosity > 3: 
            logging.info(f'    {col} is converted to category')
    else:
        static[col] = static[col].astype('float32')
lstm_static = static.loc[[stn]].copy()
lstm_static.index = ['-9999']  # Inference.make_inference uses stn='-9999' internally

# LSTM baseline -- updated GlobalLSTM1 (training job 47122937); 44 static
# features (q_0.8 added vs the old 43-feature run), best validation weights at
# epoch 57. scalers.json for this run must live alongside config.json in lstm_dir.
lstm_dir = '/ibex/user/abbaa0a/results/hydrodeepai/47122937_all_GlobalLSTM1_ddp'
lstm_weights = os.path.join(lstm_dir, 'weights', '57_0.0371_0.5004.pt')

# This training run renamed the dynamic-feature key in config.json from
# 'dynamic_features' to 'dyn_features'; Inference/run_lstm still expect
# 'dynamic_features', so load the config and alias the key before use.
with open(os.path.join(lstm_dir, 'config.json'), 'r') as f:
    lstm_config = json.load(f)
_dc = lstm_config['data_config']
if 'dynamic_features' not in _dc and 'dyn_features' in _dc:
    _dc['dynamic_features'] = _dc['dyn_features']

lstm_inference = Inference(
    path=lstm_dir,
    best_weights=lstm_weights,
    config=lstm_config,
    batch_size=2048,
    verbosity=CONFIG.verbosity
)
lstm_baseline_qsim = run_lstm(hist_data, lstm_inference, lstm_static, warmup_days=365)

lstm_baseline_stats = compute_flow_stats(
    lstm_baseline_qsim, area_km2=area_val, precip=hist_data['pcp'],
    lat=lat_val)

# HBV baseline run

with open(hbv_param_fpath, 'r') as f:
    params = json.load(f)

hbv_baseline_qsim, baseline_stats = run_hbv_and_save(
    data=hist_data,
    parameters=params,
    warmup_days=365,
    baseline_stats=None,
    area_km2=area_val,
    lat=lat_val,
    metadata={
        "station": stn,
        "run_type": "historical_baseline",
        "start_date": CONFIG.historical_period[0],
        "end_date": CONFIG.historical_period[1]
    },
)

# Fixed low-flow thresholds for dry-spell detection: each engine uses its own
# baseline q05, reused for that engine's baseline + all future runs (Option 2),
# so a drier future shows up as longer/more frequent dry spells.
hbv_dry_threshold = baseline_stats.get("q05")
lstm_dry_threshold = lstm_baseline_stats.get("q05")

# Fixed high-flow thresholds (9 x baseline median), computed by the baseline
# compute_flow_stats and reused for that engine's future runs, so a flashier
# future shows up as more/longer high-flow events (same logic as the dry
# thresholds above).
hbv_high_flow_threshold = baseline_stats.get("high_q_threshold")
lstm_high_flow_threshold = lstm_baseline_stats.get("high_q_threshold")

# backfill dry-spell stats AND the marginal low-flow frequency/severity pair
# (low_q_freq, low_q_deficit) onto the baselines (their q05 wasn't known when
# compute_flow_stats first ran for them). Same fixed threshold for both
# families, exactly as on the future runs.
if hbv_dry_threshold is not None:
    baseline_stats.update(dry_spell_stats(hbv_baseline_qsim, hbv_dry_threshold))
    baseline_stats.update(low_flow_freq_deficit(hbv_baseline_qsim, hbv_dry_threshold))
if lstm_dry_threshold is not None:
    lstm_baseline_stats.update(dry_spell_stats(lstm_baseline_qsim, lstm_dry_threshold))
    lstm_baseline_stats.update(low_flow_freq_deficit(lstm_baseline_qsim, lstm_dry_threshold))

# Precipitation (forcing) stats on the baseline forcing -- ETCCDI heavy-precip
# indices + consecutive-dry-day metrics; an engine-independent climate metric,
# post-warmup window (.iloc[365:]) to match the flow-stats window in
# run_hbv_and_save. The wet-day 95th/99th percentiles are computed here and
# forwarded to the future runs so R95p/R99p share a fixed base period.
baseline_precip_stats = compute_precip_stats(hist_data['pcp'].iloc[365:])
base_p95 = baseline_precip_stats.get("p95_ref")
base_p99 = baseline_precip_stats.get("p99_ref")

# baseline -> one row per engine in the same table
for engine, es in (("hbv", baseline_stats), ("lstm", lstm_baseline_stats)):
    row = {
        "scenario": "baseline",
        "model":    "baseline",
        "member":   "baseline",
        "period":   "historical",
        "engine":   engine,
        **{k: v for k, v in es.items() if k not in ("metadata", "monthly_mean_qsim")},
    }
    for mm, val in (es.get("monthly_mean_qsim") or {}).items():
        row[f"monthly_mean_{mm}"] = val
    row.update(baseline_precip_stats)   # same precip metrics on both engine rows
    stats_records.append(row)

# -----------------------------
# 3) Future delta runs
# -----------------------------
# make future climate series and run the model

all_hbv_qsim = {}
all_lstm_qsim = {}

idx = 0
for scen_mod_ee, deltas in mean_deltas.items():

    if CONFIG.verbosity>1:
        logging.info(f"  Calculating projections for {scen_mod_ee} ({idx+1}/{len(mean_deltas)})")
    
    scenario, model, member = scen_mod_ee.split('_', 2)

    for fut_per in CONFIG.future_periods:

        st_yr, en_yr = fut_per[0].split('-')[0], fut_per[1].split('-')[0]

        key = scen_mod_ee
       
        try:
            fut_data = build_future_data(
            deltas, fut_per, obs_pr, obs_tmin,
            obs_tmax, obs_temp, lat_val, CONFIG
        )
        except Exception as e:
            logging.error(f"Error building future data for {key} period {fut_per}: {e}")
            continue

        # Precipitation (forcing) stats for this realization+period -- ETCCDI
        # heavy-precip + CDD, engine-independent, with change vs the baseline
        # forcing. R95p/R99p use the baseline wet-day percentiles (base_p95/99).
        fut_precip_stats = compute_precip_stats(
            fut_data['pcp'].iloc[365:], p95_ref=base_p95, p99_ref=base_p99)
        fut_precip_stats = add_change_vs_baseline(
            fut_precip_stats, baseline_precip_stats, compare_keys=PRECIP_COMPARE_KEYS)

        qsim_series, hbv_stats = run_hbv_and_save(
            data=fut_data,
            parameters=params,
            warmup_days=365,
            baseline_stats=baseline_stats,
            dry_threshold=hbv_dry_threshold,
            area_km2=area_val,
            high_flow_threshold=hbv_high_flow_threshold,
            lat=lat_val,
            metadata={
                "station": stn,
                "run_type": "future_delta",
                "scenario": scenario,
                "model": model,
                "member": member,
                "future_period": fut_per,
                "start_date": CONFIG.historical_period[0],
                "end_date": CONFIG.historical_period[1]
            },
        )

        all_hbv_qsim[(scenario, model, member, f"{st_yr}-{en_yr}")] = qsim_series

        # LSTM projection
        lstm_qsim = run_lstm(fut_data, lstm_inference, lstm_static, warmup_days=365)
        lstm_stats = compute_flow_stats(
            lstm_qsim,
            dry_threshold=lstm_dry_threshold,
            area_km2=area_val,
            precip=fut_data['pcp'],
            high_flow_threshold=lstm_high_flow_threshold,
            lat=lat_val,
        )
        lstm_stats = add_change_vs_baseline(lstm_stats, lstm_baseline_stats)
        lstm_stats["metadata"] = {
            "station": stn,
            "run_type": "future_delta_lstm",
            "scenario": scenario,
            "model": model,
            "member": member,
            "future_period": fut_per,
        }

        for engine, es in (("hbv", hbv_stats), ("lstm", lstm_stats)):
            row = {
                "scenario": scenario,
                "model":    model,
                "member":   member,
                "period":   f"{st_yr}-{en_yr}",
                "engine":   engine,
                **{k: v for k, v in es.items() if k not in ("metadata", "monthly_mean_qsim")},
            }
            for mm, val in (es.get("monthly_mean_qsim") or {}).items():
                row[f"monthly_mean_{mm}"] = val
            row.update(fut_precip_stats)   # same precip metrics (+change) on both engine rows
            stats_records.append(row)

        all_lstm_qsim[(scenario, model, member, f"{st_yr}-{en_yr}")] = lstm_qsim

        if CONFIG.verbosity ==2 and idx % 50 == 0 and fut_per == '2081-2100':
            logging.info(f"{idx}/{len(mean_deltas)} Completed for {key}.")
        elif CONFIG.verbosity > 2 and idx % 10 == 0 and fut_per == '2081-2100':
            logging.info(f"{idx}/{len(mean_deltas)} Completed for {key}.")

    idx += 1

all_hbv_qsim[("baseline", "baseline", "baseline", "historical")] = hbv_baseline_qsim
all_lstm_qsim[("baseline", "baseline", "baseline", "historical")] = lstm_baseline_qsim

hbv_records = []
lstm_records = []

for (scen, mod, mem, period), qs in all_hbv_qsim.items():
    da = xr.DataArray(qs.values, dims=["time"], coords={"time": qs.index})
    da = da.expand_dims({"scenario": [scen], "model": [mod], "member": [mem], "period": [period]})
    hbv_records.append(da)

for (scen, mod, mem, period), qs in all_lstm_qsim.items():
    da = xr.DataArray(qs.values, dims=["time"], coords={"time": qs.index})
    da = da.expand_dims({"scenario": [scen], "model": [mod], "member": [mem], "period": [period]})
    lstm_records.append(da)

hbv_combined = xr.combine_by_coords(hbv_records)
lstm_combined = xr.combine_by_coords(lstm_records)

ds = xr.Dataset({
    "hbv_qsim": hbv_combined,
    "lstm_qsim": lstm_combined,
})
ds.attrs = {"station": stn, "units": "mm/day", "created": str(pd.Timestamp.now())}
ds.to_netcdf(
    qsim_path,
    encoding={
        "hbv_qsim": {"dtype": "float32", "zlib": True, "complevel": 4},
        "lstm_qsim": {"dtype": "float32", "zlib": True, "complevel": 4},
    }
)

stats_df = pd.DataFrame(stats_records)
stats_df = stats_df.sort_values(["scenario", "model", "member", "period", "engine"]).reset_index(drop=True)
stats_df.to_parquet(stats_parquet_fpath)

if CONFIG.verbosity > 0:
    logging.info(f"Done for {stn}. Baseline + future runs saved.")
