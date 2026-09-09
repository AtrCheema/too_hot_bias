
__all__ = ['CONFIG']

# ---------------------------------------------------------------------------
# Paths -- EDIT THESE BEFORE RUNNING ANYTHING
# ---------------------------------------------------------------------------
# All six are placeholders. Set them to the locations on your own machine.
# The three data paths point at inputs that are not distributed with this
# repository; see the "Inputs not included" section of readme.md.
#
# Large volumes are involved: the raw CMIP6 archive and the derived deltas each
# run to several terabytes, so put them on scratch or project storage rather
# than in a home directory.

# Raw CMIP6 daily fields (pr, tasmin, tasmax), as retrieved from ESGF or the
# Google Cloud CMIP6 archive.
cmip6_data = '/path/to/cmip6_data'

# Gridded climate-change deltas written by pipeline/calc_deltas_2d.py.
cmip6_deltas = '/path/to/cmip6_deltas'

# Observed streamflow database, catchment boundaries and static attributes.
gscad_path = '/path/to/streamflow_database'

# Calibrated per-catchment HBV parameter sets from the prior global calibration.
calib_path = '/path/to/hbv_calibration_results'

# Root for everything this project writes: per-catchment qsim.nc and
# mean_deltas.nc, and the fig_data parquet caches the figure scripts read.
out_path = '/path/to/outputs'

# Scratch space for intermediate files during delta calculation/application.
tmp_path = '/path/to/tmp'

config = {

    # common settings (apply to data download, delta calculation and delta application)
    'cmip6_data': cmip6_data,      # used in data download and delta calculation
    'cmip6_deltas': cmip6_deltas,  # used in delta calculationa and delta application
    'tmp_path': tmp_path,    # used in data download (esgf only) and delta application
    # overwrite can have different implications for different steps
    # for delta application, if True, then for each station even if the catchment mean deltas (mean_deltas.nc) are present, they will be recalculated
    # also for delta application, if overwrite is True and the final output files (qsim.nc and stats.parquet) are already present, they will be overwritten; if overwrite is False and the final output files are already present, then the station will be skipped for HBV runs assuming that the outputs are already there from a previous run
    'overwrite': False,
    'verbosity': 3,
    'historical_period': ('2001-01-01', '2020-12-31'),  # also callled reference/baseline period
    'scenarios': [
        #'ssp119',
        'ssp126',
        'ssp245', 
        'ssp370',
        #'ssp434', 
        #'ssp460',
        'ssp585'
        ],
    "variables": [
        #"tas", 
        #"rsds", #"rsdt", "rsut", "rlut",
        #"hurs",
        "tasmin",
        "tasmax",
        "pr",
        #"ps", # surface pressure
        #"prw", # precipitation water
        #"evspsbl",
        #"hus",
        #"ua",
        #"va",
        ],

    'models': [
        ## 20 fron Asenjan's list
        'CanESM5',
        'NESM3',
        'IPSL-CM6A-LR',
        'EC-Earth3-Veg',
        'EC-Earth3',
        'ACCESS-CM2',
        'ACCESS-ESM1-5',
        'GFDL-CM4',  # there are two versions with gr1 and gr2 grid labels
        'MRI-ESM2-0',
        'MPI-ESM1-2-LR',
        'BCC-CSM2-MR',
        'MPI-ESM1-2-HR',  # requires > 40 GB for qqmap_day method
        'FGOALS-g3',
        'GFDL-ESM4',
        'NorESM2-LM',
        'MIROC6',
        'NorESM2-MM',
        'INM-CM5-0',
        'INM-CM4-8',

        ## additional models from Beck et al., 2023
        'CIESM',
        'AWI-CM-1-1-MR',
        'AWI-ESM-1-1-LR',
        'BCC-ESM1',
        'CAMS-CSM1-0',
        'CAS-ESM2-0',
        'CESM2',
        'CESM2-FV2',
        'CESM2-WACCM',
        'CESM2-WACCM-FV2',
        'CMCC-CM2-HR4',
        'CMCC-CM2-SR5',
        'CMCC-ESM2',
        'CNRM-CM6-1',
        'CNRM-CM6-1-HR',
        'CNRM-ESM2-1',
        'CanESM5-1',
        'CanESM5-CanOE',
        'E3SM-1-0',
        'E3SM-1-1',
        'E3SM-1-1-ECA',
        'E3SM-2-0',
        'EC-Earth3-AerChem',
        'EC-Earth3-CC',
        'EC-Earth3-Veg-LR',
        'FGOALS-f3-L',
        'FIO-ESM-2-0',
        'GISS-E2-1-G',
        'GISS-E2-1-G-CC',
        'GISS-E2-1-H',
        'GISS-E2-2-H',
        'HadGEM3-GC31-LL',
        'HadGEM3-GC31-MM',
        'IITM-ESM',
        'IPSL-CM5A2-INCA',
        'IPSL-CM6A-LR-INCA',
        'KACE-1-0-G',
        'KIOST-ESM',
        'MCM-UA-1-0',
        'MIROC-ES2H',
        'MIROC-ES2L',
        'MPI-ESM-1-2-HAM',
        'NorCPM1',
        'SAM0-UNICON',
        'TaiESM1',
        'UKESM1-0-LL',
        'UKESM1-1-LL',

        ## additional models which are not in PANGEO
        'BCC-CSM2-HR',
        'CMCC-CM2-VHR4',
        'EC-Earth3-HR',
        'EC-Earth3-LR',
        'EC-Earth3P',
        'EC-Earth3P-HR',
        'ECMWF-IFS-HR',
        'ECMWF-IFS-LR',
        'ECMWF-IFS-MR',
        'FGOALS-f3-H',
        'GFDL-AM4',
        'GFDL-CM4C192',
        'GISS-E2-2-G',
        'GISS-E3-G',
        'INM-CM5-H',
        'IPSL-CM6A-ATM-HR',
        'IPSL-CM6A-ATM-ICO-HR',
        'IPSL-CM6A-ATM-ICO-LR',
        'IPSL-CM6A-ATM-ICO-MR',
        'IPSL-CM6A-ATM-ICO-VHR',
        'IPSL-CM6A-ATM-LR-REPROBUS',
        'IPSL-CM6A-MR1',
        'MPI-ESM1-2-XR',
        'MRI-AGCM3-2-H',
        'MRI-AGCM3-2-S',   ## very high res
        'UKESM1-ice-LL',

        'E3SM-2-0-NARRM',
        'E3SM-2-1',
        'ICON-ESM-LR',
        ],

    'future_periods': [#('2015-01-01', '2020-12-31'),  # use this only to download data for ssp585 and timestep is 'day'
                       ('2041-01-01', '2060-12-31'), 
                       ('2081-01-01', '2100-12-31')
                       ],

    'timestep': 'day',  # Amon is for Monthly and 'day' is for daily


    # *** DATA DOWNLOAD SETTINGS ***
    'data_source': 'esgf',          # 'pangeo' | 'esgf'
    'esgf_search_node': 'https://esgf-node.llnl.gov/esg-search',
    'esgf_distrib': True,             # federate across ESGF nodes
    'esgf_latest_only': True,         # keep only latest version per dataset
    'esgf_prefer_replica': False,     # primary copies first, replicas as fallback

    "grid_label": ["gn", "gr", "gr1", "gr2"],  # order of preference


    # *** DELTA CALCULATION SETTINGS ***
    'make_delta_figures': False,
    # downscaling method 
    # scaler : a single value delta applied to all time steps
    # mon_mean : monthly mean delta applied to all days in a month 
    #            (e.g. all days in Jan 2040 get the same delta based on the difference between historical and future monthly means for Jan)
    #            The shape of delta map will be (12, lat, lon) for each relaization
    # qqmap_bin_day : quantile-quantile mapping on daily data but with bins to speed up the calculation;
    #                the shape of delta map will be (12, n_bins, lat, lon) for each realization
    # qqmap_day : daily quantile-quantile mapping; intensity from quantile_mapping
    #             (NOTE: has the dry-tail inflation artefact)
    # qqmap_day2 : daily quantile-quantile mapping using quantile_mapping2; wet-day-only
    #              intensity, fixes both the dry-tail inflation and the freq/intensity
    #              entanglement. Same on-disk structure as qqmap_day (sample axis).
    'delta_method': 'qqmap_day2',   # qqmap_day2, qqmap_day, mon_mean, qqmap_bin_day, scalar


    # *** DELTA APPLICATION SETTINGS ***
    # max number of members to process per scenario/model combination; set to 
    # None to process all members (note: some combinations have > 100 members 
    # which can be very time consuming to apply deltas to, especially with qqmap_day method)
    'max_members_per_model': 20,

    'calib_path': calib_path,      # used in delta application
    'gscad_path': gscad_path,      # used in delta application
    'out_path': out_path,          # used in delta application
}


# Following models do not have daily data for (pr, tasmax and tas)
MODELS_WITH_NO_DAILY_DATA = [
    'AWI-CM-1-1-MR', 'AWI-ESM-1-1-LR', 'BCC-ESM1', 'CAMS-CSM1-0', 'CAS-ESM2-0', 'CESM2', 'CESM2-FV2', 
    'CESM2-WACCM', 'CESM2-WACCM-FV2', 'CIESM', 'CMCC-CM2-HR4', 'CanESM5-CanOE', 'E3SM-1-1', 'E3SM-1-1-ECA', 
    'E3SM-2-0', 'EC-Earth3-AerChem', 'FGOALS-f3-L', 'FIO-ESM-2-0', 'GISS-E2-1-G-CC', 'GISS-E2-1-H', 
    'GISS-E2-2-H', 'IITM-ESM', 'IPSL-CM5A2-INCA', 'IPSL-CM6A-LR-INCA', 'MCM-UA-1-0', 'MPI-ESM-1-2-HAM', 
    'NorCPM1', 'SAM0-UNICON', 'UKESM1-1-LL']

class CONF:
    def __init__(self) -> None:
        for k, v in config.items():
            setattr(self, k, v)
        
        if self.timestep == 'day':
            for m in MODELS_WITH_NO_DAILY_DATA:
                if m in self.models:
                    if self.verbosity>3: print(f"WARNING: model {m} is in the config.models list but it has no daily data, removing it from the list ...")
                    self.models.remove(m)


CONFIG = CONF()
