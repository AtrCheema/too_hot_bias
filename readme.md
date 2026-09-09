# Influence of too-hot climate models on hydrological climate-change assessment

Analysis code for a global study of how over-sensitive CMIP6 climate models bias
projected streamflow and design-flood change across roughly 20,000 catchments.

Climate models are weighted by a Gaussian kernel in transient climate response
(TCR) whose central 90% interval is the IPCC AR6 *very likely* range of
1.2-2.4 degC. The weighted ensemble is compared against the unweighted
one-model-one-vote ensemble. Streamflow is simulated with two structurally
independent hydrological models, the conceptual HBV and a global LSTM network.

## Workflow

CMIP6 daily precipitation and minimum/maximum temperature must be obtained
separately from the Earth System Grid Federation or the Google Cloud CMIP6
archive. The download script is not part of this repository.

1. `pipeline/calc_deltas_2d.py` computes the gridded quantile-wise change
   between the reference period (2001-2020) and each future period.
2. `pipeline/apply_deltas.py` with job type `catch_avg_deltas` averages those
   changes over each catchment boundary, calling
   `pipeline/catch_avg_deltas_stn.py` and writing `mean_deltas.nc`.
3. `pipeline/apply_deltas.py` with job type `apply_deltas` perturbs the observed
   forcing, runs HBV and the LSTM, and writes `qsim.nc` per catchment through
   `pipeline/apply_deltas_stn.py`.
4. `paper/figures/scripts/build_streamflow_cache.py` collects the per-catchment
   output into the parquet caches every figure script reads.

Run every script from the repository root, for example

    python paper/figures/scripts/fig02_precip_change.py

The scripts add the repository root to `sys.path` themselves, so imports resolve
regardless of the working directory, but figure and table output paths are
resolved relative to the root.

## Figures and analysis

`paper/figures/scripts/run_tcr_weighting.py` is the driver for the weighting
treatment used in the manuscript. It selects the treatment in process through
the `ensemble_method` API and then runs any of its steps, for example

    python paper/figures/scripts/run_tcr_weighting.py headline variant=vlike
    python paper/figures/scripts/run_tcr_weighting.py signif sigarg=flow variant=vlike

Its module docstring lists every step. The individual generators can also be run
on their own:

| Script | Produces |
| --- | --- |
| `fig01_model_screening.py` | Model universe, TCR/ECS scatter, model table |
| `model_weights.py`, `fig_model_weights.py` | Gaussian TCR weights and their figure |
| `fig02_precip_change.py` | Projected precipitation and temperature change |
| `hydro_indices_change.py` | Streamflow index change, bias ranking, spread |
| `gev_fdr_dipole.py` | GEV return levels, false-discovery control, snow dipole |
| `analyze_full_vs_reduced_significance.py` | Per-catchment significance, signal-to-noise |
| `weight_permutation_null.py`, `local_vs_aggregate_null.py` | Shuffled-weight null tests |
| `bootstrap_random_removal.py` | Random-removal null test |
| `baseline.py`, `fig03.py` | Present-day baseline and hydrological model performance |

`ensemble_method.py` is the single switch between the unweighted, screened and
weighted ensembles. `realization_blocklist.py` is the single source of truth for
the excluded corrupt CMIP6 realizations.

## Layout

    config.py                  paths, scenarios, climate model list, delta method
    utils.py                   CMIP6 catalogue access and delta calculation
    apply_deltas_utils.py      catchment averaging, delta application, flow statistics
    ml_utils.py                LSTM inference
    models/hbv.py              HBV rainfall-runoff model (numba kernel)
    pipeline/                  the numbered workflow above
    paper/figures/scripts/     figure and analysis generators
    data/                      calibrated station list

## Inputs not included

The repository holds analysis code only. It does not contain the observed
streamflow database and catchment boundaries, the calibrated HBV parameter sets,
the trained LSTM weights, or the MSWEP v2.8 and MSWX v1.0 forcing. Paths to all
of these are set in `config.py`, which must be edited for a new machine. HBV
calibration and LSTM training were carried out in a separate project and their
code is not part of this repository.

## Licence

MIT. See `LICENSE`.
