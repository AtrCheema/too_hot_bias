"""
scenario_gradient.py
=================
Full / reduced ensemble means AND medians of the projected change, for every
(scenario, period, engine) cell, on the phase0_headline.py aggregation
(member -> model mean -> one-model-one-vote ensemble mean per station; then
mean / median across stations). Backs the main4.tex sentences on the scenario
gradient of the AnnMax bias ("+0.21 pp under SSP1-2.6 ... -7.20 pp under
SSP5-8.5"; mid-century -0.60/-0.41, medians +0.15/+0.05) and the Discussion
sentence on the mid-century Q05 median bias.

The treatment follows ensemble_method.py: under METHOD="removal" the second
ensemble is the unweighted mean over the retained models (SCREEN), under
METHOD="weighting" it is the weighted mean over every model present. Output
is tagged:
    fig_data/scenario_gradient_v1<tag>.csv
Run only via slurm (reads the 10.7 GB stats_change.parquet with row-group
filters, one cell at a time).
"""
import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
sys.path.insert(0, _THIS)

from config import CONFIG                                      # noqa: E402
import ensemble_method as em                                   # noqa: E402

CACHE_DIR = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")
CLIMATE_CACHE = os.path.join(CACHE_DIR, "fig02_change.parquet")
FLOW_CACHE    = os.path.join(CACHE_DIR, "stats_change.parquet")
OUT_CSV = em.tag(os.path.join(CACHE_DIR, "scenario_gradient_v1.csv"))

SCENARIOS = ["ssp126", "ssp245", "ssp370", "ssp585"]
PERIODS   = ["2041-2060", "2081-2100"]
# (name, dataset, column, engine, scale100)
METRICS = [
    ("warming_degC",    "climate", "temp_change_degC",         None,   False),
    ("precip_mean_pct", "climate", "precip_change_pct",        None,   False),
    ("annmax_pct",      "flow",    "delta_rel_mean_annual_max", "hbv",  True),
    ("rl20_pct",        "flow",    "delta_rel_flood_q20",       "hbv",  True),
    ("qmean_pct",       "flow",    "delta_rel_qmean",           "hbv",  True),
    ("q05_pct",         "flow",    "delta_rel_q05",             "hbv",  True),
    ("annmax_pct",      "flow",    "delta_rel_mean_annual_max", "lstm", True),
    ("rl20_pct",        "flow",    "delta_rel_flood_q20",       "lstm", True),
    ("qmean_pct",       "flow",    "delta_rel_qmean",           "lstm", True),
    ("q05_pct",         "flow",    "delta_rel_q05",             "lstm", True),
]


def load_slice(dataset, column, engine, scen, per, universe):
    cache = FLOW_CACHE if dataset == "flow" else CLIMATE_CACHE
    id_cols = ["station", "scenario", "model", "member", "period"]
    if dataset == "flow":
        id_cols.append("engine")
    df = pd.read_parquet(cache, columns=id_cols + [column],
                         filters=[("scenario", "==", scen), ("period", "==", per)])
    if engine is not None:
        df = df[df["engine"].astype(str) == engine]
    df = df[df["model"].astype(str).isin(universe)]
    v = df[column].astype("float64").replace([np.inf, -np.inf], np.nan)
    df = df.assign(**{column: v}).dropna(subset=[column])
    return df.pivot_table(index="station", columns="model", values=column,
                          aggfunc="mean")


def second_ensemble(M):
    """The treatment ensemble mean per station, as a Series aligned to M.index.

    Under removal this is the unweighted mean over the retained models; under
    weighting it is the weighted mean over EVERY model present, with the
    weights renormalised per station over the models that are finite there
    (model coverage varies by scenario and period, so a single global
    renormalisation would be wrong for the stations that are missing a model).
    Returns the second-ensemble Series and the count to report as `n_reduced`
    (the retained-model count under removal, the Kish effective size under
    weighting)."""
    cols = list(M.columns)
    if not em.is_weighting():
        keep = [m for m in em.retained_models() if m in M.columns]
        return M[keep].mean(axis=1, skipna=True), float(len(keep))
    A = M.to_numpy(dtype="float64")
    w = em.weight_array(cols)                      # renormalised over cols
    finite = np.isfinite(A)
    num = np.where(finite, A * w[None, :], 0.0).sum(axis=1)
    den = np.where(finite, w[None, :], 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(den > 0, num / den, np.nan)
    return pd.Series(vals, index=M.index), em.n_eff_present(cols)


def main():
    em.log_treatment()
    universe = list(em.full_models())
    rows = []
    for scen in SCENARIOS:
        for per in PERIODS:
            for name, dataset, column, engine, scale100 in METRICS:
                M = load_slice(dataset, column, engine, scen, per, universe)
                if M.empty:
                    print(f"  [{scen} {per}] {name} {engine or ''}: empty"); continue
                fac = 100.0 if scale100 else 1.0
                full = M.mean(axis=1, skipna=True) * fac
                red, n_second = second_ensemble(M)
                red = red * fac
                bias = red - full           # reduced/weighted - full (manuscript)
                ok = np.isfinite(full) & np.isfinite(red)
                rows.append(dict(
                    scenario=scen, period=per, variable=name, engine=engine or "",
                    n_sta=int(ok.sum()), n_full=int(M.shape[1]), n_reduced=n_second,
                    full_mean=float(full[ok].mean()), reduced_mean=float(red[ok].mean()),
                    bias_mean=float(bias[ok].mean()),
                    full_median=float(full[ok].median()), reduced_median=float(red[ok].median()),
                    bias_median=float(bias[ok].median())))
                r = rows[-1]
                print(f"  [{scen} {per}] {name:16s} {engine or '':5s} n={r['n_sta']:6d} "
                      f"models {r['n_full']}/{r['n_reduced']:.1f} | mean full={r['full_mean']:+.3f} "
                      f"red={r['reduced_mean']:+.3f} bias={r['bias_mean']:+.3f} | "
                      f"median full={r['full_median']:+.3f} red={r['reduced_median']:+.3f} "
                      f"bias={r['bias_median']:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n-> {OUT_CSV}")


if __name__ == "__main__":
    em.reject_cli_tokens(sys.argv[1:])
    main()
