"""
phase0_headline.py
==================

The headline full / reduced / weighted comparison table, for one scenario and
period, over every weight column in the frozen weights CSV.

This reproduces (and supersedes) the ad-hoc script behind job 50364512, whose
output is `fig_data/phase0_headline_wt.csv`. That run predates the AR6
assessed-likelihood scheme, so it carries no `w_like_ar6_tcr` column; this
script recomputes the same quantities and adds it.

Method (identical to the tcr-sweep path in
analyze_full_vs_reduced_significance.py):

  1. read the Stage-1 cache with the metric column and the FOCUS scenario and
     period pushed down into the parquet read;
  2. restrict to the 33-model universe, drop non-finite values;
  3. pivot to a station x model matrix of realization means;
  4. across models, per station:
        full     = unweighted mean over all models present
        reduced  = unweighted mean over models with TCR <= 2.2 degC
        weighted = sum(w_m * x_m) / sum(w_m), renormalised over models present
  5. across stations, report the mean and the median.

The realization blocklist is NOT applied, matching the original run (and hence
main4.tex); the number of rows it WOULD drop is reported so the choice is
visible rather than silent.

Validation: the script reloads the original CSV and checks every full /
reduced / w_bma / w_imp / w_kde cell it can match, printing PASS/FAIL. A
reproduction failure means the aggregation drifted and the new column must not
be trusted.

    python paper/figures/scripts/phase0_headline.py            # -> *_v2.csv
    python paper/figures/scripts/phase0_headline.py --out X.csv
"""

import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))
sys.path.insert(0, _THIS)

from config import CONFIG                                      # noqa: E402
from realization_blocklist import apply_blocklist              # noqa: E402
import ensemble_method as em                                   # noqa: E402

CACHE_DIR = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")
CLIMATE_CACHE = os.path.join(CACHE_DIR, "fig02_change.parquet")
FLOW_CACHE    = os.path.join(CACHE_DIR, "stats_change.parquet")
WEIGHTS_CSV   = os.path.join(CACHE_DIR, "model_weights_wt.csv")
ORIGINAL_CSV  = os.path.join(CACHE_DIR, "phase0_headline_wt.csv")
DEFAULT_OUT   = em.tag(os.path.join(CACHE_DIR, "phase0_headline_v2.csv"))

FOCUS_SCEN = "ssp585"
FOCUS_PER  = "2081-2100"
TOO_HOT_TCR_THRESHOLD = em.TCR_UPPER   # AR6 likely upper bound, from em

# (row name in the output, cache, parquet column, engine, x100?)
# Row names match the original CSV so the validation can join on them.
METRICS = [
    ("warming_degC",    "climate", "temp_change_degC",          None,   False),
    ("precip_mean_pct", "climate", "precip_change_pct",         None,   False),
    ("precip_q99_pct",  "climate", "precip_q99_change_pct",     None,   False),
    ("annmax_pct",      "flow",    "delta_rel_mean_annual_max", "hbv",  True),
    ("rl20_pct",        "flow",    "delta_rel_flood_q20",       "hbv",  True),
    ("qmean_pct",       "flow",    "delta_rel_qmean",           "hbv",  True),
    ("q05_pct",         "flow",    "delta_rel_q05",             "hbv",  True),
    ("annmax_pct",      "flow",    "delta_rel_mean_annual_max", "lstm", True),
    ("rl20_pct",        "flow",    "delta_rel_flood_q20",       "lstm", True),
    ("qmean_pct",       "flow",    "delta_rel_qmean",           "lstm", True),
    ("q05_pct",         "flow",    "delta_rel_q05",             "lstm", True),
]


def load_weights():
    """Frozen weights CSV -> (universe, retained, {column: {model: weight}})."""
    w = pd.read_csv(WEIGHTS_CSV)
    w["model"] = w["model"].astype(str)
    universe = w["model"].tolist()
    # follow the ACTIVE screen, not a hardcoded hot-only cut: under
    # SCREEN='both' the reduced column here must match the rest of the run
    excluded = w["model"].map(em.is_excluded).astype(bool)
    retained = w.loc[~excluded, "model"].tolist()
    cols = [c for c in w.columns if c.startswith("w_")]
    weights = {c: dict(zip(w["model"], w[c].astype(float))) for c in cols}
    return universe, retained, weights


def load_slice(dataset, column, engine, universe):
    """Station x model matrix of realization means for one metric."""
    cache = FLOW_CACHE if dataset == "flow" else CLIMATE_CACHE
    id_cols = ["station", "scenario", "model", "member", "period"]
    if dataset == "flow":
        id_cols.append("engine")
    df = pd.read_parquet(
        cache, columns=id_cols + [column],
        filters=[("scenario", "==", FOCUS_SCEN), ("period", "==", FOCUS_PER)])
    n_raw = len(df)
    # report, but do not apply — the original run had no blocklist (see docstring)
    n_blocked = n_raw - len(apply_blocklist(df.copy(), label="phase0 probe"))
    if engine is not None:
        df = df[df["engine"].astype(str) == engine]
    df = df[df["model"].astype(str).isin(universe)]
    v = df[column].astype("float64").replace([np.inf, -np.inf], np.nan)
    df = df.assign(**{column: v}).dropna(subset=[column])
    M = df.pivot_table(index="station", columns="model", values=column,
                       aggfunc="mean")
    print(f"    rows={n_raw:,} (blocklist would drop {n_blocked:,}) -> "
          f"matrix {M.shape[0]:,} stations x {M.shape[1]} models")
    return M


def aggregate(M, retained, weights, scale100):
    """full / reduced / one column per weight set, as station-wise Series."""
    factor = 100.0 if scale100 else 1.0
    out = {"full": M.mean(axis=1, skipna=True) * factor}
    keep = [m for m in retained if m in M.columns]
    out["reduced"] = M[keep].mean(axis=1, skipna=True) * factor

    A = M.to_numpy(float)
    present = np.isfinite(A)
    for col, wmap in weights.items():
        w = np.array([wmap[m] for m in M.columns], float)
        num = np.nansum(np.where(present, A, 0.0) * w[None, :], axis=1)
        den = (present * w[None, :]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[col] = pd.Series(np.where(den > 0, num / den, np.nan) * factor,
                                 index=M.index)
    return out


def build(out_csv=DEFAULT_OUT):
    universe, retained, weights = load_weights()
    print(f"universe = {len(universe)} models | retained "
          f"({em.screen()} screen, {em.screen_expr()}) = {len(retained)} | "
          f"weight columns = {list(weights)}")
    print(f"scenario = {FOCUS_SCEN} | period = {FOCUS_PER}\n")

    rows = []
    for name, dataset, column, engine, scale100 in METRICS:
        print(f"  {name:16s} {engine or '':5s} <- {dataset}:{column}")
        M = load_slice(dataset, column, engine, universe)
        agg = aggregate(M, retained, weights, scale100)
        n_sta = int(np.isfinite(agg["full"]).sum())
        for stat, fn in (("mean", np.nanmean), ("median", np.nanmedian)):
            row = {"variable": name, "engine": engine or "", "stat": stat,
                   "n_sta": n_sta}
            for key, series in agg.items():
                row[key] = float(fn(series.to_numpy(float)))
            rows.append(row)

    df = pd.DataFrame(rows)
    # share of the full-ensemble signal removed by each treatment, on the one
    # convention (full - X)/full; meaningful for the mean rows only
    for key in [c for c in df.columns if c in ("reduced",) or c.startswith("w_")]:
        df[f"pct_removed_{key}"] = np.where(
            df["stat"].eq("mean"),
            (df["full"] - df[key]) / df["full"] * 100.0, np.nan)
    df.to_csv(out_csv, index=False)
    print(f"\nheadline table -> {out_csv}")
    return df


def validate(df):
    """Compare every reproducible cell against the original phase-0 CSV."""
    if not os.path.exists(ORIGINAL_CSV):
        print(f"\nvalidation skipped: {ORIGINAL_CSV} absent")
        return True
    old = pd.read_csv(ORIGINAL_CSV)
    old["engine"] = old["engine"].fillna("").astype(str)
    key = ["variable", "engine", "stat"]
    shared = [c for c in ("full", "reduced", "w_bma", "w_imp", "w_kde")
              if c in old.columns]
    if em.is_two_sided() or em.metric() != "tcr":
        # The reference CSV was produced by the TCR HOT-ONLY screen, so its
        # "reduced" column is a different quantity under any other treatment
        # and must not be compared. The test is on em.metric() rather than
        # em.is_ecs(), because is_ecs() is False under METHOD='weighting' even
        # when an ECS-fitted variant is active -- and load_weights() below
        # still builds "reduced" from em.is_excluded(), i.e. from the ECS cut.
        # full / w_* are screen-independent, so validating those still proves
        # nothing drifted.
        shared = [c for c in shared if c != "reduced"]
        print(f"\n{em.describe()}: 'reduced' excluded from validation (the "
              "reference is TCR hot-only); full and w_* still checked.")
    rename = {"w_bma": "w_bma_tcr", "w_imp": "w_bma_tcr_imp",
              "w_kde": "w_kde_tcr"}
    # rename the shared names on the OLD side before merging: pandas suffixes
    # only fire on collisions, so "full"/"reduced" would silently become
    # full_old/full_new and the loop below would skip them (it did, once).
    old = old[key + shared].rename(columns={c: f"old_{c}" for c in shared})
    m = old.merge(df, on=key)
    print(f"\n=== validation against {os.path.basename(ORIGINAL_CSV)} "
          f"({len(m)} rows matched) ===")
    ok = True
    for c in shared:
        new_c = rename.get(c, c)
        if new_c not in m.columns:
            print(f"  {c:8s} -> no counterpart column, skipped")
            continue
        a = m[f"old_{c}"].to_numpy(float)
        b = m[new_c].to_numpy(float)
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-9)
        worst = int(np.nanargmax(rel))
        good = bool(np.nanmax(rel) < 1e-6)
        ok &= good
        print(f"  {c:8s} vs {new_c:16s} max|rel diff| = {np.nanmax(rel):.3e} "
              f"({'PASS' if good else 'FAIL'})  worst: "
              f"{m['variable'][worst]}/{m['engine'][worst]}/{m['stat'][worst]} "
              f"{a[worst]:.6f} vs {b[worst]:.6f}")
    print("VERDICT:", "PASS — aggregation reproduced exactly." if ok else
          "FAIL — do NOT trust the new column until this is resolved.")
    return ok


def report(df):
    """Print the manuscript-facing table: full, reduced, BMA, AR6 likelihood."""
    ROWS = [("warming_degC", "", "mean",   "Warming (degC)"),
            ("annmax_pct", "hbv",  "mean",   "AnnMax HBV (%)"),
            ("annmax_pct", "lstm", "mean",   "AnnMax LSTM (%)"),
            ("rl20_pct",   "hbv",  "mean",   "RL20 HBV (%)"),
            ("rl20_pct",   "lstm", "mean",   "RL20 LSTM (%)"),
            ("qmean_pct",  "hbv",  "mean",   "Mean runoff HBV (%)"),
            ("q05_pct",    "hbv",  "median", "Q05 HBV (median, %)")]
    # main4.tex uses TWO conventions and both are reported here so a number can
    # be lifted straight into either sentence:
    #   share  = (full - X)/full     "too-hot models account for 23-29% of the
    #                                 projected increase" (abstract)
    #   inflat = (full - X)/X        "from 4.89 to 4.05 degC, a 21% ensemble-
    #                                 composition inflation" (Results)
    # (column, short header). Columns absent from the CSV are dropped, so the
    # same report works before and after `model_weights.py addlike` / `addecs`.
    CANDIDATES = [("reduced",            "reduced"),
                  ("w_bma_tcr",          "BMA"),
                  ("w_like_ar6_tcr",     "AR6gam"),
                  ("w_likenorm_ar6_tcr", "AR6norm"),
                  ("w_gaussjoint_ar6_tcr", "TCRgauss"),
                  ("w_likemean_ar6_ecs", "ECSmean"),
                  ("w_like_ar6_ecs",     "ECSeq"),
                  ("w_tilt_ar6_ecs",     "ECStilt")]
    cols = [c for c, _ in CANDIDATES if c in df.columns]
    heads = [h for c, h in CANDIDATES if c in df.columns]
    print("\n=== headline table: full | " + " | ".join(heads) + " ===")
    print(f"'reduced' here is the ACTIVE screen: {em.screen_expr()}")
    print(f"{'Quantity':22s} {'n':>7s} {'full':>9s} "
          + " ".join(f"{h:>9s}" for h in heads))
    for var, eng, stat, label in ROWS:
        r = df[(df["variable"] == var) & (df["engine"] == eng)
               & (df["stat"] == stat)]
        if r.empty:
            print(f"{label:22s}  -- not computed --"); continue
        r = r.iloc[0]
        f = r["full"]
        print(f"{label:22s} {int(r['n_sta']):7d} {f:9.3f} "
              + " ".join(f"{r[c]:9.3f}" for c in cols))
        print(f"{'  share (full-X)/full':22s} {'':7s} {'':9s} "
              + " ".join(f"{(f - r[c]) / f * 100.0:8.1f}%" for c in cols))
        print(f"{'  inflation (full-X)/X':22s} {'':7s} {'':9s} "
              + " ".join(f"{(f - r[c]) / r[c] * 100.0:8.1f}%" for c in cols))


if __name__ == "__main__":
    em.reject_cli_tokens(sys.argv[1:])
    # Provenance: the treatment lives in an edited file, so this log line
    # is the only record of what this job actually ran.
    em.log_treatment()
    out = DEFAULT_OUT
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    d = build(out)
    validate(d)
    report(d)
