"""
fig_detectability.py
====================
Detectability curve: how large an ensemble must be before the too-hot bias
emerges above the model spread, per variable. Turns the "S/N < 1 everywhere but
temperature" paradox into an actionable number.

Reads ONLY the per-catchment focus CSVs already written by
analyze_full_vs_reduced_significance.py (its export_focus()), so there is NO
delta / hydrology recompute — it is light enough to run interactively.
Each CSV  (signif_<col>[_<engine>]_<scen>_<per>_nNNNNsta.csv, under
CONFIG.out_path/<delta_method>/fig_data/) carries, per catchment:
    full_minus_reduced : the too-hot bias in the ensemble MEAN (the paper's bias)
    hot_minus_retained : the hot-vs-retained group-mean contrast
    inter_model_std    : sigma across per-model mean changes (the "noise")
    signal_to_noise    : |full_minus_reduced| / inter_model_std   (as stored)
    n_hot, n_retained  : models present in each group (n_full = n_hot+n_retained)
                         -- REMOVAL ONLY. Under method="weighting" no such split
                         exists and the CSVs carry n_models (available) plus
                         n_eff (effective size after weighting) instead; see
                         _models_available().

METHOD
------
For a catchment with bias b and inter-model spread sigma, an ensemble mean built
from N independent model realizations has standard error ~ sigma/sqrt(N). The
bias is "detectable" (full- and reduced-ensemble means separate above the
sampling noise) when
        |b| >= z * sigma / sqrt(N)     <=>     r * sqrt(N) >= z ,
with r = |b|/sigma the single-realization signal-to-noise and z the detection
level (z=1 "exceeds the spread"; z=1.96 ~ p<0.05, two-sided). Hence the required
ensemble size at a catchment is  N_req = (z / r)^2 , and the fraction of
catchments detectable at ensemble size N is  mean_i[ r_i * sqrt(N) >= z ].

This is the standard "signal fixed, noise ~ 1/sqrt(N)" detection argument
(Hawkins & Sutton 2011; Deser et al. 2012). It is an idealization: averaging
fully removes internal variability but only pins down the model-uncertainty
part of the mean, so read the curve as an UPPER BOUND on what more realizations
buy. State this in the caption.

SIGNAL CONVENTION (read this)
-----------------------------
SIGNAL_COL selects what counts as the "bias":
  * "full_minus_reduced" (DEFAULT) — the shift in the ensemble mean when the
    too-hot models are kept vs dropped; the quantity the maps, boxplots and the
    headline "23-29% of the AnnMax intensification" use. Keeps the
    detectability curve consistent with the rest of the paper.
  * "hot_minus_retained" — the raw hot-vs-retained group contrast (~ N/h larger).
    Under this convention temperature median r ~ 1.37; under full_minus_reduced
    (DEFAULT, and now the stored signal_to_noise convention) it is ~0.64. The
    choice moves every curve, so name it in the caption.

Run:
    python paper/figures/scripts/fig_detectability.py
    python paper/figures/scripts/fig_detectability.py hot_minus_retained
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))                 # .../figures/scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))  # repo root
from config import CONFIG
# Nature-style bold top-left panel letters (multi-panel figures only)
from figstyle import add_panel_letters, plabel
import ensemble_method as em

# Screen-aware label text: "too-hot" under SCREEN="hot", but the excluded set
# is not too-hot under SCREEN="both", so every figure/label string uses these
# instead of hardcoding the word.
_XHOT  = em.excluded_label()
_XHOTT = em.excluded_label(title=True)

CACHE_DIR = os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data")
SIG_DIR   = os.path.join(os.path.dirname(_THIS), "significance")   # .../figures/significance
SCEN, PER = "ssp585", "2081-2100"

em.reject_cli_tokens(sys.argv[1:])   # this script has no __main__ block
em.log_treatment()
SIGNAL_COL = sys.argv[1] if len(sys.argv) > 1 else "full_minus_reduced"
assert SIGNAL_COL in ("full_minus_reduced", "hot_minus_retained")

Z_LEVELS = {"exceeds spread (z=1)": 1.0, "p<0.05 (z=1.96)": 1.96}
Z_MAIN   = 1.96                       # threshold drawn in panel (a)
NMAX     = 48
N_GRID   = np.arange(1, NMAX + 1)

# (label, focus-CSV column stem [+_engine for flow], colour). Colours are the
# Okabe-Ito colourblind-safe set. AnnMax = mean annual maximum daily flow (the
# design flood index); mean runoff is the water-balance control that is ~unbiased.
#
# The labels carry the SAME index names as main4.tex and Fig. 3, not the terse
# shorthand this script used to hold ("Mean annual max", "Mean runoff"): one
# concept, one term across the manuscript. None of the five is a percentile
# index, so figstyle.percentile_label has nothing to contribute here; add any
# percentile variable through that table rather than by typing "Q95" here.
#
# LONG and SHORT, following the same split figstyle.PercentileLabel uses. The
# names reach the reader in two places with very different room: the panel (a)
# legend takes the long form, and the five panel (b) x tick labels take the
# short one. Both come from this one list, so they cannot disagree, and
# AnnMax is expanded on the same canvas by the (a) legend before (b) abbreviates
# it. Wrapping the long form to fit (b) instead produced four-line tick stacks.
VARIABLES = [
    ("Temperature, mean",   "Temperature\n(mean)",   "temp_change_degC",               "#0072B2"),
    ("Precipitation, mean", "Precipitation\n(mean)", "precip_change_pct",              "#56B4E9"),
    ("Mean annual maximum (AnnMax), HBV",  "AnnMax\nHBV",  "delta_rel_mean_annual_max_hbv",  "#D55E00"),
    ("Mean annual maximum (AnnMax), LSTM", "AnnMax\nLSTM", "delta_rel_mean_annual_max_lstm", "#E69F00"),
    ("Mean annual runoff, HBV", "Mean annual\nrunoff, HBV", "delta_rel_qmean_hbv",     "#009E73"),
]


def latest_csv(stem):
    """Newest (by mtime) focus CSV for `stem` at the focus scen/period, or None.
    Mirrors analyze_full_vs_reduced_significance._latest_focus_csv so the curve
    uses the same vintage as the maps."""
    # anchored on the ACTIVE treatment: without the suffix this would read the
    # untagged removal+hot focus CSVs under every other treatment
    hits = glob.glob(os.path.join(
        CACHE_DIR, f"signif_{stem}_{SCEN}_{PER}_n*sta{em.suffix()}.csv"))
    return max(hits, key=os.path.getmtime) if hits else None


def _models_available(df, fname):
    """Median number of CMIP6 models with data per catchment.

    The two treatments name this differently in the focus CSVs written by
    analyze_full_vs_reduced_significance.export_focus():

      removal    n_hot + n_retained  -- the two groups the screen splits
      weighting  n_models            -- no split exists; every model carries a
                                        weight, and n_eff is the EFFECTIVE size
                                        after weighting, not the number of
                                        models available

    Only the count of available models belongs on the reference line, so the
    weighting branch takes n_models and never n_eff."""
    if {"n_hot", "n_retained"}.issubset(df.columns):
        n = (df["n_hot"] + df["n_retained"]).to_numpy(float)
    elif "n_models" in df.columns:
        n = df["n_models"].to_numpy(float)
    else:
        raise SystemExit(
            f"{fname}: no model-count column (expected 'n_hot'+'n_retained' "
            f"under removal or 'n_models' under weighting); columns are "
            f"{list(df.columns)}")
    return float(np.median(n))


def load_ratio(stem):
    """Return (r, median_available_N, filename) for one column; r = |signal|/sigma
    over catchments with a finite, positive spread. (None, None, None) if the CSV
    is missing."""
    f = latest_csv(stem)
    if f is None:
        print(f"  [skip] no focus CSV for {stem}")
        return None, None, None
    df = pd.read_csv(f)
    if SIGNAL_COL not in df.columns:
        raise SystemExit(
            f"{os.path.basename(f)} has no column {SIGNAL_COL!r}; under "
            f"method={em.method()!r} the focus CSVs carry "
            f"{[c for c in df.columns if 'minus' in c]}. "
            f"'hot_minus_retained' exists only under removal, where the screen "
            f"splits the ensemble into two groups")
    b     = df[SIGNAL_COL].to_numpy(float)
    sigma = df["inter_model_std"].to_numpy(float)
    r = np.abs(b) / np.where(sigma > 0, sigma, np.nan)
    r = r[np.isfinite(r)]
    n_avail = _models_available(df, os.path.basename(f))
    return r, n_avail, os.path.basename(f)


# ---- gather -----------------------------------------------------------------
data = {}
print(f"detectability | signal = {SIGNAL_COL} | {SCEN} {PER}")
for label, short, stem, colour in VARIABLES:
    r, n_avail, fname = load_ratio(stem)
    if r is None or len(r) == 0:
        continue
    data[label] = dict(r=r, short=short, colour=colour, n_avail=n_avail)
    print(f"  {label:36s} n={len(r):5d}  median r={np.median(r):.2f}  "
          f"detectable@N=1: {(r >= Z_MAIN).mean()*100:4.1f}%  "
          f"avail N~{n_avail:.0f}  ({fname})")

if not data:
    sys.exit("no focus CSVs found — run "
             "analyze_full_vs_reduced_significance.py (climate 'all' + 'flow') first.")

n_avail_all = float(np.median([d["n_avail"] for d in data.values()]))

# ---- figure: (a) detectable fraction vs N; (b) N required at median catchment
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(13.0, 4.7), gridspec_kw={"width_ratios": [1.35, 1.0]})

# (a) fraction of catchments detectable vs ensemble size N -------------------
for label, d in data.items():
    frac = [(d["r"] * np.sqrt(N) >= Z_MAIN).mean() * 100 for N in N_GRID]
    axA.plot(N_GRID, frac, color=d["colour"], lw=2.2, label=label)
axA.axvline(n_avail_all, color="0.35", ls="--", lw=1.1)
axA.text(n_avail_all, 3, f"  ~{n_avail_all:.0f} CMIP6\n  models available",
         fontsize=8, color="0.35", va="bottom")
axA.axvline(1, color="0.75", ls=":", lw=1.0)
axA.set_xlabel("Ensemble size N  (independent model realizations)")
axA.set_ylabel(f"% catchments where the {_XHOT} bias is detectable\n"
               f"(|bias|·√N / σ$_{{model}}$ ≥ {Z_MAIN})")
axA.set_xlim(1, NMAX)
axA.set_ylim(0, 100)
axA.grid(color="0.92", lw=0.5)
axA.set_axisbelow(True)
axA.legend(fontsize=11, loc="upper left")
# The raw column name used to leak into the title as "signal =
# full_minus_reduced": internal shorthand, and under METHOD='weighting' the
# wrong word, because nothing is reduced. The stored column keeps its name --
# the focus CSVs and this figure's own file name are keyed on it, and
# supplement.tex cites that file name -- so only the drawn text changes here.
_SIGNAL_TXT = (f"|{em.second_label().lower()} − full|"
               if SIGNAL_COL == "full_minus_reduced"
               else f"|{_XHOT} − retained|")
axA.set_title(f"Detectability of the {_XHOT} bias vs ensemble size\n"
              f"({SCEN.upper()} {plabel(PER)}; signal = {_SIGNAL_TXT})",
              fontsize=10)

# (b) ensemble size required so the MEDIAN catchment is detectable ------------
labels = list(data.keys())
xpos = np.arange(len(labels))
for (zlab, z), off, hatch in zip(Z_LEVELS.items(), (-0.19, 0.19), (None, "////")):
    nreq = []
    for lab in labels:
        med_r = float(np.median(data[lab]["r"]))
        nreq.append((z / med_r) ** 2 if med_r > 0 else np.nan)
    axB.bar(xpos + off, nreq, width=0.36, hatch=hatch, label=zlab,
            color=[data[lab]["colour"] for lab in labels],
            edgecolor="k", lw=0.4, alpha=0.85)
axB.axhline(n_avail_all, color="0.35", ls="--", lw=1.1)
axB.text(len(labels) - 0.5, n_avail_all * 1.12, f"~{n_avail_all:.0f} available",
         fontsize=8, color="0.35", ha="right")
axB.set_yscale("log")
axB.set_xticks(xpos)
# The SHORT form from VARIABLES, which carries its own line break, rather than
# textwrap over the long form: wrapping "Mean annual maximum (AnnMax), HBV" to
# fit this panel produced a four-line tick stack. The panel (a) legend on the
# same canvas gives every short name in full.
axB.set_xticklabels([data[lab]["short"] for lab in labels], fontsize=9.5)
axB.set_ylabel("Ensemble size N required\n(median catchment)")
axB.grid(axis="y", color="0.92", lw=0.5)
axB.set_axisbelow(True)
axB.legend(fontsize=11)
axB.set_title("Ensemble size to detect the bias\nat the median catchment",
              fontsize=10)

add_panel_letters([axA, axB])
fig.tight_layout()
os.makedirs(SIG_DIR, exist_ok=True)
out = em.tag(os.path.join(SIG_DIR,
                          f"detectability_{SCEN}_{PER}_{SIGNAL_COL}.png"))
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nDetectability figure -> {out}")
