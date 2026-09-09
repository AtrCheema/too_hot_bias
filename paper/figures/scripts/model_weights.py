"""
model_weights.py
================

SINGLE SOURCE OF TRUTH for climate-model WEIGHTS in this project — the
weighting-scheme counterpart of the TOO_HOT/RETAINED exclusion lists. Every
consumer (fig02_precip_change.py, hydro_indices_change.py, gev_fdr_dipole.py,
analyze_full_vs_reduced_significance.py, ...) must take weights from HERE
(load_frozen() for figure runs, get_weights() to recompute) and never derive
its own, mirroring how the exclusion flows from fig01_model_screening.METRICS.

Method (primary): Bayesian Model Averaging (BMA) weights on the Transient
Climate Response, following Massoud, Lee, Terando & Wehner (2023),
Communications Earth & Environment 4:365, doi:10.1038/s43247-023-01009-8
(text: paper/literature/text/massoud2023.txt). The paper's Methods prose is
ambiguous in places, so the algorithm below follows their PUBLISHED CODE
(github.com/EliasMassoud1/BMA_ECS, ECS_BMA_test.m, checked 11 Aug 2026).

*** NOT AN EXACT REPRODUCTION — describe as "following the algorithm of
Massoud et al. (2023) as released in their public code", never as a
reproduction of their published weights. Two input-level differences are
unavoidable: (i) the released code's ECS vector differs from their published
Table 1 for 13 of 16 models (e.g. ACCESS-CM2 4.66 vs 4.72, IPSL-CM6A-LR 4.70
vs 4.56), so the published weights do not correspond to the published metric
values; (ii) their gamma target is refitted from ECS_Target_Values.mat while
this module uses the article's rounded parameters. Consequently
validate_against_massoud() (published-weight comparison from Table-1 inputs)
measures input+algorithm differences TOGETHER and its strict criteria FAIL
(TCR r=0.85, weighted-mean 1.732 vs 1.753); the ALGORITHM itself is verified
by diagnostics/validate_bma_exact.py on their released x_pars_15000.mat
samples and released inputs (their script is seed-free and loads those
samples, so the comparison is deterministic).

EXACT-REPLICATION RESULT (job 50368752, 11 Aug 2026): on their artifacts the
implementation reproduces the published ECS weights with r=0.998,
slope=1.03, MAD=0.00054 — 15 of 16 models within ~0.0007 (Table-1 rounding);
the single outlier is IPSL-CM6A-LR (0.0416 vs 0.0449), attributable to their
own Table-1/code input inconsistency for that model. Two further findings:
(a) the ARTICLE's stated ECS gamma (67.696, 0.0476; mean 3.22) does NOT
match their released target sample (fit: a=109.73, scale=0.0281; mean 3.09)
— trust the artifacts, not the printed parameters; (b) rerunning with
article-gamma/Table-1 inputs reproduces the validate_against_massoud FAIL
numbers exactly (MAD 0.0039-0.0040, r 0.948), proving that gap is
input-driven, not algorithmic. For TCR no artifacts are released; the
production TCR target keeps the article parameters (119.734, 0.0147) as a
documented choice. ***

The released algorithm:

  1. Sample n weight vectors from the "power prior":
         u ~ Uniform(0,1)^(n x K);  W = u**8 row-normalised to sum 1.
     (Their MATLAB: x = rand(n,K).^8; x_pars = x./sum(x,2). This is what the
     paper's "Jeffreys prior ... covering a larger range" sentence refers to;
     it is much sparser than Dirichlet(1) — single rows concentrate weight on
     a few models. A Dirichlet option is kept for prior-sensitivity checks.)
  2. Likelihood of a sample = gamma pdf of its weighted-mean metric:
         L(w) = gamma_pdf( sum_i w_i x_i ; a, scale=b )
     (Their MATLAB: b_BMA = A*x'; lik = pdf('gam', b_BMA, pd.a, pd.b).
      The paper's sentence "L ∝ [Target − Generated]" is not implementable as
      written — a signed difference is not a probability — and is superseded
      by the code.)
  3. Posterior mean weights = mean of the top KEEP_FRAC samples ranked by
     likelihood ("topfrac"): 2/3 for ECS (their main text / MATLAB
     n_posterior = n*(2/3)), 3/4 for TCR (their supplement text).
     A self-normalised importance-sampling estimator ("importance") is kept
     as a statistically cleaner cross-check; it is NOT the published method.

Targets (their Methods): TCR ~ gamma(a=119.734, scale=0.0147)
(mean 1.760 degC, sd 0.161, "range 1.4-2.2, peak near 1.8"); ECS ~
gamma(a=67.696, scale=0.0476) (mean 3.222, sd 0.392). A rate reading of b
would give a mean of ~8000 degC, so scale is the only physical reading.

Comparison variant: kernel-density-ratio weights ("KDE approach", credited in
Massoud et al. 2023 to Massoud et al. 2020: each model weighted by the ratio
of the target density to the local sample density of models in TCR space):
w_i ∝ gamma_pdf(x_i)/kde_sample(x_i). Far sharper than BMA (it zeroes the
hottest models); reported as a sensitivity, NOT the primary scheme.

THIRD SCHEME (added 11 Aug 2026): direct assessed-likelihood weighting,
w_i ∝ pdf_target(x_i) — each model weighted by the density the assessed TCR
distribution puts on its own TCR. No Monte Carlo, no simplex sampling, no
tuning parameter; the weights are a deterministic function of the TCR vector.

  Target width matters more than the scheme. Massoud et al. (2023) state
  "TCR has an assessed likely range of 1.4-2.2 degC (c.f. the IPCC AR6 WG1
  technical summary)" (their p. 6) and then fit gamma(119.734, 0.0147), whose
  mean is 1.760 and sd 0.161, so 1.4-2.2 spans +/-2.24 sd = 97.5% of the mass.
  But AR6 "likely" means 66%: Hausfather et al. (2022) p. 195 ("the range
  1.4-2.2 degC deemed as 'likely' in AR6") and Beck et al. (2023, Fig. 2
  caption: "the likely range (66% confidence limit) also from IPCC AR6").
  Their target is therefore ~2.6x too narrow for the assessment it cites.
  Applying likelihood weights at THEIR width is not a weighting at all — the
  hottest model gets 5e-09 and N_eff falls to 11.2, harsher than the TCR>2.2
  screen (N_eff 18).

  TCR_TARGET_AR6_GAMMA is therefore fitted here from the VERIFIED fact alone —
  P(1.4 <= TCR <= 2.2) = 0.66, centred in probability (17% in each tail) — via
  fit_gamma_to_interval(). No "best estimate" value is asserted, because the
  literature cache does not carry the printed AR6 number; the fitted mean is a
  RESULT of the interval fit, not an input. At K=33 this gives a genuinely
  graded weighting (hot share 0.19, N_eff 24.4, max/min 16.6) sitting between
  the near-inert BMA (hot share 0.39) and the screen (hot share 0).

Known behaviour at K=33 (this project's universe, 15/33 models above 2.2 K):
the BMA posterior mean stays CLOSE TO UNIFORM (the likelihood constrains only
the 1-D projection sum(w*TCR), and averaging over the kept samples shrinks
towards the prior mean 1/K), so the weighted-mean TCR does NOT reach the
target 1.76. Report `generated_mean` next to the target; never claim the
weighted ensemble matches the assessed TCR distribution without checking it.

Conventions
-----------
* Weights are defined once over the full model UNIVERSE (the 33 models in
  the results archive). Consumers slicing to scenarios/catchments with fewer
  models MUST renormalise via renormalize() — identically project-wide.
* Weights are per MODEL (generalised one-model-one-vote), as in Massoud
  et al. 2023. Weighted statistics over realizations use w_model/n_members.
* Frozen weights: `python model_weights.py weights` (via slurm) writes
  fig_data/model_weights_wt.csv INCLUDING seed/n_samples/estimator/prior
  columns; figure runs read it back with load_frozen() so a numpy upgrade
  cannot silently shift the published numbers.

FOURTH SCHEME (added 23 Aug 2026): the ECS counterpart of the third one.
w_i ∝ pdf_target(ECS_i) against the AR6 assessed ECS distribution. Two facts
drive it and BOTH are PDF-verified in paper/literature/text/massoud2023.txt
(p. 3, quoting AR6): best estimate 3.0 degC, likely range 2.5-4.0 degC. Because
the best estimate is verified for ECS — it is not for TCR — the primary ECS
target is the MEAN-CONSTRAINED fit (mean pinned to 3.0, 66% of the mass in
2.5-4.0), not the equal-tail one. See add_ecs_likelihood_columns().

CLI (submit through slurm — no python from the login/VSCode shell):
    python model_weights.py validate   # reproduce Massoud Table 1 (ECS & TCR)
    python model_weights.py weights    # production weights -> fig_data CSV
    python model_weights.py all        # both
    python model_weights.py ar6fit     # print/verify the fitted AR6 TCR target
    python model_weights.py addlike    # append the AR6 TCR likelihood columns
    python model_weights.py ecsfit     # print/verify the fitted AR6 ECS targets
    python model_weights.py addecs     # append the AR6 ECS likelihood columns
    python model_weights.py addgauss   # append the joint Gaussian TCR column
    python model_weights.py addvlike   # append the very-likely-range TCR column
"""

import os
import sys
import math

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))          # paper/figures/scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_THIS))))  # repo root
sys.path.insert(0, _THIS)

# ---------------------------------------------------------------------------
# Targets (Massoud et al. 2023, Methods) and sampling configuration
# ---------------------------------------------------------------------------
TCR_TARGET_GAMMA = (119.734, 0.0147)   # (a, scale) -> mean 1.760, sd 0.161 degC
ECS_TARGET_GAMMA = (67.696, 0.0476)    # (a, scale) -> mean 3.222, sd 0.392 degC

# IPCC AR6 assessed TCR: likely range 1.4-2.2 degC, where "likely" = 66%.
# Both facts are PDF-verified (see module docstring). TCR_TARGET_AR6_GAMMA is
# fitted from them below, after fit_gamma_to_interval() is defined.
AR6_TCR_LIKELY      = (1.4, 2.2)   # degC, IPCC AR6 WG1 assessed likely range
AR6_TCR_LIKELY_PROB = 0.66         # IPCC calibrated language: "likely" = 66%
# *** NOT YET PDF-VERIFIED *** No IPCC/AR6 document is in paper/literature/,
# and no PDF in that folder prints the AR6 TCR best estimate. 1.80 is used here
# only by the mean-constrained fit below; it is also the midpoint of the likely
# range, and Massoud et al. (2023) refer to a peak "near 1.8 degC". Verify
# against IPCC AR6 WG1 before any manuscript sentence attributes it to AR6.
AR6_TCR_BEST_ESTIMATE = 1.80       # degC

SEED      = 20260811      # frozen; stamped into the production CSV
N_SAMPLES = 15_000        # Massoud et al. 2023 (validation MUST use this)
N_SAMPLES_PRODUCTION = 200_000   # same method, smaller Monte Carlo error for
                                 # the K=33 universe (documented choice)
# Their keep fraction is METRIC-DEPENDENT: main text (ECS, Fig 1b) says "2/3
# of the model weights from the posterior samples"; supplement (TCR, Fig S1B)
# says "3/4". Their MATLAB (ECS) has n_posterior = n_sample*(2/3).
KEEP_FRAC = {"ecs": 2.0 / 3.0, "tcr": 0.75}
POWER_EXPONENT  = 8       # their rand()^8 prior
DIRICHLET_ALPHA = 0.5     # sensitivity-only alternative prior
DEFAULT_PRIOR     = "power"      # faithful to their code
DEFAULT_ESTIMATOR = "topfrac"    # faithful to their code

# Acceptance criteria for validate_against_massoud() (set by the adversarial
# review): the faithful variant must beat the uniform-weights baseline MAD
# (0.0069 TCR / 0.0116 ECS) decisively, reproduce the SPAN of the published
# weights (slope of fit est~published within 1 +/- 0.3), correlate r>=0.9,
# and land the weighted-mean metric within +/-0.02 of their published means
# (1.7534 TCR / 3.1431 ECS). Their own Monte Carlo noise puts a hard floor of
# ~0.002 on the achievable MAD (BCC-CSM2-MR and MIROC6 share TCR 1.55 yet
# publish weights 0.0639 vs 0.0664).
VALIDATION_MAD_MAX   = {"tcr": 0.004, "ecs": 0.005}
VALIDATION_CORR_MIN  = 0.90
VALIDATION_SLOPE     = (0.7, 1.3)
VALIDATION_WMEAN_TOL = 0.02
PUBLISHED_BMA_MEAN   = {"ecs": 3.1431, "tcr": 1.7534}

# ---------------------------------------------------------------------------
# Massoud et al. (2023) Table 1 — their 16 models, their metric values and
# their published BMA mean weights. Used ONLY by validate_against_massoud().
# model: (ECS, w_ECS, TCR, w_TCR)   [verified digit-by-digit vs the paper]
# ---------------------------------------------------------------------------
MASSOUD_TABLE1 = {
    "ACCESS-CM2":    (4.72, 0.0412, 1.96, 0.0708),
    "ACCESS-ESM1-5": (3.87, 0.0581, 1.97, 0.0715),
    "BCC-CSM2-MR":   (3.04, 0.0723, 1.55, 0.0639),
    "CanESM5":       (5.62, 0.0290, 2.71, 0.0494),
    "EC-Earth3":     (4.30, 0.0498, 2.30, 0.0626),
    "FGOALS-g3":     (2.88, 0.0716, 1.50, 0.0608),
    "GFDL-ESM4":     (3.90, 0.0589, 1.63, 0.0688),
    "INM-CM4-8":     (1.83, 0.0646, 1.30, 0.0501),
    "INM-CM5-0":     (1.92, 0.0649, 1.41, 0.0552),
    "IPSL-CM6A-LR":  (4.56, 0.0449, 2.35, 0.0614),
    "MIROC6":        (2.61, 0.0767, 1.55, 0.0664),
    "MPI-ESM1-2-HR": (2.98, 0.0731, 1.64, 0.0686),
    "MPI-ESM1-2-LR": (3.00, 0.0755, 1.82, 0.0737),
    "MRI-ESM2-0":    (3.15, 0.0730, 1.67, 0.0712),
    "NorESM2-LM":    (2.54, 0.0736, 1.49, 0.0596),
    "NorESM2-MM":    (2.50, 0.0727, 1.22, 0.0461),
}


# ---------------------------------------------------------------------------
# numerics (no scipy dependency: log-space gamma pdf, Silverman KDE)
# ---------------------------------------------------------------------------
def gamma_logpdf(x, a, scale):
    """log pdf of gamma(a, scale) at x (elementwise); -inf for x <= 0."""
    x = np.asarray(x, float)
    out = np.full(x.shape, -np.inf)
    ok = x > 0
    out[ok] = ((a - 1.0) * np.log(x[ok]) - x[ok] / scale
               - math.lgamma(a) - a * math.log(scale))
    return out


def gamma_pdf(x, a, scale):
    return np.exp(gamma_logpdf(x, a, scale))


def _gammainc_lower_reg(a, x):
    """Regularised lower incomplete gamma P(a, x), scalar, pure numpy.

    Series representation for x < a+1, Lentz continued fraction for Q(a, x)
    otherwise (Numerical Recipes 6.2). Kept dependency-free on purpose: this
    module must import without scipy, exactly like gamma_logpdf above. Checked
    against scipy.special.gammainc by the `ar6fit` CLI command."""
    a = float(a)
    x = float(x)
    if x <= 0.0:
        return 0.0
    lead = math.exp(-x + a * math.log(x) - math.lgamma(a))
    if x < a + 1.0:                                   # series -> P directly
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(10_000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return min(1.0, total * lead)
    tiny = 1e-300                                     # continued fraction -> Q
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0.0 else 1.0 / tiny
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, 1.0 - lead * h)


def gamma_cdf(x, a, scale):
    """CDF of gamma(a, scale) at x (elementwise over `x`)."""
    xs = np.atleast_1d(np.asarray(x, float))
    out = np.array([_gammainc_lower_reg(a, v / scale) for v in xs])
    return out if np.ndim(x) else float(out[0])


def _gamma_ppf_unit(a, p, iters=80):
    """Quantile of gamma(a, scale=1) at probability p, by bisection on the
    CDF. Deterministic and monotone, so it needs no starting guess. 80 halvings
    take any starting bracket below double precision; the count is bounded
    because this runs at import time (see TCR_TARGET_AR6_GAMMA)."""
    lo, hi = 0.0, a + 20.0 * math.sqrt(a) + 40.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _gammainc_lower_reg(a, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_gamma_to_interval(lo, hi, prob):
    """(a, scale) of the gamma whose central `prob` interval is [lo, hi], i.e.
    CDF(lo) = (1-prob)/2 and CDF(hi) = (1+prob)/2.

    Used to turn an IPCC calibrated-language statement ("likely range 1.4-2.2
    degC", likely = 66%) into a target distribution WITHOUT assuming a best
    estimate: the resulting mean is an output of the fit, not an input.

    The shape parameter is found from the quantile RATIO, which is free of
    `scale` and decreases monotonically in `a` (gamma -> normal as a -> inf),
    so a plain bisection is safe."""
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < lo < hi, got lo={lo!r} hi={hi!r}")
    if not (0.0 < prob < 1.0):
        raise ValueError(f"prob must be in (0,1), got {prob!r}")
    p_lo = 0.5 * (1.0 - prob)
    p_hi = 0.5 * (1.0 + prob)
    want = hi / lo

    def ratio(a):
        return _gamma_ppf_unit(a, p_hi) / _gamma_ppf_unit(a, p_lo)

    # a_lo is 0.5 rather than ~0: below that the lower quantile of gamma(a,1)
    # underflows to 0 and the ratio becomes inf/nan. a_hi = 1e6 is already far
    # into the normal limit (ratio there is 1.002, i.e. below any real range).
    a_lo, a_hi = 0.5, 1e6
    if ratio(a_lo) < want or ratio(a_hi) > want:
        raise ValueError(f"interval ratio {want:.4f} outside the gamma family "
                         f"for prob={prob}")
    for _ in range(100):
        a_mid = math.sqrt(a_lo * a_hi)                # geometric: a spans decades
        if ratio(a_mid) > want:
            a_lo = a_mid
        else:
            a_hi = a_mid
    a = math.sqrt(a_lo * a_hi)
    scale = lo / _gamma_ppf_unit(a, p_lo)
    got = _gammainc_lower_reg(a, hi / scale) - _gammainc_lower_reg(a, lo / scale)
    if abs(got - prob) > 1e-6:                        # never ship a silent miss
        raise RuntimeError(f"gamma fit to [{lo}, {hi}] at p={prob} did not "
                           f"converge: coverage {got:.8f}")
    return a, scale


def _gamma_interval_mass(a, scale, lo, hi):
    """P(lo <= X <= hi) for gamma(a, scale)."""
    return (_gammainc_lower_reg(a, hi / scale)
            - _gammainc_lower_reg(a, lo / scale))


def fit_gamma_to_mean_and_interval(mean, lo, hi, prob):
    """(a, scale) of the gamma with a GIVEN mean whose interval [lo, hi] holds
    probability `prob`.

    The alternative to fit_gamma_to_interval(): both are exactly determined
    (two equations, two unknowns), but they consume different facts.

      fit_gamma_to_interval        CDF(lo) = (1-p)/2 and CDF(hi) = (1+p)/2
                                   -> uses the range + probability, and ASSUMES
                                      the two tails carry equal probability.
                                      The mean is an output.
      fit_gamma_to_mean_and_interval   a*scale = mean and CDF(hi) - CDF(lo) = p
                                   -> uses the range + probability + the
                                      assessed best estimate. The tail split is
                                      an output, so no symmetry is assumed.

    The second is the more faithful reading IF the best estimate is verified,
    which for TCR it is not yet (see AR6_TCR_BEST_ESTIMATE). With the mean
    pinned, sd = mean/sqrt(a), so the interval mass rises monotonically with a
    and a plain bisection is safe."""
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < lo < hi, got lo={lo!r} hi={hi!r}")
    if not (0.0 < prob < 1.0):
        raise ValueError(f"prob must be in (0,1), got {prob!r}")
    if not (lo < mean < hi):
        raise ValueError(f"mean {mean!r} must lie inside [{lo}, {hi}]")

    def mass(a):
        return _gamma_interval_mass(a, mean / a, lo, hi)

    a_lo, a_hi = 0.5, 1e6
    if mass(a_lo) > prob or mass(a_hi) < prob:
        raise ValueError(f"no gamma with mean {mean} puts {prob} of its mass "
                         f"in [{lo}, {hi}]")
    for _ in range(100):
        a_mid = math.sqrt(a_lo * a_hi)
        if mass(a_mid) < prob:
            a_lo = a_mid
        else:
            a_hi = a_mid
    a = math.sqrt(a_lo * a_hi)
    scale = mean / a
    got = _gamma_interval_mass(a, scale, lo, hi)
    if abs(got - prob) > 1e-6:
        raise RuntimeError(f"mean-constrained gamma fit did not converge: "
                           f"coverage {got:.8f}")
    return a, scale


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p):
    """Standard-normal quantile by bisection (no scipy)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_normal_to_interval(lo, hi, prob):
    """(mu, sigma) of the normal whose central `prob` interval is [lo, hi].
    Closed form, unlike the gamma: mu is the midpoint and sigma follows from
    the standard-normal quantile. The Gaussian counterpart of
    fit_gamma_to_interval(), kept for the target-family sensitivity."""
    if not (lo < hi):
        raise ValueError(f"need lo < hi, got lo={lo!r} hi={hi!r}")
    if not (0.0 < prob < 1.0):
        raise ValueError(f"prob must be in (0,1), got {prob!r}")
    z = _norm_ppf(0.5 * (1.0 + prob))
    return 0.5 * (lo + hi), 0.5 * (hi - lo) / z


def normal_logpdf(x, mu, sigma):
    """log pdf of Normal(mu, sigma) at x (elementwise)."""
    z = (np.asarray(x, float) - mu) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2.0 * math.pi))


# Fitted here rather than hard-coded so the two AR6 inputs above stay the only
# stated facts. Deterministic (no RNG, no scipy); the published numbers are
# frozen at the CSV, as for every other scheme in this module.
TCR_TARGET_AR6_GAMMA  = fit_gamma_to_interval(*AR6_TCR_LIKELY,
                                              prob=AR6_TCR_LIKELY_PROB)
# Gaussian target over the SAME assessed interval — sensitivity only, so the
# family assumption inherited from Massoud et al. (2023) can be quantified
# rather than asserted. Never the production scheme.
TCR_TARGET_AR6_NORMAL = fit_normal_to_interval(*AR6_TCR_LIKELY,
                                               prob=AR6_TCR_LIKELY_PROB)
# Mean-constrained variant: pins the mean to the assessed best estimate instead
# of assuming equal tail probabilities. Candidate primary scheme, but it can
# only become primary once AR6_TCR_BEST_ESTIMATE is verified against AR6.
TCR_TARGET_AR6_MEANFIT = fit_gamma_to_mean_and_interval(
    AR6_TCR_BEST_ESTIMATE, *AR6_TCR_LIKELY, prob=AR6_TCR_LIKELY_PROB)

# --- very-likely-range TCR normal: the "vlike" scheme (27 Aug 2026) ---------
# Co-author proposal: weight each model by the density of the normal whose
# central 90% interval is the AR6 very likely TCR range 1.2-2.4 degC
# (mu 1.80, sigma 0.3648). Unlike gaussjoint, nothing is solved against the
# archive: the kernel is fixed by the assessed interval alone, so the
# likely-range mass (0.727, not 0.66) and the weighted-ensemble mean TCR
# (~1.846, not 1.80) are OUTCOMES of the scheme, not conditions of it.
# *** NOT PDF-VERIFIED *** No document in paper/literature/ prints the AR6
# very likely TCR range; 1.2-2.4 degC at 90% must be verified against IPCC
# AR6 WG1 Ch. 7 before any manuscript sentence attributes it to AR6.
AR6_TCR_VERY_LIKELY      = (1.2, 2.4)   # degC; NOT PDF-verified, see above
AR6_TCR_VERY_LIKELY_PROB = 0.90         # IPCC calibrated language: "very likely"
TCR_TARGET_AR6_VL_NORMAL = fit_normal_to_interval(*AR6_TCR_VERY_LIKELY,
                                                  prob=AR6_TCR_VERY_LIKELY_PROB)

# --- the ECS counterparts (added 23 Aug 2026) -------------------------------
# IPCC AR6 assessed ECS: best estimate 3.0 degC, likely range 2.5-4.0 degC.
# PDF-VERIFIED, unlike AR6_TCR_BEST_ESTIMATE: paper/literature/text/
# massoud2023.txt p. 3 states "Based on multiple lines of evidence, the IPCC
# AR6 assessed best estimate of ECS is 3 degC with a likely range of 2.5-4 degC
# (high confidence)", citing AR6. "likely" is the same IPCC calibrated-language
# term as for TCR, i.e. at least 66% probability.
#
# Because the best estimate IS verified here, ECS_TARGET_AR6_MEANFIT (mean
# pinned to 3.0, 66% of the mass in 2.5-4.0) is the PRIMARY ECS target: it
# consumes both assessed facts and assumes no tail symmetry.
# ECS_TARGET_AR6_GAMMA (equal-tail) is kept as the sensitivity that asserts no
# best estimate, exactly as for TCR.
#
# Note the same width problem as for TCR: Massoud et al. (2023) fit
# ECS_TARGET_GAMMA = (67.696, 0.0476) to this range, but that gamma puts 94.5%
# of its mass in 2.5-4.0, not 66%, so it is ~2.4x too narrow for the
# assessment it cites. Both targets below are refitted at 66%.
AR6_ECS_LIKELY        = (2.5, 4.0)   # degC, IPCC AR6 WG1 assessed likely range
AR6_ECS_LIKELY_PROB   = 0.66         # IPCC calibrated language: "likely" = 66%
AR6_ECS_BEST_ESTIMATE = 3.00         # degC, AR6 assessed best estimate

ECS_TARGET_AR6_GAMMA   = fit_gamma_to_interval(*AR6_ECS_LIKELY,
                                               prob=AR6_ECS_LIKELY_PROB)
ECS_TARGET_AR6_MEANFIT = fit_gamma_to_mean_and_interval(
    AR6_ECS_BEST_ESTIMATE, *AR6_ECS_LIKELY, prob=AR6_ECS_LIKELY_PROB)


# ---------------------------------------------------------------------------
# FIFTH SCHEME (chosen 23 Aug 2026): jointly solved Gaussian kernel on TCR.
# Production port of tmp_gaussian_weight_shape.solve_centre_and_width()
# (panel c of weighting/fig_gaussian_weight_shape_tcr.png). The kernel
#
#     w_i  proportional to  exp( -(TCR_i - mu)^2 / (2 sigma^2) )
#
# has its two parameters solved so that BOTH
#   (i)  the kernel keeps AR6_TCR_LIKELY_PROB (66%) of its own mass inside the
#        AR6 likely range 1.4-2.2 degC, and
#   (ii) the WEIGHTED-ENSEMBLE mean TCR equals AR6_TCR_BEST_ESTIMATE (1.80).
#
# *** PROVENANCE: our construction, not a published climate model-weighting
# scheme. Condition (ii) makes the kernel ENSEMBLE-DEPENDENT (it is retuned
# against the 33 TCR values in this archive), exactly like the ECS tilt
# (exponential_tilt); the solved centre sits below the assessed best estimate
# (mu = 1.705 at K = 33), so the kernel is NOT the assessed TCR distribution.
# The weighted mean of 1.80 is true BY CONSTRUCTION and must never be quoted
# as evidence of agreement with the assessment. Note also the asymmetry with
# the primary ECS weighting: 'ecslike' pins the TARGET's own mean to the
# assessed best estimate, whereas this scheme pins the ENSEMBLE mean — the
# ECS analogue of this construction is 'ecstilt', which is sensitivity-only
# there. ***
# ---------------------------------------------------------------------------
def _normal_interval_mass(mu, sigma, lo, hi):
    """P(lo <= X <= hi) for Normal(mu, sigma)."""
    return _norm_cdf((hi - mu) / sigma) - _norm_cdf((lo - mu) / sigma)


def _gauss_sigma_for_mass(mu, lo, hi, prob):
    """The unique sigma putting `prob` of a Normal(mu, .) inside [lo, hi].
    For mu inside the interval the mass decreases monotonically with sigma,
    so a plain bisection is safe (asserted by the mu bound check)."""
    if not (lo < mu < hi):
        raise ValueError(f"centre {mu!r} must lie inside [{lo}, {hi}] for the "
                         "mass to be monotone in sigma")
    a, b = 1e-4, 50.0
    for _ in range(200):
        m = 0.5 * (a + b)
        if _normal_interval_mass(mu, m, lo, hi) > prob:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def fit_gauss_joint_tcr(values, target_mean=None, likely=None, prob=None,
                        bracket=(1.42, 1.84), n_scan=2101):
    """(mu, sigma) of the Gaussian kernel satisfying conditions (i) and (ii)
    above for the TCR values in `values` ({model: TCR}).

    sigma is eliminated through the mass condition, leaving one equation in mu,
    which is solved by SCAN-THEN-BISECT rather than a single bisection: the
    weighted mean along the constraint curve is not strictly monotone near the
    lower edge of the likely range (verified 23 Aug 2026: decreases of
    <= 1.4e-4 degC below mu = 1.415 at K = 33), so the root count is checked
    instead of assumed. Raises unless exactly one root lies in `bracket`."""
    _, x = _check_values(values)
    target_mean = AR6_TCR_BEST_ESTIMATE if target_mean is None else target_mean
    lo, hi = AR6_TCR_LIKELY if likely is None else likely
    prob = AR6_TCR_LIKELY_PROB if prob is None else prob

    def dev(mu):
        sg = _gauss_sigma_for_mass(mu, lo, hi, prob)
        lw = -0.5 * ((x - mu) / sg) ** 2
        w = np.exp(lw - lw.max())
        w = w / w.sum()
        return float(w @ x) - target_mean

    grid = np.linspace(bracket[0], bracket[1], n_scan)
    dv = np.array([dev(m) for m in grid])
    crossings = np.where(np.diff(np.sign(dv)) != 0)[0]
    if len(crossings) != 1:
        raise RuntimeError(
            f"joint Gaussian solve found {len(crossings)} roots in "
            f"{bracket}, expected exactly one; inspect the constraint curve")
    a, b = grid[crossings[0]], grid[crossings[0] + 1]
    fa = dev(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        if dev(m) * fa > 0:
            a = m
        else:
            b = m
    mu = 0.5 * (a + b)
    sigma = _gauss_sigma_for_mass(mu, lo, hi, prob)
    if abs(_normal_interval_mass(mu, sigma, lo, hi) - prob) > 1e-6:
        raise RuntimeError("joint Gaussian solve: mass condition not met")
    if abs(dev(mu)) > 1e-8:
        raise RuntimeError("joint Gaussian solve: mean condition not met")
    return mu, sigma


def exponential_tilt(values, base_logw, target_mean):
    """Weights closest to a base vector that hit a MEAN CONSTRAINT exactly.

        w_i  proportional to  b_i * exp(lambda * x_i),   sum_i w_i x_i = mu*

    `values` is {model: metric value}, `base_logw` the log of the base weights
    aligned to list(values) -- the dict's own insertion order, which is what
    _check_values() uses -- (zeros = uniform base), and `target_mean` is the
    mu* the weighted ensemble must reach. Returns (lambda, {model: weight}).

    Among all weight vectors satisfying the mean constraint this is the one
    that minimises the Kullback-Leibler divergence from the base, i.e. the
    smallest departure from the base weights that still meets the constraint.
    sum(w*x) increases monotonically in lambda, so a bisection is safe.

    *** PROVENANCE: exponential tilting is the standard device for imposing a
    moment constraint on a set of weights (importance sampling / maximum-
    entropy reweighting). It is NOT a method taken from the climate
    model-weighting literature. Any manuscript sentence must describe it as our
    construction. ***

    One consequence is worth stating because it decides how the result should
    be read: tilting a gamma density by exp(lambda*x) returns ANOTHER gamma
    with the same shape a and scale s' = s/(1 - lambda*s). So a tilted-gamma
    weighting is not a new kind of scheme; it is the SAME density weighting
    against a DIFFERENT gamma - one whose own mean and interval coverage no
    longer match the assessment, because it has been retuned against the
    particular models in this archive. tilted_gamma_params() returns it."""
    models, x = _check_values(values)
    base_logw = np.asarray(base_logw, float)
    if base_logw.shape != x.shape:
        raise ValueError(f"base_logw has shape {base_logw.shape}, expected "
                         f"{x.shape} (aligned to list(values))")

    def at(lam):
        lw = base_logw + lam * x
        w = np.exp(lw - lw.max())
        w /= w.sum()
        return float(w @ x), w

    lo, hi = -200.0, 200.0
    if at(lo)[0] > target_mean or at(hi)[0] < target_mean:
        raise ValueError(f"target mean {target_mean} outside the achievable "
                         f"range [{x.min()}, {x.max()}]")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if at(mid)[0] < target_mean:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    got, w = at(lam)
    if abs(got - target_mean) > 1e-10:
        raise RuntimeError(f"tilt did not converge: mean {got}")
    return lam, dict(zip(models, w.tolist()))


def tilted_gamma_params(target, lam):
    """(a, scale) of the gamma that a gamma `target` becomes after an
    exponential tilt by exp(lam*x): same shape, scale s/(1 - lam*s).
    Raises if the tilt leaves the gamma family (1 - lam*s <= 0)."""
    a, s = target
    den = 1.0 - lam * s
    if den <= 0:
        raise ValueError(f"tilt lambda={lam} leaves the gamma family "
                         f"(1 - lambda*scale = {den})")
    return a, s / den


def _silverman_kde_at(points, x):
    """Gaussian KDE of `points` evaluated at `x`, Silverman bandwidth."""
    pts = np.asarray(points, float)
    n = len(pts)
    sd = pts.std(ddof=1)
    iqr = np.subtract(*np.percentile(pts, [75, 25]))
    sigma = min(sd, iqr / 1.349) if iqr > 0 else sd
    h = 0.9 * sigma * n ** (-0.2)          # Silverman's rule of thumb
    if not np.isfinite(h) or h <= 0:
        raise ValueError("degenerate KDE bandwidth (identical metric values?)")
    z = (np.asarray(x, float)[:, None] - pts[None, :]) / h
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * h * math.sqrt(2.0 * math.pi))


def _check_values(values, minimum=2):
    models = list(values)
    x = np.array([values[m] for m in models], float)
    if len(models) < minimum:
        raise ValueError(f"need at least {minimum} models to weight")
    if not np.all(np.isfinite(x)):
        bad = [m for m, v in zip(models, x) if not np.isfinite(v)]
        raise ValueError(f"non-finite metric value(s) for: {bad}")
    return models, x


def _sample_prior(rng, n_samples, k, prior):
    """Weight-vector prior samples, shape (n_samples, k), rows sum to 1."""
    if prior == "power":                       # their rand()^8, row-normalised
        u = rng.random((n_samples, k)) ** POWER_EXPONENT
        return u / u.sum(axis=1, keepdims=True)
    if prior == "dirichlet":                   # sensitivity-only alternative
        return rng.dirichlet(np.full(k, DIRICHLET_ALPHA), size=n_samples)
    raise ValueError(f"unknown prior {prior!r}")


# ---------------------------------------------------------------------------
# core estimators
# ---------------------------------------------------------------------------
def bma_weights(values, target, n_samples=N_SAMPLES, estimator=DEFAULT_ESTIMATOR,
                keep_frac=0.75, prior=DEFAULT_PRIOR, seed=SEED,
                return_diagnostics=False):
    """BMA mean weights for `values` ({model: metric value}) against a gamma
    `target` = (a, scale). Returns {model: weight}; with return_diagnostics,
    (weights, diag dict). `keep_frac` must be passed per metric (2/3 ECS,
    3/4 TCR — see KEEP_FRAC); it is ignored by the 'importance' estimator."""
    models, x = _check_values(values)
    a, scale = target

    rng = np.random.default_rng(seed)
    W = _sample_prior(rng, n_samples, len(models), prior)
    g = W @ x                                       # generated metric per sample
    logL = gamma_logpdf(g, a, scale)
    if not np.isfinite(logL.max()):
        raise ValueError("all sampled likelihoods are zero — metric values "
                         "incompatible with the target distribution")

    if estimator == "topfrac":                      # the published method
        k = max(1, int(round(keep_frac * n_samples)))
        idx = np.argsort(logL)[-k:]                 # highest-likelihood samples
        wbar = W[idx].mean(axis=0)
        ess = float(k)
    elif estimator == "importance":                 # cross-check only
        logL = logL - logL.max()                    # underflow guard
        L = np.exp(logL)
        L_sum = L.sum()
        wbar = (L @ W) / L_sum
        ess = float(L_sum ** 2 / (L * L).sum())     # Kish ESS of the sampler
    else:
        raise ValueError(f"unknown estimator {estimator!r}")

    wbar = wbar / wbar.sum()                        # exact renormalisation
    out = dict(zip(models, wbar.tolist()))
    if not return_diagnostics:
        return out
    diag = {
        "estimator": estimator, "prior": prior, "n_samples": n_samples,
        "keep_frac": keep_frac, "seed": seed, "sampler_ess": ess,
        "generated_mean": float(wbar @ x),          # weighted-ensemble metric
        "equal_weight_mean": float(x.mean()),
        "target_mean": a * scale,
        "n_eff": n_eff(out),
    }
    return out, diag


def _normalise_logweights(models, lp):
    """Shared tail of every likelihood scheme: exponentiate in a shift-safe
    way and normalise. The shift cancels exactly under normalisation."""
    if not np.isfinite(lp.max()):
        raise ValueError("all target densities are zero — metric values "
                         "incompatible with the target distribution")
    w = np.exp(lp - lp.max())
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"likelihood weights sum to {s!r}")
    return dict(zip(models, (w / s).tolist()))


def likelihood_weights_normal(values, target):
    """Gaussian counterpart of likelihood_weights(): w_i proportional to
    Normal(mu, sigma) evaluated at model i's metric value. Sensitivity only."""
    models, x = _check_values(values)
    return _normalise_logweights(models, normal_logpdf(x, *target))


def likelihood_weights(values, target):
    """Direct assessed-likelihood weights: w_i proportional to the density the
    target distribution places on model i's own metric value,

        w_i = pdf_target(x_i) / sum_j pdf_target(x_j).

    Deterministic (no Monte Carlo), no tuning parameter, and — unlike the BMA
    scheme — no dependence on a keep fraction. Computed in log space because
    the pdf underflows to 0 for the hottest models under a narrow target."""
    models, x = _check_values(values)
    a, scale = target
    return _normalise_logweights(models, gamma_logpdf(x, a, scale))


def kde_weights(values, target):
    """Density-ratio weights (the 'KDE approach' credited by Massoud et al.
    2023 to Massoud et al. 2020): w_i ∝ target_pdf(x_i)/sample_kde(x_i)."""
    models, x = _check_values(values)
    a, scale = target
    num = gamma_pdf(x, a, scale)
    den = _silverman_kde_at(x, x)
    w = num / den
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("KDE weights degenerate (all target densities ~0)")
    return dict(zip(models, (w / s).tolist()))


# ---------------------------------------------------------------------------
# shared helpers for every consumer
# ---------------------------------------------------------------------------
def n_eff(weights):
    """Kish effective ensemble size 1/sum(w^2) of a normalised weight dict."""
    w = np.array(list(weights.values()), float)
    w = w / w.sum()
    return float(1.0 / (w * w).sum())


def renormalize(weights, present):
    """Subset `weights` to the models in `present` and renormalise to sum 1.
    EVERY consumer slicing to a scenario/catchment with missing models must go
    through this. Fails loudly on models without a weight (misspelled or new
    model names must not silently re-base the ensemble)."""
    present = list(present)
    missing = [m for m in present if m not in weights]
    if missing:
        raise KeyError(f"models with no weight: {sorted(missing)}")
    sub = {m: weights[m] for m in present}
    if not sub:
        raise ValueError("no models given to renormalize()")
    s = sum(sub.values())
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"present-model weights sum to {s!r}")
    return {m: v / s for m, v in sub.items()}


def weighted_mean(values, weights):
    """Weighted mean of {model: value} under {model: weight}; renormalises
    over the models present in BOTH dicts (NaN values are dropped first)."""
    vals = {m: v for m, v in values.items() if np.isfinite(v)}
    w = renormalize(weights, vals.keys())
    return float(sum(w[m] * vals[m] for m in w))


# ---------------------------------------------------------------------------
# production weights for THIS project's model universe
# ---------------------------------------------------------------------------
def universe_metric_values(metric="tcr", models=None):
    """{model: metric value} for the results universe (or an explicit model
    list) from fig01_model_screening.METRICS. Fails loudly on a missing
    model. NOTE: importing fig01 pulls in matplotlib and applies the house
    style; models_in_results() additionally opens one mean_deltas.nc."""
    from fig01_model_screening import METRICS, models_in_results
    if models is None:
        models = models_in_results()
    models = sorted(models)                 # deterministic regardless of input
    idx = {"ecs": 0, "tcr": 1}[metric]      # METRICS tuple = (ECS, TCR, trend)
    missing = [m for m in models if m not in METRICS]
    if missing:
        raise RuntimeError(
            f"{len(missing)} model(s) have no METRICS entry: {sorted(missing)}; "
            "add them to fig01_model_screening.METRICS first.")
    return {m: float(METRICS[m][idx]) for m in models}


def get_weights(metric="tcr", method="bma", models=None,
                n_samples=N_SAMPLES_PRODUCTION, estimator=DEFAULT_ESTIMATOR,
                prior=DEFAULT_PRIOR, seed=SEED):
    """Recompute weights for the model universe. Figure runs should prefer
    load_frozen(). method: 'bma' (primary), 'kde', 'likelihood' or
    'likelihood_mean' (sensitivities).

    'likelihood'      w_i proportional to the EQUAL-TAIL AR6 target density at
                      model i's metric value. Defined for both metrics: the
                      assessed likely range is verified for TCR (1.4-2.2 degC)
                      and for ECS (2.5-4.0 degC).
    'likelihood_mean' the same rule against the MEAN-CONSTRAINED target, which
                      additionally pins the target's own mean to the assessed
                      best estimate. For ECS that best estimate (3.0 degC) is
                      PDF-verified, so this is the primary ECS scheme; for TCR
                      1.80 degC is not verified, so it stays a sensitivity."""
    values = universe_metric_values(metric=metric, models=models)
    target = TCR_TARGET_GAMMA if metric == "tcr" else ECS_TARGET_GAMMA
    ar6_equal = {"tcr": TCR_TARGET_AR6_GAMMA, "ecs": ECS_TARGET_AR6_GAMMA}
    ar6_mean = {"tcr": TCR_TARGET_AR6_MEANFIT, "ecs": ECS_TARGET_AR6_MEANFIT}
    if method == "bma":
        return bma_weights(values, target, n_samples=n_samples,
                           estimator=estimator, keep_frac=KEEP_FRAC[metric],
                           prior=prior, seed=seed)
    if method == "kde":
        return kde_weights(values, target)
    if method == "likelihood":
        return likelihood_weights(values, ar6_equal[metric])
    if method == "likelihood_mean":
        return likelihood_weights(values, ar6_mean[metric])
    raise ValueError(f"unknown method {method!r}")


def frozen_csv_path():
    from config import CONFIG
    return os.path.join(CONFIG.out_path, CONFIG.delta_method, "fig_data",
                        "model_weights_wt.csv")


def load_frozen(column="w_bma_tcr", path=None):
    """{model: weight} from the frozen production CSV — the weight source for
    every figure/cache run, so published numbers cannot drift with numpy or
    seed changes. Raises with instructions if the CSV is absent."""
    import pandas as pd
    path = path or frozen_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python model_weights.py weights` "
            "(via slurm) first to freeze the weights.")
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(f"{column!r} not in {path} "
                       f"(have: {[c for c in df.columns]})")
    return dict(zip(df["model"].astype(str), df[column].astype(float)))


# ---------------------------------------------------------------------------
# validation against Massoud et al. (2023) Table 1
# ---------------------------------------------------------------------------
def _fit_slope(pub, est):
    """OLS slope of est on pub (1.0 = published span reproduced)."""
    return float(np.polyfit(pub, est, 1)[0])


def validate_against_massoud(n_samples=N_SAMPLES, seeds=(SEED, 1, 2, 3, 4)):
    """Recompute the BMA weights from Massoud's OWN 16-model inputs and
    compare with their published Table 1 mean weights. Variants: the faithful
    (power prior, topfrac) plus importance/Dirichlet sensitivity readings and
    the KDE alternative. Seed-averaged (their Monte Carlo draw is a fixed
    .mat file we do not have, so agreement must hold across seeds). Prints
    per-model tables, acceptance statistics and a PASS/FAIL verdict for the
    faithful variant; returns the summary dict."""
    VARIANTS = [("power",     "topfrac"),      # published method
                ("power",     "importance"),
                ("dirichlet", "topfrac"),
                ("dirichlet", "importance")]
    summary = {}
    for metric, target, vcol, wcol in (("ecs", ECS_TARGET_GAMMA, 0, 1),
                                       ("tcr", TCR_TARGET_GAMMA, 2, 3)):
        values = {m: t[vcol] for m, t in MASSOUD_TABLE1.items()}
        published = {m: t[wcol] for m, t in MASSOUD_TABLE1.items()}
        pub = np.array([published[m] for m in values])
        x = np.array([values[m] for m in values])
        kf = KEEP_FRAC[metric]
        uniform_mad = float(np.mean(np.abs(pub - 1.0 / len(pub))))
        print(f"\n=== validate [{metric.upper()}] target gamma{target} | "
              f"n_samples={n_samples} | keep_frac={kf:.3f} | "
              f"uniform-baseline MAD={uniform_mad:.4f} ===")
        cols = {}
        for pr, est in VARIANTS:
            per_seed = [np.array([bma_weights(values, target,
                                              n_samples=n_samples,
                                              estimator=est, keep_frac=kf,
                                              prior=pr, seed=s)[m]
                                  for m in values]) for s in seeds]
            cols[f"{pr[:3]}/{est[:3]}"] = np.mean(per_seed, axis=0)
        kw = kde_weights(values, target)
        cols["kde"] = np.array([kw[m] for m in values])

        hdr = " ".join(f"{c:>9s}" for c in cols)
        print(f"{'model':16s} {'value':>6s} {'published':>9s} {hdr}")
        for i, m in enumerate(values):
            row = " ".join(f"{cols[c][i]:9.4f}" for c in cols)
            print(f"{m:16s} {values[m]:6.2f} {pub[i]:9.4f} {row}")

        for cname, est_w in cols.items():
            mad = float(np.mean(np.abs(est_w - pub)))
            r = float(np.corrcoef(est_w, pub)[0, 1])
            slope = _fit_slope(pub, est_w)
            wm = float(est_w @ x)
            summary[(metric, cname)] = {"mad": mad, "corr": r,
                                        "slope": slope, "weighted_mean": wm}
            print(f"  [{cname:9s}] MAD={mad:.4f} (uniform {uniform_mad:.4f}) "
                  f"| r={r:+.3f} | slope={slope:+.2f} | "
                  f"weighted-mean={wm:.4f} "
                  f"(published {PUBLISHED_BMA_MEAN[metric]:.4f})")

        s = summary[(metric, "pow/top")]
        ok = (s["mad"] <= VALIDATION_MAD_MAX[metric]
              and s["mad"] < uniform_mad
              and s["corr"] >= VALIDATION_CORR_MIN
              and VALIDATION_SLOPE[0] <= s["slope"] <= VALIDATION_SLOPE[1]
              and abs(s["weighted_mean"] - PUBLISHED_BMA_MEAN[metric])
                  <= VALIDATION_WMEAN_TOL)
        summary[(metric, "verdict")] = "PASS" if ok else "FAIL"
        print(f"  VERDICT [{metric.upper()}, faithful power/topfrac]: "
              f"{'PASS' if ok else 'FAIL'} "
              f"(criteria: MAD<={VALIDATION_MAD_MAX[metric]}, "
              f"MAD<uniform, r>={VALIDATION_CORR_MIN}, "
              f"slope in {VALIDATION_SLOPE}, "
              f"|wmean-published|<={VALIDATION_WMEAN_TOL})")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _report_ar6_target():
    """Print the fitted AR6 TCR target and cross-check the pure-numpy
    incomplete gamma against scipy where scipy is importable. Verification
    only — nothing downstream depends on scipy being present."""
    a, scale = TCR_TARGET_AR6_GAMMA
    lo, hi = AR6_TCR_LIKELY
    mean = a * scale
    sd = math.sqrt(a) * scale
    cov = gamma_cdf(hi, a, scale) - gamma_cdf(lo, a, scale)
    print("\n=== AR6 TCR target fitted from the assessed likely range ===")
    print(f"inputs (PDF-verified): likely range {lo}-{hi} degC, "
          f"'likely' = {AR6_TCR_LIKELY_PROB:.0%}")
    print(f"fitted gamma: a = {a:.4f}, scale = {scale:.6f}  ->  "
          f"mean {mean:.4f} degC, sd {sd:.4f} degC")
    print(f"check: P({lo} <= TCR <= {hi}) = {cov:.6f} "
          f"(target {AR6_TCR_LIKELY_PROB})")
    print(f"Massoud target for contrast: gamma{TCR_TARGET_GAMMA} -> "
          f"mean {TCR_TARGET_GAMMA[0] * TCR_TARGET_GAMMA[1]:.4f}, sd "
          f"{math.sqrt(TCR_TARGET_GAMMA[0]) * TCR_TARGET_GAMMA[1]:.4f}; it "
          f"puts {gamma_cdf(hi, *TCR_TARGET_GAMMA) - gamma_cdf(lo, *TCR_TARGET_GAMMA):.4f} "
          f"of its mass in {lo}-{hi} (i.e. treats a 66% range as ~97%)")
    print(f"sd ratio AR6/Massoud = "
          f"{sd / (math.sqrt(TCR_TARGET_GAMMA[0]) * TCR_TARGET_GAMMA[1]):.3f}")
    try:
        from scipy.special import gammainc as _sp_gammainc
    except Exception as exc:                      # scipy absent -> skip check
        print(f"scipy cross-check skipped ({exc.__class__.__name__})")
        return
    probe = [0.3, 1.0, 1.4, 1.8, 2.2, 3.0, 6.0]
    err = max(abs(_gammainc_lower_reg(a, v / scale) - _sp_gammainc(a, v / scale))
              for v in probe)
    print(f"scipy cross-check of _gammainc_lower_reg: max |diff| = {err:.3e} "
          f"over TCR = {probe}")


def _report_ar6_ecs_target():
    """Print both fitted AR6 ECS targets and check their coverage. The ECS
    counterpart of _report_ar6_target(); prints the mean-constrained fit too,
    because for ECS (unlike TCR) the assessed best estimate is PDF-verified and
    the mean-constrained target is therefore the primary one."""
    lo, hi = AR6_ECS_LIKELY
    print("\n=== AR6 ECS targets fitted from the assessed range ===")
    print(f"inputs (PDF-verified, massoud2023.txt p. 3 quoting AR6): best "
          f"estimate {AR6_ECS_BEST_ESTIMATE} degC, likely range {lo}-{hi} "
          f"degC, 'likely' = {AR6_ECS_LIKELY_PROB:.0%}")
    for name, (a, scale) in (("equal-tail  ", ECS_TARGET_AR6_GAMMA),
                             ("mean-pinned ", ECS_TARGET_AR6_MEANFIT)):
        cov = gamma_cdf(hi, a, scale) - gamma_cdf(lo, a, scale)
        print(f"  {name}: a = {a:.4f}, scale = {scale:.6f} -> mean "
              f"{a * scale:.4f}, sd {math.sqrt(a) * scale:.4f}, "
              f"P({lo} <= ECS <= {hi}) = {cov:.6f}")
    ma, ms = ECS_TARGET_GAMMA
    print(f"  Massoud target for contrast: gamma{ECS_TARGET_GAMMA} -> mean "
          f"{ma * ms:.4f}, sd {math.sqrt(ma) * ms:.4f}; it puts "
          f"{gamma_cdf(hi, ma, ms) - gamma_cdf(lo, ma, ms):.4f} of its mass in "
          f"{lo}-{hi} (i.e. treats a 66% range as ~95%)")


def add_ecs_likelihood_columns(path=None):
    """Append the ECS-based assessed-likelihood weights to an EXISTING frozen
    CSV, computed from the `ecs` column already in it.

    Three columns, all of the form w_i proportional to gamma_pdf(ECS_i):

      w_likemean_ar6_ecs  PRIMARY. Target = gamma with mean exactly the AR6
                          best estimate 3.0 degC and 66% of its mass in
                          2.5-4.0 degC. Both facts are PDF-verified, so the
                          target IS the assessment; the weighted ensemble mean
                          that comes out is a RESULT, not an input.
      w_like_ar6_ecs      Equal-tail sensitivity: target's central 66% interval
                          is exactly 2.5-4.0 and its mean is an output of the
                          fit (3.256). It does not take 3.0 as an INPUT, but it
                          is not estimate-free either: its MODE lands at 3.062,
                          so it is close to reading the assessed best estimate
                          as the mode rather than the mean. The mean / median /
                          mode readings are laid out, with the weighted-mean
                          ECS each produces, by
                          tmp_ecs_gamma_feasibility.report_central_estimate_readings().
      w_tilt_ar6_ecs      Mean-CONSTRAINED sensitivity: the mean-pinned target
                          exponentially tilted until the WEIGHTED ENSEMBLE mean
                          ECS equals 3.000 exactly. Provenance warning in
                          exponential_tilt(): our construction, not a published
                          climate model-weighting scheme. Reported so the cost
                          of forcing the constraint is visible.

    Deliberately additive, exactly like add_likelihood_columns(): every
    previously written column is carried through untouched and a paranoia check
    aborts if any of them moves."""
    import pandas as pd
    path = path or frozen_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python model_weights.py weights` first.")
    df = pd.read_csv(path)
    for col in ("model", "ecs"):
        if col not in df.columns:
            raise KeyError(f"{col!r} missing from {path}")

    ecs = dict(zip(df["model"].astype(str), df["ecs"].astype(float)))
    if len(ecs) != len(df):
        raise ValueError(f"duplicate model names in {path}")
    w_mean = likelihood_weights(ecs, ECS_TARGET_AR6_MEANFIT)
    w_equal = likelihood_weights(ecs, ECS_TARGET_AR6_GAMMA)
    models = list(ecs)          # _check_values() keeps dict insertion order
    base = gamma_logpdf(np.array([ecs[m] for m in models], float),
                        *ECS_TARGET_AR6_MEANFIT)
    lam, w_tilt = exponential_tilt(ecs, base, AR6_ECS_BEST_ESTIMATE)
    tilt_gamma = tilted_gamma_params(ECS_TARGET_AR6_MEANFIT, lam)
    # the tilt must stay inside the gamma family, or the figure's density curve
    # would not pass through the model points; assert it rather than assume it
    curve = gamma_pdf(np.array([ecs[m] for m in models], float), *tilt_gamma)
    curve = curve / curve.sum()
    err = float(np.max(np.abs(curve - np.array([w_tilt[m] for m in models]))))
    if err > 1e-12:
        raise RuntimeError(f"tilted-gamma identity failed (max |diff| {err:.3e})")

    OWNED = ("w_likemean_ar6_ecs", "w_like_ar6_ecs", "w_tilt_ar6_ecs",
             "ecs_ar6_gamma_a", "ecs_ar6_gamma_scale",
             "ecs_ar6_meanfit_a", "ecs_ar6_meanfit_scale",
             "ecs_tilt_gamma_a", "ecs_tilt_gamma_scale", "ecs_tilt_lambda")
    before = {c: df[c].copy() for c in df.columns if c not in OWNED}
    name = df["model"].astype(str)
    df["w_likemean_ar6_ecs"] = name.map(w_mean)
    df["w_like_ar6_ecs"] = name.map(w_equal)
    df["w_tilt_ar6_ecs"] = name.map(w_tilt)
    df["ecs_ar6_gamma_a"], df["ecs_ar6_gamma_scale"] = ECS_TARGET_AR6_GAMMA
    df["ecs_ar6_meanfit_a"], df["ecs_ar6_meanfit_scale"] = ECS_TARGET_AR6_MEANFIT
    df["ecs_tilt_gamma_a"], df["ecs_tilt_gamma_scale"] = tilt_gamma
    df["ecs_tilt_lambda"] = lam
    if df[["w_likemean_ar6_ecs", "w_like_ar6_ecs",
           "w_tilt_ar6_ecs"]].isna().any().any():
        raise RuntimeError("model name mismatch while mapping the ECS weights")
    for c, old in before.items():                 # paranoia: nothing else moved
        if not old.equals(df[c]):
            raise RuntimeError(f"existing column {c!r} changed — aborting")

    _report_ar6_ecs_target()
    print(f"tilt: lambda = {lam:+.5f} -> tilted target gamma"
          f"({tilt_gamma[0]:.4f}, {tilt_gamma[1]:.6f}), own mean "
          f"{tilt_gamma[0] * tilt_gamma[1]:.4f} degC, mass in "
          f"{AR6_ECS_LIKELY[0]}-{AR6_ECS_LIKELY[1]} = "
          f"{_gamma_interval_mass(*tilt_gamma, *AR6_ECS_LIKELY):.4f} "
          f"(the tilted target is NOT the AR6 distribution)")

    x = df["ecs"].to_numpy(float)
    hot = x > 4.0                       # ECS_UPPER of ensemble_method
    k = len(df)
    print(f"\n=== ECS likelihood weights | K = {k} | equal weight = "
          f"{1.0 / k:.4f} | equal-weight mean ECS = {x.mean():.4f} ===")
    print(df[["model", "ecs", "tcr", "w_likemean_ar6_ecs", "w_like_ar6_ecs",
              "w_tilt_ar6_ecs"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    for col, wmap in (("w_likemean_ar6_ecs", w_mean),
                      ("w_like_ar6_ecs", w_equal),
                      ("w_tilt_ar6_ecs", w_tilt)):
        w = df[col].to_numpy(float)
        print(f"\n{col}: weighted-mean ECS = {np.average(x, weights=w):.4f} "
              f"(assessed best estimate {AR6_ECS_BEST_ESTIMATE}) | "
              f"weighted-mean TCR = "
              f"{np.average(df['tcr'].to_numpy(float), weights=w):.4f}")
        print(f"    N_eff = {n_eff(wmap):.2f} of {k} | span {w.min():.5f} to "
              f"{w.max():.5f} (ratio {w.max() / w.min():.1f}) | "
              f"ECS>4.0 share = {w[hot].sum():.4f} "
              f"(equal {hot.sum() / k:.4f}) -> hot influence removed = "
              f"{1 - w[hot].sum() / (hot.sum() / k):.1%}")

    df.to_csv(path, index=False)
    print(f"\nweights CSV (ECS columns appended) -> {path}")
    return df


def add_gaussjoint_columns(path=None):
    """Append the jointly solved Gaussian TCR kernel weights to an EXISTING
    frozen CSV, computed from the `tcr` column already in it.

    One weight column plus the two solved kernel parameters as provenance:

      w_gaussjoint_ar6_tcr  w_i proportional to Normal(mu, sigma) at TCR_i,
                            with (mu, sigma) from fit_gauss_joint_tcr(): 66%
                            of the kernel mass inside 1.4-2.2 degC AND the
                            weighted-ensemble mean TCR pinned to 1.80 degC.
                            See the PROVENANCE warning at fit_gauss_joint_tcr:
                            our construction; the 1.80 is imposed, not found.
      gaussjoint_mu         solved kernel centre (constant column)
      gaussjoint_sigma      solved kernel width  (constant column)

    Deliberately additive, exactly like add_likelihood_columns(): every
    previously written column is carried through untouched and a paranoia
    check aborts if any of them moves."""
    import pandas as pd
    path = path or frozen_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python model_weights.py weights` first.")
    df = pd.read_csv(path)
    for col in ("model", "tcr"):
        if col not in df.columns:
            raise KeyError(f"{col!r} missing from {path}")

    tcr = dict(zip(df["model"].astype(str), df["tcr"].astype(float)))
    if len(tcr) != len(df):
        raise ValueError(f"duplicate model names in {path}")
    mu, sigma = fit_gauss_joint_tcr(tcr)
    w_gj = likelihood_weights_normal(tcr, (mu, sigma))

    OWNED = ("w_gaussjoint_ar6_tcr", "gaussjoint_mu", "gaussjoint_sigma")
    before = {c: df[c].copy() for c in df.columns if c not in OWNED}
    df["w_gaussjoint_ar6_tcr"] = df["model"].astype(str).map(w_gj)
    df["gaussjoint_mu"] = mu
    df["gaussjoint_sigma"] = sigma
    if df["w_gaussjoint_ar6_tcr"].isna().any():
        raise RuntimeError("model name mismatch while mapping the joint "
                           "Gaussian weights")
    for c, old in before.items():                 # paranoia: nothing else moved
        if not old.equals(df[c]):
            raise RuntimeError(f"existing column {c!r} changed — aborting")

    x = df["tcr"].to_numpy(float)
    w = df["w_gaussjoint_ar6_tcr"].to_numpy(float)
    k = len(df)
    hot, cold = x > 2.2, x < 1.4          # TCR bounds of ensemble_method
    m = float(np.average(x, weights=w))
    sd = float(np.sqrt(np.average((x - m) ** 2, weights=w)))
    print(f"=== joint Gaussian TCR kernel | K = {k} | equal weight = "
          f"{1.0 / k:.4f} | equal-weight mean TCR = {x.mean():.4f} ===")
    print(f"solved kernel: mu = {mu:.4f} degC, sigma = {sigma:.4f} degC "
          f"(offset from best estimate {AR6_TCR_BEST_ESTIMATE}: "
          f"{mu - AR6_TCR_BEST_ESTIMATE:+.4f} degC)")
    print(f"  kernel mass in {AR6_TCR_LIKELY[0]}-{AR6_TCR_LIKELY[1]} = "
          f"{_normal_interval_mass(mu, sigma, *AR6_TCR_LIKELY):.4f} "
          f"(condition (i), target {AR6_TCR_LIKELY_PROB})")
    print(f"  weighted mean TCR = {m:.4f} degC (condition (ii), IMPOSED) | "
          f"weighted sd = {sd:.4f} degC")
    print(f"  N_eff = {n_eff(w_gj):.2f} of {k} | span {w.min():.5f} to "
          f"{w.max():.5f} (ratio {w.max() / w.min():.1f})")
    print(f"  TCR>2.2 share = {w[hot].sum():.4f} (equal {hot.mean():.4f}) -> "
          f"hot influence removed = {1 - w[hot].sum() / hot.mean():.1%}")
    print(f"  TCR<1.4 share = {w[cold].sum():.4f} (equal {cold.mean():.4f})")
    print(df[["model", "tcr", "too_hot", "w_bma_tcr", "w_likenorm_ar6_tcr",
              "w_gaussjoint_ar6_tcr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    df.to_csv(path, index=False)
    print(f"\nweights CSV (joint Gaussian column appended) -> {path}")
    return df


def add_vlikenorm_columns(path=None):
    """Append the very-likely-range Gaussian TCR weights (the co-author
    "vlike" scheme, 27 Aug 2026) to an EXISTING frozen CSV, computed from the
    `tcr` column already in it.

    One weight column plus the two kernel parameters as provenance:

      w_vlikenorm_ar6_tcr  w_i proportional to Normal(mu, sigma) at TCR_i,
                           with (mu, sigma) = TCR_TARGET_AR6_VL_NORMAL: the
                           normal whose central 90% interval is the AR6 very
                           likely range 1.2-2.4 degC (mu 1.80, sigma 0.3648).
                           *** the very likely range is NOT PDF-verified;
                           see the constant's comment block ***
      vlikenorm_mu         kernel centre (constant column)
      vlikenorm_sigma      kernel width  (constant column)

    Unlike add_gaussjoint_columns(), nothing is solved against the archive,
    so the likely-range mass and the weighted-ensemble mean TCR printed below
    are OUTCOMES. Deliberately additive, exactly like the other add_*
    functions: every previously written column is carried through untouched
    and a paranoia check aborts if any of them moves."""
    import pandas as pd
    path = path or frozen_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python model_weights.py weights` first.")
    df = pd.read_csv(path)
    for col in ("model", "tcr"):
        if col not in df.columns:
            raise KeyError(f"{col!r} missing from {path}")

    tcr = dict(zip(df["model"].astype(str), df["tcr"].astype(float)))
    if len(tcr) != len(df):
        raise ValueError(f"duplicate model names in {path}")
    mu, sigma = TCR_TARGET_AR6_VL_NORMAL
    w_vl = likelihood_weights_normal(tcr, (mu, sigma))

    OWNED = ("w_vlikenorm_ar6_tcr", "vlikenorm_mu", "vlikenorm_sigma")
    before = {c: df[c].copy() for c in df.columns if c not in OWNED}
    df["w_vlikenorm_ar6_tcr"] = df["model"].astype(str).map(w_vl)
    df["vlikenorm_mu"] = mu
    df["vlikenorm_sigma"] = sigma
    if df["w_vlikenorm_ar6_tcr"].isna().any():
        raise RuntimeError("model name mismatch while mapping the "
                           "very-likely-range weights")
    for c, old in before.items():                 # paranoia: nothing else moved
        if not old.equals(df[c]):
            raise RuntimeError(f"existing column {c!r} changed — aborting")

    x = df["tcr"].to_numpy(float)
    w = df["w_vlikenorm_ar6_tcr"].to_numpy(float)
    k = len(df)
    hot, cold = x > 2.2, x < 1.4          # TCR bounds of ensemble_method
    m = float(np.average(x, weights=w))
    sd = float(np.sqrt(np.average((x - m) ** 2, weights=w)))
    print(f"=== very-likely-range Gaussian TCR kernel | K = {k} | equal "
          f"weight = {1.0 / k:.4f} | equal-weight mean TCR = {x.mean():.4f} ===")
    print(f"kernel: mu = {mu:.4f} degC, sigma = {sigma:.4f} degC "
          f"(fitted to {AR6_TCR_VERY_LIKELY} at {AR6_TCR_VERY_LIKELY_PROB:.0%};"
          f" range NOT PDF-verified)")
    print(f"  kernel mass in {AR6_TCR_VERY_LIKELY[0]}-{AR6_TCR_VERY_LIKELY[1]}"
          f" = {_normal_interval_mass(mu, sigma, *AR6_TCR_VERY_LIKELY):.4f} "
          f"(imposed by the fit)")
    print(f"  kernel mass in {AR6_TCR_LIKELY[0]}-{AR6_TCR_LIKELY[1]} = "
          f"{_normal_interval_mass(mu, sigma, *AR6_TCR_LIKELY):.4f} "
          f"(OUTCOME; the likely-mass condition 0.66 is NOT imposed)")
    print(f"  weighted mean TCR = {m:.4f} degC (OUTCOME; 1.80 is NOT imposed)"
          f" | weighted sd = {sd:.4f} degC")
    print(f"  N_eff = {n_eff(w_vl):.2f} of {k} | span {w.min():.5f} to "
          f"{w.max():.5f} (ratio {w.max() / w.min():.1f})")
    print(f"  TCR>2.2 share = {w[hot].sum():.4f} (equal {hot.mean():.4f}) -> "
          f"hot influence removed = {1 - w[hot].sum() / hot.mean():.1%}")
    print(f"  TCR<1.4 share = {w[cold].sum():.4f} (equal {cold.mean():.4f})")
    print(df[["model", "tcr", "too_hot", "w_likenorm_ar6_tcr",
              "w_gaussjoint_ar6_tcr", "w_vlikenorm_ar6_tcr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    df.to_csv(path, index=False)
    print(f"\nweights CSV (very-likely-range column appended) -> {path}")
    return df


def _report_target_families():
    """Sensitivity of the likelihood weights to the DISTRIBUTION FAMILY and to
    WHICH assessed interval is fitted.

    AR6 publishes intervals, not a density, so any likelihood weighting must
    assume a family. Gamma is inherited from Massoud et al. (2023) — it is not
    prescribed by AR6 — so the size of that assumption has to be reported, not
    asserted away. Report-only: nothing here feeds the frozen weights."""
    from fig01_model_screening import TOO_HOT_TCR_THRESHOLD
    tcr = universe_metric_values("tcr")
    x = np.array(list(tcr.values()), float)
    hot = x > TOO_HOT_TCR_THRESHOLD
    k = len(x)
    lo, hi = AR6_TCR_LIKELY

    def norm_cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def norm_ppf(p):                                  # bisection; no scipy
        a, b = -40.0, 40.0
        for _ in range(200):
            m = 0.5 * (a + b)
            if norm_cdf(m) < p:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    def summarise(tag, logpdf):
        lp = logpdf(x)
        w = np.exp(lp - lp.max())
        w = w / w.sum()
        print(f"  {tag:34s} hot share = {w[hot].sum():.4f} | N_eff = "
              f"{1.0 / (w * w).sum():5.2f} | weighted-mean TCR = "
              f"{w @ x:.4f} | max/min = {w.max() / w.min():8.2f} | "
              f"hot influence removed = "
              f"{1 - w[hot].sum() / (hot.sum() / k):5.1%}")

    print("\n=== target-family sensitivity of the likelihood weights ===")
    print(f"universe K = {k}, of which {hot.sum()} have TCR > "
          f"{TOO_HOT_TCR_THRESHOLD}; equal-weight hot share = "
          f"{hot.sum() / k:.4f}")
    print("all rows below weight by w_i proportional to the target density at "
          "model i's TCR;\nthey differ ONLY in what that target is.\n")

    a66, s66 = TCR_TARGET_AR6_GAMMA
    print(f"  [production] gamma fitted to the LIKELY (66%) range {lo}-{hi}: "
          f"a={a66:.4f}, scale={s66:.6f}, mean={a66 * s66:.4f}, "
          f"sd={math.sqrt(a66) * s66:.4f}")
    summarise("gamma, 66% range 1.4-2.2", lambda v: gamma_logpdf(v, a66, s66))

    am, sm = TCR_TARGET_AR6_MEANFIT
    tail_lo = gamma_cdf(lo, am, sm)
    tail_hi = 1.0 - gamma_cdf(hi, am, sm)
    print(f"\n  gamma with the mean PINNED to the best estimate "
          f"{AR6_TCR_BEST_ESTIMATE} (best estimate NOT yet PDF-verified): "
          f"a={am:.4f}, scale={sm:.6f}, mean={am * sm:.4f}, "
          f"sd={math.sqrt(am) * sm:.4f}, mode={(am - 1) * sm:.4f}")
    print(f"    resulting tail split: {tail_lo:.4f} below {lo}, "
          f"{tail_hi:.4f} above {hi} (equal-tail fit assumes "
          f"{0.5 * (1 - AR6_TCR_LIKELY_PROB):.4f} each)")
    summarise("gamma, mean pinned to 1.80", lambda v: gamma_logpdf(v, am, sm))

    # AR6 also assesses a VERY LIKELY (90%) range; fitting that instead is an
    # equally faithful reading of the same assessment.
    VERY_LIKELY, VL_PROB = (1.2, 2.4), 0.90
    a90, s90 = fit_gamma_to_interval(*VERY_LIKELY, prob=VL_PROB)
    print(f"\n  gamma fitted to the VERY LIKELY (90%) range "
          f"{VERY_LIKELY[0]}-{VERY_LIKELY[1]}: a={a90:.4f}, "
          f"scale={s90:.6f}, mean={a90 * s90:.4f}, "
          f"sd={math.sqrt(a90) * s90:.4f}")
    summarise("gamma, 90% range 1.2-2.4", lambda v: gamma_logpdf(v, a90, s90))
    cov = (gamma_cdf(VERY_LIKELY[1], a66, s66)
           - gamma_cdf(VERY_LIKELY[0], a66, s66))
    print(f"  consistency check: the 66%-fitted gamma puts {cov:.4f} of its "
          f"mass in {VERY_LIKELY[0]}-{VERY_LIKELY[1]} (AR6 says {VL_PROB})")

    z = norm_ppf(0.5 * (1.0 + AR6_TCR_LIKELY_PROB))
    mu = 0.5 * (lo + hi)
    sd = 0.5 * (hi - lo) / z
    print(f"\n  normal fitted to the 66% range: mu={mu:.4f}, sd={sd:.4f}")
    summarise("normal, 66% range 1.4-2.2",
              lambda v: -0.5 * ((np.asarray(v, float) - mu) / sd) ** 2)

    mul = 0.5 * (math.log(lo) + math.log(hi))
    sdl = 0.5 * (math.log(hi) - math.log(lo)) / z
    print(f"  lognormal fitted to the 66% range: mu_log={mul:.4f}, "
          f"sd_log={sdl:.4f} (median {math.exp(mul):.4f})")
    summarise("lognormal, 66% range 1.4-2.2",
              lambda v: (-0.5 * ((np.log(np.asarray(v, float)) - mul) / sdl) ** 2
                         - np.log(np.asarray(v, float))))

    print(f"\n  for contrast, Massoud's own target gamma{TCR_TARGET_GAMMA} "
          f"(mean {TCR_TARGET_GAMMA[0] * TCR_TARGET_GAMMA[1]:.4f}, sd "
          f"{math.sqrt(TCR_TARGET_GAMMA[0]) * TCR_TARGET_GAMMA[1]:.4f})")
    summarise("gamma, Massoud width (sd 0.16)",
              lambda v: gamma_logpdf(v, *TCR_TARGET_GAMMA))


def add_likelihood_columns(path=None):
    """Append the AR6 likelihood weights to an EXISTING frozen CSV, computed
    from the `tcr` column already in it.

    Deliberately additive: every previously written column is carried through
    untouched, so the published BMA/KDE numbers cannot move when the new
    scheme is added. Use `weights` instead to rebuild the whole table."""
    import pandas as pd
    path = path or frozen_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python model_weights.py weights` first.")
    df = pd.read_csv(path)
    for col in ("model", "tcr"):
        if col not in df.columns:
            raise KeyError(f"{col!r} missing from {path}")

    tcr = dict(zip(df["model"].astype(str), df["tcr"].astype(float)))
    if len(tcr) != len(df):
        raise ValueError(f"duplicate model names in {path}")
    w_lik = likelihood_weights(tcr, TCR_TARGET_AR6_GAMMA)
    w_nrm = likelihood_weights_normal(tcr, TCR_TARGET_AR6_NORMAL)
    w_mfit = likelihood_weights(tcr, TCR_TARGET_AR6_MEANFIT)

    # columns this function owns; excluded from the no-drift check below so a
    # re-run can legitimately update them after a change to the fit
    OWNED = ("w_like_ar6_tcr", "w_likenorm_ar6_tcr", "w_likemean_ar6_tcr",
             "ar6_gamma_a", "ar6_gamma_scale")
    before = {c: df[c].copy() for c in df.columns if c not in OWNED}
    df["w_like_ar6_tcr"] = df["model"].astype(str).map(w_lik)
    df["w_likenorm_ar6_tcr"] = df["model"].astype(str).map(w_nrm)
    df["w_likemean_ar6_tcr"] = df["model"].astype(str).map(w_mfit)
    df["ar6_gamma_a"] = TCR_TARGET_AR6_GAMMA[0]
    df["ar6_gamma_scale"] = TCR_TARGET_AR6_GAMMA[1]
    if df[["w_like_ar6_tcr", "w_likenorm_ar6_tcr",
           "w_likemean_ar6_tcr"]].isna().any().any():
        raise RuntimeError("model name mismatch while mapping new weights")
    for c, old in before.items():                 # paranoia: nothing else moved
        if not old.equals(df[c]):
            raise RuntimeError(f"existing column {c!r} changed — aborting")

    _report_ar6_target()
    equal = 1.0 / len(df)
    hot = df["too_hot"].astype(str).str.lower().eq("true").to_numpy()
    w = df["w_like_ar6_tcr"].to_numpy(float)
    print(f"\n=== AR6 likelihood weights | K = {len(df)} | "
          f"equal weight = {equal:.4f} ===")
    print(df[["model", "tcr", "too_hot", "w_bma_tcr", "w_like_ar6_tcr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nN_eff = {n_eff(w_lik):.2f} of {len(df)} | "
          f"weighted-mean TCR = {np.average(df['tcr'], weights=w):.4f} | "
          f"weight span {w.min():.5f} to {w.max():.5f} "
          f"(ratio {w.max() / w.min():.2f})")
    print(f"too-hot share of weight = {w[hot].sum():.4f} "
          f"(equal {hot.sum() / len(df):.4f}, "
          f"bma {df['w_bma_tcr'].to_numpy(float)[hot].sum():.4f}) -> "
          f"hot influence removed = "
          f"{1 - w[hot].sum() / (hot.sum() / len(df)):.1%}")

    df.to_csv(path, index=False)
    print(f"\nweights CSV (columns appended) -> {path}")
    return df


def _production_table():
    """Compute + print the production weight sets for the model universe and
    write the frozen CSV (the *_wt cache every consumer reads back through
    load_frozen()). Returns the DataFrame."""
    import pandas as pd
    from fig01_model_screening import TOO_HOT_TCR_THRESHOLD, models_in_results

    models = models_in_results()
    tcr = universe_metric_values("tcr", models)
    ecs = universe_metric_values("ecs", models)
    models = sorted(tcr)                       # same order universe_metric used

    w_bma, diag = bma_weights(tcr, TCR_TARGET_GAMMA,
                              n_samples=N_SAMPLES_PRODUCTION,
                              estimator="topfrac", keep_frac=KEEP_FRAC["tcr"],
                              prior=DEFAULT_PRIOR, seed=SEED,
                              return_diagnostics=True)
    w_imp = bma_weights(tcr, TCR_TARGET_GAMMA, n_samples=N_SAMPLES_PRODUCTION,
                        estimator="importance", prior=DEFAULT_PRIOR, seed=SEED)
    w_kde = kde_weights(tcr, TCR_TARGET_GAMMA)
    w_ecs = bma_weights(ecs, ECS_TARGET_GAMMA, n_samples=N_SAMPLES_PRODUCTION,
                        estimator="topfrac", keep_frac=KEEP_FRAC["ecs"],
                        prior=DEFAULT_PRIOR, seed=SEED)
    w_lik = likelihood_weights(tcr, TCR_TARGET_AR6_GAMMA)

    df = pd.DataFrame({
        "model": models,
        "tcr": [tcr[m] for m in models],
        "ecs": [ecs[m] for m in models],
        "too_hot": [tcr[m] > TOO_HOT_TCR_THRESHOLD for m in models],
        "w_bma_tcr": [w_bma[m] for m in models],
        "w_bma_tcr_imp": [w_imp[m] for m in models],
        "w_kde_tcr": [w_kde[m] for m in models],
        "w_bma_ecs": [w_ecs[m] for m in models],
        "w_like_ar6_tcr": [w_lik[m] for m in models],
        # provenance (constant columns, so the CSV is self-describing)
        "seed": SEED, "n_samples": N_SAMPLES_PRODUCTION,
        "estimator": "topfrac", "prior": DEFAULT_PRIOR,
        "ar6_gamma_a": TCR_TARGET_AR6_GAMMA[0],
        "ar6_gamma_scale": TCR_TARGET_AR6_GAMMA[1],
    }).sort_values(["tcr", "model"], ascending=[False, True],
                   kind="mergesort")

    uni = 1.0 / len(models)
    import ensemble_method as em
    # The frozen weights and this CSV's too_hot column are TCR-fitted, so the
    # "screen-retained" diagnostics below must be the TCR screen too. Without
    # this, a run left at METRIC='ecs' would print an ECS-retained count beside
    # a TCR too_hot column in the same provenance block.
    em.set_metric("tcr")
    retained = [m for m in models if not em.is_excluded(m)]
    print(f"\n=== production weights | universe = {len(models)} models | "
          f"equal weight = {uni:.4f} | seed = {SEED} | "
          f"n_samples = {N_SAMPLES_PRODUCTION} | prior = {DEFAULT_PRIOR} | "
          f"estimator = topfrac (keep {KEEP_FRAC['tcr']:.2f}) ===")
    print(df.drop(columns=["seed", "n_samples", "estimator", "prior"])
            .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nN_eff  bma_tcr = {n_eff(w_bma):.2f} | importance = "
          f"{n_eff(w_imp):.2f} | kde_tcr = {n_eff(w_kde):.2f} | "
          f"bma_ecs = {n_eff(w_ecs):.2f} | like_ar6 = {n_eff(w_lik):.2f} | "
          f"equal = {len(models)} | screen-retained = {len(retained)}")
    print(f"weighted-mean TCR: bma = {weighted_mean(tcr, w_bma):.3f} | "
          f"importance = {weighted_mean(tcr, w_imp):.3f} | "
          f"kde = {weighted_mean(tcr, w_kde):.3f} | "
          f"like_ar6 = {weighted_mean(tcr, w_lik):.3f} | "
          f"equal = {np.mean(list(tcr.values())):.3f} | "
          f"screen-retained = {np.mean([tcr[m] for m in retained]):.3f} | "
          f"target = {TCR_TARGET_GAMMA[0] * TCR_TARGET_GAMMA[1]:.3f}")
    _report_ar6_target()
    print(f"sampler kept samples (topfrac) = {diag['sampler_ess']:.0f} "
          f"of {N_SAMPLES_PRODUCTION}")

    out_csv = frozen_csv_path()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nweights CSV -> {out_csv}")
    print("NOTE: this rebuild writes the BMA/KDE/equal-tail-TCR columns only. "
          "Re-run `python model_weights.py addlike` and `... addecs` to "
          "restore the appended AR6 TCR and ECS likelihood columns, which "
          "ensemble_method's 'like' and 'ecs*' variants read.")
    return df


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg not in ("validate", "weights", "all", "ar6fit", "addlike",
                   "families", "ecsfit", "addecs", "addgauss", "addvlike"):
        raise SystemExit(
            f"unknown command {arg!r} (expected: validate | weights | all | "
            "ar6fit | addlike | families | ecsfit | addecs | addgauss | "
            "addvlike)")
    if arg == "families":               # target-family sensitivity (report only)
        _report_target_families()
    elif arg == "ar6fit":               # print + verify the fitted AR6 target
        _report_ar6_target()
    elif arg == "ecsfit":               # print + verify the fitted AR6 ECS targets
        _report_ar6_ecs_target()
    elif arg == "addecs":               # append the ECS columns to the frozen CSV
        add_ecs_likelihood_columns()
    elif arg == "addgauss":             # append the joint Gaussian TCR column
        add_gaussjoint_columns()
    elif arg == "addvlike":             # append the very-likely-range column
        add_vlikenorm_columns()
    elif arg == "addlike":              # append AR6 columns to the frozen CSV
        add_likelihood_columns()
    else:
        if arg in ("validate", "all"):
            validate_against_massoud()
        if arg in ("weights", "all"):
            _production_table()
