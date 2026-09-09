"""
For every station calibrated with MSWEP_V280_Past, pick the model with the
highest kge_mod and produce two publication-ready maps:

    best_model_<PCP>.png       categorical -- best model per station
    best_kge_<PCP>.png         continuous  -- best kge_mod per station

Reads:
    RESULTS_DIR/<station>/metrics_summary.csv     (written by on_station_end)
    <gscad_path>/daily/dataframes/static.csv      (lat / long per station)
"""
import os
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy import stats


# Cartopy is used for coastlines / land background. Imported lazily so the
# script still runs (without the basemap) on machines where cartopy isn't
# installed.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False
    print("note: cartopy not installed; maps will render without coastlines. "
          "Install with `pip install cartopy` for the basemap.")

# config.py lives at the repo root (three dirs up from this scripts/ dir). Put
# it on sys.path so the script imports cleanly when launched as
# `python paper/figures/scripts/fig03.py` (mirrors fig02 / fig05). Done via
# sys.path here rather than the shell PYTHONPATH.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from config import CONFIG as C
# Nature-style bold top-left panel letters (multi-panel figures only)
from figstyle import add_panel_letters

MODELS = ['hbv', 'gr4j', 'gr6j', 'simhyd', 'hymod', 'xaj']

PCP = 'MSWEP_V280_Past'
# Human-readable PCP name for titles / suptitle only. PCP itself stays the raw
# data key (filters metrics_summary.csv) and the output-filename stem.
PCP_DISPLAY_NAMES = {
    'MSWEP_V280_Past': 'MSWEP v2.80 (Past)',
}
PCP_LABEL = PCP_DISPLAY_NAMES.get(PCP, PCP)

# Which split the lumped (conceptual) model results are based on. Switch here
# (or via the `phase=` kwarg of collect_best_per_station). 'validation'
# requires the run to have used cal_frac < 1.0 so metrics_summary.csv contains
# valid_kge_mod. The LSTM is always its own test set -- unaffected by this.
PHASE = 'validation'   # 'calibration' | 'validation'
PHASE_METRIC = {
    'calibration': 'calib_kge_mod',
    'validation':  'valid_kge_mod',
}
METRIC = PHASE_METRIC[PHASE]

# Internal metric column -> label shown in titles / colorbars. The
# calibration, validation and LSTM-test columns are all the modified KGE,
# so every variant reads as "KGE" in the figure.
METRIC_DISPLAY = {
    'calib_kge_mod': 'KGE',
    'valid_kge_mod': 'KGE',
    'kge_mod':       'KGE',
}
METRIC_LABEL = METRIC_DISPLAY.get(METRIC, METRIC)

#RESULTS_DIR = os.path.join(C.results_dir, '36_6_0.7_40_24_48_0.01_5_25')
RESULTS_DIR = C.calib_path

# Panel (d): global LSTM test metrics. The CSV is indexed by station name with
# one column per metric; panel (d) is coloured by its ``kge_mod`` column,
# restricted to the same catchments shown in panels (a)-(c).
# LSTM_METRICS_FP = ('/ibex/user/abbaa0a/results/hydrodeepai/'
#                    '46169901_all_GlobalLSTM1_ddp/test_metrics.csv')
LSTM_METRICS_FP = ('/ibex/user/abbaa0a/results/hydrodeepai/'
                   '47122937_all_GlobalLSTM1_ddp/inference_57/test_metrics.csv')


# Display names for the legend / titles. Internal config uses lowercase keys
# (hbv, gr4j, ...) but the literature uses these standard forms.
MODEL_DISPLAY_NAMES = {
    'hbv':    'HBV',
    'gr4j':   'GR4J',
    'gr6j':   'GR6J',
    'simhyd': 'SIMHYD',
    'hymod':  'HyMod',
    'xaj':    'XAJ',
}

# Professional axis / title labels for static features (the raw column names
# are terse internal identifiers). Falls back to the raw name if unmapped.
FEATURE_AXIS_LABELS = {
    'KGZ_detailed': 'Köppen–Geiger climate class',
    'KGZ_major':    'Köppen–Geiger major climate zone',
    'source':       'Data source',
}


# Non-font rcParams specific to this figure. Fonts, savefig dpi/bbox and the
# TrueType embedding now come from figstyle.py (imported above), which is the
# single source of truth for type across every figure in the paper -- do not
# re-set font sizes here or fig03 drifts away from the rest again.
mpl.rcParams.update({
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 110,
})


def _collect_one_station(task):
    """Read one station's metrics_summary.csv and reduce it to a single result.

    Worker for the parallel scan in ``collect_best_per_station``. Defined at
    module level (not a closure) so it stays picklable for multiprocessing.Pool.
    ``task`` is ``(results_dir, stn, pcp, metric)``.

    Returns ``(metric_seen, row)``:
      metric_seen -- True if the file held ``pcp`` rows *and* the ``metric``
                     column (even if every value was NaN); drives the helpful
                     "phase column never found" error in the caller.
      row         -- the per-station dict, or None when there is nothing usable.
    This is exactly the same per-station computation as the original serial
    loop, so the result is independent of worker count / order.
    """
    results_dir, stn, pcp, metric = task
    summary_fp = os.path.join(results_dir, stn, 'metrics_summary.csv')
    if not os.path.isfile(summary_fp):
        return (False, None)
    df = pd.read_csv(summary_fp)
    df = df[df['pcp_data_name'] == pcp]
    if df.empty or metric not in df.columns:
        return (False, None)
    df = df.dropna(subset=[metric])
    if df.empty:
        return (True, None)
    best = df.loc[df[metric].idxmax()]
    worst = df.loc[df[metric].idxmin()]
    per_model = dict(zip(df['model'], df[metric]))
    row = {
        'station': stn,
        'best_model': best['model'],
        'best_kge_mod': float(best[metric]),
        'worst_model': worst['model'],
        'worst_kge_mod': float(worst[metric]),
        # Inter-model spread at this station -- how much model choice matters
        # here. Always >= 0 since best >= worst by construction.
        'kge_spread': float(best[metric]) - float(worst[metric]),
        'n_models': int(len(df)),
    }
    # Per-model KGE (one column per configured model) so downstream plots can
    # show each model's distribution, not just best/worst. A model missing for
    # this station -> NaN (dropped per-series by the plots).
    for m in MODELS:
        row[f'kge_{m}'] = float(per_model[m]) if m in per_model else np.nan
    return (True, row)


def collect_best_per_station(results_dir: str, pcp: str,
                             phase: str = PHASE,
                             n_workers: int = 1) -> pd.DataFrame:
    """Walk ``results_dir`` and return one row per station with the winning
    conceptual model for ``pcp``.

    ``phase`` selects which metrics_summary.csv column the conceptual-model
    results use:
        'calibration' -> calib_kge_mod
        'validation'  -> valid_kge_mod   (run must have used cal_frac < 1.0)
    The LSTM (loaded separately via load_lstm_kge) is always its own test set
    and is NOT affected by this argument. Joined with lat/long from static.csv.

    NOTE: the returned columns keep generic names ('best_kge_mod',
    'worst_kge_mod', 'kge_<model>', ...) regardless of phase -- they hold
    whichever phase's KGE was selected here.
    """
    if phase not in PHASE_METRIC:
        raise SystemExit(
            f"phase must be one of {list(PHASE_METRIC)}, got {phase!r}")
    metric = PHASE_METRIC[phase]

    static_fpath = os.path.join(C.gscad_path, 'dataframes', 'static.csv')
    static_df = pd.read_csv(static_fpath, index_col='stations', comment='#')

    # Scan every station's metrics_summary.csv. The per-station reduction lives
    # in the module-level worker `_collect_one_station`, so the scan runs either
    # serially (n_workers <= 1) or across a multiprocessing Pool. Station order
    # is the sorted-listdir order in BOTH paths (Pool.map preserves input
    # order), and each station is independent, so the result is identical
    # regardless of n_workers.
    stns = sorted(os.listdir(results_dir))
    tasks = [(results_dir, stn, pcp, metric) for stn in stns]

    if n_workers and n_workers > 1:
        # Chunk so ~20k tiny tasks don't pay per-task IPC overhead.
        chunksize = max(1, len(tasks) // (n_workers * 8))
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_collect_one_station, tasks, chunksize=chunksize)
    else:
        results = [_collect_one_station(t) for t in tasks]

    # did *any* file even have the requested phase column?
    metric_seen = any(seen for seen, _ in results)
    rows = [row for _, row in results if row is not None]

    best = pd.DataFrame(rows).set_index('station')
    if best.empty:
        if not metric_seen:
            raise SystemExit(
                f"'{metric}' (phase={phase!r}) not found in any "
                f"metrics_summary.csv under {results_dir} for pcp {pcp!r}. "
                f"phase='validation' needs a run with cal_frac < 1.0."
            )
        raise SystemExit(f"No stations found with {pcp} results under {results_dir}")

    best = best.join(static_df[['lat', 'long']], how='left')
    missing = best[best[['lat', 'long']].isna().any(axis=1)]
    if len(missing):
        print(f"warning: {len(missing)} stations have no lat/long in static.csv; dropping")
        best = best.drop(missing.index)
    return best


def _make_world_axes(fig, *subplot_args):
    """Create a world-map axes with a coastline / land basemap if cartopy is
    available, otherwise a plain matplotlib axes with the same extent.

    ``subplot_args`` is forwarded to ``fig.add_subplot`` so callers can
    request a grid slot (e.g. ``2, 2, 1``); defaults to ``(1, 1, 1)``.

    Returns ``(ax, transform_kwargs)``. ``transform_kwargs`` is a dict that
    should be unpacked into ``ax.scatter`` so points are projected correctly
    when cartopy is in use; it's empty otherwise.
    """
    if not subplot_args:
        subplot_args = (1, 1, 1)
    if _HAS_CARTOPY:
        # Robinson is the conventional projection for global hydrology maps:
        # far less high-latitude area distortion than PlateCarree and a more
        # compact globe (less ocean whitespace) for a multi-panel layout.
        ax = fig.add_subplot(*subplot_args, projection=ccrs.Robinson())
        ax.set_global()
        # Ocean is left white (axes background) so the blue end of the
        # diverging colormap and HBV's blue markers don't fight a blue sea.
        # Land gets a very light neutral grey -- just enough to read shape.
        ax.set_facecolor('white')
        ax.add_feature(cfeature.LAND, facecolor='#ededed', zorder=0)
        ax.add_feature(cfeature.COASTLINE, edgecolor='#555555',
                       linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, edgecolor='#aaaaaa',
                       linewidth=0.3, linestyle=':', zorder=1)
        # Graticule only, no per-panel lon/lat labels: across four global
        # maps the repeated tick labels are clutter, and Robinson label
        # placement is unreliable across cartopy versions. The grid lines
        # alone still convey orientation.
        ax.gridlines(draw_labels=False, linewidth=0.5,
                     color='#888888', alpha=0.5, linestyle=':')
        transform_kwargs = {'transform': ccrs.PlateCarree()}
    else:
        ax = fig.add_subplot(*subplot_args)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-60, 91, 30))
        ax.grid(True, which='major', linestyle=':', linewidth=0.6,
                color='#888888', alpha=0.6)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_facecolor('#f7f7f7')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        transform_kwargs = {}
    return ax, transform_kwargs


def plot_best_model_map(best: pd.DataFrame, out_path: str) -> None:
    """Categorical scatter: one colour per hydrological model."""
    # Okabe-Ito colourblind-safe qualitative palette. Yellow (#F0E442) and
    # black are omitted -- low contrast as small markers on the light land
    # fill. Hues kept close to the old mapping so models stay recognisable.
    palette = {
        'hbv':    '#0072B2',  # blue
        'gr4j':   '#D55E00',  # vermillion
        'gr6j':   '#009E73',  # bluish green
        'simhyd': '#E69F00',  # orange
        'hymod':  '#CC79A7',  # reddish purple
        'xaj':    '#56B4E9',  # sky blue
    }
    # Fallback for any model name not in the palette (shouldn't happen).
    model_colors = {m: palette.get(m, plt.get_cmap('tab10')(i))
                    for i, m in enumerate(MODELS)}

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor('white')
    ax, tkw = _make_world_axes(fig)

    counts = best['best_model'].value_counts()
    # Plot models with smaller counts last so they sit on top of dense ones.
    for model_name in counts.sort_values().index:
        sub = best[best['best_model'] == model_name]
        ax.scatter(
            sub['long'], sub['lat'],
            s=6,
            c=[model_colors[model_name]],
            label=f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}",
            linewidths=0,
            alpha=0.7,
            zorder=3,
            **tkw,
        )

    ax.set_title(f'Best-performing hydrological model per catchment\n'
                 f'{PCP_LABEL}:  ranked by {METRIC}  (N = {len(best):,} catchments)',
                 pad=12)

    leg = ax.legend(
        title='Model',
        loc='lower left',
        ncol=1,
        frameon=True,
        framealpha=0.95,
        edgecolor='#333333',
        borderpad=0.7,
        labelspacing=0.55,
        handletextpad=0.6,
        markerscale=3.8,
    )
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_best_kge_map(best: pd.DataFrame, out_path: str) -> None:
    """Continuous scatter coloured by the best kge_mod value at each station.

    Diverging colormap (RdYlBu) centred on 0.5 -- a common 'satisfactory'
    threshold for KGE: red below, blue above. Values are clipped to [-0.5, 1]
    so a handful of very-poor catchments don't compress the rest of the range.
    """
    vmin, vcenter, vmax = -0.5, 0.5, 1.0
    scores = best['best_kge_mod'].clip(lower=vmin, upper=vmax)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = plt.get_cmap('RdYlBu')

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor('white')
    ax, tkw = _make_world_axes(fig)

    # Order points by score so high-performing catchments (blue) are drawn on
    # top of poor ones (red) -- avoids hiding success behind failure visually.
    order = scores.sort_values().index
    sc = ax.scatter(
        best.loc[order, 'long'],
        best.loc[order, 'lat'],
        c=scores.loc[order],
        cmap=cmap, norm=norm,
        s=14,
        linewidths=0,
        alpha=0.95,
        zorder=3,
        **tkw,
    )

    ax.set_title(f'Best-model performance per catchment\n'
                 f'{PCP_LABEL}:  {METRIC_LABEL}  (N = {len(best):,} catchments)',
                 pad=12)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, extend='both')
    cbar.set_label(f'{METRIC_LABEL}', fontsize=13, fontweight='normal')
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_edgecolor('#333333')
    # Mark the 0.5 'satisfactory' threshold the cmap is centred on.
    cbar.ax.axhline(0.5, color='black', linewidth=0.8)

    # Stats annotation in an unobtrusive corner box.
    raw = best['best_kge_mod']
    txt = (f"median = {raw.median():.2f}\n"
           f"mean   = {raw.mean():.2f}\n"
           f"≥ 0.5 : {(raw >= 0.5).mean()*100:.1f} %")
    ax.text(
        0.012, 0.025, txt,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=11, family='monospace',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor='#333333', linewidth=0.8, alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_kge_spread_map(best: pd.DataFrame, out_path: str) -> None:
    """Continuous scatter coloured by (best - worst) kge_mod per station.

    Large spread => model choice matters a lot here (one structure captures
    the catchment, another fails). Small spread => all six models agree
    (could be uniformly good OR uniformly bad -- read alongside the
    best_kge map to disambiguate).

    The spread is non-negative by construction, so a sequential colormap
    (``viridis``) is appropriate. The upper end of the colour scale is
    capped at 1.5 -- a semantically meaningful threshold: best is bounded
    above by 1.0, so spread = 1.5 implies worst <= -0.5 (worse than the
    long-term-mean benchmark, KGE ~ -0.41). Catchments where one model
    has completely diverged are sent to the ``extend='max'`` triangle so
    they don't compress the meaningful range.
    """
    vmin, vmax = 0.0, 1.5
    spread = best['kge_spread'].clip(lower=vmin, upper=vmax)
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor('white')
    ax, tkw = _make_world_axes(fig)

    # Draw high-spread points on top -- they're the interesting ones.
    order = spread.sort_values().index
    sc = ax.scatter(
        best.loc[order, 'long'],
        best.loc[order, 'lat'],
        c=spread.loc[order],
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=14,
        linewidths=0,
        alpha=0.95,
        zorder=3,
        **tkw,
    )

    ax.set_title(f'Inter-model performance spread per catchment\n'
                 f'{PCP_LABEL}:  best − worst {METRIC_LABEL}  (N = {len(best):,} catchments)',
                 pad=12)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, extend='max')
    cbar.set_label(f'best − worst {METRIC_LABEL}', fontsize=13, fontweight='normal')
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_edgecolor('#333333')

    raw = best['kge_spread']
    txt = (f"median = {raw.median():.2f}\n"
           f"mean   = {raw.mean():.2f}\n"
           f"> 0.3 : {(raw > 0.3).mean()*100:.1f} %")
    ax.text(
        0.012, 0.025, txt,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=11, family='monospace',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor='#333333', linewidth=0.8, alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_kge_relative_spread_map(best: pd.DataFrame, out_path: str) -> None:
    """Scatter coloured by (best - worst) / best -- scale-aware spread.

    A raw spread of 0.2 means very different things at a station where the
    best model scores 0.95 (=> 21% relative spread, models basically agree)
    vs one where the best scores 0.30 (=> 67% relative spread, large
    disagreement relative to the available skill). This ratio captures that.

    Stations where best_kge_mod <= 0.05 are excluded -- the ratio is
    numerically unstable and conceptually meaningless when no model has any
    useful skill. The excluded count is shown in the stats box.
    """
    eligible = best[best['best_kge_mod'] > 0.05].copy()
    excluded = len(best) - len(eligible)
    eligible['rel_spread'] = eligible['kge_spread'] / eligible['best_kge_mod']

    vmin, vmax = 0.0, 1.5
    rel = eligible['rel_spread'].clip(lower=vmin, upper=vmax)
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(15, 7.5))
    fig.patch.set_facecolor('white')
    ax, tkw = _make_world_axes(fig)

    order = rel.sort_values().index
    sc = ax.scatter(
        eligible.loc[order, 'long'],
        eligible.loc[order, 'lat'],
        c=rel.loc[order],
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=14, linewidths=0, alpha=0.95, zorder=3, **tkw,
    )

    ax.set_title(f'Relative inter-model spread per catchment\n'
                 f'{PCP_LABEL}:  (best − worst) / best {METRIC_LABEL}  '
                 f'(N = {len(eligible):,} catchments)',
                 pad=12)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, extend='max')
    cbar.set_label(f'(best − worst) / best  {METRIC_LABEL}',
                   fontsize=13, fontweight='normal')
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_edgecolor('#333333')

    raw = eligible['rel_spread']
    txt = (f"median = {raw.median():.2f}\n"
           f"mean   = {raw.mean():.2f}\n"
           f"> 0.5 : {(raw > 0.5).mean()*100:.1f} %\n"
           f"excluded (best ≤ 0.05): {excluded:,}")
    ax.text(
        0.012, 0.025, txt,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=11, family='monospace',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                  edgecolor='#333333', linewidth=0.8, alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Panel-level helpers used by the combined 4-panel figure. Each one just
# draws the scatter onto a given axes -- titles, colorbars, legends, and
# stats boxes are handled by the caller so layout decisions live in one
# place (plot_combined_panels) rather than being spread across helpers.
# ---------------------------------------------------------------------------

# Palette for the categorical best-model panel. Pulled out of
# plot_best_model_map so the combined figure uses identical colours.
_MODEL_PALETTE = {
    # Okabe-Ito colourblind-safe palette (yellow & black omitted).
    'hbv':    '#0072B2',  # blue
    'gr4j':   '#D55E00',  # vermillion
    'gr6j':   '#009E73',  # bluish green
    'simhyd': '#E69F00',  # orange
    'hymod':  '#CC79A7',  # reddish purple
    'xaj':    '#56B4E9',  # sky blue
}


def _draw_best_model_panel(ax, tkw, best: pd.DataFrame, marker_size: int = 9):
    model_colors = {m: _MODEL_PALETTE.get(m, plt.get_cmap('tab10')(i))
                    for i, m in enumerate(MODELS)}
    counts = best['best_model'].value_counts()
    for model_name in counts.sort_values().index:
        sub = best[best['best_model'] == model_name]
        ax.scatter(
            sub['long'], sub['lat'],
            s=marker_size,
            c=[model_colors[model_name]],
            label=f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}",
            linewidths=0, alpha=0.7, zorder=3, **tkw,
        )


def _add_stats_box(ax, lines, fontsize: int = 12):
    """Lower-left stats annotation with a TRANSPARENT fill so the catchments
    underneath stay visible. Tucked into the empty southern-ocean corner of the
    Robinson map (down + left of centre) to use that otherwise-dead space."""
    ax.text(
        0.43, 0.05, '\n'.join(lines),
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=fontsize, family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='none',
                  edgecolor='#333333', linewidth=0.6),
        zorder=5,
    )


def _draw_best_kge_panel(ax, tkw, best: pd.DataFrame, marker_size: int = 9):
    vmin, vcenter, vmax = -0.5, 0.5, 1.0
    scores = best['best_kge_mod'].clip(lower=vmin, upper=vmax)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    order = scores.sort_values().index
    sc = ax.scatter(
        best.loc[order, 'long'], best.loc[order, 'lat'],
        c=scores.loc[order], cmap=plt.get_cmap('RdYlBu'), norm=norm,
        s=marker_size, linewidths=0, alpha=0.95, zorder=3, **tkw,
    )
    raw = best['best_kge_mod']
    _add_stats_box(ax, [
        f"median = {raw.median():.2f}",
        f"mean   = {raw.mean():.2f}",
        f"≥ 0.5 : {(raw >= 0.5).mean()*100:.1f} %",
    ])
    return sc


def _draw_spread_panel(ax, tkw, best: pd.DataFrame, marker_size: int = 9):
    vmin, vmax = 0.0, 1.5
    spread = best['kge_spread'].clip(lower=vmin, upper=vmax)
    order = spread.sort_values().index
    sc = ax.scatter(
        best.loc[order, 'long'], best.loc[order, 'lat'],
        c=spread.loc[order], cmap=plt.get_cmap('viridis'),
        vmin=vmin, vmax=vmax,
        s=marker_size, linewidths=0, alpha=0.95, zorder=3, **tkw,
    )
    raw = best['kge_spread']
    _add_stats_box(ax, [
        f"median = {raw.median():.2f}",
        f"mean   = {raw.mean():.2f}",
        f"> 0.3 : {(raw > 0.3).mean()*100:.1f} %",
    ])
    return sc


def load_lstm_kge(best: pd.DataFrame,
                  metrics_fp: str = LSTM_METRICS_FP) -> pd.Series:
    """Return LSTM test ``kge_mod`` aligned to the catchments in ``best``.

    test_metrics.csv is indexed by station name with one column per metric and
    covers ~27k catchments. Panels (a)-(c) only show the catchments calibrated
    for ``PCP``, so we intersect on station name and keep just those -- panel
    (d) then has the same markers as the first three panels.
    """
    lstm = pd.read_csv(metrics_fp, index_col=0)
    if 'kge_mod' not in lstm.columns:
        raise SystemExit(
            f"'kge_mod' not a column in {metrics_fp}; found {list(lstm.columns)}"
        )
    kge = lstm['kge_mod'].dropna()

    common = best.index.intersection(kge.index)
    if len(common) == 0:
        raise SystemExit(
            "No station overlap between the LSTM test_metrics.csv and the "
            f"{len(best):,} calibrated catchments.\n"
            f"  example calibrated station : {best.index[0]!r}\n"
            f"  example LSTM station       : {kge.index[0]!r}\n"
            "Station-name formats differ -- reconcile them in load_lstm_kge()."
        )
    missing = len(best) - len(common)
    if missing:
        print(f"note: {missing:,} of {len(best):,} calibrated stations are "
              f"absent from the LSTM file; panel (d) shows {len(common):,}")
    else:
        print(f"panel (d): all {len(common):,} calibrated stations have an "
              f"LSTM test result")
    return kge.loc[common]


def _draw_kge_map_panel(ax, tkw, best: pd.DataFrame, kge: pd.Series,
                        marker_size: int = 9):
    """Map any per-station KGE series onto ``best``'s lat/long, using the
    common RdYlBu / TwoSlopeNorm(-0.5, 0.5, 1.0) mapping shared by every KGE
    map in this module -- so panels drawn by this helper are directly
    comparable off one colour scale. ``kge`` is indexed by station; only the
    stations it carries are drawn."""
    vmin, vcenter, vmax = -0.5, 0.5, 1.0
    scores = kge.clip(lower=vmin, upper=vmax)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    order = scores.sort_values().index
    sc = ax.scatter(
        best.loc[order, 'long'], best.loc[order, 'lat'],
        c=scores.loc[order], cmap=plt.get_cmap('RdYlBu'), norm=norm,
        s=marker_size, linewidths=0, alpha=0.95, zorder=3, **tkw,
    )
    _add_stats_box(ax, [
        f"median = {kge.median():.2f}",
        f"mean   = {kge.mean():.2f}",
        f"≥ 0.5 : {(kge >= 0.5).mean()*100:.1f} %",
    ])
    return sc


def _draw_lstm_panel(ax, tkw, best: pd.DataFrame, lstm_kge: pd.Series,
                     marker_size: int = 9):
    """LSTM test kge_mod scatter. Identical colour mapping to the best-model
    KGE panel (RdYlBu, TwoSlopeNorm centred on 0.5) so the LSTM and the best
    process model can be compared off the same scale."""
    return _draw_kge_map_panel(ax, tkw, best, lstm_kge, marker_size=marker_size)


def plot_combined_panels(best: pd.DataFrame, out_path: str) -> None:
    """4-panel summary figure. Panels (a)-(c) summarise the lumped conceptual
    models; panel (d) is the global LSTM on the same catchments:
    (a) best conceptual model, (b) best conceptual-model KGE,
    (c) conceptual inter-model spread, (d) LSTM test KGE.

    Each map is drawn on its own world-map axes; colorbars live to the right
    of panels b, c and d so the geographic extent of all maps stays aligned
    within a row.
    """
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    fig.patch.set_facecolor('white')
    # Tighten constrained_layout gaps to trim inter-panel and figure-margin
    # white space (figure size / ratios kept at their originals).
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02,
                                    wspace=0.02, hspace=0.03)

    # --- Panel (a): best model -------------------------------------------
    ax_a, tkw_a = _make_world_axes(fig, 2, 2, 1)
    _draw_best_model_panel(ax_a, tkw_a, best, marker_size=2)
    ax_a.set_title('Best conceptual model',
                   fontsize=13, fontweight='normal', loc='left', pad=6)
    leg = ax_a.legend(
        title='Model', loc='center left', bbox_to_anchor=(0.06, 0.5),
        ncol=1, frameon=True,
        framealpha=0.75, edgecolor='#333333', borderpad=0.45,
        labelspacing=0.35, handletextpad=0.5, markerscale=6,
        fontsize=10, title_fontsize=11,
    )
    leg.get_frame().set_linewidth(0.6)

    # --- Panel (b): best KGE ---------------------------------------------
    ax_b, tkw_b = _make_world_axes(fig, 2, 2, 2)
    sc_b = _draw_best_kge_panel(ax_b, tkw_b, best, marker_size=2)
    ax_b.set_title(f'Best-conceptual {METRIC_LABEL}',
                   fontsize=13, fontweight='normal', loc='left', pad=6)
    # Colorbar deferred: (b) and (d) share one common KGE scale, so a single
    # shared colorbar is added after panel (d) is drawn.

    # --- Panel (c): inter-model spread -----------------------------------
    ax_c, tkw_c = _make_world_axes(fig, 2, 2, 3)
    sc_c = _draw_spread_panel(ax_c, tkw_c, best, marker_size=2)
    ax_c.set_title('Conceptual inter-model spread',
                   fontsize=13, fontweight='normal', loc='left', pad=6)
    cbar_c = fig.colorbar(sc_c, ax=ax_c, fraction=0.035, pad=0.02, extend='max')
    cbar_c.set_label(f'best − worst {METRIC_LABEL}', fontsize=12, fontweight='normal')
    cbar_c.ax.tick_params(labelsize=11)
    cbar_c.outline.set_edgecolor('#333333')

    # --- Panel (d): LSTM test performance --------------------------------
    # Same RdYlBu / TwoSlopeNorm(-0.5, 0.5, 1.0) as panel (b) so the LSTM and
    # the best process model are read off an identical colour scale. Only the
    # catchments present in `best` (panels a-c) are drawn.
    ax_d, tkw_d = _make_world_axes(fig, 2, 2, 4)
    lstm_kge = load_lstm_kge(best)
    sc_d = _draw_lstm_panel(ax_d, tkw_d, best, lstm_kge, marker_size=2)
    ax_d.set_title(f'LSTM test {METRIC_DISPLAY.get("kge_mod", "KGE")}',
                   fontsize=13, fontweight='normal', loc='left', pad=6)

    # --- Shared KGE colorbar for panels (b) and (d) ----------------------
    # (b) and (d) use an identical RdYlBu / TwoSlopeNorm(-0.5, 0.5, 1.0)
    # mapping; one colorbar spanning the right column makes the
    # best-conceptual vs LSTM comparison a direct visual read-off. sc_b and
    # sc_d are interchangeable as the mappable since the norm/cmap match.
    cbar_bd = fig.colorbar(sc_b, ax=[ax_b, ax_d], fraction=0.025, pad=0.02,
                           extend='both')
    cbar_bd.set_label(METRIC_LABEL, fontsize=12, fontweight='normal')
    cbar_bd.ax.tick_params(labelsize=11)
    cbar_bd.outline.set_edgecolor('#333333')
    cbar_bd.ax.axhline(0.5, color='black', linewidth=0.7)
    # Uniform, round tick labels on the shared b/d colorbar (the TwoSlopeNorm
    # default ticker otherwise lands them on uneven values).
    cbar_bd.set_ticks([-0.5, 0.0, 0.5, 1.0])

    fig.suptitle(
        f'{PCP_LABEL}:  lumped conceptual models vs. LSTM  '
        f'(N = {len(best):,} catchments)',
        fontsize=15, fontweight='normal',
    )

    # explicit order: the 2x2 grid is filled a=top-left, b=top-right,
    # c=bottom-left, d=bottom-right, matching the letters the titles used to carry.
    add_panel_letters([ax_a, ax_b, ax_c, ax_d])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Alternative layout: the four maps in a left 2x2 block, plus a right-hand
# summary column (ECDF + Köppen boxplots) that uses the otherwise-empty
# Robinson whitespace. Produced *alongside* plot_combined_panels (separate
# output file) so both layouts stay available -- nothing above is changed.
# ---------------------------------------------------------------------------

# Köppen major-class labels. Falls back to str(class) for any code (numeric or
# otherwise) not listed here, so an unexpected encoding still renders.
_KGZ_MAJOR_NAMES = {
    'A': 'A: Tropical',
    'B': 'B: Arid',
    'C': 'C: Temperate',
    'D': 'D: Continental',
    'E': 'E: Polar',
}


def _ecdf(values):
    """(sorted_values, cumulative_fraction) for an empirical CDF; NaNs dropped."""
    v = np.asarray(values, dtype=float)
    v = np.sort(v[~np.isnan(v)])
    if len(v) == 0:
        return v, v
    return v, np.arange(1, len(v) + 1) / len(v)


def _draw_ecdf_panel(ax, best: pd.DataFrame, lstm_kge: pd.Series, common,
                     models=None, model_linewidth: float = 1.3,
                     lstm_linewidth: float = 2.6,
                     lstm_color: str = '#000000'):
    """Overlaid ECDFs of KGE on the shared catchment set: each conceptual
    model in ``models`` individually (colours match panel (a)) and the LSTM.
    ``models`` defaults to all of ``MODELS``; pass e.g. ['hbv'] for the
    two-model (HBV vs LSTM) variant. ``lstm_color`` defaults to black, which
    reads as a reference line against six coloured model curves; the two-model
    figure overrides it to match its boxplot panel. The gap between the model
    curves and the LSTM curve is the headline result."""
    # Individual conceptual models -- thin lines, panel-(a) palette.
    for m in (MODELS if models is None else models):
        col = f'kge_{m}'
        if col not in best.columns:
            continue
        x, y = _ecdf(best.loc[common, col])
        if len(x) == 0:
            continue
        med = float(np.median(x))  # x is the NaN-dropped, sorted array
        ax.plot(x, y, color=_MODEL_PALETTE.get(m, '#999999'),
                linestyle='-', linewidth=model_linewidth, alpha=0.85,
                label=f"{MODEL_DISPLAY_NAMES.get(m, m)}  ({med:.2f})")

    # Bold reference: the LSTM (the per-station best-conceptual envelope line
    # was dropped here on request -- the individual model curves remain).
    refs = [
        ('LSTM (test)', lstm_kge.loc[common], lstm_color, '-', lstm_linewidth),
    ]
    for label, vals, color, ls, lw in refs:
        x, y = _ecdf(vals)
        if len(x) == 0:
            continue
        med = float(np.median(x))  # x is the NaN-dropped, sorted array
        ax.plot(x, y, color=color, linestyle=ls, linewidth=lw,
                label=f"{label}  ({med:.2f})")

    ax.axvline(0.5, color='black', linewidth=0.7, linestyle=':')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(METRIC_LABEL)
    ax.set_ylabel('Cumulative fraction')
    ax.set_title(f'{METRIC_LABEL} distribution',
                 fontsize=13, fontweight='normal', loc='left', pad=6)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=10, frameon=True,
              framealpha=0.9, edgecolor='#333333', labelspacing=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def _draw_kgz_box_panel(ax, best: pd.DataFrame, lstm_kge: pd.Series, common,
                        min_per_class: int = 10,
                        concept_col: str = 'best_kge_mod',
                        concept_label: str = 'Best conceptual',
                        concept_color: str = '#0072B2',
                        lstm_color: str = '#D55E00'):
    """Paired KGE boxplots per Köppen major zone (``concept_col`` of ``best``
    vs LSTM). Defaults reproduce the original best-conceptual comparison;
    pass e.g. concept_col='kge_hbv' for a single-model comparison. The two
    colours are exposed so a caller can keep this panel and the ECDF panel on
    one colour key. Degrades gracefully if static.csv lacks the KGZ_major
    column."""
    static = _load_static_with_extras()
    col = 'KGZ_major'
    if col not in static.columns:
        ax.text(0.5, 0.5, f"'{col}' not in static.csv\n(boxplot skipped)",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#999999', style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if concept_col not in best.columns:
        raise SystemExit(
            f"'{concept_col}' not a column of `best`; available: "
            f"{[c for c in best.columns if c.startswith('kge_') or c.endswith('kge_mod')]}"
        )

    df = pd.DataFrame({
        'kgz': static.reindex(common)[col],
        'concept': best.loc[common, concept_col],
        'lstm': lstm_kge.loc[common],
    }).dropna(subset=['kgz'])

    counts = df['kgz'].value_counts()
    classes = sorted([c for c in counts.index if counts[c] >= min_per_class],
                     key=lambda c: str(c))

    c_concept, c_lstm = concept_color, lstm_color
    data_c, data_l, posc, posl, ticks, labels = [], [], [], [], [], []
    pos = 0
    for cls in classes:
        sub = df[df['kgz'] == cls]
        dc = sub['concept'].dropna().values
        dl = sub['lstm'].dropna().values
        if len(dc) == 0 or len(dl) == 0:
            continue
        data_c.append(dc)
        data_l.append(dl)
        posc.append(pos - 0.18)
        posl.append(pos + 0.18)
        ticks.append(pos)
        labels.append(f"{_KGZ_MAJOR_NAMES.get(str(cls), str(cls))} (n={len(sub):,})")
        pos += 1

    if not data_c:
        ax.text(0.5, 0.5, "no Köppen classes with enough data",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#999999', style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    base_bp = dict(widths=0.30, patch_artist=True, showfliers=False,
                   medianprops=dict(color='black', linewidth=1.2))
    ax.boxplot(data_c, positions=posc, **base_bp,
               boxprops=dict(facecolor=c_concept, edgecolor=c_concept, alpha=0.65),
               whiskerprops=dict(color=c_concept),
               capprops=dict(color=c_concept))
    ax.boxplot(data_l, positions=posl, **base_bp,
               boxprops=dict(facecolor=c_lstm, edgecolor=c_lstm, alpha=0.65),
               whiskerprops=dict(color=c_lstm),
               capprops=dict(color=c_lstm))

    ax.axhline(0.5, color='black', linewidth=0.7, linestyle=':')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=10, rotation=25, ha='right',
                       rotation_mode='anchor')
    ax.set_xlim(-0.6, (len(ticks) - 1) + 0.6)
    ax.set_ylim(-1.0, 1.05)
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title('KGE by Köppen major zone',
                 fontsize=13, fontweight='normal', loc='left', pad=6)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=c_concept, edgecolor=c_concept, alpha=0.65,
              label=concept_label),
        Patch(facecolor=c_lstm, edgecolor=c_lstm, alpha=0.65,
              label='LSTM (test)'),
    ], loc='lower left', fontsize=10, frameon=True, framealpha=0.9,
       edgecolor='#333333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def plot_maps_with_summary(best: pd.DataFrame, out_path: str) -> None:
    """Same four maps as plot_combined_panels, but in a left 2x2 block with a
    right-hand summary column:
        (e) ECDF of KGE -- best-conceptual vs LSTM (vs worst-conceptual)
        (f) KGE boxplots by Köppen major zone -- conceptual vs LSTM
    Written to its own file so this layout and plot_combined_panels coexist.
    """
    lstm_kge = load_lstm_kge(best)
    common = best.index.intersection(lstm_kge.index)

    # Sized for embedding at ~\linewidth in main.tex: a smaller figure means the
    # (fixed point-size) fonts render larger relative to the figure, so they stay
    # readable once LaTeX scales the PNG down (was (23, 10)). The short height
    # keeps the 2x2 map cells close to the Robinson maps' own ~2:1 aspect, which
    # trims the big empty band that otherwise opens between the map rows.
    fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
    fig.patch.set_facecolor('white')
    # Tighten constrained_layout gaps to trim inter-panel and figure-margin
    # white space.
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02,
                                    wspace=0.02, hspace=0.03)

    outer = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.0])
    gs_maps = outer[0, 0].subgridspec(2, 2)
    gs_summ = outer[0, 1].subgridspec(2, 1)

    # --- Panel (a): best conceptual model --------------------------------
    ax_a, tkw_a = _make_world_axes(fig, gs_maps[0, 0])
    _draw_best_model_panel(ax_a, tkw_a, best, marker_size=2)
    ax_a.set_title('Best conceptual model',
                   fontsize=13, fontweight='normal', loc='left', pad=6)
    leg = ax_a.legend(
        loc='lower center', bbox_to_anchor=(0.5, 0.0),
        ncol=2, frameon=True,
        framealpha=0.75, edgecolor='#333333', borderpad=0.45,
        labelspacing=0.35, handletextpad=0.5, markerscale=6,
        fontsize=10, title_fontsize=11,
    )
    leg.get_frame().set_linewidth(0.6)

    # --- Panel (b): best conceptual KGE ----------------------------------
    ax_b, tkw_b = _make_world_axes(fig, gs_maps[0, 1])
    sc_b = _draw_best_kge_panel(ax_b, tkw_b, best, marker_size=2)
    ax_b.set_title(f'Best-conceptual {METRIC_LABEL}',
                   fontsize=13, fontweight='normal', loc='left', pad=6)

    # --- Panel (c): inter-model spread -----------------------------------
    ax_c, tkw_c = _make_world_axes(fig, gs_maps[1, 0])
    sc_c = _draw_spread_panel(ax_c, tkw_c, best, marker_size=2)
    ax_c.set_title('Conceptual inter-model spread',
                   fontsize=13, fontweight='normal', loc='left', pad=6)
    cbar_c = fig.colorbar(sc_c, ax=ax_c, fraction=0.035, pad=0.02, extend='max')
    cbar_c.set_label(f'best − worst {METRIC_LABEL}', fontsize=12, fontweight='normal')
    cbar_c.ax.tick_params(labelsize=11)
    cbar_c.outline.set_edgecolor('#333333')

    # --- Panel (d): LSTM test KGE ----------------------------------------
    ax_d, tkw_d = _make_world_axes(fig, gs_maps[1, 1])
    sc_d = _draw_lstm_panel(ax_d, tkw_d, best, lstm_kge, marker_size=2)
    ax_d.set_title(f'LSTM test {METRIC_DISPLAY.get("kge_mod", "KGE")}',
                   fontsize=13, fontweight='normal', loc='left', pad=6)

    # Shared KGE colorbar for (b) and (d) -- identical norm/cmap.
    cbar_bd = fig.colorbar(sc_b, ax=[ax_b, ax_d], fraction=0.025, pad=0.02,
                           extend='both')
    cbar_bd.set_label(METRIC_LABEL, fontsize=12, fontweight='normal')
    cbar_bd.ax.tick_params(labelsize=11)
    cbar_bd.outline.set_edgecolor('#333333')
    cbar_bd.ax.axhline(0.5, color='black', linewidth=0.7)
    # Uniform, round tick labels on the shared b/d colorbar (the TwoSlopeNorm
    # default ticker otherwise lands them on uneven values).
    cbar_bd.set_ticks([-0.5, 0.0, 0.5, 1.0])

    # --- Right column: summary plots -------------------------------------
    ax_e = fig.add_subplot(gs_summ[0, 0])
    _draw_ecdf_panel(ax_e, best, lstm_kge, common)

    ax_f = fig.add_subplot(gs_summ[1, 0])
    _draw_kgz_box_panel(ax_f, best, lstm_kge, common)

    # No figure suptitle: this figure is embedded in main.tex where the LaTeX
    # caption already describes it, so an in-figure title would be redundant.

    # Explicit order, NOT reading order: the outer gridspec is [2x2 maps |
    # 2x1 summary column], so a-d are the maps and e-f the summary panels --
    # exactly the letters the titles used to carry.
    add_panel_letters([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Two-model variant of the extended summary figure: HBV vs the global LSTM
# only, with no comparison across the six conceptual models. Written as a
# standalone function (own output file, own CLI selector) so nothing above is
# affected.
# ---------------------------------------------------------------------------

def _indent_title(ax, x: float = 0.06):
    """Nudge a left-aligned panel title to the right so it clears the bold
    panel letter, which add_panel_letters draws at axes x = 0. Re-sets the
    title already on ``ax`` rather than requiring the caller to know the text,
    so it also works for titles written by the shared _draw_* helpers."""
    label = ax.get_title(loc='left')
    if label:
        ax.set_title(label, loc='left', x=x, pad=6,
                     fontsize=13, fontweight='normal')


def plot_hbv_lstm_summary(best: pd.DataFrame, out_path: str,
                          model: str = 'hbv') -> None:
    """Four-panel HBV-vs-LSTM summary -- the two-model reduction of
    ``plot_maps_with_summary``:

        (a) map of HBV KGE per catchment
        (b) map of LSTM KGE per catchment   (same colour scale as (a))
        (c) ECDF of KGE -- HBV vs LSTM      (panel (e) of the extended figure)
        (d) KGE by Köppen major zone        (panel (f) of the extended figure)

    All four panels use ONE paired catchment set: the stations that have both
    a non-NaN ``kge_<model>`` in ``best`` and an LSTM test score. Pairing is
    what makes the two-model comparison defensible -- an unpaired figure would
    confound a skill difference with a difference in which catchments each
    model happens to cover.

    Note on the phases being compared: the conceptual-model score is the
    PHASE-selected column (``valid_kge_mod`` by default) while the LSTM score
    is its own held-out test set. State this in the caption; it is not a
    like-for-like split, only a like-for-like catchment sample.
    """
    col = f'kge_{model}'
    if col not in best.columns:
        raise SystemExit(
            f"'{col}' not a column of `best` -- re-run collect_best_per_station "
            f"(it writes one kge_<model> column per entry of MODELS={MODELS})."
        )

    lstm_kge = load_lstm_kge(best)
    # Paired sample: both models scored at the same catchments.
    common = best.loc[best.index.intersection(lstm_kge.index), col].dropna().index
    if len(common) == 0:
        raise SystemExit(
            f"no catchment has both a non-NaN {col} and an LSTM test score")
    model_kge = best.loc[common, col]
    lstm_kge = lstm_kge.loc[common]
    model_label = MODEL_DISPLAY_NAMES.get(model, model)
    print(f"{model_label} vs LSTM: {len(common):,} paired catchments")

    # One colour key across panels (c) and (d): the conceptual model keeps its
    # palette colour from the map panels, the LSTM keeps a single contrasting
    # colour, in both the ECDF and the boxplots.
    model_color = _MODEL_PALETTE.get(model, '#0072B2')
    lstm_color = '#D55E00'

    # Two wide Robinson maps on top, two summary panels below. The map row is
    # the information-dense one, so it takes the larger share of the height
    # (1.3 : 1). A cartopy GeoAxes holds a fixed ~2:1 aspect, so height alone
    # cannot enlarge a map -- the figure is also wider than the map row strictly
    # needs, which is what actually sets the map size; the row ratio then keeps
    # the summary panels from dominating.
    fig = plt.figure(figsize=(14, 7.6), constrained_layout=True)
    fig.patch.set_facecolor('white')
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02,
                                    wspace=0.02, hspace=0.03)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])
    # The summary panels are narrower than the maps above them: the bottom row
    # is re-split into five slots -- outer margins, panel, inner gap, panel --
    # so (c) and (d) each take ~40% of the figure width instead of ~48%. Their
    # content (two ECDF curves, five paired boxes) does not need map width, and
    # the indent keeps the visual weight on the map row.
    gs_bot = gs[1, :].subgridspec(
        1, 5, width_ratios=[0.15, 1.0, 0.22, 1.0, 0.15])

    # --- Panel (a): conceptual-model KGE map ------------------------------
    ax_a, tkw_a = _make_world_axes(fig, gs[0, 0])
    sc_a = _draw_kge_map_panel(ax_a, tkw_a, best, model_kge, marker_size=2)
    # Name the phase in the title: this panel is the PHASE-selected split
    # ('validation' by default) while panel (b) is the LSTM's own test set. The
    # asymmetry is real and the titles should not hide it.
    ax_a.set_title(f'{model_label} {PHASE} {METRIC_LABEL}',
                   fontsize=13, fontweight='normal', loc='left', pad=6)

    # --- Panel (b): LSTM KGE map ------------------------------------------
    ax_b, tkw_b = _make_world_axes(fig, gs[0, 1])
    _draw_kge_map_panel(ax_b, tkw_b, best, lstm_kge, marker_size=2)
    ax_b.set_title(f'LSTM test {METRIC_DISPLAY.get("kge_mod", "KGE")}',
                   fontsize=13, fontweight='normal', loc='left', pad=6)

    # Shared colorbar under both maps -- identical norm/cmap, so one bar. Placed
    # horizontally beneath the map row rather than to its right, which would
    # narrow the maps themselves.
    cbar = fig.colorbar(sc_a, ax=[ax_a, ax_b], orientation='horizontal',
                        location='bottom', fraction=0.055, pad=0.02,
                        aspect=55, extend='both')
    cbar.set_label(METRIC_LABEL, fontsize=12, fontweight='normal')
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_edgecolor('#333333')
    cbar.ax.axvline(0.5, color='black', linewidth=0.7)
    cbar.set_ticks([-0.5, 0.0, 0.5, 1.0])

    # --- Panel (c): ECDF, this model vs LSTM ------------------------------
    ax_c = fig.add_subplot(gs_bot[0, 1])
    _draw_ecdf_panel(ax_c, best, lstm_kge, common, models=[model],
                     model_linewidth=2.6, lstm_color=lstm_color)

    # --- Panel (d): KGE by Köppen major zone ------------------------------
    ax_d = fig.add_subplot(gs_bot[0, 3])
    _draw_kgz_box_panel(ax_d, best, lstm_kge, common,
                        concept_col=col, concept_label=model_label,
                        concept_color=model_color, lstm_color=lstm_color)

    # No suptitle: the LaTeX caption carries the description, as in
    # plot_maps_with_summary.
    for _ax in (ax_a, ax_b, ax_c, ax_d):
        _indent_title(_ax)
    add_panel_letters([ax_a, ax_b, ax_c, ax_d])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Driver analysis: which static catchment attributes (columns of static.csv)
# are related to which model wins?
# ---------------------------------------------------------------------------

def _load_static_with_extras() -> pd.DataFrame:
    """Load static.csv and inject synthetic columns derived from the station
    index (so they're available to every downstream analysis without each
    caller re-deriving them).

    Currently adds:
      - ``source``: prefix of the station name before "__"
                    (e.g. "GRDC__6731585" -> "GRDC").
    """
    static_fpath = os.path.join(C.gscad_path, 'dataframes', 'static.csv')
    static_full = pd.read_csv(static_fpath, index_col='stations', comment='#',
                              low_memory=False)
    static_full = static_full.assign(
        source=static_full.index.to_series().str.split('__').str[0]
    )
    return static_full


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR). Vectorised, no statsmodels
    dependency. Returns adjusted p-values in the original input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.argsort(order) + 1
    sorted_p = p[order]
    adj_sorted = sorted_p * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p downward, then clip to [0, 1].
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    return adj_sorted[ranks - 1]


def analyze_static_relations(
        best: pd.DataFrame,
        alpha: float = 0.05,
        min_effect: float = 0.10,
        max_cardinality: int = 40,
        numeric_as_categorical_threshold: int = 12,
) -> pd.DataFrame:
    """For each column of static.csv, test whether the distribution differs
    across best-model classes. Returns a DataFrame of relations that pass
    BH-FDR ``alpha`` AND have ``effect_size >= min_effect``, ranked.

    Tests used
    ----------
    Numeric column:        one-way ANOVA   -> effect size = eta-squared
    Categorical column:    chi-squared     -> effect size = Cramer's V

    Both effect sizes live on a [0, 1] scale (negligible < 0.1, small ~0.1,
    medium ~0.3, large >= 0.5) so they can be ranked together.

    Heuristic: numeric columns with at most ``numeric_as_categorical_threshold``
    unique values are treated as categorical (catches climate classes encoded
    as integers, etc.). High-cardinality categoricals (> ``max_cardinality``)
    are skipped -- chi-squared on station-ID-like columns is meaningless.
    """
    static_full = _load_static_with_extras()

    # Align on the stations actually present in `best`.
    common = static_full.index.intersection(best.index)
    static_full = static_full.loc[common]
    target = best.loc[common, 'best_model']

    # Per-reason skip log so the caller can see which columns were excluded
    # and why. Reported as a printed summary at the end.
    skipped: dict[str, list[tuple[str, int]]] = {
        'too_few_observations': [],         # < 50 non-NaN stations
        'fewer_than_2_model_classes': [],   # after dropna, only one model wins everywhere
        'too_few_groups_or_no_variance': [],# numeric: <2 groups with non-zero std
        'invalid_p_value': [],              # NaN/inf from the test
        'constant_or_single_category': [],  # categorical with < 2 unique values
        'high_cardinality': [],             # categorical with > max_cardinality
    }

    results = []
    for col in static_full.columns:
        series = static_full[col]
        joined = pd.DataFrame({'y': target, 'x': series}).dropna()
        if len(joined) < 50:
            skipped['too_few_observations'].append((col, len(joined)))
            continue
        if joined['y'].nunique() < 2:
            skipped['fewer_than_2_model_classes'].append((col, int(joined['y'].nunique())))
            continue

        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric and joined['x'].nunique() <= numeric_as_categorical_threshold:
            is_numeric = False
            joined['x'] = joined['x'].astype('category')

        if is_numeric:
            groups = [g['x'].values for _, g in joined.groupby('y') if len(g) >= 5]
            groups = [g for g in groups if np.std(g) > 0]
            if len(groups) < 2:
                skipped['too_few_groups_or_no_variance'].append((col, len(groups)))
                continue
            try:
                f_stat, p = stats.f_oneway(*groups)
            except Exception:
                skipped['invalid_p_value'].append((col, -1))
                continue
            if not np.isfinite(p):
                skipped['invalid_p_value'].append((col, -1))
                continue
            grand_mean = joined['x'].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = float(((joined['x'] - grand_mean) ** 2).sum())
            eta2 = ss_between / ss_total if ss_total > 0 else 0.0
            results.append({
                'feature': col, 'kind': 'numeric',
                'test': 'ANOVA',
                'effect_metric': 'eta²',
                'effect_size': float(eta2),
                'p_value': float(p),
                'n': len(joined),
            })
        else:
            uniq = int(joined['x'].nunique())
            if uniq < 2:
                skipped['constant_or_single_category'].append((col, uniq))
                continue
            if uniq > max_cardinality:
                skipped['high_cardinality'].append((col, uniq))
                continue
            ct = pd.crosstab(joined['y'], joined['x'])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                skipped['constant_or_single_category'].append((col, uniq))
                continue
            try:
                chi2, p, _, _ = stats.chi2_contingency(ct)
            except Exception:
                skipped['invalid_p_value'].append((col, -1))
                continue
            if not np.isfinite(p):
                skipped['invalid_p_value'].append((col, -1))
                continue
            n = int(ct.values.sum())
            r, c = ct.shape
            cramers_v = float(np.sqrt(chi2 / (n * (min(r, c) - 1))))
            results.append({
                'feature': col, 'kind': 'categorical',
                'test': 'chi-squared',
                'effect_metric': "Cramer's V",
                'effect_size': cramers_v,
                'p_value': float(p),
                'n': n,
            })

    # --- Print the skip summary ----------------------------------------------
    total_skipped = sum(len(v) for v in skipped.values())
    if total_skipped:
        print(f"\n[analyze_static_relations] {total_skipped} columns skipped "
              f"(of {len(static_full.columns)} tested):")
        for reason, items in skipped.items():
            if not items:
                continue
            print(f"  - {reason} ({len(items)}):")
            # Sort high-cardinality ones by count descending; others by name.
            sorted_items = (sorted(items, key=lambda x: -x[1])
                            if reason == 'high_cardinality'
                            else sorted(items, key=lambda x: x[0]))
            for name, info in sorted_items:
                extra = f"  [n_unique={info}]" if reason == 'high_cardinality' \
                        else (f"  [n_obs={info}]" if reason == 'too_few_observations'
                              else "")
                print(f"      {name}{extra}")
        print()

    if not results:
        return pd.DataFrame(columns=['feature', 'kind', 'test', 'effect_metric',
                                     'effect_size', 'p_value', 'p_adj', 'n'])

    df = pd.DataFrame(results)
    df['p_adj'] = _bh_fdr(df['p_value'].values)
    df = df[(df['p_adj'] < alpha) & (df['effect_size'] >= min_effect)]
    df = df.sort_values('effect_size', ascending=False).reset_index(drop=True)
    return df


def plot_model_drivers(rel_df: pd.DataFrame, out_path: str,
                       alpha: float = 0.05, min_effect: float = 0.10) -> None:
    """Horizontal bar chart of significantly-related static features, ranked
    by effect size. Empty input is reported but doesn't error."""
    if rel_df.empty:
        print(f"no static features passed BH-FDR p<{alpha} & effect>= {min_effect}; "
              f"skipping {out_path}")
        return

    color_map = {'numeric': '#3a6fa5', 'categorical': '#c46b3a'}
    bar_colors = [color_map[k] for k in rel_df['kind']]

    height = max(3.5, 0.42 * len(rel_df) + 1.8)
    fig, ax = plt.subplots(figsize=(11, height))
    fig.patch.set_facecolor('white')

    y = np.arange(len(rel_df))
    ax.barh(y, rel_df['effect_size'], color=bar_colors,
            edgecolor='#333333', linewidth=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(rel_df['feature'])
    ax.invert_yaxis()  # Strongest at the top.

    # Effect-size convention guidelines (Cohen-style, applies to both metrics).
    x_max = max(0.35, rel_df['effect_size'].max() * 1.25)
    for x, lbl in [(0.1, 'small'), (0.3, 'medium'), (0.5, 'large')]:
        if x < x_max:
            ax.axvline(x, color='#888888', linestyle=':', linewidth=0.7, zorder=2)
            ax.text(x, -0.6, lbl, ha='center', va='bottom',
                    fontsize=9, color='#666666')

    # Per-bar annotation: effect size + corrected p-value.
    for i, row in rel_df.iterrows():
        ax.text(row['effect_size'] + 0.005, i,
                f"  {row['effect_size']:.2f}   "
                f"p$_\\mathrm{{adj}}$ = {row['p_adj']:.1e}   n={row['n']:,}",
                va='center', fontsize=9.5)

    ax.set_xlabel("Effect size  (eta²  /  Cramer's V)")
    ax.set_title(
        f"Static catchment attributes related to best-model choice\n"
        f"{PCP_LABEL}:  BH-FDR p < {alpha}, effect size $\\geq$ {min_effect}  "
        f"({len(rel_df)} of all tested features shown)",
        pad=12,
    )

    legend_elements = [
        Patch(facecolor=color_map['numeric'], edgecolor='#333333',
              label='Numeric feature (ANOVA, eta²)'),
        Patch(facecolor=color_map['categorical'], edgecolor='#333333',
              label="Categorical feature (chi², Cramer's V)"),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              frameon=True, edgecolor='#333333', framealpha=0.95)

    ax.set_xlim(0, x_max)
    ax.grid(True, axis='x', linestyle=':', alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_best_model_by_category(
        best: pd.DataFrame,
        feature: str,
        out_path: str,
        max_categories: int = 12,
) -> None:
    """Two stacked, x-aligned panels:

      (top)    stacked-proportion bars -- how best_model distributes across
               the categories of a static feature (each bar sums to 1.0,
               coloured by which model won).
      (bottom) boxplots of the best-conceptual KGE per category -- the skill
               level, so a category dominated by one model but poorly
               modelled by all is not mistaken for a success.

    An ``Overall`` reference column on the right shows the global model mix
    and the global KGE distribution, so per-category deviations from the
    baseline are visually obvious in both panels.

    For low-cardinality numerics encoded as ints (Koppen-Geiger major codes,
    etc.) this works directly. For genuine numerics (area, slope), bin first
    with ``pd.qcut`` before calling.
    """
    static_full = _load_static_with_extras()
    if feature not in static_full.columns:
        raise SystemExit(
            f"feature '{feature}' not in static.csv "
            f"(or in the synthetic columns added by _load_static_with_extras)"
        )

    common = static_full.index.intersection(best.index)
    df = pd.DataFrame({
        'best_model': best.loc[common, 'best_model'],
        'best_kge': best.loc[common, 'best_kge_mod'],
        'cat': static_full.loc[common, feature],
    }).dropna(subset=['best_model', 'cat'])  # keep rows w/ NaN KGE for the
    #   composition counts; the KGE panel drops NaN per-class itself.
    if df.empty:
        raise SystemExit(f"no overlapping stations have non-NaN {feature}")

    # Which classes to show is still decided by frequency (rare ones folded
    # into 'other' for readability), but the bars are ordered in Köppen
    # *climatic* sequence -- major zone A->E then sub-classes -- not by count.
    # _clim_key sorts numeric KG codes numerically and letter codes ("Af",
    # "BWh", ...) alphabetically; both yield the climatic order. 'other'
    # is appended last since it isn't a climate class.
    def _clim_key(c):
        try:
            return (0, float(c))
        except (TypeError, ValueError):
            return (1, str(c))

    cat_counts = df['cat'].value_counts()
    if len(cat_counts) > max_categories:
        keep = cat_counts.index[:max_categories]
        df.loc[~df['cat'].isin(keep), 'cat'] = 'other'
        cat_order = sorted(keep, key=_clim_key) + ['other']
    else:
        cat_order = sorted(cat_counts.index, key=_clim_key)

    ct = pd.crosstab(df['cat'], df['best_model']).reindex(cat_order, fill_value=0)
    proportions = ct.div(ct.sum(axis=1), axis=0)

    # Global baseline row appended to the right of the figure.
    overall = df['best_model'].value_counts(normalize=True)
    proportions.loc['Overall'] = overall.reindex(proportions.columns, fill_value=0)
    totals = ct.sum(axis=1)
    totals_with_overall = pd.concat([totals, pd.Series({'Overall': int(totals.sum())})])

    palette = {
        'hbv':    '#0072B2', 'gr4j':   '#D55E00', 'gr6j':   '#009E73',
        'simhyd': '#E69F00', 'hymod':  '#CC79A7', 'xaj':    '#56B4E9',
    }
    model_order = list(MODELS)
    colors = [palette.get(m, plt.get_cmap('tab10')(i))
              for i, m in enumerate(model_order)]
    proportions = proportions.reindex(columns=model_order, fill_value=0)

    # Width scales sub-linearly with category count so 20+ categories still
    # produce a screen-reasonable figure. Two x-aligned panels (sharex):
    # composition on top, best-conceptual KGE distribution below.
    n_bars = len(proportions)
    fig_w = max(9, 0.6 * n_bars + 3)
    fig_h = 10.0 if n_bars > 10 else 9.0
    fig, (ax, ax_kge) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={'height_ratios': [3.0, 1.5], 'hspace': 0.07},
    )
    fig.patch.set_facecolor('white')

    x = np.arange(len(proportions))
    overall_idx = list(proportions.index).index('Overall')

    # --- Top panel: stacked best-model composition -----------------------
    bottoms = np.zeros(len(proportions))
    for model_name, color in zip(model_order, colors):
        values = proportions[model_name].values
        ax.bar(x, values, bottom=bottoms, color=color, width=0.78,
               edgecolor='white', linewidth=0.6,
               label=MODEL_DISPLAY_NAMES.get(model_name, model_name))
        # Annotate large-enough segments with their percentage so the deviations
        # are quantitatively readable, not just visually.
        for xi, v, b in zip(x, values, bottoms):
            if v >= 0.06:
                ax.text(xi, b + v / 2, f"{v*100:.0f}%",
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='normal')
        bottoms += values

    ax.axvline(overall_idx - 0.5, color='#333333',
               linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of catchments')
    ax.set_title(
        f'Best model & its skill by {feature}\n'
        f'{PCP_LABEL}:  N = {int(totals.sum()):,} catchments  '
        f'(rightmost column = global baseline)',
        pad=12,
    )
    ax.legend(title='Best model', loc='upper left',
              bbox_to_anchor=(1.01, 1.0), frameon=True,
              edgecolor='#333333', framealpha=0.95)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelbottom=False)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # --- Bottom panel: best-conceptual KGE per class ---------------------
    # 'Overall' uses every catchment (matches the baseline column's meaning);
    # each other column uses its class. Same x positions/order as the bars
    # (shared axis) so a column's "who wins" and "how good" line up.
    kge_by_cat, box_positions = [], []
    for xi, cat in zip(x, proportions.index):
        if cat == 'Overall':
            vals = df['best_kge'].dropna().values
        else:
            vals = df.loc[df['cat'] == cat, 'best_kge'].dropna().values
        if len(vals):
            kge_by_cat.append(vals)
            box_positions.append(xi)

    if kge_by_cat:
        ax_kge.boxplot(
            kge_by_cat, positions=box_positions, widths=0.6,
            patch_artist=True, showfliers=False,
            medianprops=dict(color='black', linewidth=1.3),
            whiskerprops=dict(color='#0072B2'),
            capprops=dict(color='#0072B2'),
            boxprops=dict(facecolor='#0072B2', edgecolor='#0072B2',
                          alpha=0.55),
        )
    ax_kge.axhline(0.5, color='black', linewidth=0.7, linestyle=':')
    ax_kge.axvline(overall_idx - 0.5, color='#333333',
                   linewidth=0.8, linestyle='--', alpha=0.6)
    ax_kge.set_ylim(-1.0, 1.05)
    ax_kge.set_ylabel(f'Best-conceptual {METRIC_LABEL}')
    ax_kge.set_xlabel(feature)
    ax_kge.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax_kge.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax_kge.spines[spine].set_visible(False)

    tick_labels = [f"{cat}\n(n={totals_with_overall[cat]:,})"
                   for cat in proportions.index]
    ax_kge.set_xticks(x)
    if n_bars > 10:
        # Rotate so high-cardinality names don't collide. ha='right'
        # anchors the rotation pivot at the tick mark.
        ax_kge.set_xticklabels(tick_labels, rotation=45, ha='right',
                               rotation_mode='anchor', fontsize=10)
    else:
        ax_kge.set_xticklabels(tick_labels)
    ax_kge.set_xlim(-0.6, len(proportions) - 0.4)

    add_panel_letters([ax, ax_kge])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_model_kge_by_climate(best: pd.DataFrame, feature: str, out_path: str,
                              max_categories: int = 20) -> None:
    """Grouped boxplots of *each* conceptual model's own validation KGE per
    class of ``feature`` (e.g. KGZ_detailed).

    The honest "which model is good in which climate" view: unlike the
    argmax winner-share figure (plot_best_model_by_category), this shows
    every model's full skill distribution -- including climates where a
    model is competitive but not the single best. Same class selection,
    climatic ordering, 'other' folding and 'Overall' baseline as that
    figure, so the two line up column-for-column. Uses the per-model
    kge_<model> columns written by collect_best_per_station.
    """
    static_full = _load_static_with_extras()
    if feature not in static_full.columns:
        raise SystemExit(
            f"feature '{feature}' not in static.csv "
            f"(or the synthetic columns added by _load_static_with_extras)"
        )

    models = [m for m in MODELS if f'kge_{m}' in best.columns]
    if not models:
        raise SystemExit(
            "no kge_<model> columns in `best` -- re-run collect_best_per_station "
            "(it writes one KGE column per configured model)."
        )

    common = static_full.index.intersection(best.index)
    cols = {f'kge_{m}': best.loc[common, f'kge_{m}'] for m in models}
    df = pd.DataFrame({'cat': static_full.loc[common, feature], **cols}
                      ).dropna(subset=['cat'])
    if df.empty:
        raise SystemExit(f"no overlapping stations have non-NaN {feature}")

    # Identical class handling to plot_best_model_by_category so the two
    # figures share an x-axis.
    def _clim_key(c):
        try:
            return (0, float(c))
        except (TypeError, ValueError):
            return (1, str(c))

    cat_counts = df['cat'].value_counts()
    if len(cat_counts) > max_categories:
        keep = cat_counts.index[:max_categories]
        df.loc[~df['cat'].isin(keep), 'cat'] = 'other'
        cat_order = sorted(keep, key=_clim_key) + ['other']
    else:
        cat_order = sorted(cat_counts.index, key=_clim_key)
    cat_order = cat_order + ['Overall']

    totals = df['cat'].value_counts()
    n_by_cat = {c: int(totals.get(c, 0)) for c in cat_order}
    n_by_cat['Overall'] = int(len(df))

    n_cat, n_mod = len(cat_order), len(models)
    # Fixed box width (independent of how many models a class has) so every
    # box is the same size; within a class the boxes are RANKED by median
    # validation KGE, best -> worst, left -> right. Colour still identifies
    # the model (see legend), so a model's colour changes slot per class.
    width = 0.8 / n_mod

    fig_w = max(12, 0.95 * n_cat + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 7.0))
    fig.patch.set_facecolor('white')

    for i, cat in enumerate(cat_order):
        # (model, values) for every model that has data in this class.
        per_model = []
        for m in models:
            col = f'kge_{m}'
            if cat == 'Overall':
                vals = df[col].dropna().values
            else:
                vals = df.loc[df['cat'] == cat, col].dropna().values
            if len(vals):
                per_model.append((m, vals))
        if not per_model:
            continue
        # Rank best -> worst by median KGE within this class.
        per_model.sort(key=lambda mv: np.median(mv[1]), reverse=True)

        k = len(per_model)
        local_off = (np.arange(k) - (k - 1) / 2.0) * width
        for slot, (m, vals) in enumerate(per_model):
            color = _MODEL_PALETTE.get(m, '#999999')
            # manage_ticks=False: keep our integer class ticks; don't let
            # boxplot reset the x-axis to the offset float positions.
            ax.boxplot(
                [vals], positions=[i + local_off[slot]], widths=width * 0.9,
                patch_artist=True, showfliers=False, manage_ticks=False,
                medianprops=dict(color='black', linewidth=1.0),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                boxprops=dict(facecolor=color, edgecolor=color, alpha=0.6),
            )

    overall_idx = cat_order.index('Overall')
    ax.axvline(overall_idx - 0.5, color='#333333',
               linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(0.5, color='black', linewidth=0.7, linestyle=':')
    ax.set_ylim(-1.0, 1.05)
    ax.set_xlim(-0.6, n_cat - 0.4)
    ax.set_xticks(np.arange(n_cat))
    tick_labels = [f"{c}\n(n={n_by_cat[c]:,})" for c in cat_order]
    if n_cat > 10:
        ax.set_xticklabels(tick_labels, rotation=45, ha='right',
                           rotation_mode='anchor', fontsize=10)
    else:
        ax.set_xticklabels(tick_labels)
    feature_label = FEATURE_AXIS_LABELS.get(feature, feature)
    ax.set_ylabel(f'{METRIC_LABEL} (validation)')
    ax.set_xlabel(feature_label)
    ax.set_title(
        f'Per-model validation {METRIC_LABEL} by {feature_label}\n'
        f'{PCP_LABEL}:  N = {len(df):,} catchments  (boxes ranked '
        f'best→worst within each class; rightmost = global baseline)',
        pad=12,
    )
    ax.legend(
        handles=[Patch(facecolor=_MODEL_PALETTE.get(m, '#999999'),
                       edgecolor=_MODEL_PALETTE.get(m, '#999999'),
                       alpha=0.6, label=MODEL_DISPLAY_NAMES.get(m, m))
                 for m in models],
        title='Model', loc='upper left', bbox_to_anchor=(1.01, 1.0),
        frameon=True, edgecolor='#333333', framealpha=0.95,
    )
    ax.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# Type sizes for plot_hbv_lstm_by_climate ONLY. figstyle.py remains the house
# default for every other figure; this one is ~20 in wide and is reproduced at
# a small fraction of that width, so it needs its own larger sizes (figstyle's
# docstring allows a per-call-site override for exactly this case).
TICK_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 17
LEGEND_FONTSIZE = 15
LEGEND_TITLE_FONTSIZE = 16


def plot_hbv_lstm_by_climate(best: pd.DataFrame, feature: str, out_path: str,
                             model: str = 'hbv',
                             max_categories: int = 20) -> None:
    """Two-model companion to ``plot_model_kge_by_climate``: paired HBV and
    LSTM KGE boxplots per class of ``feature`` (e.g. KGZ_detailed).

    Same class selection, climatic ordering, 'other' folding and 'Overall'
    baseline as the six-model version, so the two figures line up
    column-for-column. Two differences follow from having only two models:
    the box slots are FIXED (conceptual model left, LSTM right) rather than
    ranked by median, because with a fixed slot the reader can scan one
    direction to see which model leads in each climate; and the catchment
    sample is paired, so a class shows the same catchments for both models.
    """
    col = f'kge_{model}'
    if col not in best.columns:
        raise SystemExit(
            f"'{col}' not a column of `best` -- re-run collect_best_per_station "
            f"(it writes one kge_<model> column per entry of MODELS={MODELS}).")

    static_full = _load_static_with_extras()
    if feature not in static_full.columns:
        raise SystemExit(
            f"feature '{feature}' not in static.csv "
            f"(or the synthetic columns added by _load_static_with_extras)")

    lstm_kge = load_lstm_kge(best)
    # Paired sample, as in plot_hbv_lstm_summary: both models scored at the
    # same catchments, so a per-class difference is a skill difference and not
    # a difference in catchment coverage.
    common = best.loc[best.index.intersection(lstm_kge.index), col].dropna().index
    common = common.intersection(static_full.index)
    if len(common) == 0:
        raise SystemExit(
            f"no catchment has both a non-NaN {col}, an LSTM test score and a "
            f"{feature} value")

    model_label = MODEL_DISPLAY_NAMES.get(model, model)
    df = pd.DataFrame({
        'cat': static_full.loc[common, feature],
        'concept': best.loc[common, col],
        'lstm': lstm_kge.loc[common],
    }).dropna(subset=['cat'])
    if df.empty:
        raise SystemExit(f"no overlapping stations have non-NaN {feature}")

    # Identical class handling to plot_model_kge_by_climate /
    # plot_best_model_by_category so all three share an x-axis.
    def _clim_key(c):
        try:
            return (0, float(c))
        except (TypeError, ValueError):
            return (1, str(c))

    cat_counts = df['cat'].value_counts()
    if len(cat_counts) > max_categories:
        keep = cat_counts.index[:max_categories]
        df.loc[~df['cat'].isin(keep), 'cat'] = 'other'
        cat_order = sorted(keep, key=_clim_key) + ['other']
    else:
        cat_order = sorted(cat_counts.index, key=_clim_key)
    cat_order = cat_order + ['Overall']

    totals = df['cat'].value_counts()
    n_by_cat = {c: int(totals.get(c, 0)) for c in cat_order}
    n_by_cat['Overall'] = int(len(df))

    # Same colour key as panels (c) and (d) of plot_hbv_lstm_summary.
    c_concept = _MODEL_PALETTE.get(model, '#0072B2')
    c_lstm = '#D55E00'
    # Narrower boxes than the six-model figure, and correspondingly fewer
    # inches per class: at 0.8 in/class the figure came out 20.6 in wide, so
    # everything on it shrank on the page. 0.58 in/class keeps the same box
    # geometry in data units while cutting ~25% of the width, which is what
    # makes the larger type below read as larger.
    width = 0.7 / 2

    n_cat = len(cat_order)
    fig_w = max(11, 0.58 * n_cat + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    fig.patch.set_facecolor('white')

    for i, cat in enumerate(cat_order):
        for slot, (key, color) in enumerate([('concept', c_concept),
                                             ('lstm', c_lstm)]):
            if cat == 'Overall':
                vals = df[key].dropna().values
            else:
                vals = df.loc[df['cat'] == cat, key].dropna().values
            if not len(vals):
                continue
            # manage_ticks=False: keep our integer class ticks; don't let
            # boxplot reset the x-axis to the offset float positions.
            ax.boxplot(
                [vals], positions=[i + (slot - 0.5) * width],
                widths=width * 0.9,
                patch_artist=True, showfliers=False, manage_ticks=False,
                medianprops=dict(color='black', linewidth=1.0),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                boxprops=dict(facecolor=color, edgecolor=color, alpha=0.6),
            )

    overall_idx = cat_order.index('Overall')
    ax.axvline(overall_idx - 0.5, color='#333333',
               linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(0.5, color='black', linewidth=0.7, linestyle=':')
    ax.set_ylim(-1.0, 1.05)
    ax.set_xlim(-0.6, n_cat - 0.4)
    ax.set_xticks(np.arange(n_cat))
    tick_labels = [f"{c}\n(n={n_by_cat[c]:,})" for c in cat_order]
    if n_cat > 10:
        ax.set_xticklabels(tick_labels, rotation=45, ha='right',
                           rotation_mode='anchor', fontsize=TICK_FONTSIZE)
    else:
        ax.set_xticklabels(tick_labels, fontsize=TICK_FONTSIZE)
    # This figure is wide (fig_w ~ 20 in) and gets shrunk hard on the printed
    # page, so the house tick/label sizes come out unreadable. Override at the
    # call site only -- figstyle.py stays the default for every other figure.
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    feature_label = FEATURE_AXIS_LABELS.get(feature, feature)
    ax.set_ylabel(METRIC_LABEL, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel(feature_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        f'{model_label} ({PHASE}) and LSTM (test) {METRIC_LABEL} '
        f'by {feature_label}\n'
        f'{PCP_LABEL}:  N = {len(df):,} paired catchments  '
        f'(rightmost = global baseline)',
        pad=12,
    )
    # Legend inside the axes rather than beside them: the old
    # bbox_to_anchor=(1.01, 1.0) placement spent ~10% of the figure width on
    # white space. It sits in the empty bottom strip under the Df* classes,
    # whose whiskers all stop above KGE 0.35 -- NOT under the C/D block
    # generally, where BSh, BSk and Cwa reach below -0.8. Single column so the
    # box stays narrow enough to clear those three.
    leg = ax.legend(
        handles=[
            Patch(facecolor=c_concept, edgecolor=c_concept, alpha=0.6,
                  label=f'{model_label} ({PHASE})'),
            Patch(facecolor=c_lstm, edgecolor=c_lstm, alpha=0.6,
                  label='LSTM (test)'),
        ],
        title='Hydrological model', loc='lower center',
        bbox_to_anchor=(0.72, 0.02), ncol=1,
        frameon=True, edgecolor='#333333', framealpha=0.95,
        facecolor='white',
        fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
    )
    # draw over the gridlines / axvline rather than under them
    leg.set_zorder(5)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Best-conceptual-model-per-catchment maps + driver analysis.")
    parser.add_argument(
        '--n-workers', type=int, default=1,
        help="processes for the per-station metrics_summary.csv scan. Request "
             "cores via SLURM and pass $SLURM_CPUS_PER_TASK here; default 1 "
             "(serial). Results are identical regardless of this value.")
    parser.add_argument(
        '--plot', default='all',
        choices=['all', 'extended', 'hbv_lstm', 'model_kge_by_climate',
                 'hbv_lstm_by_climate', 'drivers', 'best_model_by_category'],
        help="which figure to produce; 'all' (default) reproduces the full set. "
             "'hbv_lstm' is the two-model (HBV vs LSTM) reduction of the "
             "extended summary figure, and 'hbv_lstm_by_climate' the two-model "
             "reduction of the per-Koppen-class boxplots.")
    parser.add_argument(
        '--from-cache', action='store_true',
        help="load the per-station frame from the cached "
             "fig03_best_model_<PCP>.csv instead of re-scanning ~20k "
             "metrics_summary.csv files. Skips rewriting that cache. Use only "
             "when the calibration results have not changed since it was "
             "written.")
    cli = parser.parse_args()

    # Output dirs, anchored to the script location so they're independent of
    # the cwd the (slurm) job happens to start in:
    #   figure PNGs                    -> paper/figures/calibration/
    #   data CSVs (best-model cache +  -> paper/figures/fig_data/   (shared
    #   model_drivers)                    fig_data dir, figNN_<desc> naming as
    #                                     used by fig02/fig04/fig05).
    FIG_DIR = os.path.dirname(_HERE)               # paper/figures
    OUT_DIR = os.path.join(FIG_DIR, 'calibration')
    FIG_DATA_DIR = os.path.join(C.out_path, 'fig_data')  # relocated off /home (200G quota) to CONFIG.out_path on /ibex
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DATA_DIR, exist_ok=True)

    out_csv = os.path.join(FIG_DATA_DIR, f'fig03_best_model_{PCP}.csv')

    if cli.from_cache:
        if not os.path.isfile(out_csv):
            raise SystemExit(
                f"--from-cache given but {out_csv} does not exist; run once "
                f"without --from-cache to build it.")
        best = pd.read_csv(out_csv, index_col='station')
        print(f"read {out_csv}  ({len(best)} stations)")
    else:
        # phase='validation' -> conceptual results use valid_kge_mod; switch to
        # 'calibration' here (or via the PHASE constant) for calib_kge_mod.
        best = collect_best_per_station(RESULTS_DIR, PCP, phase=PHASE,
                                        n_workers=cli.n_workers)
        # Cache the collected per-station best-model frame (the expensive scan
        # result) so downstream/repeat runs can reuse it without re-walking
        # ~20k metrics_summary.csv files.
        best.to_csv(out_csv)
        print(f"wrote {out_csv}  ({len(best)} stations)")

    want = cli.plot

    #plot_best_model_map(best,           os.path.join(os.getcwd(), f'best_model_{PCP}.png'))
    #plot_best_kge_map(best,             os.path.join(os.getcwd(), f'best_kge_{PCP}.png'))
    #plot_kge_spread_map(best,           os.path.join(os.getcwd(), f'kge_spread_{PCP}.png'))
    ##plot_kge_relative_spread_map(best,  os.path.join(os.getcwd(), f'kge_relspread_{PCP}.png'))
    # plot_combined_panels(best,          os.path.join(OUT_DIR,f'summary_4panel_{PCP}.png'))
    # Alternative layout (maps left, ECDF + Köppen boxplots right). Separate
    # file so both this and the plain 4-panel figure above are available.
    if want in ('all', 'extended'):
        plot_maps_with_summary(best,        os.path.join(OUT_DIR,f'summary_4panel_extended_{PCP}.png'))

    # Two-model reduction of the figure above: HBV vs the global LSTM only.
    if want in ('all', 'hbv_lstm'):
        plot_hbv_lstm_summary(
            best, os.path.join(OUT_DIR, f'summary_hbv_lstm_{PCP}.png'),
            model='hbv',
        )

    # Two-model companion to the figure below: HBV vs LSTM per Köppen class.
    if want in ('all', 'hbv_lstm_by_climate'):
        plot_hbv_lstm_by_climate(
            best, 'KGZ_detailed',
            os.path.join(OUT_DIR,
                         f'hbv_lstm_kge_by_KGZ_detailed_{PCP}.png'),
            model='hbv', max_categories=20,
        )

    # Per-model validation KGE by Köppen detailed class -- the honest
    # "which model is good in which climate" companion (separate file).
    if want in ('all', 'model_kge_by_climate'):
        plot_model_kge_by_climate(
            best, 'KGZ_detailed',
            os.path.join(OUT_DIR,f'model_kge_by_KGZ_detailed_{PCP}.png'),
            max_categories=20,
        )

    # Driver analysis. The CSV captures every tested feature that survived the
    # filter -- handy for the paper / for inspecting borderline cases.
    if want in ('all', 'drivers'):
        rel_df = analyze_static_relations(best, alpha=0.05, min_effect=0.05)
        rel_df.to_csv(os.path.join(FIG_DATA_DIR, f'fig03_model_drivers_{PCP}.csv'),
                      index=False)
        print(f"wrote fig03_model_drivers_{PCP}.csv  ({len(rel_df)} significant features)")
        plot_model_drivers(rel_df, os.path.join(OUT_DIR,f'model_drivers_{PCP}.png'))

    if want in ('all', 'best_model_by_category'):
        # Drill-down on the top climate-related driver. Compare the per-class
        # model mix against the global baseline (rightmost bar) to see which
        # models over/under-represent in each Koppen class. KGZ_detailed has
        # higher cardinality than KGZ_major and a higher effect size here.
        plot_best_model_by_category(
            best, 'KGZ_detailed',
            os.path.join(OUT_DIR,f'best_model_by_KGZ_detailed_{PCP}.png'),
            max_categories=20,
        )

        # Source is the strongest categorical driver (Cramer's V = 0.23) --
        # worth its own drill-down. ~36 sources total; show the 15 largest and
        # fold the long tail into 'other' so the figure stays readable.
        plot_best_model_by_category(
            best, 'source',
            os.path.join(OUT_DIR,f'best_model_by_source_{PCP}.png'),
            max_categories=15,
        )

    print("done.")
