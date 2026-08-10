"""Representative example images from each pipeline stage — reads real
pipeline output produced by tomo_run.sh and writes ONE PNG PER STEP:

    pipeline_examples_step0_ups{UPS}.png    init.h5           xy + xz
    pipeline_examples_step1_ups{UPS}.png    big{UPS}x.h5      xy + xz
    pipeline_examples_step2_ups{UPS}.png    proj.h5           sino + θ=0
    pipeline_examples_step3_ups{UPS}.png    data.h5           sino + θ=0
    pipeline_examples_step4_ups{UPS}.png    mosaic_h5         diagonal tile picks

All panels are grayscale + colorbar, aspect='equal' so pixel geometry is
preserved.  Step 4 walks a diagonal across the (n_z, n_x) mosaic — n_x
linearly-spaced tiles from (0,0) to (n_z-1, n_x-1) — so the panels sample
the whole FOV instead of clustering in one corner.

Missing stages are silently skipped, so this can be run partway through a
pipeline run.
"""

import argparse
import glob
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- CLI --------------------------------------------------------------------
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--ups",  type=int, default=1, help="pipeline UPS to visualize (default 1)")
p.add_argument("--path", default="/data2/brain_sym_mosaic",
               help="base dir containing init.h5, big{UPS}x.h5, model_big{UPS}x/, mosaic_h5/")
p.add_argument("--bin",  type=int, default=4, help="display binning factor")
args = p.parse_args()

UPS  = args.ups
BASE = args.path
BIN  = args.bin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _shape_check(path, expected_nz, label):
    """Print a warning if the file's z-dimension isn't what we expect
    (e.g. init.h5 should be 3072, big{UPS}x.h5 should be 3072·UPS)."""
    with h5py.File(path, 'r') as f:
        actual = f['/exchange/data'].shape
    if actual[0] != expected_nz:
        print(f'  ⚠  {label}  NZ = {actual[0]}  '
              f'(expected {expected_nz}) — likely stale step00/step1 default; '
              f'big fraction may be zero-filled')
    else:
        print(f'  ✓  {label}  shape {actual}')


# --- helpers ---------------------------------------------------------------
def bin2d(a, factor=BIN):
    if factor <= 1:
        return a
    h, w = a.shape
    h2, w2 = h // factor, w // factor
    return a[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def robust_range(a, lo=1.0, hi=99.0):
    return np.percentile(a, [lo, hi])


# VDS bank files stay open for the master file's lifetime; on this system the
# fd limit is 1024 (unraisable), and big{UPS}x.h5 has ~3000 banks — reading
# every 4th z hits ~1500 banks and past ~fd#1020 HDF5 silently returns the
# fill value (-5.0) instead of raising. Reopen the master file every
# _REOPEN_EVERY reads to flush the bank fd table.
_REOPEN_EVERY = 500


def read_xz(path, y, z_step=BIN, x_step=BIN):
    """xz slice at fixed y — chunks are per-z so we read one chunk per z."""
    with h5py.File(path, 'r') as f:
        NZ, _, NX = f['/exchange/data'].shape
    rows = list(range(0, NZ // z_step * z_step, z_step))
    out = np.zeros((len(rows), NX // x_step), dtype=np.float32)
    for i0 in range(0, len(rows), _REOPEN_EVERY):
        with h5py.File(path, 'r') as f:
            d = f['/exchange/data']
            for i in range(i0, min(i0 + _REOPEN_EVERY, len(rows))):
                out[i] = d[rows[i], y, :NX // x_step * x_step:x_step]
    return out


def read_fixed_theta(path, theta_idx=0, z_step=BIN, x_step=BIN):
    """Projection at fixed θ — sub-sampled in z and x."""
    with h5py.File(path, 'r') as f:
        _, NZ, NX = f['/exchange/data'].shape
    rows = list(range(0, NZ // z_step * z_step, z_step))
    out = np.zeros((len(rows), NX // x_step), dtype=np.float32)
    for i0 in range(0, len(rows), _REOPEN_EVERY):
        with h5py.File(path, 'r') as f:
            d = f['/exchange/data']
            for i in range(i0, min(i0 + _REOPEN_EVERY, len(rows))):
                out[i] = d[theta_idx, rows[i], :NX // x_step * x_step:x_step]
    return out


def _panel(ax, img, title, subtitle, orig_shape=None):
    """Draw img as a panel.  orig_shape is the ORIGINAL (un-binned) shape
    for the corner label; if None, we assume img is already at original
    resolution (label = img.shape)."""
    vmin, vmax = robust_range(img, 1, 99)
    im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    ax.text(0.0, 1.055, title, transform=ax.transAxes,
            fontsize=10.5, fontweight='bold', ha='left', va='bottom')
    ax.text(0.0, 1.008, subtitle, transform=ax.transAxes,
            fontsize=8.8, ha='left', va='bottom', color='#444')
    ax.set_xticks([]); ax.set_yticks([])
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='3%', pad=0.06)
    cb  = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    oh, ow = orig_shape if orig_shape is not None else img.shape
    ax.text(0.02, 0.02, f'{oh:,} × {ow:,}',
            transform=ax.transAxes, fontsize=8, ha='left', va='bottom',
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                      edgecolor='none', alpha=0.6))


def _save(fig, name):
    out = os.path.join(SCRIPT_DIR, name)
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')


# --- per-step renderers ----------------------------------------------------
def _render_volume(step, path, expected_nz, label, suptitle, xy_subtitle_extra=''):
    """Two-panel xy + xz figure for a 3-D volume (init.h5 or big{UPS}x.h5)."""
    if not os.path.exists(path):
        print(f'skip step {step}: {path} not found')
        return
    _shape_check(path, expected_nz, label)
    with h5py.File(path, 'r') as f:
        NZ, NY, NX = f['/exchange/data'].shape
        mid_z, mid_y = NZ // 2, NY // 2
        print(f'{label} xy @ z={mid_z} …')
        xy_raw = f['/exchange/data'][mid_z, :, :].astype(np.float32)
    xy = bin2d(xy_raw)
    print(f'{label} xz @ y={mid_y} …')
    xz = read_xz(path, mid_y)

    panel_title = f'step {step} · {label}'
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2),
                             gridspec_kw={'wspace': 0.28})
    _panel(axes[0], xy, panel_title, f'xy slice at z = {mid_z}{xy_subtitle_extra}',
           orig_shape=xy_raw.shape)
    _panel(axes[1], xz, panel_title, f'xz slice at y = {mid_y}  (side view)',
           orig_shape=(NZ, NX))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.995)
    _save(fig, f'pipeline_examples_step{step}_ups{UPS}.png')


def _render_projections(step, path, label, suptitle, sino_note, theta_note):
    """Two-panel sinogram + θ=0 projection for proj.h5 or data.h5
    (shape (NTHETA, NZ, N))."""
    if not os.path.exists(path):
        print(f'skip step {step}: {path} not found')
        return
    with h5py.File(path, 'r') as f:
        _, NZ_p, NX_p = f['/exchange/data'].shape
        mid_z = NZ_p // 2
        print(f'{label} sinogram @ z={mid_z} …')
        sino_raw = f['/exchange/data'][:, mid_z, :].astype(np.float32)
    sino = bin2d(sino_raw)
    print(f'{label} θ=0 …')
    theta0 = read_fixed_theta(path, 0)

    panel_title = f'step {step} · {label}'
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2),
                             gridspec_kw={'wspace': 0.28})
    _panel(axes[0], sino, panel_title, f'sinogram at z = {mid_z}  {sino_note}',
           orig_shape=sino_raw.shape)
    _panel(axes[1], theta0, panel_title, f'projection at θ = 0°  {theta_note}',
           orig_shape=(NZ_p, NX_p))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.995)
    _save(fig, f'pipeline_examples_step{step}_ups{UPS}.png')


def _list_tile_indices(mosaic_dir):
    tiles = []
    for f in glob.glob(os.path.join(mosaic_dir, '*_*.h5')):
        try:
            zi, xi = (int(x) for x in os.path.basename(f)[:-3].split('_'))
        except ValueError:
            continue
        tiles.append((zi, xi))
    return sorted(tiles)


def _render_mosaic(step, mosaic_dir):
    if not os.path.isdir(mosaic_dir):
        print(f'skip step {step}: {mosaic_dir} not found')
        return
    tiles_idx = _list_tile_indices(mosaic_dir)
    if not tiles_idx:
        print(f'skip step {step}: no *_*.h5 files in {mosaic_dir}')
        return
    n_z = max(zi for zi, _ in tiles_idx) + 1
    n_x = max(xi for _, xi in tiles_idx) + 1
    # Walk diagonal: n_x linearly-spaced tiles from (0,0) to (n_z-1, n_x-1)
    picks = [(int(round(k * (n_z - 1) / max(1, n_x - 1))), k)
             for k in range(n_x)]
    print(f'mosaic tiles: {picks}')

    nrows = len(picks)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 4.2 * nrows),
                             gridspec_kw={'hspace': 0.35})
    if nrows == 1:
        axes = [axes]
    for ax, (zi, xi) in zip(axes, picks):
        with h5py.File(f'{mosaic_dir}/{zi}_{xi}.h5', 'r') as f:
            img = f['/exchange/data'][0, :, :].astype(np.float32)
        pos = ('first' if (zi, xi) == picks[0]
               else 'last' if (zi, xi) == picks[-1]
               else 'mid')
        _panel(ax, img,
               f'step {step} · mosaic_h5/{zi}_{xi}.h5',
               f'{pos} tile,  θ = 0°')
    fig.suptitle(f'step {step} — mosaic tiles ({n_z}×{n_x} grid, UPS = {UPS})',
                 fontsize=13, fontweight='bold', y=0.995)
    _save(fig, f'pipeline_examples_step{step}_ups{UPS}.png')


# --- run each step ---------------------------------------------------------
_render_volume(
    step=0, path=f'{BASE}/init.h5', expected_nz=3072, label='init.h5',
    suptitle=f'step 0 — extract + mask (init.h5, UPS = {UPS})',
)
_render_volume(
    step=1, path=f'{BASE}/big{UPS}x.h5', expected_nz=3072 * UPS, label=f'big{UPS}x.h5',
    suptitle=f'step 1 — upsample ×{UPS} (big{UPS}x.h5)',
    xy_subtitle_extra='  (= init at UPS = 1)' if UPS == 1 else '',
)
_render_projections(
    step=2, path=f'{BASE}/model_big{UPS}x/proj.h5', label='proj.h5',
    suptitle=f'step 2 — Radon transform (proj.h5, UPS = {UPS})',
    sino_note='(θ × s)', theta_note='(z × s)',
)
_render_projections(
    step=3, path=f'{BASE}/model_big{UPS}x/data.h5', label='data.h5',
    suptitle=f'step 3 — Fresnel propagation (data.h5, UPS = {UPS})',
    sino_note='(after Fresnel)', theta_note='(Fresnel-propagated)',
)
_render_mosaic(step=4, mosaic_dir=f'{BASE}/mosaic_h5')
