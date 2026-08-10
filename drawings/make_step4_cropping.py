"""Visualize step 4 cropping: overlay all mosaic-tile crop rectangles on
the data.h5 fixed-θ projection (real UPS=1 data).

Reads data.h5 (θ=0 view over the full N × N plane), computes tile origins
via step0_schematic.compute_x_layout / compute_z_stack, and draws every
(z, x) tile as a rectangle on top of the projection.  Uses step4's actual
z_pad derivation: `(NZ - SAMPLE_H_PX)/2` — sample centered vertically in
the (3072·UPS)^3 dataset.
"""

import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- reuse layout from step0_schematic (loads DET_W, DET_H, SAMPLE_D_PX, N, …)
# step0_schematic.py lives one directory up from this drawings/ folder.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCHEMATIC_PY = os.path.join(os.path.dirname(_SCRIPT_DIR), 'step0_schematic.py')
sys.argv = ['x', '--ups', '2']
exec(open(_SCHEMATIC_PY).read().split('def main')[0])

# --- step4_extract's z_pad default (matches step4_extract.py's derivation):
# sample cylinder is centered in the (NZ, N, N) pipeline dataset, so the
# sample top sits at row (NZ - SAMPLE_H_PX)/2 in the data.h5 coordinate
# frame.  For the current 3072·UPS pipeline with SAMPLE_H_PX=2972·UPS,
# Z_PAD_PIPE = (3072·UPS − 2972·UPS)/2 = 50·UPS.
Z_PAD_PIPE   = (_N_PIPELINE - SAMPLE_H_PX) // 2

# --- tile origins in data.h5 (row, col) coordinates
_, x_origins, _, _ = compute_x_layout(SAMPLE_D_PX / 2.0)
z_positions, _     = compute_z_stack(float(SAMPLE_H_PX))
z_starts = [int(round(z + Z_PAD_PIPE)) for z in z_positions]
x_starts = [int(round(x + _N_PIPELINE / 2)) for x in x_origins]
n_z, n_x = len(z_starts), len(x_starts)
N_TILES  = n_z * n_x

# --- read data.h5 projection at θ = 0 (fixed θ, all z, all x) ------------
BASE = '/local/tomodata2/brain_sym_mosaic'
BIN  = 4                              # spatial binning for display

def read_fixed_theta(path, theta_idx=0, z_step=BIN, x_step=BIN):
    with h5py.File(path, 'r') as f:
        d = f['/exchange/data']
        _, NZ, NX = d.shape
        rows = range(0, NZ // z_step * z_step, z_step)
        out = np.zeros((len(rows), NX // x_step), dtype=np.float32)
        for i, z in enumerate(rows):
            out[i] = d[theta_idx, z, :NX // x_step * x_step:x_step]
    return out

print('reading data.h5 @ θ = 0 …')
proj = read_fixed_theta(f'{BASE}/model_big1x/data.h5', 0)
vmin, vmax = np.percentile(proj, [1, 99])

# tiles in DISPLAY (binned) coordinates
def tile_rect_disp(z_start, x_start):
    """Rectangle (col0, row0, w, h) in the binned image."""
    return (x_start / BIN, z_start / BIN, DET_W / BIN, DET_H / BIN)

# highlighted tiles (diagonal-ish sweep from (0,0) to (n_z-1, n_x-1))
HIGHLIGHT = {}
if n_z >= 1 and n_x >= 1:
    for k in range(n_x):
        zi = int(round(k * (n_z - 1) / max(1, n_x - 1)))
        xi = k
        label = f'{zi}_{xi}'
        if k == 0:
            label = f'first ({label})'
        elif k == n_x - 1:
            label = f'last ({label})'
        HIGHLIGHT[(zi, xi)] = label


# --- figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 11))

ax.imshow(proj, cmap='gray', vmin=vmin, vmax=vmax,
          aspect='equal', interpolation='nearest',
          extent=(0, _N_PIPELINE, _N_PIPELINE, 0))       # extent in FULL data px

# colour by x tile index (columns)
cmap = plt.get_cmap('tab10')
x_colors = [cmap(i) for i in range(n_x)]

# draw every tile
for zi in range(n_z):
    for xi in range(n_x):
        r0, c0 = z_starts[zi], x_starts[xi]
        color  = x_colors[xi]
        highlight = (zi, xi) in HIGHLIGHT
        lw = 2.2 if highlight else 1.0
        alpha_fill = 0.20 if highlight else 0.0
        alpha_edge = 1.0 if highlight else 0.75
        # fill
        if alpha_fill > 0:
            ax.add_patch(Rectangle((c0, r0), DET_W, DET_H,
                                   facecolor=color, alpha=alpha_fill,
                                   edgecolor='none', zorder=2))
        # outline
        ax.add_patch(Rectangle((c0, r0), DET_W, DET_H,
                               fill=False, edgecolor=color,
                               lw=lw, alpha=alpha_edge, zorder=3))
        # label (tile index) in tile centre
        cx, cy = c0 + DET_W/2, r0 + DET_H/2
        ax.text(cx, cy, f'{zi}_{xi}',
                color='white', fontsize=7, ha='center', va='center',
                fontweight='bold' if highlight else 'normal',
                path_effects=None,
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor=color, edgecolor='none',
                          alpha=0.75 if highlight else 0.5),
                zorder=4)

# rotation axis reference (vertical line at N/2)
ax.axvline(_N_PIPELINE/2, color='red', ls='--', lw=0.9, alpha=0.65, zorder=1)
ax.text(_N_PIPELINE/2 + 12, 20, 'rotation axis  (col = N/2)',
        color='red', fontsize=9, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='red', alpha=0.85))

# sample bounds reference (SAMPLE_H_PX in z, centred by z_pad)
z_sample_top = Z_PAD_PIPE
z_sample_bot = Z_PAD_PIPE + SAMPLE_H_PX
ax.axhline(z_sample_top, color='#0088aa', ls=':', lw=0.9, alpha=0.7)
ax.axhline(z_sample_bot, color='#0088aa', ls=':', lw=0.9, alpha=0.7)
ax.text(20, z_sample_top - 12, f'sample top (row {z_sample_top})',
        color='#0088aa', fontsize=8, ha='left', va='bottom')
ax.text(20, z_sample_bot + 25, f'sample bottom (row {z_sample_bot})',
        color='#0088aa', fontsize=8, ha='left', va='top')

# axes
ax.set_xlim(0, _N_PIPELINE)
ax.set_ylim(_N_PIPELINE, 0)
ax.set_xlabel('column  (detector s)')
ax.set_ylabel('row  (z, sample vertical)')
ax.set_title(f'step 4 cropping — {n_z} × {n_x} = {N_TILES} mosaic tiles overlaid on '
             f'data.h5 (θ = 0°, full N × N = {_N_PIPELINE} × {_N_PIPELINE})',
             fontsize=12, fontweight='bold', pad=10)

# legend for x colours
from matplotlib.lines import Line2D
legend_items = [Line2D([], [], marker='s', color='none',
                       markerfacecolor=x_colors[i],
                       markeredgecolor=x_colors[i], markersize=12,
                       label=f'x-tile {i}   (col start {x_starts[i]})')
                for i in range(n_x)]
ax.legend(handles=legend_items, loc='upper right', fontsize=9,
          frameon=True, facecolor='white', framealpha=0.9)

# summary text
info = (
    f'DET_W × DET_H = {DET_W} × {DET_H}   (per-tile crop)\n'
    f'AXIS_INSET   = {AXIS_INSET}   OVERLAP = {OVERLAP}   Z_OVERLAP = {Z_OVERLAP}\n'
    f'z_pad (row offset) = {Z_PAD_PIPE}   ·   rotation axis col = {_N_PIPELINE // 2}'
)
ax.text(0.02, 0.02, info, transform=ax.transAxes,
        fontsize=9, ha='left', va='bottom', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#aaa', alpha=0.9))

out = os.path.join(_SCRIPT_DIR, 'step4_cropping_ups1.png')
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'saved: {out}')
