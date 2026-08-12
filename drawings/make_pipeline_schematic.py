"""Data-simulation pipeline schematic — one figure per step + a combined
overview.  Uses the current 3072³ init.h5 geometry (see step0_schematic.py
for the physical numbers).

Vertical flow of the whole pipeline:
    source TIFF → init.h5 → big{UPS}x.h5 → proj.h5 → data.h5 → mosaic tiles

Outputs (all saved next to this script, in mosaic_modeling/drawings/):
    pipeline_step0_ups{UPS}.png    step00 EXTRACT + MASK
    pipeline_step1_ups{UPS}.png    step1  UPSAMPLE
    pipeline_step2_ups{UPS}.png    step2  RADON
    pipeline_step3_ups{UPS}.png    step3  FRESNEL
    pipeline_step4_ups{UPS}.png    step4  MOSAIC EXTRACT
    pipeline_step5_ups{UPS}.png    step5  CORRECT (dezinger + darkflat + FW rings)
    pipeline_overview_ups{UPS}.png full vertical flow (all steps stacked)
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --- CLI --------------------------------------------------------------------
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--ups", type=int, default=8, help="upsample factor (default 8)")
args = p.parse_args()
UPS = args.ups

# All PNGs land next to this script (mosaic_modeling/drawings/), no matter
# where the user launched python from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- pipeline parameters (geometry from step0_schematic.py) -----------------
DET_W     = 3232 * UPS // 8       # detector px in pipeline voxels
DET_H     = 2426 * UPS // 8
PIXEL_UM  = 1.38 * 8 / UPS        # pipeline voxel = detector px at UPS=8
INIT_N    = 3072                  # init.h5 side (step00 output)
N         = INIT_N * UPS          # big{UPS}x side
NTHETA    = 3 * N // 4
SAMPLE_D_PX = 2918 * UPS          # sample cylinder diameter
SAMPLE_H_PX = 2972 * UPS          # sample cylinder height
SAMPLE_D_MM = SAMPLE_D_PX * PIXEL_UM * 1e-3
SAMPLE_H_MM = SAMPLE_H_PX * PIXEL_UM * 1e-3
# 4 x-tiles × 11 z-stacks at the current geometry (200 px overlap).
NX_TILES  = 4
NZ_STACK  = 11
N_TILES   = NX_TILES * NZ_STACK


def fmt(nbytes):
    for unit, sz in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6)):
        if nbytes >= sz:
            return f'{nbytes/sz:.1f} {unit}' if nbytes/sz < 100 else f'{nbytes/sz:.0f} {unit}'
    return f'{nbytes/1e6:.0f} MB'


# --- pipeline stages: alternating data → step → data → step … ---------------
tiff_shape = (3264, 3264, 3264)
init_shape = (INIT_N, INIT_N, INIT_N)
big_shape  = (N, N, N)
proj_shape = (NTHETA, N, N)
data_shape = proj_shape
tile_shape = (NTHETA, DET_H, DET_W)

stages = [
    dict(kind='data', name='reconstructed volume',
         file='brain_recon.tif  (source TIFF stack)',
         shape=tiff_shape, extra='pre-existing input',
         color='#e8e8e8'),
    dict(kind='step', num=0, script='step00_upsample_extract.py',
         title='EXTRACT + MASK',
         body=f'center-crop to 2560³  ·  cylindrical mask (⌀ ≈ 0.95·N, cosine taper)\n'
              f'~50 zero voxels at each z end with cosine ramp\n'
              f'bilinear xy up + linear z lerp → init.h5 ({INIT_N}³)'),
    dict(kind='data', name='init volume',
         file='init.h5   /exchange/data',
         shape=init_shape,
         extra=f'sample cylinder ⌀{SAMPLE_D_MM:.1f} × {SAMPLE_H_MM:.1f} mm centered',
         color='#fff2b0'),
    dict(kind='step', num=1, script='step1_upsample.py',
         title=f'UPSAMPLE  ×{UPS}',
         body=f'trilinear (bilinear xy + linear z blend)\n'
              f'MPI  ·  GPU  ·  VDS + banks output'),
    dict(kind='data', name='high-res volume',
         file=f'big{UPS}x.h5   /exchange/data',
         shape=big_shape,
         extra=f'{PIXEL_UM:g} µm/voxel  (=  detector pixel at UPS=8)',
         color='#ffde7a'),
    dict(kind='step', num=2, script='step2_radon.py',
         title='RADON TRANSFORM',
         body=f'{NTHETA} angles over 360°  (NTHETA = 3·N/4)\n'
              f'TomoReal (USFFT, rfft/float32)  ·  GPU  ·  MPI z-slabs'),
    dict(kind='data', name='sinograms (phase)',
         file=f'model_big{UPS}x/proj.h5   /exchange/data',
         shape=proj_shape, extra='φ(θ, z, s)  before propagation',
         color='#ffb98a'),
    dict(kind='step', num=3, script='step3_propagation.py',
         title='FRESNEL PROPAGATION',
         body='ψ = exp(i·φ − φ/β_ratio)  →  I = |D_R₂ψ|²\n'
              'E = 30 keV,  R₂ = 1 m  ·  GPU  ·  MPI'),
    dict(kind='data', name='detector-plane intensity',
         file=f'model_big{UPS}x/data.h5   /exchange/data',
         shape=data_shape, extra='full-N virtual detector, all θ',
         color='#f88b6c'),
    dict(kind='step', num=4, script='step4_extract.py',
         title='MOSAIC TILE EXTRACT',
         body=f'crop each of {NX_TILES}×{NZ_STACK} = {N_TILES} tile positions\n'
              'air-fill (=1.0) outside sample  ·  MPI round-robin'),
    dict(kind='data', name=f'{N_TILES} mosaic tiles',
         file='mosaic_h5/{z_idx}_{x_idx}.h5   ×' + f' {N_TILES}',
         shape=tile_shape, extra=f'one file per (z-stack, x-tile) position',
         color='#c66b52'),
    dict(kind='step', num=5, script='step5_correct.py',
         title='PREPROCESS TILES',
         body='dezinger + dark-flat correction + FW ring removal\n'
              'per-tile GPU pass (nzchunk row-strips)  ·  MPI round-robin'),
    dict(kind='data', name=f'{N_TILES} corrected tiles',
         file='mosaic_h5_pre/{z_idx}_{x_idx}.h5   ×' + f' {N_TILES}',
         shape=tile_shape, extra='same schema as mosaic_h5/, ring-suppressed',
         color='#a05840'),
]


# --- draw helpers -----------------------------------------------------------
def draw_step(ax, s, y_bot, y_top, box_left, box_right):
    y_mid = (y_top + y_bot) / 2
    box = FancyBboxPatch(
        (box_left, y_bot), box_right - box_left, y_top - y_bot,
        boxstyle='round,pad=0.02,rounding_size=0.10',
        facecolor='#3f6bad', edgecolor='#1e3f6e', lw=1.8, zorder=3)
    ax.add_patch(box)
    cx = box_left + 0.7
    ax.add_patch(plt.Circle((cx, y_mid), 0.42, facecolor='white',
                            edgecolor='#1e3f6e', lw=1.8, zorder=4))
    ax.text(cx, y_mid, f"{s['num']}", ha='center', va='center',
            fontsize=18, fontweight='bold', color='#1e3f6e', zorder=5)
    ax.text(cx + 0.8, y_mid + 0.32, s['title'], ha='left', va='center',
            fontsize=13, fontweight='bold', color='white', zorder=5)
    ax.text(cx + 0.8, y_mid - 0.28, s['body'], ha='left', va='center',
            fontsize=9.5, color='#e8eefb', zorder=5)
    ax.text(box_right - 0.15, y_mid, s['script'], ha='right', va='center',
            fontsize=9, color='#c8d4ec', style='italic', zorder=5,
            family='monospace')


def draw_data(ax, s, y_bot, y_top, box_left, box_right):
    y_mid = (y_top + y_bot) / 2
    box = FancyBboxPatch(
        (box_left, y_bot), box_right - box_left, y_top - y_bot,
        boxstyle='round,pad=0.02,rounding_size=0.20',
        facecolor=s['color'], edgecolor='#7a6415', lw=1.4, zorder=3)
    ax.add_patch(box)
    cx = box_left + 0.7
    ax.text(cx, y_mid + 0.02, '≡', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#5c4f0f', zorder=5)
    shape = s['shape']
    size  = int(np.prod(shape)) * 4
    if s['file'].startswith('mosaic_h5/'):
        total_size = size * N_TILES
        size_text  = f'{fmt(size)} × {N_TILES}  =  {fmt(total_size)}'
    else:
        size_text  = fmt(size)
    shape_text = '(' + ', '.join(f'{d:,}' for d in shape) + ')  float32'
    ax.text(cx + 0.7, y_mid + 0.30, f"{s['file']}", ha='left', va='center',
            fontsize=11, fontweight='bold', color='#3a2f00', zorder=5,
            family='monospace')
    ax.text(cx + 0.7, y_mid - 0.03, f"shape {shape_text}", ha='left',
            va='center', fontsize=9.5, color='#3a2f00', zorder=5,
            family='monospace')
    ax.text(cx + 0.7, y_mid - 0.32, f"{s['extra']}", ha='left', va='center',
            fontsize=9, color='#5a4b10', zorder=5, style='italic')
    ax.text(box_right - 0.25, y_mid, size_text, ha='right', va='center',
            fontsize=12, fontweight='bold', color='#7a1919', zorder=5)


def draw_arrow(ax, x, y_from, y_to):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle='-|>',
                                 color='#333', lw=1.6, mutation_scale=18,
                                 zorder=2))


# --- per-step figures (input data → step → output data) ---------------------
# Find each step index and the data cards immediately above / below it.
step_indices = [i for i, s in enumerate(stages) if s['kind'] == 'step']

box_left, box_right = 1.0, 13.0
box_center = (box_left + box_right) / 2

for step_i in step_indices:
    step   = stages[step_i]
    in_dat = stages[step_i - 1] if step_i > 0 else None
    out_dat = stages[step_i + 1] if step_i + 1 < len(stages) else None

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.set_aspect('equal'); ax.axis('off')

    # Header
    ax.text(7, 6.6,
            f"Pipeline step {step['num']} — {step['title']}   (UPS = {UPS})",
            ha='center', va='center', fontsize=14, fontweight='bold')

    y_in_top, y_in_bot   = 5.6, 4.4
    y_step_top, y_step_bot = 4.0, 2.4
    y_out_top, y_out_bot = 2.0, 0.8

    if in_dat is not None:
        draw_data(ax, in_dat, y_in_bot, y_in_top, box_left, box_right)
        ax.text(box_center, y_in_top + 0.30, 'INPUT', ha='center',
                va='center', fontsize=9, color='#666', fontweight='bold')
        draw_arrow(ax, box_center, y_in_bot - 0.05, y_step_top + 0.05)

    draw_step(ax, step, y_step_bot, y_step_top, box_left, box_right)

    if out_dat is not None:
        draw_arrow(ax, box_center, y_step_bot - 0.05, y_out_top + 0.05)
        draw_data(ax, out_dat, y_out_bot, y_out_top, box_left, box_right)
        ax.text(box_center, y_out_top + 0.30, 'OUTPUT', ha='center',
                va='center', fontsize=9, color='#666', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR,
                       f'pipeline_step{step["num"]}_ups{UPS}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')


# --- combined overview (all stages stacked, same style as before) -----------
fig, ax = plt.subplots(figsize=(15, 15))
ax.set_xlim(0, 14); ax.set_ylim(0, len(stages) * 1.55 + 1)
ax.set_aspect('equal'); ax.axis('off')

y_top = len(stages) * 1.55 + 0.5
ax.text(7, y_top,
        f'Mosaic tomography data simulation — pipeline for  UPS = {UPS}',
        ha='center', va='center', fontsize=14.5, fontweight='bold')
ax.text(7, y_top - 0.55,
        f'N = {N:,}    NTHETA = {NTHETA:,}    '
        f'detector {DET_W}×{DET_H} px @ {PIXEL_UM:g} µm    '
        f'sample ⌀{SAMPLE_D_MM:.1f} × {SAMPLE_H_MM:.1f} mm',
        ha='center', va='center', fontsize=10, color='#333')

row_h = 1.35
gap   = 0.30
y_cursor = y_top - 1.4

for i, s in enumerate(stages):
    y_top_row = y_cursor
    y_bot_row = y_cursor - (row_h if s['kind'] == 'step' else 1.15)

    if s['kind'] == 'step':
        draw_step(ax, s, y_bot_row, y_top_row, box_left, box_right)
    else:
        draw_data(ax, s, y_bot_row, y_top_row, box_left, box_right)

    if i < len(stages) - 1:
        draw_arrow(ax, box_center, y_bot_row - 0.02, y_bot_row - gap + 0.02)

    y_cursor = y_bot_row - gap

y_foot = y_cursor - 0.4
ax.text(7, y_foot,
        f'Peak on-disk footprint:  '
        f'init  {fmt(INIT_N**3*4)}  →  big{UPS}x  {fmt(N**3*4)}'
        f'  →  proj/data  {fmt(NTHETA*N*N*4)} each  →  '
        f'{N_TILES} tiles  {fmt(N_TILES*NTHETA*DET_H*DET_W*4)}',
        ha='center', va='center', fontsize=10, color='#333',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7',
                  edgecolor='#aaa', lw=1))

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, f'pipeline_overview_ups{UPS}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'saved: {out}')
