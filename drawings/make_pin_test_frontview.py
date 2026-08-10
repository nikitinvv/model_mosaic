"""Pin-based test to check samplex roll — FRONT VIEW (looking along beam).

x horizontal, y vertical.  Beam is out of the page (⊙).
The pin stands vertically. Below it: topx block on the rotation stage on the samplex stage.

Ideal motors:
  C1 & C2 place the pin at the SAME lab (x, y)    -> samplex/topx compensate cleanly
  C3 mirrors x to -1 mm at same y                 -> rotation axis is well-calibrated

Any y-shift of the pin across configs = samplex roll / rot-axis wobble.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

# --- detector / beam ---
det_px = 1.4e-3
det_nx = 3232
det_ny = 2426
det_W = det_px * det_nx   # 4.5248 mm horizontal FOV
det_H = det_px * det_ny   # 3.3964 mm vertical FOV

configs = [
    dict(name='Config 1', topx=1.0,        samplex=0.0,   theta=0.0),
    dict(name='Config 2', topx=1.0 + 15.0, samplex=-15.0, theta=0.0),
    dict(name='Config 3', topx=1.0 + 15.0, samplex=+15.0, theta=180.0),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), sharey=True)

x_range = 24.0
y_bot   = -14.0
y_top   = 10.0

# --- element sizes (mm) ---
pin_diam        = 0.25
pin_height      = 6.0
topx_block_w    = 3.0
topx_block_h    = 1.5
rot_stage_r     = 4.0
rot_stage_h     = 3.0
samplex_stage_w = 44.0
samplex_stage_h = 2.5

# --- y stacking ---
y_samplex_bot = -12.0
y_samplex_top = y_samplex_bot + samplex_stage_h
y_rot_bot     = y_samplex_top
y_rot_top     = y_rot_bot + rot_stage_h
y_topx_bot    = y_rot_top
y_topx_top    = y_topx_bot + topx_block_h
y_pin_bot     = y_topx_top
y_pin_top     = y_pin_bot + pin_height    # = 1 mm

# beam center: put detector so the pin tip lies inside its FOV
# FOV is y_beam ± det_H/2 = y_beam ± 1.70 mm; place y_beam = 0
y_beam = 0.0

for i, (ax, cfg) in enumerate(zip(axes, configs)):
    topx      = cfg['topx']
    samplex   = cfg['samplex']
    theta_deg = cfg['theta']
    theta     = np.deg2rad(theta_deg)

    pin_x = samplex + topx * np.cos(theta)

    # --- detector FOV (behind everything, faint) ---
    ax.add_patch(Rectangle((-det_W/2, y_beam - det_H/2), det_W, det_H,
                           facecolor='#d7ecd9', edgecolor='#1b4332',
                           lw=1.5, alpha=0.55, zorder=0.4))
    ax.text(det_W/2 + 0.5, y_beam, 'detector\nFOV',
            fontsize=8, va='center', ha='left', color='#1b4332')
    # beam annotation outside FOV, to the left
    ax.text(-det_W/2 - 0.5, y_beam, '⊙\nbeam',
            fontsize=8, ha='right', va='center', color='#b36b00')

    # --- samplex stage (bottom, wide slab) ---
    ax.add_patch(Rectangle((-samplex_stage_w/2, y_samplex_bot),
                           samplex_stage_w, samplex_stage_h,
                           facecolor='#c9c9c9', edgecolor='#555', lw=1, zorder=1))
    ax.text(-samplex_stage_w/2 + 0.4, y_samplex_bot + samplex_stage_h/2,
            'samplex stage', fontsize=7.5, va='center', color='#333')

    # --- rotation stage (on samplex, centered at x=samplex) ---
    ax.add_patch(Rectangle((samplex - rot_stage_r, y_rot_bot),
                           2*rot_stage_r, rot_stage_h,
                           facecolor='#a8a8a8', edgecolor='#333', lw=1, zorder=2))
    ax.text(samplex, y_rot_bot + rot_stage_h/2, 'rot',
            fontsize=8, ha='center', va='center', color='#111')

    # --- rotation axis: vertical dashed line through x=samplex ---
    rot_top_y = y_top - 1.5
    ax.plot([samplex, samplex], [y_rot_top, rot_top_y],
            color='k', ls='--', lw=1.1, alpha=0.75, zorder=2.5)
    ax.text(samplex + 0.4, rot_top_y, 'rot axis',
            fontsize=8, ha='left', va='top', color='k')

    # --- topx block (on rot stage, offset by dx_topx from rot axis) ---
    dx_topx = topx * np.cos(theta)
    ax.add_patch(Rectangle((samplex + dx_topx - topx_block_w/2, y_topx_bot),
                           topx_block_w, topx_block_h,
                           facecolor='#8a8a8a', edgecolor='#111', lw=1, zorder=3))

    # --- pin (vertical rod on top of topx block) ---
    ax.add_patch(Rectangle((pin_x - pin_diam/2, y_pin_bot),
                           pin_diam, pin_height,
                           facecolor='#c0392b', edgecolor='#5c1408',
                           lw=1.0, zorder=6))
    # pin label + lab-x readout, placed above the detector FOV so nothing collides
    ax.text(pin_x, y_beam + det_H/2 + 0.6,
            f'pin tip\nlab x = {pin_x:+.1f} mm',
            fontsize=9.5, color='#c0392b', fontweight='bold',
            ha='center', va='bottom')

    # --- samplex arrow (beam center → rot axis) ---
    y_arr_sx = y_samplex_bot - 1.2
    if samplex != 0:
        ax.add_patch(FancyArrowPatch((0, y_arr_sx), (samplex, y_arr_sx),
                                     arrowstyle='<->', color='#7a3fa8',
                                     lw=1.9, mutation_scale=14))
    else:
        ax.plot(0, y_arr_sx, marker='|', color='#7a3fa8', markersize=12, mew=2)
    ax.text(samplex/2 if samplex != 0 else 0,
            y_arr_sx - 0.8, f'samplex = {samplex:+.0f} mm',
            fontsize=9.5, color='#7a3fa8',
            ha='center', va='top', fontweight='bold')

    # --- topx arrow: BELOW the detector FOV, in the clear band between
    # FOV bottom (-1.7) and topx block top (-5)
    y_arr_tx = -3.3
    if abs(dx_topx) < 0.5:
        ax.plot(samplex, y_arr_tx, marker='|', color='#b8442a',
                markersize=12, mew=2.5, zorder=5)
        ax.text(samplex + 1.5, y_arr_tx - 0.4,
                f'topx = {topx:+.0f} mm  (very short)',
                fontsize=9.5, color='#b8442a',
                ha='left', va='top', fontweight='bold')
    else:
        ax.add_patch(FancyArrowPatch((samplex, y_arr_tx),
                                     (samplex + dx_topx, y_arr_tx),
                                     arrowstyle='->', color='#b8442a',
                                     lw=1.9, mutation_scale=14, zorder=5))
        ax.text(samplex + dx_topx/2, y_arr_tx - 0.4,
                f'topx = {topx:+.0f} mm',
                fontsize=9.5, color='#b8442a',
                ha='center', va='top', fontweight='bold')

    # --- θ indicator (top-right corner badge) ---
    ax.text(0.97, 0.97, f'θ = {theta_deg:.0f}°',
            transform=ax.transAxes,
            fontsize=11, color='#1a6a2f', fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='#e0f5e0', edgecolor='#2a9d3f'))

    # --- panel title ---
    ax.set_title(cfg['name'], fontsize=12.5, fontweight='bold')

    # --- axis ---
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(y_bot - 3, y_top + 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x  (lab, horizontal)   [mm]', fontsize=10)
    if i == 0:
        ax.set_ylabel('y  (lab, vertical)   [mm]', fontsize=10)
    ax.grid(alpha=0.22)

fig.suptitle(
    'Pin test — front view (looking upstream, from detector side)\n'
    'ideal motors:  C1 ≡ C2 → pin at x = +1 mm   ·   C3 mirrors to x = −1 mm   ·   y should not change',
    fontsize=12.5, y=0.98
)

legend_text = (
    'purple ↔  samplex (linear stage under rot. stage)      red →  topx (linear stage on top of rot. stage)\n'
    'black dashed = rotation axis (vertical)      red rod = pin      green box = detector FOV\n'
    '⊙  = X-ray beam coming out of the page toward viewer\n'
    'Roll of samplex would shift pin in y between C1 and C2 (same nominal lab x, different samplex value).'
)
fig.text(0.5, 0.02, legend_text, ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.45',
                   facecolor='#f6f6f6', edgecolor='gray'))

plt.tight_layout(rect=[0, 0.09, 1, 0.95])
out = '/home/beams2/VNIKITIN/mosaic_modeling/mosaic_pin_test_frontview.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f'saved: {out}')
