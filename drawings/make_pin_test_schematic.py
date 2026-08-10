"""Pin-based test to check samplex roll/parallelism.

3 configurations of (topx, samplex, rotation) applied to a thin pin.
Each panel shows top-down view with beam propagating vertically (top->bottom).

Ideal motors:
  C1 & C2 place the pin at the SAME lab x (+1 mm)  -> tests samplex/topx compensation
  C3 mirrors the pin to lab x = -1 mm              -> tests rotation-axis calibration

Any deviation on the detector reveals roll, axis wobble, or scale error.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Arc, FancyArrowPatch

# --- detector / beam ---
det_px = 1.4e-3
det_nx = 3232
det_W = det_px * det_nx  # 4.5248 mm

# vertical layout (per panel): x is horizontal, z is vertical (beam goes down)
z_source = -55
z_sample = 0
z_det = 40

configs = [
    dict(name='Config 1', topx=1.0,        samplex=0.0,   theta=0.0),
    dict(name='Config 2', topx=1.0 + 15.0, samplex=-15.0, theta=0.0),
    dict(name='Config 3', topx=1.0 + 15.0, samplex=+15.0, theta=180.0),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True)

x_range = 22.0  # ±22 mm

for i, (ax, cfg) in enumerate(zip(axes, configs)):
    topx      = cfg['topx']
    samplex   = cfg['samplex']
    theta_deg = cfg['theta']
    theta     = np.deg2rad(theta_deg)

    # pin lab position (parallel beam: only lab x matters for projection)
    pin_x   = samplex + topx * np.cos(theta)
    pin_dz  = topx * np.sin(theta)          # 0 for both θ=0 and θ=180

    # --- beam (vertical yellow strip, ±FOV/2) ---
    beam_half = det_W / 2
    ax.fill_betweenx([z_source, z_det + 4], -beam_half, beam_half,
                     color='#ffcc33', alpha=0.22, zorder=1)
    for edge in (-beam_half, beam_half):
        ax.plot([edge, edge], [z_source, z_det + 4],
                color='#e69500', ls='--', lw=0.8, alpha=0.7)
    for xa in np.linspace(-beam_half*0.7, beam_half*0.7, 3):
        ax.annotate('', xy=(xa, z_sample - 14), xytext=(xa, z_source + 3),
                    arrowprops=dict(arrowstyle='->', color='#e69500', lw=1.2))
    ax.text(0, z_source - 2, 'X-ray beam', ha='center', va='bottom',
            fontsize=9, color='#b36b00', fontweight='bold')

    # beam center line
    ax.axvline(0, color='gray', ls=':', lw=0.7, alpha=0.6, zorder=0)

    # --- rotation axis (+ marker + dot) ---
    ax.plot(samplex, z_sample, marker='+', color='k',
            markersize=18, markeredgewidth=2.5, zorder=5)
    ax.plot(samplex, z_sample, marker='o', color='k', markersize=5, zorder=5)

    # rotation arc / θ indicator
    arc_r = 3.2
    if theta_deg == 0:
        # small tick at 0
        ax.plot([samplex, samplex + arc_r], [z_sample, z_sample],
                color='#2a9d3f', lw=2)
        ax.plot(samplex + arc_r, z_sample, marker='>', color='#2a9d3f',
                markersize=8)
    else:
        arc = Arc((samplex, z_sample), arc_r*2, arc_r*2,
                  angle=0, theta1=0, theta2=180, color='#2a9d3f', lw=2)
        ax.add_patch(arc)
        ax.plot(samplex - arc_r, z_sample, marker='<', color='#2a9d3f',
                markersize=8)
    ax.text(samplex + arc_r + 0.5, z_sample - arc_r - 1.5,
            f'θ = {theta_deg:.0f}°',
            fontsize=10, color='#2a9d3f', fontweight='bold')

    # --- samplex arrow (beam center → rotation axis) ---
    if samplex != 0:
        y_a = z_sample - 9
        ax.add_patch(FancyArrowPatch((0, y_a), (samplex, y_a),
                                     arrowstyle='<->', color='#7a3fa8',
                                     lw=1.8, mutation_scale=13))
        ax.text(samplex/2, y_a - 1.5, f'samplex = {samplex:+.0f}',
                fontsize=9, color='#7a3fa8',
                ha='center', va='top', fontweight='bold')
    else:
        ax.text(0, z_sample - 10, 'samplex = 0',
                fontsize=9, color='#7a3fa8',
                ha='center', va='top', fontweight='bold')

    # --- topx arrow (rot axis → pin, in the rotating frame projected to lab) ---
    y_b = z_sample + 8
    dx_topx = topx * np.cos(theta)
    ax.add_patch(FancyArrowPatch((samplex, y_b), (samplex + dx_topx, y_b),
                                 arrowstyle='->', color='#b8442a',
                                 lw=1.8, mutation_scale=13))
    ax.text(samplex + dx_topx/2, y_b + 1.5, f'topx = {topx:+.0f}',
            fontsize=9, color='#b8442a',
            ha='center', va='bottom', fontweight='bold')

    # --- pin (dot at sample plane) ---
    ax.plot(pin_x, z_sample, marker='o', color='#c0392b',
            markersize=11, zorder=8, markeredgecolor='k', markeredgewidth=1.2)
    ax.text(pin_x, z_sample + 2.5,
            f'pin\nlab x = {pin_x:+.1f}',
            fontsize=9, color='#c0392b',
            ha='center', va='top', fontweight='bold')

    # --- ray from pin down to detector (parallel beam projection) ---
    ax.plot([pin_x, pin_x], [z_sample, z_det],
            color='#c0392b', ls='--', lw=1, alpha=0.6)
    ax.plot(pin_x, z_det, marker='v', color='#c0392b', markersize=9, zorder=6)

    # --- detector ---
    det_th = 2
    ax.add_patch(Rectangle((-det_W/2, z_det), det_W, det_th,
                           facecolor='#b7e4c7', edgecolor='#1b4332',
                           lw=1.5, zorder=4))
    ax.text(-det_W/2 - 1, z_det + det_th/2, 'det', fontsize=8,
            va='center', ha='right')
    # detector center tick
    ax.plot(0, z_det + det_th, marker='|', color='k', markersize=8)
    # spot position on detector
    ax.text(pin_x, z_det + det_th + 1.5, f'{pin_x:+.1f}',
            fontsize=8.5, color='#c0392b',
            ha='center', va='bottom', fontweight='bold')

    # --- panel title ---
    ax.set_title(f'{cfg["name"]}\n'
                 f'topx = {topx:+.0f}, samplex = {samplex:+.0f}, θ = {theta_deg:.0f}°',
                 fontsize=11)

    # --- axis ---
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(z_det + 8, z_source - 6)  # inverted -> beam goes top→bottom
    ax.set_aspect('equal')
    ax.set_xlabel('lab x  [mm]', fontsize=10)
    if i == 0:
        ax.set_ylabel('z (along beam) ↓', fontsize=10)
    ax.grid(alpha=0.25)

# Overall title
fig.suptitle(
    'Pin test — check samplex roll & rotation-axis calibration\n'
    'ideal motors:  C1 ≡ C2 (both pin at +1 mm)   |   C3 mirrors to −1 mm',
    fontsize=12.5, y=0.98
)

# Legend
legend_text = (
    'purple ↔  samplex (under rot. stage) — moves the rotation axis\n'
    'red →     topx (on top of rot. stage) — moves pin in the rotating frame\n'
    'green arc rotation θ    ●red pin at sample plane    ▼red pin on detector\n'
    'yellow strip parallel X-ray beam (FOV 4.52 mm across the detector)'
)
fig.text(0.5, 0.02, legend_text, ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.45',
                   facecolor='#f6f6f6', edgecolor='gray'))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
out = '/home/beams2/VNIKITIN/mosaic_modeling/mosaic_pin_test.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f'saved: {out}')
