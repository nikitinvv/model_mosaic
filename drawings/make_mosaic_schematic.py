"""Schematic of mosaic CT scanning geometry (top-down view).

Sample: 30 mm dia cylinder
topx (on rotation stage): -30 mm  --> sample offset -30 from rotation axis
samplex (below rotation stage): +30 mm --> rotation axis offset +30 from beam
Detector: centered on beam axis
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Arc, FancyArrowPatch

# --- geometry (mm) ---
sample_diam = 30.0
sample_R = sample_diam / 2
det_px = 1.4e-3           # mm
det_nx = 3232
det_ny = 2426
det_W = det_px * det_nx   # 4.5248 mm
topx = -30.0
samplex = +30.0

sample_x0 = samplex + topx   # 0
rot_x = samplex              # +30

# --- plot layout: horizontal beam, top view (bird's eye) ---
# z: beam direction (horizontal), x: transverse (vertical in plot)
fig, ax = plt.subplots(figsize=(15, 8.5))

# z positions (not to scale for clarity)
z_beam0 = -90
z_sample = 0
z_det = 55
det_th = 3

# --- beam ---
beam_half = det_W / 2  # parallel beam ~ detector width
ax.fill_between([z_beam0, z_det], -beam_half, beam_half,
                color='#ffcc33', alpha=0.20, zorder=1, label='X-ray beam')
ax.plot([z_beam0, z_det], [beam_half, beam_half], color='#e69500', ls='--', lw=0.8)
ax.plot([z_beam0, z_det], [-beam_half, -beam_half], color='#e69500', ls='--', lw=0.8)
for x in np.linspace(-beam_half*0.85, beam_half*0.85, 5):
    ax.annotate('', xy=(z_sample - sample_R - 4, x), xytext=(z_beam0 + 2, x),
                arrowprops=dict(arrowstyle='->', color='#e69500', lw=1.3, alpha=0.9))
ax.text(z_beam0 + 4, beam_half + 3.5, 'X-ray beam (parallel, from source 50 m upstream)',
        fontsize=11, color='#b36b00', fontweight='bold')

# --- beam center line ---
ax.axhline(0, color='gray', ls=':', lw=0.8, alpha=0.6, zorder=0)
ax.text(z_beam0 + 2, -1.7, 'beam axis (x = 0)', fontsize=8, color='gray')

# --- sample cylinder (top view = circle) at current pos ---
sample = Circle((z_sample, sample_x0), sample_R,
                facecolor='#a6cee3', edgecolor='#1f5b8a', lw=2, alpha=0.65, zorder=3)
ax.add_patch(sample)
ax.text(z_sample, sample_x0 - sample_R - 3.5,
        'Sample (30 mm ⌀)\nat θ = 0°',
        ha='center', va='top', fontsize=10, color='#1f5b8a', fontweight='bold')

# --- illuminated strip (chord through sample within beam) ---
strip = Rectangle((z_sample - sample_R, -beam_half),
                  2*sample_R, det_W,
                  facecolor='#e63946', alpha=0.35, edgecolor='#a01a2a',
                  lw=1.2, zorder=4)
ax.add_patch(strip)
ax.text(z_sample + sample_R + 1, 0, f'illuminated\nstrip\n({det_W:.2f} mm)',
        fontsize=8.5, color='#a01a2a', va='center')

# --- rotation axis (out of page, marked as +/dot) ---
ax.plot(z_sample, rot_x, marker='+', color='k', markersize=22, markeredgewidth=3, zorder=6)
ax.plot(z_sample, rot_x, marker='o', color='k', markersize=6, zorder=6)
ax.text(z_sample + 3.5, rot_x + 0.5, 'rotation axis\n(vertical, out of page)',
        fontsize=10, va='center')

# rotation arc + arrow
arc = Arc((z_sample, rot_x), 9, 9, angle=0, theta1=200, theta2=520,
          color='#2a9d3f', lw=2.2, zorder=6)
ax.add_patch(arc)
ax.annotate('', xy=(z_sample - 4.4, rot_x - 1.5), xytext=(z_sample - 4.2, rot_x - 0.5),
            arrowprops=dict(arrowstyle='->', color='#2a9d3f', lw=2.2))
ax.text(z_sample - 6.5, rot_x + 5.5, 'θ', fontsize=13, color='#2a9d3f', fontweight='bold')

# --- orbit of sample center as it rotates around rot axis ---
r_orbit = abs(topx)
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(z_sample + r_orbit*np.sin(theta), rot_x - r_orbit*np.cos(theta),
        color='#666', ls='--', lw=0.9, alpha=0.6, zorder=1)

# ghost sample positions at 90°, 180°, 270° (as it rotates)
for th_deg, alpha_g in [(90, 0.25), (180, 0.25), (270, 0.25)]:
    th = np.deg2rad(th_deg)
    cx = z_sample + r_orbit*np.sin(th)
    cy = rot_x - r_orbit*np.cos(th)
    ghost = Circle((cx, cy), sample_R, facecolor='none',
                   edgecolor='#1f5b8a', lw=1.1, ls=':', alpha=alpha_g*2.4, zorder=2)
    ax.add_patch(ghost)
    ax.text(cx, cy, f'θ={th_deg}°', ha='center', va='center',
            fontsize=8, color='#1f5b8a', alpha=0.75)

# --- samplex motor arrow (beam center → rotation axis) ---
arr1 = FancyArrowPatch((z_sample - 8, 0), (z_sample - 8, rot_x),
                       arrowstyle='<->', color='#7a3fa8', lw=2.2,
                       mutation_scale=15)
ax.add_patch(arr1)
ax.text(z_sample - 9.5, rot_x/2,
        f'samplex\n= +{samplex:.0f} mm\n(under rot. stage)',
        fontsize=10, color='#7a3fa8', ha='right', va='center', fontweight='bold')

# --- topx motor arrow (rotation axis → sample center) ---
arr2 = FancyArrowPatch((z_sample + 8, rot_x), (z_sample + 8, sample_x0),
                       arrowstyle='<->', color='#b8442a', lw=2.2,
                       mutation_scale=15)
ax.add_patch(arr2)
ax.text(z_sample + 9.5, (rot_x + sample_x0)/2,
        f'topx\n= {topx:.0f} mm\n(on top of rot. stage)',
        fontsize=10, color='#b8442a', ha='left', va='center', fontweight='bold')

# --- detector ---
det = Rectangle((z_det, -det_W/2), det_th, det_W,
                facecolor='#b7e4c7', edgecolor='#1b4332', lw=2, zorder=5)
ax.add_patch(det)
ax.text(z_det + det_th + 2, 0,
        f'Detector\n3232 × 2426 px @ 1.4 μm\nFOV = {det_W:.2f} × {det_px*det_ny:.2f} mm',
        fontsize=9.5, va='center')
# label det center
ax.plot(z_det + det_th/2, 0, 'k.', markersize=5)
ax.annotate('', xy=(z_det, -det_W/2 - 5), xytext=(z_sample, -det_W/2 - 5),
            arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
ax.text((z_sample + z_det)/2, -det_W/2 - 6.5, 'R₂ = 1 m',
        fontsize=9, color='#333', ha='center', va='top')

# --- source arrow off-plot (left) ---
ax.annotate('', xy=(z_beam0, 0), xytext=(z_beam0 - 8, 0),
            arrowprops=dict(arrowstyle='->', color='#b36b00', lw=1.8))
ax.text(z_beam0 - 4, -3.5, 'from source\n(R₁ = 50 m)',
        fontsize=9, color='#b36b00', ha='center', va='top')

# --- annotation box: interpretation ---
info = (
    f'Config: sample center at beam (x = samplex + topx = 0)\n'
    f'Rotation axis offset +{samplex:.0f} mm from beam\n'
    f'Sample orbits rot. axis with radius |topx| = {abs(topx):.0f} mm\n'
    f'→ 30 mm sample only enters the {det_W:.2f} mm FOV over a\n'
    f'   narrow angular range near θ = 0°; mosaic in (samplex, topx)\n'
    f'   builds full sinogram.'
)
ax.text(0.98, 0.03, info, transform=ax.transAxes, fontsize=9,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e1',
                  edgecolor='#c9a227', lw=1))

# --- axes ---
ax.set_xlim(z_beam0 - 12, z_det + 32)
ax.set_ylim(-30, 65)
ax.set_aspect('equal')
ax.set_xlabel('z  (along beam)   [mm; z-axis compressed vs. actual R₂ = 1 m]', fontsize=10)
ax.set_ylabel('x  (transverse, mosaic direction)   [mm]', fontsize=10)
ax.set_title('Mosaic CT scan geometry — top-down view\n'
             f'sample 30 mm ⌀,  topx = {topx:.0f} mm,  samplex = +{samplex:.0f} mm,  detector centered on beam',
             fontsize=12)
ax.grid(alpha=0.25)

plt.tight_layout()
out = '/home/beams2/VNIKITIN/mosaic_modeling/mosaic_schematic.png'
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f'saved: {out}')
