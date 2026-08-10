"""Parallel-beam approximation validity for APS 2-BM mosaic scans.

Geometry:  R1 = 50 m,  R2 = 1 m,  sample ⌀30 mm,  detector 3232×2426 @ 1.38 µm.
Scanning the center of the sample (rotation axis on beam axis).

The plot shows:
  (left)  true-scale schematic of source→sample→detector (broken axis for R1)
          with the ray fan from the source through the visible FOV
  (right) numerical breakdown — magnification, divergence angles, and the
          depth-dependent pixel shift, i.e., the actual error incurred by
          treating the geometry as parallel-beam Radon.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arc, Polygon

# --- physical parameters (mm) ---
R1 = 50_000.0
R2 =  1_000.0
sample_D = 30.0
sample_R = 15.0
det_px_um = 1.38
det_W_mm  = 3232 * det_px_um / 1000.0     # 4.460
det_H_mm  = 2426 * det_px_um / 1000.0     # 3.348
M         = (R1 + R2) / R1                # 1.020

x_vis     = det_W_mm / 2.0 / M            # sample-plane x that projects to FOV edge

def cone_x_det(x, z):
    return x * (R1 + R2) / (R1 + z)

def px(mm): return mm / (det_px_um / 1000.0)
def urad(rad): return rad * 1e6

alpha_fov       = (det_W_mm/2) / R1           # divergence half-angle (rad)
cone_spread_mm  = alpha_fov * sample_D        # lateral spread of one ray traversing the sample
cone_spread_um  = cone_spread_mm * 1000
cone_spread_px  = cone_spread_um / det_px_um

# =====================================================================
fig, (axL, axR) = plt.subplots(2, 1, figsize=(15, 11),
                                gridspec_kw={'height_ratios': [1.0, 1.0]})

# =====================================================================
# LEFT: schematic geometry (display units; distances labeled)
# =====================================================================
axL.set_aspect('equal')
axL.axis('off')

# display x positions
x_src, x_smp, x_det = -9.0, 0.0, 4.0

# y-scale: display 1.0 = 15 mm (sample radius).  So visible half-x (2.19 mm)
# is 0.146 display units.
def mm_to_disp(y_mm): return y_mm / sample_R    # 15 mm → 1.0 display units
y_smp_edge = mm_to_disp(sample_R)                # 1.0
y_vis      = mm_to_disp(x_vis)                   # 0.146

# --- beam axis ---
axL.axhline(0, color='gray', ls=':', lw=0.6, alpha=0.6, zorder=0)

# --- source ---
axL.plot(x_src, 0, marker='*', color='#ff9800',
         markersize=42, markeredgecolor='#b35a00', mew=1.4, zorder=5)
axL.text(x_src, -0.55, 'X-ray source\n(APS 2-BM)',
         ha='center', va='top', fontsize=10.5, fontweight='bold')

# --- sample (30 mm diameter circle in x-z plane, seen from above) ---
smp = Circle((x_smp, 0), y_smp_edge, facecolor='#a6cee3',
             edgecolor='#1f5b8a', lw=2, alpha=0.7, zorder=3)
axL.add_patch(smp)
# rotation axis dot at center
axL.plot(x_smp, 0, marker='+', color='k', markersize=14, mew=2, zorder=6)
axL.text(x_smp, y_smp_edge + 0.15,
         f'sample  ⌀ {sample_D:.0f} mm',
         ha='center', va='bottom', fontsize=10.5, color='#1f5b8a', fontweight='bold')

# --- shade the "visible strip" through sample (what center tile captures) ---
axL.add_patch(Rectangle((x_smp - y_smp_edge - 0.05, -y_vis),
                        2*y_smp_edge + 0.1, 2*y_vis,
                        facecolor='#ffe082', alpha=0.55, zorder=2))

# --- detector ---
det_disp_H = 2 * mm_to_disp(det_H_mm/2)          # ~0.223
det_thk    = 0.15
axL.add_patch(Rectangle((x_det - det_thk/2, -det_disp_H/2),
                        det_thk, det_disp_H,
                        facecolor='#b7e4c7', edgecolor='#1b4332', lw=1.8, zorder=4))
axL.text(x_det + 0.35, det_disp_H/2 + 0.05,
         f'detector\n3232 × 2426 px\n1.38 µm pixel\nFOV {det_W_mm:.2f} × {det_H_mm:.2f} mm',
         ha='left', va='center', fontsize=9.5)

# --- EXAGGERATED cone ray fan (real divergence 44.6 µrad is invisible) ---
m_exag = 0.06                                    # slope; real would be ~5e-6
x_end  = x_smp + y_smp_edge + 0.3                # stop rays just past the sample
for m in np.linspace(-m_exag, m_exag, 9):
    axL.plot([x_src, x_end], [0, m * (x_end - x_src)],
             color='#ff8c00', lw=1.0, alpha=0.55, zorder=1)

axL.text(x_src + 1.4, 0.85, 'cone rays from source',
         color='#c25f00', fontsize=10)
axL.text(x_src + 1.4, 0.60, '(divergence exaggerated ~1200×)',
         color='#c25f00', fontsize=9, style='italic')

# --- Δx = α · sample_size triangle (top ray through sample) ---
# The topmost cone ray enters the sample at its highest entry point A,
# and exits at its highest exit point C (higher than A by Δx).
# Triangle: A (entry, top-left), C (exit, top-right), B (right-angle at C_x, A_y).
TRI = '#c1272d'

# Intersect line y = m_exag·(x − x_src) with circle (x−x_smp)² + y² = y_smp_edge²
_a = 1 + m_exag**2
_b = -2 * x_smp - 2 * m_exag**2 * x_src
_c = x_smp**2 + m_exag**2 * x_src**2 - y_smp_edge**2
_disc = _b**2 - 4*_a*_c
_sd = np.sqrt(_disc)
x_A = (-_b - _sd) / (2*_a)                     # entry (left root)
x_C = (-_b + _sd) / (2*_a)                     # exit  (right root)
y_A = m_exag * (x_A - x_src)
y_C = m_exag * (x_C - x_src)
tri_A = (x_A, y_A)                             # entry (bottom-left of triangle)
tri_C = (x_C, y_C)                             # exit  (top-right)
tri_B = (x_C, y_A)                             # right-angle corner
delta_x_exag = y_C - y_A

# highlight the top ray (hypotenuse) — extend from source through triangle
axL.plot([x_src, x_end],
         [0, m_exag * (x_end - x_src)],
         color=TRI, lw=1.8, alpha=0.95, zorder=6)

# triangle: fill + outline
axL.add_patch(Polygon([tri_A, tri_B, tri_C], closed=True,
                      facecolor=TRI, alpha=0.22, zorder=6))
axL.add_patch(Polygon([tri_A, tri_B, tri_C], closed=True,
                      fill=False, edgecolor=TRI, lw=2, zorder=7))
# emphasize horizontal side (line through the sample at ray-entry height)
axL.plot([tri_A[0], tri_B[0]], [y_A, y_A], color=TRI, lw=2.4, zorder=7)

# vertical Δx arrow
axL.annotate('', xy=tri_C, xytext=tri_B,
             arrowprops=dict(arrowstyle='<->', color=TRI, lw=1.8))

# Δx callout — placed in the empty space between sample and detector
lbl_x = x_smp + y_smp_edge + 0.35
lbl_y = y_C + 0.25
axL.annotate('',
             xy=(tri_C[0] + 0.02, tri_C[1]),
             xytext=(lbl_x, lbl_y),
             arrowprops=dict(arrowstyle='-', color=TRI, lw=1.0))
axL.text(lbl_x, lbl_y,
         f'Δx  =  α · sample_size\n     =  {cone_spread_um:.2f} µm\n     ≈  1 pixel',
         color=TRI, fontsize=10, fontweight='bold',
         va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='#fdecec', edgecolor=TRI, lw=1))

# angle α arc at vertex A
arc_r = 0.4
arc_theta_end = np.degrees(np.arctan(m_exag))
axL.add_patch(Arc(tri_A, 2*arc_r, 2*arc_r, angle=0,
                  theta1=0, theta2=arc_theta_end,
                  color=TRI, lw=1.8, zorder=8))
axL.text(tri_A[0] + arc_r + 0.04,
         y_A + arc_r * np.tan(np.radians(arc_theta_end/2)),
         'α', color=TRI, fontsize=13, fontweight='bold',
         ha='left', va='center')

# sample_size label (below the horizontal side)
axL.text((x_A + x_C)/2, y_A - 0.08,
         f'sample_size = {sample_D:.0f} mm',
         color=TRI, fontsize=9.5, ha='center', va='top', fontweight='bold')

# exaggeration note — put it below the whole sample
axL.text(x_smp, -y_smp_edge - 0.45,
         '(Δx and α exaggerated — real Δx ≈ 1 detector pixel)',
         color=TRI, fontsize=8.5, ha='center', va='top', style='italic')

# --- distance annotations ---
y_dim = -1.65
axL.annotate('', xy=(x_smp - 0.05, y_dim), xytext=(x_src + 0.6, y_dim),
             arrowprops=dict(arrowstyle='<->', color='#333', lw=1.3))
axL.text((x_src + x_smp)/2, y_dim - 0.15, 'R₁ = 50 m  (source → sample)',
         ha='center', va='top', fontsize=11, fontweight='bold', color='#333')

axL.annotate('', xy=(x_det - det_thk/2 - 0.02, y_dim),
             xytext=(x_smp + 0.05, y_dim),
             arrowprops=dict(arrowstyle='<->', color='#333', lw=1.3))
axL.text((x_smp + x_det)/2, y_dim - 0.15, 'R₂ = 1 m',
         ha='center', va='top', fontsize=11, fontweight='bold', color='#333')

# --- inset callout: exaggerated divergence at detector edge ---
# little box at bottom-right showing the tiny half-angle
axL.text(0.98, 0.02,
         (f'divergence half-angle to visible FOV edge\n'
          f'  α  =  ({det_W_mm/2:.2f} mm) / R₁  =  {urad(alpha_fov):.1f} µrad  =  '
          f'{np.degrees(alpha_fov)*3600:.1f}″'),
         transform=axL.transAxes, ha='right', va='bottom',
         fontsize=9.5, color='#c25f00',
         bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='#fff3e0', edgecolor='#c25f00', lw=1))

# --- title ---
axL.set_title('Parallel-beam approximation — beamline schematic\n'
              'scanning the CENTER of a 30 mm sample',
              fontsize=12, fontweight='bold', pad=8)

# axis limits
axL.set_xlim(x_src - 1.5, x_det + 5.0)
axL.set_ylim(-2.2, 1.7)

# =====================================================================
# BOTTOM: numerical breakdown table
# =====================================================================
axR.set_aspect('auto')
axR.axis('off')
axR.set_xlim(0, 1)
axR.set_ylim(0, 1)

axR.set_title('Numerical check — does parallel beam hold?',
              fontsize=12, fontweight='bold', pad=8, loc='left')

# Build the table as multiple text rows for control
rows = [
    ('Geometric magnification',
     f'M = (R₁+R₂)/R₁ = 51/50 = 1.020',
     'global 2 % zoom;\ncan be absorbed into voxel size'),
    ('Divergence half-angle\n(source → visible FOV edge)',
     f'α = (det_W / 2) / R₁\n'
     f'  = ({det_W_mm/2:.2f} mm) / (50 m)\n'
     f'  = {urad(alpha_fov):.1f} µrad  =  {np.degrees(alpha_fov)*3600:.1f} arcsec',
     'source-to-sample distance only —\ndoes NOT involve the sample size'),
    ('Cone-effect estimate\ninside the sample',
     f'Δx = α · sample_size\n'
     f'  = {urad(alpha_fov):.1f} µrad · {sample_D:.0f} mm\n'
     f'  = {cone_spread_um:.2f} µm  =  {cone_spread_px:.2f} pixels',
     'how far a single ray drifts\nlaterally while traversing the\n30 mm sample'),
]

y0 = 0.88
dy = 0.24
for i, (name, formula, note) in enumerate(rows):
    y = y0 - i * dy
    # row background
    axR.add_patch(Rectangle((0.0, y - dy*0.85), 1.0, dy*0.9,
                            facecolor='#fafafa' if i % 2 else '#eef2f7',
                            edgecolor='none', zorder=0))
    axR.text(0.02, y - dy*0.10, name,
             ha='left', va='top', fontsize=9.5,
             fontweight='bold', color='#1e3f6e')
    axR.text(0.36, y - dy*0.10, formula,
             ha='left', va='top', fontsize=9,
             family='monospace', color='#111')
    axR.text(0.78, y - dy*0.10, note,
             ha='left', va='top', fontsize=8.5,
             style='italic', color='#555')

# verdict box at bottom
axR.text(0.5, 0.05,
         f'Verdict:  α·sample_size = {cone_spread_px:.2f} pixels  <  1 pixel\n'
         '⇒  parallel-beam Radon transform is a good approximation.',
         transform=axR.transAxes, ha='center', va='bottom',
         fontsize=11, fontweight='bold', color='#1b4332',
         bbox=dict(boxstyle='round,pad=0.5',
                   facecolor='#d7ecd9', edgecolor='#1b4332', lw=1.2))

plt.tight_layout()
out = '/home/beams2/VNIKITIN/mosaic_modeling/parallel_beam_approx.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'saved: {out}')
