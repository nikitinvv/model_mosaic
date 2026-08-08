#!/usr/bin/env python
"""Mosaic scan schematic for the experimental setup.

Physical dimensions stay constant across --ups; only the voxel-count
representation of the pipeline changes.  At UPS=1 (init/prototype) the
schematic reports coarse voxel counts (detector 404×303, sample 2560×2460)
at 11.2 µm/voxel.  At UPS=8 (full real experiment) it reports the real
counts (detector 3232×2426, sample 20480×19680) at 1.4 µm/voxel.

Sample cylinder ~28.67 × 27.55 mm tall (real physical size), rotating
about the vertical axis.  Reconstruction voxel = detector pixel.
Angles-per-scan follows NTHETA = 3·N/4 (matches step2_model_*.py).

360-deg extended-FOV mosaic:
  - rotation axis fixed at horizontal centre
  - tile 0 has the rotation axis AXIS_INSET (=200) px INSIDE its left border
    (tile + its 180-deg mirror overlap by 2*AXIS_INSET, giving a clean
    central disk in virtual radial coverage)
  - tiles 1..N are shifted in +x by (DET_W - OVERLAP) each => concentric
    annuli in virtual (r, phi) sample space
  - number of x-tiles auto-computed to cover the full sample radius
Detector height < sample height => z-stacking required, drawn as a side view.

Produces:
  mosaic_schematic.png    figure
  mosaic_positions.txt    x tile-origin list (physical detector px)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge, FancyArrowPatch

#
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=1,
                   help="pipeline upsample factor (matches step1_upsample --ups). "
                        "1 = init/prototype (small dims, coarse voxel); "
                        "8 = full real experiment (3232×2426 detector, 1.4 µm voxel).")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory (positions txt is written here)")
    return p.parse_args()


_A = _parse_args()
UPS  = _A.ups
BASE = _A.path

# Detector at real physical size (3232 × 2426 px @ 1.4 µm/px, overlaps 200 px)
# expressed in pipeline-voxel units.  Voxel size scales inversely with UPS
# to keep physical dimensions constant: at UPS=1 voxel=11.2 µm, at UPS=8
# voxel=1.4 µm (matches step2_model's default VOXELSIZE).
DET_W        = 3232 * UPS // 8
DET_H        = 2426 * UPS // 8
PIXEL_UM     = 1.4 * 8 / UPS
VOXEL_UM     = PIXEL_UM
OVERLAP      = 200 * UPS // 8
AXIS_INSET   = 200 * UPS // 8
Z_OVERLAP    = 200 * UPS // 8

# Sample cylinder in pipeline voxels.  Init (before upsampling) is
# 2560 × 2744 × 2744 with CIRCLE_DIAM=2560 and Z_PAD=50 → sample z 2460.
# After upsampling by UPS, dimensions scale linearly.
SAMPLE_D_PX  = 2560 * UPS                         # mask diameter in voxels
SAMPLE_H_PX  = 2460 * UPS                         # sample z-extent
SAMPLE_D_MM  = SAMPLE_D_PX * VOXEL_UM * 1e-3      # ~28.67 mm (real brain size)
SAMPLE_H_MM  = SAMPLE_H_PX * VOXEL_UM * 1e-3      # ~27.55 mm

# Nyquist-scaled angle count (matches step2_model_*.py: NTHETA = 3·N/4,
# with N = 2744·UPS).  Draw only up to MAX_DRAWN_SPOKES spokes to keep the
# right-panel figure legible when NTHETA is in the thousands.
_N_PIPELINE      = 2744 * UPS
NTHETA           = 3 * _N_PIPELINE // 4
MAX_DRAWN_SPOKES = 64
ANG_MAX          = 360


def px_to_mm(v):
    return v * PIXEL_UM * 1e-3


def mm_to_px(v):
    return v / (PIXEL_UM * 1e-3)


def compute_x_layout(sample_r_px: float):
    """x_axis (px, ref=0), tile origins (physical detector px), and virtual
    radial coverage (r_in, r_out): tile 0 is a disk, tiles 1..N annuli."""
    step   = DET_W - OVERLAP
    r0_max = DET_W - AXIS_INSET
    if r0_max >= sample_r_px:
        n_shifts = 0
    else:
        n_shifts = int(np.ceil((sample_r_px - r0_max) / step))
    n_tiles = n_shifts + 1
    x_axis = 0.0
    origins = np.empty(n_tiles)
    origins[0] = x_axis - AXIS_INSET
    for i in range(1, n_tiles):
        origins[i] = origins[i - 1] + step
    r_in  = np.zeros(n_tiles)
    r_out = np.empty(n_tiles)
    r_out[0] = r0_max
    for i in range(1, n_tiles):
        r_in[i]  = origins[i] - x_axis
        r_out[i] = r_in[i] + DET_W
    return x_axis, origins, r_in, r_out


def compute_z_stack(sample_h_px: float):
    """Uniform-step z-positions with symmetric overshoot on top and bottom
    so that EVERY neighbouring pair (including the last one) has the same
    Z_OVERLAP.  z0 starts at -overshoot; z_{N-1} ends at sample_h_px + overshoot."""
    step_z = DET_H - Z_OVERLAP
    if sample_h_px <= DET_H:
        overshoot = (DET_H - sample_h_px) / 2.0
        return np.array([-overshoot]), overshoot
    n_z = int(np.ceil((sample_h_px - DET_H) / step_z)) + 1
    total_span = n_z * DET_H - (n_z - 1) * Z_OVERLAP
    overshoot = (total_span - sample_h_px) / 2.0
    positions = -overshoot + np.arange(n_z) * step_z
    return positions, overshoot


def main() -> None:
    sample_r_px = mm_to_px(SAMPLE_D_MM) / 2.0
    sample_h_px = mm_to_px(SAMPLE_H_MM)

    x_axis, origins, r_in, r_out = compute_x_layout(sample_r_px)
    z_positions, z_overshoot = compute_z_stack(sample_h_px)
    n_tiles = len(origins)
    n_z     = len(z_positions)
    virt_R  = r_out[-1]
    step_mm = px_to_mm(DET_W - OVERLAP)
    total_proj = NTHETA * n_tiles * n_z

    print("=== mosaic layout ===")
    print(f"pixel size      : {PIXEL_UM} µm")
    print(f"detector        : {DET_W} x {DET_H} px "
          f"({px_to_mm(DET_W):.3f} x {px_to_mm(DET_H):.3f} mm)")
    print(f"sample cylinder : ⌀{SAMPLE_D_MM:.3f} mm × {SAMPLE_H_MM:.3f} mm tall  "
          f"({SAMPLE_D_PX} × {SAMPLE_H_PX} voxels @ {VOXEL_UM} µm)")
    print(f"axis inset      : {AXIS_INSET} px = {px_to_mm(AXIS_INSET)*1e3:.1f} µm")
    print(f"x-overlap       : {OVERLAP} px = {px_to_mm(OVERLAP)*1e3:.1f} µm")
    print(f"x-shift step    : {DET_W - OVERLAP} px = {step_mm:.3f} mm")
    print(f"x tiles         : {n_tiles} (0 + {n_tiles-1} shifts)")
    print(f"x origins  (mm) : {[f'{px_to_mm(o):+.3f}' for o in origins]}")
    for i in range(n_tiles):
        print(f"  tile {i}: x=[{px_to_mm(origins[i]):+.3f}, "
              f"{px_to_mm(origins[i]+DET_W):+.3f}] mm  "
              f"virtual r=[{px_to_mm(r_in[i]):.3f}, "
              f"{px_to_mm(r_out[i]):.3f}] mm")
    print(f"virtual FOV     : diameter {px_to_mm(2*virt_R):.3f} mm "
          f"= {int(2*virt_R)} px "
          f"({'OK' if 2*virt_R >= 2*sample_r_px else 'INSUFFICIENT'})")
    # Reconstruction voxel = detector pixel (upsample_big.py runs UPS=1).
    rec_ratio = VOXEL_UM / PIXEL_UM
    rec_nx = int(round(2 * virt_R / rec_ratio))
    rec_nz = int(round(sample_h_px / rec_ratio))
    print(f"REC volume      : {rec_nx} x {rec_nx} x {rec_nz} voxels "
          f"@ {VOXEL_UM} µm  "
          f"(nx mod 8 = {rec_nx%8}, nz mod 8 = {rec_nz%8})")
    print(f"                : {rec_nx*rec_nx*rec_nz:.3e} voxels "
          f"= {rec_nx*rec_nx*rec_nz*4/1e12:.2f} TB (float32)")
    print(f"z stacks        : {n_z} (step {px_to_mm(DET_H-Z_OVERLAP):.3f} mm, "
          f"uniform overlap {Z_OVERLAP} px)")
    print(f"z overshoot     : {z_overshoot:.1f} px = "
          f"{px_to_mm(z_overshoot)*1e3:.1f} µm (each end)")
    print(f"z origins  (mm) : {[f'{px_to_mm(z):+.3f}' for z in z_positions]}")
    print(f"angles          : {NTHETA} over {ANG_MAX}°")
    print(f"TOTAL PROJ.     : {NTHETA} × {n_tiles} × {n_z} = {total_proj}")

    os.makedirs(BASE, exist_ok=True)
    positions_path = os.path.join(BASE, f"mosaic_positions{UPS}.txt")
    np.savetxt(
        positions_path,
        origins.astype(int), fmt="%d",
        header=(f"x tile origins (detector px).  axis={int(x_axis)}, "
                f"tile={DET_W}, overlap={OVERLAP}, axis_inset={AXIS_INSET}, "
                f"z stacks={n_z}, sample ⌀{SAMPLE_D_MM:.2f} mm"),
    )

    # ============ figure ============
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(20, 8),
                                         gridspec_kw={"width_ratios": [1.3, 0.55, 1.0]})

    tile_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
                   "tab:brown", "tab:pink", "tab:olive", "tab:cyan"][:n_tiles]

    # --------- LEFT: full physical view (mm) ----------
    axL.set_aspect("equal")
    x_lo_mm = -SAMPLE_D_MM / 2 - 3.0
    x_hi_mm = px_to_mm(origins[-1] + DET_W) + 3.0
    y_lim   = SAMPLE_D_MM / 2 + 4.0
    axL.set_xlim(x_lo_mm, x_hi_mm)
    axL.set_ylim(y_lim, -y_lim)
    axL.set_xlabel("x [mm]  (relative to rotation axis)")
    axL.set_ylabel("y [mm]")
    axL.set_title(
        f"360° extended-FOV mosaic — {n_tiles} x-tiles × {n_z} z-stacks × {NTHETA} angles\n"
        f"detector {DET_W}×{DET_H} px @ {PIXEL_UM} µm  |  "
        f"voxel = det.pixel = {VOXEL_UM} µm  |  "
        f"overlap {OVERLAP} px = {px_to_mm(OVERLAP)*1e3:.0f} µm"
    )

    # Full sample cross-section (cylinder → circle in xy plane)
    axL.add_patch(Circle((0, 0), SAMPLE_D_MM / 2,
                         facecolor="tab:red", alpha=0.08,
                         edgecolor="tab:red", lw=1.8))
    axL.text(0, SAMPLE_D_MM / 2 + 0.4,
             f"sample cylinder ⌀{SAMPLE_D_MM:.2f} mm",
             ha="center", color="tab:red", fontsize=10)

    # Detector tiles at mid-z, drawn as thin horizontal bars because
    # DET_H (3.4 mm) is much smaller than sample height/diameter
    y_tile_mm = -px_to_mm(DET_H) / 2
    tile_h_mm = px_to_mm(DET_H)
    for i, (x0, col) in enumerate(zip(origins, tile_colors)):
        x0_mm = px_to_mm(x0)
        w_mm  = px_to_mm(DET_W)
        axL.add_patch(Rectangle((x0_mm, y_tile_mm), w_mm, tile_h_mm,
                                facecolor=col, alpha=0.35,
                                edgecolor=col, lw=1.4))
        axL.text(x0_mm + w_mm / 2, y_tile_mm + tile_h_mm / 2, f"{i}",
                 ha="center", va="center", fontsize=16, color=col,
                 fontweight="bold")

    # Overlap bands (physical x-overlap)
    for i in range(n_tiles - 1):
        x_b = px_to_mm(origins[i + 1])
        band_w = px_to_mm(origins[i] + DET_W) - x_b
        axL.add_patch(Rectangle((x_b, y_tile_mm), band_w, tile_h_mm,
                                facecolor="0.15", alpha=0.55, edgecolor="none"))

    # Rotation axis
    axL.axvline(0, color="black", lw=1.5)
    axL.plot(0, 0, "+", color="black", ms=16, mew=2.4)

    # Axis-inset annotation
    axL.annotate(
        f"rot. axis is {AXIS_INSET} px = {px_to_mm(AXIS_INSET)*1e3:.0f} µm\n"
        f"inside tile 0's left border",
        xy=(0, y_tile_mm),
        xytext=(-SAMPLE_D_MM / 2 + 2, y_tile_mm - 4),
        ha="left", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1))

    # x-shift arrows: one per shift, labelled in mm, positioned BELOW tiles
    y_arrow = y_tile_mm + tile_h_mm + 1.6
    for i in range(n_tiles - 1):
        x_from = px_to_mm(origins[i]) + px_to_mm(DET_W) / 2
        x_to   = px_to_mm(origins[i+1]) + px_to_mm(DET_W) / 2
        axL.annotate("", xy=(x_to, y_arrow), xytext=(x_from, y_arrow),
                     arrowprops=dict(arrowstyle="->", color="0.25", lw=1.2))
        axL.text((x_from + x_to) / 2, y_arrow + 0.5,
                 f"{step_mm:.2f} mm", ha="center", va="top",
                 color="0.25", fontsize=9)

    # Tile-origin ticks in mm above the tiles
    for x0 in origins:
        x0_mm = px_to_mm(x0)
        axL.plot([x0_mm, x0_mm], [y_tile_mm - 0.15, y_tile_mm - 0.6],
                 color="0.4", lw=0.8)
        axL.text(x0_mm, y_tile_mm - 0.7, f"{x0_mm:+.2f}",
                 ha="center", va="bottom", fontsize=8, color="0.35")

    # Virtual FOV bracket
    y_br = -y_lim + 1.0
    axL.annotate("", xy=(px_to_mm(virt_R), y_br),
                 xytext=(px_to_mm(-virt_R), y_br),
                 arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.6))
    axL.text(0, y_br + 0.6,
             f"virtual FOV via 360° = {px_to_mm(2*virt_R):.2f} mm",
             ha="center", color="tab:red", fontsize=10)

    # --------- MIDDLE: vertical z-scan (side view, mm) ----------
    axM.set_aspect("equal")
    det_w_mm = px_to_mm(DET_W)
    det_h_mm = px_to_mm(DET_H)
    z_step_mm = px_to_mm(DET_H - Z_OVERLAP)
    overshoot_mm = px_to_mm(z_overshoot)
    z_x_lo = -SAMPLE_D_MM / 2 - 1.0
    z_x_hi =  SAMPLE_D_MM / 2 + 5.5  # room for shift arrows/labels on the right
    z_z_lo = -overshoot_mm - 2.5
    z_z_hi =  SAMPLE_H_MM + overshoot_mm + 2.5
    axM.set_xlim(z_x_lo, z_x_hi)
    axM.set_ylim(z_z_hi, z_z_lo)                 # z-axis increasing downward
    axM.set_xlabel("x [mm]")
    axM.set_ylabel("z [mm]  (vertical)")
    axM.set_title(
        f"z-mosaic (side view) — {n_z} vertical positions\n"
        f"detector {det_h_mm:.2f} mm tall, uniform z-overlap {Z_OVERLAP} px "
        f"= {px_to_mm(Z_OVERLAP)*1e3:.0f} µm\n"
        f"symmetric overshoot {z_overshoot:.0f} px "
        f"= {overshoot_mm*1e3:.0f} µm at z0 and z{n_z-1}"
    )

    # Sample cylinder projected on xz plane (rectangle SAMPLE_D_MM × SAMPLE_H_MM)
    axM.add_patch(Rectangle((-SAMPLE_D_MM / 2, 0),
                             SAMPLE_D_MM, SAMPLE_H_MM,
                             facecolor="tab:red", alpha=0.08,
                             edgecolor="tab:red", lw=1.6))
    axM.text(0, -0.6, f"sample cylinder ⌀{SAMPLE_D_MM:.2f} × {SAMPLE_H_MM:.2f} mm",
             ha="center", color="tab:red", fontsize=9)

    # Rotation axis
    axM.axvline(0, color="black", lw=1.2)

    # 10 z-stack detector rectangles at x = tile 0 position
    x0_mm = px_to_mm(origins[0])
    zcolors = plt.cm.viridis(np.linspace(0.05, 0.9, n_z))
    for i, z0 in enumerate(z_positions):
        z0_mm = px_to_mm(z0)
        axM.add_patch(Rectangle((x0_mm, z0_mm), det_w_mm, det_h_mm,
                                facecolor=zcolors[i], alpha=0.55,
                                edgecolor=zcolors[i], lw=1.0))
        axM.text(x0_mm + det_w_mm / 2, z0_mm + det_h_mm / 2, f"z{i}",
                 ha="center", va="center", fontsize=9, color="white",
                 fontweight="bold")

    # z-overlap bands
    for i in range(n_z - 1):
        z_b = px_to_mm(z_positions[i + 1])
        band_h = px_to_mm(z_positions[i] + DET_H) - z_b
        axM.add_patch(Rectangle((x0_mm, z_b), det_w_mm, band_h,
                                facecolor="0.1", alpha=0.5, edgecolor="none"))

    # z-shift arrows on the right, labeled in mm
    x_arrow = x0_mm + det_w_mm + 0.8
    for i in range(n_z - 1):
        z_from = px_to_mm(z_positions[i] + DET_H / 2)
        z_to   = px_to_mm(z_positions[i + 1] + DET_H / 2)
        axM.annotate("", xy=(x_arrow, z_to), xytext=(x_arrow, z_from),
                     arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
    # single mm label (uniform step, except last which is clamped)
    axM.text(x_arrow + 0.3,
             (px_to_mm(z_positions[0] + DET_H / 2) +
              px_to_mm(z_positions[1] + DET_H / 2)) / 2,
             f"z-shift {z_step_mm:.2f} mm\n({DET_H - Z_OVERLAP} px)",
             ha="left", va="center", color="0.3", fontsize=8)

    # Note about x-mosaic repetition per z
    axM.text(z_x_lo + 0.3, z_z_hi - 0.5,
             f"× {n_tiles} x-tiles\nper z-position",
             ha="left", va="bottom", color="0.35", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="0.7", pad=0.3))

    # --------- RIGHT: virtual radial coverage (mm) ----------
    axR.set_aspect("equal")
    lim_mm = px_to_mm(virt_R) * 1.15
    axR.set_xlim(-lim_mm, lim_mm)
    axR.set_ylim(-lim_mm, lim_mm)
    axR.set_xlabel("x - x_axis [mm]")
    axR.set_ylabel("y - y_axis [mm]")
    axR.set_title("Virtual sample coverage (top-down, 360° scan)\n"
                  "tile 0 → central disk, tiles 1..N → concentric annuli")

    for i, (ri, ro, col) in enumerate(zip(r_in, r_out, tile_colors)):
        w = Wedge((0, 0), px_to_mm(ro), 0, 360,
                  width=px_to_mm(ro - ri),
                  facecolor=col, alpha=0.30, edgecolor=col, lw=1.0)
        axR.add_patch(w)
        r_mid = px_to_mm((ri + ro) / 2)
        axR.text(r_mid, -0.4, f"{i}", ha="center", va="top",
                 color=col, fontsize=13, fontweight="bold")

    # Sample cylinder cross-section
    axR.add_patch(Circle((0, 0), SAMPLE_D_MM / 2,
                         fill=False, edgecolor="tab:red", lw=2.0))
    axR.text(0, SAMPLE_D_MM / 2 + 0.5, f"sample ⌀{SAMPLE_D_MM:.2f} mm",
             ha="center", color="tab:red")

    # Angle spokes (capped at MAX_DRAWN_SPOKES so the figure stays legible
    # when NTHETA is in the thousands).
    n_drawn = min(NTHETA, MAX_DRAWN_SPOKES)
    thetas  = np.linspace(0, np.pi * ANG_MAX / 180, n_drawn, endpoint=False)
    for th in thetas:
        axR.plot([0, px_to_mm(virt_R) * np.cos(th)],
                 [0, px_to_mm(virt_R) * np.sin(th)],
                 color="0.4", lw=0.4, alpha=0.6)

    # 360° arc arrow
    arc_r = px_to_mm(virt_R) * 1.05
    arc_th = np.linspace(np.deg2rad(15), np.deg2rad(345), 200)
    axR.plot(arc_r * np.cos(arc_th), arc_r * np.sin(arc_th),
             color="black", lw=1.0)
    axR.add_patch(FancyArrowPatch(
        (arc_r * np.cos(np.deg2rad(345)), arc_r * np.sin(np.deg2rad(345))),
        (arc_r * np.cos(np.deg2rad(348)), arc_r * np.sin(np.deg2rad(348))),
        arrowstyle="->", mutation_scale=16, color="black"))
    axR.text(0, arc_r + 0.5,
             f"{ANG_MAX}° / {NTHETA} angles"
             + (f" ({n_drawn} drawn)" if n_drawn < NTHETA else ""),
             ha="center", fontsize=10)

    axR.plot(0, 0, "+", color="black", ms=16, mew=2.4)

    # Parameter box
    txt = (f"pixel = voxel: {PIXEL_UM} µm\n"
           f"detector     : {DET_W}x{DET_H} px\n"
           f"             = {px_to_mm(DET_W):.2f}x{px_to_mm(DET_H):.2f} mm\n"
           f"sample       : ⌀{SAMPLE_D_MM:.2f} × {SAMPLE_H_MM:.2f} mm cyl.\n"
           f"x-overlap    : {OVERLAP} px ({px_to_mm(OVERLAP)*1e3:.0f} µm)\n"
           f"x-shift step : {step_mm:.2f} mm ({DET_W-OVERLAP} px)\n"
           f"axis inset   : {AXIS_INSET} px\n"
           f"x tiles      : {n_tiles} (0 + {n_tiles-1} shifts)\n"
           f"z stacks     : {n_z} "
           f"(step {px_to_mm(DET_H-Z_OVERLAP):.2f} mm)\n"
           f"z overshoot  : ±{overshoot_mm*1e3:.0f} µm\n"
           f"angles       : {NTHETA} × {ANG_MAX}°\n"
           f"virtual FOV  : {px_to_mm(2*virt_R):.2f} mm\n"
           f"recon volume : {rec_nx}×{rec_nx}×{rec_nz} vx @ {VOXEL_UM} µm\n"
           f"             = {rec_nx*rec_nx*rec_nz*4/1e12:.2f} TB f32\n"
           f"total proj.  : {NTHETA}×{n_tiles}×{n_z} = {total_proj}")
    axR.text(-lim_mm * 0.98, -lim_mm * 0.98, txt, family="monospace",
             fontsize=9, va="bottom", ha="left",
             bbox=dict(boxstyle="round", facecolor="white",
                       edgecolor="0.7", pad=0.4))

    plt.tight_layout()
    out_png = os.path.join(BASE, f"mosaic_schematic{UPS}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")
    print(f"saved {positions_path}")


if __name__ == "__main__":
    main()
