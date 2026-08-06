#!/usr/bin/env python
"""Simulate mosaic-scan h5 files from single-tile projection tifs.

Reads NTHETA `data_XXXXX.tif` projections from {path}/model_big{UPS}x
(produced by model_radon_big.py / model_radon_large.py) and slices them
into per-tile h5 files using the tile layout from mosaic_schematic.py.

For every (z_tile, x_tile) mosaic position:
  {path}/mosaic_h5/{z_idx}_{x_idx}.h5
    /exchange/data   (NTHETA, DET_H, DET_W) float32   cropped intensity
    /exchange/theta  (NTHETA,)              float32   angles in DEGREES

Coordinate mapping:
  schematic z=0 (sample top)     → projection row  Z_PAD
  schematic x=0 (rotation axis)  → projection col  N/2
Out-of-projection regions get filled with air (intensity = 1.0).
"""
from __future__ import annotations

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import tifffile

from step0_schematic import (
    DET_W, DET_H,
    SAMPLE_D_PX, SAMPLE_H_PX,
    ANG_MAX,
    compute_x_layout, compute_z_stack,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="matches model_radon_big.py --ups (drives SRC path)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/model_big{UPS}x, writes {path}/mosaic_h5")
    p.add_argument("--z-pad", type=int, default=50,
                   help="projection air-padding (matches upsample_extract --z-pad)")
    p.add_argument("--n-read", type=int, default=8,
                   help="parallel disk-read threads")
    p.add_argument("--air-fill", type=float, default=1.0,
                   help="OOB tile pixel fill (transmission intensity of air)")
    return p.parse_args()


def _fill_tile(src: np.ndarray, row0: int, col0: int,
               h: int, w: int, fill: float) -> np.ndarray:
    """(h, w) crop of src starting at (row0, col0); OOB pixels get `fill`."""
    out = np.full((h, w), fill, dtype=np.float32)
    r_lo = max(0, -row0)
    r_hi = min(h, src.shape[0] - row0)
    c_lo = max(0, -col0)
    c_hi = min(w, src.shape[1] - col0)
    if r_lo < r_hi and c_lo < c_hi:
        out[r_lo:r_hi, c_lo:c_hi] = src[row0 + r_lo : row0 + r_hi,
                                        col0 + c_lo : col0 + c_hi]
    return out


def main() -> None:
    args = _parse_args()

    src_dir  = f"{args.path}/model_big{args.ups}x"
    dst_dir  = f"{args.path}/mosaic_h5"
    os.makedirs(dst_dir, exist_ok=True)

    proj_paths = sorted(glob.glob(os.path.join(src_dir, "data_*.tif")))
    if not proj_paths:
        raise SystemExit(f"no data_*.tif found in {src_dir}")
    ntheta = len(proj_paths)

    # Infer projection shape from the first file.
    proj0 = tifffile.imread(proj_paths[0])
    if proj0.ndim != 2:
        raise SystemExit(f"expected 2-D projection, got shape {proj0.shape}")
    NZ, N = proj0.shape

    # Tile layout from the schematic (positions in schematic coords).
    _, x_origins, _, _ = compute_x_layout(SAMPLE_D_PX / 2.0)
    z_positions, _     = compute_z_stack(float(SAMPLE_H_PX))
    n_x = len(x_origins)
    n_z = len(z_positions)

    # Sub-pixel schematic positions → nearest integer projection index.
    z_starts = [int(round(z + args.z_pad)) for z in z_positions]
    x_starts = [int(round(x + N / 2))      for x in x_origins]

    theta_deg = np.linspace(0.0, ANG_MAX, ntheta, endpoint=False).astype("float32")

    print(f"src : {src_dir}  ({ntheta} projections, {NZ}×{N})")
    print(f"dst : {dst_dir}")
    print(f"tiles: {n_z} z-positions × {n_x} x-positions  "
          f"({n_z*n_x} h5 files)")
    print(f"det  : {DET_H} × {DET_W} px (h × w)")

    # Load every projection once (host peak ≈ ntheta·NZ·N·4 B).
    projs = np.empty((ntheta, NZ, N), dtype=np.float32)

    def _load(i: int) -> None:
        im = tifffile.imread(proj_paths[i])
        projs[i] = im if im.dtype == np.float32 else im.astype(np.float32, copy=False)

    with ThreadPoolExecutor(max_workers=args.n_read) as pool:
        list(pool.map(_load, range(ntheta)))

    # Emit one h5 per tile.
    for zi in range(n_z):
        for xi in range(n_x):
            r0, c0 = z_starts[zi], x_starts[xi]
            tile = np.empty((ntheta, DET_H, DET_W), dtype=np.float32)
            for ti in range(ntheta):
                tile[ti] = _fill_tile(projs[ti], r0, c0, DET_H, DET_W, args.air_fill)

            path = os.path.join(dst_dir, f"{zi}_{xi}.h5")
            with h5py.File(path, "w") as f:
                g = f.create_group("exchange")
                d = g.create_dataset("data", data=tile)
                g.create_dataset("theta", data=theta_deg)  # DEGREES
                d.attrs["z_tile"]  = zi
                d.attrs["x_tile"]  = xi
                d.attrs["z_start"] = r0
                d.attrs["x_start"] = c0
                d.attrs["det_h"]   = DET_H
                d.attrs["det_w"]   = DET_W
            print(f"  {path}  z_start={r0}  x_start={c0}")

    print(f"done. wrote {n_z*n_x} h5 files.")


if __name__ == "__main__":
    main()
