#!/usr/bin/env python
"""Slice a single-file `data.h5` (from step2_model_*.py) into per-tile HDF5
files using the mosaic layout from step0_schematic.py.

Reads {path}/model_big{UPS}x/data.h5 which has
    /exchange/data   (NTHETA, NZ, N) float32
    /exchange/theta  (NTHETA,)       float32   angles in DEGREES

For every (z_tile, x_tile) mosaic position, writes one h5:
    {path}/mosaic_h5/{z_idx}_{x_idx}.h5
        /exchange/data   (NTHETA, DET_H, DET_W) float32
        /exchange/theta  (NTHETA,)              float32

Coordinate mapping (same as before):
    schematic z=0 (sample top)     → data row  Z_PAD
    schematic x=0 (rotation axis)  → data col  N/2
Out-of-projection regions are filled with air (intensity = 1.0).

Multi-rank via MPI: tiles are round-robin sharded across ranks so many
tiles can be extracted in parallel.
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from step0_schematic import (
    DET_W, DET_H,
    SAMPLE_D_PX, SAMPLE_H_PX,
    ANG_MAX,
    compute_x_layout, compute_z_stack,
)

import time
from utils import RANK, SIZE, barrier as _barrier, rprint, report_stage
from iohdf5.h5_vchunks import describe_input


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="matches step2_model_*.py --ups (drives src path)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/model_big{UPS}x/data.h5, "
                        "writes {path}/mosaic_h5/*.h5")
    p.add_argument("--z-pad", type=int, default=50,
                   help="projection air-padding (matches upsample_extract --z-pad)")
    p.add_argument("--air-fill", type=float, default=1.0,
                   help="OOB tile pixel fill (transmission intensity of air)")
    return p.parse_args()


def _fill_tile(src_plane: np.ndarray, row0: int, col0: int,
               h: int, w: int, fill: float) -> np.ndarray:
    """(h, w) crop of src_plane starting at (row0, col0); OOB pixels get `fill`."""
    out = np.full((h, w), fill, dtype=np.float32)
    r_lo = max(0, -row0)
    r_hi = min(h, src_plane.shape[0] - row0)
    c_lo = max(0, -col0)
    c_hi = min(w, src_plane.shape[1] - col0)
    if r_lo < r_hi and c_lo < c_hi:
        out[r_lo:r_hi, c_lo:c_hi] = src_plane[row0 + r_lo : row0 + r_hi,
                                              col0 + c_lo : col0 + c_hi]
    return out


def main() -> None:
    args = _parse_args()

    src_h5   = f"{args.path}/model_big{args.ups}x/data.h5"
    dst_dir  = f"{args.path}/mosaic_h5"
    if RANK == 0:
        os.makedirs(dst_dir, exist_ok=True)
    _barrier()

    if not os.path.exists(src_h5):
        raise SystemExit(f"missing input: {src_h5}")

    # Peek at data.h5 to get shape + theta.
    with h5py.File(src_h5, "r") as f:
        dset  = f["exchange/data"]
        NTHETA, NZ, N = dset.shape
        theta_deg = f["exchange/theta"][:]

    # Tile layout.
    _, x_origins, _, _ = compute_x_layout(SAMPLE_D_PX / 2.0)
    z_positions, _     = compute_z_stack(float(SAMPLE_H_PX))
    n_x = len(x_origins)
    n_z = len(z_positions)

    z_starts = [int(round(z + args.z_pad)) for z in z_positions]
    x_starts = [int(round(x + N / 2))      for x in x_origins]

    if RANK == 0:
        describe_input(src_h5)
        per_tile = NTHETA * DET_H * DET_W * 4
        print(f"  OUT: {dst_dir}/{{z}}_{{x}}.h5   (per-tile, plain HDF5)")
        print(f"       shape=({NTHETA}, {DET_H}, {DET_W}) float32   "
              f"HDF5 chunk=(1, {DET_H}, {DET_W})")
        print(f"       {n_z} z-tiles × {n_x} x-tiles = {n_z*n_x} files   "
              f"({per_tile/1e9:.2f} GB/tile, "
              f"{per_tile*n_z*n_x/1e12:.2f} TB total)   MPI ranks={SIZE}",
              flush=True)

    # Enumerate tiles and round-robin across ranks.
    tiles = [(zi, xi) for zi in range(n_z) for xi in range(n_x)]
    my_tiles = tiles[RANK::SIZE]

    t_read = t_write = 0.0
    b_read = b_write = 0

    with h5py.File(src_h5, "r") as fsrc:
        data_dset = fsrc["exchange/data"]

        for zi, xi in my_tiles:
            r0, c0 = z_starts[zi], x_starts[xi]
            tile = np.empty((NTHETA, DET_H, DET_W), dtype=np.float32)

            # Efficient read: pull the relevant z-strip of each plane once,
            # then crop columns.  Falls back to per-plane whole reads if the
            # crop extends outside the volume in z (air fill happens then).
            rz_lo = max(0, r0)
            rz_hi = min(NZ, r0 + DET_H)
            cc_lo = max(0, c0)
            cc_hi = min(N,  c0 + DET_W)
            t0 = time.perf_counter()
            if rz_lo < rz_hi and cc_lo < cc_hi:
                strip = data_dset[:, rz_lo:rz_hi, cc_lo:cc_hi]     # (NTHETA, h', w')
                b_read += strip.nbytes
            else:
                strip = None
            t_read += time.perf_counter() - t0

            out_r_lo = max(0, -r0)
            out_c_lo = max(0, -c0)
            # Fill with air, then place strip content.
            tile[:] = args.air_fill
            if strip is not None:
                tile[:, out_r_lo:out_r_lo + strip.shape[1],
                        out_c_lo:out_c_lo + strip.shape[2]] = strip

            path = os.path.join(dst_dir, f"{zi}_{xi}.h5")
            t0 = time.perf_counter()
            with h5py.File(path, "w") as fout:
                g = fout.create_group("exchange")
                d = g.create_dataset("data", data=tile,
                                     chunks=(1, DET_H, DET_W))
                g.create_dataset("theta", data=theta_deg)   # DEGREES
                # Synthetic flat/dark fields for downstream reconstruction:
                # data_white = ones (perfect flat), data_dark = zeros (no noise).
                # One image each, matching the standard tomography convention.
                g.create_dataset(
                    "data_white",
                    data=np.ones((1, DET_H, DET_W), dtype=np.float32),
                )
                g.create_dataset(
                    "data_dark",
                    data=np.zeros((1, DET_H, DET_W), dtype=np.float32),
                )
                d.attrs["z_tile"]  = zi
                d.attrs["x_tile"]  = xi
                d.attrs["z_start"] = r0
                d.attrs["x_start"] = c0
                d.attrs["det_h"]   = DET_H
                d.attrs["det_w"]   = DET_W
            t_write += time.perf_counter() - t0
            b_write += tile.nbytes
            print(f"  [rank {RANK}] {path}  z_start={r0}  x_start={c0}",
                  flush=True)

    _barrier()
    report_stage("step4 read (data)",   b_read,  t_read)
    report_stage("step4 write (tiles)", b_write, t_write)
    rprint(f"done. wrote {n_z*n_x} h5 files.")


if __name__ == "__main__":
    from utils import run_main
    run_main(main)
