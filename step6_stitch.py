#!/usr/bin/env python
"""Stitch the per-tile HDF5 files from step5_correct.py back into a single
180° big projection (NTHETA/2 angles) matching the (NZ, N) format of
step3_propagation.py's data.h5.

Reads
    {path}/mosaic_h5_pre/{zi}_{xi}.h5             (one per tile, 360°;
                                                   corrected tiles from step5)
    mosaic_modeling/mosaic_positions/mosaic_positions{UPS}.txt
Writes
    {path}/model_big{UPS}x/stitched.h5            (VDS master + banks)
        /exchange/data   (N_HALF, NZ, N) float32
        /exchange/theta  (N_HALF,)       float32   angles in DEGREES,
                                                   the first half of the
                                                   tile files' theta.

Mirror-fold — for each output angle θ in [0, N_HALF):
  * place tile[θ]             at (z_center, x_center)              (no flip)
  * place tile[θ + N_HALF][:, :, ::-1] at (z_center, N - x_center)
The second contribution fills the far side of the rotation axis via
proj(θ, x) = proj(θ+π, N-1-x); tomo reconstruction is 180°-only.

Blending — each placement gets a tent weight of shape (h, w),
    w[i, j] = min(i+0.5, h-0.5-i, cap) * min(j+0.5, w-0.5-j, cap)
with cap = OVERLAP (from the positions header).  This gives an exact
linear ramp inside overlap bands of width `OVERLAP`, and lets the
direct/mirror cross-overlap (typically 2·OVERLAP wide) resolve
smoothly.  Contributions are accumulated additively; a final divide by
sum(weights) yields the blended big projection.  Pixels no placement
covers are filled with 1.0 (air transmission).

Multi-rank via MPI: θ-slab vchunks are round-robin sharded across ranks
(same pattern as step3_propagation.py); tomo_writex fans each rank's vchunk
buffer across --nbanks bank files.
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from iohdf5.dxchange_hdf5_chunks import tomo_writex
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_output,
)
from iohdf5.layout import add_layout_args, resolve_step
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, report_stage

from step4_extract import read_placements


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POSITIONS_DIR = os.path.join(_SCRIPT_DIR, "mosaic_positions")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="matches step3_propagation --ups (drives paths + positions file)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/mosaic_h5_pre/*.h5 (step5 output), "
                        "writes {path}/model_big{UPS}x/stitched.h5")
    p.add_argument("--nbanks", type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--nthetachunk", type=int, default=0,
                   help="θ per super-chunk (=vchunk C0); 0 = take it from "
                        "iohdf5.layout (--mem-budget / --chunk-bytes)")
    add_layout_args(p)
    return p.parse_args()


def _tent_weight(h: int, w: int, cap: int) -> np.ndarray:
    """Separable tent, capped at `cap` on each axis.  Result is
    positive everywhere (>= 0.5*0.5 at corners) so 'sum-of-weights'
    never divides by 0 on covered pixels.  With cap == OVERLAP, the
    ramp is exactly linear over every normal (OVERLAP-wide) overlap
    band and yields a linear transition between two adjacent tiles."""
    yi = np.arange(h, dtype=np.float32) + 0.5
    xi = np.arange(w, dtype=np.float32) + 0.5
    wz = np.minimum(np.minimum(yi, h - yi), float(cap))
    wx = np.minimum(np.minimum(xi, w - xi), float(cap))
    return (wz[:, None] * wx[None, :]).astype(np.float32)


def _theta_range_wraps(start: int, size: int, total: int) -> bool:
    return (start + size) > total


def main() -> None:
    args = _parse_args()

    from mpi_utils import banner
    banner("6", "mosaic_h5_pre/*.h5 -> stitched.h5  (tent-weight blend, mirror-fold to 180 deg)")

    src_dir  = f"{args.path}/mosaic_h5_pre"
    dst_dir  = f"{args.path}/model_big{args.ups}x"
    dst_h5   = f"{dst_dir}/stitched.h5"
    positions_path = os.path.join(_POSITIONS_DIR,
                                  f"mosaic_positions{args.ups}.txt")

    if not os.path.exists(positions_path):
        raise SystemExit(
            f"missing positions file: {positions_path}\n"
            f"run `python step0_schematic.py --ups {args.ups}` first.")
    if not os.path.isdir(src_dir):
        raise SystemExit(f"missing tile dir: {src_dir}")

    meta, placements = read_placements(positions_path)
    NZ, N       = meta["NZ"], meta["N"]
    n_z, n_x    = meta["n_z"], meta["n_x"]
    NTHETA      = meta["NTHETA"]
    N_HALF      = meta["N_HALF"]
    OVERLAP     = meta["OVERLAP"]

    # Pull theta from any tile file.  Stitched output covers the first
    # 180° (first N_HALF entries of theta).
    probe_p = placements[0]
    probe   = os.path.join(src_dir, f"{probe_p['zi']}_{probe_p['xi']}.h5")
    with h5py.File(probe, "r") as f:
        tile_ntheta = f["exchange/data"].shape[0]
        theta_deg   = f["exchange/theta"][:]
    if tile_ntheta != NTHETA:
        raise SystemExit(
            f"tile ntheta ({tile_ntheta}) != positions NTHETA ({NTHETA})")
    theta_out = theta_deg[:N_HALF]

    # Layout from the shared byte-budget policy.  A flat 64 θ is 24 GB at
    # UPS=1 but 1.5 TB at UPS=8, and the default (1, NZ, N) chunk is past
    # HDF5's 4 GiB limit from UPS=16 on.  The planned C0 always divides
    # N_HALF, so the mirror read range can never wrap past NTHETA.
    _plan = resolve_step("stitched", ups=args.ups,
                         in_nz=NZ // args.ups, in_nyx=N // args.ups,
                         ntheta=NTHETA, nbanks=args.nbanks,
                         mem_budget_gb=args.mem_budget,
                         chunk_mb=args.chunk_bytes, nranks=SIZE,
                         vchunks=((args.nthetachunk, NZ, N)
                                  if args.nthetachunk else None))
    VCHUNKS  = _plan.vchunks
    NBANKS   = _plan.nbanks
    H5CHUNKS = _plan.chunks
    if N_HALF % VCHUNKS[0] != 0:
        rprint(f"WARN: N_HALF={N_HALF} not divisible by vchunk C0="
               f"{VCHUNKS[0]}; the final vchunk will be short and the "
               f"mirror read range may wrap around NTHETA.")

    if RANK == 0:
        os.makedirs(dst_dir, exist_ok=True)
    barrier()

    rprint(f"UPS={args.ups}  NTHETA(tiles)={NTHETA}  N_HALF(out)={N_HALF}  "
           f"NZ={NZ}  N={N}  tiles={n_z}x{n_x}={n_z*n_x}  "
           f"OVERLAP={OVERLAP}  ranks={SIZE}")
    if RANK == 0:
        describe_output(dst_h5, (N_HALF, NZ, N), np.float32,
                        VCHUNKS, "proj", NBANKS, chunks=H5CHUNKS,
                        read_granule=_plan.read_granule)

    ctx = initx_and_bcast(dst_h5, shape=(N_HALF, NZ, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="proj", nbanks=NBANKS, chunks=H5CHUNKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(dst_h5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_out)
    barrier()

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((N_HALF, NZ, N), VCHUNKS)}")

    # Precompute per-tile 2-D tent weight (once, all ranks).  Direct and
    # mirror share the same stored shape, so one weight per (zi, xi).
    weights = {}
    for p in placements:
        h = p["r_hi_dir"] - p["r_lo_dir"]
        w = p["c_hi_dir"] - p["c_lo_dir"]
        weights[(p["zi"], p["xi"])] = _tent_weight(h, w, OVERLAP)

    ivchunks = list(iter_vchunks((N_HALF, NZ, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)
    # Weight accumulator is a plane (NZ, N) — one per vchunk — reused.
    wgt_acc = np.zeros((NZ, N), dtype=np.float32)

    t_read = t_write = t_blend = 0.0
    b_read = b_write = 0

    try:
        for k, ivc in enumerate(my_ivchunks, start=1):
            t0_vc = ivc[0] * VCHUNKS[0]
            t1_vc = min(t0_vc + VCHUNKS[0], N_HALF)
            n = t1_vc - t0_vc
            mt0 = t0_vc + N_HALF
            # We rely on nthetachunk dividing N_HALF; assert it doesn't wrap.
            if _theta_range_wraps(mt0, n, NTHETA):
                raise RuntimeError(
                    f"mirror θ range [{mt0},{mt0+n}) wraps NTHETA={NTHETA}; "
                    f"pick --nthetachunk that divides N_HALF={N_HALF}")

            buf.fill(0.0)
            wgt_acc.fill(0.0)

            for p in placements:
                tile_path = os.path.join(src_dir,
                                         f"{p['zi']}_{p['xi']}.h5")
                w2 = weights[(p["zi"], p["xi"])]
                t0 = time.perf_counter()
                with h5py.File(tile_path, "r") as ftile:
                    dset = ftile["exchange/data"]
                    direct_slab = dset[t0_vc:t1_vc, :, :]        # (n, h, w)
                    mirror_slab = dset[mt0:mt0 + n, :, :]        # (n, h, w)
                t_read += time.perf_counter() - t0
                b_read += direct_slab.nbytes + mirror_slab.nbytes

                t0 = time.perf_counter()
                # Direct — as-is at (r_lo_dir, c_lo_dir).
                buf[:n,
                    p["r_lo_dir"]:p["r_hi_dir"],
                    p["c_lo_dir"]:p["c_hi_dir"]] += direct_slab * w2
                wgt_acc[p["r_lo_dir"]:p["r_hi_dir"],
                        p["c_lo_dir"]:p["c_hi_dir"]] += w2
                # Mirror — h-flip and place at (r_lo_mir, c_lo_mir).  The
                # weight is also flipped in x (since crop_left/right swap
                # under the flip), but the tent's x profile is symmetric,
                # so the flipped weight equals the direct weight.
                mirror_flipped = mirror_slab[:, :, ::-1]
                buf[:n,
                    p["r_lo_mir"]:p["r_hi_mir"],
                    p["c_lo_mir"]:p["c_hi_mir"]] += mirror_flipped * w2
                wgt_acc[p["r_lo_mir"]:p["r_hi_mir"],
                        p["c_lo_mir"]:p["c_hi_mir"]] += w2
                t_blend += time.perf_counter() - t0
                del direct_slab, mirror_slab, mirror_flipped

            # Normalize covered pixels; fill uncovered pixels with 1.0
            # (air transmission) so downstream flat-field division sees
            # 1 instead of 0.
            t0 = time.perf_counter()
            covered = wgt_acc > 0
            inv = np.zeros_like(wgt_acc)
            np.divide(1.0, wgt_acc, out=inv, where=covered)
            buf[:n] *= inv[None, :, :]
            if (~covered).any():
                buf[:n, ~covered] = 1.0
            t_blend += time.perf_counter() - t0

            t0 = time.perf_counter()
            tomo_writex(dst_h5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
            t_write += time.perf_counter() - t0
            b_write += n * NZ * N * 4

            print(f"  [rank {RANK}] vchunk {k}/{len(my_ivchunks)}  "
                  f"θ_out=[{t0_vc},{t1_vc})  "
                  f"(read={t_read:.1f}s blend={t_blend:.1f}s "
                  f"write={t_write:.1f}s)", flush=True)
    finally:
        free_shm(shm)

    barrier()
    report_stage("step6 read (tiles)",    b_read,  t_read)
    report_stage("step6 write (stitched)", b_write, t_write)
    rprint(f"wrote {N_HALF} angles to {dst_h5}")


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
