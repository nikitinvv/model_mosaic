#!/usr/bin/env python
"""Compute Radon projections R(delta) of the upsampled init volume.

Writes proj.h5 (VDS + banks; vchunks pattern from test_h5_buffer_io.py):

    {path}/big{UPS}x.h5                        VDS master (input)
    {path}/model_big{UPS}x/proj.h5             VDS master (output)
    {path}/model_big{UPS}x/proj/proj_data_*.h5 bank files

For each z-super-chunk (--vchunks C1) this rank owns, loop NZCHUNK-sized
Radon calls to fill a shared-memory buffer, then tomo_writex fans it across
--nbanks POSIX writers.  Uses the GPU-only TomoReal class (whole (nz, 2N, N+1)
rfft-half frequency-domain buffer lives on the GPU) — for very large N use
step2_radon_large.py which host-chunks.

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step2_radon.py \\
        --ups 2 --path /data2/brain_sym_mosaic

Uses TomoReal — rfft/float32 path.  The obj is REAL (imag=0 in the
old complex64 pipeline was wasted memory + wasted FFT bandwidth), so
we send float32 to the GPU and receive float32 sinograms back with
zero conversion at either end.
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.tomo import TomoReal
from iohdf5.dxchange_hdf5_chunks import tomo_writex, read_projs_vchunkx
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
)
from iohdf5.layout import add_layout_args, resolve_step
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="upsample factor (matches step1_upsample --ups)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/big{UPS}x.h5, writes {path}/model_big{UPS}x/proj.h5")
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 180°; default = 3·N/4")
    p.add_argument("--nzchunk", type=int, default=8,
                   help="z-slices per Radon call")
    p.add_argument("--nbanks",  type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--ntasks",  type=int, default=8,
                   help="parallel workers for read_projs_vchunkx (big-vol prefetch)")
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for proj.h5; default comes from "
                        "iohdf5.layout (--mem-budget / --chunk-bytes)")
    add_layout_args(p)
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
SRC_H5   = f"{BASE_DIR}/big{UPS}x.h5"
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
PROJ_H5  = f"{DST_DIR}/proj.h5"

IN_NZ = IN_N = 3072            # init.h5 dims after step00; UPS scales from here
NZ    = IN_NZ * UPS
N     = IN_N  * UPS
NTHETA  = _A.ntheta if _A.ntheta is not None else 3 * N // 4
ANG_MAX = np.pi          # tomo needs 180°; step4 synthesises the 360° tile-scan
                         # via the tomo identity proj(θ,x) = proj(θ+π, N-1-x)
NTASKS  = _A.ntasks

# Layout from the shared byte-budget policy.  The old default,
# (NTHETA, 8·NZCHUNK, N), is 2.3 GB at UPS=1 but 9.7 TB at UPS=8: it counts
# z-slices when what has to stay bounded is bytes, and one z-sinogram alone
# is 7.2 GB at UPS=16.  NZCHUNK follows the plan's alignment -- the Radon
# loop can run in smaller pieces, so it bends rather than pushing the
# super-chunk over the RAM budget.
_PLAN   = resolve_step("proj", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                       ntheta=NTHETA, nbanks=_A.nbanks,
                       mem_budget_gb=_A.mem_budget, chunk_mb=_A.chunk_bytes,
                       nzchunk=_A.nzchunk, vchunks=_A.vchunks, nranks=SIZE)
NBANKS   = _PLAN.nbanks
VCHUNKS  = _PLAN.vchunks
H5CHUNKS = _PLAN.chunks
NZCHUNK  = _PLAN.align


def load_chunk(src_dset, z_start: int, z_end: int) -> np.ndarray:
    """(k, N, N) float32 host read from big{UPS}x.h5's /exchange/data
    (VDS from step1_upsample; plain h5py handles cross-bank routing)."""
    return src_dset[z_start:z_end, :, :].astype(np.float32, copy=False)


def main() -> None:
    if VCHUNKS[1] % NZCHUNK != 0:
        raise SystemExit(
            f"--vchunks C1={VCHUNKS[1]} must be a multiple of "
            f"--nzchunk={NZCHUNK}.")
    if VCHUNKS[0] != NTHETA:
        raise SystemExit(
            f"--vchunks C0={VCHUNKS[0]} must equal NTHETA={NTHETA} "
            f"(single-θ-batch mode).")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    theta_rad = np.linspace(0.0, ANG_MAX, NTHETA, endpoint=False).astype("float32")
    theta_deg = np.rad2deg(theta_rad).astype("float32")

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()
    rprint(f"UPS={UPS}  nz={NZ} n={N} ntheta={NTHETA} nzchunk={NZCHUNK}")
    # TomoReal keeps the padded (2N × 2N) buffer as REAL float32 and the
    # rfft output as (2N × (N+1)) complex64 — half the memory of a
    # full-complex64 path.
    rprint(f"GPU est. — TomoReal padded (f32): "
           f"{NZCHUNK * (2*N)**2 * 4 / 1e9:.1f} GB  "
           f"+ rfft fde (c64): {NZCHUNK * (2*N) * (N+1) * 8 / 1e9:.1f} GB")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(PROJ_H5, (NTHETA, NZ, N), np.float32,
                        VCHUNKS, "slice", NBANKS, chunks=H5CHUNKS,
                        read_granule=_PLAN.read_granule,
                        companion_bytes=VCHUNKS[1] * N * N * 4)

    ctx = initx_and_bcast(PROJ_H5, shape=(NTHETA, NZ, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="slice", nbanks=NBANKS, chunks=H5CHUNKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(PROJ_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((NTHETA, NZ, N), VCHUNKS)}")

    proj_min, proj_max = np.inf, -np.inf

    rprint("building TomoReal (allocating buffers + cuFFT plans)...")
    cl_tomo = TomoReal(N, NZCHUNK, theta_rad)
    rprint("TomoReal ready.")

    ivchunks = list(iter_vchunks((NTHETA, NZ, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)

    # Prefetch shm for the vchunkx z-slab from big{UPS}x.h5 (aligned along
    # its axis 0 = z).  One read_projs_vchunkx per output vchunk with
    # NTASKS parallel workers; inner NZCHUNK loop slices from RAM.
    big_slab_shape = (VCHUNKS[1], N, N)
    shm_slab, big_slab_buf = alloc_shm(big_slab_shape, np.float32)

    t_read = t_radon = t_write = 0.0
    b_read = b_write = 0
    try:
        for k, ivc in enumerate(my_ivchunks, start=1):
            z0_vc = ivc[1] * VCHUNKS[1]
            z1_vc = min(z0_vc + VCHUNKS[1], NZ)
            buf.fill(0)

            t0 = time.perf_counter()
            read_projs_vchunkx(SRC_H5, shm_slab, ntasks=NTASKS,
                               vchunksx=big_slab_shape,
                               ivchunkx=(ivc[1], 0, 0))
            t_read += time.perf_counter() - t0
            b_read += (z1_vc - z0_vc) * N * N * 4

            for z0 in range(z0_vc, z1_vc, NZCHUNK):
                z1 = min(z0 + NZCHUNK, NZ)
                kz = z1 - z0

                # Slice from the pre-fetched RAM buffer — free.
                chunk_h = big_slab_buf[z0 - z0_vc : z0 - z0_vc + kz]
                if kz < NZCHUNK:
                    pad = np.zeros((NZCHUNK, N, N), dtype=np.float32)
                    pad[:kz] = chunk_h
                    chunk_h = pad

                t0 = time.perf_counter()
                # TomoReal takes float32 obj directly (no complex64
                # wrapping) and returns float32 sino (no .real / astype
                # dance).
                vol_d = cp.asarray(chunk_h)
                proj_d = cl_tomo.R(vol_d)      # (NTHETA, NZCHUNK, N) f32
                del vol_d

                proj_chunk_h = cp.asnumpy(proj_d[:, :kz])
                del proj_d
                cp.get_default_memory_pool().free_all_blocks()
                t_radon += time.perf_counter() - t0

                buf[:, z0 - z0_vc : z1 - z0_vc, :] = proj_chunk_h
                proj_min = min(proj_min, float(proj_chunk_h.min()))
                proj_max = max(proj_max, float(proj_chunk_h.max()))
                del proj_chunk_h

            t0 = time.perf_counter()
            tomo_writex(PROJ_H5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
            t_write += time.perf_counter() - t0
            b_write += NTHETA * (z1_vc - z0_vc) * N * 4

            print(f"  [rank {RANK}] vchunk {k}/{len(my_ivchunks)}  "
                  f"z=[{z0_vc},{z1_vc})  "
                  f"(read={t_read:.1f}s radon={t_radon:.1f}s "
                  f"write={t_write:.1f}s)", flush=True)
    finally:
        free_shm(shm)
        free_shm(shm_slab)
        del cl_tomo
        cp.get_default_memory_pool().free_all_blocks()
    barrier()

    proj_min = allreduce(proj_min, MPI.MIN)
    proj_max = allreduce(proj_max, MPI.MAX)
    barrier()

    report_stage("step2 read (big)",  b_read,  t_read)
    report_stage("step2 write (proj)", b_write, t_write)

    norm_const = float(np.sqrt(N / NTHETA))
    rprint(f"proj = R(delta) stats: min={proj_min:.4g} max={proj_max:.4g}  "
           f"(after scaling by 1/NORM_CONST={1.0/norm_const:.4g}: "
           f"phase in [{proj_min/norm_const:.4g}, "
           f"{proj_max/norm_const:.4g}] rad)")


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
