#!/usr/bin/env python
"""Compute Radon projections R(delta) of the upsampled init volume.

Writes proj.h5 (VDS + banks; vchunks pattern from test_h5_buffer_io.py):

    {path}/big{UPS}x.h5                        VDS master (input)
    {path}/model_big{UPS}x/proj.h5             VDS master (output)
    {path}/model_big{UPS}x/proj/proj_data_*.h5 bank files

For each z-super-chunk (--proj-vchunks C1) this rank owns, loop NZCHUNK-sized
Radon calls to fill a shared-memory buffer, then tomo_writex fans it across
--nbanks POSIX writers.  Uses the GPU-only Tomo class (whole (nz, 2N, 2N)
frequency-domain buffer lives on the GPU) — for very large N use
step2_radon_large.py which host-chunks.

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step2_radon.py \\
        --ups 2 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.tomo import Tomo
from iohdf5.dxchange_hdf5_chunks import tomo_writex
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
)
from utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="upsample factor (matches step1_upsample --ups)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/big{UPS}x.h5, writes {path}/model_big{UPS}x/proj.h5")
    p.add_argument("--in-nz",  type=int, default=2560, help="init nz (before UPS)")
    p.add_argument("--in-n",   type=int, default=2744, help="init N  (before UPS)")
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 360°; default = 3·N/4")
    p.add_argument("--mask-r", type=float, default=0.0,
                   help="soft circular mask radius (0 disables)")
    p.add_argument("--nzchunk", type=int, default=8,
                   help="z-slices per Radon call")
    p.add_argument("--nbanks",  type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--proj-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for proj.h5 (default: NTHETA, "
                        "8·NZCHUNK, N).  RAM buffer = C0·C1·C2·4 bytes/rank.")
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
SRC_H5   = f"{BASE_DIR}/big{UPS}x.h5"
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
PROJ_H5  = f"{DST_DIR}/proj.h5"

NZ      = _A.in_nz * UPS
N       = _A.in_n  * UPS
NTHETA  = _A.ntheta if _A.ntheta is not None else 3 * N // 4
ANG_MAX = 2 * np.pi
MASK_R  = _A.mask_r
NZCHUNK = _A.nzchunk
NBANKS  = _A.nbanks
PROJ_VCHUNKS = tuple(_A.proj_vchunks) if _A.proj_vchunks else (NTHETA, 8 * NZCHUNK, N)


def load_chunk(src_dset, z_start: int, z_end: int) -> np.ndarray:
    """(k, N, N) float32 host read from big{UPS}x.h5's /exchange/data
    (VDS from step1_upsample; plain h5py handles cross-bank routing)."""
    return src_dset[z_start:z_end, :, :].astype(np.float32, copy=False)


def main() -> None:
    if PROJ_VCHUNKS[1] % NZCHUNK != 0:
        raise SystemExit(
            f"--proj-vchunks C1={PROJ_VCHUNKS[1]} must be a multiple of "
            f"--nzchunk={NZCHUNK}.")
    if PROJ_VCHUNKS[0] != NTHETA:
        raise SystemExit(
            f"--proj-vchunks C0={PROJ_VCHUNKS[0]} must equal NTHETA={NTHETA} "
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
    rprint(f"UPS={UPS}  nz={NZ} n={N} ntheta={NTHETA} nzchunk={NZCHUNK}  "
           f"mask_r={MASK_R}")
    rprint(f"GPU est. — Tomo._buf_fde: "
           f"{NZCHUNK * (2*N)**2 * 8 / 1e9:.1f} GB")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(PROJ_H5, (NTHETA, NZ, N), np.float32,
                        PROJ_VCHUNKS, "slice", NBANKS)

    ctx = initx_and_bcast(PROJ_H5, shape=(NTHETA, NZ, N),
                          dtype=np.float32, vchunks=PROJ_VCHUNKS,
                          stype="slice", nbanks=NBANKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(PROJ_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    buf_gb = vchunk_bytes(PROJ_VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((NTHETA, NZ, N), PROJ_VCHUNKS)}")

    proj_min, proj_max = np.inf, -np.inf

    rprint("building Tomo (allocating buffers + cuFFT plans)...")
    cl_tomo = Tomo(N, NZCHUNK, theta_rad, mask_r=MASK_R)
    rprint("Tomo ready.")

    ivchunks = list(iter_vchunks((NTHETA, NZ, N), PROJ_VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(PROJ_VCHUNKS, np.float32)

    try:
        with h5py.File(SRC_H5, "r") as fsrc:
            src_dset = fsrc["exchange/data"]
            t_read = t_radon = t_write = 0.0
            b_read = b_write = 0
            for k, ivc in enumerate(my_ivchunks, start=1):
                z0_vc = ivc[1] * PROJ_VCHUNKS[1]
                z1_vc = min(z0_vc + PROJ_VCHUNKS[1], NZ)
                buf.fill(0)

                for z0 in range(z0_vc, z1_vc, NZCHUNK):
                    z1 = min(z0 + NZCHUNK, NZ)
                    kz = z1 - z0

                    t0 = time.perf_counter()
                    chunk_h = load_chunk(src_dset, z0, z1)
                    if kz < NZCHUNK:
                        pad = np.zeros((NZCHUNK, N, N), dtype=np.float32)
                        pad[:kz] = chunk_h
                        chunk_h = pad
                    t_read += time.perf_counter() - t0
                    b_read += kz * N * N * 4

                    t0 = time.perf_counter()
                    delta_d = cp.asarray(chunk_h)
                    vol_d   = cp.empty(delta_d.shape, dtype=cp.complex64)
                    vol_d.real = delta_d
                    vol_d.imag = cp.float32(0)
                    del delta_d

                    proj_d_c = cl_tomo.R(vol_d)   # (NTHETA, NZCHUNK, N)
                    del vol_d

                    proj_chunk_h = cp.asnumpy(proj_d_c[:, :kz].real).astype(
                        np.float32, copy=False)
                    del proj_d_c
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
    from utils import run_main
    run_main(main)
