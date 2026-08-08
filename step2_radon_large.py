#!/usr/bin/env python
"""Radon projections — host-chunked variant of step2_radon.py.

Same output format (VDS + banks proj.h5) but uses the host-chunked
TomoLarge implementation from tomo_large.py: the big (nz, 2N, 2N) `fde`
and (nz, ntheta, N) `sino` buffers live on the HOST, and small pieces
are staged through the GPU.  Peak GPU memory becomes proportional to
--chunk-n/--chunk-theta/--chunk-xy, not to (2N)², so much larger N
can be modelled on a 40 GB GPU.

Chunk sizes passed to TomoLarge.R must divide the sizes they slice into:
  --chunk-n     divides N and 2N
  --chunk-theta divides NTHETA
  --chunk-xy    divides 2N

Defaults 686 / 343 / 686 are divisors of 2744 and 2058, so they work for
any integer --ups ≥ 1.

Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step2_radon_large.py \\
        --ups 8 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.tomo_large import TomoLarge
from iohdf5.dxchange_hdf5_chunks import tomo_writex
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks,
)
from utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=8)
    p.add_argument("--path", default="/data2/brain_sym_mosaic")
    p.add_argument("--in-nz",  type=int, default=2560)
    p.add_argument("--in-n",   type=int, default=2744)
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 360°; default = 3·N/4")
    p.add_argument("--nzchunk", type=int, default=1,
                   help="z-slices per Radon call")
    p.add_argument("--chunk-n",     type=int, default=686)
    p.add_argument("--chunk-theta", type=int, default=343)
    p.add_argument("--chunk-xy",    type=int, default=686)
    p.add_argument("--nbanks", type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--proj-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for proj.h5 (default: NTHETA, 8·NZCHUNK, N)")
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
ROTATION_AXIS = N / 2

NZCHUNK     = _A.nzchunk
CHUNK_N     = _A.chunk_n
CHUNK_THETA = _A.chunk_theta
CHUNK_XY    = _A.chunk_xy
NBANKS      = _A.nbanks
PROJ_VCHUNKS = tuple(_A.proj_vchunks) if _A.proj_vchunks else (NTHETA, 8 * NZCHUNK, N)


def _validate_chunks() -> None:
    problems = []
    if N % CHUNK_N or (2 * N) % CHUNK_N:
        problems.append(f"--chunk-n={CHUNK_N} must divide N={N} and 2N={2*N}")
    if NTHETA % CHUNK_THETA:
        problems.append(f"--chunk-theta={CHUNK_THETA} must divide NTHETA={NTHETA}")
    if (2 * N) % CHUNK_XY:
        problems.append(f"--chunk-xy={CHUNK_XY} must divide 2N={2*N}")
    if problems:
        raise SystemExit("chunk-size problems:\n  " + "\n  ".join(problems))


def load_chunk(src_dset, z_start: int, z_end: int) -> np.ndarray:
    """(k, N, N) complex64 host array (imag=0) for TomoLarge input."""
    k = z_end - z_start
    buf = np.empty((k, N, N), dtype=np.complex64)
    buf.real = src_dset[z_start:z_end, :, :]
    buf.imag = 0
    return buf


def main() -> None:
    _validate_chunks()
    if PROJ_VCHUNKS[1] % NZCHUNK != 0:
        raise SystemExit(
            f"--proj-vchunks C1={PROJ_VCHUNKS[1]} must be a multiple of "
            f"--nzchunk={NZCHUNK}.")
    if PROJ_VCHUNKS[0] != NTHETA:
        raise SystemExit(
            f"--proj-vchunks C0={PROJ_VCHUNKS[0]} must equal NTHETA={NTHETA}.")

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
    rprint(f"src ={SRC_H5}")
    rprint(f"proj={PROJ_H5}")
    rprint(f"chunks: CHUNK_N={CHUNK_N}  CHUNK_THETA={CHUNK_THETA}  CHUNK_XY={CHUNK_XY}")
    fde_gb  = NZCHUNK * (2*N)**2 * 8 / 1e9
    sino_gb = NZCHUNK * NTHETA * N * 8 / 1e9
    rprint(f"HOST est. per R call: fde≈{fde_gb:.1f} GB, sino≈{sino_gb:.2f} GB")

    ctx = initx_and_bcast(PROJ_H5, shape=(NTHETA, NZ, N),
                          dtype=np.float32, vchunks=PROJ_VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(PROJ_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    buf_gb = vchunk_bytes(PROJ_VCHUNKS, np.float32) / 1e9
    rprint(f"proj.h5 VDS + banks  (vchunks={PROJ_VCHUNKS}, nbanks={NBANKS}; "
           f"buffer/rank={buf_gb:.2f} GB; "
           f"{NTHETA * NZ * N * 4 / 1e12:.2f} TB total, "
           f"nvchunks={n_vchunks((NTHETA, NZ, N), PROJ_VCHUNKS)})")

    proj_min, proj_max = np.inf, -np.inf
    chunks_arg = [CHUNK_N, CHUNK_THETA, CHUNK_XY]

    rprint("building TomoLarge...")
    cl_tomo = TomoLarge(N, theta_rad, ROTATION_AXIS)
    rprint("TomoLarge ready.")

    ivchunks = list(iter_vchunks((NTHETA, NZ, N), PROJ_VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(PROJ_VCHUNKS, np.float32)

    try:
        with h5py.File(SRC_H5, "r") as fsrc:
            src_dset = fsrc["exchange/data"]
            t_read = t_radon = t_write = 0.0
            b_read = b_write = 0
            for k_i, ivc in enumerate(my_ivchunks, start=1):
                z0_vc = ivc[1] * PROJ_VCHUNKS[1]
                z1_vc = min(z0_vc + PROJ_VCHUNKS[1], NZ)
                buf.fill(0)

                for z0 in range(z0_vc, z1_vc, NZCHUNK):
                    z1 = min(z0 + NZCHUNK, NZ)
                    kz = z1 - z0

                    t0 = time.perf_counter()
                    chunk_h = load_chunk(src_dset, z0, z1)
                    if kz < NZCHUNK:
                        pad = np.zeros((NZCHUNK, N, N), dtype=np.complex64)
                        pad[:kz] = chunk_h
                        chunk_h = pad
                    t_read += time.perf_counter() - t0
                    b_read += kz * N * N * 4    # source is float32 on disk

                    t0 = time.perf_counter()
                    res_h = cl_tomo.R(chunk_h, chunks_arg)
                    del chunk_h
                    proj_chunk_h = res_h[:, :kz].real.astype(
                        np.float32, copy=False)
                    del res_h
                    t_radon += time.perf_counter() - t0

                    buf[:, z0 - z0_vc : z1 - z0_vc, :] = proj_chunk_h
                    proj_min = min(proj_min, float(proj_chunk_h.min()))
                    proj_max = max(proj_max, float(proj_chunk_h.max()))
                    del proj_chunk_h

                t0 = time.perf_counter()
                tomo_writex(PROJ_H5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
                t_write += time.perf_counter() - t0
                b_write += NTHETA * (z1_vc - z0_vc) * N * 4

                print(f"  [rank {RANK}] vchunk {k_i}/{len(my_ivchunks)}  "
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

    report_stage("step2 read (big)",   b_read,  t_read)
    report_stage("step2 write (proj)", b_write, t_write)

    norm_const = float(np.sqrt(N / NTHETA))
    rprint(f"proj = R(delta) stats: min={proj_min:.4g} max={proj_max:.4g}  "
           f"(after scaling by 1/NORM_CONST={1.0/norm_const:.4g}: "
           f"phase in [{proj_min/norm_const:.4g}, "
           f"{proj_max/norm_const:.4g}] rad)")


if __name__ == "__main__":
    main()
    from utils import hard_exit
    hard_exit()
