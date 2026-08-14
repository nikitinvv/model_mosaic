#!/usr/bin/env python
"""Radon projections — host-chunked variant of step2_radon.py.

Same output format (VDS + banks proj.h5) but uses the host-chunked
TomoLargeReal implementation from tomo_large.py: the big (nz, 2N, N+1)
`fde` and (nz, ntheta, N) `sino` buffers live on the HOST (pinned), and
small pieces are staged through the GPU.  Peak GPU memory becomes
proportional to --chunk-n/--chunk-theta/--chunk-xy, not to (2N)², so
much larger N can be modelled on a 40 GB GPU.

TomoLargeReal uses rfft/float32 throughout — half the host fde memory
of a full-complex64 path, half the x-FFT bandwidth, and no
complex↔real wrapping at the boundaries.

Chunk sizes passed to TomoLargeReal.R must divide the sizes they slice into:
  --chunk-n     divides N and 2N
  --chunk-theta divides NTHETA
  --chunk-xy    divides 2N
Defaults (768) divide every 3072·UPS grid; override per-run if needed.

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

from processing.tomo_large import TomoLargeReal
from iohdf5.dxchange_hdf5_chunks import tomo_writex
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
)
from iohdf5.layout import add_layout_args, resolve_step
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=8)
    p.add_argument("--path", default="/data2/brain_sym_mosaic")
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 180°; default = 3·N/4")
    p.add_argument("--nzchunk", type=int, default=1,
                   help="z-slices per Radon call")
    p.add_argument("--chunk-n",     type=int, default=768,
                   help="x/y FFT strip width")
    p.add_argument("--chunk-theta", type=int, default=768,
                   help="angle batch for r-IFFT")
    p.add_argument("--chunk-xy",    type=int, default=768,
                   help="NUFFT gather bin edge")
    p.add_argument("--nbanks", type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    add_layout_args(p)
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for proj.h5; default comes from "
                        "iohdf5.layout (--mem-budget / --chunk-bytes)")
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

CHUNK_N     = _A.chunk_n
CHUNK_THETA = _A.chunk_theta
CHUNK_XY    = _A.chunk_xy
# Layout from the shared byte-budget policy -- same plan step2_radon uses,
# so the host-chunked twin writes an identically-laid-out proj.h5.
_PLAN    = resolve_step("proj", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                        ntheta=NTHETA, nbanks=_A.nbanks,
                        mem_budget_gb=_A.mem_budget,
                        chunk_mb=_A.chunk_bytes, nzchunk=_A.nzchunk,
                        vchunks=_A.vchunks, nranks=SIZE)
NBANKS      = _PLAN.nbanks
VCHUNKS     = _PLAN.vchunks
H5CHUNKS    = _PLAN.chunks
NZCHUNK     = _PLAN.align


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


def load_chunk_into(src_dset, dst: np.ndarray, z_start: int, z_end: int) -> None:
    """Read src_dset[z_start:z_end] straight into the first (z_end-z_start)
    rows of `dst` (a pinned float32 buffer owned by TomoLargeReal).  Skips
    the fresh (k, N, N) allocation that `astype(np.float32, copy=False)`
    would trigger when the on-disk dtype ≠ float32."""
    kz = z_end - z_start
    src_dset.read_direct(dst, source_sel=np.s_[z_start:z_end, :, :],
                         dest_sel=np.s_[:kz, :, :])


def main() -> None:
    _validate_chunks()
    if VCHUNKS[1] % NZCHUNK != 0:
        raise SystemExit(
            f"--vchunks C1={VCHUNKS[1]} must be a multiple of "
            f"--nzchunk={NZCHUNK}.")
    if VCHUNKS[0] != NTHETA:
        raise SystemExit(
            f"--vchunks C0={VCHUNKS[0]} must equal NTHETA={NTHETA}.")

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
    rprint(f"chunks: CHUNK_N={CHUNK_N}  CHUNK_THETA={CHUNK_THETA}  CHUNK_XY={CHUNK_XY}")
    # TomoLargeReal buffers: fde is rfft-half (2N × (N+1)) c64, and
    # sino_real is a f32 view of sino_c's memory (no extra allocation).
    fde_gb  = NZCHUNK * (2*N) * (N + 1) * 8 / 1e9
    sino_gb = NTHETA * NZCHUNK * N * 8 / 1e9
    rprint(f"HOST est. per R call: fde≈{fde_gb:.1f} GB, sino≈{sino_gb:.2f} GB")

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
    chunks_arg = [CHUNK_N, CHUNK_THETA, CHUNK_XY]

    rprint("building TomoLargeReal...")
    cl_tomo = TomoLargeReal(N, theta_rad, CHUNK_XY)     # eager index precompute
    rprint("TomoLargeReal ready.")

    ivchunks = list(iter_vchunks((NTHETA, NZ, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)

    # Pinned obj input, allocated once and reused across all iterations.
    # h5py reads straight into it; last (kz < NZCHUNK) chunk keeps the
    # tail zero-padded by a one-time fill below.
    obj_pinned = cl_tomo.obj_buffer(NZCHUNK)
    obj_pinned.fill(0)

    try:
        with h5py.File(SRC_H5, "r") as fsrc:
            src_dset = fsrc["exchange/data"]
            t_read = t_radon = t_write = 0.0
            b_read = b_write = 0
            for k_i, ivc in enumerate(my_ivchunks, start=1):
                z0_vc = ivc[1] * VCHUNKS[1]
                z1_vc = min(z0_vc + VCHUNKS[1], NZ)
                buf.fill(0)

                for z0 in range(z0_vc, z1_vc, NZCHUNK):
                    z1 = min(z0 + NZCHUNK, NZ)
                    kz = z1 - z0

                    t0 = time.perf_counter()
                    load_chunk_into(src_dset, obj_pinned, z0, z1)
                    if kz < NZCHUNK:
                        obj_pinned[kz:].fill(0)
                    t_read += time.perf_counter() - t0
                    b_read += kz * N * N * 4    # source is float32 on disk

                    t0 = time.perf_counter()
                    # TomoLargeReal.R returns a (NTHETA, NZCHUNK, N) float32
                    # VIEW into the class's pinned sino buffer — valid until
                    # the next R() call.  We copy out via buf[...]= below,
                    # so the view lifetime doesn't leak past this iteration.
                    res_h = cl_tomo.R(obj_pinned, chunks_arg)
                    proj_chunk_h = res_h[:, :kz]
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
        cl_tomo.free()           # return cached pinned/GPU buffers to pools
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
    from mpi_utils import run_main
    run_main(main)
