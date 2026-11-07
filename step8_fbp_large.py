#!/usr/bin/env python
"""FBP reconstruction — host-chunked variant of step8_fbp.py.

Same math as step8_fbp.py but the (chunk_nz, 2N, 2N) complex64 fde
never lives entirely on the GPU: TomoLargeReal.RT streams the four
reversed passes through the device one strip at a time.  Use this at
UPS ≥ 4 where the GPU-only TomoReal.RT no longer fits.

Reads {path}/model_big{UPS}x/sino.h5 (step7b's z-banked transpose; see
step8_fbp.py for why, and for the `--src paganin` A/B escape hatch) and
writes
{path}/model_big{UPS}x/rec.h5.  Same shape and theta grid as
step8_fbp.py.

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step8_fbp_large.py \\
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
from processing.fbp_filter import FBPFilter
from iohdf5.dxchange_hdf5_chunks import tomo_writex, read_slices_vchunkx
from iohdf5.h5_vchunks import (
    initx_and_bcast, open_out, close_out, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
    read_src_header, rm_src, add_rm_src_arg,
)
from iohdf5.layout import add_layout_args, add_geometry_args, input_dims, resolve_step
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=8,
                   help="matches step6 --ups (drives paths)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic")
    p.add_argument("--src", default="sino", choices=("sino", "paganin"),
                   help="input store: 'sino' = step7b's z-banked transpose "
                        "(default, sequential read); 'paganin' = step7 "
                        "output directly (transposed read, for A/B timing). "
                        "'sino' falls back to paganin.h5 when absent.")
    p.add_argument("--filter", default="ramp",
                   choices=("none", "ramp", "shepp", "cosine", "cosine2",
                            "hamming", "hann", "parzen"),
                   help="FBP filter (default: ramp)")
    p.add_argument("--center-shift", type=float, default=0.0,
                   help="move the sinogram by this many pixels (+ = towards "
                        "higher columns) in the filter, as a phase ramp "
                        "(tomocupy's centre shift).  TomoLargeReal takes the "
                        "axis at column N/2 + 0.5; step56 folds it onto N/2, "
                        "so real data wants +0.5.  Simulated data made with "
                        "TomoLargeReal.R needs 0")
    p.add_argument("--nzchunk", type=int, default=1,
                   help="z-slices per TomoLargeReal.RT call.  Default 1 at "
                        "high UPS since the host fde scales with nz.")
    p.add_argument("--chunk-n",     type=int, default=768,
                   help="x/y FFT strip width")
    p.add_argument("--chunk-theta", type=int, default=768,
                   help="theta batch for passRT1 r-FFT")
    p.add_argument("--chunk-xy",    type=int, default=768,
                   help="RT-scatter ky-band size")
    p.add_argument("--nbanks",      type=int, default=8)
    p.add_argument("--ntasks",      type=int, default=8,
                   help="parallel workers for read_slices_vchunkx (sino prefetch)")
    add_layout_args(p)
    add_geometry_args(p)
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for rec.h5; default comes from "
                        "iohdf5.layout (--mem-budget / --chunk-bytes)")
    add_rm_src_arg(p, "the input sino.h5 (or paganin.h5)")
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
SINO_H5  = f"{DST_DIR}/sino.h5"
PGN_H5   = f"{DST_DIR}/paganin.h5"
SRC_H5   = (SINO_H5 if (_A.src == "sino" and os.path.isfile(SINO_H5))
            else PGN_H5)
DST_H5   = f"{DST_DIR}/rec.h5"

IN_NZ, IN_N = input_dims(_A, UPS)   # 3072 (init.h5 after step00) unless --nz/--n
NZ    = IN_NZ * UPS
N     = IN_N  * UPS
FILTER     = _A.filter
CHUNK_N     = _A.chunk_n
CHUNK_THETA = _A.chunk_theta
CHUNK_XY    = _A.chunk_xy
NTASKS      = _A.ntasks

# Layout from the shared byte-budget policy -- same plan step8_fbp uses,
# so the host-chunked twin writes an identically-laid-out rec.h5.
# Resolved in main(): the policy needs the true angle count, which only
# the source file knows (see step8_fbp._resolve_plan).
_PLAN = NBANKS = VCHUNKS = H5CHUNKS = NZCHUNK = None


def _resolve_plan(ntheta):
    global _PLAN, NBANKS, VCHUNKS, H5CHUNKS, NZCHUNK
    _PLAN       = resolve_step("rec", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                               ntheta=2 * ntheta,
                               nbanks=_A.nbanks, mem_budget_gb=_A.mem_budget,
                               chunk_mb=_A.chunk_bytes, nzchunk=_A.nzchunk,
                               vchunks=_A.vchunks, nranks=SIZE)
    NBANKS      = _PLAN.nbanks
    VCHUNKS     = _PLAN.vchunks
    H5CHUNKS    = _PLAN.chunks
    NZCHUNK     = _PLAN.align
    return resolve_step("sino", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                        ntheta=2 * ntheta,
                        nbanks=_A.nbanks, mem_budget_gb=_A.mem_budget,
                        chunk_mb=_A.chunk_bytes, nzchunk=_A.nzchunk,
                        nranks=SIZE)


def _validate_chunks(NTHETA: int, sino_plan) -> None:
    problems = []
    if SRC_H5 == SINO_H5 and VCHUNKS[0] % sino_plan.vchunks[1] != 0:
        problems.append(f"rec --vchunks C0={VCHUNKS[0]} must be a multiple "
                        f"of sino.h5's z-bank {sino_plan.vchunks[1]}")
    if N % CHUNK_N:
        problems.append(f"--chunk-n={CHUNK_N} must divide N={N}")
    if NTHETA % CHUNK_THETA:
        problems.append(f"--chunk-theta={CHUNK_THETA} must divide NTHETA={NTHETA}")
    if (2 * N) % CHUNK_XY:
        problems.append(f"--chunk-xy={CHUNK_XY} must divide 2N={2*N}")
    if VCHUNKS[0] % NZCHUNK != 0:
        problems.append(f"--vchunks C0={VCHUNKS[0]} must be a multiple "
                        f"of --nzchunk={NZCHUNK}")
    if VCHUNKS[1] != N or VCHUNKS[2] != N:
        problems.append(f"--vchunks C1×C2 must equal N×N ({N}×{N}) "
                        f"— got {VCHUNKS[1]}×{VCHUNKS[2]}")
    if problems:
        raise SystemExit("chunk-size problems:\n  " + "\n  ".join(problems))


def main() -> None:
    from mpi_utils import banner
    banner("8", f"{os.path.basename(SRC_H5)} -> rec.h5  "
                f"(FBP, large/host-chunked, filter={FILTER})")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    src_shape, theta_deg = read_src_header(SRC_H5, rank=RANK, comm=COMM)
    NTHETA = src_shape[0]
    if src_shape[1:] != (NZ, N):
        raise SystemExit(
            f"{SRC_H5} shape {src_shape} incompatible with (NZ={NZ}, N={N})")
    theta_rad = np.deg2rad(theta_deg).astype(np.float32)

    _validate_chunks(NTHETA, _resolve_plan(NTHETA))

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()

    rprint(f"UPS={UPS}  nz={NZ}  n={N}  ntheta={NTHETA}  "
           f"nzchunk={NZCHUNK}  filter={FILTER}  "
           f"chunks=(chunk_n={CHUNK_N}, chunk_theta={CHUNK_THETA}, chunk_xy={CHUNK_XY})")
    # Host fde per RT call is (nzchunk, 2N, 2N) c64 banded.
    rprint(f"host — fde ≈ "
           f"{NZCHUNK * (2*N) * (2*N) * 8 / 1e9:.2f} GB (banded)")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(DST_H5, (NZ, N, N), np.float32,
                        VCHUNKS, "proj", NBANKS, chunks=H5CHUNKS,
                        companion_bytes=NTHETA * VCHUNKS[0] * N * 4)

    ctx = initx_and_bcast(DST_H5, shape=(NZ, N, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="proj", nbanks=NBANKS, chunks=H5CHUNKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(DST_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()
    open_out(ctx)                 # single-file output: collective MPI-IO open

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((NZ, N, N), VCHUNKS)}")

    cl_filter = FBPFilter(N)
    w_gpu     = cl_filter.calc_filter(FILTER)
    if _A.center_shift:
        # x(j - s) <-> X(f) exp(-2 pi i f s), f in cycles per pixel of the
        # 4N-padded line the filter works on
        f_gpu = cp.fft.rfftfreq(cl_filter.ne).astype(cp.float32)
        w_gpu = (w_gpu * cp.exp(-2j * cp.pi * f_gpu * np.float32(_A.center_shift))
                 ).astype(cp.complex64)
        rprint(f"centre shift {_A.center_shift:+g} px (filter phase ramp)")
    cl_tomo   = TomoLargeReal(N, theta_rad, CHUNK_XY)   # eager index precompute

    r_min, r_max = np.inf, -np.inf
    r_sum, r_cnt = 0.0, 0
    r_has_nan    = False

    t_read = t_comp = t_write = 0.0
    b_read = b_write = 0

    ivchunks = list(iter_vchunks((NZ, N, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)

    # Prefetch shm for the sinogram vchunkx (NTHETA, VCHUNKS[0], N).
    # One read_slices_vchunkx per rec vchunk (NTASKS parallel workers)
    # replaces NZCHUNK-many per-inner plain-h5py reads.
    sino_shape = (NTHETA, VCHUNKS[0], N)
    shm_sino, sino_buf = alloc_shm(sino_shape, np.float32)
    sino_c = np.empty((NTHETA, NZCHUNK, N), dtype=np.float32)

    try:
        for k, ivc in enumerate(my_ivchunks, start=1):
            z0_vc = ivc[0] * VCHUNKS[0]
            z1_vc = min(z0_vc + VCHUNKS[0], NZ)
            buf.fill(0)

            # Vchunkx read: all sinogram data for this rec vchunk in one shot.
            t0 = time.perf_counter()
            read_slices_vchunkx(SRC_H5, shm_sino, ntasks=NTASKS,
                                vchunksx=sino_shape,
                                ivchunkx=(0, ivc[0], 0))
            t_read += time.perf_counter() - t0
            b_read += NTHETA * (z1_vc - z0_vc) * N * 4

            for zc0 in range(z0_vc, z1_vc, NZCHUNK):
                zc1 = min(zc0 + NZCHUNK, z1_vc)
                b   = zc1 - zc0

                # A z-range of the (NTHETA, C0, N) prefetch is a strided
                # view whenever C0 > NZCHUNK; filter_host / RT need it
                # C-contiguous, so copy into a reused buffer.
                sino_h = sino_c[:, :b, :] if b == NZCHUNK else \
                    np.empty((NTHETA, b, N), dtype=np.float32)
                np.copyto(sino_h, sino_buf[:, zc0 - z0_vc : zc1 - z0_vc, :])

                # Filter via filter_host — chunks the H2D → GPU-filter
                # → D2H roundtrip by FBPFilter.batch_chunk (default 256
                # rows), so peak GPU stays bounded regardless of NTHETA.
                # At UPS≥32 the full-sino upload here would OOM (29 GB
                # f32 at UPS=32, 116 GB at UPS=64).
                t0 = time.perf_counter()
                cl_filter.filter_host(sino_h, w_gpu)

                rec_bh = cl_tomo.RT(sino_h,
                                    [CHUNK_N, CHUNK_THETA, CHUNK_XY])
                cp.get_default_memory_pool().free_all_blocks()
                t_comp += time.perf_counter() - t0

                # rec_bh is a pinned f32 ndarray of shape (b, N, N).
                rec_arr = np.asarray(rec_bh)
                r_min = min(r_min, float(rec_arr.min()))
                r_max = max(r_max, float(rec_arr.max()))
                r_sum += float(rec_arr.sum())
                r_cnt += b * N * N
                if np.isnan(rec_arr).any():
                    r_has_nan = True

                buf[zc0 - z0_vc : zc1 - z0_vc] = rec_arr

            t0 = time.perf_counter()
            tomo_writex(DST_H5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
            t_write += time.perf_counter() - t0
            b_write += (z1_vc - z0_vc) * N * N * 4

            print(f"  [rank {RANK}] vchunk {k}/{len(my_ivchunks)}  "
                  f"z=[{z0_vc},{z1_vc})  "
                  f"(read={t_read:.1f}s comp={t_comp:.1f}s "
                  f"write={t_write:.1f}s)", flush=True)
    finally:
        free_shm(shm)
        free_shm(shm_sino)
    close_out(ctx)

    r_min     = allreduce(r_min,     MPI.MIN)
    r_max     = allreduce(r_max,     MPI.MAX)
    r_sum     = allreduce(r_sum,     MPI.SUM)
    r_cnt     = allreduce(r_cnt,     MPI.SUM)
    r_has_nan = allreduce(r_has_nan, MPI.LOR)
    barrier()

    report_stage(f"step8 read ({os.path.basename(SRC_H5)})", b_read, t_read)
    report_stage("step8 write (rec)",    b_write, t_write)

    rprint(f"rec stats: min={r_min:.4g} max={r_max:.4g} "
           f"mean={r_sum/max(r_cnt,1):.4g} nan={r_has_nan}")
    rprint(f"wrote {NZ} z-slices to {DST_H5}")

    rm_src(SRC_H5, enabled=_A.rm_src, rank=RANK, comm=COMM)


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
