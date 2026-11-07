#!/usr/bin/env python
"""Filtered backprojection (FBP) reconstruction of the paganin-retrieved
180° big projection.

Reads {path}/model_big{UPS}x/sino.h5 (N_HALF, NZ, N) — step7b's z-banked
transpose of the line integrals from step7_paganin.py — and writes
{path}/model_big{UPS}x/rec.h5 with shape (NZ, N, N) float32:  the
reconstructed 3-D volume.

For each --nzchunk z-slab this rank owns:
  1. Read (N_HALF, chunk_nz, N) sino slab from sino.h5.  Because that
     store is banked on z — the same axis this loop shards on — the slab
     is one whole super-chunk and each worker stays inside its own bank
     files.  `--src paganin` reads step7's output directly instead, which
     is banked on theta and so reads transposed (~500 MB/s against
     ~20 GB/s in the UPS=1 log); it exists for A/B timing only.
  2. Apply FBP filter along the sample axis (rfft → w → irfft) —
     see processing.fbp_filter.FBPFilter (ported from tomocupy).
  3. Backproject via processing.tomo.TomoReal.RT → (chunk_nz, N, N).
  4. tomo_writex fans the z-slab buffer to disk across --nbanks writers.

GPU-only TomoReal — the (chunk_nz, 2N, 2N) complex64 fde_full buffer
(lazily allocated on first RT() call) must fit on the GPU (~600 MB at
UPS=1 for chunk_nz=8).  For UPS ≥ 4 swap to step8_fbp_large.py which
uses the host-chunked TomoLargeReal.RT (4-pass streaming: r-FFT → adj
scatter → y-IFFT → x-IFFT+phi+crop).

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step8_fbp.py \\
        --ups 1 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.tomo       import TomoReal
from processing.fbp_filter import FBPFilter
from iohdf5.dxchange_hdf5_chunks import tomo_writex, read_slices_vchunkx
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
    read_src_header, rm_src, add_rm_src_arg,
)
from iohdf5.layout import add_layout_args, add_geometry_args, input_dims, resolve_step
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=1,
                   help="matches step6 --ups (drives paths)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/model_big{UPS}x/sino.h5, "
                        "writes {path}/model_big{UPS}x/rec.h5")
    p.add_argument("--src", default="sino", choices=("sino", "paganin"),
                   help="input store: 'sino' = step7b's z-banked transpose "
                        "(default, sequential read); 'paganin' = step7 "
                        "output directly (transposed read, for A/B timing). "
                        "'sino' falls back to paganin.h5 when sino.h5 is "
                        "absent.")
    p.add_argument("--filter", default="ramp",
                   choices=("none", "ramp", "shepp", "cosine", "cosine2",
                            "hamming", "hann", "parzen"),
                   help="FBP filter (default: ramp)")
    p.add_argument("--nzchunk", type=int, default=8,
                   help="z-slices per TomoReal.RT call (bounds GPU memory)")
    p.add_argument("--nbanks",  type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--ntasks",  type=int, default=8,
                   help="parallel workers for read_slices_vchunkx (sino prefetch)")
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for rec.h5; default comes from "
                        "iohdf5.layout (--mem-budget / --chunk-bytes)")
    add_rm_src_arg(p, "the input sino.h5 (or paganin.h5)")
    add_layout_args(p)
    add_geometry_args(p)
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
SINO_H5  = f"{DST_DIR}/sino.h5"
PGN_H5   = f"{DST_DIR}/paganin.h5"
# 'sino' is the default but step7b may not have run (or may have been
# skipped at UPS=1, where the transposed read is cheap) -- fall back
# rather than dying three hours into a job.
SRC_H5   = (SINO_H5 if (_A.src == "sino" and os.path.isfile(SINO_H5))
            else PGN_H5)
DST_H5   = f"{DST_DIR}/rec.h5"

IN_NZ, IN_N = input_dims(_A, UPS)   # 3072 (init.h5 after step00) unless --nz/--n
NZ    = IN_NZ * UPS
N     = IN_N  * UPS
FILTER     = _A.filter
NTASKS     = _A.ntasks

# Layout from the shared byte-budget policy.  This step holds two buffers
# per rank -- the (C0, N, N) rec vchunk and the (NTHETA, C0, N) sinogram
# prefetch -- so the real cost per z is (N·N + NTHETA·N)·4.  The old
# 8·NZCHUNK literal counted neither: it already asked for 9.7 GB + 1.8 GB
# at UPS=1 and 155 GB at UPS=2.  NZCHUNK follows the plan's alignment,
# since the backprojection loop can just run in smaller pieces.
# Resolved in main(), once the source file has told us the true angle
# count: without ntheta= the policy assumes NTHETA=3N/4 and prices the
# sinogram prefetch at half its real size.
_PLAN = NBANKS = VCHUNKS = H5CHUNKS = NZCHUNK = None


def _resolve_plan(ntheta):
    """rec's super-chunk / HDF5 chunk / z-alignment for this angle count."""
    global _PLAN, NBANKS, VCHUNKS, H5CHUNKS, NZCHUNK
    _PLAN    = resolve_step("rec", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                            ntheta=2 * ntheta,
                            nbanks=_A.nbanks, mem_budget_gb=_A.mem_budget,
                            chunk_mb=_A.chunk_bytes, nzchunk=_A.nzchunk,
                            vchunks=_A.vchunks, nranks=SIZE)
    NBANKS   = _PLAN.nbanks
    VCHUNKS  = _PLAN.vchunks
    H5CHUNKS = _PLAN.chunks
    NZCHUNK  = _PLAN.align
    return resolve_step("sino", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                        ntheta=2 * ntheta,
                        nbanks=_A.nbanks, mem_budget_gb=_A.mem_budget,
                        chunk_mb=_A.chunk_bytes, nzchunk=_A.nzchunk,
                        nranks=SIZE)


def main() -> None:
    from mpi_utils import banner
    banner("8", f"{os.path.basename(SRC_H5)} -> rec.h5  "
                f"(FBP, filter={FILTER})")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    src_shape, theta_deg = read_src_header(SRC_H5, rank=RANK, comm=COMM)
    NTHETA = src_shape[0]        # = N_HALF from step5/step6
    if src_shape[1:] != (NZ, N):
        raise SystemExit(
            f"{SRC_H5} shape {src_shape} incompatible with (NZ={NZ}, N={N})")
    theta_rad = np.deg2rad(theta_deg).astype(np.float32)

    sino_plan = _resolve_plan(NTHETA)

    if VCHUNKS[0] % NZCHUNK != 0:
        raise SystemExit(
            f"--vchunks C0={VCHUNKS[0]} must be a multiple of "
            f"--nzchunk={NZCHUNK}.")
    if VCHUNKS[1] != N or VCHUNKS[2] != N:
        raise SystemExit(
            f"--vchunks C1×C2 must equal N×N ({N}×{N}).  "
            f"Got {VCHUNKS[1]}×{VCHUNKS[2]}.")
    # sino.h5 is banked every sino_plan.vchunks[1] z-rows.  A read slab
    # that is not a whole number of those banks would make two ranks
    # touch one bank file, which is the access pattern step7b exists to
    # avoid -- the policy snaps Zb to a divisor of VCHUNKS[0], so this
    # only fires when --vchunks was overridden by hand.
    if SRC_H5 == SINO_H5 and VCHUNKS[0] % sino_plan.vchunks[1] != 0:
        raise SystemExit(
            f"rec --vchunks C0={VCHUNKS[0]} must be a multiple of sino.h5's "
            f"z-bank {sino_plan.vchunks[1]} (drop --vchunks and let the "
            f"layout policy pick, or re-run step7b with a matching Zb).")

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()

    rprint(f"UPS={UPS}  nz={NZ}  n={N}  ntheta={NTHETA}  "
           f"nzchunk={NZCHUNK}  filter={FILTER}")
    rprint(f"GPU est. — TomoReal._buf_fde_full: "
           f"{NZCHUNK * (2*N) * (2*N) * 8 / 1e9:.2f} GB   "
           f"+ TomoReal._buf_sino: {NTHETA * NZCHUNK * N * 8 / 1e9:.2f} GB")

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

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((NZ, N, N), VCHUNKS)}")

    # FBP filter + TomoReal backprojector both scoped to NZCHUNK z-slices.
    cl_filter = FBPFilter(N)
    w_gpu     = cl_filter.calc_filter(FILTER)
    cl_tomo   = TomoReal(N, NZCHUNK, theta_rad)

    r_min, r_max = np.inf, -np.inf
    r_sum, r_cnt = 0.0, 0
    r_has_nan    = False

    t_read = t_comp = t_write = 0.0
    b_read = b_write = 0

    ivchunks = list(iter_vchunks((NZ, N, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)

    # Prefetch shm for the sinogram vchunkx (NTHETA, VCHUNKS[0], N).
    # One parallel read per rec vchunk instead of NZCHUNK-many single-
    # threaded plain-h5py reads.  Buffer size at UPS=1: 2304·64·3072·4 =
    # 1.8 GB per rank.  Amp per read = NZ/VCHUNKS[0] = 3072/64 = 48× (vs
    # 3072/NZCHUNK=384× when reading per-inner-slab), and the read runs
    # across NTASKS parallel workers instead of one plain-h5py handle.
    sino_shape = (NTHETA, VCHUNKS[0], N)
    shm_sino, sino_buf = alloc_shm(sino_shape, np.float32)

    try:
        for k, ivc in enumerate(my_ivchunks, start=1):
            z0_vc = ivc[0] * VCHUNKS[0]
            z1_vc = min(z0_vc + VCHUNKS[0], NZ)
            buf.fill(0)

            # ── Vchunkx read: (NTHETA, VCHUNKS[0], N) in parallel ──
            t0 = time.perf_counter()
            read_slices_vchunkx(SRC_H5, shm_sino, ntasks=NTASKS,
                                vchunksx=sino_shape,
                                ivchunkx=(0, ivc[0], 0))
            t_read += time.perf_counter() - t0
            b_read += NTHETA * (z1_vc - z0_vc) * N * 4

            for zc0 in range(z0_vc, z1_vc, NZCHUNK):
                zc1 = min(zc0 + NZCHUNK, z1_vc)
                b   = zc1 - zc0

                # Slice from the pre-fetched RAM buffer — free.
                sino_h = sino_buf[:, zc0 - z0_vc : zc1 - z0_vc, :]

                # Filter + backproject.
                t0 = time.perf_counter()
                sino_d = cp.asarray(sino_h)
                cl_filter.filter(sino_d, w_gpu)
                rec_d  = cl_tomo.RT(sino_d)               # (b, N, N) f32
                del sino_d
                # rec_d is real f32 (TomoReal.RT returns .real when sino is float32).
                rec_batch_h = cp.asnumpy(rec_d)
                del rec_d
                cp.get_default_memory_pool().free_all_blocks()
                t_comp += time.perf_counter() - t0

                r_min = min(r_min, float(rec_batch_h.min()))
                r_max = max(r_max, float(rec_batch_h.max()))
                r_sum += float(rec_batch_h.sum())
                r_cnt += b * N * N
                if np.isnan(rec_batch_h).any():
                    r_has_nan = True

                buf[zc0 - z0_vc : zc1 - z0_vc] = rec_batch_h
                del rec_batch_h

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
