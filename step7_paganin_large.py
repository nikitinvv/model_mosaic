#!/usr/bin/env python
"""Single-distance Paganin phase retrieval on the stitched big projection,
host-chunked variant — same math as step7_paganin.py but the full 2-D
FFT never lives on the GPU.  For UPS ≥ 8 where the (NPGNCHUNK, NZ, N)
complex64 buffer no longer fits on a 40 GB device.

Reads {path}/model_big{UPS}x/stitched.h5 and writes
{path}/model_big{UPS}x/paganin.h5.  Same shape and theta grid as the
non-large variant.

For each --vchunks super-chunk (θ_super, NZ, N) this rank owns:
  1. Read the θ-slab from stitched.h5.
  2. Loop NPGNCHUNK θ-batches through PaganinLarge.retrieve() —
     3-pass streaming x-FFT / y-FFT+H·mult / x-IFFT+log+scale, one
     CHUNK_NZ or CHUNK_N strip on the GPU at a time.
  3. tomo_writex fans the vchunk buffer to disk across --nbanks writers.

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step7_paganin_large.py \\
        --ups 8 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.paganin_large import PaganinLarge
from iohdf5.dxchange_hdf5_chunks import tomo_writex, read_projs_vchunkx
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
)
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=8,
                   help="matches step3/step5 --ups (drives paths)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic")
    p.add_argument("--energy",     type=float, default=30.0,   help="keV")
    p.add_argument("--voxelsize",  type=float, default=1.38e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",   type=float, default=0.2,
                   help="sample → detector distance, meters (matches step3)")
    p.add_argument("--alpha",      type=float, default=1e-3,
                   help="Tikhonov regularisation added to T² in the filter")
    p.add_argument("--npgnchunk",  type=int, default=1,
                   help="angles per PaganinLarge.retrieve() call.  Default 1 "
                        "at high UPS since the host fde scales with ntheta.")
    p.add_argument("--chunk-nz",   type=int, default=768,
                   help="pass1/3 x-strip depth")
    p.add_argument("--chunk-n",    type=int, default=768,
                   help="pass2 y-strip width")
    p.add_argument("--nbanks",     type=int, default=8)
    p.add_argument("--ntasks",     type=int, default=8,
                   help="parallel workers for read_projs_vchunkx (stitched prefetch)")
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for paganin.h5 (default: 8·NPGNCHUNK, NZ, N)")
    return p.parse_args()


_A = _parse_args()

UPS       = _A.ups
BASE_DIR  = _A.path
DST_DIR   = f"{BASE_DIR}/model_big{UPS}x"
SRC_H5    = f"{DST_DIR}/stitched.h5"
DST_H5    = f"{DST_DIR}/paganin.h5"

IN_NZ = IN_N = 3072            # init.h5 dims after step00; UPS scales from here
NZ    = IN_NZ * UPS
N     = IN_N  * UPS

ENERGY     = _A.energy
VOXELSIZE  = _A.voxelsize
DISTANCE   = _A.distance
ALPHA      = _A.alpha

NPGNCHUNK   = _A.npgnchunk
NBANKS      = _A.nbanks
NTASKS      = _A.ntasks
VCHUNKS = tuple(_A.vchunks) if _A.vchunks else (8 * NPGNCHUNK, NZ, N)

CHUNK_NZ = _A.chunk_nz
CHUNK_N  = _A.chunk_n


def _validate_chunks() -> None:
    problems = []
    if NZ % CHUNK_NZ:
        problems.append(f"--chunk-nz={CHUNK_NZ} must divide NZ={NZ}")
    if N  % CHUNK_N:
        problems.append(f"--chunk-n={CHUNK_N} must divide N={N}")
    if VCHUNKS[0] % NPGNCHUNK != 0:
        problems.append(f"--vchunks C0={VCHUNKS[0]} must be a multiple "
                        f"of --npgnchunk={NPGNCHUNK}")
    if VCHUNKS[1] != NZ or VCHUNKS[2] != N:
        problems.append(f"--vchunks C1×C2 must equal NZ×N ({NZ}×{N}) "
                        f"— got {VCHUNKS[1]}×{VCHUNKS[2]}")
    if problems:
        raise SystemExit("chunk-size problems:\n  " + "\n  ".join(problems))


def main() -> None:
    from mpi_utils import banner
    banner("7", f"stitched.h5 -> paganin.h5  "
                f"(single-distance Paganin, large/host-chunked, α={ALPHA})")

    _validate_chunks()

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    wavelength = 1.24e-9 / ENERGY

    if RANK == 0:
        with h5py.File(SRC_H5, "r") as f:
            src_shape = tuple(f["exchange/data"].shape)
            theta_deg = f["exchange/theta"][:]
    else:
        src_shape = None
        theta_deg = None
    src_shape = COMM.bcast(src_shape, root=0)
    theta_deg = COMM.bcast(theta_deg, root=0)
    N_HALF = src_shape[0]
    if src_shape[1:] != (NZ, N):
        raise SystemExit(
            f"stitched.h5 shape {src_shape} incompatible with (NZ={NZ}, N={N})")

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()

    rprint(f"UPS={UPS}  nz={NZ}  n={N}  n_half={N_HALF}  "
           f"npgnchunk={NPGNCHUNK}  chunks=(chunk_nz={CHUNK_NZ}, chunk_n={CHUNK_N})")
    rprint(f"paganin: E={ENERGY} keV  λ={wavelength:.4e} m  "
           f"voxel={VOXELSIZE} m  distance={DISTANCE} m  α={ALPHA}")
    peak_gb = max(NPGNCHUNK * CHUNK_NZ * N * 12,
                  NPGNCHUNK * NZ * CHUNK_N * 16) * 2 / 1e9
    rprint(f"GPU est. — peak stream buffers ≈ {peak_gb:.2f} GB")
    rprint(f"host — fde + out ≈ "
           f"{NPGNCHUNK * NZ * N * (8 + 4) / 1e9:.2f} GB (banded)")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(DST_H5, (N_HALF, NZ, N), np.float32,
                        VCHUNKS, "proj", NBANKS)

    ctx = initx_and_bcast(DST_H5, shape=(N_HALF, NZ, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(DST_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((N_HALF, NZ, N), VCHUNKS)}")

    cl = PaganinLarge(N, NZ,
                      wavelength=wavelength, voxelsize=VOXELSIZE,
                      distance=DISTANCE, alpha=ALPHA)

    p_min, p_max = np.inf, -np.inf
    p_sum, p_cnt = 0.0, 0
    p_has_nan    = False

    t_read = t_comp = t_write = 0.0
    b_read = b_write = 0

    ivchunks = list(iter_vchunks((N_HALF, NZ, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)

    # Prefetch shm for the vchunkx θ-slab (VCHUNKS[0], NZ, N) — aligned
    # read on proj-stored stitched.h5 via NTASKS parallel workers.
    stitched_slab_shape = (VCHUNKS[0], NZ, N)
    shm_slab, stitched_slab_buf = alloc_shm(stitched_slab_shape, np.float32)

    try:
        for k, ivc in enumerate(my_ivchunks, start=1):
            t0_vc = ivc[0] * VCHUNKS[0]
            t1_vc = min(t0_vc + VCHUNKS[0], N_HALF)
            buf.fill(0)

            t0 = time.perf_counter()
            read_projs_vchunkx(SRC_H5, shm_slab, ntasks=NTASKS,
                               vchunksx=stitched_slab_shape,
                               ivchunkx=(ivc[0], 0, 0))
            slab_h = stitched_slab_buf
            t_read += time.perf_counter() - t0
            b_read += (t1_vc - t0_vc) * NZ * N * 4

            for tb0 in range(t0_vc, t1_vc, NPGNCHUNK):
                tb1 = min(tb0 + NPGNCHUNK, t1_vc)
                b   = tb1 - tb0

                t0 = time.perf_counter()
                phase_bh = cl.retrieve(
                    slab_h[tb0 - t0_vc : tb1 - t0_vc],
                    chunks=[CHUNK_NZ, CHUNK_N])
                cp.get_default_memory_pool().free_all_blocks()
                t_comp += time.perf_counter() - t0

                # phase_bh is a BandedPinned view of cl._out — materialise
                # the (b, NZ, N) slice into buf via BandedPinned.copy_to.
                if hasattr(phase_bh, 'copy_to'):
                    phase_bh.copy_to(buf[tb0 - t0_vc : tb1 - t0_vc],
                                     np.s_[:, :, :])
                else:
                    buf[tb0 - t0_vc : tb1 - t0_vc] = phase_bh

                sub = buf[tb0 - t0_vc : tb1 - t0_vc]
                p_min = min(p_min, float(sub.min()))
                p_max = max(p_max, float(sub.max()))
                p_sum += float(sub.sum())
                p_cnt += b * NZ * N
                if np.isnan(sub).any():
                    p_has_nan = True

            del slab_h

            t0 = time.perf_counter()
            tomo_writex(DST_H5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
            t_write += time.perf_counter() - t0
            b_write += (t1_vc - t0_vc) * NZ * N * 4

            print(f"  [rank {RANK}] vchunk {k}/{len(my_ivchunks)}  "
                  f"θ=[{t0_vc},{t1_vc})  "
                  f"(read={t_read:.1f}s comp={t_comp:.1f}s "
                  f"write={t_write:.1f}s)", flush=True)
    finally:
        free_shm(shm)
        free_shm(shm_slab)

    p_min     = allreduce(p_min,     MPI.MIN)
    p_max     = allreduce(p_max,     MPI.MAX)
    p_sum     = allreduce(p_sum,     MPI.SUM)
    p_cnt     = allreduce(p_cnt,     MPI.SUM)
    p_has_nan = allreduce(p_has_nan, MPI.LOR)
    barrier()

    report_stage("step7 read (stitched)", b_read,  t_read)
    report_stage("step7 write (paganin)", b_write, t_write)

    rprint(f"paganin stats: min={p_min:.4g} max={p_max:.4g} "
           f"mean={p_sum/max(p_cnt,1):.4g} nan={p_has_nan}")
    rprint(f"wrote {N_HALF} angles to {DST_H5}")


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
