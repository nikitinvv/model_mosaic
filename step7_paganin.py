#!/usr/bin/env python
"""Single-distance Paganin phase retrieval on the stitched big projection.

Reads {path}/model_big{UPS}x/stitched.h5 (N_HALF, NZ, N) — the 180°
mirror-fold stitched output from step6_stitch.py, already flat-field
normalised (uncovered pixels = 1) — and writes
{path}/model_big{UPS}x/paganin.h5 with the same shape and theta grid.

For each --vchunks super-chunk (θ_super, NZ, N) this rank owns:
  1. Read the θ-slab from stitched.h5 (one big read).
  2. Loop NPGNCHUNK θ-batches through the Paganin filter (per-angle 2-D
     FFT / K-mult / IFFT / log / scale) — see processing.paganin.Paganin.
  3. tomo_writex fans the vchunk buffer to disk across --nbanks writers.

GPU-only 2-D FFT (same size class as step3_propagation.py).  For UPS ≥ 8
the (NPGNCHUNK, NZ, N) complex64 buffer no longer fits on a 40 GB GPU;
swap to step7_paganin_large.py which uses the host-chunked PaganinLarge
(3-pass streaming x-FFT / y-FFT+H·mult / x-IFFT, one strip at a time).

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step7_paganin.py \\
        --ups 1 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.paganin import Paganin
from iohdf5.dxchange_hdf5_chunks import tomo_writex, read_projs_vchunkx
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
)
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=1,
                   help="matches step3/step5 --ups (drives paths)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/model_big{UPS}x/stitched.h5, "
                        "writes {path}/model_big{UPS}x/paganin.h5")
    p.add_argument("--energy",     type=float, default=30.0,   help="keV")
    p.add_argument("--voxelsize",  type=float, default=1.38e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",   type=float, default=0.2,
                   help="sample → detector distance, meters (matches step3)")
    p.add_argument("--alpha",      type=float, default=5e-4,
                   help="Tikhonov regularisation added to T² in the filter")
    p.add_argument("--npgnchunk",  type=int, default=8,
                   help="angles per Paganin batch (== per-GPU 2-D FFT batch size)")
    p.add_argument("--nbanks",     type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--ntasks",     type=int, default=8,
                   help="parallel workers for read_projs_vchunkx (stitched prefetch)")
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for paganin.h5 (default: 8·NPGNCHUNK, NZ, N)")
    p.add_argument("--chunk-order", choices=("sino", "proj"), default="sino",
                   help="HDF5 chunk order inside paganin.h5's bank files.  "
                        "'sino' (default) = (θ_per_bank, 1, N), what step8's "
                        "z-slab FBP read wants.  'proj' = (1, NZ, N), the old "
                        "layout — makes that read touch every chunk to use "
                        "zslab/NZ of it.  Banking is θ-split either way.")
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

# HDF5 chunk shape inside the bank files.  Banking stays θ-split (stype
# 'proj') because ranks shard on θ and each bank file must have exactly one
# writer — but the chunks are laid out for the reader.  step8 reads
# (NTHETA, zslab, N) sinograms, so a chunk spanning one z row and a whole
# bank's worth of angles is read whole; the old (1, NZ, N) projection chunk
# was read NZ/zslab-fold over.  θ_per_bank is what tomo_initx's banking plan
# puts in each bank file, so this covers a chunk exactly.
THETA_PER_BANK = (VCHUNKS[0] + NBANKS - 1) // NBANKS
H5CHUNKS = (THETA_PER_BANK, 1, N) if _A.chunk_order == "sino" else (1, NZ, N)


def main() -> None:
    from mpi_utils import banner
    banner("7", f"stitched.h5 -> paganin.h5  "
                f"(single-distance Paganin, α={ALPHA})")

    if VCHUNKS[0] % NPGNCHUNK != 0:
        raise SystemExit(
            f"--vchunks C0={VCHUNKS[0]} must be a multiple of "
            f"--npgnchunk={NPGNCHUNK}.")
    if VCHUNKS[1] != NZ or VCHUNKS[2] != N:
        raise SystemExit(
            f"--vchunks C1×C2 must equal NZ×N ({NZ}×{N}).  "
            f"Got {VCHUNKS[1]}×{VCHUNKS[2]}.")

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

    rprint(f"UPS={UPS}  nz={NZ}  n={N}  n_half={N_HALF}  npgnchunk={NPGNCHUNK}")
    rprint(f"paganin: E={ENERGY} keV  λ={wavelength:.4e} m  "
           f"voxel={VOXELSIZE} m  distance={DISTANCE} m  α={ALPHA}")
    rprint(f"GPU est. — Paganin._buf + filt: "
           f"{(NPGNCHUNK * NZ * N + NZ * N) * 8 / 1e9:.3f} GB")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(DST_H5, (N_HALF, NZ, N), np.float32,
                        VCHUNKS, "proj", NBANKS, chunks=H5CHUNKS)

    ctx = initx_and_bcast(DST_H5, shape=(N_HALF, NZ, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=RANK, comm=COMM, chunks=H5CHUNKS)
    if RANK == 0:
        with h5py.File(DST_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((N_HALF, NZ, N), VCHUNKS)}")

    cl = Paganin(N, NZ, NPGNCHUNK,
                 wavelength=wavelength, voxelsize=VOXELSIZE, distance=DISTANCE,
                 alpha=ALPHA)

    p_min, p_max = np.inf, -np.inf
    p_sum, p_cnt = 0.0, 0
    p_has_nan    = False

    t_read = t_comp = t_write = 0.0
    b_read = b_write = 0

    ivchunks = list(iter_vchunks((N_HALF, NZ, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)

    # Prefetch shm for the vchunkx θ-slab (VCHUNKS[0], NZ, N).
    # stitched.h5 is proj-stored so this is an ALIGNED read — parallel
    # workers each read their own θ-shard (aligned chunk access).
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
                intens_d = cp.asarray(slab_h[tb0 - t0_vc : tb1 - t0_vc])
                phase_d  = cl.retrieve(intens_d)
                del intens_d
                phase_batch_h = cp.asnumpy(phase_d)
                del phase_d
                cp.get_default_memory_pool().free_all_blocks()
                t_comp += time.perf_counter() - t0

                p_min = min(p_min, float(phase_batch_h.min()))
                p_max = max(p_max, float(phase_batch_h.max()))
                p_sum += float(phase_batch_h.sum())
                p_cnt += b * NZ * N
                if np.isnan(phase_batch_h).any():
                    p_has_nan = True

                buf[tb0 - t0_vc : tb1 - t0_vc] = phase_batch_h
                del phase_batch_h

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
