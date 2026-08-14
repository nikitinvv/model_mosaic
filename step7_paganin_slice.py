#!/usr/bin/env python
"""Option B variant of step7_paganin.py:

    READ from stitched.h5   — unchanged (θ-slab reads, chunk-aligned)
    WRITE to paganin.h5     — CHANGED: slice-stored, chunks (N_HALF, 1, N)

So step 8's z-slab read [:, z0:z0+K, :] becomes chunk-aligned (K chunks
touched, 100% used per chunk) instead of the current cross-axis pattern
that scans the whole file per inner read.

Design notes:

  · paganin.h5 becomes a PLAIN HDF5 file (no VDS+banks) with HDF5 chunks
    (N_HALF, 1, N).  Effective vchunk shape is (N_HALF, VC, N) where VC
    = --theta-slab default 8·NPGNCHUNK; VDS+banks isn't used because step 8
    already reads paganin.h5 via plain h5py, so parallel bank fan-out on
    the read side isn't in play — a plain slice-stored file is enough.

  · Step 7's compute still iterates θ-slabs (Paganin's 2-D FFT needs the
    full (NZ, N) plane per angle — we cannot restructure to z-slabs
    without either 48× redundant Paganin compute or MPI transpose).

  · The mismatch: compute outputs a (θ-slab, NZ, N) buffer per iteration,
    but paganin.h5's chunks are (N_HALF, 1, N).  We therefore cannot use
    tomo_writex (which needs vchunk-aligned buffers).  Instead we do
    plain h5py partial writes — HDF5 does read-modify-write on the
    z-row chunks automatically.

  · Concurrent writes: slice-stored chunks span all θ, so any partial-θ
    write from one rank touches every chunk.  Multiple ranks cannot RMW
    the same chunk simultaneously safely, so writes are SERIALIZED across
    ranks (rank 0 writes all its slabs, then rank 1, …).  Each rank
    holds all its computed slabs in RAM (~20 GB/rank at UPS=1) between
    the compute phase and the write phase.

Trade-off vs OLD (proj-stored paganin.h5, step 8 at NZCHUNK=1):
    step 7 write physical:  ~5.8 TB   (was 81 GB, +72×)
    step 8 read  physical:  ~81 GB    (was ~249 TB, −3072×)
    total physical I/O:     ~6 TB     (was ~249 TB, −40×)

Launch (identical to step7_paganin.py):
    mpirun -n <NGPU> set_affinity_gpu.sh python step7_paganin_slice.py \\
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
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=1,
                   help="matches step3/step5 --ups (drives paths)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/model_big{UPS}x/stitched.h5, "
                        "writes {path}/model_big{UPS}x/paganin.h5 (slice-stored)")
    p.add_argument("--energy",     type=float, default=30.0,   help="keV")
    p.add_argument("--voxelsize",  type=float, default=1.38e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",   type=float, default=0.2,
                   help="sample → detector distance, meters (matches step3)")
    p.add_argument("--alpha",      type=float, default=5e-4,
                   help="Tikhonov regularisation added to T² in the filter")
    p.add_argument("--npgnchunk",  type=int, default=8,
                   help="angles per Paganin batch (== per-GPU 2-D FFT batch size)")
    p.add_argument("--theta-slab", type=int, default=None,
                   help="θ per outer iteration on each rank "
                        "(default 8·npgnchunk).  Larger = fewer h5py opens, "
                        "more RAM per rank.")
    return p.parse_args()


_A = _parse_args()

UPS       = _A.ups
BASE_DIR  = _A.path
DST_DIR   = f"{BASE_DIR}/model_big{UPS}x"
SRC_H5    = f"{DST_DIR}/stitched.h5"
DST_H5    = f"{DST_DIR}/paganin.h5"

IN_NZ = IN_N = 3072            # init.h5 dims after step00
NZ    = IN_NZ * UPS
N     = IN_N  * UPS

ENERGY     = _A.energy
VOXELSIZE  = _A.voxelsize
DISTANCE   = _A.distance
ALPHA      = _A.alpha
NPGNCHUNK  = _A.npgnchunk
THETA_SLAB = _A.theta_slab if _A.theta_slab is not None else 8 * NPGNCHUNK


def main() -> None:
    from mpi_utils import banner
    banner("7", f"stitched.h5 -> paganin.h5  "
                f"(single-distance Paganin, α={ALPHA})  "
                f"[Option B — slice-stored output]")

    if THETA_SLAB % NPGNCHUNK != 0:
        raise SystemExit(
            f"--theta-slab={THETA_SLAB} must be a multiple of "
            f"--npgnchunk={NPGNCHUNK}.")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    wavelength = 1.24e-9 / ENERGY

    # ── read source header ────────────────────────────────────────────────
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

    # ── GPU affinity banner ──────────────────────────────────────────────
    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()

    rprint(f"UPS={UPS}  nz={NZ}  n={N}  n_half={N_HALF}  "
           f"npgnchunk={NPGNCHUNK}  theta_slab={THETA_SLAB}")
    rprint(f"paganin: E={ENERGY} keV  λ={wavelength:.4e} m  "
           f"voxel={VOXELSIZE} m  distance={DISTANCE} m  α={ALPHA}")
    rprint(f"GPU est. — Paganin._buf + filt: "
           f"{(NPGNCHUNK * NZ * N + NZ * N) * 8 / 1e9:.3f} GB")

    # ── Create plain slice-stored paganin.h5 (rank 0) ────────────────────
    # Chunk shape (N_HALF, 1, N) = one chunk per z-row × all θ × all x.
    # Total chunks = NZ.  Each chunk = N_HALF·N·4 bytes ≈ 27 MB at UPS=1.
    if RANK == 0:
        if os.path.exists(DST_H5):
            os.remove(DST_H5)
        rprint(f"  OUT: {DST_H5}  (plain HDF5, slice-stored)")
        rprint(f"       shape=({N_HALF}, {NZ}, {N}) float32  "
               f"total={N_HALF*NZ*N*4/1024**3:.1f} GB")
        rprint(f"       chunk=({N_HALF}, 1, {N})  "
               f"= {N_HALF*N*4/1024**2:.1f} MB × {NZ} z-row chunks")
        with h5py.File(DST_H5, "w", libver="latest") as f:
            g = f.create_group("exchange")
            g.create_dataset(
                "data",
                shape=(N_HALF, NZ, N),
                dtype=np.float32,
                chunks=(N_HALF, 1, N),      # slice-stored
            )
            g.create_dataset("theta", data=theta_deg)
    barrier()

    # ── Paganin GPU object ───────────────────────────────────────────────
    cl = Paganin(N, NZ, NPGNCHUNK,
                 wavelength=wavelength, voxelsize=VOXELSIZE, distance=DISTANCE,
                 alpha=ALPHA)

    # ── Shard θ-slabs across ranks ────────────────────────────────────────
    n_theta_slabs = (N_HALF + THETA_SLAB - 1) // THETA_SLAB
    my_slab_indices = list(range(RANK, n_theta_slabs, SIZE))
    rprint(f"total θ-slabs={n_theta_slabs}  (each rank owns "
           f"~{n_theta_slabs // SIZE} slabs)")

    # ── Metrics ──────────────────────────────────────────────────────────
    p_min, p_max = np.inf, -np.inf
    p_sum, p_cnt = 0.0, 0
    p_has_nan    = False
    t_read = t_comp = t_write = 0.0
    b_read = b_write = 0

    # ── Phase 1: compute all owned θ-slabs, buffer outputs in RAM ────────
    # RAM per rank ≈ len(my_slab_indices) · THETA_SLAB · NZ · N · 4
    #             ≈ (N_HALF/SIZE) · NZ · N · 4  bytes
    #             ≈ 20 GB per rank at UPS=1 (4 ranks, N_HALF=2304)
    my_slab_data = []   # list of (tb0, tb1, phase_h)

    for k, slab_i in enumerate(my_slab_indices, start=1):
        tb0 = slab_i * THETA_SLAB
        tb1 = min(tb0 + THETA_SLAB, N_HALF)

        # Read θ-slab from stitched.h5 (aligned — chunk axis is θ)
        t0 = time.perf_counter()
        with h5py.File(SRC_H5, "r") as fp:
            slab_h = fp["exchange/data"][tb0:tb1, :, :]  # (K, NZ, N)
        t_read += time.perf_counter() - t0
        b_read += (tb1 - tb0) * NZ * N * 4

        # Compute Paganin per NPGNCHUNK batch
        out_h = np.empty((tb1 - tb0, NZ, N), dtype=np.float32)
        for sub_t in range(tb0, tb1, NPGNCHUNK):
            sub_te = min(sub_t + NPGNCHUNK, tb1)
            b = sub_te - sub_t

            t0 = time.perf_counter()
            intens_d = cp.asarray(slab_h[sub_t - tb0 : sub_te - tb0])
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

            out_h[sub_t - tb0 : sub_te - tb0] = phase_batch_h
            del phase_batch_h

        del slab_h

        my_slab_data.append((tb0, tb1, out_h))
        print(f"  [rank {RANK}] compute {k}/{len(my_slab_indices)}  "
              f"θ=[{tb0},{tb1})  "
              f"(read={t_read:.1f}s comp={t_comp:.1f}s)", flush=True)

    barrier()
    rprint("compute done on all ranks; starting serialized writes…")

    # ── Phase 2: serialized writes ───────────────────────────────────────
    # Slice-stored chunks span ALL θ, so a partial-θ write is RMW.
    # Multiple ranks cannot safely RMW the same chunk concurrently, so
    # ranks take turns.  Each rank writes ALL its slabs in one file open
    # (lets the OS page cache amortize repeated chunk touches).
    for r in range(SIZE):
        if r == RANK:
            t0 = time.perf_counter()
            with h5py.File(DST_H5, "r+", libver="latest") as fp:
                dset = fp["exchange/data"]
                for tb0, tb1, out_h in my_slab_data:
                    dset[tb0:tb1, :, :] = out_h
                    b_write += (tb1 - tb0) * NZ * N * 4
            t_write += time.perf_counter() - t0
            print(f"  [rank {RANK}] wrote {len(my_slab_data)} slabs  "
                  f"({t_write:.1f}s)", flush=True)
        barrier()

    # Drop the RAM buffers.
    del my_slab_data

    # ── Reduce metrics + report ──────────────────────────────────────────
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
    rprint(f"wrote {N_HALF} angles to {DST_H5} "
           f"(slice-stored, ready for step 8 aligned reads)")


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
