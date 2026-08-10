#!/usr/bin/env python
"""Fresnel propagation of the Radon projections to detector intensities
using the host-chunked PropagationLarge — same output format as
step3_fresnel.py but the padded (2NZ × 2N) Fresnel work buffer lives on
the HOST, only per-strip pieces on the GPU.  Use this for UPS ≥ 8 (or
whenever the (NPROPCHUNK, 2NZ, 2N) complex64 buffer no longer fits on a
40 GB GPU).

Reads {path}/model_big{UPS}x/proj.h5 (VDS + banks) and writes
{path}/model_big{UPS}x/data.h5.

For each --data-vchunks super-chunk (θ_super, NZ, N) this rank owns:
  1. Read the θ-slab from proj.h5 via VDS (one big read).
  2. Loop NPROPCHUNK θ-batches through PropagationLarge (HOST pre/post):
         psi_h  = exp(1j·proj/NORM + …)                (numpy, host)
         prop_h = cl_prop_large.D(psi_h, 0, chunks)     (host-staged GPU)
         data_h = |prop_h|²                             (numpy, host)
  3. tomo_writex fans the vchunk buffer to disk across --nbanks writers.

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step3_fresnel_large.py \\
        --ups 8 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.propagation_large import PropagationLarge
from processing.chunk_pick import pick_prop_chunks
from iohdf5.dxchange_hdf5_chunks import tomo_writex
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
)
from utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=8,
                   help="upsample factor (matches step1_upsample --ups)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic")
    p.add_argument("--in-nz",  type=int, default=3072, help="init nz (before UPS)")
    p.add_argument("--in-n",   type=int, default=3072, help="init N  (before UPS)")
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 360°; default = 3·N/4")
    p.add_argument("--beta-ratio", type=float, default=100.0,
                   help="weak absorption: β = phase/beta_ratio")
    p.add_argument("--phase-scale", type=float, default=1.0,
                   help=">1 amplifies the phase to make Fresnel fringes visible")
    p.add_argument("--energy",    type=float, default=30.0, help="keV")
    p.add_argument("--voxelsize", type=float, default=1.38e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",  type=float, default=1.0,
                   help="sample → detector distance, meters")
    p.add_argument("--npropchunk", type=int, default=1,
                   help="angles per Fresnel batch (each D() call takes "
                        "NPROPCHUNK angles; default 1 = one angle at a time, "
                        "which minimises host RAM for very large NZ·N)")
    p.add_argument("--nbanks",     type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--chunk-nz", type=int, default=0,
                   help="PropagationLarge x-strip depth; 0 = auto from --gpu-budget-gb")
    p.add_argument("--chunk-2n", type=int, default=0,
                   help="PropagationLarge y-strip width; 0 = auto from --gpu-budget-gb")
    p.add_argument("--gpu-budget-gb", type=float, default=30.0,
                   help="target GPU memory for chunk-picker (default 30 GB, "
                        "leaves headroom on a 40 GB A100 for cuFFT plans + "
                        "transient allocations across pipeline stages)")
    p.add_argument("--data-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for data.h5 (default: 8·NPROPCHUNK, NZ, N)")
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
PROJ_H5  = f"{DST_DIR}/proj.h5"
DATA_H5  = f"{DST_DIR}/data.h5"

NZ         = _A.in_nz * UPS
N          = _A.in_n  * UPS
NTHETA     = _A.ntheta if _A.ntheta is not None else 3 * N // 4
BETA_RATIO = _A.beta_ratio
PHASE_SCALE = _A.phase_scale
NORM_CONST  = np.float32(np.sqrt(N / NTHETA))

ENERGY     = _A.energy
VOXELSIZE  = _A.voxelsize
DISTANCE   = _A.distance

NPROPCHUNK = _A.npropchunk
NBANKS      = _A.nbanks
DATA_VCHUNKS = tuple(_A.data_vchunks) if _A.data_vchunks else (8 * NPROPCHUNK, NZ, N)

_auto_nz, _auto_2n = pick_prop_chunks(
    nz=NZ, n=N, ntheta=NPROPCHUNK,
    gpu_budget_bytes=int(_A.gpu_budget_gb * 1e9))
CHUNK_NZ = _A.chunk_nz or _auto_nz
CHUNK_2N = _A.chunk_2n or _auto_2n


def _validate_chunks() -> None:
    problems = []
    if NZ % CHUNK_NZ:
        problems.append(f"--chunk-nz={CHUNK_NZ} must divide NZ={NZ}")
    if (2 * N) % CHUNK_2N:
        problems.append(f"--chunk-2n={CHUNK_2N} must divide 2N={2*N}")
    if problems:
        raise SystemExit("chunk-size problems:\n  " + "\n  ".join(problems))


def main() -> None:
    _validate_chunks()
    if DATA_VCHUNKS[0] % NPROPCHUNK != 0:
        raise SystemExit(
            f"--data-vchunks C0={DATA_VCHUNKS[0]} must be a multiple of "
            f"--npropchunk={NPROPCHUNK}.")
    if DATA_VCHUNKS[1] != NZ or DATA_VCHUNKS[2] != N:
        raise SystemExit(
            f"--data-vchunks C1×C2 must equal NZ×N ({NZ}×{N}).  "
            f"Got {DATA_VCHUNKS[1]}×{DATA_VCHUNKS[2]}.")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    wavelength = 1.24e-9 / ENERGY
    fresnel_number = (VOXELSIZE ** 2) / (wavelength * DISTANCE)

    if RANK == 0:
        with h5py.File(PROJ_H5, "r") as f:
            theta_deg = f["exchange/theta"][:]
    else:
        theta_deg = None
    theta_deg = COMM.bcast(theta_deg, root=0)

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()

    rprint(f"UPS={UPS}  nz={NZ} n={N} ntheta={NTHETA}  npropchunk={NPROPCHUNK}")
    rprint(f"prop: E={ENERGY} keV  lambda={wavelength:.4e} m  "
           f"voxel={VOXELSIZE} m  distance={DISTANCE} m  "
           f"Fresnel number (per pixel)={fresnel_number:.4g}")
    rprint(f"norm_const={float(NORM_CONST):.4g}  phase_scale={PHASE_SCALE}  "
           f"beta_ratio={BETA_RATIO}")
    rprint(f"chunks: CHUNK_NZ={CHUNK_NZ}  CHUNK_2N={CHUNK_2N}  "
           f"(auto from gpu-budget={_A.gpu_budget_gb} GB)")
    host_pass2_gb = NPROPCHUNK * 2 * NZ * CHUNK_2N * 8 / 1e9
    rprint(f"host-side psi/fde ≈ {NPROPCHUNK * NZ * N * 8 / 1e9:.1f} + "
           f"{NPROPCHUNK * NZ * 2 * N * 8 / 1e9:.1f} GB per D() call")

    if RANK == 0:
        describe_input(PROJ_H5)
        describe_output(DATA_H5, (NTHETA, NZ, N), np.float32,
                        DATA_VCHUNKS, "proj", NBANKS)

    ctx = initx_and_bcast(DATA_H5, shape=(NTHETA, NZ, N),
                          dtype=np.float32, vchunks=DATA_VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(DATA_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    buf_gb = vchunk_bytes(DATA_VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((NTHETA, NZ, N), DATA_VCHUNKS)}")

    cl_prop = PropagationLarge(N, NZ, wavelength, VOXELSIZE, [DISTANCE])

    d_min, d_max = np.inf, -np.inf
    d_sum, d_cnt = 0.0, 0
    d_has_nan    = False

    inv_norm       = np.float32(PHASE_SCALE / float(NORM_CONST))
    inv_beta_ratio = np.float32(1.0 / BETA_RATIO)

    t_read = t_prop = t_write = 0.0
    b_read = b_write = 0

    ivchunks = list(iter_vchunks((NTHETA, NZ, N), DATA_VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(DATA_VCHUNKS, np.float32)

    try:
        with h5py.File(PROJ_H5, "r") as fp:
            proj_dset = fp["exchange/data"]

            for k, ivc in enumerate(my_ivchunks, start=1):
                t0_vc = ivc[0] * DATA_VCHUNKS[0]
                t1_vc = min(t0_vc + DATA_VCHUNKS[0], NTHETA)
                buf.fill(0)

                t0 = time.perf_counter()
                proj_slab_h = proj_dset[t0_vc:t1_vc, :, :]  # (K, NZ, N) float32
                t_read += time.perf_counter() - t0
                b_read += (t1_vc - t0_vc) * NZ * N * 4

                # Pinned psi input reused across sub-batches — one alloc per
                # iteration instead of one fresh np.empty per NPROPCHUNK batch.
                psi_h_pinned = cl_prop.psi_buffer(NPROPCHUNK)

                for tb0 in range(t0_vc, t1_vc, NPROPCHUNK):
                    tb1 = min(tb0 + NPROPCHUNK, t1_vc)
                    b   = tb1 - tb0

                    t0 = time.perf_counter()
                    # HOST-side psi = exp(1j·(proj/NORM + 1j·(proj/(NORM·β))))
                    # Written elementwise via numpy — for very large NZ·N this
                    # avoids a GPU round-trip on the psi build itself.
                    proj_batch = proj_slab_h[tb0 - t0_vc : tb1 - t0_vc]
                    phase = proj_batch * inv_norm
                    atten = np.exp(-phase * inv_beta_ratio).astype(np.float32)
                    psi_h_pinned.real[:b] = atten * np.cos(phase)
                    psi_h_pinned.imag[:b] = atten * np.sin(phase)
                    if b < NPROPCHUNK:
                        psi_h_pinned.real[b:] = 0
                        psi_h_pinned.imag[b:] = 0
                    del phase, atten

                    # Fresnel propagate through PropagationLarge (host-staged GPU)
                    prop_h = cl_prop.D(psi_h_pinned, 0, [CHUNK_NZ, CHUNK_2N])[:b]

                    # |prop|² on host into a float32 batch
                    data_batch_h = (prop_h.real * prop_h.real
                                    + prop_h.imag * prop_h.imag).astype(
                        np.float32, copy=False)
                    del prop_h
                    t_prop += time.perf_counter() - t0

                    d_min = min(d_min, float(data_batch_h.min()))
                    d_max = max(d_max, float(data_batch_h.max()))
                    d_sum += float(data_batch_h.sum())
                    d_cnt += b * NZ * N
                    if np.isnan(data_batch_h).any():
                        d_has_nan = True

                    buf[tb0 - t0_vc : tb1 - t0_vc] = data_batch_h
                    del data_batch_h

                del proj_slab_h

                t0 = time.perf_counter()
                tomo_writex(DATA_H5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
                t_write += time.perf_counter() - t0
                b_write += (t1_vc - t0_vc) * NZ * N * 4

                print(f"  [rank {RANK}] vchunk {k}/{len(my_ivchunks)}  "
                      f"θ=[{t0_vc},{t1_vc})  "
                      f"(read={t_read:.1f}s prop={t_prop:.1f}s "
                      f"write={t_write:.1f}s)", flush=True)
    finally:
        free_shm(shm)

    d_min     = allreduce(d_min,     MPI.MIN)
    d_max     = allreduce(d_max,     MPI.MAX)
    d_sum     = allreduce(d_sum,     MPI.SUM)
    d_cnt     = allreduce(d_cnt,     MPI.SUM)
    d_has_nan = allreduce(d_has_nan, MPI.LOR)
    barrier()

    report_stage("step3 read (proj)",  b_read,  t_read)
    report_stage("step3 write (data)", b_write, t_write)

    rprint(f"data stats: min={d_min:.4g} max={d_max:.4g} "
           f"mean={d_sum/max(d_cnt,1):.4g} nan={d_has_nan}")
    rprint(f"wrote {NTHETA} angles to {DATA_H5}")


if __name__ == "__main__":
    from utils import run_main
    run_main(main)
