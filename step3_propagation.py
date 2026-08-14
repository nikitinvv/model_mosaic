#!/usr/bin/env python
"""Fresnel propagation of the Radon projections to detector intensities.

Reads {path}/model_big{UPS}x/proj.h5 (VDS + banks from step2_radon.py or
step2_radon_large.py) and writes {path}/model_big{UPS}x/data.h5.

For each --vchunks super-chunk (θ_super, NZ, N) this rank owns:
  1. Read the θ-slab from proj.h5 via VDS (one big read).
  2. Loop NPROPCHUNK θ-batches through the Fresnel pipeline:
         psi  = exp(1j·(proj/NORM_CONST + 1j·(proj/(NORM_CONST·BETA_RATIO))))
         data = |D_prop(psi)|²
  3. tomo_writex fans the vchunk buffer to disk across --nbanks writers.

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step3_propagation.py \\
        --ups 2 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.propagation import Propagation
from iohdf5.dxchange_hdf5_chunks import tomo_writex, read_projs_vchunkx
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
)
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="upsample factor (matches step1_upsample --ups)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/model_big{UPS}x/proj.h5, "
                        "writes {path}/model_big{UPS}x/data.h5")
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 180°; default = 3·N/4")
    p.add_argument("--beta-ratio", type=float, default=100.0,
                   help="weak absorption: β = phase/beta_ratio")
    p.add_argument("--phase-scale", type=float, default=1.0,
                   help=">1 amplifies the phase to make Fresnel fringes visible")
    p.add_argument("--energy",    type=float, default=30.0, help="keV")
    p.add_argument("--voxelsize", type=float, default=1.38e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",  type=float, default=0.2,
                   help="sample → detector distance, meters")
    p.add_argument("--npropchunk", type=int, default=8,
                   help="angles per Fresnel batch")
    p.add_argument("--nbanks",     type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--ntasks",     type=int, default=8,
                   help="parallel workers for read_projs_vchunkx (proj prefetch)")
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for data.h5 (default: 8·NPROPCHUNK, NZ, N)")
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
PROJ_H5  = f"{DST_DIR}/proj.h5"
DATA_H5  = f"{DST_DIR}/data.h5"

IN_NZ = IN_N = 3072            # init.h5 dims after step00; UPS scales from here
NZ    = IN_NZ * UPS
N     = IN_N  * UPS
NTHETA     = _A.ntheta if _A.ntheta is not None else 3 * N // 4
BETA_RATIO = _A.beta_ratio
PHASE_SCALE = _A.phase_scale
NORM_CONST  = np.float32(np.sqrt(N / NTHETA))

ENERGY     = _A.energy
VOXELSIZE  = _A.voxelsize
DISTANCE   = _A.distance

NPROPCHUNK = _A.npropchunk
NBANKS      = _A.nbanks
NTASKS      = _A.ntasks
VCHUNKS = tuple(_A.vchunks) if _A.vchunks else (8 * NPROPCHUNK, NZ, N)


_psi_from_proj = cp.ElementwiseKernel(
    "float32 delta_raw, float32 inv_norm, float32 inv_beta_ratio",
    "complex64 psi",
    """
    float phase = delta_raw * inv_norm;
    float atten = expf(-phase * inv_beta_ratio);
    psi = complex<float>(atten * cosf(phase), atten * sinf(phase));
    """,
    "psi_from_proj",
)

_abs2_c64_to_f32 = cp.ElementwiseKernel(
    "complex64 z",
    "float32 out",
    "out = z.real() * z.real() + z.imag() * z.imag();",
    "abs2_c64",
)


def main() -> None:
    if VCHUNKS[0] % NPROPCHUNK != 0:
        raise SystemExit(
            f"--vchunks C0={VCHUNKS[0]} must be a multiple of "
            f"--npropchunk={NPROPCHUNK}.")
    if VCHUNKS[1] != NZ or VCHUNKS[2] != N:
        raise SystemExit(
            f"--vchunks C1×C2 must equal NZ×N ({NZ}×{N}).  "
            f"Got {VCHUNKS[1]}×{VCHUNKS[2]}.")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    wavelength = 1.24e-9 / ENERGY
    fresnel_number = (VOXELSIZE ** 2) / (wavelength * DISTANCE)

    # Pull theta from proj.h5 so we can attach it to data.h5 too.
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
    rprint(f"GPU est. — Prop._buf_big + fker: "
           f"{(NPROPCHUNK + 1) * (2*NZ) * (2*N) * 8 / 1e9:.3f} GB")

    if RANK == 0:
        describe_input(PROJ_H5)
        describe_output(DATA_H5, (NTHETA, NZ, N), np.float32,
                        VCHUNKS, "proj", NBANKS)

    ctx = initx_and_bcast(DATA_H5, shape=(NTHETA, NZ, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(DATA_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((NTHETA, NZ, N), VCHUNKS)}")

    cl_prop = Propagation(N, NZ, NPROPCHUNK, 1,
                          wavelength, VOXELSIZE, [DISTANCE])

    d_min, d_max = np.inf, -np.inf
    d_sum, d_cnt = 0.0, 0
    d_has_nan    = False

    inv_norm       = np.float32(PHASE_SCALE / float(NORM_CONST))
    inv_beta_ratio = np.float32(1.0 / BETA_RATIO)

    t_read = t_prop = t_write = 0.0
    b_read = b_write = 0

    ivchunks = list(iter_vchunks((NTHETA, NZ, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)

    # Prefetch shm for the vchunkx proj-slab (VCHUNKS[0], NZ, N).
    # read_projs_vchunkx fans the read across NTASKS workers (each doing
    # its own θ-shard slice on the VDS master) instead of one plain-h5py
    # handle.  proj.h5 is slice-stored so the read is still cross-axis
    # (amp per worker ≈ NTHETA/VCHUNKS[0]), but parallel wall-time.
    proj_slab_shape = (VCHUNKS[0], NZ, N)
    shm_slab, proj_slab_buf = alloc_shm(proj_slab_shape, np.float32)

    try:
        for k, ivc in enumerate(my_ivchunks, start=1):
            t0_vc = ivc[0] * VCHUNKS[0]
            t1_vc = min(t0_vc + VCHUNKS[0], NTHETA)
            buf.fill(0)

            t0 = time.perf_counter()
            read_projs_vchunkx(PROJ_H5, shm_slab, ntasks=NTASKS,
                               vchunksx=proj_slab_shape,
                               ivchunkx=(ivc[0], 0, 0))
            proj_slab_h = proj_slab_buf
            t_read += time.perf_counter() - t0
            b_read += (t1_vc - t0_vc) * NZ * N * 4

            for tb0 in range(t0_vc, t1_vc, NPROPCHUNK):
                tb1 = min(tb0 + NPROPCHUNK, t1_vc)
                b   = tb1 - tb0

                t0 = time.perf_counter()
                proj_d = cp.asarray(proj_slab_h[tb0 - t0_vc : tb1 - t0_vc])
                psi_d = _psi_from_proj(proj_d, inv_norm, inv_beta_ratio)
                del proj_d

                prop_d = cl_prop.D(psi_d, 0)
                del psi_d
                intens_d = _abs2_c64_to_f32(prop_d)
                del prop_d
                data_batch_h = cp.asnumpy(intens_d[:b])
                del intens_d
                cp.get_default_memory_pool().free_all_blocks()
                t_prop += time.perf_counter() - t0

                d_min = min(d_min, float(data_batch_h.min()))
                d_max = max(d_max, float(data_batch_h.max()))
                d_sum += float(data_batch_h.sum())
                d_cnt += b * NZ * N
                if np.isnan(data_batch_h).any():
                    d_has_nan = True

                buf[tb0 - t0_vc : tb1 - t0_vc] = data_batch_h
                del data_batch_h

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
        free_shm(shm_slab)

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
    from mpi_utils import run_main
    run_main(main)
