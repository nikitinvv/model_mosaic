#!/usr/bin/env python
"""Model detector intensities for the upsampled init volume.

Pipeline (matches rec_mpi.gen_sqrt_data(), with attenuation via BETA_RATIO):
    proj = R(delta)                              (linear; written to proj.h5)
    x22  = R(delta) / NORM_CONST                 (norm_const = sqrt(N/NTHETA))
    beta = x22 / BETA_RATIO                      (weak absorption)
    psi  = exp(1j·(x22 + 1j·beta)) = exp(-beta)·(cos + 1j·sin)
    data = |D_prop(psi)|²                        (parallel-beam Fresnel, → data.h5)

I/O uses per-rank bank files behind a top-level VDS master (see h5_banks.py).
Reads go through the master transparently.  Each rank writes only its own
bank via POSIX (no MPI-IO):
    {path}/big{UPS}x.h5         VDS master (input) — z-banked bank files
    {path}/model_big{UPS}x/
        proj.h5                 VDS master, z-banked (axis 1)
        proj/proj_data_*.h5     bank files, one per rank
        data.h5                 VDS master, θ-banked (axis 0)
        data/data_data_*.h5     bank files, one per rank

Two stages, streaming:
  1. RADON — z-chunks fanned across ranks; each rank writes into its own
             proj bank via `--accum-r` accumulator.
  2. FRESNEL — angle-batched; each rank writes into its own data bank via
               `--accum-f` accumulator.  proj reads go via VDS.

Multi-GPU via MPI (mpi4py optional).  GPU affinity via set_affinity_gpu.sh.
Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step2_model_big.py \\
        --ups 2 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import cupy as cp

from tomo import Tomo
from propagation import Propagation
from h5_mpi_slab import check_chunk_bytes
from h5_banks import BankedH5, Accumulator


# ---------- MPI (optional) --------------------------------------------------
try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    RANK  = _COMM.Get_rank()
    SIZE  = _COMM.Get_size()
except ImportError:
    MPI   = None
    _COMM = None
    RANK  = 0
    SIZE  = 1


def _barrier() -> None:
    if _COMM is not None:
        _COMM.Barrier()


def _allreduce(val, op):
    if _COMM is None:
        return val
    return _COMM.allreduce(val, op=op)


def rprint(*a, **k) -> None:
    if RANK == 0:
        k.setdefault("flush", True)
        print(*a, **k)


_H5_HAS_MPI = h5py.get_config().mpi
_H5_MPI_KW  = ({"driver": "mpio", "comm": _COMM}
               if _COMM is not None and _H5_HAS_MPI else {})


# ---------- CLI ------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="upsample factor (matches step1_upsample --ups)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/big{UPS}x.h5, writes {path}/model_big{UPS}x/")
    p.add_argument("--in-nz",  type=int, default=2560,  help="init nz (before UPS)")
    p.add_argument("--in-n",   type=int, default=2744,  help="init N  (before UPS)")
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 360°; default = 3·N/4")
    p.add_argument("--mask-r",     type=float, default=0.0,
                   help="soft circular mask radius (0 disables)")
    p.add_argument("--beta-ratio", type=float, default=100.0,
                   help="weak absorption: β = phase/beta_ratio")
    p.add_argument("--phase-scale", type=float, default=1.0,
                   help=">1 amplifies the phase to make Fresnel fringes visible")
    p.add_argument("--energy",    type=float, default=30.0, help="keV")
    p.add_argument("--voxelsize", type=float, default=1.4e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",  type=float, default=1.0,
                   help="sample → detector distance, meters")
    p.add_argument("--nzchunk",       type=int, default=8,
                   help="z-slices per Radon call")
    p.add_argument("--npropchunk",  type=int, default=8,
                   help="angles per Fresnel batch")
    p.add_argument("--n-load-threads", type=int, default=8)
    p.add_argument("--stage", choices=("both", "radon", "prop"), default="both")
    p.add_argument("--theta-batch", type=int, default=0,
                   help="angles per Tomo batch (0 = all in one)")
    p.add_argument("--nthetachunk", type=int, default=64,
                   help="θ-chunk size for proj.h5 (larger = fewer h5 chunks per "
                        "stage-1 write; too large amplifies stage-2 read of "
                        "NPROPCHUNK angles.  Chunk bytes = θchunk · NZCHUNK · N · 4)")
    p.add_argument("--accum-r", type=int, default=1,
                   help="radon z-chunks to accumulate before flushing to the "
                        "proj bank file (1 = flush every chunk).  Buffer/rank "
                        "= accum-r · NTHETA · NZCHUNK · N · 4 bytes.")
    p.add_argument("--accum-f", type=int, default=1,
                   help="fresnel batches to accumulate before flushing to the "
                        "data bank file (1 = flush every batch).  Buffer/rank "
                        "= accum-f · NPROPCHUNK · NZ · N · 4 bytes.")
    return p.parse_args()


_A = _parse_args()

# ---------- config from CLI -------------------------------------------------
UPS         = _A.ups
BASE_DIR    = _A.path
SRC_H5      = f"{BASE_DIR}/big{UPS}x.h5"
DST_DIR     = f"{BASE_DIR}/model_big{UPS}x"
PROJ_H5     = f"{DST_DIR}/proj.h5"
DATA_H5     = f"{DST_DIR}/data.h5"

NZ         = _A.in_nz * UPS
N          = _A.in_n  * UPS
NTHETA     = _A.ntheta if _A.ntheta is not None else 3 * N // 4
ANG_MAX    = 2 * np.pi
MASK_R     = _A.mask_r
BETA_RATIO = _A.beta_ratio

NORM_CONST  = np.float32(np.sqrt(N / NTHETA))
PHASE_SCALE = _A.phase_scale

ENERGY    = _A.energy
VOXELSIZE = _A.voxelsize
DISTANCE  = _A.distance

NZCHUNK          = _A.nzchunk
NPROPCHUNK     = _A.npropchunk
N_LOAD_THREADS  = _A.n_load_threads
STAGE           = _A.stage
NTHETACHUNK     = max(1, min(_A.nthetachunk, NTHETA))
ACCUM_R          = max(1, _A.accum_r)
ACCUM_F          = max(1, _A.accum_f)

THETA_BATCH = _A.theta_batch
if THETA_BATCH <= 0 or THETA_BATCH >= NTHETA:
    THETA_BATCH = NTHETA
N_THETA_BATCHES = (NTHETA + THETA_BATCH - 1) // THETA_BATCH


def load_chunk(src_dset, z_start: int, z_end: int) -> np.ndarray:
    """(k, N, N) float32 host read from big{UPS}x.h5's /exchange/data."""
    return src_dset[z_start:z_end, :, :].astype(np.float32, copy=False)


def _radon_bank_ranges():
    """z-partition for proj.h5 z-banking: contiguous NZCHUNK-aligned blocks
    per rank, matching _run_radon()'s own compute assignment."""
    n_chunks_total = (NZ + NZCHUNK - 1) // NZCHUNK
    per_rank = (n_chunks_total + SIZE - 1) // SIZE
    ranges = []
    for r in range(SIZE):
        lo = min(r * per_rank, n_chunks_total) * NZCHUNK
        hi = min(min(r * per_rank, n_chunks_total) + per_rank,
                 n_chunks_total) * NZCHUNK
        ranges.append((lo, min(hi, NZ)))
    return ranges


def _theta_bank_ranges():
    """θ-partition for data.h5 θ-banking: contiguous θ per rank, matching
    _run_propagation()'s per-rank assignment."""
    per_rank = (NTHETA + SIZE - 1) // SIZE
    return [(min(r * per_rank, NTHETA), min((r + 1) * per_rank, NTHETA))
            for r in range(SIZE)]


def main() -> None:
    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    _barrier()

    theta_rad = np.linspace(0.0, ANG_MAX, NTHETA, endpoint=False).astype("float32")
    theta_deg = np.rad2deg(theta_rad).astype("float32")

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)  h5 mpi={_H5_HAS_MPI}")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    _barrier()
    rprint(f"UPS={UPS}  nz={NZ} n={N} ntheta={NTHETA} nchunk={NZCHUNK}  "
           f"mask_r={MASK_R}  norm_const={float(NORM_CONST):.4g} "
           f"(applied at propagation)")
    rprint(f"src={SRC_H5}")
    rprint(f"proj={PROJ_H5}")
    rprint(f"data={DATA_H5}")
    rprint(f"GPU est. — Tomo._buf_fde: "
           f"{NZCHUNK * (2*N)**2 * 8 / 1e9:.1f} GB")

    if STAGE in {"both", "radon"}:
        _run_radon(theta_rad, theta_deg)
    else:
        rprint(f"STAGE={STAGE}: skipping Radon stage; assuming proj.h5 already on disk")
    _barrier()

    if STAGE == "radon":
        rprint("STAGE=radon: skipping propagation.")
        return

    _run_propagation(theta_deg)


def _run_radon(theta_rad: np.ndarray, theta_deg: np.ndarray) -> None:
    rprint(f"THETA_BATCH={THETA_BATCH}  n_theta_batches={N_THETA_BATCHES}  "
           f"(re-reads input volume {N_THETA_BATCHES}×)")

    proj_chunks = (NTHETACHUNK, NZCHUNK, N)
    check_chunk_bytes(proj_chunks, 4, label="proj.h5")

    z_ranges = _radon_bank_ranges()
    proj = BankedH5(PROJ_H5, shape=(NTHETA, NZ, N), dtype="float32",
                    axis=1, chunks=proj_chunks,
                    rank=RANK, size=SIZE, comm=_COMM,
                    bank_ranges=z_ranges)
    proj.create(extra_datasets={"theta": theta_deg})

    my_zlo, my_zhi = z_ranges[RANK]
    my_chunks = list(range(my_zlo // NZCHUNK,
                           (my_zhi + NZCHUNK - 1) // NZCHUNK))

    accum_bytes = ACCUM_R * NTHETACHUNK * NZCHUNK * N * 4  # radon batch=1 chunk
    rprint(f"proj.h5 VDS + {SIZE} bank files  (chunks={proj_chunks}, "
           f"{np.prod(proj_chunks)*4/1e6:.1f} MB/chunk; "
           f"{NTHETA * NZ * N * 4 / 1e12:.2f} TB total)   "
           f"accum-r={ACCUM_R}  buffer/rank≈{ACCUM_R * NTHETA * NZCHUNK * N * 4 / 1e9:.2f} GB")

    proj_min, proj_max = np.inf, -np.inf

    with h5py.File(SRC_H5, "r") as fsrc, \
         proj.open_writer() as proj_w:
        src_dset = fsrc["exchange/data"]

        for tb in range(N_THETA_BATCHES):
            tb0 = tb * THETA_BATCH
            tb1 = min(tb0 + THETA_BATCH, NTHETA)
            b_theta = theta_rad[tb0:tb1]
            rprint(f"[theta batch {tb+1}/{N_THETA_BATCHES}] angles [{tb0}, {tb1})")

            rprint("building Tomo (allocating buffers + cuFFT plans)...")
            cl_tomo = Tomo(N, NZCHUNK, b_theta, mask_r=MASK_R)
            rprint("Tomo ready.")

            # Accumulator batches ACCUM_R contiguous z-chunks along the
            # banking axis (dataset axis 1) before hitting disk.
            acc = Accumulator(proj_w, axis=1,
                              capacity=ACCUM_R * NZCHUNK,
                              dtype=np.float32)

            t_read = t_radon = t_write = 0.0

            for ci, chunk_idx in enumerate(my_chunks):
                z0 = chunk_idx * NZCHUNK
                z1 = min(z0 + NZCHUNK, NZ)
                k  = z1 - z0

                t0 = time.perf_counter()
                chunk_h = load_chunk(src_dset, z0, z1)
                if k < NZCHUNK:
                    pad = np.zeros((NZCHUNK, N, N), dtype=np.float32)
                    pad[:k] = chunk_h
                    chunk_h = pad
                t_read += time.perf_counter() - t0

                t0 = time.perf_counter()
                delta_d = cp.asarray(chunk_h)
                vol_d   = cp.empty(delta_d.shape, dtype=cp.complex64)
                vol_d.real = delta_d
                vol_d.imag = cp.float32(0)
                del delta_d

                proj_d_c = cl_tomo.R(vol_d)          # [len(b_theta), NZCHUNK, N]
                del vol_d

                proj_chunk_h = cp.asnumpy(proj_d_c[:, :k].real).astype(
                    np.float32, copy=False)
                del proj_d_c
                cp.get_default_memory_pool().free_all_blocks()
                t_radon += time.perf_counter() - t0

                # Accumulate into the RAM buffer; flush is automatic when
                # the buffer's z-capacity fills.
                t0 = time.perf_counter()
                acc.append((slice(tb0, tb1), slice(z0, z1), slice(None)),
                           proj_chunk_h)
                t_write += time.perf_counter() - t0

                proj_min = min(proj_min, float(proj_chunk_h.min()))
                proj_max = max(proj_max, float(proj_chunk_h.max()))
                del proj_chunk_h

                if (ci + 1) % 4 == 0 or (ci + 1) == len(my_chunks):
                    print(f"  [rank {RANK}] tb{tb+1}/{N_THETA_BATCHES}  "
                          f"chunk {ci+1}/{len(my_chunks)}  z={z1}", flush=True)

            # Flush any leftover in the accumulator before next θ-batch.
            t0 = time.perf_counter()
            acc.close()
            t_write += time.perf_counter() - t0

            print(f"  [rank {RANK}] radon timing tb{tb+1}: "
                  f"read={t_read:.1f}s radon={t_radon:.1f}s write={t_write:.1f}s",
                  flush=True)

            del cl_tomo
            cp.get_default_memory_pool().free_all_blocks()
            _barrier()

    if MPI is not None:
        proj_min = _allreduce(proj_min, MPI.MIN)
        proj_max = _allreduce(proj_max, MPI.MAX)
    _barrier()

    rprint(f"proj = R(delta) stats: min={proj_min:.4g} max={proj_max:.4g}  "
           f"(after scaling by 1/NORM_CONST={1.0/float(NORM_CONST):.4g}: "
           f"phase in [{proj_min/float(NORM_CONST):.4g}, "
           f"{proj_max/float(NORM_CONST):.4g}] rad)")


# --------------------- Stage 2: Fresnel propagation kernels -------------------
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


def _run_propagation(theta_deg: np.ndarray) -> None:
    wavelength = 1.24e-9 / ENERGY
    fresnel_number = (VOXELSIZE ** 2) / (wavelength * DISTANCE)

    rprint(f"prop: E={ENERGY} keV  lambda={wavelength:.4e} m  "
           f"voxel={VOXELSIZE} m  distance={DISTANCE} m  "
           f"Fresnel number (per pixel)={fresnel_number:.4g}")
    rprint(f"propagating full sinogram {NZ}×{N} (no crop)")
    rprint(f"GPU est. — Prop._buf_big + fker: "
           f"{(NPROPCHUNK + 1) * (2*NZ) * (2*N) * 8 / 1e9:.3f} GB  "
           f"(NPROPCHUNK={NPROPCHUNK})")

    data_chunks = (1, NZ, N)
    check_chunk_bytes(data_chunks, 4, label="data.h5")

    theta_ranges = _theta_bank_ranges()
    data = BankedH5(DATA_H5, shape=(NTHETA, NZ, N), dtype="float32",
                    axis=0, chunks=data_chunks,
                    rank=RANK, size=SIZE, comm=_COMM,
                    bank_ranges=theta_ranges)
    data.create(extra_datasets={"theta": theta_deg})
    i_start, i_end = theta_ranges[RANK]

    rprint(f"data.h5 VDS + {SIZE} bank files  (chunks={data_chunks}, "
           f"{np.prod(data_chunks)*4/1e6:.1f} MB/chunk; "
           f"{NTHETA * NZ * N * 4 / 1e12:.2f} TB total)   "
           f"accum-f={ACCUM_F}  buffer/rank≈{ACCUM_F * NPROPCHUNK * NZ * N * 4 / 1e9:.2f} GB")

    cl_prop = Propagation(N, NZ, NPROPCHUNK, 1,
                          wavelength, VOXELSIZE, [DISTANCE])

    d_min, d_max = np.inf, -np.inf
    d_sum, d_cnt = 0.0, 0
    d_has_nan    = False

    inv_norm       = np.float32(PHASE_SCALE / float(NORM_CONST))
    inv_beta_ratio = np.float32(1.0 / BETA_RATIO)
    rprint(f"PHASE_SCALE={PHASE_SCALE}  effective inv_norm={float(inv_norm):.4g}")

    t_read = t_prop = t_write = 0.0

    # Bootstrap needs the earlier BankedH5 for proj as a reader; reconstruct it.
    proj_ro = BankedH5(PROJ_H5, shape=(NTHETA, NZ, N), dtype="float32",
                       axis=1, chunks=(NTHETACHUNK, NZCHUNK, N),
                       rank=RANK, size=SIZE, comm=_COMM,
                       bank_ranges=_radon_bank_ranges())

    with proj_ro.open_reader() as proj_r, \
         data.open_writer() as data_w:
        # Accumulator batches ACCUM_F Fresnel batches along θ before flushing.
        acc = Accumulator(data_w, axis=0,
                          capacity=ACCUM_F * NPROPCHUNK,
                          dtype=np.float32)

        for i0 in range(i_start, i_end, NPROPCHUNK):
            i1 = min(i0 + NPROPCHUNK, i_end)
            b  = i1 - i0

            # Read the angle batch (b, NZ, N) via the VDS master; the
            # underlying banks are z-sharded and h5py stitches them.
            t0 = time.perf_counter()
            proj_batch_h = proj_r.read((slice(i0, i1), slice(None), slice(None)))
            t_read += time.perf_counter() - t0

            t0 = time.perf_counter()
            proj_d = cp.asarray(proj_batch_h)
            del proj_batch_h

            psi_d = _psi_from_proj(proj_d, inv_norm, inv_beta_ratio)
            del proj_d

            prop_d = cl_prop.D(psi_d, 0)
            del psi_d
            intens_d = _abs2_c64_to_f32(prop_d)
            del prop_d
            data_batch_h = cp.asnumpy(intens_d)
            del intens_d
            cp.get_default_memory_pool().free_all_blocks()
            t_prop += time.perf_counter() - t0

            d_min = min(d_min, float(data_batch_h.min()))
            d_max = max(d_max, float(data_batch_h.max()))
            d_sum += float(data_batch_h.sum())
            d_cnt += b * NZ * N
            if np.isnan(data_batch_h).any():
                d_has_nan = True

            t0 = time.perf_counter()
            acc.append((slice(i0, i1), slice(None), slice(None)),
                       data_batch_h)
            t_write += time.perf_counter() - t0
            del data_batch_h

            print(f"  [rank {RANK}] prop  angles {i0}..{i1-1}", flush=True)

        t0 = time.perf_counter()
        acc.close()
        t_write += time.perf_counter() - t0

    print(f"  [rank {RANK}] prop timing: "
          f"read={t_read:.1f}s prop={t_prop:.1f}s write={t_write:.1f}s",
          flush=True)

    if MPI is not None:
        d_min     = _allreduce(d_min,     MPI.MIN)
        d_max     = _allreduce(d_max,     MPI.MAX)
        d_sum     = _allreduce(d_sum,     MPI.SUM)
        d_cnt     = _allreduce(d_cnt,     MPI.SUM)
        d_has_nan = _allreduce(d_has_nan, MPI.LOR)
    _barrier()

    rprint(f"data stats: min={d_min:.4g} max={d_max:.4g} "
           f"mean={d_sum/d_cnt:.4g} nan={d_has_nan}")
    rprint(f"wrote {NTHETA} angles to {DATA_H5}")


if __name__ == "__main__":
    main()
