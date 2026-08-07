#!/usr/bin/env python
"""Same pipeline as step2_model_big.py but uses the host-chunked Radon
implementation in tomo_large.py, which stages small pieces of the padded
frequency-domain buffer through the GPU and keeps the big (nz, 2N, 2N)
`fde` and (nz, ntheta, N) `sino` arrays on the HOST.  Peak GPU memory
becomes proportional to the chunk sizes, not to (2N)², so much larger N
can be modelled on a 40 GB GPU.

Pipeline (identical to step2_model_big.py):
    x22  = R(delta) / NORM_CONST     (norm_const = sqrt(N/NTHETA))
    beta = x22 / BETA_RATIO          (weak absorption; BETA_RATIO=100)
    psi  = exp(1j·(x22 + 1j·beta))   = exp(-beta)·(cos + 1j·sin)
    data = |D_prop(psi)|²

I/O uses single HDF5 files with parallel writes (h5py MPI-IO):
    {path}/big{UPS}x.h5         input volume
    {path}/model_big{UPS}x/proj.h5, data.h5    outputs

Chunk sizes passed to TomoLarge.R must divide the sizes they slice into:
  --chunk-n     divides N and 2N       (stripes across the xy axis)
  --chunk-theta divides NTHETA         (angle grouping)
  --chunk-xy    divides 2N             (spatial gather-bin size)

Defaults 686 / 343 / 686 are divisors of 2744 and 2058, so they work for
any integer --ups ≥ 1.

Multi-GPU via MPI, GPU affinity via set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step2_model_large.py \\
        --ups 8 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import cupy as cp

from tomo_large  import TomoLarge
from propagation import Propagation
from h5_mpi_slab import (
    check_chunk_bytes,
    mpiio_read_axis0,
    mpiio_write_axis0,
    mpiio_write_slab,
)


# ---------- MPI (optional) -----------------------------------------------
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


# ---------- CLI ----------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=8,
                   help="upsample factor (matches step1_upsample --ups)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/big{UPS}x.h5, writes {path}/model_big{UPS}x/")
    p.add_argument("--in-nz",  type=int, default=2560, help="init nz (before UPS)")
    p.add_argument("--in-n",   type=int, default=2744, help="init N  (before UPS)")
    p.add_argument("--ntheta", type=int, default=None,
                   help="angles over 360°; default = 3·N/4")
    p.add_argument("--beta-ratio", type=float, default=100.0,
                   help="β = phase/beta_ratio (weak absorption)")
    p.add_argument("--energy",    type=float, default=30.0, help="keV")
    p.add_argument("--voxelsize", type=float, default=1.4e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",  type=float, default=1.0,
                   help="sample → detector distance, meters")
    p.add_argument("--nzchunk",       type=int, default=1,
                   help="z-slices per Radon call")
    p.add_argument("--npropchunk",  type=int, default=1,
                   help="angles per Fresnel batch")
    p.add_argument("--n-load-threads", type=int, default=8)
    p.add_argument("--stage", choices=("both", "radon", "prop"), default="both")
    p.add_argument("--theta-batch", type=int, default=0,
                   help="angles per Tomo batch (0 = all in one)")
    p.add_argument("--chunk-n",     type=int, default=686)
    p.add_argument("--chunk-theta", type=int, default=343)
    p.add_argument("--chunk-xy",    type=int, default=686)
    p.add_argument("--nthetachunk", type=int, default=64,
                   help="θ-chunk size for proj.h5 (larger = fewer h5 chunks per "
                        "stage-1 write; too large amplifies stage-2 read of "
                        "NPROPCHUNK angles.  Chunk bytes = θchunk · NZCHUNK · N · 4)")
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
SRC_H5   = f"{BASE_DIR}/big{UPS}x.h5"
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
PROJ_H5  = f"{DST_DIR}/proj.h5"
DATA_H5  = f"{DST_DIR}/data.h5"

NZ         = _A.in_nz * UPS
N          = _A.in_n  * UPS
NTHETA     = _A.ntheta if _A.ntheta is not None else 3 * N // 4
ANG_MAX    = 2 * np.pi
ROTATION_AXIS = N / 2

NORM_CONST = np.float32(np.sqrt(N / NTHETA))
BETA_RATIO = _A.beta_ratio

ENERGY     = _A.energy
VOXELSIZE  = _A.voxelsize
DISTANCE   = _A.distance

NZCHUNK          = _A.nzchunk
NPROPCHUNK     = _A.npropchunk
N_LOAD_THREADS  = _A.n_load_threads
STAGE           = _A.stage

CHUNK_N     = _A.chunk_n
CHUNK_THETA = _A.chunk_theta
CHUNK_XY    = _A.chunk_xy
NTHETACHUNK  = max(1, min(_A.nthetachunk, NTHETA))

THETA_BATCH = _A.theta_batch
if THETA_BATCH <= 0 or THETA_BATCH >= NTHETA:
    THETA_BATCH = NTHETA
N_THETA_BATCHES = (NTHETA + THETA_BATCH - 1) // THETA_BATCH


def _validate_chunks() -> None:
    problems = []
    if N % CHUNK_N or (2 * N) % CHUNK_N:
        problems.append(f"--chunk-n={CHUNK_N} must divide N={N} and 2N={2*N}")
    if THETA_BATCH % CHUNK_THETA:
        problems.append(f"--chunk-theta={CHUNK_THETA} must divide THETA_BATCH={THETA_BATCH}")
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
           f"norm_const={float(NORM_CONST):.4g} (applied at propagation)")
    rprint(f"src={SRC_H5}")
    rprint(f"proj={PROJ_H5}")
    rprint(f"data={DATA_H5}")
    rprint(f"chunks: CHUNK_N={CHUNK_N}  CHUNK_THETA={CHUNK_THETA}  CHUNK_XY={CHUNK_XY}")
    rprint(f"THETA_BATCH={THETA_BATCH}  n_theta_batches={N_THETA_BATCHES}  "
           f"(re-reads input volume {N_THETA_BATCHES}×)")
    fde_gb  = NZCHUNK * (2*N)**2 * 8 / 1e9
    sino_gb = NZCHUNK * THETA_BATCH * N * 8 / 1e9
    rprint(f"HOST est. per R call: fde≈{fde_gb:.1f} GB, sino≈{sino_gb:.2f} GB, "
           f"res≈{sino_gb:.2f} GB")

    if STAGE in {"both", "radon"}:
        _run_radon(theta_rad, theta_deg)
    else:
        rprint(f"STAGE={STAGE}: skipping Radon; assuming proj.h5 already on disk")
    _barrier()

    if STAGE == "radon":
        rprint("STAGE=radon: skipping propagation.")
        return

    _run_propagation(theta_deg)


def _run_radon(theta_rad: np.ndarray, theta_deg: np.ndarray) -> None:
    # Bootstrap proj.h5.
    if RANK == 0 and os.path.exists(PROJ_H5):
        os.remove(PROJ_H5)
    _barrier()

    proj_chunks = (NTHETACHUNK, NZCHUNK, N)
    check_chunk_bytes(proj_chunks, 4, label="proj.h5")
    with h5py.File(PROJ_H5, "w", **_H5_MPI_KW) as f:
        g = f.create_group("exchange")
        g.create_dataset("data", shape=(NTHETA, NZ, N), dtype="float32",
                         chunks=proj_chunks)
        g.create_dataset("theta", data=theta_deg)
    _barrier()
    rprint(f"proj.h5 created  (chunks={proj_chunks}, "
           f"{np.prod(proj_chunks)*4/1e6:.1f} MB/chunk; "
           f"{NTHETA * NZ * N * 4 / 1e12:.2f} TB total)")

    proj_min, proj_max = np.inf, -np.inf

    # Contiguous z-chunk assignment so each h5 chunk is written by one rank.
    n_chunks_total = (NZ + NZCHUNK - 1) // NZCHUNK
    per_rank = (n_chunks_total + SIZE - 1) // SIZE
    my_chunk_lo = RANK * per_rank
    my_chunk_hi = min(my_chunk_lo + per_rank, n_chunks_total)
    my_chunks = list(range(my_chunk_lo, my_chunk_hi))

    chunks_arg = [CHUNK_N, CHUNK_THETA, CHUNK_XY]

    with h5py.File(SRC_H5,  "r",  **_H5_MPI_KW) as fsrc, \
         h5py.File(PROJ_H5, "r+", **_H5_MPI_KW) as fdst:
        src_dset  = fsrc["exchange/data"]
        proj_dset = fdst["exchange/data"]

        for tb in range(N_THETA_BATCHES):
            tb0 = tb * THETA_BATCH
            tb1 = min(tb0 + THETA_BATCH, NTHETA)
            b_theta = theta_rad[tb0:tb1]
            rprint(f"[theta batch {tb+1}/{N_THETA_BATCHES}] angles [{tb0}, {tb1})")

            rprint("building TomoLarge...")
            cl_tomo = TomoLarge(N, b_theta, ROTATION_AXIS)
            rprint("TomoLarge ready.")

            t_read = t_radon = t_write = 0.0

            for ci, chunk_idx in enumerate(my_chunks):
                z0 = chunk_idx * NZCHUNK
                z1 = min(z0 + NZCHUNK, NZ)
                k  = z1 - z0

                t0 = time.perf_counter()
                chunk_h = load_chunk(src_dset, z0, z1)
                if k < NZCHUNK:
                    pad = np.zeros((NZCHUNK, N, N), dtype=np.complex64)
                    pad[:k] = chunk_h
                    chunk_h = pad
                t_read += time.perf_counter() - t0

                # TomoLarge.R: (nz, N, N) complex64 host → (ntheta, nz, N) complex64 host
                t0 = time.perf_counter()
                res_h = cl_tomo.R(chunk_h, chunks_arg)
                del chunk_h
                proj_chunk_h = res_h[:, :k].real.astype(np.float32, copy=False)
                del res_h
                t_radon += time.perf_counter() - t0

                # Independent parallel write to disjoint z-chunk, slab-safe.
                t0 = time.perf_counter()
                mpiio_write_slab(proj_dset,
                                 (slice(tb0, tb1), slice(z0, z1), slice(None)),
                                 proj_chunk_h)
                t_write += time.perf_counter() - t0

                proj_min = min(proj_min, float(proj_chunk_h.min()))
                proj_max = max(proj_max, float(proj_chunk_h.max()))
                del proj_chunk_h

                if (ci + 1) % 4 == 0 or (ci + 1) == len(my_chunks):
                    print(f"  [rank {RANK}] tb{tb+1}/{N_THETA_BATCHES}  "
                          f"chunk {ci+1}/{len(my_chunks)}  z={z1}", flush=True)

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


# --------------------- Stage 2 kernels + Fresnel prop -----------------------
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
    rprint(f"GPU est. — Prop._buf_big + fker: "
           f"{(NPROPCHUNK + 1) * (2*NZ) * (2*N) * 8 / 1e9:.3f} GB  "
           f"(NPROPCHUNK={NPROPCHUNK})")

    if RANK == 0 and os.path.exists(DATA_H5):
        os.remove(DATA_H5)
    _barrier()

    # One full (NZ, N) plane per chunk — matches the per-angle Fresnel
    # write pattern exactly (each write touches one chunk).
    data_chunks = (1, NZ, N)
    check_chunk_bytes(data_chunks, 4, label="data.h5")
    with h5py.File(DATA_H5, "w", **_H5_MPI_KW) as f:
        g = f.create_group("exchange")
        g.create_dataset("data", shape=(NTHETA, NZ, N), dtype="float32",
                         chunks=data_chunks)
        g.create_dataset("theta", data=theta_deg)
    _barrier()
    rprint(f"data.h5 created  (chunks={data_chunks}, "
           f"{np.prod(data_chunks)*4/1e6:.1f} MB/chunk; "
           f"{NTHETA * NZ * N * 4 / 1e12:.2f} TB total)")

    cl_prop = Propagation(N, NZ, NPROPCHUNK, 1,
                          wavelength, VOXELSIZE, [DISTANCE])

    per_rank = (NTHETA + SIZE - 1) // SIZE
    i_start  = min(RANK * per_rank, NTHETA)
    i_end    = min(i_start + per_rank, NTHETA)

    d_min, d_max = np.inf, -np.inf
    d_sum, d_cnt = 0.0, 0
    d_has_nan    = False

    inv_norm       = np.float32(1.0 / float(NORM_CONST))
    inv_beta_ratio = np.float32(1.0 / BETA_RATIO)

    t_read = t_prop = t_write = 0.0

    with h5py.File(PROJ_H5, "r",  **_H5_MPI_KW) as fp, \
         h5py.File(DATA_H5, "r+", **_H5_MPI_KW) as fd:
        proj_dset = fp["exchange/data"]
        data_dset = fd["exchange/data"]

        for i0 in range(i_start, i_end, NPROPCHUNK):
            i1 = min(i0 + NPROPCHUNK, i_end)
            b  = i1 - i0

            t0 = time.perf_counter()
            proj_batch_h = mpiio_read_axis0(proj_dset, i0, i1)
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

            t0 = time.perf_counter()
            mpiio_write_axis0(data_dset, i0, i1, data_batch_h)
            t_write += time.perf_counter() - t0
            del data_batch_h

            print(f"  [rank {RANK}] prop  angles {i0}..{i1-1}", flush=True)

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
