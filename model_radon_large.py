#!/usr/bin/env python
"""Same pipeline as model_radon_big.py but uses the CHUNKED-ON-HOST Radon
implementation in ~/radon_large/tomo_large.py, which stages small pieces
of the padded frequency-domain buffer through the GPU and keeps the big
(nz, 2N, 2N) `fde` and (nz, ntheta, N) `sino` arrays on the HOST.  Peak
GPU memory becomes proportional to the chunk sizes, not to (2N)², so
much larger N can be modelled on a 40 GB GPU.

Trade-off vs. model_radon_big.py:
  - HOST memory per Tomo.R call: (fde + sino + res) ≈
    NCHUNK×(2N)²×8 + 2×NCHUNK×NTHETA_BATCH×N×8 bytes.
    At N=21952, NTHETA=14400, NCHUNK=1, THETA_BATCH=NTHETA:
        fde   ≈ 15.4 GB, sino ≈ 2.5 GB, res ≈ 2.5 GB → ~20 GB / rank.
  - GPU memory per R call: only the chunk stripes on device
    (obj0 + phi0 + fftshift + fde0 ≈ a few 100 MB depending on chunks).
  - Tomo is recreated per THETA_BATCH so the *host* sino/res shrink too.

The chunk sizes must divide the sizes they slice into:
  CHUNK_N        divides both N and 2N   (stripes across the xy axis)
  CHUNK_THETA    divides NTHETA and 2*NTHETA
  CHUNK_XY       divides 2N              (spatial gather-bin size)

Pipeline (identical to model_radon_big.py):
  x22  = R(delta) / NORM_CONST     (norm_const = sqrt(N/NTHETA))
  beta = x22 / BETA_RATIO          (weak absorption; BETA_RATIO=100)
  psi  = exp(1j · (x22 + 1j·beta)) = exp(-beta)·(cos + 1j·sin)
  data = |D_prop(psi)|²

Two stages, MPI multi-GPU (unchanged), memmap TIFF outputs, sparse
truncated bootstrap files, batched memmap opens (MMAP_BATCH).
"""
from __future__ import annotations

import os
import sys
import resource
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp
import tifffile


def _raise_fd_limit(target: int) -> int:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = max(target, soft)
        if hard > 0:
            want = min(want, hard)
        if want > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            soft = want
        return soft
    except Exception:
        return -1


# ---------- imports for chunked Tomo + Propagation ------------------------
sys.path.insert(0, os.path.expanduser("~/radon_large"))
from tomo_large import Tomo as TomoLarge

sys.path.insert(0, "/home/beams2/VNIKITIN/holotomocupy_mpi/src")
from holotomocupy.propagation import Propagation


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
        print(*a, **k)


# ---------- config -------------------------------------------------------
SRC_DIR = "/data2/brain_sym_mosaic/big8x"
DST_DIR = "/data2/brain_sym_mosaic/model_big8x"

NZ         = 10240 * 2                  # z voxels (input volume z-extent)
N          = 10976 * 2                  # in-plane voxel count (n)
NTHETA     = 7200 * 2                   # projection angles over 360°
ANG_MAX    = 2 * np.pi                  # 360°
ROTATION_AXIS = N / 2                   # centred (tomo_large stores; not used in R)

# rec_mpi-style obj-side normalisation, applied at propagation time.
NORM_CONST = np.float32(np.sqrt(N / NTHETA))
# beta = (delta/NORM_CONST) / BETA_RATIO; large => weak absorption.
# Matches model_radon_big.py.
BETA_RATIO = 100.0

# Fresnel (all lengths in meters).  Cone-beam geometry with source-to-sample
# distance Z1 and sample-to-detector distance Z2 modelled via the
# Fresnel-scaling equivalent parallel-beam geometry — same convention as
# model_radon_big.py:
#     VOXELSIZE = P_DET * Z1 / (Z1 + Z2)   (effective pixel at sample plane)
#     DISTANCE  = Z1 * Z2 / (Z1 + Z2)      (effective propagation distance)
ENERGY     = 30.0
Z1         = 40
Z2         = 1
P_DET      = 1.4e-6
VOXELSIZE  = P_DET * Z1 / (Z1 + Z2)          # ≈ 1.366 µm
DISTANCE   = Z1 * Z2 / (Z1 + Z2)             # ≈ 0.9756 m

NCHUNK          = int(os.environ.get("NCHUNK", "1"))          # z-slices per R call
NPROP_BATCH     = int(os.environ.get("NPROP_BATCH", "1"))
N_LOAD_THREADS  = int(os.environ.get("N_LOAD_THREADS", "8"))
STAGE           = os.environ.get("STAGE", "both").lower()

# Chunk sizes passed to TomoLarge.R (must divide the corresponding axis).
CHUNK_N     = int(os.environ.get("CHUNK_N",     "448"))       # divides N and 2N
CHUNK_THETA = int(os.environ.get("CHUNK_THETA", "480"))       # divides NTHETA and 2*NTHETA
CHUNK_XY    = int(os.environ.get("CHUNK_XY",    "448"))       # divides 2N

MMAP_BATCH  = int(os.environ.get("MMAP_BATCH", "256"))

# Angle batching: bounds host sino/res AND lets Tomo see fewer angles.
# 0 or >=NTHETA means single batch = all angles at once.
THETA_BATCH = int(os.environ.get("THETA_BATCH", "0"))
if THETA_BATCH <= 0 or THETA_BATCH >= NTHETA:
    THETA_BATCH = NTHETA
N_THETA_BATCHES = (NTHETA + THETA_BATCH - 1) // THETA_BATCH

if STAGE not in {"both", "radon", "prop"}:
    raise SystemExit(f"STAGE must be one of both|radon|prop, got {STAGE!r}")

_FD_TARGET   = MMAP_BATCH * 4 + 256
_FD_ACHIEVED = _raise_fd_limit(_FD_TARGET)


def _validate_chunks() -> None:
    problems = []
    if N % CHUNK_N or (2 * N) % CHUNK_N:
        problems.append(f"CHUNK_N={CHUNK_N} must divide both N={N} and 2N={2*N}")
    if THETA_BATCH % CHUNK_THETA or (2 * THETA_BATCH) % CHUNK_THETA:
        problems.append(f"CHUNK_THETA={CHUNK_THETA} must divide THETA_BATCH={THETA_BATCH} "
                        f"and 2*THETA_BATCH={2*THETA_BATCH}")
    if (2 * N) % CHUNK_XY:
        problems.append(f"CHUNK_XY={CHUNK_XY} must divide 2N={2*N}")
    if problems:
        raise SystemExit("chunk-size problems:\n  " + "\n  ".join(problems))


def _pick_gpu() -> int:
    if "GPU_ID" in os.environ:
        return int(os.environ["GPU_ID"])
    for k in ("LOCAL_RANK",
              "OMPI_COMM_WORLD_LOCAL_RANK",
              "MV2_COMM_WORLD_LOCAL_RANK",
              "MPI_LOCALRANKID",
              "SLURM_LOCALID"):
        if k in os.environ:
            return int(os.environ[k])
    return RANK % max(cp.cuda.runtime.getDeviceCount(), 1)

GPU_ID = _pick_gpu()


def load_chunk(z_start: int, z_end: int, pool: ThreadPoolExecutor) -> np.ndarray:
    """Return a (k, N, N) complex64 host array for the requested z-strip."""
    k = z_end - z_start
    buf = np.empty((k, N, N), dtype=np.complex64)

    def _read(i: int) -> None:
        im = tifffile.imread(os.path.join(SRC_DIR, f"big_{z_start + i:05d}.tif"))
        buf[i].real = im
        buf[i].imag = 0

    list(pool.map(_read, range(k)))
    return buf


# ============ main =========================================================
def main() -> None:
    _validate_chunks()
    os.makedirs(DST_DIR, exist_ok=True)
    cp.cuda.Device(GPU_ID).use()

    theta = np.linspace(0.0, ANG_MAX, NTHETA, endpoint=False).astype("float32")
    rprint(f"[MPI] size={SIZE}  ranks pinned round-robin over local GPUs")
    print(f"  rank {RANK}: gpu={GPU_ID}  "
          f"({cp.cuda.runtime.getDeviceProperties(GPU_ID)['name'].decode()})",
          flush=True)
    _barrier()
    rprint(f"nz={NZ} n={N} ntheta={NTHETA} nchunk={NCHUNK}  "
           f"norm_const={float(NORM_CONST):.4g} (applied at propagation)")
    rprint(f"theta (rad, over 360°): {theta}")
    rprint(f"chunks: CHUNK_N={CHUNK_N}  CHUNK_THETA={CHUNK_THETA}  CHUNK_XY={CHUNK_XY}")
    rprint(f"THETA_BATCH={THETA_BATCH}  n_theta_batches={N_THETA_BATCHES}  "
           f"(re-reads input volume {N_THETA_BATCHES}×)")
    fde_gb   = NCHUNK * (2*N)**2 * 8 / 1e9
    sino_gb  = NCHUNK * THETA_BATCH * N * 8 / 1e9
    rprint(f"HOST est. per R call: fde≈{fde_gb:.1f} GB, sino≈{sino_gb:.2f} GB, "
           f"res≈{sino_gb:.2f} GB")

    proj_paths = [os.path.join(DST_DIR, f"proj_{i:05d}.tif")
                  for i in range(NTHETA)]

    if STAGE in {"both", "radon"}:
        _run_radon(theta, proj_paths)
    else:
        rprint(f"STAGE={STAGE}: skipping Radon; assuming proj_*.tif already on disk")
    _barrier()

    if STAGE == "radon":
        rprint("STAGE=radon: skipping propagation.")
        return

    _run_propagation(proj_paths)


def _run_radon(theta: np.ndarray, proj_paths: list[str]) -> None:
    rprint(f"fd limit: soft={_FD_ACHIEVED} (requested >= {_FD_TARGET}); "
           f"MMAP_BATCH={MMAP_BATCH}")

    # Bootstrap: rank 0 creates + sparse-extends each proj file.
    expected_pix_bytes = NZ * N * 4
    if RANK == 0:
        for p in proj_paths:
            if os.path.exists(p):
                os.remove(p)
            mm = tifffile.memmap(p, shape=(NZ, N), dtype=np.float32,
                                 bigtiff=True)
            mm.flush()
            del mm
            with tifffile.TiffFile(p) as tf:
                pix_off = int(tf.pages[0].dataoffsets[0])
            need = pix_off + expected_pix_bytes
            if os.path.getsize(p) < need:
                os.truncate(p, need)
    _barrier()

    proj_min, proj_max = np.inf, -np.inf

    n_chunks_total = (NZ + NCHUNK - 1) // NCHUNK
    my_chunks = range(RANK, n_chunks_total, SIZE)

    chunks_arg = [CHUNK_N, CHUNK_THETA, CHUNK_XY]

    with ThreadPoolExecutor(max_workers=N_LOAD_THREADS) as pool:
        for tb in range(N_THETA_BATCHES):
            tb0 = tb * THETA_BATCH
            tb1 = min(tb0 + THETA_BATCH, NTHETA)
            b_theta = theta[tb0:tb1]
            rprint(f"[theta batch {tb+1}/{N_THETA_BATCHES}] angles [{tb0}, {tb1})")

            cl_tomo = TomoLarge(N, b_theta, ROTATION_AXIS)

            for ci, chunk_idx in enumerate(my_chunks):
                z0 = chunk_idx * NCHUNK
                z1 = min(z0 + NCHUNK, NZ)
                k  = z1 - z0
                chunk_h = load_chunk(z0, z1, pool)      # (k, N, N) complex64 host
                if k < NCHUNK:
                    pad = np.zeros((NCHUNK, N, N), dtype=np.complex64)
                    pad[:k] = chunk_h
                    chunk_h = pad

                # TomoLarge.R: (nz, N, N) complex64 host → (ntheta, nz, N) complex64 host
                res_h = cl_tomo.R(chunk_h, chunks_arg)
                del chunk_h

                # Take real part as float32 and drop any z-padding.
                proj_chunk_h = res_h[:, :k].real.astype(np.float32, copy=False)
                del res_h

                # Scatter into per-angle memmaps for this batch's slice.
                batch_ntheta = tb1 - tb0
                for b0 in range(0, batch_ntheta, MMAP_BATCH):
                    b1 = min(b0 + MMAP_BATCH, batch_ntheta)
                    mms = [tifffile.memmap(proj_paths[tb0 + i], mode='r+')
                           for i in range(b0, b1)]
                    for j, mm in enumerate(mms):
                        mm[z0:z1] = proj_chunk_h[b0 + j]
                    for mm in mms:
                        mm.flush()
                    del mms

                proj_min = min(proj_min, float(proj_chunk_h.min()))
                proj_max = max(proj_max, float(proj_chunk_h.max()))
                del proj_chunk_h

                if (ci + 1) % 32 == 0 or chunk_idx == n_chunks_total - 1:
                    print(f"  [rank {RANK}] tb{tb+1}/{N_THETA_BATCHES}  "
                          f"chunk {ci+1}/{len(my_chunks)}  z={z1}", flush=True)

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


# ============ stage 2 kernels + Fresnel prop (identical to model_radon_big) =
_psi_from_proj = cp.ElementwiseKernel(
    "float32 delta_raw, float32 inv_norm, float32 inv_beta_ratio",
    "complex64 psi",
    """
    float phase = delta_raw * inv_norm;             // = R(delta) / norm_const
    float atten = expf(-phase * inv_beta_ratio);    // weak absorption
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


def _run_propagation(proj_paths: list[str]) -> None:
    wavelength = 1.24e-9 / ENERGY
    fresnel_number = (VOXELSIZE ** 2) / (wavelength * DISTANCE)

    rprint(f"prop: E={ENERGY} keV  lambda={wavelength:.4e} m  "
           f"voxel={VOXELSIZE} m  distance={DISTANCE} m  "
           f"Fresnel number (per pixel)={fresnel_number:.4g}")
    rprint(f"propagating full sinogram {NZ}×{N} (no crop)")
    rprint(f"GPU est. — Prop._buf_big + fker: "
           f"{(NPROP_BATCH + 1) * (2*NZ) * (2*N) * 8 / 1e9:.3f} GB  "
           f"(NPROP_BATCH={NPROP_BATCH})")

    cl_prop = Propagation(N, NZ, NPROP_BATCH, 1,
                          wavelength, VOXELSIZE, [DISTANCE])

    per_rank = (NTHETA + SIZE - 1) // SIZE
    i_start  = min(RANK * per_rank, NTHETA)
    i_end    = min(i_start + per_rank, NTHETA)

    d_min, d_max = np.inf, -np.inf
    d_sum, d_cnt = 0.0, 0
    d_has_nan    = False

    inv_norm       = np.float32(1.0 / float(NORM_CONST))
    inv_beta_ratio = np.float32(1.0 / BETA_RATIO)

    with ThreadPoolExecutor(max_workers=max(NPROP_BATCH, 2)) as io_pool:
        for i0 in range(i_start, i_end, NPROP_BATCH):
            i1 = min(i0 + NPROP_BATCH, i_end)
            b  = i1 - i0

            proj_batch_h = np.zeros((NPROP_BATCH, NZ, N), dtype=np.float32)

            def _load(kk: int) -> None:
                proj_batch_h[kk] = tifffile.imread(proj_paths[i0 + kk])

            list(io_pool.map(_load, range(b)))

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

            d_min = min(d_min, float(data_batch_h.min()))
            d_max = max(d_max, float(data_batch_h.max()))
            d_sum += float(data_batch_h.sum())
            d_cnt += b * NZ * N
            if np.isnan(data_batch_h).any():
                d_has_nan = True

            def _write(kk: int) -> None:
                tifffile.imwrite(
                    os.path.join(DST_DIR, f"data_{i0 + kk:05d}.tif"),
                    data_batch_h[kk], compression=None,
                )

            list(io_pool.map(_write, range(b)))
            del data_batch_h

            print(f"  [rank {RANK}] prop  angles {i0}..{i1-1}", flush=True)

    if MPI is not None:
        d_min     = _allreduce(d_min,     MPI.MIN)
        d_max     = _allreduce(d_max,     MPI.MAX)
        d_sum     = _allreduce(d_sum,     MPI.SUM)
        d_cnt     = _allreduce(d_cnt,     MPI.SUM)
        d_has_nan = _allreduce(d_has_nan, MPI.LOR)
    _barrier()

    rprint(f"data stats: min={d_min:.4g} max={d_max:.4g} "
           f"mean={d_sum/d_cnt:.4g} nan={d_has_nan}")
    rprint(f"wrote {NTHETA} data tiffs to {DST_DIR}  "
           f"(intermediate proj_*.tif left in place)")


if __name__ == "__main__":
    main()
