#!/usr/bin/env python
"""Model detector intensities for the 4×-upsampled init volume (from
upsample_big.py).

  input volume  : 10240 z × 10976 y × 10976 x  float32, per-slice TIFFs
  volume voxel  : 2.8 µm (= 2 × detector pixel of 1.4 µm)
  sample inside : ⌀10240 × 10240 cylinder (28.67 × 28.67 mm)
  angles        : 16 over 360°
  pipeline      : matches rec_mpi.gen_sqrt_data() exactly —
                    x22  = R(delta) / NORM_CONST    (norm_const=sqrt(N/NTHETA))
                    beta = x22 / BETA_RATIO         (weak absorption; BETA_RATIO=100)
                    psi  = exp(1j·(x22 + 1j·beta))  = exp(-beta)·(cos + 1j·sin)
                    data = |D_prop(psi)|²           (30 keV, 1.4 µm det pixel)
                  Radon is linear so proj = R(delta) is stored raw on disk
                  and NORM_CONST is applied at propagation time.

Two-stage design (no big host arrays; psi never lives fully in RAM):

  Stage 1 — RADON, streamed z-chunks
    For each z-chunk on the GPU: Radon → proj = R(delta) (real float32).
    The chunk's (NTHETA, k, N) result is scattered by angle into
    per-angle memmapped TIFFs on disk:  proj_{i:05d}.tif  (NZ, N) f32.

  Stage 2 — FRESNEL PROPAGATION, batched by angle
    For each NPROP_BATCH angles: read a DET_UP_NZ × DET_UP_N central
    window (default 2048×2048) from each proj_*.tif via TiffFile.slice,
    build psi = exp(1j · proj / NORM_CONST) on GPU, propagate via
    Propagation.D at that crop size, take |·|², write
    data_{i:05d}.tif (f32).  psi is not saved.

Multi-GPU via MPI (mpi4py, optional — falls back to single-rank if it
can't be imported):

  Stage 1  Z-chunks partitioned round-robin across ranks; each rank
           opens all NTHETA proj_*.tif memmaps but writes only its own
           disjoint z-strips (rank 0 creates the files first; barrier).

  Stage 2  Angles partitioned contiguously; each rank handles its own
           range with no shared writes.

  Launch   mpirun -n <NGPU> --map-by ppr:1:socket python model_radon_big.py
           (or a Slurm/OMPI equivalent that sets LOCAL_RANK).

Memory (per GPU/rank, 48 GB RTX 8000, ≥16 GB host):
  Tomo._buf_fde   NCHUNK × (2N)² × 8 B ≈ NCHUNK × 3.86 GB   (NCHUNK=1 → 3.86 GB)
  Prop._buf_big   BATCH  × 2NZ × 2N × 8 B ≈ BATCH × 3.60 GB (full sinogram; +3.6 GB fker)
  host peak       ~NPROP_BATCH × NZ × N × 12 B ≈ 5.4 GB     (BATCH=4)
"""
from __future__ import annotations

import os
import sys
import resource
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as _gpu_gaussian_filter
import tifffile


def _raise_fd_limit(target: int) -> int:
    """Best-effort bump of RLIMIT_NOFILE to at least `target` (never above the
    hard limit).  Returns the resulting soft limit."""
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

sys.path.insert(0, "/home/beams2/VNIKITIN/holotomocupy_mpi/src")
from holotomocupy.tomo import Tomo
from holotomocupy.propagation import Propagation


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
        print(*a, **k)


# ---------- config ---------------------------------------------------------
SRC_DIR = "/data2/brain_sym_mosaic/big1x"       # 10240 × 10976 × 10976 slices, Y350a UPS=4
DST_DIR = "/data2/brain_sym_mosaic/model_big1x"

NZ         = 2560                # = 10240
N          = 2744                # = 10976
NTHETA     = 16                # roughly half of the previous 7200 (matches ratio to N)
ANG_MAX    = 2 * np.pi          # 360°
MASK_R     = 0.0
BETA_RATIO = 100.0               # beta = (delta/NORM_CONST) / BETA_RATIO; large => weak abs

# Same obj-side normalization that rec_mpi.gen_sqrt_data() applies before
# fwd_tomo (line 778: `vars["obj"] /= self.norm_const`).  Radon is linear
# so we store proj = R(delta) on disk and divide by NORM_CONST at
# propagation time.  norm_const = sqrt(nobj / ntheta)  (rec_mpi.py:94).
NORM_CONST = np.float32(np.sqrt(N / NTHETA))
# Optional Stage-2 phase multiplier (=1 gives the strict gen_data-style
# phase R(δ)/NORM_CONST; >1 amplifies to make Fresnel fringes visible).
PHASE_SCALE = float(os.environ.get("PHASE_SCALE", "1.0"))
# Anti-alias Gaussian applied per-angle to the (z, s) sinogram plane
# BEFORE Fresnel.  The Fresnel kernel is isotropic in the continuous
# Fourier plane, but on the Cartesian grid the highest |f| samples live
# at the four corners (±fx_Nyq, ±fy_Nyq), where the kernel phase is
# maximal.  Any residual Nyquist content in psi gets amplified there and
# shows up as 4-fold cross / diagonal fringes in |D(psi)|².  Band-limit
# psi's spectrum before the FFT to suppress this.
# =0 disables.  Typical: 0.5–1.5 (reconstruction pixels).
SINO_AA_SIGMA = float(os.environ.get("SINO_AA_SIGMA", "0.0"))

# Fresnel (all lengths in meters).  Cone-beam geometry with source-to-sample
# distance Z1 and sample-to-detector distance Z2 is modelled with the
# Fresnel-scaling equivalent parallel-beam geometry:
#     VOXELSIZE = P_det * Z1 / (Z1 + Z2)     (effective pixel at sample plane)
#     DISTANCE  = Z1 * Z2 / (Z1 + Z2)        (effective propagation distance)
# Magnification M = (Z1 + Z2) / Z1.
ENERGY     = 30.0
Z1         = 40
Z2         = 1
P_DET      = 1.4e-6
VOXELSIZE  = P_DET * Z1 / (Z1 + Z2)          # ≈ 2.568 µm
DISTANCE   = Z1 * Z2 / (Z1 + Z2)             # ≈ 0.4938 m

NCHUNK          = int(os.environ.get("NCHUNK", "8"))
NPROP_BATCH     = int(os.environ.get("NPROP_BATCH", "8"))
N_LOAD_THREADS  = int(os.environ.get("N_LOAD_THREADS", "8"))
STAGE           = os.environ.get("STAGE", "both").lower()  # both|radon|prop
# Cap on how many proj_*.tif memmaps we open at once per rank in stage 1
# to stay under RLIMIT_NOFILE (default 1024).  Each chunk writes to all
# NTHETA angles in slices of MMAP_BATCH so at most that many fds are open.
MMAP_BATCH      = int(os.environ.get("MMAP_BATCH", "256"))
# Propagation crop: read only a DET_UP_NZ × DET_UP_N central window of
# each proj_*.tif and propagate that.  Bounds Prop._buf_big / fker and
# keeps FFT sizes power-of-2 for cuFFT.
DET_UP_N   = int(os.environ.get("DET_UP_N",  "2048"))   # x width  (2N padded → 4096)
DET_UP_NZ  = int(os.environ.get("DET_UP_NZ", "2048"))   # z height (2NZ padded → 4096)
# Angle batching for stage 1 to bound Tomo._buf_sino + 1D FFT plan (both
# scale linearly with ntheta).  0 or >=NTHETA means "single batch = all
# angles at once".  Each batch reads the input volume once (I/O multiplies
# by num_batches) but shrinks _buf_sino to (batch × NCHUNK × N) complex64.
THETA_BATCH     = int(os.environ.get("THETA_BATCH", "0"))
if THETA_BATCH <= 0 or THETA_BATCH >= NTHETA:
    THETA_BATCH = NTHETA
N_THETA_BATCHES = (NTHETA + THETA_BATCH - 1) // THETA_BATCH

if STAGE not in {"both", "radon", "prop"}:
    raise SystemExit(f"STAGE must be one of both|radon|prop, got {STAGE!r}")

# NTHETA can be much larger than the default 1024 fd limit; try to bump
# to accommodate MMAP_BATCH + tifffile/mpi/stdio slack.
_FD_TARGET   = MMAP_BATCH * 4 + 256
_FD_ACHIEVED = _raise_fd_limit(_FD_TARGET)

# Pick a GPU for this rank.  Priority:
#   1) explicit GPU_ID env
#   2) LOCAL_RANK / OMPI_COMM_WORLD_LOCAL_RANK / SLURM_LOCALID (MPI launcher)
#   3) RANK % <visible device count>
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
    k = z_end - z_start
    buf = np.empty((k, N, N), dtype=np.float32)

    def _read(i: int) -> None:
        buf[i] = tifffile.imread(os.path.join(SRC_DIR, f"big_{z_start + i:05d}.tif"))

    list(pool.map(_read, range(k)))
    return buf


def main() -> None:
    os.makedirs(DST_DIR, exist_ok=True)
    cp.cuda.Device(GPU_ID).use()

    theta = np.linspace(0.0, ANG_MAX, NTHETA, endpoint=False).astype("float32")
    rprint(f"[MPI] size={SIZE}  ranks pinned round-robin over local GPUs")
    print(f"  rank {RANK}: gpu={GPU_ID}  ({cp.cuda.runtime.getDeviceProperties(GPU_ID)['name'].decode()})",
          flush=True)
    _barrier()
    rprint(f"nz={NZ} n={N} ntheta={NTHETA} nchunk={NCHUNK}  "
           f"mask_r={MASK_R}  norm_const={float(NORM_CONST):.4g} "
           f"(applied at propagation)")
    rprint(f"theta (rad, over 360°): {theta}")
    rprint(f"GPU est. — Tomo._buf_fde: "
           f"{NCHUNK * (2*N)**2 * 8 / 1e9:.1f} GB")

    proj_paths = [os.path.join(DST_DIR, f"proj_{i:05d}.tif")
                  for i in range(NTHETA)]

    # =========== Stage 1: real Radon ⇒ stream proj to per-angle TIFFs =====
    if STAGE not in {"both", "radon"}:
        rprint(f"STAGE={STAGE}: skipping Radon stage; assuming proj_*.tif already on disk")
    else:
        _run_radon(theta, proj_paths)
    _barrier()

    if STAGE == "radon":
        rprint("STAGE=radon: skipping propagation.")
        return

    # =========== Stage 2: read proj tifs ⇒ propagate ⇒ write data tifs ====
    _run_propagation(proj_paths)


def _run_radon(theta: np.ndarray, proj_paths: list[str]) -> None:
    rprint(f"fd limit: soft={_FD_ACHIEVED} (requested >= {_FD_TARGET}); "
           f"MMAP_BATCH={MMAP_BATCH}")
    rprint(f"THETA_BATCH={THETA_BATCH}  n_theta_batches={N_THETA_BATCHES}  "
           f"(re-reads input volume {N_THETA_BATCHES}×)")

    # Rank 0 creates the (fresh) memmapped files ONE AT A TIME so we never
    # spike above the fd limit even for NTHETA in the thousands.
    #
    # tifffile.memmap() writes the BigTIFF header but does not always
    # extend the file to include the full pixel-data area on disk, which
    # makes a subsequent numpy.memmap fail with "mmap length is greater
    # than file size".  We fix that by truncating each file to
    # (pixel_offset + NZ*N*4) bytes.  Sparse extension: no data written.
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

    # Round-robin sharding of z-chunks across ranks (chunk_idx % SIZE == RANK).
    n_chunks_total = (NZ + NCHUNK - 1) // NCHUNK
    my_chunks = range(RANK, n_chunks_total, SIZE)

    # Outer loop: angle batches.  For each batch we construct a smaller
    # Tomo (ntheta = batch size), run the full z-sweep against it, then
    # tear it down before the next batch.  This bounds _buf_sino and the
    # 1-D FFT plan at the cost of re-reading the input volume once per batch.
    with ThreadPoolExecutor(max_workers=N_LOAD_THREADS) as pool:
        for tb in range(N_THETA_BATCHES):
            tb0 = tb * THETA_BATCH
            tb1 = min(tb0 + THETA_BATCH, NTHETA)
            b_theta = theta[tb0:tb1]
            rprint(f"[theta batch {tb+1}/{N_THETA_BATCHES}] angles [{tb0}, {tb1})")

            cl_tomo = Tomo(N, NCHUNK, b_theta, mask_r=MASK_R)

            for ci, chunk_idx in enumerate(my_chunks):
                z0 = chunk_idx * NCHUNK
                z1 = min(z0 + NCHUNK, NZ)
                k  = z1 - z0
                chunk_h = load_chunk(z0, z1, pool)
                if k < NCHUNK:
                    pad = np.zeros((NCHUNK, N, N), dtype=np.float32)
                    pad[:k] = chunk_h
                    chunk_h = pad

                # Give Tomo.R a complex64 volume (imag=0 → equivalent to beta=0).
                # Float32 obj triggers a cupy multiply-into-strided-complex64-view
                # bug on this build (fills only NCHUNK/2 z-slices).
                #
                # The obj-side NORM_CONST scaling used by rec_mpi.BH() /
                # gen_sqrt_data is applied in stage 2 (Radon is linear:
                # R(delta / NORM_CONST) = R(delta) / NORM_CONST), so proj on
                # disk stays as R(delta).
                delta_d = cp.asarray(chunk_h)
                vol_d   = cp.empty(delta_d.shape, dtype=cp.complex64)
                vol_d.real = delta_d
                vol_d.imag = cp.float32(0)                # beta = 0
                del delta_d

                proj_d_c = cl_tomo.R(vol_d)               # [len(b_theta), NCHUNK, N]
                del vol_d

                proj_chunk_h = cp.asnumpy(proj_d_c[:, :k].real).astype(
                    np.float32, copy=False)               # (len(b_theta), k, N) f32
                del proj_d_c
                cp.get_default_memory_pool().free_all_blocks()

                # Scatter into the per-angle memmaps for this batch's slice.
                # We open at most MMAP_BATCH files at a time so the fd count
                # stays well under RLIMIT_NOFILE.  passing shape/dtype to
                # tifffile.memmap triggers the WRITE path (which would
                # truncate our pre-sized files), so we only pass the path.
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

    # Merge stats across ranks.
    if MPI is not None:
        proj_min = _allreduce(proj_min, MPI.MIN)
        proj_max = _allreduce(proj_max, MPI.MAX)
    _barrier()

    rprint(f"proj = R(delta) stats: min={proj_min:.4g} max={proj_max:.4g}  "
           f"(after scaling by 1/NORM_CONST={1.0/float(NORM_CONST):.4g}: "
           f"phase in [{proj_min/float(NORM_CONST):.4g}, "
           f"{proj_max/float(NORM_CONST):.4g}] rad)")


# Fused kernel for stage 2:
#   x22   = R(delta) / norm_const            (obj-side normalization)
#   beta  = x22 / BETA_RATIO                 (weak absorption)
#   psi   = exp( 1j * (x22 + 1j * beta) )
#         = exp(-beta) * (cos(x22) + 1j * sin(x22))
# BETA_RATIO=∞ (== inv_beta_ratio=0) recovers the pure-phase gen_data F2.
_psi_from_proj = cp.ElementwiseKernel(
    "float32 delta_raw, float32 inv_norm, float32 inv_beta_ratio",
    "complex64 psi",
    """
    float phase = delta_raw * inv_norm;   // = R(delta) / norm_const
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

    # Propagator sized to the full sinogram.
    cl_prop = Propagation(N, NZ, NPROP_BATCH, 1,
                          wavelength, VOXELSIZE, [DISTANCE])

    # Contiguous angle sharding: rank r owns [i_start, i_end).
    per_rank = (NTHETA + SIZE - 1) // SIZE
    i_start  = min(RANK * per_rank, NTHETA)
    i_end    = min(i_start + per_rank, NTHETA)

    d_min, d_max = np.inf, -np.inf
    d_sum, d_cnt = 0.0, 0
    d_has_nan    = False

    inv_norm       = np.float32(PHASE_SCALE / float(NORM_CONST))
    inv_beta_ratio = np.float32(1.0 / BETA_RATIO)
    rprint(f"PHASE_SCALE={PHASE_SCALE}  effective inv_norm={float(inv_norm):.4g}")

    with ThreadPoolExecutor(max_workers=max(NPROP_BATCH, 2)) as io_pool:
        for i0 in range(i_start, i_end, NPROP_BATCH):
            i1 = min(i0 + NPROP_BATCH, i_end)
            b  = i1 - i0

            # Full-plane read (no crop). Size to the actual batch count b
            # so downstream psi_d / padded_d shapes match on the final
            # (possibly smaller) batch after MPI angle sharding.
            proj_batch_h = np.zeros((b, NZ, N), dtype=np.float32)

            def _load(kk: int) -> None:
                proj_batch_h[kk] = tifffile.imread(proj_paths[i0 + kk])

            list(io_pool.map(_load, range(b)))

            proj_d = cp.asarray(proj_batch_h)
            del proj_batch_h

            # Band-limit the sinogram before Fresnel to kill Nyquist-band
            # content that the isotropic Fresnel kernel would amplify at
            # the (fx_Nyq, fy_Nyq) corners → 4-fold diagonal artifacts.
            if SINO_AA_SIGMA > 0.0:
                proj_d = _gpu_gaussian_filter(
                    proj_d, sigma=(0.0, SINO_AA_SIGMA, SINO_AA_SIGMA),
                    mode="nearest")

            # x22 = R(delta)/norm_const, beta = x22/BETA_RATIO,
            # psi = exp(1j·(x22 + 1j·beta)) = exp(-beta)·(cos + 1j·sin)
            psi_d = _psi_from_proj(proj_d, inv_norm, inv_beta_ratio)
            del proj_d

            # Fresnel propagation via Propagation.D
            prop_d = cl_prop.D(psi_d, 0)
            del psi_d
            intens_d = _abs2_c64_to_f32(prop_d)
            del prop_d
            data_batch_h = cp.asnumpy(intens_d)
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
    rprint(f"wrote {NTHETA} psi + {NTHETA} data tiffs to {DST_DIR}  "
           f"(intermediate proj_*.tif left in place)")


if __name__ == "__main__":
    main()
