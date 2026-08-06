#!/usr/bin/env python
"""Model detector intensities for the upsampled init volume.

Pipeline (matches rec_mpi.gen_sqrt_data(), with attenuation via BETA_RATIO):
    proj = R(delta)                              (linear; stored raw on disk)
    x22  = R(delta) / NORM_CONST                 (norm_const = sqrt(N/NTHETA))
    beta = x22 / BETA_RATIO                      (weak absorption)
    psi  = exp(1j·(x22 + 1j·beta)) = exp(-beta)·(cos + 1j·sin)
    data = |D_prop(psi)|²                        (parallel-beam Fresnel)

Two stages, streaming (no big host arrays):

  Stage 1 — RADON, z-chunks fanned across MPI ranks (round-robin)
    For each z-chunk on the GPU: Radon → proj = R(delta) (float32).
    Result is scattered by angle into per-angle memmapped TIFFs on disk:
      proj_{i:05d}.tif  (NZ, N) float32

  Stage 2 — FRESNEL PROPAGATION, angle-batched
    For each NPROP_BATCH angles: read from disk, build psi on GPU,
    propagate via Propagation.D, take |·|², write data_{i:05d}.tif.

Multi-GPU via MPI (mpi4py, optional).  GPU affinity is delegated to the
launcher: wrap with set_affinity_gpu.sh so each rank sees one GPU via
CUDA_VISIBLE_DEVICES.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step1_model_big.py \\
        --ups 2 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import resource
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp
import tifffile

from tomo import Tomo
from propagation import Propagation


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


# ---------- CLI ------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="upsample factor (matches upsample_big.py --ups)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/big{UPS}x, writes {path}/model_big{UPS}x")
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
    p.add_argument("--nchunk",       type=int, default=8,
                   help="z-slices per Radon call")
    p.add_argument("--nprop-batch",  type=int, default=8,
                   help="angles per Fresnel batch")
    p.add_argument("--n-load-threads", type=int, default=8,
                   help="parallel disk-read threads")
    p.add_argument("--stage", choices=("both", "radon", "prop"), default="both")
    p.add_argument("--mmap-batch",  type=int, default=256,
                   help="max simultaneous memmapped proj files per rank")
    p.add_argument("--theta-batch", type=int, default=0,
                   help="angles per Tomo batch (0 = all in one)")
    return p.parse_args()


_A = _parse_args()

# ---------- config from CLI -------------------------------------------------
UPS         = _A.ups
BASE_DIR    = _A.path
SRC_DIR     = f"{BASE_DIR}/big{UPS}x"
DST_DIR     = f"{BASE_DIR}/model_big{UPS}x"

NZ         = _A.in_nz * UPS
N          = _A.in_n  * UPS
NTHETA     = _A.ntheta if _A.ntheta is not None else 3 * N // 4
ANG_MAX    = 2 * np.pi          # 360°
MASK_R     = _A.mask_r
BETA_RATIO = _A.beta_ratio

NORM_CONST  = np.float32(np.sqrt(N / NTHETA))
PHASE_SCALE = _A.phase_scale

# Fresnel (parallel beam)
ENERGY    = _A.energy
VOXELSIZE = _A.voxelsize
DISTANCE  = _A.distance

NCHUNK          = _A.nchunk
NPROP_BATCH     = _A.nprop_batch
N_LOAD_THREADS  = _A.n_load_threads
STAGE           = _A.stage
MMAP_BATCH      = _A.mmap_batch

THETA_BATCH = _A.theta_batch
if THETA_BATCH <= 0 or THETA_BATCH >= NTHETA:
    THETA_BATCH = NTHETA
N_THETA_BATCHES = (NTHETA + THETA_BATCH - 1) // THETA_BATCH

_FD_TARGET   = MMAP_BATCH * 4 + 256
_FD_ACHIEVED = _raise_fd_limit(_FD_TARGET)


def load_chunk(z_start: int, z_end: int, pool: ThreadPoolExecutor) -> np.ndarray:
    k = z_end - z_start
    buf = np.empty((k, N, N), dtype=np.float32)

    def _read(i: int) -> None:
        buf[i] = tifffile.imread(os.path.join(SRC_DIR, f"big_{z_start + i:05d}.tif"))

    list(pool.map(_read, range(k)))
    return buf


def main() -> None:
    os.makedirs(DST_DIR, exist_ok=True)
    # GPU affinity is set externally (set_affinity_gpu.sh → CUDA_VISIBLE_DEVICES).
    theta = np.linspace(0.0, ANG_MAX, NTHETA, endpoint=False).astype("float32")
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    _barrier()
    rprint(f"UPS={UPS}  nz={NZ} n={N} ntheta={NTHETA} nchunk={NCHUNK}  "
           f"mask_r={MASK_R}  norm_const={float(NORM_CONST):.4g} "
           f"(applied at propagation)")
    rprint(f"src={SRC_DIR}")
    rprint(f"dst={DST_DIR}")
    rprint(f"GPU est. — Tomo._buf_fde: "
           f"{NCHUNK * (2*N)**2 * 8 / 1e9:.1f} GB")

    proj_paths = [os.path.join(DST_DIR, f"proj_{i:05d}.tif")
                  for i in range(NTHETA)]

    if STAGE in {"both", "radon"}:
        _run_radon(theta, proj_paths)
    else:
        rprint(f"STAGE={STAGE}: skipping Radon stage; assuming proj_*.tif already on disk")
    _barrier()

    if STAGE == "radon":
        rprint("STAGE=radon: skipping propagation.")
        return

    _run_propagation(proj_paths)


def _run_radon(theta: np.ndarray, proj_paths: list[str]) -> None:
    rprint(f"fd limit: soft={_FD_ACHIEVED} (requested >= {_FD_TARGET}); "
           f"MMAP_BATCH={MMAP_BATCH}")
    rprint(f"THETA_BATCH={THETA_BATCH}  n_theta_batches={N_THETA_BATCHES}  "
           f"(re-reads input volume {N_THETA_BATCHES}×)")

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

                # complex64 obj with imag=0 (avoids a cupy strided-view bug
                # observed with float32 obj on some builds).
                delta_d = cp.asarray(chunk_h)
                vol_d   = cp.empty(delta_d.shape, dtype=cp.complex64)
                vol_d.real = delta_d
                vol_d.imag = cp.float32(0)
                del delta_d

                proj_d_c = cl_tomo.R(vol_d)               # [len(b_theta), NCHUNK, N]
                del vol_d

                proj_chunk_h = cp.asnumpy(proj_d_c[:, :k].real).astype(
                    np.float32, copy=False)
                del proj_d_c
                cp.get_default_memory_pool().free_all_blocks()

                # Scatter into per-angle memmaps.  Fanned over the pool so
                # open+write+flush run concurrently across MMAP_BATCH files.
                # Each thread holds one memmap for its scope, so concurrent
                # fd usage is bounded by pool size, not batch_ntheta.
                batch_ntheta = tb1 - tb0

                def _scatter_one(i: int) -> None:
                    mm = tifffile.memmap(proj_paths[tb0 + i], mode='r+')
                    mm[z0:z1] = proj_chunk_h[i]
                    mm.flush()

                list(pool.map(_scatter_one, range(batch_ntheta)))

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


# Fused kernel for stage 2:
#   x22   = R(delta) / norm_const
#   beta  = x22 / BETA_RATIO
#   psi   = exp(1j · (x22 + 1j · beta)) = exp(-beta) · (cos + 1j·sin)
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

    inv_norm       = np.float32(PHASE_SCALE / float(NORM_CONST))
    inv_beta_ratio = np.float32(1.0 / BETA_RATIO)
    rprint(f"PHASE_SCALE={PHASE_SCALE}  effective inv_norm={float(inv_norm):.4g}")

    with ThreadPoolExecutor(max_workers=max(NPROP_BATCH, 2)) as io_pool:
        for i0 in range(i_start, i_end, NPROP_BATCH):
            i1 = min(i0 + NPROP_BATCH, i_end)
            b  = i1 - i0

            proj_batch_h = np.zeros((b, NZ, N), dtype=np.float32)

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
    rprint(f"wrote {NTHETA} data tiffs to {DST_DIR}  "
           f"(intermediate proj_*.tif left in place)")


if __name__ == "__main__":
    main()
