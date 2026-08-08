#!/usr/bin/env python
"""Extract a centred sub-volume from a 3-D reconstruction TIFF, apply a
soft-edged circular mask (and optional 3-D sharpen + smoothing), and save
as a VDS+banks HDF5 store at {path}/init.h5 (see test_h5_buffer_io.py for
the layout).

    {path}/init.h5              VDS master
    {path}/init/init_data_*.h5  nvchunks·nbanks bank files
        /exchange/data   (OUT_NZ, OUT_NYX, OUT_NYX) float32
                         chunks (1, OUT_NYX, OUT_NYX)

Vchunk (super-chunk) shape --init-vchunks controls the RAM buffer per
rank; nbanks-per-vchunk parallel POSIX writers fan the buffer to disk.
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import tifffile

from iohdf5.dxchange_hdf5_chunks import tomo_writex
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks,
)

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as _cp_gaussian_filter

from utils import (
    COMM as _COMM, RANK as _RANK, SIZE as _SIZE,
    barrier as _barrier, report_stage,
)


# ---------- CLI ----------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=(
        "/data3/vnikitin/brain_rec/20251115/Y350a1234/"
        "4500_2048_0_0.0_0.003_0.05_0.02_20_1.1_0/rec_obj_real/0096.tiff"),
        help="source 3-D reconstruction TIFF (multi-page)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
        help="base output directory; init.h5 goes to {path}/init.h5")
    p.add_argument("--src-nz",  type=int, default=3264, help="source nz (pages)")
    p.add_argument("--src-nyx", type=int, default=3264, help="source ny=nx")
    p.add_argument("--out-nz",  type=int, default=2560, help="output total z")
    p.add_argument("--out-nyx", type=int, default=2744, help="output y=x")
    p.add_argument("--circle-diam", type=int, default=2560,
                   help="mask diameter in output voxels")
    p.add_argument("--z-pad",   type=int, default=50,
                   help="zero-padding on each end of z (air region)")
    p.add_argument("--z-taper", type=int, default=15,
                   help="cosine taper into the padded ends (px)")
    p.add_argument("--mask-taper", type=float, default=7.0,
                   help="cosine taper on the circular xy mask (px)")
    p.add_argument("--sharpen-amount", type=float, default=0.0,
                   help="unsharp-mask amount; 0 disables")
    p.add_argument("--sharpen-sigma", type=float, default=1.0,
                   help="unsharp-mask Gaussian σ (px)")
    p.add_argument("--smooth-sigma", type=float, default=0.0,
                   help="post-smoothing Gaussian σ (px); 0 disables")
    p.add_argument("--n-threads", type=int, default=32,
                   help="threads per rank for I/O + separable filter")
    p.add_argument("--chunk-z", type=int, default=64,
                   help="z-slices processed per compute chunk")
    p.add_argument("--nbanks", type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--init-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk shape for init.h5 (default: "
                        "--chunk-z, OUT_NYX, OUT_NYX = one compute chunk per vchunk)")
    return p.parse_args()


_A = _parse_args()

SRC_TIFF      = _A.src
DST_H5        = f"{_A.path}/init.h5"
SRC_NZ        = _A.src_nz
SRC_NY = SRC_NX = _A.src_nyx
OUT_NZ        = _A.out_nz
OUT_NYX       = _A.out_nyx
CIRCLE_DIAM   = _A.circle_diam
Z_PAD         = _A.z_pad
Z_TAPER       = _A.z_taper
MASK_TAPER    = _A.mask_taper
SHARPEN_AMOUNT = _A.sharpen_amount
SHARPEN_SIGMA  = _A.sharpen_sigma
SMOOTH_SIGMA   = _A.smooth_sigma
N_THREADS     = _A.n_threads
CHUNK_Z       = _A.chunk_z
NBANKS        = _A.nbanks
INIT_VCHUNKS  = tuple(_A.init_vchunks) if _A.init_vchunks else (CHUNK_Z, _A.out_nyx, _A.out_nyx)

SAMPLE_NZ = OUT_NZ - 2 * Z_PAD
if SAMPLE_NZ <= 0:
    raise SystemExit(f"--z-pad {Z_PAD} leaves no sample slices (--out-nz {OUT_NZ}).")

Z0 = (SRC_NZ - SAMPLE_NZ) // 2
Y0 = (SRC_NY - OUT_NYX) // 2
X0 = (SRC_NX - OUT_NYX) // 2

cp.cuda.Device(_RANK % cp.cuda.runtime.getDeviceCount()).use()

# Zero-copy memmap of the source TIFF (uncompressed contiguous pages).
try:
    _SRC = tifffile.memmap(SRC_TIFF, mode='r')
except Exception:
    _SRC = None
_tiff_local = threading.local()


def _read_src_plane(z: int) -> np.ndarray:
    if _SRC is not None:
        return _SRC[z]
    tf = getattr(_tiff_local, 'tf', None)
    if tf is None:
        tf = tifffile.TiffFile(SRC_TIFF)
        _tiff_local.tf = tf
    return tf.pages[z].asarray()


def _z_weight(z_out: int) -> float:
    if z_out < Z_PAD or z_out >= Z_PAD + SAMPLE_NZ:
        return 0.0
    d = min(z_out - Z_PAD, Z_PAD + SAMPLE_NZ - 1 - z_out)
    if d >= Z_TAPER:
        return 1.0
    t = (d + 0.5) / Z_TAPER
    return float(0.5 - 0.5 * np.cos(np.pi * t))


def make_mask() -> np.ndarray:
    r = CIRCLE_DIAM / 2.0
    c = (OUT_NYX - 1) / 2.0
    y, x = np.ogrid[:OUT_NYX, :OUT_NYX]
    rho = np.sqrt((y - c) ** 2 + (x - c) ** 2)
    if MASK_TAPER <= 0:
        return (rho <= r).astype(np.float32)
    t = np.clip((r - rho) / MASK_TAPER + 0.5, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def _read_masked(z_out: int, mask: np.ndarray) -> np.ndarray:
    w = _z_weight(z_out)
    if w == 0.0:
        return np.zeros((OUT_NYX, OUT_NYX), dtype=np.float32)
    z_src = (z_out - Z_PAD) + Z0
    im = _read_src_plane(z_src)
    plane = im[Y0:Y0 + OUT_NYX, X0:X0 + OUT_NYX].astype(np.float32,
                                                        copy=False) * mask
    if w < 1.0:
        plane = plane * np.float32(w)
    return plane


def _apply_filters_gpu(vol: np.ndarray) -> np.ndarray:
    d = cp.asarray(vol)
    if SHARPEN_AMOUNT != 0.0 and SHARPEN_SIGMA > 0:
        blur = _cp_gaussian_filter(d, sigma=SHARPEN_SIGMA,
                                   mode="constant", cval=0.0)
        d = d + cp.float32(SHARPEN_AMOUNT) * (d - blur)
        del blur
    if SMOOTH_SIGMA > 0:
        d = _cp_gaussian_filter(d, sigma=SMOOTH_SIGMA,
                                mode="constant", cval=0.0)
    d.get(out=vol)
    del d
    cp.get_default_memory_pool().free_all_blocks()
    return vol


def compute_chunk(z_start: int, z_end: int, mask: np.ndarray,
                  pool: ThreadPoolExecutor) -> np.ndarray:
    """Read + mask + optional 3-D filter for one compute chunk.  Returns
    a (z_end-z_start, OUT_NYX, OUT_NYX) float32 array."""
    _sig_max = max(SHARPEN_SIGMA if SHARPEN_AMOUNT != 0.0 else 0.0,
                   SMOOTH_SIGMA)
    halo = int(np.ceil(3 * _sig_max)) if _sig_max > 0 else 0
    z_lo = max(0,      z_start - halo)
    z_hi = min(OUT_NZ, z_end   + halo)
    n = z_hi - z_lo

    vol = np.empty((n, OUT_NYX, OUT_NYX), dtype=np.float32)

    def _read(i: int) -> None:
        vol[i] = _read_masked(z_lo + i, mask)
    list(pool.map(_read, range(n)))

    _apply_filters_gpu(vol)

    offset = z_start - z_lo
    return vol[offset:offset + (z_end - z_start)]


def main() -> None:
    if _RANK == 0:
        os.makedirs(os.path.dirname(DST_H5) or ".", exist_ok=True)

    mask = make_mask()
    _sig_max = max(SHARPEN_SIGMA if SHARPEN_AMOUNT != 0.0 else 0.0,
                   SMOOTH_SIGMA)
    _halo = int(np.ceil(3 * _sig_max)) if _sig_max > 0 else 0
    if _RANK == 0:
        backend = f"gpu×{_SIZE}"
        print(f"src : {SRC_TIFF}", flush=True)
        print(f"dst : {DST_H5}  (VDS + banks)", flush=True)
        print(f"backend={backend}  threads/rank={N_THREADS}  "
              f"chunk_z={CHUNK_Z}  nz={OUT_NZ}  nyx={OUT_NYX}  "
              f"nbanks={NBANKS}  init-vchunks={INIT_VCHUNKS}", flush=True)
        buf_gb = vchunk_bytes(INIT_VCHUNKS, np.float32) / 1e9
        print(f"per-rank shm buffer: {buf_gb:.2f} GB   nvchunks="
              f"{n_vchunks((OUT_NZ, OUT_NYX, OUT_NYX), INIT_VCHUNKS)}",
              flush=True)
        print(f"z: sample slices [{Z_PAD},{Z_PAD+SAMPLE_NZ})={SAMPLE_NZ}px "
              f"from src z=[{Z0},{Z0+SAMPLE_NZ}); "
              f"padded 0 in [0,{Z_PAD}) and [{Z_PAD+SAMPLE_NZ},{OUT_NZ}); "
              f"cosine taper {Z_TAPER} px each end", flush=True)
        print(f"yx=[{Y0},{Y0+OUT_NYX})  "
              f"mask: circle ⌀{CIRCLE_DIAM} px, cosine taper {MASK_TAPER} px  "
              f"(sum={int(mask.sum())}, pi*r^2 ~ {np.pi*(CIRCLE_DIAM/2)**2:.0f})",
              flush=True)
        print(f"3-D unsharp:  amount={SHARPEN_AMOUNT} sigma={SHARPEN_SIGMA}",
              flush=True)
        print(f"3-D smoothing: sigma={SMOOTH_SIGMA}   halo={_halo} slices",
              flush=True)

    # Create VDS + empty bank files; broadcast ctx to non-zero ranks.
    ctx = initx_and_bcast(DST_H5, shape=(OUT_NZ, OUT_NYX, OUT_NYX),
                          dtype=np.float32, vchunks=INIT_VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=_RANK, comm=_COMM)

    # Round-robin vchunk sharding across ranks (matches test_h5_buffer_io).
    ivchunks = list(iter_vchunks((OUT_NZ, OUT_NYX, OUT_NYX), INIT_VCHUNKS))
    my_ivchunks = ivchunks[_RANK::_SIZE]

    shm, buf = alloc_shm(INIT_VCHUNKS, np.float32)
    try:
        with ThreadPoolExecutor(max_workers=N_THREADS) as tpool:
            t_total = t_write = 0.0
            b_write = 0
            for i, ivc in enumerate(my_ivchunks):
                # This vchunk covers absolute z-range [z0, z1).
                z0 = ivc[0] * INIT_VCHUNKS[0]
                z1 = min(z0 + INIT_VCHUNKS[0], OUT_NZ)
                buf.fill(0)  # pad tail (last vchunk may be short)

                t_iter = time.perf_counter()
                # Fill buffer one compute chunk (--chunk-z) at a time.
                for zc0 in range(z0, z1, CHUNK_Z):
                    zc1 = min(zc0 + CHUNK_Z, z1)
                    piece = compute_chunk(zc0, zc1, mask, tpool)
                    buf[zc0 - z0 : zc1 - z0] = piece
                # One tomo_writex per vchunk: fans across NBANKS bank files.
                t_w = time.perf_counter()
                tomo_writex(DST_H5, data=buf, shm=shm,
                            ivchunk=ivc, ctx=ctx)
                t_write += time.perf_counter() - t_w
                b_write += (z1 - z0) * OUT_NYX * OUT_NYX * 4

                dt = time.perf_counter() - t_iter
                t_total += dt
                print(f"[rank {_RANK}] {i+1}/{len(my_ivchunks)}  "
                      f"z=[{z0},{z1})  {dt:.1f}s", flush=True)
            print(f"[rank {_RANK}] total per-rank time: "
                  f"{t_total:.1f}s ({len(my_ivchunks)} vchunks)", flush=True)
    finally:
        free_shm(shm)

    if _COMM is not None:
        _COMM.Barrier()
    report_stage("step00 write (init)", b_write, t_write)
    if _RANK == 0:
        print("init.h5 done.", flush=True)


if __name__ == "__main__":
    main()
    from utils import hard_exit
    hard_exit()
