#!/usr/bin/env python
"""Extract a centred sub-volume from a 3-D reconstruction TIFF, apply a
soft-edged circular mask (and optional 3-D sharpen + smoothing), and save
as a per-slice TIFF stack.  Threaded I/O, optional MPI + GPU.
"""
from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as _cp_gaussian_filter
    _HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAVE_GPU = False

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
except Exception:
    _COMM = None
    _RANK = 0
    _SIZE = 1


# ---------- CLI ----------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=(
        "/data3/vnikitin/brain_rec/20251115/Y350a1234/"
        "4500_2048_0_0.0_0.003_0.05_0.02_20_1.1_0/rec_obj_real/0096.tiff"),
        help="source 3-D reconstruction TIFF (multi-page)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
        help="base output directory; init tifs go to {path}/init")
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
                   help="z-slices processed per chunk")
    return p.parse_args()


_A = _parse_args()

SRC_TIFF      = _A.src
DST_DIR       = f"{_A.path}/init"
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

SAMPLE_NZ = OUT_NZ - 2 * Z_PAD
if SAMPLE_NZ <= 0:
    raise SystemExit(f"--z-pad {Z_PAD} leaves no sample slices (--out-nz {OUT_NZ}).")

Z0 = (SRC_NZ - SAMPLE_NZ) // 2
Y0 = (SRC_NY - OUT_NYX) // 2
X0 = (SRC_NX - OUT_NYX) // 2

# Bind each rank to its own GPU (round-robin over visible devices).
if _HAVE_GPU:
    cp.cuda.Device(_RANK % cp.cuda.runtime.getDeviceCount()).use()

# Zero-copy memmap of the source TIFF (uncompressed contiguous pages).
try:
    _SRC = tifffile.memmap(SRC_TIFF, mode='r')
except Exception:
    _SRC = None
_tiff_local = threading.local()


def _read_src_plane(z: int) -> np.ndarray:
    """Return one (SRC_NY, SRC_NX) plane from the source volume."""
    if _SRC is not None:
        return _SRC[z]
    tf = getattr(_tiff_local, 'tf', None)
    if tf is None:
        tf = tifffile.TiffFile(SRC_TIFF)
        _tiff_local.tf = tf
    return tf.pages[z].asarray()


def _z_weight(z_out: int) -> float:
    """Cosine-tapered z-window: 0 in the padded ends, cosine ramp of
    Z_TAPER px inside each sample edge, 1 in the middle."""
    if z_out < Z_PAD or z_out >= Z_PAD + SAMPLE_NZ:
        return 0.0
    d = min(z_out - Z_PAD, Z_PAD + SAMPLE_NZ - 1 - z_out)
    if d >= Z_TAPER:
        return 1.0
    t = (d + 0.5) / Z_TAPER
    return float(0.5 - 0.5 * np.cos(np.pi * t))


def make_mask() -> np.ndarray:
    """Circular mask with a smooth cosine taper of MASK_TAPER pixels at the
    boundary.  Limits the mask's spectrum to ~1/MASK_TAPER cycles/px."""
    r = CIRCLE_DIAM / 2.0
    c = (OUT_NYX - 1) / 2.0
    y, x = np.ogrid[:OUT_NYX, :OUT_NYX]
    rho = np.sqrt((y - c) ** 2 + (x - c) ** 2)
    if MASK_TAPER <= 0:
        return (rho <= r).astype(np.float32)
    t = np.clip((r - rho) / MASK_TAPER + 0.5, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def _read_masked(z_out: int, mask: np.ndarray) -> np.ndarray:
    """Return the (OUT_NYX, OUT_NYX) plane at output-z index z_out."""
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


def _write_slice(z_out: int, plane: np.ndarray) -> None:
    tifffile.imwrite(
        os.path.join(DST_DIR, f"init_{z_out:05d}.tif"),
        plane.astype(np.float32, copy=False),
        compression=None,
    )


def _apply_filters_gpu(vol: np.ndarray) -> np.ndarray:
    """Run the unsharp mask (if enabled) and the main 3-D Gaussian on the
    GPU in a single host↔device round-trip; in-place."""
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


def _gaussian_3d_threaded(vol: np.ndarray, sigma: float,
                          pool: ThreadPoolExecutor) -> None:
    """In-place 3-D Gaussian using scipy 1-D separable passes across threads."""
    if sigma <= 0:
        return
    nz, ny, _ = vol.shape
    kw = dict(sigma=sigma, mode="constant", cval=0.0)

    def _xy(z: int) -> None:
        gaussian_filter1d(vol[z], axis=1, output=vol[z], **kw)
        gaussian_filter1d(vol[z], axis=0, output=vol[z], **kw)
    list(pool.map(_xy, range(nz)))

    y_chunk = max(1, ny // (N_THREADS * 4))
    def _z(y0: int) -> None:
        y1 = min(y0 + y_chunk, ny)
        strip = vol[:, y0:y1, :]
        gaussian_filter1d(strip, axis=0, output=strip, **kw)
    list(pool.map(_z, range(0, ny, y_chunk)))


def process_chunk(z_start: int, z_end: int, mask: np.ndarray,
                  pool: ThreadPoolExecutor) -> None:
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

    if _HAVE_GPU:
        _apply_filters_gpu(vol)
    else:
        if SHARPEN_AMOUNT != 0.0:
            blur = vol.copy()
            _gaussian_3d_threaded(blur, SHARPEN_SIGMA, pool)
            vol = vol + np.float32(SHARPEN_AMOUNT) * (vol - blur)
            del blur
        _gaussian_3d_threaded(vol, SMOOTH_SIGMA, pool)

    offset = z_start - z_lo
    def _write(i: int) -> None:
        _write_slice(z_start + i, vol[offset + i])

    list(pool.map(_write, range(z_end - z_start)))


def main() -> None:
    if _RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    if _COMM is not None:
        _COMM.Barrier()
    mask = make_mask()
    _sig_max = max(SHARPEN_SIGMA if SHARPEN_AMOUNT != 0.0 else 0.0,
                   SMOOTH_SIGMA)
    _halo = int(np.ceil(3 * _sig_max)) if _sig_max > 0 else 0
    if _RANK == 0:
        backend = f"gpu×{_SIZE}" if _HAVE_GPU else f"cpu×{_SIZE}"
        print(f"src : {SRC_TIFF}")
        print(f"dst : {DST_DIR}")
        print(f"backend={backend}  threads/rank={N_THREADS}  "
              f"chunk_z={CHUNK_Z}  nz={OUT_NZ}  nyx={OUT_NYX}")
        print(f"z: sample slices [{Z_PAD},{Z_PAD+SAMPLE_NZ})={SAMPLE_NZ}px "
              f"from src z=[{Z0},{Z0+SAMPLE_NZ}); "
              f"padded 0 in [0,{Z_PAD}) and [{Z_PAD+SAMPLE_NZ},{OUT_NZ}); "
              f"cosine taper {Z_TAPER} px each end")
        print(f"yx=[{Y0},{Y0+OUT_NYX})  "
              f"mask: circle ⌀{CIRCLE_DIAM} px, cosine taper {MASK_TAPER} px  "
              f"(sum={int(mask.sum())}, pi*r^2 ~ {np.pi*(CIRCLE_DIAM/2)**2:.0f})")
        print(f"3-D unsharp:  amount={SHARPEN_AMOUNT} sigma={SHARPEN_SIGMA}")
        print(f"3-D smoothing: sigma={SMOOTH_SIGMA}   halo={_halo} slices")

    chunks = [(z, min(z + CHUNK_Z, OUT_NZ))
              for z in range(0, OUT_NZ, CHUNK_Z)]
    my_chunks = chunks[_RANK::_SIZE]

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for i, (z_start, z_end) in enumerate(my_chunks):
            process_chunk(z_start, z_end, mask, pool)
            print(f"[rank {_RANK}] {i+1}/{len(my_chunks)}  "
                  f"z=[{z_start},{z_end})", flush=True)

    if _COMM is not None:
        _COMM.Barrier()


if __name__ == "__main__":
    main()
