#!/usr/bin/env python
"""Extract a centred sub-volume from a 3-D reconstruction TIFF, apply a
soft-edged CYLINDRICAL MASK (circle in xy + cosine taper in z), bilinear-
upsample to (OUT_NZ, OUT_NYX, OUT_NYX), and save as a SINGLE HDF5 file
at {path}/init.h5.

Pipeline (per rank, per vchunk in OUT z):
  1. Center-crop the source to (CROP_NZ, CROP_NYX, CROP_NYX).  Default
     2560^3 — matches the physical brain size and always ≤ typical source.
  2. Apply cylindrical mask IN THE CROP GRID:
       * circle of --circle-diam voxels with --mask-taper cosine edge in xy
       * cosine z-weight with --z-pad zero-band + --z-taper cosine ramp
         at each end (smooth transition to air on top/bottom)
  3. Bilinear-upsample xy from CROP_NYX to OUT_NYX + linear-interpolate
     z from CROP_NZ to OUT_NZ (per chunk, so the whole volume never lives
     on host at once).
  4. Write the resulting z-slab straight to init.h5 (single file, HDF5
     chunked storage with ALLOC_TIME_EARLY so disjoint z-regions are
     safe to write concurrently from multiple ranks).

    {path}/init.h5
        /exchange/data   (OUT_NZ, OUT_NYX, OUT_NYX) float32
                         chunks (1, OUT_NYX, OUT_NYX)
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

import shutil

from iohdf5.h5_vchunks import (
    alloc_shm, free_shm, iter_vchunks, vchunk_bytes, n_vchunks,
)

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as _cp_gaussian_filter
from scipy.ndimage import zoom as _zoom

from mpi_utils import (
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
    # --- CROP (pre-upsample) grid: source is center-cropped to this size
    # and the cylindrical mask is applied here.
    p.add_argument("--crop-nz",  type=int, default=2560,
                   help="z crop size from source (default 2560; must be ≤ SRC_NZ)")
    p.add_argument("--crop-nyx", type=int, default=2560,
                   help="xy crop size from source (default 2560; must be ≤ SRC_NYX)")
    # Pipeline convention: we CROP the middle 2560^3 of the source TIFF,
    # UPSAMPLE it to a 3072^3 init.h5 (factor 3072/2560 = 1.2), apply a
    # cylindrical mask of diameter ≈ 0.95·OUT_NYX (~2918 in the OUT grid,
    # so a ~2432 mask in the CROP grid) with a cosine taper, and leave
    # ~50 OUT voxels of zero (with smooth cosine ramp) at the top and
    # bottom of z (~42 in the CROP grid).  Result: init.h5 is a 3072^3
    # cube containing a right circular cylinder that fills 95% of the FOV
    # in xy and ~97% of the FOV in z (both with smooth air borders).
    p.add_argument("--circle-diam", type=int, default=2432,
                   help="mask diameter in the CROP_NYX=2560 grid.  Default "
                        "2432 ≈ 0.95·CROP_NYX → after 2560→3072 upsample "
                        "gives a 2918-voxel sample cylinder in init.h5 "
                        "(≈ 32.2 mm at 11.04 µm/vx at UPS=1).  Matches "
                        "step0_schematic's SAMPLE_D_PX = 2918·UPS.")
    p.add_argument("--z-pad",   type=int, default=42,
                   help="zero-padding on each end of z, in CROP_NZ=2560 "
                        "grid.  Default 42 (≈50 in the OUT_NZ=3072 grid): "
                        "leaves 2476 CROP voxels of sample core → 2971 "
                        "voxels in init.h5, matching schematic's "
                        "SAMPLE_H_PX = 2972·UPS (rounded).")
    p.add_argument("--z-taper", type=int, default=15,
                   help="cosine taper into the padded ends, in CROP_NZ grid")
    p.add_argument("--mask-taper", type=float, default=7.0,
                   help="cosine taper on the circular xy mask (px)")
    # --- OUT (post-upsample) grid: init.h5 shape.
    p.add_argument("--out-nz",  type=int, default=3072,
                   help="output nz after upsample (default 3072; "
                        "N_z downstream = OUT_NZ·UPS).  Was 4096 in prior "
                        "revisions; 3072 matches a 2560 → 3072 upsample "
                        "of the CROP grid (factor 1.2).")
    p.add_argument("--out-nyx", type=int, default=3072,
                   help="output y=x after upsample (default 3072; "
                        "N downstream = OUT_NYX·UPS)")
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
    p.add_argument("--init-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk shape for init.h5 (default: "
                        "--chunk-z, OUT_NYX, OUT_NYX = one compute chunk per vchunk)")
    return p.parse_args()


_A = _parse_args()

SRC_TIFF      = _A.src
DST_H5        = f"{_A.path}/init.h5"
CROP_NZ       = _A.crop_nz
CROP_NYX      = _A.crop_nyx
OUT_NZ        = _A.out_nz
OUT_NYX       = _A.out_nyx
CIRCLE_DIAM   = _A.circle_diam
Z_PAD         = _A.z_pad         # in CROP_NZ grid
Z_TAPER       = _A.z_taper       # in CROP_NZ grid
MASK_TAPER    = _A.mask_taper    # in CROP_NYX grid
SHARPEN_AMOUNT = _A.sharpen_amount
SHARPEN_SIGMA  = _A.sharpen_sigma
SMOOTH_SIGMA   = _A.smooth_sigma
N_THREADS     = _A.n_threads
CHUNK_Z       = _A.chunk_z
INIT_VCHUNKS  = tuple(_A.init_vchunks) if _A.init_vchunks else (CHUNK_Z, _A.out_nyx, _A.out_nyx)

SAMPLE_NZ = CROP_NZ - 2 * Z_PAD    # sample-slice count in CROP grid
if SAMPLE_NZ <= 0:
    raise SystemExit(f"--z-pad {Z_PAD} leaves no sample slices in "
                     f"--crop-nz {CROP_NZ}.")

# Upsample ratios (float, may be non-integer)
Z_ZOOM  = OUT_NZ  / CROP_NZ
XY_ZOOM = OUT_NYX / CROP_NYX

cp.cuda.Device(_RANK % cp.cuda.runtime.getDeviceCount()).use()

# Zero-copy memmap of the source TIFF (uncompressed contiguous pages).
try:
    _SRC = tifffile.memmap(SRC_TIFF, mode='r')
except Exception:
    _SRC = None
_tiff_local = threading.local()

# Source dims come straight from the TIFF header.
if _SRC is not None:
    SRC_NZ, SRC_NY, SRC_NX = _SRC.shape
else:
    with tifffile.TiffFile(SRC_TIFF) as _tf_probe:
        SRC_NZ = len(_tf_probe.pages)
        SRC_NY, SRC_NX = _tf_probe.pages[0].shape

# Source-to-CROP offsets.  If SRC is bigger than CROP we center-crop; if
# smaller (unusual) we center-pad with zeros (air).  _read_crop_plane
# handles both via non-negative windows.
Z_SRC_OFF = (SRC_NZ - CROP_NZ) // 2
Y_SRC_OFF = (SRC_NY - CROP_NYX) // 2
X_SRC_OFF = (SRC_NX - CROP_NYX) // 2


def _read_src_plane(z: int) -> np.ndarray:
    if _SRC is not None:
        return _SRC[z]
    tf = getattr(_tiff_local, 'tf', None)
    if tf is None:
        tf = tifffile.TiffFile(SRC_TIFF)
        _tiff_local.tf = tf
    return tf.pages[z].asarray()


def _z_weight_crop(z_crop: int) -> float:
    """Cosine z-taper in the CROP_NZ grid: zero in the outer Z_PAD bands,
    cosine ramp over the next Z_TAPER slices, unity in the sample core."""
    if z_crop < Z_PAD or z_crop >= Z_PAD + SAMPLE_NZ:
        return 0.0
    d = min(z_crop - Z_PAD, Z_PAD + SAMPLE_NZ - 1 - z_crop)
    if d >= Z_TAPER:
        return 1.0
    t = (d + 0.5) / Z_TAPER
    return float(0.5 - 0.5 * np.cos(np.pi * t))


def make_mask() -> np.ndarray:
    """(CROP_NYX, CROP_NYX) soft-edged circular mask, applied in CROP grid."""
    r = CIRCLE_DIAM / 2.0
    c = (CROP_NYX - 1) / 2.0
    y, x = np.ogrid[:CROP_NYX, :CROP_NYX]
    rho = np.sqrt((y - c) ** 2 + (x - c) ** 2)
    if MASK_TAPER <= 0:
        return (rho <= r).astype(np.float32)
    t = np.clip((r - rho) / MASK_TAPER + 0.5, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def _read_crop_plane(z_crop: int, mask_xy: np.ndarray) -> np.ndarray:
    """Read one source plane at z_src = z_crop + Z_SRC_OFF, center-crop
    xy to (CROP_NYX, CROP_NYX), apply xy circle mask + z-taper.  Returns
    a (CROP_NYX, CROP_NYX) float32 plane in the CROP grid."""
    w = _z_weight_crop(z_crop)
    if w == 0.0:
        return np.zeros((CROP_NYX, CROP_NYX), dtype=np.float32)
    z_src = z_crop + Z_SRC_OFF
    if not (0 <= z_src < SRC_NZ):
        return np.zeros((CROP_NYX, CROP_NYX), dtype=np.float32)
    im = _read_src_plane(z_src).astype(np.float32, copy=False)
    # Center-place source into (CROP_NYX, CROP_NYX) with non-negative
    # windows on both sides.  SRC ≥ CROP  → center-crop.  SRC < CROP →
    # center-pad with zeros (air).
    y_dst_lo = max(0, -Y_SRC_OFF)
    x_dst_lo = max(0, -X_SRC_OFF)
    y_src_lo = max(0,  Y_SRC_OFF)
    x_src_lo = max(0,  X_SRC_OFF)
    y_len = min(CROP_NYX - y_dst_lo, im.shape[0] - y_src_lo)
    x_len = min(CROP_NYX - x_dst_lo, im.shape[1] - x_src_lo)
    plane = np.zeros((CROP_NYX, CROP_NYX), dtype=np.float32)
    if y_len > 0 and x_len > 0:
        plane[y_dst_lo:y_dst_lo + y_len, x_dst_lo:x_dst_lo + x_len] = \
            im[y_src_lo:y_src_lo + y_len, x_src_lo:x_src_lo + x_len]
    plane *= mask_xy
    if w < 1.0:
        plane *= np.float32(w)
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


def _crop_to_out_z(z_out_lo: int, z_out_hi: int) -> tuple[int, int]:
    """Given an output z-range [z_out_lo, z_out_hi), return the CROP
    z-range [z_crop_lo, z_crop_hi) needed to reconstruct it with a
    ±1-slice halo for linear z-interp."""
    if Z_ZOOM == 1.0:
        return z_out_lo, z_out_hi
    z_crop_lo = max(0, int(np.floor(z_out_lo / Z_ZOOM)) - 1)
    z_crop_hi = min(CROP_NZ, int(np.ceil((z_out_hi - 1) / Z_ZOOM)) + 2)
    return z_crop_lo, z_crop_hi


def compute_chunk(z_start: int, z_end: int, mask_xy: np.ndarray,
                  pool: ThreadPoolExecutor) -> np.ndarray:
    """Produce an OUT-grid chunk covering [z_start, z_end) in OUT_NZ.

    Reads only the CROP z-range needed (+halo for zoom), applies mask in
    the CROP grid, then 3-D bilinear upsamples to the OUT chunk shape
    (linear in z, bilinear in xy).  Returns (z_end - z_start, OUT_NYX,
    OUT_NYX) float32.
    """
    # Extra halo for optional GPU filters — kept in OUT-grid units, then
    # widened in CROP grid via _crop_to_out_z.
    _sig_max = max(SHARPEN_SIGMA if SHARPEN_AMOUNT != 0.0 else 0.0,
                   SMOOTH_SIGMA)
    filter_halo = int(np.ceil(3 * _sig_max)) if _sig_max > 0 else 0
    z_out_lo = max(0,      z_start - filter_halo)
    z_out_hi = min(OUT_NZ, z_end   + filter_halo)
    n_out    = z_out_hi - z_out_lo

    # 1. Read + mask CROP z-range (with halo for zoom).
    z_crop_lo, z_crop_hi = _crop_to_out_z(z_out_lo, z_out_hi)
    n_crop = z_crop_hi - z_crop_lo
    crop_vol = np.empty((n_crop, CROP_NYX, CROP_NYX), dtype=np.float32)

    def _read(i: int) -> None:
        crop_vol[i] = _read_crop_plane(z_crop_lo + i, mask_xy)
    list(pool.map(_read, range(n_crop)))

    # 2. Bilinear-upsample xy per plane: (n_crop, CROP_NYX, CROP_NYX) →
    #    (n_crop, OUT_NYX, OUT_NYX).  Skipped when XY_ZOOM == 1.
    if XY_ZOOM != 1.0:
        xy_vol = np.empty((n_crop, OUT_NYX, OUT_NYX), dtype=np.float32)

        def _zoom_xy(i: int) -> None:
            xy_vol[i] = _zoom(crop_vol[i], (XY_ZOOM, XY_ZOOM),
                              order=1, mode="nearest", grid_mode=False)
        list(pool.map(_zoom_xy, range(n_crop)))
        del crop_vol
    else:
        xy_vol = crop_vol

    # 3. Linear-interpolate along z into the OUT-grid chunk.
    out_vol = np.empty((n_out, OUT_NYX, OUT_NYX), dtype=np.float32)
    if Z_ZOOM == 1.0:
        out_vol[:] = xy_vol[z_out_lo - z_crop_lo :
                            z_out_lo - z_crop_lo + n_out]
    else:
        for i in range(n_out):
            z_out = z_out_lo + i
            z_crop_f = z_out / Z_ZOOM                    # continuous CROP z
            z_lo_i = int(np.floor(z_crop_f)) - z_crop_lo # local index
            z_hi_i = z_lo_i + 1
            a = z_crop_f - np.floor(z_crop_f)
            if 0 <= z_lo_i < n_crop and z_hi_i < n_crop:
                out_vol[i] = ((1.0 - a) * xy_vol[z_lo_i]
                              +      a  * xy_vol[z_hi_i])
            elif 0 <= z_lo_i < n_crop:
                out_vol[i] = xy_vol[z_lo_i]
            else:
                out_vol[i] = 0
    del xy_vol

    # 4. Optional GPU 3-D unsharp / smooth (rarely used).
    if SHARPEN_AMOUNT != 0.0 or SMOOTH_SIGMA > 0:
        _apply_filters_gpu(out_vol)

    offset = z_start - z_out_lo
    return out_vol[offset:offset + (z_end - z_start)]


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
        print(f"dst : {DST_H5}  (single HDF5 file, "
              f"chunks=(1, {OUT_NYX}, {OUT_NYX}), ALLOC_TIME_EARLY)",
              flush=True)
        print(f"backend={backend}  threads/rank={N_THREADS}  "
              f"chunk_z={CHUNK_Z}  nz={OUT_NZ}  nyx={OUT_NYX}  "
              f"init-vchunks={INIT_VCHUNKS}", flush=True)
        buf_gb = vchunk_bytes(INIT_VCHUNKS, np.float32) / 1e9
        print(f"per-rank shm buffer: {buf_gb:.2f} GB   nvchunks="
              f"{n_vchunks((OUT_NZ, OUT_NYX, OUT_NYX), INIT_VCHUNKS)}",
              flush=True)
        print(f"src shape : ({SRC_NZ}, {SRC_NY}, {SRC_NX})", flush=True)
        print(f"crop grid : ({CROP_NZ}, {CROP_NYX}, {CROP_NYX})   "
              f"center-crop offsets Z_SRC_OFF={Z_SRC_OFF} "
              f"Y_SRC_OFF={Y_SRC_OFF} X_SRC_OFF={X_SRC_OFF}", flush=True)
        print(f"z-mask    : air [0,{Z_PAD}) and [{Z_PAD+SAMPLE_NZ},{CROP_NZ}); "
              f"cosine taper {Z_TAPER} px; sample core "
              f"[{Z_PAD+Z_TAPER},{Z_PAD+SAMPLE_NZ-Z_TAPER})", flush=True)
        print(f"xy-mask   : circle ⌀{CIRCLE_DIAM} px, cosine taper "
              f"{MASK_TAPER} px  (sum={int(mask.sum())}, "
              f"pi*r^2 ~ {np.pi*(CIRCLE_DIAM/2)**2:.0f})", flush=True)
        print(f"upsample  : ({CROP_NZ}, {CROP_NYX}, {CROP_NYX}) → "
              f"({OUT_NZ}, {OUT_NYX}, {OUT_NYX})   "
              f"z_zoom={Z_ZOOM:.4f}  xy_zoom={XY_ZOOM:.4f}   "
              f"(linear z, bilinear xy)", flush=True)
        print(f"3-D unsharp:  amount={SHARPEN_AMOUNT} sigma={SHARPEN_SIGMA}",
              flush=True)
        print(f"3-D smoothing: sigma={SMOOTH_SIGMA}   halo={_halo} slices",
              flush=True)

    # Rank 0 creates the SINGLE init.h5 file (removing any prior
    # single-file or old VDS+bank layout).  ALLOC_TIME_EARLY pre-reserves
    # every chunk so subsequent writes never touch the chunk-index B-tree
    # — that lets multiple ranks safely write disjoint z-regions in
    # parallel when HDF5_USE_FILE_LOCKING=FALSE.
    if _RANK == 0:
        if os.path.isfile(DST_H5):
            os.remove(DST_H5)
        _stem = os.path.splitext(os.path.basename(DST_H5))[0]
        _bank_dir = os.path.join(os.path.dirname(DST_H5) or ".", _stem)
        if os.path.isdir(_bank_dir):
            shutil.rmtree(_bank_dir)

        with h5py.File(DST_H5, "w", libver="latest") as _hf_create:
            g = _hf_create.create_group("exchange")
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            dcpl.set_chunk((1, OUT_NYX, OUT_NYX))
            dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
            dcpl.set_fill_value(np.array(0.0, dtype=np.float32))
            sid = h5py.h5s.create_simple((OUT_NZ, OUT_NYX, OUT_NYX))
            tid = h5py.h5t.py_create(np.dtype(np.float32))
            h5py.h5d.create(g.id, b"data", tid, sid, dcpl=dcpl)
        print(f"[step00] created {DST_H5}  "
              f"({OUT_NZ*OUT_NYX*OUT_NYX*4/1e9:.1f} GB dense)", flush=True)
    _barrier()

    # Round-robin vchunk sharding across ranks.
    ivchunks = list(iter_vchunks((OUT_NZ, OUT_NYX, OUT_NYX), INIT_VCHUNKS))
    my_ivchunks = ivchunks[_RANK::_SIZE]

    shm, buf = alloc_shm(INIT_VCHUNKS, np.float32)
    try:
        with ThreadPoolExecutor(max_workers=N_THREADS) as tpool, \
             h5py.File(DST_H5, "r+", libver="latest") as hf:
            dset = hf["exchange/data"]
            t_total = t_write = 0.0
            b_write = 0
            for i, ivc in enumerate(my_ivchunks):
                z0 = ivc[0] * INIT_VCHUNKS[0]
                z1 = min(z0 + INIT_VCHUNKS[0], OUT_NZ)
                buf.fill(0)  # pad tail (last vchunk may be short)

                t_iter = time.perf_counter()
                for zc0 in range(z0, z1, CHUNK_Z):
                    zc1 = min(zc0 + CHUNK_Z, z1)
                    piece = compute_chunk(zc0, zc1, mask, tpool)
                    buf[zc0 - z0 : zc1 - z0] = piece

                t_w = time.perf_counter()
                dset.write_direct(buf[: z1 - z0],
                                  dest_sel=np.s_[z0:z1, :, :])
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
        print(f"init.h5 done ({DST_H5}).", flush=True)


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
