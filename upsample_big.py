#!/usr/bin/env python
"""Upsample a 2560×2744×2744 (z,y,x) init volume by 8× in each axis
=> 20480 × 21952 × 21952 output volume, saved as one TIFF per output
z-slice.

Method: trilinear (3-D linear) interpolation.  Trilinear is separable,
so xy-linear per plane + linear blend in z gives the true 3-D linear
result at a fraction of the memory of an all-at-once 3-D zoom.

  - xy: cupyx.scipy.ndimage.zoom(order=1) on GPU (fast; ~10× vs PIL on
        chunks this big).  Falls back to PIL BILINEAR on CPU when no
        usable GPU is visible.
  - z : convex combination between two adjacent xy-upsampled planes,
        done on-device.
  - Pipeline: a background CPU thread prefetches the next input plane
        from disk while the GPU is blending/writing the current pair.

Multi-GPU via MPI (mpi4py, optional — single-rank fallback if unavailable):
  Input z-slices are partitioned contiguously across ranks.  Rank r owns
  [i_start, i_end); it also reads one extra slice at i_end for the final
  blend (unless i_end == IN_NZ, in which case the last slice is held).
  Each rank is pinned to a local GPU (LOCAL_RANK / OMPI_COMM_WORLD_LOCAL_RANK
  / SLURM_LOCALID / RANK % ndev).  Launch:
      mpirun -n <NGPU> --map-by ppr:1:socket python upsample_big.py

Sizes (per plane / total):
  input  plane : 2744  × 2744  float32  ≈  30 MB
  output plane : 21952 × 21952 float32  ≈  1.93 GB
  total output : 20480 × 21952² × 4 B   ≈  39.5 TB
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile


# --------------------- MPI (optional) ------------------------------------
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


def rprint(*a, **k) -> None:
    if RANK == 0:
        print(*a, **k)


# --------------------- GPU backend ---------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as _gpu_zoom
    _HAS_GPU = True
except Exception as _e:                                       # noqa: BLE001
    from PIL import Image
    _HAS_GPU = False
    rprint(f"[upsample] GPU disabled ({_e}); using CPU PIL BILINEAR")


def _pick_gpu() -> int:
    """Pick a GPU for this rank:
      1) explicit GPU_ID env
      2) LOCAL_RANK / OMPI_COMM_WORLD_LOCAL_RANK / SLURM_LOCALID (MPI launcher)
      3) RANK % <visible device count>
    """
    if "GPU_ID" in os.environ:
        return int(os.environ["GPU_ID"])
    for k in ("LOCAL_RANK",
              "OMPI_COMM_WORLD_LOCAL_RANK",
              "MV2_COMM_WORLD_LOCAL_RANK",
              "MPI_LOCALRANKID",
              "SLURM_LOCALID"):
        if k in os.environ:
            return int(os.environ[k])
    if _HAS_GPU:
        return RANK % max(cp.cuda.runtime.getDeviceCount(), 1)
    return 0


# --------------------- config ---------------------------------------------
SRC_DIR = "/data2/brain_sym_mosaic/init"       # Y350a init: 2560 × 2744 × 2744 slices
DST_DIR = "/data2/brain_sym_mosaic/big2x"      # 10240 × 10976 × 10976 slices

IN_NZ   = 2560
IN_NYX  = 2744
UPS     = 2                                    # in every axis
OUT_NZ  = IN_NZ  * UPS                         # 10240
OUT_NYX = IN_NYX * UPS                         # 10976

N_WRITE = int(os.environ.get("N_WRITE", "8"))  # parallel SSD writers per rank
N_READ  = int(os.environ.get("N_READ",  "2"))  # background input prefetchers

# 3-D Gaussian smoothing of the FINAL output volume (post-pass, overwrites
# in place).  σ in output-voxel units.  0 disables the pass entirely.
# Chunked in z with halo=ceil(3·σ) so chunk seams have no discontinuity.
SMOOTH_SIGMA   = float(os.environ.get("SMOOTH_SIGMA",   "0"))
SMOOTH_CHUNK_Z = int(  os.environ.get("SMOOTH_CHUNK_Z", "3"))


# --------------------- I/O helpers ---------------------------------------
def _read_plane(zi: int) -> np.ndarray:
    src = os.path.join(SRC_DIR, f"init_{zi:05d}.tif")
    im = tifffile.imread(src)
    if im.shape != (IN_NYX, IN_NYX):
        raise RuntimeError(
            f"unexpected shape {im.shape} in {src}, expected "
            f"({IN_NYX},{IN_NYX})")
    return im.astype(np.float32, copy=False)


def _write(path: str, data: np.ndarray) -> None:
    tifffile.imwrite(path, data, compression=None)


# --------------------- upsample + blend (GPU or CPU) ---------------------
#   XY: spline zoom of order XY_ORDER (default 3 = cubic).
#   Z : Z_ORDER=1 linear blend between 2 planes (default);
#       Z_ORDER=3 Catmull-Rom cubic between 4 planes (true tricubic).
XY_ORDER = int(os.environ.get("XY_ORDER", "1"))     # bilinear
Z_ORDER  = int(os.environ.get("Z_ORDER",  "1"))     # linear blend between 2 planes
if XY_ORDER not in (0, 1, 3, 5):
    raise SystemExit(f"XY_ORDER must be 0|1|3|5, got {XY_ORDER!r}")
if Z_ORDER not in (1, 3):
    raise SystemExit(f"Z_ORDER must be 1 or 3, got {Z_ORDER!r}")

if _HAS_GPU:
    def _upsample_xy(im_np: np.ndarray):
        """H2D + spline zoom by UPS in xy (order=XY_ORDER), returns cupy."""
        im_d = cp.asarray(im_np)
        return _gpu_zoom(im_d, zoom=UPS, order=XY_ORDER, mode="nearest")

    def _blend_and_pull(up_curr_d, up_next_d, r: int) -> np.ndarray:
        if r == 0:
            out_d = up_curr_d
        else:
            t = cp.float32(r / UPS)
            out_d = (cp.float32(1.0) - t) * up_curr_d + t * up_next_d
        return cp.asnumpy(out_d)
else:
    _PIL_MODE = {
        0: Image.Resampling.NEAREST,
        1: Image.Resampling.BILINEAR,
        3: Image.Resampling.BICUBIC,
        5: Image.Resampling.LANCZOS,     # PIL has no true order-5 spline
    }[XY_ORDER]

    def _upsample_xy(im_np: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(im_np, mode="F")
        return np.asarray(pil.resize((IN_NYX * UPS, IN_NYX * UPS), _PIL_MODE))

    def _blend_and_pull(up_curr, up_next, r: int) -> np.ndarray:
        if r == 0:
            return up_curr
        t = np.float32(r / UPS)
        return ((np.float32(1.0) - t) * up_curr +
                t * up_next).astype(np.float32, copy=False)


# --------------------- main -----------------------------------------------
def main() -> None:
    GPU_ID = _pick_gpu()
    if _HAS_GPU:
        cp.cuda.Device(GPU_ID).use()

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    _barrier()

    # Partition input z-slices contiguously across ranks.
    per_rank = (IN_NZ + SIZE - 1) // SIZE
    i_start  = min(RANK * per_rank, IN_NZ)
    i_end    = min(i_start + per_rank, IN_NZ)
    local_n  = i_end - i_start

    rprint(f"input : {IN_NZ}×{IN_NYX}×{IN_NYX}   dir={SRC_DIR}")
    rprint(f"output: {OUT_NZ}×{OUT_NYX}×{OUT_NYX}  dir={DST_DIR}")
    _xy_name = {0: "nearest", 1: "linear (bilinear)", 3: "cubic spline",
                5: "quintic spline / LANCZOS"}[XY_ORDER]
    _z_name  = {1: "linear", 3: "cubic (Catmull-Rom)"}[Z_ORDER]
    rprint(f"upsample: {UPS}×  xy={_xy_name} (order={XY_ORDER}), "
           f"z={_z_name} (order={Z_ORDER})  "
           f"backend={'GPU' if _HAS_GPU else 'CPU'}  "
           f"read={N_READ}  write={N_WRITE}  MPI ranks={SIZE}")
    rprint(f"estimated storage: "
           f"{OUT_NZ * OUT_NYX * OUT_NYX * 4 / 1e12:.2f} TB")

    dev_name = ""
    if _HAS_GPU:
        try:
            dev_name = cp.cuda.runtime.getDeviceProperties(GPU_ID)["name"].decode()
        except Exception:
            dev_name = ""
    print(f"  rank {RANK}/{SIZE}: gpu={GPU_ID} ({dev_name})  "
          f"input z=[{i_start}, {i_end})  ({local_n} slices)", flush=True)
    _barrier()

    if local_n == 0:
        rprint("nothing to do for this rank")
        return

    read_pool  = ThreadPoolExecutor(max_workers=N_READ,
                                    thread_name_prefix=f"r{RANK}-read")
    write_pool = ThreadPoolExecutor(max_workers=N_WRITE,
                                    thread_name_prefix=f"r{RANK}-write")

    pending: list = []
    max_pending = 2 * N_WRITE

    def submit_write(path: str, buf: np.ndarray) -> None:
        while len(pending) >= max_pending:
            pending.pop(0).result()
        pending.append(write_pool.submit(_write, path, buf))

    def _fetch_up(zi: int):
        """Read + xy-upsample plane at global input index zi (clamped to
        the volume's own edges — NOT to this rank's slab — so cubic-z
        boundary handling matches the whole-volume geometry)."""
        zc = min(max(zi, 0), IN_NZ - 1)
        return _upsample_xy(_read_plane(zc))

    if Z_ORDER == 1:
        # -------- LINEAR in z (2-plane rolling buffer) --------------------
        up_curr_d   = _fetch_up(i_start)
        fut_next_np = (read_pool.submit(_read_plane, i_start + 1)
                       if i_start + 1 < IN_NZ else None)

        for zi in range(i_start, i_end):
            if fut_next_np is not None:
                next_np     = fut_next_np.result()
                fut_next_np = (read_pool.submit(_read_plane, zi + 2)
                               if zi + 2 < IN_NZ else None)
                up_next_d = _upsample_xy(next_np)
                del next_np
            else:
                up_next_d = up_curr_d          # end of volume: hold

            for r in range(UPS):
                submit_write(
                    os.path.join(DST_DIR, f"big_{zi * UPS + r:05d}.tif"),
                    _blend_and_pull(up_curr_d, up_next_d, r),
                )

            up_curr_d = up_next_d

            done = zi - i_start + 1
            if done % 8 == 0 or done == local_n:
                print(f"  [rank {RANK}] input {done}/{local_n}  "
                      f"(wrote up to output slice {zi*UPS + UPS - 1})",
                      flush=True)

    else:
        # -------- CUBIC in z (Catmull-Rom, 4-plane rolling buffer) --------
        # For each output slice at parameter t ∈ [0,1) between input pair
        # (curr, next), we need 4 planes: prev, curr, next, next2.
        # Weights:
        #   w-1 = -0.5t + t^2 - 0.5t^3
        #   w0  =  1     - 2.5t^2 + 1.5t^3
        #   w1  =  0.5t + 2t^2  - 1.5t^3
        #   w2  =         -0.5t^2 + 0.5t^3
        up_prev_d = _fetch_up(i_start - 1)
        up_curr_d = _fetch_up(i_start)
        up_next_d = _fetch_up(i_start + 1)

        def _cubic_pull(prev_d, curr_d, next_d, next2_d, r: int) -> np.ndarray:
            if r == 0:
                return cp.asnumpy(curr_d)
            t  = float(r) / UPS
            t2 = t * t
            t3 = t2 * t
            w_1 = cp.float32(-0.5 * t + t2 - 0.5 * t3)
            w0  = cp.float32( 1.0        - 2.5 * t2 + 1.5 * t3)
            w1  = cp.float32( 0.5 * t + 2.0 * t2 - 1.5 * t3)
            w2  = cp.float32(              -0.5 * t2 + 0.5 * t3)
            out_d = (w_1 * prev_d + w0 * curr_d
                     + w1 * next_d + w2 * next2_d)
            return cp.asnumpy(out_d)

        for zi in range(i_start, i_end):
            up_next2_d = _fetch_up(zi + 2)

            for r in range(UPS):
                submit_write(
                    os.path.join(DST_DIR, f"big_{zi * UPS + r:05d}.tif"),
                    _cubic_pull(up_prev_d, up_curr_d, up_next_d,
                                up_next2_d, r),
                )

            # rotate: prev←curr, curr←next, next←next2
            up_prev_d = up_curr_d
            up_curr_d = up_next_d
            up_next_d = up_next2_d

            done = zi - i_start + 1
            if done % 8 == 0 or done == local_n:
                print(f"  [rank {RANK}] input {done}/{local_n}  "
                      f"(wrote up to output slice {zi*UPS + UPS - 1})",
                      flush=True)

    for f in pending:
        f.result()
    read_pool.shutdown()
    write_pool.shutdown()
    _barrier()
    rprint("upsample done.")

    if SMOOTH_SIGMA > 0:
        _post_smooth()
    _barrier()
    rprint("all done.")


def _post_smooth() -> None:
    """3-D Gaussian smoothing of the final output volume, in place.

    Two speedups vs. the naive chunked+scipy version:
      1. GPU filter (cupyx.scipy.ndimage.gaussian_filter) when available.
      2. Halo caching — the last `halo` slices of each chunk are kept in
         memory and reused as the next chunk's lower halo, so each disk
         slice is read at most once per rank instead of 3× (halo overlap).
    """
    if _HAS_GPU:
        from cupyx.scipy.ndimage import gaussian_filter as _gf
        _asarray = cp.asarray
        _asnumpy = cp.asnumpy
        _backend = "GPU"
    else:
        from scipy.ndimage import gaussian_filter as _gf
        _asarray = lambda a: a
        _asnumpy = lambda a: a
        _backend = "CPU"

    halo = int(np.ceil(3 * SMOOTH_SIGMA))

    # Contiguous z-partition across MPI ranks.
    per_rank = (OUT_NZ + SIZE - 1) // SIZE
    z_start_rank = min(RANK * per_rank, OUT_NZ)
    z_end_rank   = min(z_start_rank + per_rank, OUT_NZ)
    if z_start_rank >= z_end_rank:
        return

    rprint(f"post-smooth: sigma={SMOOTH_SIGMA} px  halo={halo}  "
           f"chunk_z={SMOOTH_CHUNK_Z}  backend={_backend}  MPI ranks={SIZE}")
    print(f"  [rank {RANK}] post-smooth z=[{z_start_rank},{z_end_rank})",
          flush=True)

    # Cached lower halo (host numpy) so we don't re-read those slices from
    # disk on the next chunk.  On the first chunk, this is None and we
    # read the lower halo from disk.
    lower_halo = None                                 # shape (h, Y, X) or None

    def _read_disk(zi_lo: int, count: int) -> np.ndarray:
        """Read `count` slices starting at global z=zi_lo, in parallel."""
        buf = np.empty((count, OUT_NYX, OUT_NYX), dtype=np.float32)
        def _r(i: int) -> None:
            zi = zi_lo + i
            buf[i] = tifffile.imread(
                os.path.join(DST_DIR, f"big_{zi:05d}.tif"))
        with ThreadPoolExecutor(max_workers=N_READ) as p:
            list(p.map(_r, range(count)))
        return buf

    for z_start in range(z_start_rank, z_end_rank, SMOOTH_CHUNK_Z):
        z_end = min(z_start + SMOOTH_CHUNK_Z, z_end_rank)
        # Full window we want in memory for the filter:
        z_lo = max(0,      z_start - halo)
        z_hi = min(OUT_NZ, z_end   + halo)

        # Where the cached lower halo starts / ends (may be < z_lo).
        # It always ends at z_start (we cached tail of previous chunk).
        if lower_halo is not None:
            need_lower_h_from = z_start - lower_halo.shape[0]   # cache start
            # Some of the cache may be outside [z_lo, z_hi); trim.
            trim = max(0, z_lo - need_lower_h_from)
            cached = lower_halo[trim:]
            new_lo = z_start                                    # after cache
        else:
            cached  = None
            new_lo  = z_lo                                       # read from z_lo

        # Read the *new* slices from disk: [new_lo, z_hi).
        new_count = z_hi - new_lo
        new_buf = _read_disk(new_lo, new_count) if new_count > 0 else \
                  np.empty((0, OUT_NYX, OUT_NYX), dtype=np.float32)

        # Assemble: [cached | new_buf]
        vol_h = np.concatenate([cached, new_buf], axis=0) \
                if cached is not None and cached.shape[0] > 0 else new_buf

        # 3-D Gaussian smooth (GPU if available).
        vol_d = _asarray(vol_h)
        vol_d = _gf(vol_d, sigma=SMOOTH_SIGMA, mode="constant", cval=0.0)
        vol_h = _asnumpy(vol_d)
        if _HAS_GPU:
            del vol_d
            cp.get_default_memory_pool().free_all_blocks()

        # Write only the interior [z_start, z_end).
        offset = z_start - z_lo
        def _w(i: int) -> None:
            zo = z_start + i
            tifffile.imwrite(
                os.path.join(DST_DIR, f"big_{zo:05d}.tif"),
                vol_h[offset + i], compression=None)
        with ThreadPoolExecutor(max_workers=N_WRITE) as p:
            list(p.map(_w, range(z_end - z_start)))

        # Cache the *unsmoothed* tail of this chunk's inputs for the next
        # chunk's lower halo.  We saved `new_buf`'s last `halo` slices;
        # those come from disk untouched, not from the smoothed vol_h.
        if new_buf.shape[0] >= halo:
            lower_halo = new_buf[-halo:].copy()
        elif cached is not None:
            # Combine cached tail + new_buf tail to give a `halo`-length window
            combined = np.concatenate([cached, new_buf], axis=0)
            lower_halo = combined[-halo:].copy() if combined.shape[0] >= halo \
                         else combined.copy()
        del new_buf, vol_h

        print(f"  [rank {RANK}] post-smooth chunk [{z_start},{z_end})",
              flush=True)


if __name__ == "__main__":
    main()
