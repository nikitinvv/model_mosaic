#!/usr/bin/env python
"""Upsample a (2560, 2744, 2744) init volume by UPS× in each axis and save
one TIFF per output z-slice.  Method: bilinear xy + linear z blend
(trilinear).  Trilinear is separable, so xy-linear per plane + linear
blend between two adjacent upsampled planes gives the true 3-D linear
result at a fraction of the memory of a whole-volume 3-D zoom.

  - xy: cupyx.scipy.ndimage.zoom(order=1) on GPU (PIL BILINEAR fallback).
  - z : convex combination between two adjacent xy-upsampled planes.
  - Pipeline: a background CPU thread prefetches the next input plane
        from disk while the GPU is blending/writing the current pair.

Multi-GPU via MPI (mpi4py optional).  GPU affinity is delegated to the
launcher: wrap with set_affinity_gpu.sh so each rank sees exactly one GPU
via CUDA_VISIBLE_DEVICES.

    mpirun -n <NGPU> set_affinity_gpu.sh python step0_upsample.py --ups 4
"""
from __future__ import annotations

import argparse
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


# --------------------- CLI -----------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ups",  type=int, default=4,
                   help="upsample factor (in every axis)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help=("base directory; reads {path}/init/init_*.tif and "
                         "writes {path}/big{UPS}x/big_*.tif"))
    p.add_argument("--in-nz",  type=int, default=2560, help="input nz")
    p.add_argument("--in-nyx", type=int, default=2744, help="input ny=nx")
    p.add_argument("--n-write", type=int, default=8,
                   help="parallel SSD writers per rank")
    p.add_argument("--n-read",  type=int, default=2,
                   help="background input prefetchers per rank")
    return p.parse_args()


_A = _parse_args()

UPS     = _A.ups
BASE    = _A.path
SRC_DIR = f"{BASE}/init"
DST_DIR = f"{BASE}/big{UPS}x"

IN_NZ   = _A.in_nz
IN_NYX  = _A.in_nyx
OUT_NZ  = IN_NZ  * UPS
OUT_NYX = IN_NYX * UPS

N_WRITE = _A.n_write
N_READ  = _A.n_read


# --------------------- GPU backend ---------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as _gpu_zoom
    _HAS_GPU = True
except Exception as _e:                                       # noqa: BLE001
    from PIL import Image
    _HAS_GPU = False
    rprint(f"[upsample] GPU disabled ({_e}); using CPU PIL BILINEAR")


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


# --------------------- upsample + linear-z blend --------------------------
if _HAS_GPU:
    def _upsample_xy(im_np: np.ndarray):
        """H2D + bilinear zoom by UPS in xy, returns cupy array."""
        im_d = cp.asarray(im_np)
        return _gpu_zoom(im_d, zoom=UPS, order=1, mode="nearest")

    def _blend_and_pull(up_curr_d, up_next_d, r: int) -> np.ndarray:
        if r == 0:
            out_d = up_curr_d
        else:
            t = cp.float32(r / UPS)
            out_d = (cp.float32(1.0) - t) * up_curr_d + t * up_next_d
        return cp.asnumpy(out_d)
else:
    def _upsample_xy(im_np: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(im_np, mode="F")
        return np.asarray(pil.resize((IN_NYX * UPS, IN_NYX * UPS),
                                     Image.Resampling.BILINEAR))

    def _blend_and_pull(up_curr, up_next, r: int) -> np.ndarray:
        if r == 0:
            return up_curr
        t = np.float32(r / UPS)
        return ((np.float32(1.0) - t) * up_curr +
                t * up_next).astype(np.float32, copy=False)


# --------------------- main -----------------------------------------------
def main() -> None:
    # GPU affinity is set externally (set_affinity_gpu.sh → CUDA_VISIBLE_DEVICES),
    # so cupy sees exactly one device per rank and we just use it.
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
    rprint(f"upsample: {UPS}×  method=trilinear (bilinear xy + linear z)  "
           f"backend={'GPU' if _HAS_GPU else 'CPU'}  "
           f"read={N_READ}  write={N_WRITE}  MPI ranks={SIZE}")
    rprint(f"estimated storage: "
           f"{OUT_NZ * OUT_NYX * OUT_NYX * 4 / 1e12:.2f} TB")

    dev_name, dev_id = "", ""
    if _HAS_GPU:
        try:
            dev_id   = cp.cuda.runtime.getDevice()
            dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)["name"].decode()
        except Exception:
            pass
    print(f"  rank {RANK}/{SIZE}: gpu={dev_id} ({dev_name}) "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}  "
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

    # -------- LINEAR in z (2-plane rolling buffer) ------------------------
    up_curr_d   = _upsample_xy(_read_plane(i_start))
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

    for f in pending:
        f.result()
    read_pool.shutdown()
    write_pool.shutdown()
    _barrier()
    rprint("upsample done.")


if __name__ == "__main__":
    main()
