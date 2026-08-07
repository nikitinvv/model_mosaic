#!/usr/bin/env python
"""Upsample a (2560, 2744, 2744) init volume by UPS× in each axis, saving
the result as a single HDF5 file at {path}/big{UPS}x.h5.

    /exchange/data   (2560·UPS, 2744·UPS, 2744·UPS) float32
                     chunks (1, 2744·UPS, 2744·UPS) — per-slice reads hit one chunk

Method: bilinear xy + linear z blend (trilinear, separable).  For each
input z, a background thread prefetches the next input plane while the
GPU upsamples in xy and blends UPS output planes between the current pair.

Multi-GPU via MPI (mpi4py optional).  GPU affinity is delegated to the
launcher: wrap with set_affinity_gpu.sh so each rank sees one GPU via
CUDA_VISIBLE_DEVICES.

    mpirun -n <NGPU> set_affinity_gpu.sh python step1_upsample.py --ups 4
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from h5_mpi_slab import check_chunk_bytes, mpiio_write_axis0


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
        k.setdefault("flush", True)
        print(*a, **k)


_H5_HAS_MPI = h5py.get_config().mpi
_H5_MPI_KW  = ({"driver": "mpio", "comm": _COMM}
               if _COMM is not None and _H5_HAS_MPI else {})


# --------------------- CLI -----------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ups",  type=int, default=4,
                   help="upsample factor (in every axis)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/init.h5, writes {path}/big{UPS}x.h5")
    p.add_argument("--in-nz",  type=int, default=2560, help="input nz")
    p.add_argument("--in-nyx", type=int, default=2744, help="input ny=nx")
    p.add_argument("--n-read",  type=int, default=2,
                   help="background input prefetchers per rank")
    return p.parse_args()


_A = _parse_args()

UPS     = _A.ups
BASE    = _A.path
SRC_H5  = f"{BASE}/init.h5"
DST_H5  = f"{BASE}/big{UPS}x.h5"

IN_NZ   = _A.in_nz
IN_NYX  = _A.in_nyx
OUT_NZ  = IN_NZ  * UPS
OUT_NYX = IN_NYX * UPS
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
def _read_plane(src_dset, zi: int) -> np.ndarray:
    im = src_dset[zi, :, :]
    if im.shape != (IN_NYX, IN_NYX):
        raise RuntimeError(
            f"unexpected shape {im.shape} at z={zi}, expected "
            f"({IN_NYX},{IN_NYX})")
    return im.astype(np.float32, copy=False)


# --------------------- upsample + linear-z blend --------------------------
if _HAS_GPU:
    def _upsample_xy(im_np: np.ndarray):
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
    # Collective create of big{UPS}x.h5 on all ranks (rank 0 also removes stale file first).
    if RANK == 0:
        if os.path.exists(DST_H5):
            os.remove(DST_H5)
    _barrier()

    dst_chunks = (1, OUT_NYX, OUT_NYX)
    check_chunk_bytes(dst_chunks, 4, label=f"{DST_H5}")
    with h5py.File(DST_H5, "w", **_H5_MPI_KW) as f:
        g = f.create_group("exchange")
        g.create_dataset("data", shape=(OUT_NZ, OUT_NYX, OUT_NYX),
                         dtype="float32",
                         chunks=dst_chunks)
    _barrier()

    # Partition input z-slices contiguously across ranks.
    per_rank = (IN_NZ + SIZE - 1) // SIZE
    i_start  = min(RANK * per_rank, IN_NZ)
    i_end    = min(i_start + per_rank, IN_NZ)
    local_n  = i_end - i_start

    rprint(f"input : {IN_NZ}×{IN_NYX}×{IN_NYX}   src={SRC_H5}")
    rprint(f"output: {OUT_NZ}×{OUT_NYX}×{OUT_NYX}  dst={DST_H5}")
    rprint(f"upsample: {UPS}×  method=trilinear (bilinear xy + linear z)  "
           f"backend={'GPU' if _HAS_GPU else 'CPU'}  "
           f"read={N_READ}  h5 mpi={_H5_HAS_MPI}  MPI ranks={SIZE}")
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

    with h5py.File(SRC_H5, "r", **_H5_MPI_KW) as fsrc, \
         h5py.File(DST_H5, "r+", **_H5_MPI_KW) as fdst, \
         ThreadPoolExecutor(max_workers=N_READ,
                            thread_name_prefix=f"r{RANK}-read") as read_pool:
        src_dset = fsrc["exchange/data"]
        dst_dset = fdst["exchange/data"]

        # 2-plane rolling buffer.  Prefetch next input plane in a background thread.
        up_curr_d   = _upsample_xy(_read_plane(src_dset, i_start))
        fut_next_np = (read_pool.submit(_read_plane, src_dset, i_start + 1)
                       if i_start + 1 < IN_NZ else None)

        t_read = t_upsample = t_write = 0.0

        for zi in range(i_start, i_end):
            t0 = time.perf_counter()
            if fut_next_np is not None:
                next_np     = fut_next_np.result()
                fut_next_np = (read_pool.submit(_read_plane, src_dset, zi + 2)
                               if zi + 2 < IN_NZ else None)
                t_read += time.perf_counter() - t0

                t0 = time.perf_counter()
                up_next_d = _upsample_xy(next_np)
                del next_np
                t_upsample += time.perf_counter() - t0
            else:
                up_next_d = up_curr_d          # end of volume: hold

            for r in range(UPS):
                out_z = zi * UPS + r
                t0 = time.perf_counter()
                plane = _blend_and_pull(up_curr_d, up_next_d, r)
                t_upsample += time.perf_counter() - t0

                # Single-plane write — slab helper is a no-op here (already <2 GiB
                # unless OUT_NYX > ~23000), but reuses the same code path.
                t0 = time.perf_counter()
                mpiio_write_axis0(dst_dset, out_z, out_z + 1, plane[None, ...])
                t_write += time.perf_counter() - t0

            up_curr_d = up_next_d

            done = zi - i_start + 1
            if done % 8 == 0 or done == local_n:
                print(f"  [rank {RANK}] input {done}/{local_n}  "
                      f"(wrote up to output slice {zi*UPS + UPS - 1})",
                      flush=True)

        print(f"  [rank {RANK}] timing: read={t_read:.1f}s "
              f"upsample={t_upsample:.1f}s write={t_write:.1f}s", flush=True)

    _barrier()
    rprint("upsample done.")


if __name__ == "__main__":
    main()
