#!/usr/bin/env python
"""Upsample a (2560, 2744, 2744) init volume by UPS× in each axis, saving
the result as a VDS+banks HDF5 store at {path}/big{UPS}x.h5.

    {path}/big{UPS}x.h5              VDS master
    {path}/big{UPS}x/big{UPS}x_data_*.h5   bank files
        /exchange/data   (2560·UPS, 2744·UPS, 2744·UPS) float32
                         chunks (1, 2744·UPS, 2744·UPS)

Method: bilinear xy + linear z blend (trilinear, separable).  For each
input z, a background thread prefetches the next input plane while the
GPU upsamples in xy and blends UPS output planes between the current pair.

Each rank owns a subset of super-chunks (--big-vchunks) round-robin
across ranks, fills a shared-memory buffer of that shape by looping
compute over its input z-range, then tomo_writex fans the buffer across
--nbanks bank files.  Reads of init.h5 go through its VDS master via
plain h5py (transparent).

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

from iohdf5.dxchange_hdf5_chunks import tomo_writex
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, iter_vchunks,
    describe_input, describe_output,
    vchunk_bytes,
)
from utils import COMM, RANK, SIZE, barrier, rprint, report_stage


# --------------------- CLI -----------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ups",  type=int, default=4,
                   help="upsample factor (in every axis)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base directory; reads {path}/init.h5, writes {path}/big{UPS}x.h5")
    p.add_argument("--in-nz", type=int, default=2560, help="input nz")
    p.add_argument("--in-n",  type=int, default=2744,
                   help="input ny=nx (same name as step2/3 --in-n)")
    p.add_argument("--n-read",  type=int, default=2,
                   help="background input prefetchers per rank")
    p.add_argument("--nbanks",  type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--big-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk shape for big{UPS}x.h5 (default: "
                        "8·UPS, OUT_NYX, OUT_NYX; RAM buffer = C0·C1·C2·4 bytes)")
    return p.parse_args()


_A = _parse_args()

UPS     = _A.ups
BASE    = _A.path
SRC_H5  = f"{BASE}/init.h5"
DST_H5  = f"{BASE}/big{UPS}x.h5"

IN_NZ   = _A.in_nz
IN_NYX  = _A.in_n
OUT_NZ  = IN_NZ  * UPS
OUT_NYX = IN_NYX * UPS
N_READ  = _A.n_read
NBANKS  = _A.nbanks
BIG_VCHUNKS = tuple(_A.big_vchunks) if _A.big_vchunks else (8 * UPS, OUT_NYX, OUT_NYX)


# --------------------- GPU backend ---------------------------------------
import cupy as cp
from cupyx.scipy.ndimage import zoom as _gpu_zoom


def _read_plane(src_dset, zi: int) -> np.ndarray:
    im = src_dset[zi, :, :]
    if im.shape != (IN_NYX, IN_NYX):
        raise RuntimeError(
            f"unexpected shape {im.shape} at z={zi}, expected "
            f"({IN_NYX},{IN_NYX})")
    return im.astype(np.float32, copy=False)


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


# --------------------- main -----------------------------------------------
def main() -> None:
    if BIG_VCHUNKS[0] % UPS != 0:
        raise SystemExit(
            f"--big-vchunks C0={BIG_VCHUNKS[0]} must be a multiple of "
            f"UPS={UPS} so vchunk boundaries align with input planes.")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(DST_H5, (OUT_NZ, OUT_NYX, OUT_NYX), np.float32,
                        BIG_VCHUNKS, "proj", NBANKS)

    ctx = initx_and_bcast(DST_H5, shape=(OUT_NZ, OUT_NYX, OUT_NYX),
                          dtype=np.float32, vchunks=BIG_VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=RANK, comm=COMM)

    buf_gb = vchunk_bytes(BIG_VCHUNKS, np.float32) / 1e9
    rprint(f"upsample: {UPS}×  method=trilinear (bilinear xy + linear z)  "
           f"read={N_READ}  MPI ranks={SIZE}  per-rank shm buffer={buf_gb:.2f} GB")

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)["name"].decode()
    print(f"  rank {RANK}/{SIZE}: gpu={dev_id} ({dev_name}) "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()

    # Round-robin vchunk sharding — matches test_h5_buffer_io.py.
    ivchunks = list(iter_vchunks((OUT_NZ, OUT_NYX, OUT_NYX), BIG_VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]

    shm, buf = alloc_shm(BIG_VCHUNKS, np.float32)
    try:
        with h5py.File(SRC_H5, "r") as fsrc, \
             ThreadPoolExecutor(max_workers=N_READ,
                                thread_name_prefix=f"r{RANK}-read") as read_pool:
            # init.h5 may follow the holotomo convention (/exchange/data,
            # written by step00_upsample_extract.py) or the tomoscan
            # convention (/data at root, from tomopy/dxchange).  Accept
            # either.  The output big{UPS}x.h5 always uses /exchange/data.
            if "exchange/data" in fsrc:
                src_dset = fsrc["exchange/data"]
            elif "data" in fsrc:
                src_dset = fsrc["data"]
            else:
                raise SystemExit(
                    f"{SRC_H5}: neither /exchange/data nor /data found.  "
                    f"Root groups: {list(fsrc.keys())}")

            t_read = t_upsample = t_write = 0.0
            b_read = b_write = 0
            for k, ivc in enumerate(my_ivchunks, start=1):
                # Output z-range for this vchunk.
                z0_out = ivc[0] * BIG_VCHUNKS[0]
                z1_out = min(z0_out + BIG_VCHUNKS[0], OUT_NZ)
                # Input planes needed to fill it (integer division; C0%UPS==0).
                z0_in  = z0_out // UPS
                z1_in  = (z1_out + UPS - 1) // UPS

                buf.fill(0)  # zero any tail padding

                # 2-plane rolling upsample, local to this vchunk.
                t0 = time.perf_counter()
                up_curr_d = _upsample_xy(_read_plane(src_dset, z0_in))
                fut_next  = (read_pool.submit(_read_plane, src_dset, z0_in + 1)
                             if z0_in + 1 < IN_NZ else None)
                t_read += time.perf_counter() - t0
                b_read += IN_NYX * IN_NYX * 4    # one input plane

                for zi in range(z0_in, z1_in):
                    if fut_next is not None:
                        t0 = time.perf_counter()
                        next_np  = fut_next.result()
                        fut_next = (read_pool.submit(_read_plane, src_dset, zi + 2)
                                    if zi + 2 < IN_NZ else None)
                        t_read += time.perf_counter() - t0
                        b_read += IN_NYX * IN_NYX * 4
                        t0 = time.perf_counter()
                        up_next_d = _upsample_xy(next_np)
                        del next_np
                        t_upsample += time.perf_counter() - t0
                    else:
                        up_next_d = up_curr_d      # end of volume

                    for r in range(UPS):
                        out_z = zi * UPS + r
                        if not (z0_out <= out_z < z1_out):
                            continue
                        t0 = time.perf_counter()
                        plane = _blend_and_pull(up_curr_d, up_next_d, r)
                        t_upsample += time.perf_counter() - t0
                        buf[out_z - z0_out] = plane

                    up_curr_d = up_next_d

                # Fan the buffer across nbanks bank files.
                t0 = time.perf_counter()
                tomo_writex(DST_H5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
                t_write += time.perf_counter() - t0
                b_write += (z1_out - z0_out) * OUT_NYX * OUT_NYX * 4

                print(f"  [rank {RANK}] vchunk {k}/{len(my_ivchunks)}  "
                      f"z_out=[{z0_out},{z1_out})  "
                      f"(read={t_read:.1f}s upsample={t_upsample:.1f}s "
                      f"write={t_write:.1f}s)", flush=True)
    finally:
        free_shm(shm)

    barrier()
    report_stage("step1 read (init)",  b_read,  t_read)
    report_stage("step1 write (big)",  b_write, t_write)
    rprint("upsample done.")


if __name__ == "__main__":
    from utils import run_main
    run_main(main)
