#!/usr/bin/env python
"""Per-tile preprocessing: dezinger + dark-flat field correction + FW ring removal.

Reads the per-tile HDF5 files produced by step4_extract.py at
    {path}/mosaic_h5/{zi}_{xi}.h5
and writes cleaned tiles (same schema) to
    {path}/mosaic_h5_pre/{zi}_{xi}.h5

For every tile:
  1. dezinger  — cupyx.scipy.ndimage.median_filter with a (w, 1, w) footprint,
     replaces outliers where (data - median) > threshold.  Skipped when
     --dezinger 0 (default — the simulated data_white=1 / data_dark=0 tiles
     produced by step4_extract have no zingers).
  2. dark-flat correction — (data - dark_mean) / (flat_mean - dark_mean) with
     the same +1e-5 flat guard tomocupy uses.  With synthetic flats/darks
     it's numerically a no-op, but the code path is exercised so real
     acquisitions drop in unchanged.
  3. FW ring removal — processing.remove_stripe.remove_stripe_fw (vendored
     wavelet-FFT method from tomocupy.processing.remove_stripe).  Needs the
     FULL θ axis at once, so tiles are streamed through the GPU row-chunk
     at a time (--nzchunk rows per pass).

Output tile schema matches step4_extract.py byte-for-byte:
    /exchange/data        (NTHETA_TILE, h, w) float32   chunks=(1, h, w)
    /exchange/theta       (NTHETA_TILE,)      float32
    /exchange/data_white  (1, h, w)           float32   ones
    /exchange/data_dark   (1, h, w)           float32   zeros
    all step4 tile attrs preserved on /exchange/data.

Multi-rank via MPI: tiles are round-robin sharded across ranks (same pattern
as step4_extract).  Multi-GPU via `set_affinity_gpu.sh` — one GPU per rank.

Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step5_correct.py \\
        --ups 1 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import median_filter

from mpi_utils import RANK, SIZE, barrier as _barrier, rprint, report_stage
from processing.remove_stripe import remove_stripe_fw
from step4_extract import read_placements


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POSITIONS_DIR = os.path.join(_SCRIPT_DIR, "mosaic_positions")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="matches step4_extract --ups (drives src/dst paths + positions file)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/mosaic_h5/*.h5, "
                        "writes {path}/mosaic_h5_pre/*.h5")
    p.add_argument("--dezinger", type=int, default=0,
                   help="median-filter footprint (odd int); 0 disables (default 0 "
                        "— simulated tiles have no zingers)")
    p.add_argument("--dezinger-threshold", type=float, default=1000.0,
                   help="pixels with (data - median) > threshold get replaced")
    p.add_argument("--fw-sigma",  type=float, default=2.0,
                   help="FW ring removal sigma (per-band Gaussian damping)")
    p.add_argument("--fw-wname",  type=str,   default="sym16",
                   help="FW wavelet name (pywt Wavelet id)")
    p.add_argument("--fw-level",  type=int,   default=7,
                   help="FW wavelet decomposition levels")
    p.add_argument("--nzchunk",   type=int,   default=8,
                   help="rows per GPU pass (FW requires full θ, so we only "
                        "chunk over nz)")
    return p.parse_args()


def _dezinger(x: cp.ndarray, w: int, thr: float) -> cp.ndarray:
    """Median-filter based outlier replacement (mirrors tomocupy's
    ProcFunctions.remove_outliers).  In-place; returns x."""
    if w <= 0:
        return x
    if x.ndim == 3:
        med = median_filter(x, [w, 1, w])
    else:
        med = median_filter(x, [w, w])
    x[:] = cp.where(cp.logical_and(x > med, (x - med) > thr), med, x)
    return x


def _darkflat(data: cp.ndarray, dark: cp.ndarray, flat: cp.ndarray) -> cp.ndarray:
    """(data - dark_mean) / (flat_mean - dark_mean), matching tomocupy's
    ProcFunctions.darkflat_correction with bright_ratio=1 and
    flat_linear=False.  All arrays float32 on GPU."""
    dark0 = cp.mean(dark, axis=0)
    flat0 = cp.mean(flat, axis=0) * cp.float32(1 + 1e-5)
    flat0 -= dark0
    return (data - dark0) / flat0


def main() -> None:
    args = _parse_args()

    from mpi_utils import banner
    banner("5", "mosaic_h5/*.h5 -> mosaic_h5_pre/*.h5  "
                "(dezinger + dark-flat + FW ring removal)")

    src_dir = f"{args.path}/mosaic_h5"
    dst_dir = f"{args.path}/mosaic_h5_pre"
    positions_path = os.path.join(_POSITIONS_DIR,
                                  f"mosaic_positions{args.ups}.txt")

    if not os.path.isdir(src_dir):
        raise SystemExit(f"missing input dir: {src_dir}")
    if not os.path.exists(positions_path):
        raise SystemExit(
            f"missing positions file: {positions_path}\n"
            f"run `python step0_schematic.py --ups {args.ups}` first.")

    if RANK == 0:
        os.makedirs(dst_dir, exist_ok=True)
    _barrier()

    meta, placements = read_placements(positions_path)
    n_z, n_x = meta["n_z"], meta["n_x"]
    NTHETA_TILE = meta["NTHETA"]

    # GPU affinity (set by set_affinity_gpu.sh via CUDA_VISIBLE_DEVICES).
    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    _barrier()

    if RANK == 0:
        print(f"  IN : {src_dir}/{{z}}_{{x}}.h5   ({n_z*n_x} tiles)")
        print(f"  OUT: {dst_dir}/{{z}}_{{x}}.h5   ({n_z*n_x} tiles)")
        print(f"       NTHETA={NTHETA_TILE}   nzchunk={args.nzchunk}   "
              f"dezinger={args.dezinger} (thr={args.dezinger_threshold})")
        print(f"       fw: sigma={args.fw_sigma} wname={args.fw_wname} "
              f"level={args.fw_level}", flush=True)

    # Round-robin tile sharding (same pattern as step4_extract).
    my_placements = placements[RANK::SIZE]

    t_read = t_comp = t_write = 0.0
    b_read = b_write = 0

    for p in my_placements:
        zi, xi = p["zi"], p["xi"]
        in_path  = os.path.join(src_dir, f"{zi}_{xi}.h5")
        out_path = os.path.join(dst_dir, f"{zi}_{xi}.h5")
        if not os.path.exists(in_path):
            raise SystemExit(f"missing tile: {in_path}")

        with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
            din = fin["exchange/data"]
            dark_in  = fin["exchange/data_dark"]     # (1, h, w) zeros
            flat_in  = fin["exchange/data_white"]    # (1, h, w) ones
            theta_in = fin["exchange/theta"]
            nt, h, w = din.shape
            if nt != NTHETA_TILE:
                raise SystemExit(
                    f"tile {in_path} has NTHETA={nt} but positions expect {NTHETA_TILE}")

            g = fout.create_group("exchange")
            dout = g.create_dataset("data",
                                    shape=(nt, h, w), dtype=np.float32,
                                    chunks=(1, h, w))
            g.create_dataset("theta", data=theta_in[...])
            g.create_dataset("data_white",
                             data=np.ones((1, h, w), dtype=np.float32))
            g.create_dataset("data_dark",
                             data=np.zeros((1, h, w), dtype=np.float32))
            # Preserve every step4 attribute on the data dataset.
            for k, v in din.attrs.items():
                dout.attrs[k] = v

            # Load flat/dark once — they're tiny (1, h, w).
            t0 = time.perf_counter()
            dark_full_h = dark_in[...]              # (1, h, w) f32
            flat_full_h = flat_in[...]              # (1, h, w) f32
            t_read += time.perf_counter() - t0
            b_read += dark_full_h.nbytes + flat_full_h.nbytes
            dark_full_d = cp.asarray(dark_full_h, dtype=cp.float32)
            flat_full_d = cp.asarray(flat_full_h, dtype=cp.float32)

            for r0 in range(0, h, args.nzchunk):
                r1 = min(r0 + args.nzchunk, h)

                t0 = time.perf_counter()
                slab_h = din[:, r0:r1, :]          # (nt, nz, w) f32
                t_read += time.perf_counter() - t0
                b_read += slab_h.nbytes

                t0 = time.perf_counter()
                slab_d  = cp.asarray(slab_h, dtype=cp.float32)
                dark_d  = dark_full_d[:, r0:r1, :]
                flat_d  = flat_full_d[:, r0:r1, :]

                if args.dezinger > 0:
                    _dezinger(slab_d, args.dezinger, args.dezinger_threshold)
                    _dezinger(dark_d, args.dezinger, args.dezinger_threshold)
                    _dezinger(flat_d, args.dezinger, args.dezinger_threshold)

                corr_d = _darkflat(slab_d, dark_d, flat_d)
                corr_d = remove_stripe_fw(corr_d, args.fw_sigma,
                                          args.fw_wname, args.fw_level)

                out_h = cp.asnumpy(corr_d)
                del slab_d, corr_d
                cp.get_default_memory_pool().free_all_blocks()
                t_comp += time.perf_counter() - t0

                t0 = time.perf_counter()
                dout[:, r0:r1, :] = out_h
                t_write += time.perf_counter() - t0
                b_write += out_h.nbytes
                del out_h, slab_h

            del dark_full_d, flat_full_d
            cp.get_default_memory_pool().free_all_blocks()

        print(f"  [rank {RANK}] {out_path}  shape=({nt},{h},{w})  "
              f"nzchunk={args.nzchunk}  "
              f"(read={t_read:.1f}s comp={t_comp:.1f}s write={t_write:.1f}s)",
              flush=True)

    _barrier()
    report_stage("step5 read (tiles)",  b_read,  t_read)
    report_stage("step5 comp (dezinger+darkflat+fw)", b_read + b_write, t_comp)
    report_stage("step5 write (tiles)", b_write, t_write)
    rprint(f"done. wrote {n_z*n_x} corrected h5 files to {dst_dir}.")


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
