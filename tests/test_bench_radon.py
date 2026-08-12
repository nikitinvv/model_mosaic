"""Stress-test TomoLargeReal.R() across problem sizes.

Sweeps UPS ∈ {1, 2, 4, 8, 16, 32, 64, 96, 128} and for each size runs
one R() call at nz=1 (matches how step2_radon_large.py uses it),
reporting the picked chunks, actual host RSS peak, GPU peak, and wall
time.  Stops on any exception (OOM, CUDA error, …).

Run:
    python -m tests.test_bench_radon
    python -m tests.test_bench_radon --ups-max 16 --gpu-budget-gb 20
"""
from __future__ import annotations

import argparse
import gc
import resource
import time
import traceback

import numpy as np
import cupy as cp

from processing.tomo_large   import TomoLargeReal
from processing.chunk_pick   import pick_tomo_chunks
from processing.pipeline     import free_pinned_pool


UPS_STEPS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
IN_NYX    = 3072                                  # matches step2 defaults


def _hb(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:6.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"


def _rss_peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def _reset_gpu() -> None:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    free_pinned_pool()


def bench_radon(ups: int, gpu_budget_gb: float) -> bool:
    n      = IN_NYX * ups
    nz     = 1
    ntheta = 3 * n // 4
    chunks = list(pick_tomo_chunks(n, ntheta, nz, int(gpu_budget_gb * 1e9)))

    print(f"\n--- TomoLargeReal.R   UPS={ups:2d}   N={n:6d}  NTHETA={ntheta:6d}  "
          f"nz={nz}   chunks={chunks}")

    theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype("float32")
    try:
        _reset_gpu()
        pool = cp.get_default_memory_pool()

        obj = np.full((nz, n, n), 0.5, dtype=np.float32)

        # Warmup R() covers pinned allocs + bin sort; second R() is the
        # timed one (all caches populated).
        t0 = time.perf_counter()
        tomo = TomoLargeReal(n, theta, chunks[2])       # chunks[2] = chunk_xy
        sino = tomo.R(obj, chunks)
        cp.cuda.runtime.deviceSynchronize()
        t_setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        sino = tomo.R(obj, chunks)
        cp.cuda.runtime.deviceSynchronize()
        t_run = time.perf_counter() - t0

        gpu_peak = pool.total_bytes()
        print(f"    SETUP+1st R={t_setup:6.2f}s   COMPUTE R={t_run:6.2f}s   "
              f"host RSS peak={_rss_peak_gb():5.1f} GB   "
              f"GPU peak={_hb(gpu_peak)}   sino[0,0,:4]={sino[0,0,:4]}")
        tomo.free()
        del sino, obj, tomo
    except Exception:
        print("    FAILED:")
        traceback.print_exc()
        return False
    finally:
        _reset_gpu()
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups-max", type=int, default=128)
    p.add_argument("--gpu-budget-gb", type=float, default=30.0,
                   help="target GPU memory for chunk-picker (default 30 GB)")
    args = p.parse_args()

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)["name"].decode()
    gpu_total = cp.cuda.runtime.memGetInfo()[1]
    print(f"[bench_radon]  GPU {dev_id}: {dev_name}   total={_hb(gpu_total)}   "
          f"chunker budget={args.gpu_budget_gb} GB")

    print("\n" + "=" * 78)
    print("TomoLargeReal.R  —  nz=1, N=3072·UPS, NTHETA=3·N/4")
    print("=" * 78)
    for ups in (u for u in UPS_STEPS if u <= args.ups_max):
        if not bench_radon(ups, args.gpu_budget_gb):
            print("[bench_radon]  stopping sweep")
            break


if __name__ == "__main__":
    main()
