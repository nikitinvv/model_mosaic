"""Stress-test filtered backprojection (FBPFilter + TomoLargeReal.RT).

Sweeps UPS ∈ {1, 2, 4, 8, 16, 32, 64, 96, 128} at nz=1 (matches
step8_fbp_large.py's default --nzchunk=1).  For each size we build the
padded shepp filter on the GPU, apply it to a (ntheta, 1, N) sinogram,
and backproject via TomoLargeReal.RT with chunks sized from
--gpu-budget-gb, then report actual host RSS peak, GPU peak, wall time.

Note: TomoLargeReal.RT keeps a full-complex64 path internally (pragmatic
port; rfft-aware adjoint scatter is deferred).  This bench measures the
full-complex RT with the ky-band-chunked scatter that bounds GPU peak.

Stops on any exception (OOM, CUDA error, …).  Run:
    python -m tests.test_bench_fbp
    python -m tests.test_bench_fbp --ups-max 16 --gpu-budget-gb 20
"""
from __future__ import annotations

import argparse
import gc
import resource
import time
import traceback

import numpy as np
import cupy as cp

from processing.tomo_large import TomoLargeReal
from processing.fbp_filter import FBPFilter
from processing.chunk_pick import pick_tomo_chunks
from processing.pipeline   import free_pinned_pool


UPS_STEPS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
IN_NYX    = 3072


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


def bench_fbp(ups: int, gpu_budget_gb: float) -> bool:
    n      = IN_NYX * ups
    nz     = 1
    ntheta = 3 * n // 4
    chunks = list(pick_tomo_chunks(n, ntheta, nz,
                                   int(gpu_budget_gb * 1e9)))
    chunk_n, chunk_theta, chunk_xy = chunks

    print(f"\n--- FBP (FBPFilter + TomoLargeReal.RT)   UPS={ups:2d}   N={n:6d}  "
          f"NTHETA={ntheta:6d}   nz={nz}   "
          f"chunks=(n={chunk_n}, theta={chunk_theta}, xy={chunk_xy})")

    theta = np.linspace(0, np.pi, ntheta, endpoint=False).astype("float32")
    try:
        _reset_gpu()
        pool = cp.get_default_memory_pool()

        # Synthetic sinogram close in shape to a real Paganin output.
        sino_h = np.full((ntheta, nz, n), -0.05, dtype=np.float32)

        t0 = time.perf_counter()
        cl_filter = FBPFilter(n)
        w_gpu     = cl_filter.calc_filter('shepp')
        cl_tomo   = TomoLargeReal(n, theta, chunk_xy)
        # Warmup: allocates pinned buffers + runs the sort.  Use
        # filter_host so the H2D roundtrip is also chunked — at UPS≥32
        # the full sino (29 GB f32 at UPS=32, 116 GB at UPS=64) does
        # not fit on the GPU as one cp.asarray call.
        sino_filt_h = sino_h.copy()
        cl_filter.filter_host(sino_filt_h, w_gpu)
        rec = cl_tomo.RT(sino_filt_h, chunks)
        cp.cuda.runtime.deviceSynchronize()
        t_setup = time.perf_counter() - t0

        # Timed call: filter and RT separately.
        sino_filt_h = sino_h.copy()
        t0 = time.perf_counter()
        cl_filter.filter_host(sino_filt_h, w_gpu)
        cp.cuda.runtime.deviceSynchronize()
        t_filter = time.perf_counter() - t0

        t0 = time.perf_counter()
        rec = cl_tomo.RT(sino_filt_h, chunks)
        cp.cuda.runtime.deviceSynchronize()
        t_rt = time.perf_counter() - t0
        t_run = t_filter + t_rt

        gpu_peak = pool.total_bytes()
        rec_arr = np.asarray(rec)
        print(f"    SETUP+1st={t_setup:6.2f}s   "
              f"filter={t_filter:6.2f}s  RT={t_rt:6.2f}s  total={t_run:6.2f}s   "
              f"host RSS peak={_rss_peak_gb():5.1f} GB   "
              f"GPU peak={_hb(gpu_peak)}   rec[0,N/2,:4]={rec_arr[0, n//2, :4]}")
        cl_tomo.free() if hasattr(cl_tomo, 'free') else None
        del rec, rec_arr, sino_filt_h, sino_h, cl_tomo, cl_filter
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
    print(f"[bench_fbp]  GPU {dev_id}: {dev_name}   total={_hb(gpu_total)}   "
          f"chunker budget={args.gpu_budget_gb} GB")

    print("\n" + "=" * 78)
    print("FBP (FBPFilter + TomoLargeReal.RT)  —  nz=1, N=3072·UPS, NTHETA=3·N/4")
    print("=" * 78)
    for ups in (u for u in UPS_STEPS if u <= args.ups_max):
        if not bench_fbp(ups, args.gpu_budget_gb):
            print("[bench_fbp]  stopping sweep")
            break


if __name__ == "__main__":
    main()
