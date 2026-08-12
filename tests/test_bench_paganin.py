"""Stress-test PaganinLarge.retrieve() across problem sizes.

Sweeps UPS ∈ {1, 2, 4, 8, 16, 32, 64, 96, 128} at ntheta=1 (matches
step7_paganin_large.py's per-call unit).  For each size we pick
(chunk_nz, chunk_n) that keep pass-1/3 strip and pass-2 column under
--gpu-budget-gb, then report actual host RSS peak, GPU peak, and wall
time.  Stops on any exception (OOM, CUDA error, …).

Run:
    python -m tests.test_bench_paganin
    python -m tests.test_bench_paganin --ups-max 16 --gpu-budget-gb 20
"""
from __future__ import annotations

import argparse
import gc
import resource
import time
import traceback

import numpy as np
import cupy as cp

from processing.paganin_large import PaganinLarge
from processing.chunk_pick    import pick_paganin_chunks
from processing.pipeline      import free_pinned_pool


UPS_STEPS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
IN_NZ, IN_NYX = 3072, 3072
ENERGY_KEV, VOXELSIZE, DISTANCE, ALPHA = 30.0, 1.38e-6, 1.0, 1e-3


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


def bench_paganin(ups: int, gpu_budget_gb: float) -> bool:
    n      = IN_NYX * ups
    nz     = IN_NZ  * ups
    ntheta = 1
    chunks = list(pick_paganin_chunks(nz, n, ntheta,
                                      int(gpu_budget_gb * 1e9)))
    chunk_nz, chunk_n = chunks

    print(f"\n--- PaganinLarge.retrieve   UPS={ups:2d}   N={n:6d}  NZ={nz:6d}   "
          f"ntheta={ntheta}   chunks=(chunk_nz={chunk_nz}, chunk_n={chunk_n})")

    wavelength = 1.24e-9 / ENERGY_KEV
    try:
        _reset_gpu()
        pool = cp.get_default_memory_pool()

        # Small perturbation around 1.0 (matches Paganin input semantics —
        # intensity ≈ 1 with weak absorption/phase modulation).
        intens = np.full((ntheta, nz, n), 0.95, dtype=np.float32)

        t0 = time.perf_counter()
        pgn = PaganinLarge(n, nz, wavelength, VOXELSIZE, DISTANCE, ALPHA)
        out = pgn.retrieve(intens, chunks)
        cp.cuda.runtime.deviceSynchronize()
        t_setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = pgn.retrieve(intens, chunks)
        cp.cuda.runtime.deviceSynchronize()
        t_run = time.perf_counter() - t0

        gpu_peak = pool.total_bytes()
        print(f"    SETUP+1st retrieve={t_setup:6.2f}s   COMPUTE retrieve={t_run:6.2f}s   "
              f"host RSS peak={_rss_peak_gb():5.1f} GB   "
              f"GPU peak={_hb(gpu_peak)}   out[0,0,:4]={out[0,0,:4]}")
        if hasattr(pgn, "free"):
            pgn.free()
        del out, intens, pgn
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
    print(f"[bench_paganin]  GPU {dev_id}: {dev_name}   total={_hb(gpu_total)}   "
          f"chunker budget={args.gpu_budget_gb} GB")

    print("\n" + "=" * 78)
    print("PaganinLarge.retrieve  —  ntheta=1, N=3072·UPS, NZ=3072·UPS")
    print("=" * 78)
    for ups in (u for u in UPS_STEPS if u <= args.ups_max):
        if not bench_paganin(ups, args.gpu_budget_gb):
            print("[bench_paganin]  stopping sweep")
            break


if __name__ == "__main__":
    main()
