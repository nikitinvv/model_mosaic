"""Stress-test TomoLargeReal and PropagationLarge across problem sizes.

The two "large" variants stage work through the GPU while keeping the
big buffers on the host — so the hard limit is host RAM, not GPU RAM.
This script sweeps UPS ∈ {1, 2, 4, 8, 16, 24, 32} and, for each size,
runs a SINGLE per-call unit (matching how step2_radon_large.py and a
future step3_fresnel_large.py would use them):

  * TomoLargeReal      — R() with  nz = 1        (one z-slice, float32
                                                   obj + rfft-half fde)
  * PropagationLarge   — D() with  ntheta = 1    (one θ angle)

For each size we print:
    N, NZ, NTHETA, auto-picked chunks, estimated & actual host RSS,
    GPU peak memory, wall time.

Stops the sweep when the estimated host RAM would exceed the caller's
--host-cap-gb (default: MemAvailable from /proc/meminfo minus a 10%
safety margin).  Also stops on any exception (OOM, CUDA error, ...).

Run:
    python -m tests.test_bench_large               # both stages
    python -m tests.test_bench_large --stage tomo
    python -m tests.test_bench_large --stage prop --ups-max 16
    python -m tests.test_bench_large --host-cap-gb 100
"""
from __future__ import annotations

import argparse
import gc
import resource
import time
import traceback

import numpy as np
import cupy as cp

from processing.tomo_large        import TomoLargeReal
from processing.propagation_large import PropagationLarge
from processing.chunk_pick        import pick_tomo_chunks, pick_prop_chunks
from processing.pipeline          import free_pinned_pool


# ---------- knobs -----------------------------------------------------------
# Tomo5 has ~4 TB host RAM, so the host cap almost never bites at these
# sizes.  Sweep goes past UPS=64 for TomoLarge and past UPS=128 for
# PropLarge; the per-chunk GPU estimates in the printout show how close
# each stage is to the 40 GB GPU budget.
UPS_STEPS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
IN_NZ, IN_NYX = 3072, 3072                # matches step2/step3 defaults
WAVELENGTH, VOXELSIZE, DISTANCE = 4.13e-11, 1.4e-6, 1.0
GPU_BUDGET_GB = 40.0                      # A100 40 GB — for advisory print


# ---------- helpers ---------------------------------------------------------
def _hb(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:6.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"


def _mem_available_gb() -> float | None:
    """Read MemAvailable from /proc/meminfo (Linux); None if unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024**2)   # kB → GB
    except OSError:
        pass
    return None


def _rss_peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _pick_tomo_chunks(n, ntheta, nz, gpu_budget_bytes):
    return list(pick_tomo_chunks(n, ntheta, nz, gpu_budget_bytes))


def _pick_prop_chunks(nz, n, ntheta, gpu_budget_bytes):
    return list(pick_prop_chunks(nz, n, ntheta, gpu_budget_bytes))


def _reset_gpu() -> None:
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    free_pinned_pool()


# ---------- benches ---------------------------------------------------------
def bench_tomo(ups: int, host_cap_gb: float, gpu_budget_gb: float) -> bool:
    """Return True if the size ran; False if skipped or failed."""
    n      = IN_NYX * ups
    nz     = 1
    ntheta = 3 * n // 4
    chunks = _pick_tomo_chunks(n, ntheta, nz, int(gpu_budget_gb * 1e9))
    c_n, c_theta, c_xy = chunks

    # TomoLargeReal host buffers.  After the 1-D phi / 1-D c2dfftshift
    # refactor and cached _sort_into_chunks (see processing/tomo_large.py),
    # the only persistent host tables are the three big pinned arrays plus
    # the sort output (idx int64) and StreamPipe ping-pong buffers.
    #   obj  (nz, n,  n)     f32  — pinned via TomoLargeReal.obj_buffer()
    #   fde  (nz, 2n, n+1)   c64  — pinned rfft half spectrum
    #   sino (ntheta, nz, n) c64  — pinned Pass-3 output
    #     sino_real (Pass-4 f32) is a VIEW into sino's first half — no
    #     separate allocation.
    obj_bytes  = nz *          n         * n     * 4
    fde_bytes  = nz *          (2 * n)   * (n + 1) * 8
    sino_bytes = nz * ntheta * n         * 8
    # Cached sort result (idx int64 for stability of ntheta·n indexing).
    sort_bytes = ntheta * n * 8
    # StreamPipe pinned ping-pong buffers.  The three staged passes share
    # a single (in, out) scratch pool sized to max across passes — each
    # side is 2 pinned byte buffers grown to fit the largest pass.
    pipe1_in   = nz * c_n     * n           * 4          # f32  Pass 1 in
    pipe1_out  = nz * c_n     * (n + 1)     * 8          # c64  Pass 1 out
    pipe2_in   = nz * (2 * n) * c_n         * 8          # c64  Pass 2 in
    pipe2_out  = nz * (2 * n) * c_n         * 8          # c64  Pass 2 out
    pipe4_in   = c_theta * nz * n           * 8          # c64  Pass 4 in
    pipe4_out  = c_theta * nz * n           * 4          # f32  Pass 4 out
    scratch_in_bytes  = max(pipe1_in,  pipe2_in,  pipe4_in)
    scratch_out_bytes = max(pipe1_out, pipe2_out, pipe4_out)
    pipe_bytes = 2 * (scratch_in_bytes + scratch_out_bytes)
    est_gb     = (obj_bytes + fde_bytes + sino_bytes
                  + sort_bytes + pipe_bytes) / 1e9

    # Per-chunk GPU peaks.  Same picker as TomoLarge (shapes are similar);
    # rfft-half fde halves Pass 2's live buffer but Pass 1's padded 2n
    # working array is still the widest.
    m_guess   = 4       # USFFT halo — stays ~4 for our N range
    gpu_xfft  = nz * c_n     * (2 * n)             * 4      # padded float32
    gpu_yfft  = nz * (2 * n) *  c_n                * 8      # complex64
    gpu_gath  = nz * (c_xy + 2 * m_guess + 1) ** 2 * 8
    gpu_ifft  = c_theta * nz *  n                  * 8
    gpu_peak_est = max(gpu_xfft, gpu_yfft, gpu_gath, gpu_ifft)

    print(f"\n--- TomoLargeReal   UPS={ups:2d}   N={n:6d}  NTHETA={ntheta:6d}  nz={nz}")
    print(f"    chunks={chunks}   est host RAM ≈ {est_gb:.1f} GB   "
          f"(fde={_hb(fde_bytes)}, sino={_hb(sino_bytes)}, "
          f"obj={_hb(obj_bytes)}, sort={_hb(sort_bytes)}, "
          f"pipes={_hb(pipe_bytes)})")
    print(f"    live-buffer per stage (~1/5 of actual peak): {_hb(gpu_peak_est)}  "
          f"[x-FFT={_hb(gpu_xfft)}, y-FFT={_hb(gpu_yfft)}, "
          f"gather={_hb(gpu_gath)}, IFFT={_hb(gpu_ifft)}]")
    if est_gb > host_cap_gb:
        print(f"    SKIP (host estimate exceeds cap {host_cap_gb:.1f} GB)")
        return False
    if gpu_peak_est > GPU_BUDGET_GB * 1e9:
        print(f"    WARN: per-chunk GPU estimate exceeds {GPU_BUDGET_GB} GB budget")

    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")
    try:
        _reset_gpu()
        pool = cp.get_default_memory_pool()

        # Constant float32 fill — bench only measures wall time; the actual
        # values don't matter as long as they're not denormals.  np.full is
        # a single write, no float64 temp (np.random.rand would double the
        # peak RSS during creation at UPS=32).
        obj = np.full((nz, n, n), 0.5, dtype=np.float32)

        # Split timing: construction + first R() include all one-time pinned
        # allocs (fde/sino/pipes) and the theta-major bin sort — none of
        # that scales with the number of R() calls in a real run.  The
        # second R() reuses every cached buffer and skips the sort, so it
        # measures the actual per-call compute + H2D/D2H cost.
        t0 = time.perf_counter()
        tomo = TomoLargeReal(n, theta)
        sino = tomo.R(obj, chunks)                  # warmup: allocates
        cp.cuda.runtime.deviceSynchronize()
        t_setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        sino = tomo.R(obj, chunks)                  # timed: load + compute
        cp.cuda.runtime.deviceSynchronize()
        t_run = time.perf_counter() - t0

        gpu_peak = pool.total_bytes()   # allocated pool footprint — close to peak
        print(f"    SETUP+1st R={t_setup:6.2f}s   COMPUTE R={t_run:6.2f}s   "
              f"host RSS peak={_rss_peak_gb():5.1f} GB   "
              f"GPU peak={_hb(gpu_peak)}   sino[0,0,:4]={sino[0,0,:4]}")
        tomo.free()               # release pinned/GPU buffers before del
        del sino, obj, tomo
    except Exception:
        print("    FAILED:")
        traceback.print_exc()
        return False
    finally:
        _reset_gpu()
    return True


def bench_prop(ups: int, host_cap_gb: float, gpu_budget_gb: float) -> bool:
    n      = IN_NYX * ups
    nz     = IN_NZ  * ups
    ntheta = 1
    chunks = _pick_prop_chunks(nz, n, ntheta, int(gpu_budget_gb * 1e9))

    c_nz, c_2n = chunks

    psi_bytes = ntheta * nz *  n      * 8
    fde_bytes = ntheta * nz * (2 * n) * 8
    out_bytes = psi_bytes
    # StreamPipe pinned ping-pong buffers (in + out × 2 each) for the
    # three passes.  Non-trivial at UPS≥16.
    pipe1_in  = 2 * ntheta * c_nz     * n           * 8
    pipe1_out = 2 * ntheta * c_nz     * (2 * n)     * 8
    pipe2_io  = 2 * ntheta * nz       * c_2n        * 8 * 2
    pipe3_in  = 2 * ntheta * c_nz     * (2 * n)     * 8
    pipe3_out = 2 * ntheta * c_nz     * n           * 8
    pipe_bytes = pipe1_in + pipe1_out + pipe2_io + pipe3_in + pipe3_out
    est_gb    = (psi_bytes + fde_bytes + out_bytes + pipe_bytes) / 1e9

    gpu_pass1 = ntheta * c_nz * (2 * n) * 8      # (nt, chunk_nz, 2n) pad+FFT
    gpu_pass2 = ntheta * (2 * nz) * c_2n * 8     # (nt, 2nz, chunk_2n) pad+FFT+K+IFFT
    gpu_pass3 = ntheta * c_nz * (2 * n) * 8      # (nt, chunk_nz, 2n) IFFT
    gpu_peak_est = max(gpu_pass1, gpu_pass2, gpu_pass3)

    print(f"\n--- PropLarge   UPS={ups:2d}   N={n:6d}  NZ={nz:6d}      ntheta={ntheta}")
    print(f"    chunks={chunks}   est host RAM ≈ {est_gb:.1f} GB   "
          f"(psi={_hb(psi_bytes)}, fde={_hb(fde_bytes)}, out={_hb(out_bytes)}, "
          f"pipes={_hb(pipe_bytes)})")
    print(f"    live-buffer per stage (~1/5 of actual peak): {_hb(gpu_peak_est)}  "
          f"[Pass1={_hb(gpu_pass1)}, Pass2={_hb(gpu_pass2)}, Pass3={_hb(gpu_pass3)}]")
    if est_gb > host_cap_gb:
        print(f"    SKIP (host estimate exceeds cap {host_cap_gb:.1f} GB)")
        return False
    if gpu_peak_est > GPU_BUDGET_GB * 1e9:
        print(f"    WARN: per-chunk GPU estimate exceeds {GPU_BUDGET_GB} GB budget")

    try:
        _reset_gpu()
        pool = cp.get_default_memory_pool()

        # Constant fill — bench only measures wall time (see bench_tomo).
        psi = np.empty((ntheta, nz, n), dtype=np.complex64)
        psi.real[...] = 0.5
        psi.imag[...] = 0.3

        # Same warmup pattern as bench_tomo: construction + first D() do the
        # one-time pinned allocations; second D() measures compute + H2D/D2H.
        t0 = time.perf_counter()
        prop = PropagationLarge(n, nz, WAVELENGTH, VOXELSIZE, [DISTANCE])
        out = prop.D(psi, 0, chunks)                # warmup: allocates
        cp.cuda.runtime.deviceSynchronize()
        t_setup = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = prop.D(psi, 0, chunks)                # timed: load + compute
        cp.cuda.runtime.deviceSynchronize()
        t_run = time.perf_counter() - t0

        gpu_peak = pool.total_bytes()   # allocated pool footprint — close to peak
        print(f"    SETUP+1st D={t_setup:6.2f}s   COMPUTE D={t_run:6.2f}s   "
              f"host RSS peak={_rss_peak_gb():5.1f} GB   "
              f"GPU peak={_hb(gpu_peak)}   out[0,0,:4]={out[0,0,:4].real}")
        prop.free()               # release pinned/GPU buffers before del
        del out, psi, prop
    except Exception:
        print("    FAILED:")
        traceback.print_exc()
        return False
    finally:
        _reset_gpu()
    return True


# ---------- main ------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("tomo", "prop", "both"), default="both")
    p.add_argument("--ups-max", type=int, default=128)
    p.add_argument("--host-cap-gb", type=float, default=None,
                   help="skip sizes whose estimate exceeds this; "
                        "default = 90%% of /proc/meminfo MemAvailable")
    p.add_argument("--gpu-budget-gb", type=float, default=30.0,
                   help="target GPU memory for chunk-picker (default 30 GB, "
                        "leaves headroom on a 40 GB A100 for cuFFT plans + "
                        "transient allocations).  Each stage is sized to "
                        "half this budget.")
    args = p.parse_args()

    cap = args.host_cap_gb
    if cap is None:
        avail = _mem_available_gb()
        if avail is None:
            cap = 100.0
            print(f"[bench_large]  MemAvailable unreadable, defaulting cap to {cap} GB")
        else:
            cap = 0.9 * avail
            print(f"[bench_large]  MemAvailable={avail:.1f} GB  → cap {cap:.1f} GB")

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)["name"].decode()
    gpu_total = cp.cuda.runtime.memGetInfo()[1]
    print(f"[bench_large]  GPU {dev_id}: {dev_name}   total={_hb(gpu_total)}   "
          f"chunker budget={args.gpu_budget_gb} GB")

    ups_list = [u for u in UPS_STEPS if u <= args.ups_max]

    if args.stage in ("tomo", "both"):
        print("\n" + "=" * 78)
        print(f"TomoLargeReal  —  R() at nz=1, N=3072·UPS, NTHETA=3·N/4")
        print("=" * 78)
        for ups in ups_list:
            if not bench_tomo(ups, cap, args.gpu_budget_gb):
                print("[bench_large]  stopping tomo sweep")
                break

    if args.stage in ("prop", "both"):
        print("\n" + "=" * 78)
        print(f"PropagationLarge  —  D() at ntheta=1, N=3072·UPS, NZ=3072·UPS")
        print("=" * 78)
        for ups in ups_list:
            if not bench_prop(ups, cap, args.gpu_budget_gb):
                print("[bench_large]  stopping prop sweep")
                break


if __name__ == "__main__":
    main()
