"""Standalone benchmark for TomoLargeReal._passRT2_scatter.

Extracts the adjoint-NUFFT scatter into a self-contained script so we
can iterate on optimizations without running the whole pipeline.

  scatter(sino_h, theta_h, chunk_theta, chunk_xy)  — baseline
      current two-nested-loops form: outer ky-band, inner theta-slab
      with per-slab sino H2D repeated per band.

  scatter_compact(sino_h, theta_h, chunk_xy, per_band=None) — optimized
      precomputed per-band sample-index lists → one compact sino
      upload per band (mirrors R's pass3_gather structure).  Total
      sino H2D per RT call ≈ 1× full sino (× tiny spillover) vs
      n_bands× in scatter().

  per_band_precompute(n, ntheta, theta_h, chunk_xy)
      computes per-band sample-index lists directly on GPU in
      theta-slabs.  No persistent centers array (was 1.8 GB at
      UPS=8, 116 GB at UPS=64); only the per-band lists (~1.8 GB at
      UPS=8) remain on host.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import cupy as cp

from processing.kernels import gather_kernel_ychunk


# Compact-input scatter: identical NUFFT footprint math to
# gather_kernel_ychunk (dir=1 branch) but reads sino values, theta,
# and x from parallel compact arrays instead of a rectangular
# (ntheta, nz, n) grid.  One thread per compact sample.  Assumes nz=1
# (matches every RT call in step8_fbp_large / test_bench_fbp).
scatter_compact_kernel = cp.RawKernel(r"""
extern "C" __global__ void scatter_compact(
    const float2* __restrict__ g,   // compact sino, (nel,)
    float2* __restrict__ f,         // fde slice, (h, 2n)  [nz=1]
    const int*   __restrict__ x_idx,      // (nel,)
    const int*   __restrict__ theta_idx,  // (nel,)
    const float* __restrict__ theta_all,  // (ntheta,)
    int m, const float* mu,
    int n, int nel,
    int y_lo, int y_hi)
{
    int tk = blockDim.x * blockIdx.x + threadIdx.x;
    if (tk >= nel) return;

    const float PI = 3.141592653589793238f;
    const int   twon    = 2 * n;
    const float ftwon   = (float)twon;
    const float mu0     = mu[0];
    const float coeff0  = PI / mu0;
    const float coeff1  = -PI * PI / mu0;
    const float inv_twon = 1.0f / ftwon;
    const float cx       = n * 0.5f;

    int   tx = x_idx[tk];
    int   ty = theta_idx[tk];
    float th = theta_all[ty];

    float x0 =  (tx - cx) / (float)n * __cosf(th);
    float y0 = -(tx - cx) / (float)n * __sinf(th);

    float2 g0 = g[tk];

    int base_x = (int)rintf(ftwon * x0) - m;
    int base_y = (int)rintf(ftwon * y0) - m;
    int len    = 2 * m + 1;

    float ex[32];
    for (int i0 = 0; i0 < len; i0++) {
        float w0 = (base_x + i0) * inv_twon - x0;
        ex[i0] = __expf(coeff1 * w0 * w0);
    }

    for (int i1 = 0; i1 < len; i1++) {
        int   ell1   = base_y + i1;
        float w1     = ell1 * inv_twon - y0;
        float ey     = coeff0 * __expf(coeff1 * w1 * w1);
        int   f_indy = (n + ell1 + twon) % twon;
        if (f_indy < y_lo || f_indy >= y_hi) continue;
        int   row_off = twon * (f_indy - y_lo);

        for (int i0 = 0; i0 < len; i0++) {
            float w    = ex[i0] * ey;
            int   ell0 = base_x + i0;
            int   f_ind = (n + ell0 + twon) % twon + row_off;
            atomicAdd(&(f[f_ind].x), w * g0.x);
            atomicAdd(&(f[f_ind].y), w * g0.y);
        }
    }
}
""", "scatter_compact")


def _compute_m_mua(n: int):
    """NUFFT window half-width m + Gaussian width mu.  Matches
    TomoLargeReal.__init__ (eps = 1e-3)."""
    eps = 1e-3
    mu  = -np.log(eps) / (2 * n * n)
    m   = int(np.ceil(2 * n / np.pi *
                      np.sqrt(-mu * np.log(eps) + (mu * n) ** 2 / 4)))
    return m, cp.array([mu], dtype=cp.float32)


def per_band_precompute(n, ntheta, theta_h, chunk_xy, chunk_theta_gpu=4096):
    """Compute per-band sample-index lists directly on GPU, streamed
    per theta-slab into host per-band buckets.  No persistent centers
    array (the full (ntheta, n) int32 would be 1.8 GB at UPS=8, 116 GB
    at UPS=64).  Only the resulting per_band list persists on host.

    For each theta slab: compute rk_y → centers on GPU (transient),
    then for each band extract that slab's contributing sample indices
    via mask + where on GPU, D2H, append to per_band[b] with global
    offset.  Wrap-around spillover at edge bands (b=0 and b=n_bands-1)
    is folded into the mask so a sample near ky=0 with a split ±m
    footprint appears in both edge bands.

    Depends only on (n, ntheta, theta, chunk_xy) — cache across RT
    calls.  Returns a list of int32 host arrays, one per ky-band.
    """
    m, _    = _compute_m_mua(n)
    twon    = 2 * n
    n_bands = (twon + chunk_xy - 1) // chunk_xy
    cx      = np.float32(n * 0.5)
    x_d     = cp.arange(n, dtype=cp.float32) - cx                       # (n,)
    sin_np  = np.sin(theta_h).astype(np.float32)

    per_band_parts = [[] for _ in range(n_bands)]
    for t0 in range(0, ntheta, chunk_theta_gpu):
        t1 = min(t0 + chunk_theta_gpu, ntheta)
        sin_d = cp.asarray(sin_np[t0:t1])
        rk_d  = cp.rint(-2.0 * x_d[None, :] * sin_d[:, None]).astype(cp.int32)
        centers_d = (((n + rk_d) % twon).astype(cp.int32)).ravel()      # ((t1-t0)*n,)
        for b in range(n_bands):
            y_lo   = b * chunk_xy
            y_hi   = min(y_lo + chunk_xy, twon)
            mask_d = (centers_d + m >= y_lo) & (centers_d - m < y_hi)
            if y_lo == 0 and m > 0:
                mask_d |= (centers_d >= twon - m)
            if y_hi == twon and m > 0:
                mask_d |= (centers_d < m)
            local_idx = cp.asnumpy(cp.where(mask_d)[0])                # int64
            if local_idx.size:
                per_band_parts[b].append(local_idx + (t0 * n))         # int64
        del sin_d, rk_d, centers_d

    return [np.concatenate(parts) if parts
            else np.zeros(0, dtype=np.int64)
            for parts in per_band_parts]


def scatter(sino_h, theta_h, chunk_theta, chunk_xy, want_fde=False):
    """The passRT2 scatter, extracted verbatim.

    sino_h : (ntheta, nz, n) c64 host
    theta_h: (ntheta,)       f32 host
    Returns (nz, 2n, 2n) c64 numpy fde when want_fde, else None.
    """
    ntheta, nz, n = sino_h.shape
    twon = 2 * n
    m, mua_d = _compute_m_mua(n)
    theta_d  = cp.asarray(theta_h)
    fde_h = np.zeros((nz, twon, twon), dtype=np.complex64) if want_fde else None
    block = (32, 32, 1)

    for y_lo in range(0, twon, chunk_xy):
        y_hi = min(y_lo + chunk_xy, twon)
        fde_slice_d = cp.zeros((nz, y_hi - y_lo, twon), dtype=cp.complex64)
        for t_lo in range(0, ntheta, chunk_theta):
            t_hi = min(t_lo + chunk_theta, ntheta)
            th   = t_hi - t_lo
            sino_slab_d = cp.asarray(sino_h[t_lo:t_hi])
            grid = (int(np.ceil(n / 32)), int(np.ceil(th / 32)), nz)
            gather_kernel_ychunk(grid, block,
                (sino_slab_d, fde_slice_d, theta_d[t_lo:t_hi], m, mua_d,
                 n, th, nz, np.int32(y_lo), np.int32(y_hi)))
            del sino_slab_d
        if want_fde:
            fde_h[:, y_lo:y_hi, :] = cp.asnumpy(fde_slice_d)
        del fde_slice_d
    return fde_h


def scatter_compact(sino_h, theta_h, chunk_xy,
                    per_band=None, want_fde=False):
    """Per-band scatter — mirrors R's pass3 structure.

    Total sino H2D per RT call ≈ 1× full sino (× tiny spillover),
    vs. n_bands× in `scatter()`.  Assumes nz=1.

    `per_band` (list of int32 host arrays, one per ky-band) depends
    only on geometry and chunk_xy — precompute via
    `per_band_precompute(n, ntheta, theta_h, chunk_xy)` once and
    cache across RT calls.  Passed here so the per-RT loop body is
    just slice + gather + upload + launch, no mask work.
    """
    ntheta, nz, n = sino_h.shape
    assert nz == 1, "scatter_compact prototype: nz=1 only"
    twon    = 2 * n
    m, mua_d = _compute_m_mua(n)
    theta_d = cp.asarray(theta_h)
    sino_flat = sino_h.reshape(-1)

    if per_band is None:
        per_band = per_band_precompute(n, ntheta, theta_h, chunk_xy)

    fde_h = np.zeros((nz, twon, twon), dtype=np.complex64) if want_fde else None
    threads = 256

    for b, sample_idx in enumerate(per_band):
        y_lo = b * chunk_xy
        y_hi = min(y_lo + chunk_xy, twon)
        nel = int(sample_idx.size)
        if nel == 0:
            continue
        g_compact_d  = cp.asarray(sino_flat[sample_idx])
        sample_idx_d = cp.asarray(sample_idx)                          # int64
        theta_i_d    = (sample_idx_d // n).astype(cp.int32)            # kernel takes int
        x_i_d        = (sample_idx_d %  n).astype(cp.int32)
        fde_slice_d  = cp.zeros((y_hi - y_lo, twon), dtype=cp.complex64)

        grid = ((nel + threads - 1) // threads,)
        scatter_compact_kernel(grid, (threads,),
            (g_compact_d, fde_slice_d, x_i_d, theta_i_d, theta_d,
             m, mua_d, n, nel,
             np.int32(y_lo), np.int32(y_hi)))

        if want_fde:
            fde_h[0, y_lo:y_hi, :] = cp.asnumpy(fde_slice_d)
        del g_compact_d, theta_i_d, x_i_d, fde_slice_d, sample_idx_d
    return fde_h


def _run_baseline(sino_h, theta_h, chunk_theta, chunk_xy):
    scatter(sino_h[:min(chunk_theta, 64)],
            theta_h[:min(chunk_theta, 64)],
            min(chunk_theta, 64), chunk_xy, want_fde=False)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    scatter(sino_h, theta_h, chunk_theta, chunk_xy, want_fde=False)
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0


def _run_compact(sino_h, theta_h, chunk_xy, per_band):
    scatter_compact(sino_h, theta_h, chunk_xy,
                    per_band=per_band, want_fde=False)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    scatter_compact(sino_h, theta_h, chunk_xy,
                    per_band=per_band, want_fde=False)
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0


def bench_ups8(chunk_theta: int, chunk_xy: int, ups: int = 8):
    """Bench BOTH baseline and compact scatter, print side-by-side."""
    nz = 1
    n  = 3072 * ups
    ntheta = 3 * n // 4
    theta_h = np.linspace(0, np.pi, ntheta, endpoint=False).astype(np.float32)
    print(f"UPS={ups}  n={n}  ntheta={ntheta}  nz={nz}")
    print(f"chunk_theta={chunk_theta}  chunk_xy={chunk_xy}  "
          f"n_ybands={(2*n + chunk_xy - 1)//chunk_xy}  "
          f"n_theta_slabs={(ntheta + chunk_theta - 1)//chunk_theta}")

    rng = np.random.default_rng(0)
    real = rng.standard_normal((ntheta, nz, n), dtype=np.float32)
    imag = rng.standard_normal((ntheta, nz, n), dtype=np.float32)
    sino_h = (real + 1j * imag).astype(np.complex64)
    print(f"sino: {sino_h.nbytes / 1e9:.2f} GB\n")

    # Baseline (needs no precompute)
    t_base = _run_baseline(sino_h, theta_h, chunk_theta, chunk_xy)
    n_bands = (2 * n + chunk_xy - 1) // chunk_xy
    n_slabs = (ntheta + chunk_theta - 1) // chunk_theta
    h2d_base = n_bands * n_slabs * chunk_theta * nz * n * 8 / 1e9

    # Compact (precompute per_band once, then time the scatter alone)
    t_pre0 = time.perf_counter()
    per_band = per_band_precompute(n, ntheta, theta_h, chunk_xy)
    cp.cuda.runtime.deviceSynchronize()
    t_pre = time.perf_counter() - t_pre0
    tot = sum(pb.size for pb in per_band)
    t_comp = _run_compact(sino_h, theta_h, chunk_xy, per_band)
    h2d_comp = tot * 8 / 1e9

    def _line(label, t_p, t_c, extra=""):
        print(f"[{label:9s}] precompute={t_p:6.2f}s  compute={t_c:6.2f}s  "
              f"total={t_p + t_c:6.2f}s   {extra}")
    _line("baseline", 0.0,   t_base,
          f"sino H2D {h2d_base:6.1f} GB ({h2d_base/max(t_base,1e-9):5.1f} GB/s eff)")
    _line("compact",  t_pre, t_comp,
          f"sino H2D {h2d_comp:6.1f} GB ({h2d_comp/t_comp:5.1f} GB/s eff)"
          f"  spillover={tot/(ntheta*n):.3f}×")

    print(f"\ncompute-only speedup baseline → compact: {t_base / max(t_comp, 1e-9):.2f}×")
    print(f"total-cost speedup: {t_base / max(t_comp + t_pre, 1e-9):.2f}×")
    print(f"H2D reduction: {h2d_base / h2d_comp:.1f}×")


def parity_check(n: int = 128, chunk_xy: int = 32):
    """Verify scatter_compact matches scatter (baseline) at small n."""
    nz = 1
    ntheta = 3 * n // 4
    theta_h = np.linspace(0, np.pi, ntheta, endpoint=False).astype(np.float32)
    rng = np.random.default_rng(0)
    sino_h = ((rng.standard_normal((ntheta, nz, n), dtype=np.float32) +
               1j * rng.standard_normal((ntheta, nz, n), dtype=np.float32))
              .astype(np.complex64))

    fde_ref = scatter(sino_h, theta_h, ntheta, chunk_xy, want_fde=True)
    fde_new = scatter_compact(sino_h, theta_h, chunk_xy, want_fde=True)
    diff    = fde_ref - fde_new
    l2_ref  = float(np.linalg.norm(fde_ref))
    l2_diff = float(np.linalg.norm(diff))
    max_d   = float(np.abs(diff).max())
    max_r   = float(np.abs(fde_ref).max())
    print(f"parity n={n} chunk_xy={chunk_xy}:")
    print(f"  ||diff||₂ / ||ref||₂ = {l2_diff / max(l2_ref, 1e-30):.3e}")
    print(f"  max|diff| / max|ref| = {max_d  / max(max_r, 1e-30):.3e}")
    print(f"  ||diff||₂ = {l2_diff:.3e}   ||ref||₂ = {l2_ref:.3e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parity", action="store_true",
                   help="verify scatter_compact matches baseline at small n")
    p.add_argument("--n-viz",   type=int, default=128,
                   help="n for --parity")
    p.add_argument("--band-xy", type=int, default=32,
                   help="chunk_xy for --parity")
    p.add_argument("--ups", type=int, default=8)
    p.add_argument("--chunk-theta", type=int, default=768)
    p.add_argument("--chunk-xy",    type=int, default=768)
    args = p.parse_args()

    if args.parity:
        parity_check(args.n_viz, args.band_xy)
    else:
        bench_ups8(args.chunk_theta, args.chunk_xy, args.ups)


if __name__ == "__main__":
    main()
