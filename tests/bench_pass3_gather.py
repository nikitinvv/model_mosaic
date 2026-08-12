"""Standalone bench + parity for TomoLargeReal._pass3_gather (forward
NUFFT gather).  Mirrors tests/bench_passrt2.py's structure so we can
iterate on optimizations independent of the full pipeline.

  gather_baseline(fde_h, theta_h, chunk_xy, per_bin):
      current tomo_large behaviour — bin samples by rk_x, then per bin
      upload an fde x-slice (nz · 2n · (chunk_xy + 2m + 1) · 8 bytes)
      and launch one kernel restricted to that bin's samples.  fde
      H2D total ≈ full fde (partitioned across bins, no re-upload);
      GPU peak bounded by the per-bin slice.

  gather_optimized(fde_h, theta_h, chunk_xy, per_bin):
      Same per-bin structure as baseline but uses `gather_compact_kernel`
      which reads pre-decoded (theta_idx, x_idx) int32 pairs per
      thread instead of decoding a flat long-long full_idx via divmod.
      Mirrors bench_passrt2.py's scatter_compact.

  per_bin_precompute(n, ntheta, theta_h, chunk_xy)
      computes per-bin sample-index lists directly on GPU in
      theta-slabs.  No persistent qid array (was 1.8 GB at UPS=8),
      no global argsort (peaked at ~3× qid).  Only the resulting
      per_bin list (~1.8 GB at UPS=8) remains on host.

Runs:
  python -m tests.bench_pass3_gather                  # bench UPS=8
  python -m tests.bench_pass3_gather --parity         # parity at small n
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import cupy as cp

from processing.kernels import gather_kernel_rfft


# Compact-input gather: identical NUFFT math to gather_kernel_rfft but
# reads pre-decoded (theta_idx, x_idx) int32 pairs per thread instead
# of a long long full_idx that the kernel would have to divmod itself.
# Structural mirror of scatter_compact_kernel in bench_passrt2.py.
gather_compact_kernel = cp.RawKernel(r"""
extern "C" __global__ void gather_compact(
    float2 *g,                            // (nel,) c64 compact sino out
    const float2 *f,                      // (twon, patch_w) c64 fde slice
    const int   *x_idx,                   // (nel,) int32
    const int   *theta_idx,               // (nel,) int32
    const float *cos_theta,               // (ntheta,) f32
    const float *sin_theta,               // (ntheta,) f32
    int m, const float *mu,
    int nel, int stx, int endx_half, int n)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx >= nel) return;

    const float PI      = 3.141592653589793238f;
    const float coeff0  = PI / mu[0];
    const float coeff1  = -PI * PI / mu[0];
    const int   twon    = 2 * n;
    const float ftwon   = (float)twon;
    const int   patch_w = endx_half - stx;

    int   k = theta_idx[tx];
    int   j = x_idx[tx];
    float r = ((float)j - 0.5f * (float)n) / (float)n;
    float x0 =  cos_theta[k] * r;
    float y0 = -sin_theta[k] * r;
    if (x0 < -0.5f)         x0 = -0.5f;
    if (x0 > 0.5f - 1e-5f)  x0 = 0.5f - 1e-5f;
    if (y0 < -0.5f)         y0 = -0.5f;
    if (y0 > 0.5f - 1e-5f)  y0 = 0.5f - 1e-5f;

    float2 g0 = {0.0f, 0.0f};
    const int base_x = (int)rintf(ftwon * x0) - m;
    const int base_y = (int)rintf(ftwon * y0) - m;

    for (int i1 = 0; i1 < 2 * m + 1; i1++) {
        int   ell1 = base_y + i1;
        float w1   = ell1 / ftwon - y0;
        int   k1   = ((ell1 % twon) + twon) % twon;

        for (int i0 = 0; i0 < 2 * m + 1; i0++) {
            int   ell0 = base_x + i0;
            float w0   = ell0 / ftwon - x0;
            float w    = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));

            int k0 = ((ell0 % twon) + twon) % twon;
            int rk_x, rk_y;
            bool conj_flag;
            if (k0 <= n) {
                rk_x = k0;
                rk_y = k1;
                conj_flag = false;
            } else {
                rk_x = twon - k0;
                rk_y = (twon - k1) % twon;
                conj_flag = true;
            }
            int f_indx = rk_x - stx;
            if (f_indx < 0 || f_indx >= patch_w) continue;

            float2 v = f[f_indx + patch_w * rk_y];
            if (conj_flag) v.y = -v.y;

            float sign_kw = ((k0 + k1) & 1) ? -w : w;
            g0.x += sign_kw * v.x;
            g0.y += sign_kw * v.y;
        }
    }

    g[tx].x = g0.x / n;
    g[tx].y = g0.y / n;
}
""", "gather_compact")


def _compute_m_mua(n: int):
    """Same NUFFT window as TomoLargeReal.__init__ (eps = 1e-3)."""
    eps = 1e-3
    mu  = -np.log(eps) / (2 * n * n)
    m   = int(np.ceil(2 * n / np.pi *
                      np.sqrt(-mu * np.log(eps) + (mu * n) ** 2 / 4)))
    return m, cp.array([mu], dtype=cp.float32)


def _bin_ids_gpu(n, ntheta, theta_h, chunk_xy, chunk_theta_gpu=4096):
    """Per-theta-slab generator yielding (t0, bin_d_gpu) with bin_d
    the ((t1-t0)*n,) int32 GPU array of rk_x-bin ids.  Shared driver
    for both sort_into_bins (baseline) and per_bin_precompute
    (optimized) — computes centres in slabs, no persistent (ntheta,n)
    array on GPU or host."""
    twon    = 2 * n
    nx_bins = (n // chunk_xy) + 1
    r_d     = (cp.arange(n, dtype=cp.float32) - n / 2) / n
    cos_th  = np.cos(theta_h).astype(np.float32)
    for t0 in range(0, ntheta, chunk_theta_gpu):
        t1     = min(t0 + chunk_theta_gpu, ntheta)
        cos_d  = cp.asarray(cos_th[t0:t1])
        x_d    = cp.clip(cos_d[:, None] * r_d[None, :], -0.5, 0.5 - 1e-5)
        k0_d   = ((cp.floor(twon * x_d).astype(cp.int32)) + twon) % twon
        rk_x_d = cp.where(k0_d > n, twon - k0_d, k0_d)
        bin_d  = (rk_x_d // chunk_xy).clip(max=nx_bins - 1).ravel()
        yield t0, t1, bin_d
        del cos_d, x_d, k0_d, rk_x_d, bin_d


def sort_into_bins(n, ntheta, theta_h, chunk_xy, chunk_theta_gpu=4096):
    """Baseline precompute (matches tomo_large._sort_into_chunks
    output shape): one flat int64 `idx` of sample-flat-indices sorted
    by rk_x-bin, plus per-bin counts `nel`.  Computed on GPU in
    theta-slabs (no persistent (ntheta,n) qid array), then sorted on
    host once.

    Returns (nel, idx, nx_bins).
    """
    twon    = 2 * n
    nx_bins = (n // chunk_xy) + 1
    qid = np.empty(ntheta * n, dtype=np.int32)
    for t0, t1, bin_d in _bin_ids_gpu(n, ntheta, theta_h, chunk_xy,
                                      chunk_theta_gpu):
        qid[t0 * n : t1 * n] = cp.asnumpy(bin_d)

    idx   = np.argsort(qid, kind="stable")
    qid_s = qid[idx]
    nel   = np.zeros(nx_bins, dtype=np.int64)
    if len(qid_s):
        change = np.flatnonzero(np.diff(qid_s, prepend=qid_s[0] - 1))
        runs   = np.diff(np.append(change, len(qid_s)))
        nel[qid_s[change]] = runs
    return nel, idx.astype(np.int64), nx_bins


def per_bin_precompute(n, ntheta, theta_h, chunk_xy, chunk_theta_gpu=4096):
    """Optimized precompute: per-bin lists of sample-flat-indices,
    streamed per theta-slab.  Each per_bin[b] is int64.  No persistent
    qid, no argsort (compared to sort_into_bins).

    Bin assignment: rk_x = post-reflection [0, n] column → bin =
    rk_x // chunk_xy.  Each sample lands in exactly one bin.
    """
    twon    = 2 * n
    nx_bins = (n // chunk_xy) + 1
    per_bin_parts = [[] for _ in range(nx_bins)]
    for t0, t1, bin_d in _bin_ids_gpu(n, ntheta, theta_h, chunk_xy,
                                      chunk_theta_gpu):
        for b in range(nx_bins):
            local = cp.asnumpy(cp.where(bin_d == b)[0])                # int64
            if local.size:
                per_bin_parts[b].append(local + (t0 * n))
    return [np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            for parts in per_bin_parts]


def _get_st_end(indx, chunk_xy, n, m):
    stx  = max(0, indx * chunk_xy - m)
    endx = min(n + 1, (indx + 1) * chunk_xy + m + 1)
    return stx, endx


def gather_baseline(fde_h, theta_h, chunk_xy,
                    sort_cache=None, want_sino=False):
    """Current tomo_large behaviour — bin samples by rk_x, upload one
    fde x-slice per bin, launch kernel restricted to that bin.

    `sort_cache = (nel, idx, nx_bins)` from `sort_into_bins`:
      idx (ntheta·n,) int64 host — sample flat indices sorted by bin
      nel (nx_bins,)  int64 host — count per bin
    Depends only on geometry + chunk_xy; cache across R() calls."""
    nz, twon, ncols = fde_h.shape
    n       = ncols - 1
    ntheta  = len(theta_h)
    assert nz == 1, "gather_baseline prototype: nz=1"
    m, mua_d = _compute_m_mua(n)
    cos_d   = cp.asarray(np.cos(theta_h).astype("float32"))
    sin_d   = cp.asarray(np.sin(theta_h).astype("float32"))

    if sort_cache is None:
        sort_cache = sort_into_bins(n, ntheta, theta_h, chunk_xy)
    nel, idx, nx_bins = sort_cache

    sino_flat_h = np.empty(ntheta * n, dtype=np.complex64) if want_sino else None
    threads = 1024
    offset  = 0
    for indx in range(nx_bins):
        nel_i = int(nel[indx])
        if nel_i == 0:
            continue
        stx, endx = _get_st_end(indx, chunk_xy, n, m)
        if endx <= stx:
            offset += nel_i
            continue
        full_idx = idx[offset : offset + nel_i]
        fde_d    = cp.asarray(fde_h[0, :, stx:endx])                   # (twon, endx-stx) c64
        fidx_d   = cp.asarray(full_idx)                                # int64
        sino_bin = cp.empty(nel_i, dtype=cp.complex64)
        grid     = ((nel_i + threads - 1) // threads,)
        gather_kernel_rfft(grid, (threads,),
            (sino_bin, fde_d, fidx_d, cos_d, sin_d, m, mua_d,
             nel_i, np.int32(stx), np.int32(endx), n))
        if want_sino:
            sino_flat_h[full_idx] = cp.asnumpy(sino_bin)
        offset += nel_i
        del fde_d, fidx_d, sino_bin

    if want_sino:
        return sino_flat_h.reshape(ntheta, n)
    return None


def gather_optimized(fde_h, theta_h, chunk_xy,
                     per_bin=None, want_sino=False):
    """Per-bin gather using the compact kernel — mirrors
    bench_passrt2.py's scatter_compact structure.

    For each rk_x bin: upload the fde x-slice (same as baseline),
    decompose per_bin[b] flat indices into (theta_idx, x_idx) int32,
    launch gather_compact_kernel (skips the flat-idx divmod that
    gather_kernel_rfft does per thread).  Sino output written to a
    compact per-bin GPU buffer, then scattered into the host sino
    array at per_bin[b] positions."""
    nz, twon, ncols = fde_h.shape
    n       = ncols - 1
    ntheta  = len(theta_h)
    assert nz == 1, "gather_optimized prototype: nz=1"
    m, mua_d = _compute_m_mua(n)
    cos_d   = cp.asarray(np.cos(theta_h).astype("float32"))
    sin_d   = cp.asarray(np.sin(theta_h).astype("float32"))

    if per_bin is None:
        per_bin = per_bin_precompute(n, ntheta, theta_h, chunk_xy)

    sino_flat_h = np.empty(ntheta * n, dtype=np.complex64) if want_sino else None
    threads = 1024
    for indx, full_idx in enumerate(per_bin):
        nel_i = int(full_idx.size)
        if nel_i == 0:
            continue
        stx, endx = _get_st_end(indx, chunk_xy, n, m)
        if endx <= stx:
            continue
        fde_d   = cp.asarray(fde_h[0, :, stx:endx])              # (twon, endx-stx) c64
        idx_d   = cp.asarray(full_idx)                           # int64
        theta_i = (idx_d // n).astype(cp.int32)
        x_i     = (idx_d %  n).astype(cp.int32)
        sino_bin = cp.empty(nel_i, dtype=cp.complex64)
        grid    = ((nel_i + threads - 1) // threads,)
        gather_compact_kernel(grid, (threads,),
            (sino_bin, fde_d, x_i, theta_i, cos_d, sin_d, m, mua_d,
             nel_i, np.int32(stx), np.int32(endx), n))
        if want_sino:
            sino_flat_h[full_idx] = cp.asnumpy(sino_bin)
        del fde_d, idx_d, theta_i, x_i, sino_bin

    if want_sino:
        return sino_flat_h.reshape(ntheta, n)
    return None


def parity_check(n=128, chunk_xy=32):
    """Compare optimized (single-launch full-fde) vs baseline
    (partitioned) — should match up to atomicAdd reordering."""
    ntheta  = 3 * n // 4
    twon    = 2 * n
    theta_h = np.linspace(0, np.pi, ntheta, endpoint=False).astype("float32")
    rng     = np.random.default_rng(0)
    fde_h   = ((rng.standard_normal((1, twon, n + 1)).astype("float32") +
                1j * rng.standard_normal((1, twon, n + 1)).astype("float32"))
               .astype(np.complex64))

    ref = gather_optimized(fde_h, theta_h, chunk_xy, want_sino=True)
    new = gather_baseline (fde_h, theta_h, chunk_xy, want_sino=True)
    diff    = ref - new
    l2_ref  = float(np.linalg.norm(ref))
    l2_diff = float(np.linalg.norm(diff))
    max_d   = float(np.abs(diff).max())
    max_r   = float(np.abs(ref).max())
    print(f"parity n={n} chunk_xy={chunk_xy}:")
    print(f"  ||diff||₂ / ||ref||₂ = {l2_diff / max(l2_ref, 1e-30):.3e}")
    print(f"  max|diff| / max|ref| = {max_d  / max(max_r, 1e-30):.3e}")
    print(f"  ||diff||₂ = {l2_diff:.3e}   ||ref||₂ = {l2_ref:.3e}")


def _run(fn, *args, **kwargs):
    fn(*args, **kwargs)                              # warmup
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    cp.cuda.runtime.deviceSynchronize()
    return time.perf_counter() - t0


def bench_ups(chunk_xy: int, ups: int = 8):
    nz      = 1
    n       = 3072 * ups
    ntheta  = 3 * n // 4
    twon    = 2 * n
    theta_h = np.linspace(0, np.pi, ntheta, endpoint=False).astype("float32")

    rng   = np.random.default_rng(0)
    fde_h = ((rng.standard_normal((nz, twon, n + 1)).astype("float32") +
              1j * rng.standard_normal((nz, twon, n + 1)).astype("float32"))
             .astype(np.complex64))
    print(f"UPS={ups}  n={n}  ntheta={ntheta}  nz={nz}")
    print(f"fde: {fde_h.nbytes / 1e9:.2f} GB   chunk_xy={chunk_xy}\n")

    # Precompute for each variant separately (they have different
    # output structures: sort_cache = (nel, flat idx, nx_bins) for
    # baseline; per_bin = list of arrays for optimized).
    t0 = time.perf_counter()
    sort_cache = sort_into_bins(n, ntheta, theta_h, chunk_xy)
    cp.cuda.runtime.deviceSynchronize()
    t_pre_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    per_bin = per_bin_precompute(n, ntheta, theta_h, chunk_xy)
    cp.cuda.runtime.deviceSynchronize()
    t_pre_opt = time.perf_counter() - t0

    nx_bins = sort_cache[2]

    t_base = _run(gather_baseline,  fde_h, theta_h, chunk_xy, sort_cache=sort_cache)
    t_opt  = _run(gather_optimized, fde_h, theta_h, chunk_xy, per_bin=per_bin)

    def _line(label, t_p, t_c, extra=""):
        print(f"[{label:9s}] precompute={t_p:6.2f}s  compute={t_c:6.2f}s  "
              f"total={t_p + t_c:6.2f}s   {extra}")
    _line("baseline",  t_pre_base, t_base, f"bins={nx_bins}")
    _line("optimized", t_pre_opt,  t_opt,  "compact kernel + (theta,x) metadata")

    print(f"\ncompute-only speedup: {t_base / max(t_opt, 1e-9):.2f}×")
    print(f"total-cost speedup:   "
          f"{(t_base + t_pre_base) / max(t_opt + t_pre_opt, 1e-9):.2f}×")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parity", action="store_true")
    p.add_argument("--n-viz",  type=int, default=128)
    p.add_argument("--band-xy", type=int, default=32,
                   help="chunk_xy for --parity")
    p.add_argument("--ups", type=int, default=8)
    p.add_argument("--chunk-xy", type=int, default=768)
    args = p.parse_args()

    if args.parity:
        parity_check(args.n_viz, args.band_xy)
    else:
        bench_ups(args.chunk_xy, args.ups)


if __name__ == "__main__":
    main()
