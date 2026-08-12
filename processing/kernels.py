"""Raw CUDA kernels used by tomo.py, tomo_large.py, propagation.py.

Vendored from holotomocupy_mpi/src/holotomocupy/cuda_kernels.py and
radon_large/cuda_kernels.py so this pipeline has no external dependencies
beyond cupy/numpy/tifffile/mpi4py.
"""
from __future__ import annotations

import cupy as cp


# --- TomoReal.RT (GPU-only, full-complex) NUFFT gather --------------------
# Full-complex (nz × 2n × 2n) c64 fde with modular wrap `(n + ell + twon) % twon`.
# Used by TomoReal.RT (dir=1 adjoint scatter path) — the forward TomoReal.R
# uses the rfft-half `gather_kernel_rfft_full` further below.
gather_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void gather(float2* g, float2* f, float* theta, int m, float* mu,
                                  int n, int ntheta, int nz, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= ntheta || tz >= nz) return;

    const float PI     = 3.141592653589793238f;
    const int   twon   = 2 * n;
    const float ftwon  = (float)twon;
    const float mu0    = mu[0];
    const float coeff0 = PI / mu0;
    const float coeff1 = -PI * PI / mu0;
    const float inv_twon = 1.0f / ftwon;

    const float cx  = n * 0.5f;
    const float x0 =  (tx - cx) / (float)n * __cosf(theta[ty]);
    const float y0 = -(tx - cx) / (float)n * __sinf(theta[ty]);

    const int g_ind = tx + tz * n + ty * n * nz;  // swapped axes
    float2 g0 = (dir == 0) ? make_float2(0.0f, 0.0f) : g[g_ind];

    // Symmetric (2m+1) gather window centred on the nearest grid point.
    // floor would bias by up to 0.5 grid units off-grid and become coherent
    // at angles aligned with the Cartesian grid (0/90/... deg).
    const int base_x  = (int)rintf(ftwon * x0) - m;
    const int base_y  = (int)rintf(ftwon * y0) - m;
    const int tz_off  = tz * twon * twon;
    const int len     = 2 * m + 1;

    // Precompute x-direction exponential factors once.
    // Reduces expf calls from (2m+1)^2 to 2*(2m+1).
    float ex[32];  // 2*m+1 entries; m is small (typically 4-5)
    for (int i0 = 0; i0 < len; i0++) {
        float w0 = (base_x + i0) * inv_twon - x0;
        ex[i0] = __expf(coeff1 * w0 * w0);
    }

    for (int i1 = 0; i1 < len; i1++)
    {
        int   ell1    = base_y + i1;
        float w1      = ell1 * inv_twon - y0;
        float ey      = coeff0 * __expf(coeff1 * w1 * w1);
        int   f_indy  = (n + ell1 + twon) % twon;
        int   row_off = twon * f_indy + tz_off;

        for (int i0 = 0; i0 < len; i0++)
        {
            float w    = ex[i0] * ey;
            int   ell0 = base_x + i0;
            int   f_ind = (n + ell0 + twon) % twon + row_off;

            if (dir == 0)
            {
                g0.x += w * f[f_ind].x;
                g0.y += w * f[f_ind].y;
            }
            else
            {
                atomicAdd(&(f[f_ind].x), w * g0.x);
                atomicAdd(&(f[f_ind].y), w * g0.y);
            }
        }
    }

    if (dir == 0)
    {
        g[g_ind].x = g0.x / n;
        g[g_ind].y = g0.y / n;
    }
}
""",
    "gather",
)


# --- ky-restricted variant of gather_kernel (scatter-only, dir=1) ---------
# Same math as gather_kernel with `direction == 1` (adjoint scatter) but
# writes only to fde rows whose absolute ky ∈ [y_lo, y_hi); rows outside
# the range are computed-and-skipped so the caller can process a full
# (2n × 2n) adjoint fde one y-band at a time — GPU peak per launch is
# `nz * (y_hi - y_lo) * 2n * 8` bytes instead of `nz * 2n * 2n * 8`.
#
# The `f` pointer addresses the ky-restricted slice fde[:, y_lo:y_hi, :]
# directly (shape (nz, h, 2n) where h = y_hi - y_lo); the kernel maps
# absolute ky → local ky - y_lo before the atomicAdd.
gather_kernel_ychunk = cp.RawKernel(
    r"""
extern "C" __global__ void gather_ychunk(float2* g, float2* f, float* theta, int m, float* mu,
                                         int n, int ntheta, int nz,
                                         int y_lo, int y_hi)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= ntheta || tz >= nz) return;

    const float PI     = 3.141592653589793238f;
    const int   twon   = 2 * n;
    const float ftwon  = (float)twon;
    const float mu0    = mu[0];
    const float coeff0 = PI / mu0;
    const float coeff1 = -PI * PI / mu0;
    const float inv_twon = 1.0f / ftwon;
    const int   h      = y_hi - y_lo;

    const float cx  = n * 0.5f;
    const float x0 =  (tx - cx) / (float)n * __cosf(theta[ty]);
    const float y0 = -(tx - cx) / (float)n * __sinf(theta[ty]);

    const int g_ind = tx + tz * n + ty * n * nz;  // swapped axes
    float2 g0 = g[g_ind];

    const int base_x = (int)rintf(ftwon * x0) - m;
    const int base_y = (int)rintf(ftwon * y0) - m;
    const int tz_off = tz * h * twon;
    const int len    = 2 * m + 1;

    // Precompute x-direction exponential factors once (same trick as
    // gather_kernel — 2·(2m+1) expf calls instead of (2m+1)^2).
    float ex[32];
    for (int i0 = 0; i0 < len; i0++) {
        float w0 = (base_x + i0) * inv_twon - x0;
        ex[i0] = __expf(coeff1 * w0 * w0);
    }

    for (int i1 = 0; i1 < len; i1++)
    {
        int   ell1    = base_y + i1;
        float w1      = ell1 * inv_twon - y0;
        float ey      = coeff0 * __expf(coeff1 * w1 * w1);
        int   f_indy  = (n + ell1 + twon) % twon;
        // Bin-chunking bounds check — skip writes outside the ky slice.
        if (f_indy < y_lo || f_indy >= y_hi) continue;
        int   f_indy_local = f_indy - y_lo;
        int   row_off = twon * f_indy_local + tz_off;

        for (int i0 = 0; i0 < len; i0++)
        {
            float w    = ex[i0] * ey;
            int   ell0 = base_x + i0;
            int   f_ind = (n + ell0 + twon) % twon + row_off;

            atomicAdd(&(f[f_ind].x), w * g0.x);
            atomicAdd(&(f[f_ind].y), w * g0.y);
        }
    }
}
""",
    "gather_ychunk",
)


# --- TomoLargeReal (rfft-half spectrum) NUFFT gather -----------------------
# Fetches from an x-strip of the stored rfft half-spectrum:
#   patch shape (nz, 2n, endx - stx)     — FULL y, columns [stx, endx) of
#                                          the (n+1)-wide stored x-axis.
# Because we always have the full 2n rows, the reflection case can index
# rk_y = (2n - k1) % 2n directly inside the kernel — no host-side y-mirror
# synthesis needed and no y-wrap corner cases at bin boundaries.  The
# host-side fetch is just fde[:, :, stx:endx] (a contiguous slice).
#
# Reflection rule (rfft conjugate symmetry, real input):
#     fde_full[k1, k0] = fde_stored[k1, k0]                        if k0 <= n
#                      = conj(fde_stored[(2n - k1) % 2n, 2n - k0]) if k0 > n
# Sign correction (-1)^(k0+k1) undoes the (-1)^(fy+fx) phase that the raw
# rfft2 leaves on a centered signal — matching the centered-spectrum layout
# that the full-complex `gather_kernel` reads from.
gather_kernel_rfft = cp.RawKernel(
    r"""
extern "C" __global__ void gather_rfft(
    float2 *g, float2 *f,
    const long long *full_idx,
    const float *cos_theta, const float *sin_theta,
    int m, float *mu,
    int nel, int stx, int endx_half, int n)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx >= nel) return;
    const float M_PI = 3.141592653589793238f;
    const float coeff0 = M_PI / mu[0];
    const float coeff1 = -M_PI * M_PI / mu[0];
    const int twon = 2 * n;
    const float ftwon = (float)twon;
    const int patch_w = endx_half - stx;

    // Recompute (x, y) from the flat sample id (see gather1 for derivation).
    long long fi = full_idx[tx];
    long long ln = (long long)n;
    int k = (int)(fi / ln);
    int j = (int)(fi - (long long)k * ln);
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
        int ell1 = base_y + i1;
        float w1 = ell1 / ftwon - y0;
        int k1 = ((ell1 % twon) + twon) % twon;

        for (int i0 = 0; i0 < 2 * m + 1; i0++) {
            int ell0 = base_x + i0;
            float w0 = ell0 / ftwon - x0;
            float w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));

            int k0 = ((ell0 % twon) + twon) % twon;
            // Reflect if k0 > n.  Full y is fetched so rk_y is always valid.
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

            // (-1)^(k0+k1) sign; invariant under reflection since 2n is even.
            float sign_kw = ((k0 + k1) & 1) ? -w : w;
            g0.x += sign_kw * v.x;
            g0.y += sign_kw * v.y;
        }
    }

    g[tx].x = g0.x / n;
    g[tx].y = g0.y / n;
}
""",
    "gather_rfft",
)


# --- TomoReal (full-grid) NUFFT gather from rfft-half spectrum -------------
# Sibling of `gather_kernel`, but reads from a (nz, 2n, n+1) half-spectrum
# (rfft output along x) instead of the full (nz, 2n, 2n) complex.  Grid /
# block launch match gather_kernel: (ceil(n/32), ceil(ntheta/32), nz) with
# (32, 32, 1) threads, one thread per (r_idx, theta_idx, z_idx) sample.
gather_kernel_rfft_full = cp.RawKernel(
    r"""
extern "C" __global__ void gather_rfft_full(
    float2 *sino,
    float2 *fde,
    float *theta,
    int m, float *mu,
    int n, int ntheta, int nz)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n || ty >= ntheta || tz >= nz) return;

    const float M_PI = 3.141592653589793238f;
    const int   twon   = 2 * n;
    const float ftwon  = (float)twon;
    const float coeff0 = M_PI / mu[0];
    const float coeff1 = -M_PI * M_PI / mu[0];
    const float cx  = n * 0.5f;
    const float x0 =  (tx - cx) / (float)n * __cosf(theta[ty]);
    const float y0 = -(tx - cx) / (float)n * __sinf(theta[ty]);

    const int g_ind  = tx + tz * n + ty * n * nz;
    const int base_x = (int)rintf(ftwon * x0) - m;
    const int base_y = (int)rintf(ftwon * y0) - m;
    const int len    = 2 * m + 1;
    const int tz_off = tz * twon * (n + 1);
    const float inv_twon = 1.0f / ftwon;

    float2 g0 = {0.0f, 0.0f};

    for (int i1 = 0; i1 < len; i1++) {
        int   ell1  = base_y + i1;
        float w1    = ell1 * inv_twon - y0;
        float ey    = coeff0 * __expf(coeff1 * w1 * w1);
        int   ky    = ((ell1 % twon) + twon) % twon;

        for (int i0 = 0; i0 < len; i0++) {
            int ell0 = base_x + i0;
            float w0 = ell0 * inv_twon - x0;
            float w  = ey * __expf(coeff1 * w0 * w0);

            int k0 = ((ell0 % twon) + twon) % twon;

            // Sign fix for rfft2 fftfreq layout vs the centered layout —
            // see gather_kernel_rfft for the derivation.
            float sign_kw = ((k0 + ky) & 1) ? -w : w;

            int rk_x, rk_y;
            bool conj_flag;
            if (k0 <= n) {
                rk_x = k0;
                rk_y = ky;
                conj_flag = false;
            } else {
                rk_x = twon - k0;
                rk_y = (twon - ky) % twon;
                conj_flag = true;
            }

            int f_ind = rk_x + (n + 1) * rk_y + tz_off;
            float2 v = fde[f_ind];
            if (conj_flag) v.y = -v.y;

            g0.x += sign_kw * v.x;
            g0.y += sign_kw * v.y;
        }
    }

    sino[g_ind].x = g0.x / n;
    sino[g_ind].y = g0.y / n;
}
""",
    "gather_rfft_full",
)


# --- Compact-input scatter (adjoint of pass3, used by _passRT2_scatter) ----
# Same NUFFT footprint math as `gather_kernel_ychunk` with `direction == 1`,
# but reads sino values, θ, and x from parallel compact arrays instead of a
# rectangular (ntheta, nz, n) grid.  One thread per compact sample; grid.z
# = nz distributes across z-slices.
#
#   g          : compact sino, shape (nz, nel) c64.
#   f          : fde slice (nz, h, 2n) c64, where h = y_hi - y_lo.
#   x_idx      : (nel,) int32 — sino x-index per sample.
#   theta_idx  : (nel,) int32 — sino θ-index per sample.
#   theta_all  : (ntheta,) f32 — full θ table.
#
# Callers precompute (x_idx, theta_idx) once per chunk_xy (see
# TomoLargeReal._per_band_precompute) and cache across RT() calls.
scatter_compact_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void scatter_compact(
    const float2 *g,
    float2       *f,
    const int    *x_idx,
    const int    *theta_idx,
    const float  *theta_all,
    int m, const float *mu,
    int n, int nel, int nz,
    int y_lo, int y_hi,
    int sino_stride, int fde_h_buf)
{
    int tk = blockDim.x * blockIdx.x + threadIdx.x;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tk >= nel || tz >= nz) return;

    const float PI      = 3.141592653589793238f;
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

    // sino_stride, fde_h_buf let the caller pass fixed-shape ping-pong
    // buffers (StreamPipe) whose actual row count may exceed the
    // current band's usable `nel` / `y_hi - y_lo`.  Kernel bounds
    // remain (`tk < nel`, `f_indy in [y_lo, y_hi)`); indexing uses
    // the buffer strides.
    float2 g0 = g[tz * sino_stride + tk];

    int base_x  = (int)rintf(ftwon * x0) - m;
    int base_y  = (int)rintf(ftwon * y0) - m;
    int len     = 2 * m + 1;
    int tz_off  = tz * fde_h_buf * twon;

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
        int   row_off = twon * (f_indy - y_lo) + tz_off;

        for (int i0 = 0; i0 < len; i0++) {
            float w     = ex[i0] * ey;
            int   ell0  = base_x + i0;
            int   f_ind = (n + ell0 + twon) % twon + row_off;
            atomicAdd(&(f[f_ind].x), w * g0.x);
            atomicAdd(&(f[f_ind].y), w * g0.y);
        }
    }
}
""",
    "scatter_compact",
)


# --- Compact-input gather (forward, used by _pass3_gather) -----------------
# Same NUFFT math as `gather_kernel_rfft` but reads pre-decoded (x_idx,
# theta_idx) int32 pairs per thread instead of decoding a `long long`
# full_idx via divmod.  z-loop stays outside (see _pass3_gather's per-z
# ComputeD2HPipe).  One thread per compact sample.
#
#   g          : compact sino out (nel,) c64 for one z.
#   f          : fde slice (2n, patch_w) c64 for one z, patch_w = endx_half - stx.
#   x_idx      : (nel,) int32 — sino x-index per sample.
#   theta_idx  : (nel,) int32 — sino θ-index per sample.
#   cos_theta  : (ntheta,) f32.
#   sin_theta  : (ntheta,) f32.
gather_compact_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void gather_compact(
    float2       *g,
    const float2 *f,
    const int    *x_idx,
    const int    *theta_idx,
    const float  *cos_theta,
    const float  *sin_theta,
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
""",
    "gather_compact",
)


# --- Symmetric (mirror) padding used by Fresnel Propagation ---------------
pad_fwd_kernel = cp.RawKernel(
    r"""
extern "C" void __global__ pad_fwd(float2* __restrict__ g,
                                    const float2* __restrict__ f,
                                    int n, int nz, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= 2*n || ty >= 2*nz || tz >= ntheta) return;

    int txx = (tx < n/2)       ? (n/2  - tx - 1)         :
              (tx >= n + n/2)   ? (2*n  - tx + n/2  - 1)  : (tx - n/2);
    int tyy = (ty < nz/2)      ? (nz/2 - ty - 1)         :
              (ty >= nz + nz/2) ? (2*nz - ty + nz/2 - 1)  : (ty - nz/2);

    g[tz*2*n*2*nz + ty*2*n + tx] = f[tz*n*nz + tyy*n + txx];
}
""",
    "pad_fwd",
)

