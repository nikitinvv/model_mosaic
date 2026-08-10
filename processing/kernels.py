"""Raw CUDA kernels used by tomo.py, tomo_large.py, propagation.py.

Vendored from holotomocupy_mpi/src/holotomocupy/cuda_kernels.py and
radon_large/cuda_kernels.py so this pipeline has no external dependencies
beyond cupy/numpy/tifffile/mpi4py.
"""
from __future__ import annotations

import cupy as cp


# --- Tomo (GPU-only) NUFFT gather: modular wrap over (2n × 2n) grid --------
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


# --- TomoLarge (host-chunked) NUFFT gather: chunk-relative indexing --------
# Sample coordinates are computed inside the kernel from (full_idx, cos_theta,
# sin_theta, n) instead of two 4·ntheta·n-byte host tables.  full_idx[tx] is
# the flat sample id (theta_k · n + r_j) in the theta-major layout; k = fi/n,
# j = fi%n, r = (j − n/2)/n, then x = cos_theta[k]·r, y = −sin_theta[k]·r
# clipped to [-0.5, 0.5 − 1e-5] to match the host np.clip in _sort_into_chunks.
gather_kernel1 = cp.RawKernel(
    r"""
extern "C" __global__ void gather1(float2 *g, float2 *f,
                       const long long *full_idx,
                       const float *cos_theta, const float *sin_theta,
                       int m, float *mu, int nel,
                       int stx, int endx, int sty, int endy, int n, bool direction)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tx >= nel) return;
    float M_PI = 3.141592653589793238f;
    float2 g0;
    float w, coeff0;
    float w0, w1, x0, y0, coeff1;
    int ell0, ell1, g_ind, f_ind, f_indx, f_indy;

    long long fi = full_idx[tx];
    long long ln = (long long)n;
    int k = (int)(fi / ln);
    int j = (int)(fi - (long long)k * ln);
    float r = ((float)j - 0.5f * (float)n) / (float)n;
    x0 =  cos_theta[k] * r;
    y0 = -sin_theta[k] * r;
    if (x0 < -0.5f)         x0 = -0.5f;
    if (x0 > 0.5f - 1e-5f)  x0 = 0.5f - 1e-5f;
    if (y0 < -0.5f)         y0 = -0.5f;
    if (y0 > 0.5f - 1e-5f)  y0 = 0.5f - 1e-5f;

    g_ind = tx;
    if (direction == 0) {
        g0.x = 0.0f;
        g0.y = 0.0f;
    } else {
        g0.x = g[g_ind].x;
        g0.y = g[g_ind].y;
    }

    coeff0 = M_PI / mu[0];
    coeff1 = -M_PI * M_PI / mu[0];

    for (int i1 = 0; i1 < 2 * m + 1; i1++)
    {
        ell1 = (int)(rintf(2 * n * y0)) - m + i1;
        for (int i0 = 0; i0 < 2 * m + 1; i0++)
        {
            ell0 = (int)(rintf(2 * n * x0)) - m + i0;
            w0 = ell0 / (float)(2 * n) - x0;
            w1 = ell1 / (float)(2 * n) - y0;
            w = coeff0 * exp(coeff1 * (w0 * w0 + w1 * w1));
            f_indx = n - stx + ell0;
            f_indy = n - sty + ell1;

            if ((f_indx < 0) || (f_indx >= endx - stx)) continue;
            if ((f_indy < 0) || (f_indy >= endy - sty)) continue;

            f_ind = f_indx + (endx - stx) * f_indy;
            if (direction == 0) {
                g0.x += w * f[f_ind].x;
                g0.y += w * f[f_ind].y;
            } else {
                float *fx = &(f[f_ind].x);
                float *fy = &(f[f_ind].y);
                atomicAdd(fx, w * g0.x);
                atomicAdd(fy, w * g0.y);
            }
        }
    }
    if (direction == 0) {
        g[g_ind].x = g0.x / n;
        g[g_ind].y = g0.y / n;
    }
}
""",
    "gather1",
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
# rfft2 leaves on a centered signal — matching TomoLarge's centered-spectrum
# layout that gather_kernel1 reads from.
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

            // Sign fix for rfft2 fftfreq layout vs Tomo's centered layout —
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

pad_adj_kernel = cp.RawKernel(
    r"""
/* Adjoint of pad_fwd: launch over f (n x nz).
   Each f[tx,ty] gathers from exactly 4 symmetric locations in g — no atomics. */
extern "C" void __global__ pad_adj(const float2* __restrict__ g,
                                    float2* __restrict__ f,
                                    int n, int nz, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n || ty >= nz || tz >= ntheta) return;

    int gx_c = tx + n/2;
    int gx_m = (tx < n/2) ? (n/2 - 1 - tx) : (2*n + n/2 - 1 - tx);
    int gy_c = ty + nz/2;
    int gy_m = (ty < nz/2) ? (nz/2 - 1 - ty) : (2*nz + nz/2 - 1 - ty);

    const float2* base = g + tz * 2*n * 2*nz;
    float2 v0 = base[gy_c*2*n + gx_c];
    float2 v1 = base[gy_c*2*n + gx_m];
    float2 v2 = base[gy_m*2*n + gx_c];
    float2 v3 = base[gy_m*2*n + gx_m];
    f[tz*n*nz + ty*n + tx] = {v0.x+v1.x+v2.x+v3.x, v0.y+v1.y+v2.y+v3.y};
}
""",
    "pad_adj",
)
