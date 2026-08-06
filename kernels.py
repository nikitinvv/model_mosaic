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
gather_kernel1 = cp.RawKernel(
    r"""
extern "C" __global__ void gather1(float2 *g, float2 *f, float *x, float *y, int m,
                       float *mu, int nel, int stx, int endx, int sty, int endy, int n, bool direction)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tx >= nel) return;
    float M_PI = 3.141592653589793238f;
    float2 g0;
    float w, coeff0;
    float w0, w1, x0, y0, coeff1;
    int ell0, ell1, g_ind, f_ind, f_indx, f_indy;

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
    x0 = x[tx];
    y0 = y[tx];

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
