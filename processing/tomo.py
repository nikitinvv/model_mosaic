"""TomoReal — GPU-only USFFT Radon (real float32 obj, rfft-along-x forward).

Depends only on numpy, cupy, cupyx, and the local kernels module.  Suitable
for volumes where a (nz, 2N, 2N) complex64 buffer fits on the GPU.  For
larger N use tomo_large.py's host-chunked TomoLargeReal instead.
"""
from __future__ import annotations

import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft

from processing.kernels import gather_kernel, gather_kernel_rfft_full


class TomoReal:
    """Radon transform via USFFT — REAL float32 obj + rfft-along-x forward.

    Forward R uses an rfft along x (halved fde memory + halved x-FFT
    cost); the gather kernel folds the missing negative-fx half of the
    spectrum via conjugate symmetry (X[fx, fy] = conj(X[-fx, -fy])).

    Adjoint RT keeps a pragmatic full-complex path (rfft-aware adjoint
    scatter is deferred): a separate (nz, 2n, 2n) complex64 `_buf_fde_full`
    buffer + C2C plan are allocated lazily on the first RT() call so
    R-only workloads don't pay for them.
    """

    def __init__(self, n, nz, theta):
        """USFFT parameter setup + forward-only preallocated buffers.

        Fftshift trick along x is dropped (rfft is asymmetric in shape).
        fde is stored in RAW fftfreq order along both x and y — no
        c2dfftshift needed for R.  For RT the c2dfftshift is rebuilt
        lazily inside _lazy_init_rt() because RT operates on a full
        (nz, 2n, 2n) complex64 fde with the fftshift-via-multiply trick.
        """
        eps = 1e-3
        mu  = -math.log(eps) / (2 * n * n)
        m   = math.ceil(2 * n / math.pi *
                        math.sqrt(-mu * math.log(eps) + (mu * n) ** 2 / 4))

        # phi — real Gaussian pre-multiplication.  Same normalisation as
        # the old complex64 Tomo so R(TomoReal) matches R(Tomo).
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        dx, dy = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) \
              * (1 - n % 4)

        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")

        mua = cp.array([mu], dtype="float32")

        self.n      = n
        self.ntheta = len(theta)
        self.theta  = cp.array(theta.astype("float32"))

        phi *= 1.0 / (n * np.sqrt(n * self.ntheta))

        # Note: `pars` matches R's needs (no c2dfftshift).  RT rebuilds
        # its own params inside _lazy_init_rt() to keep the two paths'
        # setups decoupled.
        self.pars = m, mua, phi, c1dfftshift
        self._mu  = mu

        # Buffers used by R.
        #   _buf_padded  — real, holds phi*obj center-placed in a 2n×2n grid.
        #   _buf_fde     — complex64, rfft2 output shape (nz, 2n, n+1).
        #   _buf_sino    — complex64 sinogram; .real is the output of R,
        #                  also reused by RT (same shape).
        self._buf_padded = cp.zeros([nz, 2 * n, 2 * n],     dtype="float32")
        self._buf_fde    = cp.empty([nz, 2 * n, n + 1],     dtype="complex64")
        self._buf_sino   = cp.zeros([self.ntheta, nz, n],   dtype="complex64")
        self._nz         = nz

        # Forward plans: rfft2 (R2C) along last two axes for R's pad+FFT;
        # C2C ifft along the last axis (r) for R's post-gather 1-D IFFT.
        # `_plan_1d` is also reused by RT (same buffer, same axis).
        self._plan_r2c = cufft.get_fft_plan(
            self._buf_padded, axes=(-2, -1), value_type='R2C')
        self._plan_1d  = cufft.get_fft_plan(
            self._buf_sino, axes=(-1,), value_type='C2C')

        # RT-only state (allocated on first RT() call).  See _lazy_init_rt.
        self._buf_fde_full = None      # (nz, 2n, 2n) c64
        self._plan_2d_full = None      # C2C plan for _buf_fde_full
        self._pars_rt      = None      # (m, mua, phi, c1dfftshift, c2dfftshift)

    def _lazy_init_rt(self):
        """Allocate the (nz, 2n, 2n) c64 fde buffer + its C2C plan +
        the c2dfftshift table on first RT() call.  R-only workloads pay
        nothing for these."""
        if self._buf_fde_full is not None:
            return
        n = self.n
        # c2dfftshift for the full 2n × 2n c64 fde.
        c2dtmp      = (1 - 2 * ((cp.arange(1, 2 * n + 1) % 2))).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        m, mua, phi, c1dfftshift = self.pars
        self._pars_rt = (m, mua, phi, c1dfftshift, c2dfftshift)

        self._buf_fde_full = cp.empty([self._nz, 2 * n, 2 * n],
                                      dtype="complex64")
        self._plan_2d_full = cufft.get_fft_plan(
            self._buf_fde_full, axes=(-2, -1), value_type='C2C')

    def free(self):
        """Drop persistent GPU buffers (both R and RT) — teardown for size
        sweeps.  cuFFT plans are released implicitly with their buffers."""
        self._buf_padded   = None
        self._buf_fde      = None
        self._buf_sino     = None
        self._buf_fde_full = None
        self._plan_r2c     = None
        self._plan_1d      = None
        self._plan_2d_full = None
        cp.get_default_memory_pool().free_all_blocks()

    def R(self, obj):
        """Radon transform: (nz, n, n) REAL float32 obj → (ntheta, nz, n)
        real float32 sinogram."""
        nz = obj.shape[0]
        n  = self.n
        m, mua, phi, c1dfftshift = self.pars

        # 1. Pre-mul phi + center-place into the padded 2n×2n buffer.
        self._buf_padded.fill(0)
        cp.multiply(obj, phi,
                    out=self._buf_padded[:nz, n // 2 : 3 * n // 2,
                                             n // 2 : 3 * n // 2])
        # 2. rfft2 → (nz, 2n, n+1) complex64 in raw fftfreq order.
        with self._plan_r2c:
            self._buf_fde[...] = cufft.rfft2(
                self._buf_padded, axes=(-2, -1))
        # 3. NUFFT gather (rfft-aware kernel with conj reflection).
        self._buf_sino.fill(0)
        gather_kernel_rfft_full(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta,
             m, mua, n, self.ntheta, self._nz),
        )
        # 4. 1-D IFFT along r + fftshift trick + /4 normalisation.
        self._buf_sino *= c1dfftshift
        with self._plan_1d:
            cufft.ifft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        result = self._buf_sino[:, :nz].real / 4
        return cp.ascontiguousarray(result)

    def RT(self, data):
        """Adjoint Radon: (ntheta, nz, n) sinogram → (nz, n, n) obj.

        Full complex64 path — allocates (nz, 2n, 2n) c64 fde on the GPU.
        Accepts either float32 or complex64 sinos; returns matching dtype
        (.real for float32 input, full complex64 otherwise).
        """
        self._lazy_init_rt()
        nz = data.shape[1]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self._pars_rt

        # 1-D FFT of the sinogram along the sample axis.
        self._buf_sino[:, :nz] = (data * c1dfftshift).astype('complex64')
        self._buf_sino[:, nz:] = 0
        with self._plan_1d:
            cufft.fft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        # Adjoint NUFFT scatter into the full (nz, 2n, 2n) c64 fde.
        self._buf_fde_full.fill(0)
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde_full, self.theta,
             m, mua, n, self.ntheta, self._nz, 1),
        )
        # 2-D IFFT and crop back.
        self._buf_fde_full *= c2dfftshift
        with self._plan_2d_full:
            cufft.ifft2(self._buf_fde_full, overwrite_x=True)
        self._buf_fde_full *= c2dfftshift
        result = self._buf_fde_full[:nz, n // 2 : 3 * n // 2,
                                        n // 2 : 3 * n // 2] * phi
        if data.dtype == 'float32':
            result = result.real
        return cp.ascontiguousarray(result)
