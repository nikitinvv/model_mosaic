"""Tomo (GPU-only USFFT Radon) — vendored from holotomocupy_mpi/tomo.py.

Depends only on numpy, cupy, cupyx, and the local kernels module.  Suitable
for volumes where a (nz, 2N, 2N) complex64 buffer fits on the GPU.  For
larger N use tomo_large.py's host-chunked TomoLarge instead.
"""
from __future__ import annotations

import math
import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft

from kernels import gather_kernel


class Tomo:
    """Radon transform via USFFT (real object → complex sinogram).

    Only R() and RT() are exposed here; reconstruction helpers (rec_tomo,
    fbp, _filter_sino) from the original holotomocupy Tomo were dropped
    because model_radon_* does not need them.
    """

    def __init__(self, n, nz, theta, mask_r=0.0):
        """USFFT parameters + preallocated buffers for (nz, 2n, 2n) FFTs."""
        eps = 1e-3  # accuracy of usfft
        mu = -math.log(eps) / (2 * n * n)
        m  = math.ceil(2 * n / math.pi * math.sqrt(-mu * math.log(eps) + (mu * n) ** 2 / 4))

        # Gaussian pre-multiplication (USFFT convolution kernel in real space)
        t = cp.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        dx, dy = cp.meshgrid(t, t)
        phi = cp.exp((mu * (n * n) * (dx * dx + dy * dy)).astype("float32")) * (1 - n % 4)

        # (+1,-1) sign arrays for fftshift-via-multiply.
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dtmp      = (1 - 2 * ((cp.arange(1, 2 * n + 1) % 2))).astype("int8")
        c2dfftshift = cp.outer(c2dtmp, c2dtmp)

        mua = cp.array([mu], dtype="float32")

        self.n      = n
        self.ntheta = len(theta)
        self.theta  = cp.array(theta.astype("float32"))

        # Optional soft circular mask in the pre-multiplication.
        if mask_r > 0:
            t1d = np.linspace(-1, 1, self.n)
            x, y = np.meshgrid(t1d, t1d)
            circ  = (x**2 + y**2 < mask_r).astype("float32")
            g     = np.exp(-(20**2) * (x**2 + y**2))
            fcirc = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            fg    = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))
            mask  = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(fcirc * fg))).real.astype("float32")
            mask /= np.amax(mask)
        else:
            mask = 1.0
        self.mask = mask
        phi *= cp.array(mask / (n * np.sqrt(n * self.ntheta)))

        self.pars = m, mua, phi, c1dfftshift, c2dfftshift
        self._buf_fde  = cp.empty([nz, 2 * n, 2 * n], dtype="complex64")
        self._buf_sino = cp.zeros([self.ntheta, nz, n], dtype="complex64")
        self._nz       = nz
        self._plan_2d  = cufft.get_fft_plan(self._buf_fde,  axes=(-2, -1), value_type='C2C')
        self._plan_1d  = cufft.get_fft_plan(self._buf_sino, axes=(-1,),    value_type='C2C')

    def R(self, obj):
        """Radon transform: (nz, n, n) obj → (ntheta, nz, n) sinogram."""
        nz = obj.shape[0]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # Pre-multiply and zero-pad the object into buf_fde.
        self._buf_fde.fill(0)
        cp.multiply(obj, phi, out=self._buf_fde[:nz, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2])
        # 2-D FFT via fftshift-multiply trick.
        self._buf_fde *= c2dfftshift
        with self._plan_2d:
            cufft.fft2(self._buf_fde, overwrite_x=True)
        self._buf_fde *= c2dfftshift
        # NUFFT gather onto Fourier slices at each θ.
        self._buf_sino.fill(0)
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta, m, mua, n, self.ntheta, self._nz, 0),
        )
        # 1-D IFFT along the sample axis.
        self._buf_sino *= c1dfftshift
        with self._plan_1d:
            cufft.ifft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        # Normalisation + crop back to the caller's nz.
        result = self._buf_sino[:, :nz] / 4
        if obj.dtype == 'float32':
            result = result.real
        return cp.ascontiguousarray(result)

    def RT(self, data):
        """Adjoint Radon: (ntheta, nz, n) sinogram → (nz, n, n) obj."""
        nz = data.shape[1]
        n  = self.n
        m, mua, phi, c1dfftshift, c2dfftshift = self.pars

        # 1-D FFT of the sinogram along the sample axis.
        self._buf_sino[:, :nz] = (data * c1dfftshift).astype('complex64')
        self._buf_sino[:, nz:] = 0
        with self._plan_1d:
            cufft.fft(self._buf_sino, overwrite_x=True)
        self._buf_sino *= c1dfftshift
        # Adjoint NUFFT scatter.
        self._buf_fde.fill(0)
        gather_kernel(
            (math.ceil(n / 32), math.ceil(self.ntheta / 32), self._nz),
            (32, 32, 1),
            (self._buf_sino, self._buf_fde, self.theta, m, mua, n, self.ntheta, self._nz, 1),
        )
        # 2-D IFFT and crop back.
        self._buf_fde *= c2dfftshift
        with self._plan_2d:
            cufft.ifft2(self._buf_fde, overwrite_x=True)
        self._buf_fde *= c2dfftshift
        result = self._buf_fde[:nz, n // 2 : 3 * n // 2, n // 2 : 3 * n // 2] * phi
        if data.dtype == 'float32':
            result = result.real
        return cp.ascontiguousarray(result)
