"""Fresnel propagation (parallel-beam angular spectrum) — vendored from
holotomocupy_mpi/propagation.py.  Uses only cupy + local kernels (the
cuFFTDx fast path from the original is dropped).
"""
from __future__ import annotations

import math

import cupy as cp
import cupyx.scipy.fft as cufft

from processing.kernels import pad_fwd_kernel, pad_adj_kernel


class Propagation:
    """Fresnel forward/adjoint propagator sized for (ntheta × nz × n)."""

    def __init__(self, n, nz, ntheta, ndist, wavelength, voxelsize, distance):
        self.n       = n
        self.nz      = nz
        self._ntheta = ntheta

        # Fresnel kernels on the padded (2n × 2nz) grid.
        fx = cp.fft.fftfreq(2 * n,  d=voxelsize).astype("float32")
        fy = cp.fft.fftfreq(2 * nz, d=voxelsize).astype("float32")
        fx, fy = cp.meshgrid(fx, fy)
        f2 = fx ** 2 + fy ** 2

        norm = float(4 * n * nz)
        self.fker = cp.empty([ndist, 2 * nz, 2 * n], dtype="complex64")
        for j in range(ndist):
            self.fker[j] = cp.exp(-1j * cp.pi * wavelength * distance[j] * f2) / norm

        # Pre-allocated work buffer to avoid per-call allocation.
        self._buf_big = cp.empty([ntheta, 2 * nz, 2 * n], dtype="complex64")
        self._plan_2d = cufft.get_fft_plan(self._buf_big, axes=(-2, -1), value_type='C2C')

    def _fwd_pad(self, f, fpad):
        """Symmetric padding: f (ntheta, nz, n) → fpad (ntheta, 2nz, 2n)."""
        ntheta, nz, n = f.shape
        f = cp.ascontiguousarray(f)
        pad_fwd_kernel(
            (math.ceil(2 * n / 32), math.ceil(2 * nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta),
        )

    def _adj_pad(self, fpad, f):
        """Adjoint padding: (ntheta, 2nz, 2n) → f (ntheta, nz, n)."""
        ntheta = fpad.shape[0]
        nz     = fpad.shape[1] // 2
        n      = fpad.shape[2] // 2
        fpad = cp.ascontiguousarray(fpad)
        pad_adj_kernel(
            (math.ceil(n / 32), math.ceil(nz / 32), ntheta),
            (32, 32, 1),
            (fpad, f, n, nz, ntheta),
        )

    def D(self, psi, j):
        """Forward Fresnel propagation with the j-th kernel."""
        added_dim = psi.ndim == 2
        if added_dim:
            psi = psi[cp.newaxis]

        ntheta = psi.shape[0]
        self._buf_big.fill(0)
        self._fwd_pad(psi, self._buf_big[:ntheta])
        with self._plan_2d:
            cufft.fft2(self._buf_big, overwrite_x=True)
        self._buf_big *= self.fker[j]
        with self._plan_2d:
            cufft.ifft2(self._buf_big, overwrite_x=True, norm="forward")
        result = self._buf_big[:ntheta,
                               self.nz // 2 : -self.nz // 2,
                               self.n  // 2 : -self.n  // 2].copy()
        return result[0] if added_dim else result

    def DT(self, big_psi, j):
        """Adjoint Fresnel propagator with the conjugate of the j-th kernel."""
        added_dim = big_psi.ndim == 2
        if added_dim:
            big_psi = big_psi[cp.newaxis]

        ntheta = big_psi.shape[0]
        self._buf_big.fill(0)
        self._buf_big[:ntheta,
                      self.nz // 2 : -self.nz // 2,
                      self.n  // 2 : -self.n  // 2] = big_psi
        with self._plan_2d:
            cufft.fft2(self._buf_big, overwrite_x=True)
        self._buf_big *= self.fker[j].conj()
        with self._plan_2d:
            cufft.ifft2(self._buf_big, overwrite_x=True, norm="forward")

        result = cp.zeros_like(big_psi)
        self._adj_pad(self._buf_big[:ntheta], result)
        return result[0] if added_dim else result
