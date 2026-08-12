"""Paganin single-distance phase retrieval — tomocupy variant, rfft path.

For each 2-D intensity I(y, x):
    H(kx, ky) = α / (λ·z·(kx² + ky²)/(4π) + α)     # DC = 1
    output    = −log(F⁻¹[H · F(I)])                  # line-integrated μ

Regularised TIE inversion (no homogeneous-object assumption / no δ/β in
the filter); output is the line-integrated linear attenuation
coefficient, exactly what tomocupy's `paganin_filter` + `minus_log`
produce.  Sample pixels come out POSITIVE, air ≈ 0.

Uses rfft2 / irfft2 along (y, x): the intensity is REAL float32 so the
Fourier spectrum has conj-symmetry along x and the half-plane
(nz, n//2+1) carries all the unique information.  Compared to the old
c64 fft2 / c64 mul / c64 ifft2 pipeline this halves the compute along
x and lets us store the spectrum as a smaller c64 buffer instead of a
full (ntheta, nz, n) c64.

For volumes too large to hold the (ntheta, nz, n) f32 intensity plus
the (ntheta, nz, n//2+1) c64 spectrum on the GPU, use
processing.paganin_large.PaganinLarge instead.
"""
from __future__ import annotations

import cupy as cp
import cupyx.scipy.fft as cufft


class Paganin:
    """Single-distance Paganin phase retrieval, tomocupy filter (rfft path).

    Buffers + cufft plans (R2C forward + C2R inverse) are cached across
    `.retrieve()` calls; safe to reuse across theta batches of the same
    ntheta.
    """

    def __init__(self, n, nz, ntheta,
                 wavelength, voxelsize, distance,
                 alpha):
        self.n       = n
        self.nz      = nz
        self._ntheta = ntheta

        # Build the filter directly on the rfft2 output grid so cufft's
        # native (fftfreq, rfftfreq) bin layout is used with no shift.
        fy = cp.fft.fftfreq(nz, d=voxelsize).astype("float32")     # (nz,)
        fx = cp.fft.rfftfreq(n, d=voxelsize).astype("float32")     # (n//2+1,)
        w2 = fy[:, None] ** 2 + fx[None, :] ** 2
        # Real filter is fine — cupy broadcasts real × complex; keeping
        # it in f32 saves the ×2 memory of a c64 copy.
        self.filt = (alpha / (wavelength * distance * w2 / (4.0 * cp.pi)
                              + alpha)).astype("float32")

        # Work buffers: (ntheta, nz, n) real intensity + (ntheta, nz, n//2+1)
        # c64 spectrum.  Together ~= (ntheta·nz·n) × 8B, same as the old
        # c64-only pipeline but with half the compute along x.
        self._intens = cp.empty((ntheta, nz, n),         dtype="float32")
        self._spec   = cp.empty((ntheta, nz, n // 2 + 1), dtype="complex64")
        self._plan_fwd = cufft.get_fft_plan(self._intens, axes=(-2, -1),
                                            value_type="R2C")
        self._plan_inv = cufft.get_fft_plan(self._spec,   axes=(-2, -1),
                                            value_type="C2R")

    def retrieve(self, intensity):
        """Paganin phase retrieval on a batch of real-f32 intensities.

        intensity : (ntheta, nz, n) or (nz, n) real float32 on GPU.
        Returns   : same-shape real float32 phase φ on GPU.
        """
        added_dim = intensity.ndim == 2
        if added_dim:
            intensity = intensity[cp.newaxis]
        b = intensity.shape[0]
        assert b <= self._ntheta, \
            f"batch {b} > cached ntheta {self._ntheta}"

        # Copy in and zero any tail slots so the fixed-size cuFFT plan
        # doesn't process stale data.
        self._intens[:b] = intensity
        if b < self._ntheta:
            self._intens[b:].fill(0)

        with self._plan_fwd:
            self._spec[...] = cufft.rfft2(self._intens, axes=(-2, -1))
        self._spec *= self.filt
        with self._plan_inv:
            self._intens[...] = cufft.irfft2(self._spec, s=(self.nz, self.n),
                                             axes=(-2, -1))

        # -log(clip) matches tomocupy's minus_log convention: sample → positive µL.
        out = -cp.log(self._intens[:b].clip(1e-6))
        return out[0] if added_dim else out
