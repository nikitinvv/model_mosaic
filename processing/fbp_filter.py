"""FBP filter — vendored from tomocupy/reconstruction/fbp_filter.py.

Applies a 1-D filter along the sample axis of a real sinogram via
edge-pad (n → ne = 4·n) → rfft → multiply → irfft → crop back to n.
The zero-padding factor 4× matches tomocupy's `params.ne = 4*params.n`
and suppresses the circular-convolution wrap that a bare rfft-length-n
would produce (sinogram rows are anything but zero at the edges,
because the sample cylinder is inscribed in the field of view).

The filter weights come from tomocupy's `calc_filter(name)` — the
formula is unchanged, only rescaled to the padded grid (t = k/ne,
0.5·ne coefficient, /ne IFFT-normalization).  All 8 standard tomo
filters are supported via the shared 12-point quadrature helper
`_wint`.  Weights are bit-identical to tomocupy's when its
`FBPFilter` is instantiated with `params.ne`, which is how tomocupy
uses it in `backproj_functions.BackprojFunctions`.

Tomocupy's original class calls a proprietary C++ CUDA kernel
`cfunc_filter.filter()` for the FFT-multiply-IFFT step; we replace
that with plain cupy `rfft`/`irfft` (same cuFFT plan under the hood).
The edge-replication padding, which lives in tomocupy's
`fbp_filter_center()` at the caller side, is folded into
`FBPFilter.filter()` here so step8_fbp / step8_fbp_large stay simple.

`filter()` chunks along the leading batch axis to bound GPU peak
memory: each slab holds a (batch_chunk, ne) f32 scratch + a
(batch_chunk, ne/2+1) c64 spectrum + one transient irfft output ≈
3·batch_chunk·ne·4 bytes.  Default batch_chunk=256 is enough for a
UPS=64 sinogram (N=196608 → ~600 MB scratch) and can be raised for
smaller UPS or lowered for tight GPUs.  A batch_chunk >= batch runs
in one pass, bit-exact to the pre-chunked behaviour.
"""
from __future__ import annotations

import numpy as np
import cupy as cp


class FBPFilter:
    """Cupy-only FBP 1-D filter with edge-replicated 4× zero-padding."""

    def __init__(self, n: int, batch_chunk: int = 256):
        self.n  = n
        self.ne = 4 * n                          # padded working length
        self.pad = (self.ne - self.n) // 2       # equal on both sides
        self.batch_chunk = int(batch_chunk)
        self._cache: dict[str, cp.ndarray] = {}
        self._tmp: cp.ndarray | None = None      # (batch_chunk, ne) padded scratch

    def calc_filter(self, name: str) -> cp.ndarray:
        """Return the (ne//2 + 1,) real-f32 filter weights for the named
        tomocupy filter — identical formula to tomocupy's
        `FBPFilter.calc_filter(ne)`, cached per name."""
        if name in self._cache:
            return self._cache[name]

        d = 0.5
        ne = self.ne
        t = cp.arange(0, ne // 2 + 1) / ne

        if name == 'none':
            wfa = ne * 0.5 + t * 0
            wfa[0] *= 2  # fixed later
        elif name == 'ramp':
            wfa = ne * 0.5 * self._wint(12, t)
        elif name == 'shepp':
            wfa = ne * 0.5 * self._wint(12, t) * cp.sinc(t / (2 * d)) * (t / d <= 2)
        elif name == 'cosine':
            wfa = ne * 0.5 * self._wint(12, t) * cp.cos(cp.pi * t / (2 * d)) * (t / d <= 1)
        elif name == 'cosine2':
            wfa = ne * 0.5 * self._wint(12, t) * (cp.cos(cp.pi * t / (2 * d))) ** 2 * (t / d <= 1)
        elif name == 'hamming':
            wfa = ne * 0.5 * self._wint(12, t) * (.54 + .46 * cp.cos(cp.pi * t / d)) * (t / d <= 1)
        elif name == 'hann':
            wfa = ne * 0.5 * self._wint(12, t) * (1 + cp.cos(cp.pi * t / d)) / 2.0 * (t / d <= 1)
        elif name == 'parzen':
            wfa = ne * 0.5 * self._wint(12, t) * (1 - t / d) ** 3 * (t / d <= 1)
        else:
            raise ValueError(
                f"unknown FBP filter '{name}'; expected one of "
                f"none, ramp, shepp, cosine, cosine2, hamming, hann, parzen")

        wfa = 2 * wfa * (wfa >= 0)
        wfa[0] *= 2

        # Scale adjustments, added minus and *3 to make init=rec
        wfa = (3 * self.n * wfa ).astype('float32')

        self._cache[name] = wfa
        return wfa 

    @staticmethod
    def _wint(n: int, t: cp.ndarray) -> cp.ndarray:
        """12-point Gauss-Legendre quadrature weights on [t_i, t_{i+n-1}].

        Bit-identical port of tomocupy's `FBPFilter._wint(n, t)`.
        `n` is the quadrature order (tomocupy always passes 12), `t` is
        the (n//2+1,) frequency grid.
        """
        N = len(t)
        s = cp.linspace(1e-40, 1, n)
        # Inverse Vandermonde matrix.
        tmp1 = cp.arange(n)
        tmp2 = cp.arange(1, n + 2)
        iv = cp.linalg.inv(cp.exp(cp.outer(tmp1, cp.log(s))))
        u = cp.diff(cp.exp(cp.outer(tmp2, cp.log(s))) * cp.tile(
            1.0 / tmp2[..., cp.newaxis], [1, n]))  # integration over short intervals
        W1 = cp.matmul(iv, u[1:n + 1, :])   # x·pn(x) term
        W2 = cp.matmul(iv, u[0:n, :])       # const·pn(x) term

        # Compensate for overlapping short intervals.
        tmp1 = cp.arange(1, n)
        tmp2 = (n - 1) * cp.ones((N - 2 * (n - 1) - 1))
        tmp3 = cp.arange(n - 1, 0, -1)
        p = 1 / cp.concatenate((tmp1, tmp2, tmp3))
        w = cp.zeros(N)
        for j in range(N - n + 1):
            W = ((t[j + n - 1] - t[j]) ** 2) * W1 + (t[j + n - 1] - t[j]) * t[j] * W2
            for k in range(n - 1):
                w[j:j + n] = w[j:j + n] + p[j + k] * W[:, k]
        # Tomocupy hardcodes a linear taper over the last 40 samples;
        # guard for tiny N (test sizes) where that would out-of-bound.
        wn = w
        if N > 40:
            wn[-40:] = (w[-40]) / (N - 40) * cp.arange(N - 40, N)
        return wn

    def filter(self, sino: cp.ndarray, w: cp.ndarray,
               batch_chunk: int | None = None) -> cp.ndarray:
        """In-place FBP filter of `sino` along its LAST axis, with
        tomocupy-style 4× edge-replicated zero-padding.

        sino: (..., n)         real float32 on GPU — the sinogram.
        w   : (ne//2 + 1,)     real float32 on GPU — output of calc_filter.
        batch_chunk : optional slab size along the flattened leading
            dims.  None → use the constructor default (256).  Set to
            the full batch to run in a single pass (bit-exact match to
            the pre-chunked behaviour, higher peak memory).

        Steps (per slab, matching tomocupy `fbp_filter_center` +
        `cfunc_filter`):
          1. tmp[..., pad:pad+n] = sino                     (copy in)
             tmp[..., :pad]      = sino[..., :1]            (left edge)
             tmp[..., pad+n:]    = sino[..., -1:]           (right edge)
          2. tmp = irfft(rfft(tmp) * w, n=ne)               (padded filter)
          3. sino[...] = tmp[..., pad:pad+n]                (crop back)

        We do NOT fold in the rotation-center phase shift that tomocupy
        applies (`exp(-2πj·(-center + sht + n/2)·t)`) — our synthetic
        pipeline has center = n/2 and sht = 0, so that factor is unity.

        Sino must be C-contiguous — we flatten the leading dims to a
        single batch axis via a reshape view, so a non-contiguous input
        would make the write-back land in a temporary and leak the
        filtered result.

        Returns `sino` (same object, filtered in place).
        """
        if sino.shape[-1] != self.n:
            raise ValueError(
                f"sino last-axis {sino.shape[-1]} != filter n={self.n}")
        if not sino.flags.c_contiguous:
            raise ValueError(
                "sino must be C-contiguous (reshape view for chunk loop)")

        pad, n, ne = self.pad, self.n, self.ne
        orig_shape = sino.shape
        batch = int(np.prod(orig_shape[:-1]))
        sino_flat = sino.reshape(batch, n)     # view (contiguous input)

        # Peak per slab ≈ 3·batch_chunk·ne·4 bytes (padded f32 tmp +
        # c64 spectrum ≈ same size + one transient irfft output before
        # it lands into tmp).
        if batch_chunk is None:
            batch_chunk = self.batch_chunk
        batch_chunk = min(int(batch_chunk), batch)

        # Padded scratch reused across slabs — reallocated only when
        # `batch_chunk` changes size class.
        if (self._tmp is None
                or self._tmp.shape != (batch_chunk, ne)
                or self._tmp.dtype != sino.dtype):
            self._tmp = cp.empty((batch_chunk, ne), dtype=sino.dtype)

        # rfft/irfft along last axis with default norm 'backward' (fft
        # unnormalized, ifft ÷ ne).  The ÷ ne is folded into w in
        # calc_filter, so combined normalization is 1.  A ragged last
        # slab (batch not divisible by batch_chunk) uses a view of tmp,
        # which forces cuFFT to build a second plan for that size — one
        # extra plan per whole `.filter()` call, negligible.
        for i0 in range(0, batch, batch_chunk):
            i1 = min(i0 + batch_chunk, batch)
            t  = self._tmp[: i1 - i0]
            src = sino_flat[i0:i1]
            t[:, pad:pad + n] = src
            t[:, :pad]        = src[:, :1]
            t[:, pad + n:]    = src[:, -1:]
            spec = cp.fft.rfft(t, axis=-1)
            spec *= w
            t[...] = cp.fft.irfft(spec, n=ne, axis=-1)
            sino_flat[i0:i1] = t[:, pad:pad + n]
        return sino

    def filter_host(self, sino_h: np.ndarray, w: cp.ndarray,
                    batch_chunk: int | None = None) -> np.ndarray:
        """Filter a HOST sinogram in place, chunking the H2D → filter →
        D2H roundtrip.  Bounds GPU peak to roughly 13·batch_chunk·n·4
        bytes (input slab + 3·batch_chunk·ne·4 = 12·batch_chunk·n·4
        filter scratch) instead of the full-sino upload.

        Same batch_chunk semantics as `.filter()`.  Requires C-contiguous
        input.  Returns `sino_h` (mutated in place).

        Rationale: callers otherwise do
            d = cp.asarray(host_sino); self.filter(d, w); host_sino[...] = cp.asnumpy(d)
        which peaks at (full-sino f32 on GPU) + filter scratch.  At
        UPS=32 that full sino alone is 29 GB, blowing the GPU.  This
        method makes the chunk loop include the H2D so both the slab
        and its filter scratch peak together stay bounded.
        """
        if sino_h.shape[-1] != self.n:
            raise ValueError(
                f"sino_h last-axis {sino_h.shape[-1]} != filter n={self.n}")
        if not sino_h.flags.c_contiguous:
            raise ValueError(
                "sino_h must be C-contiguous (reshape view for chunk loop)")

        orig_shape = sino_h.shape
        batch = int(np.prod(orig_shape[:-1]))
        sino_flat_h = sino_h.reshape(batch, self.n)

        if batch_chunk is None:
            batch_chunk = self.batch_chunk
        batch_chunk = min(int(batch_chunk), batch)

        for i0 in range(0, batch, batch_chunk):
            i1 = min(i0 + batch_chunk, batch)
            d = cp.asarray(sino_flat_h[i0:i1])          # H2D slab
            self.filter(d, w, batch_chunk=(i1 - i0))    # in place on GPU
            sino_flat_h[i0:i1] = cp.asnumpy(d)          # D2H slab
            del d
        return sino_h
