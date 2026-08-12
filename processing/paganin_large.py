"""PaganinLarge — host-chunked single-distance Paganin phase retrieval,
3-pass streaming, x-axis as rfft.  For volumes too big for a full-2D
GPU FFT.  Mirrors PropagationLarge's 3-pass streaming pattern: the
(ntheta, nz, n//2+1) complex spectrum lives on the HOST; only one
z-strip or x-freq strip at a time is on the GPU, so peak GPU memory
is proportional to the chunk sizes rather than to the full frame.

Formula (tomocupy `paganin_filter` + `minus_log`, no δ/β):
    H(kx, ky) = α / (λ·z·(kx² + ky²)/(4π) + α)     # DC = 1
    output    = −log(F⁻¹[H · F(I)])                  # line-integrated μ

Because the intensity is real f32, pass 1 uses rfft along x (real →
complex, spectrum has conj-symmetry along x so only n//2+1 unique
bins).  Pass 3 uses irfft along x (complex → real directly, no `.real`
dance).  The persistent host fde buffer is (ntheta, nz, n//2+1) c64 —
about half the memory of a full c64 x-spectrum.

Unlike Fresnel's kernel, Paganin's H is NOT separable (kx² + ky² lives
inside a 1/(…) rational), so we cannot factor it as K_x·K_y and apply
during the 1-D x-FFT.  Instead pass 2 rebuilds a (nz, this_chunk) H
strip per x-chunk on the GPU (cheap, few hundred kB), applies it after
the y-FFT, then IFFTs back along y.

No mirror padding.
"""
from __future__ import annotations

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np

from processing.pipeline import BandedPinned, pick_n_bands
from processing._scratch import ScratchMixin


# Bank fde (c64) and out (f32) along nz — same rationale + values as
# PropagationLarge.
FDE_N_BANDS = 4
OUT_N_BANDS = 8


class PaganinLarge(ScratchMixin):
    """Host-chunked single-distance Paganin phase retrieval (rfft path).

    Three passes over the (ntheta, nz, n//2+1) c64 host fde buffer:

      Pass 1 — for each CHUNK_NZ-tall strip of the real intensity:
                 H2D real f32 on GPU, rfft_x → c64 spectrum,
                 D2H to fde (half-plane).
      Pass 2 — for each CHUNK_N-wide column strip along the x-freq axis:
                 FFT_y, multiply by H(fx_strip, fy), IFFT_y, D2H to fde.
                 The x-freq axis has n//2+1 bins (not n), so the loop
                 stride is over n//2+1 and the last strip may be
                 shorter than CHUNK_N.
      Pass 3 — for each CHUNK_NZ-tall strip of fde:
                 irfft_x → real f32 of length n, clip, log, negate,
                 D2H to out.
    """

    def __init__(self, n, nz, wavelength, voxelsize, distance, alpha):
        self.n          = n
        self.nz         = nz
        self.wavelength = float(wavelength)
        self.voxelsize  = float(voxelsize)
        self.distance   = float(distance)
        self.alpha      = float(alpha)

        # 1-D reciprocal grids on cufft's native (fftfreq, rfftfreq) layout.
        self.fy = cp.fft.fftfreq (nz, d=voxelsize).astype("float32")   # (nz,)
        self.fx = cp.fft.rfftfreq(n,  d=voxelsize).astype("float32")   # (n//2+1,)
        # Precomputed squared 1-D grids used by _pass2_yfft_filter's
        # in-place H build — sliced per x-freq strip.
        self._fx2 = (self.fx * self.fx).astype("float32")              # (n//2+1,)
        self._fy2 = (self.fy * self.fy).astype("float32")              # (nz,)

        self._fde = None
        self._out = None
        self._H_scratch = None      # (nz, chunk_n_eff) f32, filter cache
        # Shared pinned + GPU scratch pools and StreamPipe cache from
        # ScratchMixin (see _scratch.py).  All three passes' pinned AND
        # GPU ping-pongs share the single 'in'/'out' pools; each pass's
        # StreamPipe is cached under 'p1'/'p2'/'p3'.
        self._init_scratch()

    def free(self):
        """Release cached pinned host + GPU buffers back to their pools.
        See TomoLargeReal.free for the rationale — call between iterations
        of a size sweep so previous large pinned allocations don't
        linger through the next allocation attempt."""
        self._fde = None
        self._out = None
        self._H_scratch = None
        self._free_scratch()

    def _get_fde(self, ntheta):
        # x-freq axis has n//2+1 bins (rfft half-plane) — half the memory
        # of a full-x c64 spectrum, same info thanks to conj-symmetry.
        shape = (ntheta, self.nz, self.n // 2 + 1)
        if self._fde is None or self._fde.shape != shape:
            n_bands = pick_n_bands(shape, np.complex64, band_axis=1,
                                   min_bands=FDE_N_BANDS)
            self._fde = BandedPinned(shape, np.complex64,
                                     n_bands=n_bands, band_axis=1)
        return self._fde

    def _get_out(self, ntheta):
        shape = (ntheta, self.nz, self.n)
        if self._out is None or self._out.shape != shape:
            n_bands = pick_n_bands(shape, np.float32, band_axis=1,
                                   min_bands=OUT_N_BANDS)
            self._out = BandedPinned(shape, np.float32,
                                     n_bands=n_bands, band_axis=1)
        return self._out

    def retrieve(self, intensity, chunks):
        """Paganin phase retrieval on a batch of real-f32 intensities.

        intensity : (ntheta, nz, n) or (nz, n) real float32 on HOST.
        chunks    : [CHUNK_NZ, CHUNK_N] — chunk_nz divides nz, chunk_n
                    divides n.  Both bound peak GPU memory.

        Returns   : host real float32 array with the same shape as input.
        """
        added_dim = (intensity.ndim == 2)
        if added_dim:
            intensity = intensity[np.newaxis]
        ntheta, nz_in, n_in = intensity.shape
        assert nz_in == self.nz and n_in == self.n, \
            f"intensity shape (…, {nz_in}, {n_in}) mismatches ({self.nz}, {self.n})"

        chunk_nz, chunk_n = chunks
        assert self.nz % chunk_nz == 0, \
            f"CHUNK_NZ={chunk_nz} must divide nz={self.nz}"
        assert self.n  % chunk_n  == 0, \
            f"CHUNK_N={chunk_n} must divide n={self.n}"

        fde = self._get_fde(ntheta)               # (nt, nz, n//2+1) c64 host
        out = self._get_out(ntheta)               # (nt, nz, n)      f32 host

        self._pass1_xfft(intensity, fde, chunk_nz)
        self._pass2_yfft_filter(fde, chunk_n)
        self._pass3_ifftx_log(fde, out, chunk_nz)

        return out[0] if added_dim else out

    def _pass1_xfft(self, intensity, fde, chunk_nz):
        """Pass 1 — H2D real f32 z-strip, rfft_x → c64 half-plane, D2H to fde."""
        n, nz  = self.n, self.nz
        ntheta = intensity.shape[0]
        nfx    = n // 2 + 1                       # rfft output width along x

        in_shape  = (ntheta, chunk_nz, n)
        out_shape = (ntheta, chunk_nz, nfx)
        pipe = self._get_pipe('p1', in_shape, out_shape,
                              np.float32, np.complex64,
                              use_gpu_scratch=True)

        _banded = hasattr(intensity, 'copy_to')

        def load(k, dst):
            z0 = k * chunk_nz
            if _banded:
                intensity.copy_to(dst, np.s_[:, z0:z0 + chunk_nz, :])
            else:
                dst[:] = intensity[:, z0:z0 + chunk_nz, :]

        def compute(k, in_gpu, out_gpu):
            # rfft consumes real → c64 half-plane, no need to widen the
            # input to c64 first (halves the compute vs the old fft path).
            out_gpu[...] = cp.fft.rfft(in_gpu, axis=-1)

        def store(k, src):
            z0 = k * chunk_nz
            fde.copy_from(src, np.s_[:, z0:z0 + chunk_nz, :])

        pipe.run(load, compute, store, nz // chunk_nz)

    def _pass2_yfft_filter(self, fde, chunk_n):
        """Pass 2 — H2D kx-strip of fde (from pass1), FFT_y, multiply by
        H(fx_strip, fy), IFFT_y (default norm ÷nz), D2H back to fde.

        Loops over the x-freq axis in strides of ``chunk_n_eff =
        min(chunk_n, n//2+1)`` — so a caller passing ``chunk_n = n``
        (the whole real width) collapses to a single strip covering all
        n//2+1 rfft bins.  The last strip may be shorter than
        ``chunk_n_eff`` (n//2+1 isn't necessarily a multiple of it);
        that ragged case is handled by clamping ``this_chunk`` and
        slicing the fixed-size ping-pong buffers per iteration.
        """
        n, nz  = self.n, self.nz
        nfx    = n // 2 + 1
        ntheta = fde.shape[0]
        # tomocupy filter: H = α / (λz·w²/(4π) + α), normalised DC=1
        # (analytical max at k=0 is 1/α, so /max ≡ *α).  fx/fy already in
        # cufft's native rfftfreq/fftfreq layout — index them directly.
        coef   = np.float32(self.wavelength * self.distance
                            / (4.0 * np.pi))
        alpha  = np.float32(self.alpha)

        # Clamp to the actual x-freq width so we never allocate wider
        # ping-pong buffers than the data can fill.
        chunk_n_eff = min(chunk_n, nfx)
        n_iter      = (nfx + chunk_n_eff - 1) // chunk_n_eff

        shape = (ntheta, nz, chunk_n_eff)
        pipe = self._get_pipe('p2', shape, shape,
                              np.complex64, np.complex64,
                              use_gpu_scratch=True)

        # Preallocate the (nz, chunk_n_eff) f32 H scratch once and
        # rebuild it in place per strip.  Doing `H = alpha / (coef*w2 +
        # alpha)` as a fresh expression allocates FOUR full-size
        # temporaries per iteration (12.9 GB at UPS=16, chunk=16384),
        # which is what tips the pool into OOM.  Cache + in-place ops
        # cut that to one (the H scratch itself).
        if (self._H_scratch is None
                or self._H_scratch.shape != (nz, chunk_n_eff)):
            self._H_scratch = cp.empty((nz, chunk_n_eff), dtype=cp.float32)

        def load(k, dst):
            x0 = k * chunk_n_eff
            this_chunk = min(chunk_n_eff, nfx - x0)
            fde.copy_to(dst[..., :this_chunk],
                        np.s_[:, :, x0:x0 + this_chunk])

        def compute(k, in_gpu, out_gpu):
            x0 = k * chunk_n_eff
            this_chunk = min(chunk_n_eff, nfx - x0)
            # In-place H build in cached scratch — one live (nz, this_chunk)
            # f32 buffer instead of four fresh ones.  fx²/fy² are
            # precomputed 1-D vectors on self.
            H = self._H_scratch[:, :this_chunk]
            cp.add(self._fx2[x0:x0 + this_chunk][None, :],
                   self._fy2[:, None], out=H)              # w² = fx² + fy²
            H *= coef                                        # coef · w²
            H += alpha                                       # coef · w² + α
            cp.reciprocal(H, out=H)                          # 1 / (…)
            H *= alpha                                       # α / (…)
            in_v = in_gpu[..., :this_chunk]
            arr  = cp.fft.fft(in_v, axis=1)                 # y-FFT
            arr *= H[None, :, :]                            # apply H
            arr  = cp.fft.ifft(arr, axis=1)                 # y-IFFT (÷ nz)
            out_gpu[..., :this_chunk] = arr

        def store(k, src):
            x0 = k * chunk_n_eff
            this_chunk = min(chunk_n_eff, nfx - x0)
            fde.copy_from(src[..., :this_chunk],
                          np.s_[:, :, x0:x0 + this_chunk])

        pipe.run(load, compute, store, n_iter)

    def _pass3_ifftx_log(self, fde, out, chunk_nz):
        """Pass 3 — H2D z-strip of fde (half-plane c64), irfft_x → real
        f32 of length n, clip, log, negate, D2H to out."""
        n, nz     = self.n, self.nz
        nfx       = n // 2 + 1
        ntheta    = fde.shape[0]

        in_shape  = (ntheta, chunk_nz, nfx)
        out_shape = (ntheta, chunk_nz, n)
        pipe = self._get_pipe('p3', in_shape, out_shape,
                              np.complex64, np.float32,
                              use_gpu_scratch=True)

        def load(k, dst):
            z0 = k * chunk_nz
            fde.copy_to(dst, np.s_[:, z0:z0 + chunk_nz, :])

        def compute(k, in_gpu, out_gpu):
            # irfft is already real — no `.real` dance.  Passing n=n
            # pins the output length (else irfft would infer 2*(nfx-1)
            # which happens to equal n only when n is even).
            arr = cp.fft.irfft(in_gpu, n=n, axis=-1)        # real f32
            cp.clip(arr, 1e-6, None, out=arr)
            cp.log(arr, out=arr)
            # tomocupy's minus_log convention: sample → positive attenuation.
            out_gpu[...] = -arr

        def store(k, src):
            z0 = k * chunk_nz
            out.copy_from(src, np.s_[:, z0:z0 + chunk_nz, :])

        pipe.run(load, compute, store, nz // chunk_nz)
