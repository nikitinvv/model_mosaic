"""PropagationLarge — host-chunked Fresnel propagation for volumes too big
for GPU-only Propagation.  Mirrors TomoLarge's staging pattern: the padded
(2nz × 2n) Fresnel work buffer lives on the HOST; only one strip at a time
is on the GPU, so peak GPU memory is proportional to the chunk sizes
rather than to (2nz × 2n).

Key trick: the Fresnel angular-spectrum kernel is separable,

    K(fx, fy) = exp(-i·π·λ·L·(fx²+fy²)) = K_x(fx) · K_y(fy)

so we never allocate the full 2D kernel — just two 1D arrays (a few
hundred kB each on the GPU) and apply them during the x-FFT and y-FFT
passes respectively.

Only the forward operator D is provided (step3_fresnel.py only uses D).
"""
from __future__ import annotations

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np

from processing.pipeline import StreamPipe, alloc_pinned, BandedPinned

# See tomo_large.FDE_N_BANDS — same driver-cap workaround.  fde in prop is
# (ntheta, nz, 2n) c64; nz is the biggest axis at UPS≥8 so we band on it.
FDE_N_BANDS = 4
# out / psi are half the size of fde but still cross the driver's per-alloc
# cap at UPS=64 (288 GB single alloc → observed cudaHostAlloc failure on
# tomo5).  Use 8 bands for extra headroom — per-band = 36 GB at UPS=64.
OUT_N_BANDS = 8


class PropagationLarge:
    """Fresnel forward propagator with host-staged x/y FFT chunking.

    Same math as Propagation.D, but the (ntheta, 2nz, 2n) work buffer
    stays on the host.  Three passes over that buffer:

      Pass 1 — for each CHUNK_NZ-tall strip of psi (the center rows):
                 pad x on GPU, FFT_x, multiply by K_x, back to host.
      Pass 2 — for each CHUNK_2N-wide column strip:
                 pad y on GPU, FFT_y, multiply by K_y, IFFT_y, unpad y,
                 back to host.
      Pass 3 — for each CHUNK_NZ-tall strip of the center rows:
                 IFFT_x on GPU, crop x, back to host output.

    Y-padding is done inside Pass 2 (per-strip on GPU) rather than as a
    full host-side mirror-fill: it halves the host RAM footprint (fde
    stays at (ntheta, nz, 2n) rather than (ntheta, 2nz, 2n)) at the
    cost of two extra unpadded rows worth of GPU work per x-strip.
    """

    def __init__(self, n, nz, wavelength, voxelsize, distance):
        self.n          = n
        self.nz         = nz
        self.wavelength = float(wavelength)
        self.voxelsize  = float(voxelsize)
        self.distance   = np.atleast_1d(np.asarray(distance, dtype=np.float64))
        ndist = int(self.distance.size)

        # Separable Fresnel kernels — one 1D array per axis per distance.
        # No 1/N factors: each pass pairs a default-norm FFT (unnormalized)
        # with a default-norm IFFT (divides by that axis's N), so the two
        # round-trips already reduce to identity.  Multiplying by K_x·K_y
        # in between then applies exactly the Fresnel filter with no extra
        # normalization needed.
        fx = cp.fft.fftfreq(2 * n,  d=voxelsize).astype("float32")
        fy = cp.fft.fftfreq(2 * nz, d=voxelsize).astype("float32")

        self.K_x = cp.empty((ndist, 2 * n),  dtype="complex64")
        self.K_y = cp.empty((ndist, 2 * nz), dtype="complex64")
        for j in range(ndist):
            L = float(self.distance[j])
            self.K_x[j] = cp.exp(-1j * cp.pi * self.wavelength * L
                                 * fx * fx).astype("complex64")
            self.K_y[j] = cp.exp(-1j * cp.pi * self.wavelength * L
                                 * fy * fy).astype("complex64")

        # Cached pinned host buffers (biggest allocations in D()).
        self._fde = None
        self._out = None
        self._psi = None                         # optional caller-side psi
        # Cached pipes — pinned+GPU ping-pong buffers survive across
        # D() calls, so we don't burn tens of GB of pinned pages every
        # invocation.  Rebuilt only when the driving shape changes.
        self._pipe1 = None
        self._pipe2 = None
        self._pipe3 = None
        # Shared pinned scratch pool for pipe1/2/3 (they run sequentially
        # — see TomoLarge for the rationale).  Peak pipe pinned RAM drops
        # from sum-of-passes to max-of-passes.
        self._scratch_in      = None
        self._scratch_out     = None
        self._scratch_in_cap  = 0
        self._scratch_out_cap = 0
        # Same idea on the GPU side (see TomoLarge for details).
        self._scratch_in_gpu  = None
        self._scratch_out_gpu = None
        self._scratch_in_gpu_cap  = 0
        self._scratch_out_gpu_cap = 0

    # --- padding helpers (GPU, symmetric mirror; matches pad_fwd_kernel) ---
    def _pad_x_mirror_gpu(self, strip_d):
        """(ntheta, k, n) → (ntheta, k, 2n) symmetric-mirror padded along x."""
        n = self.n
        half = n // 2
        ntheta, k, _ = strip_d.shape
        out = cp.empty((ntheta, k, 2 * n), dtype=strip_d.dtype)
        out[:, :, half : half + n]  = strip_d
        out[:, :,      : half]      = strip_d[:, :,  :half][:, :, ::-1]
        out[:, :, half + n :]       = strip_d[:, :, -half:][:, :, ::-1]
        return out

    def _pad_y_mirror_gpu(self, strip_d):
        """(ntheta, nz, m) → (ntheta, 2nz, m) symmetric-mirror padded along y."""
        nz = self.nz
        half = nz // 2
        ntheta, _, m = strip_d.shape
        out = cp.empty((ntheta, 2 * nz, m), dtype=strip_d.dtype)
        out[:, half : half + nz, :] = strip_d
        out[:,      : half,      :] = strip_d[:,  :half, :][:, ::-1, :]
        out[:, half + nz :,      :] = strip_d[:, -half:, :][:, ::-1, :]
        return out

    # ---------- explicit teardown -------------------------------------------
    def free(self):
        """Release cached pinned host + GPU buffers back to their pools.
        See TomoLarge.free for the rationale — call between iterations
        of a size sweep so previous multi-hundred-GB pinned allocations
        don't linger through the next allocation attempt."""
        self._fde = None
        self._out = None
        self._psi = None
        self._pipe1 = None
        self._pipe2 = None
        self._pipe3 = None
        self._scratch_in  = None
        self._scratch_out = None
        self._scratch_in_cap  = 0
        self._scratch_out_cap = 0
        self._scratch_in_gpu  = None
        self._scratch_out_gpu = None
        self._scratch_in_gpu_cap  = 0
        self._scratch_out_gpu_cap = 0

    # ---------- shared pinned scratch pool — see TomoLarge._scratch_views
    def _scratch_views(self, which, shape, dtype):
        dtp = np.dtype(dtype)
        need = int(np.prod(shape)) * dtp.itemsize
        if which == 'in':
            if self._scratch_in_cap < need:
                self._scratch_in = [alloc_pinned((need,), np.uint8)
                                    for _ in range(2)]
                self._scratch_in_cap = need
            bufs = self._scratch_in
        else:
            if self._scratch_out_cap < need:
                self._scratch_out = [alloc_pinned((need,), np.uint8)
                                     for _ in range(2)]
                self._scratch_out_cap = need
            bufs = self._scratch_out
        return [np.frombuffer(b, dtp, int(np.prod(shape))).reshape(shape)
                for b in bufs]

    # ---------- shared GPU scratch pool — see TomoLarge._scratch_gpu_views
    def _scratch_gpu_views(self, which, shape, dtype):
        dtp = np.dtype(dtype)
        n_elem = int(np.prod(shape))
        need = n_elem * dtp.itemsize
        n_c64 = (need + 7) // 8
        if which == 'in':
            if self._scratch_in_gpu_cap < need:
                for p in (self._pipe1, self._pipe2, self._pipe3):
                    if p is not None:
                        p.in_gpu = []
                self._scratch_in_gpu = None
                self._scratch_in_gpu = [cp.empty((n_c64,), dtype=cp.complex64)
                                        for _ in range(2)]
                self._scratch_in_gpu_cap = n_c64 * 8
            bufs = self._scratch_in_gpu
        else:
            if self._scratch_out_gpu_cap < need:
                for p in (self._pipe1, self._pipe2, self._pipe3):
                    if p is not None:
                        p.out_gpu = []
                self._scratch_out_gpu = None
                self._scratch_out_gpu = [cp.empty((n_c64,), dtype=cp.complex64)
                                         for _ in range(2)]
                self._scratch_out_gpu_cap = n_c64 * 8
            bufs = self._scratch_out_gpu
        return [b.view(dtype)[:n_elem].reshape(shape) for b in bufs]

    # ---------- pinned host-buffer cache ------------------------------------
    # Allocated PINNED so pipe load/store callbacks CPU-memcpy pinned→pinned
    # (fastest) and so step3's HDD-read pipeline can read psi chunks
    # directly into these buffers with zero extra copy.
    def _get_fde(self, ntheta):
        """(ntheta, nz, 2n) complex64 intermediate, banded along nz so
        each cudaHostAlloc stays under the driver's per-call cap.  See
        FDE_N_BANDS + tomo_large.py for rationale."""
        shape = (ntheta, self.nz, 2 * self.n)
        if self._fde is None or self._fde.shape != shape:
            self._fde = BandedPinned(shape, np.complex64,
                                     n_bands=FDE_N_BANDS, band_axis=1)
        return self._fde

    def _get_out(self, ntheta):
        """(ntheta, nz, n) c64 output buffer, banded along nz so no single
        cudaHostAlloc exceeds the driver's per-call cap at UPS=64 (see
        OUT_N_BANDS)."""
        shape = (ntheta, self.nz, self.n)
        if self._out is None or self._out.shape != shape:
            self._out = BandedPinned(shape, np.complex64,
                                     n_bands=OUT_N_BANDS, band_axis=1)
        return self._out

    def psi_buffer(self, ntheta):
        """Return a pinned (ntheta, nz, n) c64 buffer callers can fill in
        place (e.g. writing psi.real/psi.imag from proj slabs in
        step3_fresnel_large) before passing to :meth:`D`.  Cached across
        calls; contents are undefined on entry.

        Mirrors :meth:`TomoLargeReal.obj_buffer` — swaps the fresh
        ``np.empty`` in the caller for a pinned buffer, avoiding a large
        pageable alloc/free every iteration and enabling a faster
        pinned→GPU H2D inside D().

        Banded along nz (same OUT_N_BANDS as _out) so callers with very
        large nz don't trip the per-alloc cap.  Callers that write
        ``.real``/``.imag`` slices must iterate over `.bands` — see
        step3_fresnel_large for the pattern.
        """
        shape = (ntheta, self.nz, self.n)
        if self._psi is None or self._psi.shape != shape:
            self._psi = BandedPinned(shape, np.complex64,
                                     n_bands=OUT_N_BANDS, band_axis=1)
        return self._psi

    # --- forward Fresnel ---------------------------------------------------
    def D(self, psi, j, chunks):
        """Forward Fresnel propagation with the j-th distance kernel.

        psi     : (ntheta, nz, n) or (nz, n) complex64 on HOST (numpy).
        j       : distance index into K_x/K_y.
        chunks  : [CHUNK_NZ, CHUNK_2N] — CHUNK_NZ divides nz, CHUNK_2N
                  divides 2n.  Both bound peak GPU memory.

        Returns: host complex64 array with the same shape as psi.
        """
        added_dim = (psi.ndim == 2)
        if added_dim:
            psi = psi[np.newaxis]
        ntheta, nz_in, n_in = psi.shape
        assert nz_in == self.nz and n_in == self.n, \
            f"psi shape (…, {nz_in}, {n_in}) mismatches propagator ({self.nz}, {self.n})"

        chunk_nz, chunk_2n = chunks
        assert self.nz  % chunk_nz == 0, \
            f"CHUNK_NZ={chunk_nz} must divide nz={self.nz}"
        assert (2 * self.n) % chunk_2n == 0, \
            f"CHUNK_2N={chunk_2n} must divide 2n={2*self.n}"

        # Pinned host accumulators — allocated once, reused across calls.
        fde = self._get_fde(ntheta)                  # (nt, nz, 2n) c64
        out = self._get_out(ntheta)                  # (nt, nz, n)  c64

        # Pipes are cached across D() calls — see PropagationLarge.__init__
        # for the rationale.  No per-pass pool clearing so the cached
        # pipes' pinned/GPU buffers survive.
        self._pass1_xfft (psi, fde, chunk_nz, j)
        self._pass2_yfft (     fde,           chunk_2n, j)
        self._pass3_ifftx(     fde, out, chunk_nz)

        return out[0] if added_dim else out

    # ---------- individual passes -------------------------------------------
    def _pass1_xfft(self, psi, fde, chunk_nz, j):
        """Pass 1 — pipelined x-axis FFT strips.

        For each z-strip of psi: mirror-pad x from n to 2n on the GPU,
        FFT along x in-place, multiply by K_x, and D2H the result into
        the corresponding row-strip of the host `fde` buffer.
        """
        n, nz  = self.n, self.nz
        ntheta = psi.shape[0]
        K_x_d  = self.K_x[j]

        in_shape  = (ntheta, chunk_nz, n)
        out_shape = (ntheta, chunk_nz, 2 * n)
        pin_in  = self._scratch_views('in',  in_shape,  np.complex64)
        pin_out = self._scratch_views('out', out_shape, np.complex64)
        if (self._pipe1 is None
                or self._pipe1.in_shape  != in_shape
                or self._pipe1.out_shape != out_shape):
            self._pipe1 = StreamPipe(in_shape, out_shape,
                                     np.complex64, np.complex64,
                                     pinned_in=pin_in, pinned_out=pin_out)
        else:
            self._pipe1.in_pin  = pin_in
            self._pipe1.out_pin = pin_out
        pipe = self._pipe1

        # psi may be a plain ndarray (bench: user-owned) or a BandedPinned
        # (step3: filled via psi_buffer at large UPS).  copy_to handles both
        # — falls through to ndarray slicing when psi has no `.bands`.
        _psi_banded = hasattr(psi, 'copy_to')

        def load(k, dst):
            z0 = k * chunk_nz
            if _psi_banded:
                psi.copy_to(dst, np.s_[:, z0:z0 + chunk_nz, :])
            else:
                dst[:] = psi[:, z0:z0 + chunk_nz, :]

        def compute(k, in_gpu, out_gpu):
            padded = self._pad_x_mirror_gpu(in_gpu)
            cufft.fft(padded, axis=-1, overwrite_x=True)  # in-place on innermost
            padded *= K_x_d
            out_gpu[...] = padded

        def store(k, src):
            z0 = k * chunk_nz
            # BandedPinned handles the axis-1 write; typically chunk_nz
            # divides band_rows so this lands in one band.
            fde.copy_from(src, np.s_[:, z0:z0 + chunk_nz, :])

        pipe.run(load, compute, store, nz // chunk_nz)

    def _pass2_yfft(self, fde, chunk_2n, j):
        """Pass 2 — pipelined y-axis FFT/K_y/IFFT_y per x-strip.

        For each x-strip of fde: mirror-pad y from nz to 2nz on the GPU,
        FFT_y, multiply by K_y, IFFT_y (default norm ÷2nz), then crop y
        back to nz center rows and write into fde[:, :, x_strip].
        """
        n, nz = self.n, self.nz
        ntheta = fde.shape[0]
        K_y_d = self.K_y[j]

        shape = (ntheta, nz, chunk_2n)
        pin_in  = self._scratch_views('in',  shape, np.complex64)
        pin_out = self._scratch_views('out', shape, np.complex64)
        if (self._pipe2 is None
                or self._pipe2.in_shape != shape
                or self._pipe2.out_shape != shape):
            self._pipe2 = StreamPipe(shape, shape, np.complex64, np.complex64,
                                     pinned_in=pin_in, pinned_out=pin_out)
        else:
            self._pipe2.in_pin  = pin_in
            self._pipe2.out_pin = pin_out
        pipe = self._pipe2

        def load(k, dst):
            x0 = k * chunk_2n
            # Full-nz stripe crosses all fde bands — copy_to iterates them
            # into the contiguous pipe scratch with no host-side temp.
            fde.copy_to(dst, np.s_[:, :, x0:x0 + chunk_2n])

        def compute(k, in_gpu, out_gpu):
            # y-pad + FFT_y + K_y multiply + IFFT_y — cufft's overwrite_x
            # is unreliable here (silently returns garbage for the FFT/IFFT
            # pair around a scalar-broadcast multiply on axis=1), so keep
            # cp.fft.fft/ifft (return-new).
            padded = self._pad_y_mirror_gpu(in_gpu)
            padded = cp.fft.fft(padded, axis=1)
            padded *= K_y_d[:, None]
            padded = cp.fft.ifft(padded, axis=1)   # default norm ÷2nz
            out_gpu[...] = padded[:, nz // 2 : nz // 2 + nz, :]

        def store(k, src):
            x0 = k * chunk_2n
            fde.copy_from(src, np.s_[:, :, x0:x0 + chunk_2n])

        pipe.run(load, compute, store, 2 * n // chunk_2n)

    def _pass3_ifftx(self, fde, out, chunk_nz):
        """Pass 3 — pipelined IFFT along x + crop back to n.

        For each z-strip of fde: IFFT along x in-place (innermost axis
        → safe for overwrite_x), then crop the center-n columns and
        D2H into the output buffer.
        """
        n, nz  = self.n, self.nz
        ntheta = fde.shape[0]

        in_shape  = (ntheta, chunk_nz, 2 * n)
        out_shape = (ntheta, chunk_nz, n)
        pin_in  = self._scratch_views('in',  in_shape,  np.complex64)
        pin_out = self._scratch_views('out', out_shape, np.complex64)
        if (self._pipe3 is None
                or self._pipe3.in_shape  != in_shape
                or self._pipe3.out_shape != out_shape):
            self._pipe3 = StreamPipe(in_shape, out_shape,
                                     np.complex64, np.complex64,
                                     pinned_in=pin_in, pinned_out=pin_out)
        else:
            self._pipe3.in_pin  = pin_in
            self._pipe3.out_pin = pin_out
        pipe = self._pipe3

        def load(k, dst):
            z0 = k * chunk_nz
            fde.copy_to(dst, np.s_[:, z0:z0 + chunk_nz, :])

        def compute(k, in_gpu, out_gpu):
            cufft.ifft(in_gpu, axis=-1, overwrite_x=True)  # in-place
            out_gpu[...] = in_gpu[:, :, n // 2 : n // 2 + n]

        def store(k, src):
            z0 = k * chunk_nz
            # out is BandedPinned along nz — copy_from dispatches the
            # axis-1 write to the intersecting band(s).
            out.copy_from(src, np.s_[:, z0:z0 + chunk_nz, :])

        pipe.run(load, compute, store, nz // chunk_nz)
