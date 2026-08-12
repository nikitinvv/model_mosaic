"""TomoLargeReal — host-chunked USFFT Radon for volumes too big for the
GPU-only TomoReal.

Stages small pieces of the padded (2N × 2N) frequency-domain buffer
through the GPU while keeping the big `fde` and `sino` arrays on the
HOST.  Peak GPU memory is proportional to the chunk sizes rather than
to (2N)².

Forward R uses an rfft along x throughout (halved fde host RAM, halved
x-FFT cost); adjoint RT keeps a pragmatic full-complex path (rfft-aware
adjoint scatter is deferred).  RT's own persistent fde buffer is
`(nz, 2n, 2n) c64` pinned banded — allocated lazily on first RT() call
so R-only workloads pay nothing for it.

At UPS ≥ 16 the RT adjoint scatter would OOM on the GPU if a single
(nz, 2n, 2n) c64 slice had to live there; `_passRT2_scatter` chunks
the destination fde along the ky axis (`gather_kernel_ychunk`) so peak
GPU is `nz * chunk_xy * 2n * 8` bytes per launch.
"""
from __future__ import annotations

import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft

from processing.kernels import (
    gather_kernel, gather_kernel_ychunk, gather_kernel_rfft,
    scatter_compact_kernel, gather_compact_kernel,
)
from processing.pipeline import (
    ComputeD2HPipe, alloc_pinned, BandedPinned, pick_n_bands,
)
from processing._scratch import ScratchMixin

# Number of pinned bands `_get_fde` splits its (nz, 2n, ...) buffer into
# along the 2n axis.  cudaHostAlloc has a per-request cap (~64 GiB on some
# boxes, more on others); fde grows as N² so at large UPS a single alloc
# hits that ceiling.  4 bands keeps each cudaHostAlloc call well under
# any observed cap while adding only a per-load band-stitch loop.
FDE_N_BANDS = 4


# =============================================================================
# TomoLargeReal — host-chunked USFFT Radon.
# =============================================================================
class TomoLargeReal(ScratchMixin):
    """Radon transform via USFFT with host-staged chunking.

    The object is REAL (float32) and the forward x-axis FFT uses rfft —
    cutting the (nz, 2n, 2n) complex64 ``fde`` buffer down to
    (nz, 2n, n+1) (half the host RAM, half the x-axis FFT cost).  Along y
    we still do a full complex FFT of length 2n.  The gather kernel
    exploits ``X[fx, fy] = conj(X[-fx, -fy])`` (real-input symmetry) to
    reach into the missing negative-fx half without ever storing it.

    Layout choices (both fftshift tricks dropped, RAW fftfreq order):
      * fde stored in fftfreq order along BOTH x (rfft, so [0, n]) and
        y (full fft, so [0, 2n) with wrap).  No c2dfftshift needed.
      * sino stored in centered (fftshift-along-r) order for backward
        compatibility with downstream step3 / analysis code.  Pass 4's
        1-D IFFT along r therefore still uses c1dfftshift.

    Output sino: **REAL float32**, shape ``(ntheta, nz, n)`` — Pass 4
    takes ``.real`` after the r-axis IFFT (imag ≈ 0 for real obj input).

    Adjoint RT: full complex64 path (uses `_fde_rt` at (nz, 2n, 2n) c64
    pinned banded, distinct from R's (nz, 2n, n+1) `_fde`).  See the
    module docstring — this is a pragmatic port; an rfft-aware adjoint
    scatter kernel is deferred.
    """

    def __init__(self, n, theta, chunk_xy):
        eps = 1e-3
        mu  = -np.log(eps) / (2 * n * n)
        m   = int(np.ceil(2 * n / np.pi *
                          np.sqrt(-mu * np.log(eps) + (mu * n) ** 2 / 4)))
        ntheta = len(theta)

        # phi: real Gaussian pre-multiplication.  Separable — store 1-D
        # factor + ±1 scalar.  Saves n² c64 at UPS≥8.
        t = np.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        phi1d     = np.exp(mu * (n * n) * (t * t)).astype("float32")
        phi_scale = np.float32(1 - n % 4)

        # c1dfftshift kept — needed for the centered sino layout in Pass 4.
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")

        # c2dfftshift 1-D vector (outer product) — needed by the RT path's
        # full-complex y-IFFT / x-IFFT passes.  Stored as (2n,) i8; the
        # forward passes do not need it (raw fftfreq order).
        c2dfftshift1d = (1 - 2 * (np.arange(1, 2 * n + 1) % 2)).astype("int8")

        # Trig tables — the gather kernel recomputes (x, y) from these
        # plus a per-sample flat index.
        theta32   = theta.astype("float32")
        cos_theta = np.cos(theta32).astype("float32")
        sin_theta = np.sin(theta32).astype("float32")

        self.cos_theta   = cos_theta
        self.sin_theta   = sin_theta
        self.n           = n
        self.ntheta      = ntheta
        self.theta       = theta32
        self.mua         = cp.array([mu], dtype="float32")
        self.m           = m
        self.phi1d       = phi1d                 # float32 (n,)
        self.phi_scale   = phi_scale
        self.c1dfftshift = c1dfftshift
        self.c2dfftshift1d = c2dfftshift1d

        # Cached pinned host buffers.  sino_real is a VIEW into the first
        # half of sino's bytes (float32 vs complex64), so there's no
        # separate cache — it's derived per R() call in _pass4_ifft.
        self._fde       = None                   # (nz, 2n, n+1) complex64 — R
        self._fde_rt    = None                   # (nz, 2n, 2n) complex64 — RT
        self._sino      = None                   # (ntheta, nz, n) complex64
        self._obj       = None                   # (nz, n, n) float32 — pinned
        # See the pass3 gather pipe rationale (avoid re-pinning across calls).
        self._pipe_gather = None
        # Reused (nz, chunk_n, 2n) float32 scratch for _pass1_xfft's
        # center-placed pad-then-rfft input; reshaped/rezeroed on demand.
        self._pad_scratch = None
        # GPU trig tables uploaded once per R() call.
        self._cos_theta_gpu = None
        self._sin_theta_gpu = None
        # Cache the (nel, idx) result of _sort_into_chunks — deterministic
        # given chunk_xy, so re-doing it every R() call wastes a lot of
        # pageable-host alloc/free churn.
        self._sort_key   = None
        self._sort_cache = None
        # Cache the per-ky-band sample-index lists for _passRT2_scatter's
        # compact scatter kernel — same lazy-per-chunk_xy pattern.
        self._pb_key   = None
        self._pb_cache = None    # list[np.ndarray[int64]], length n_bands
        # Chunk sizes for the currently-executing RT() call — stashed on
        # self so the four passes can read them without threading extra
        # args through.  See RT() and _passRT2_scatter.
        self._chunks_rt = None
        # Shared scratch pools + StreamPipe cache from ScratchMixin.
        self._init_scratch()

        # Precompute both R and RT index tables now.  chunk_xy is fixed
        # for the lifetime of the instance — step2 / step7 / benches all
        # know it at construction, so the first R()/RT() call sees the
        # cached result rather than paying the precompute inline.
        self._sort_into_chunks(chunk_xy)
        self._per_band_precompute(chunk_xy)

    # ---------- bin sort for the gather -----------------------------------
    def _sort_into_chunks(self, chunk_xy):
        """Bin samples by rk_x only (the POST-reflection column in the
        stored [0, n+1) half-spectrum).  qid built theta-by-theta from the
        trig tables + r — no ntheta·n coordinate table ever materialised
        on the host.  qid stored as int32; max rk_x = n < 2^31.  Result
        cached per chunk_xy across R() calls.
        """
        if self._sort_key == chunk_xy:
            return self._sort_cache

        n     = self.n
        twon  = 2 * n
        nx_bins = (n // chunk_xy) + 1     # +1 for the tail bin holding kx == n
        r     = (np.arange(n, dtype="float32") - n / 2) / n
        qid   = np.empty(self.ntheta * n, dtype="int32")
        for k in range(self.ntheta):
            x_k = np.clip(self.cos_theta[k] * r, -0.5, 0.5 - 1e-5)
            k0  = ((np.floor(twon * x_k).astype("int32")) + twon) % twon
            rk_x = np.where(k0 > n, twon - k0, k0)                 # [0, n]
            qid[k * n:(k + 1) * n] = (rk_x // chunk_xy).clip(max=nx_bins - 1)

        idx = np.argsort(qid, kind="stable")
        qid_s = qid[idx]

        nel = np.zeros(nx_bins, dtype="int64")
        if len(qid_s):
            change_points = np.flatnonzero(np.diff(qid_s, prepend=qid_s[0] - 1))
            run_lengths = np.diff(np.append(change_points, len(qid_s)))
            nel[qid_s[change_points]] = run_lengths
        del qid, qid_s   # release the ~2·(ntheta·n·4)-byte scratch

        self._sort_key   = chunk_xy
        self._sort_cache = (nel, idx, nx_bins)
        return self._sort_cache

    def _per_band_precompute(self, chunk_xy, chunk_theta_gpu=4096):
        """Precompute per-ky-band (theta_i, x_i) int32 pairs for
        `scatter_compact_kernel`.  Streamed GPU compute in theta-slabs
        of `chunk_theta_gpu` angles; per band D2H'd and decoded once.
        No persistent (ntheta, n) centers array — only the per-band
        (theta_i, x_i) tuples (~3.6 GB at UPS=8, scales with total
        sample count).

        Wrap-around spillover at the top (b=0) and bottom (b=n_bands-1)
        bands is folded in so a sample with centre near ky=0 (index 0
        or 2n-1) appears in both edge bands.
        """
        n, ntheta = self.n, self.ntheta
        m         = self.m
        twon      = 2 * n
        n_bands   = (twon + chunk_xy - 1) // chunk_xy
        cx        = np.float32(n * 0.5)
        x_d       = cp.arange(n, dtype=cp.float32) - cx
        sin_np    = np.sin(self.theta).astype(np.float32)

        parts = [[] for _ in range(n_bands)]
        for t0 in range(0, ntheta, chunk_theta_gpu):
            t1     = min(t0 + chunk_theta_gpu, ntheta)
            sin_d  = cp.asarray(sin_np[t0:t1])
            rk_d   = cp.rint(-2.0 * x_d[None, :] * sin_d[:, None]).astype(cp.int32)
            centers_d = (((n + rk_d) % twon).astype(cp.int32)).ravel()
            for b in range(n_bands):
                y_lo   = b * chunk_xy
                y_hi   = min(y_lo + chunk_xy, twon)
                mask_d = (centers_d + m >= y_lo) & (centers_d - m < y_hi)
                if y_lo == 0 and m > 0:
                    mask_d |= (centers_d >= twon - m)
                if y_hi == twon and m > 0:
                    mask_d |= (centers_d < m)
                local = cp.asnumpy(cp.where(mask_d)[0])            # int64
                if local.size:
                    parts[b].append(local + (t0 * n))
            del sin_d, rk_d, centers_d

        # Store as list of (theta_i, x_i) int32 tuples — kernel takes
        # these directly, no per-RT-call decode needed.  Within each
        # band, sort samples by rk_y (their fde ky-index) so adjacent
        # threads scatter into adjacent fde rows: better L2 hit rate
        # and less atomicAdd contention per warp.
        theta_np = self.theta                            # (ntheta,) f32
        sin_np   = np.sin(theta_np).astype(np.float32)
        per_band = []
        for pb in parts:
            if not pb:
                per_band.append((np.zeros(0, dtype=np.int32),
                                 np.zeros(0, dtype=np.int32)))
                continue
            flat    = np.concatenate(pb)
            theta_i = (flat // n).astype(np.int32)
            x_i     = (flat %  n).astype(np.int32)
            # rk_y = round(-2·sin(θ[i])·(x_i − n/2)); sort ascending on GPU.
            ti_d = cp.asarray(theta_i)
            xi_d = cp.asarray(x_i)
            sd   = cp.asarray(sin_np)[ti_d]
            rk_y = cp.rint(-2.0 * sd * (xi_d.astype(cp.float32) - cx)).astype(cp.int32)
            order = cp.argsort(rk_y, kind="stable")
            per_band.append((cp.asnumpy(ti_d[order]),
                             cp.asnumpy(xi_d[order])))
            del ti_d, xi_d, sd, rk_y, order
        self._pb_cache = per_band
        return per_band

    def _get_st_end(self, indx, chunk_xy):
        """Halo-extended x-column range in the stored [0, n+1) half-spectrum."""
        n, m = self.n, self.m
        stx = max(0, indx * chunk_xy - m)
        endx = min(n + 1, (indx + 1) * chunk_xy + m + 1)
        return stx, endx

    # ---------- explicit teardown -------------------------------------------
    def free(self):
        """Release cached pinned host + GPU buffers back to their pools.
        Call this between iterations of a size sweep (or before switching
        to a very different size) so the previous instance's fde/sino
        (multi-hundred-GB pinned) don't linger through the next
        allocation attempt."""
        self._fde         = None
        self._fde_rt      = None
        self._sino        = None
        self._obj         = None
        self._obj_rt      = None
        self._pipe_gather = None
        self._pad_scratch = None
        self._sort_key    = None
        self._sort_cache  = None
        self._pb_key      = None
        self._pb_cache    = None
        self._free_scratch()

    # ---------- pinned host-buffer cache ------------------------------------
    def _get_fde(self, nz):
        """Return the pinned (nz, 2n, n+1) c64 fde buffer (for R), split
        into FDE_N_BANDS chunks along the 2n (band_axis=1) axis so no
        single cudaHostAlloc exceeds the driver's per-call cap at large UPS.
        """
        shape = (nz, 2 * self.n, self.n + 1)
        if self._fde is None or self._fde.shape != shape:
            n_bands = pick_n_bands(shape, np.complex64, band_axis=1,
                                   min_bands=FDE_N_BANDS)
            self._fde = BandedPinned(shape, np.complex64,
                                     n_bands=n_bands, band_axis=1)
        self._fde.fill(0)
        return self._fde

    def _get_fde_rt(self, nz):
        """Return the pinned (nz, 2n, 2n) c64 fde buffer (for RT).
        Separate from `_fde` because RT uses the full-complex layout —
        double the columns (2n instead of n+1) — so the two buffers cannot
        alias.  Banded on the 2n (band_axis=1) axis; see FDE_N_BANDS."""
        n = self.n
        shape = (nz, 2 * n, 2 * n)
        if self._fde_rt is None or self._fde_rt.shape != shape:
            n_bands = pick_n_bands(shape, np.complex64, band_axis=1,
                                   min_bands=FDE_N_BANDS)
            self._fde_rt = BandedPinned(shape, np.complex64,
                                        n_bands=n_bands, band_axis=1)
        # Adjoint gather atomicAdds into fde — must start at 0.
        self._fde_rt.fill(0)
        return self._fde_rt

    def _get_sino(self, nz):
        shape = (self.ntheta, nz, self.n)
        if self._sino is None or self._sino.shape != shape:
            self._sino = alloc_pinned(shape, np.complex64)
        return self._sino

    def _get_obj(self, nz, dtype=np.complex64):
        """(nz, n, n) buffer for RT output — cast to the caller's input
        dtype (complex64 for complex sinos, or float32 for real sinos —
        we return .real for float32 input to mirror TomoReal.RT)."""
        shape = (nz, self.n, self.n)
        if (getattr(self, '_obj_rt', None) is None
                or self._obj_rt.shape != shape
                or self._obj_rt.dtype != np.dtype(dtype)):
            self._obj_rt = alloc_pinned(shape, dtype)
        return self._obj_rt

    def obj_buffer(self, nz):
        """Return a pinned (nz, n, n) float32 buffer callers can fill in
        place (e.g. via ``h5py.Dataset.read_direct``) before passing to
        :meth:`R`.  Cached across calls; contents are undefined on entry.

        Using this buffer instead of a fresh ``np.empty`` avoids a
        (nz·n²·4)-byte pageable allocation and lets subsequent H2Ds skip
        the cupy pinned-bounce.  At UPS=32, nz=1 that's 36 GB pageable
        turned into 36 GB pinned — same total RAM, better throughput and
        no double-count vs the estimator's "pinned" bucket.
        """
        shape = (nz, self.n, self.n)
        if self._obj is None or self._obj.shape != shape:
            self._obj = alloc_pinned(shape, np.float32)
        return self._obj

    # ---------- forward Radon ------------------------------------------------
    def R(self, obj, chunks):
        """(nz, n, n) REAL float32 obj  →  (ntheta, nz, n) real float32 sino.

        chunks = [CHUNK_N, CHUNK_THETA, CHUNK_XY] — chunk sizes for the
        1-D FFTs, angle grouping, and gather bin size respectively.
        """
        chunk_n, chunk_theta, chunk_xy = chunks
        nz = obj.shape[0]

        fde  = self._get_fde(nz)
        sino = self._get_sino(nz)

        nel, idx, nx_bins = self._sort_cache        # precomputed in __init__

        self._cos_theta_gpu = cp.asarray(self.cos_theta)
        self._sin_theta_gpu = cp.asarray(self.sin_theta)

        # Pipes and per-bin GPU buffers are cached across R() calls to
        # avoid re-pinning tens of GB of pipe buffers on every call.
        self._pass1_xfft   (obj, fde, chunk_n)
        self._pass2_yfft   (     fde, chunk_n)
        self._pass3_gather (     fde, sino, nel, idx, nx_bins, chunk_xy)
        real_sino = self._pass4_ifft(sino, chunk_theta)

        return real_sino

    # ---------- individual passes -------------------------------------------
    def _pass1_xfft(self, obj, fde, chunk_n):
        """Pass 1 — pipelined pad + rfft along x.

        For each chunk k: multiply obj strip by phi (real), zero-pad x
        from n to 2n, rfft along x → (nz, chunk_n, n+1) complex64,
        write into fde[:, k·chunk_n:(k+1)·chunk_n, :].  No c2dfftshift
        (raw fftfreq order along x).
        """
        n, nz = self.n, obj.shape[0]

        in_shape  = (nz, chunk_n, n)
        out_shape = (nz, chunk_n, n + 1)
        pipe = self._get_pipe('p1', in_shape, out_shape,
                              np.float32, np.complex64)

        phi_scale = self.phi_scale
        phi1d_gpu = cp.asarray(self.phi1d)                    # (n,) f32

        def load(k, dst):
            st = k * chunk_n
            dst[:] = obj[:, st:st + chunk_n, :]

        def compute(k, in_gpu, out_gpu):
            st, end = k * chunk_n, (k + 1) * chunk_n
            phix = phi1d_gpu[st:end]                          # (chunk_n,) f32
            # Center-place phi*obj in the padded 2n buffer so the resulting
            # rfft matches the FFT of a centered signal (X_c[k]).
            # Left-aligning at [0, n) would give a spectrum that differs
            # by a (-i)^k phase per sample.
            need_shape = (nz, chunk_n, 2 * self.n)
            if (self._pad_scratch is None
                    or self._pad_scratch.shape != need_shape):
                self._pad_scratch = cp.zeros(need_shape, dtype=cp.float32)
            else:
                self._pad_scratch.fill(0)
            padded = self._pad_scratch
            # phi[i, j] = phi_scale · phi1d[i] · phi1d[j] — separable factor.
            padded[:, :, n // 2 : n // 2 + n] = (
                phi_scale * phix[None, :, None] * phi1d_gpu[None, None, :] * in_gpu
            )
            out_gpu[...] = cufft.rfft(padded, axis=-1)

        def store(k, src):
            st = k * chunk_n
            # Center-place along Y (offset by n//2) so the obj sits at
            # fde[:, n//2:3n//2, :] — same convention as TomoReal's
            # padded buffer.
            fde.copy_from(src, np.s_[:, n // 2 + st : n // 2 + st + chunk_n, :])

        pipe.run(load, compute, store, n // chunk_n)

    def _pass2_yfft(self, fde, chunk_n):
        """Pass 2 — pipelined complex FFT along y (chunked along x).

        The x axis stores only n+1 columns (rfft half spectrum); we chunk
        it into strips of `chunk_n` columns.  Full complex FFT of length
        2n along y, no c2dfftshift (raw fftfreq order).
        """
        n, nz = self.n, fde.shape[0]
        assert n % chunk_n == 0, \
            f"CHUNK_N={chunk_n} must divide n={n} (rfft x-axis strips)"
        n_full = n // chunk_n   # strips covering [0, n)

        shape = (nz, 2 * n, chunk_n)
        pipe = self._get_pipe('p2', shape, shape,
                              np.complex64, np.complex64)

        def load(k, dst):
            st = k * chunk_n
            fde.copy_to(dst, np.s_[:, :, st:st + chunk_n])

        def compute(k, in_gpu, out_gpu):
            # cp.fft.fft (return-new) — cufft's overwrite_x is unreliable
            # for non-innermost axes.
            out_gpu[...] = cp.fft.fft(in_gpu, axis=1)

        def store(k, src):
            st = k * chunk_n
            fde.copy_from(src, np.s_[:, :, st:st + chunk_n])

        pipe.run(load, compute, store, n_full)

        # Tail: the single last column at kx = n.
        tail_col_h = fde[:, :, n : n + 1]           # multi-band → stitched ndarray
        tail_col_d = cp.asarray(tail_col_h)
        tail_col_d = cp.fft.fft(tail_col_d, axis=1)
        fde[:, :, n : n + 1] = tail_col_d.get()

    def _pass3_gather(self, fde, sino, nel, idx, nx_bins, chunk_xy):
        """Pass 3 — NUFFT gather via `gather_compact_kernel`.

        Bin samples by rk_x (their column in the stored [0, n+1) half-
        spectrum).  For each bin, fetch a FULL-y x-strip
        fde[:, :, stx:endx] — a plain contiguous slice, no synthesis —
        and let the kernel do reflection with conj; because the strip
        covers all 2n rows, the reflected access rk_y = (2n - k1) % 2n
        is always in-range.  Per-bin (theta_idx, x_idx) int32 pairs are
        decoded once on GPU from the sorted flat idx; the compact
        kernel reads them per thread instead of doing an int64 divmod.
        """
        n = self.n
        nz = fde.shape[0]
        m, mua = self.m, self.mua
        cos_theta_gpu = self._cos_theta_gpu
        sin_theta_gpu = self._sin_theta_gpu

        max_nel = int(max((int(v) for v in nel), default=0))
        if max_nel == 0:
            return
        if (self._pipe_gather is None
                or self._pipe_gather.out_shape != (max_nel,)):
            self._pipe_gather = ComputeD2HPipe((max_nel,), np.complex64)
        gather_pipe = self._pipe_gather

        sino_flat = sino.reshape(-1)
        offset = 0

        for indx in range(nx_bins):
            nel_i = int(nel[indx])
            if nel_i == 0:
                continue
            stx, endx = self._get_st_end(indx, chunk_xy)
            if endx <= stx:       # empty x range (past kx = n)
                offset += nel_i
                continue

            # Upload on the compute stream so the kernel reads after the
            # H2D completes.  fde is banded pinned — allocate the GPU
            # patch buffer first, then copy_to_gpu iterates each band's
            # slice directly into the corresponding rows (no host-side
            # stitched temp).
            full_idx  = idx[offset : offset + nel_i]         # int64, host
            flat_base = (full_idx // n) * nz * n + (full_idx % n)
            with gather_pipe.s_comp:
                fde_d = cp.empty((nz, 2 * n, endx - stx), dtype=cp.complex64)
                fde.copy_to_gpu(fde_d, np.s_[:, :, stx:endx])
                # Compact kernel takes pre-decoded (theta_idx, x_idx)
                # int32 pairs so it can skip the flat-idx divmod that
                # gather_kernel_rfft did per thread.
                fidx_d  = cp.asarray(full_idx)               # int64
                theta_i = (fidx_d // n).astype(cp.int32)
                x_i     = (fidx_d %  n).astype(cp.int32)
                del fidx_d

            grid, block = (int(np.ceil(nel_i / 1024)),), (1024,)

            def compute(zc, out_gpu,
                        _nel=nel_i, _fde=fde_d, _ti=theta_i, _xi=x_i,
                        _stx=stx, _endx=endx,
                        _grid=grid, _block=block):
                out_gpu[:_nel].fill(0)
                gather_compact_kernel(_grid, _block,
                    (out_gpu[:_nel], _fde[zc],
                     _xi, _ti, cos_theta_gpu, sin_theta_gpu,
                     m, mua, _nel, _stx, _endx, n))

            def store(zc, src_pinned, _nel=nel_i, _base=flat_base):
                sino_flat[_base + zc * n] = src_pinned[:_nel]

            gather_pipe.run(compute, store, nz)
            offset += nel_i

    def _pass4_ifft(self, sino, chunk_theta):
        """Pass 4 — pipelined r-axis IFFT + normalisation, then .real.

        c1dfftshift-based centred-spectrum → centred-sample IFFT; the
        output is copied out as **REAL float32** (imag part ≈ 0 for real
        obj input, and is discarded).
        """
        n, ntheta = self.n, self.ntheta
        nz = sino.shape[1]
        c1d_gpu = cp.asarray(self.c1dfftshift)
        scale = np.float32(1.0 / (4 * n * np.sqrt(n * ntheta)))

        # sino_real shares the FIRST HALF of sino's pinned bytes — a f32
        # buffer takes half the byte-space of the same-shape c64 one, and
        # Pass 4's load-ahead read of sino[k+few] and store-behind write
        # of sino_real[k-3] land in disjoint byte ranges for every k (the
        # read is at 8·offset, the write at 4·offset — always separated).
        # Saves ntheta·nz·n·4 bytes of pinned RAM (~52 GB at UPS=32).
        # Caveat: the returned sino_real remains valid only until the next
        # R() call, which will overwrite sino (and therefore sino_real).
        # Callers that need to hold multiple sinos should `.copy()`.
        flat = sino.view(np.float32).reshape(-1)      # 2·ntheta·nz·n f32
        sino_real = flat[: ntheta * nz * n].reshape(ntheta, nz, n)

        in_shape  = (chunk_theta, nz, n)
        out_shape = (chunk_theta, nz, n)
        pipe = self._get_pipe('p4', in_shape, out_shape,
                              np.complex64, np.float32)

        def load(k, dst):
            st = k * chunk_theta
            dst[:] = sino[st:st + chunk_theta]

        def compute(k, in_gpu, out_gpu):
            # In-place ifft along innermost axis is safe.
            in_gpu *= c1d_gpu
            cufft.ifft(in_gpu, axis=-1, overwrite_x=True)
            in_gpu *= c1d_gpu
            in_gpu *= scale
            out_gpu[...] = in_gpu.real

        def store(k, src):
            st = k * chunk_theta
            sino_real[st:st + chunk_theta] = src

        pipe.run(load, compute, store, ntheta // chunk_theta)
        return sino_real

    # ---------- adjoint Radon (backprojection) ------------------------------
    def RT(self, sino, chunks):
        """(ntheta, nz, n) sino → (nz, n, n) obj — adjoint of R().

        chunks = [CHUNK_N, CHUNK_THETA, CHUNK_XY] — same knobs as R().
        `chunk_xy` here controls the ky-band chunking of the adjoint
        scatter (see `_passRT2_scatter`); pass `2*n` for a single
        launch (matches the historical per-slice behaviour at small UPS).

        Four passes in reverse of R (each internally structured like
        the corresponding forward pass but with reversed I/O and the
        FFT/gather direction flipped):

          passRT1 — 1-D FFT along the r axis (adjoint of pass4's IFFT).
          passRT2 — adjoint NUFFT scatter sino → fde (gather dir=1),
                    chunked along ky at `chunk_xy` rows per launch to
                    bound GPU peak memory.
          passRT3 — y-IFFT strips of fde (adjoint of pass2's y-FFT).
          passRT4 — x-IFFT strips of fde, crop center-n, multiply φ,
                    store into obj (adjoint of pass1).
        """
        chunk_n, chunk_theta, chunk_xy = chunks
        ntheta = sino.shape[0]
        nz     = sino.shape[1]

        out_dtype = np.float32 if sino.dtype == np.float32 else np.complex64

        # RT keeps its own (nz, 2n, 2n) c64 fde buffer, distinct from R's
        # (nz, 2n, n+1) buffer — see `_get_fde_rt`.
        fde = self._get_fde_rt(nz)                # (nz, 2n, 2n) c64 pinned, zeroed
        obj = self._get_obj(nz, dtype=out_dtype)

        # Stash chunks so the passes can read them without extra plumbing.
        self._chunks_rt = list(chunks)

        # Trig tables + per-bin sort not needed here (RT uses gather_kernel
        # not gather_kernel_rfft), but the theta table is uploaded once per
        # RT() call for the ychunk scatter kernel.
        # PassRT1 promotes real sino → complex in-place inside sino_c (own
        # buffer, since sino input may be real f32).  Allocated lazily.
        sino_c = self._get_sino(nz)               # (ntheta, nz, n) c64 pinned

        self._passRT1_fft    (sino, sino_c, chunk_theta)
        # fde must start at 0 — the adjoint gather atomicAdds into it.
        # `_get_fde_rt` already zeroed it on entry.
        self._passRT2_scatter(sino_c, fde)
        self._passRT3_yifft  (         fde, chunk_n)
        self._passRT4_xifft  (         fde, obj, chunk_n)

        return obj

    def _passRT1_fft(self, sino, sino_c, chunk_theta):
        """PassRT1 — adjoint of pass4.  For each θ-chunk: read sino
        (real f32 or complex64) on host, cast + multiply by c1dfftshift,
        1-D FFT along r, multiply by c1dfftshift and pass4's scale
        factor, D2H back into sino_c (the complex-c64 buffer scatter
        reads from).
        """
        n, ntheta = self.n, self.ntheta
        nz = sino.shape[1]
        c1d_gpu = cp.asarray(self.c1dfftshift)
        # The forward TomoReal / TomoLarge bakes 1/(n·√(n·ntheta)) into
        # `phi` at init — the host-chunked variant doesn't, so RT
        # compensates by removing the /4 that pass4 uses in the forward
        # direction.  Net scale = 1/(n·√(n·ntheta)).
        scale   = np.float32(1.0 / (n * np.sqrt(n * ntheta)))

        shape = (chunk_theta, nz, n)
        # Slot 'pRT1' — distinct from forward 'p4' to avoid dtype clash
        # (forward 'p4' is (c64 in, f32 out); RT is (c64 in, c64 out)).
        pipe = self._get_pipe('pRT1', shape, shape,
                              np.complex64, np.complex64)

        def load(k, dst):
            st = k * chunk_theta
            # Cast real f32 sinos to complex64 on the host side; loader
            # accepts either dtype since pin_in is c64.
            dst[:] = sino[st:st + chunk_theta]

        def compute(k, in_gpu, out_gpu):
            cp.multiply(in_gpu, c1d_gpu, out=out_gpu)
            cufft.fft(out_gpu, axis=-1, overwrite_x=True)   # in-place on innermost
            out_gpu *= c1d_gpu
            out_gpu *= scale

        def store(k, src):
            st = k * chunk_theta
            sino_c[st:st + chunk_theta] = src

        pipe.run(load, compute, store, ntheta // chunk_theta)

    def _passRT2_scatter(self, sino, fde):
        """PassRT2 — adjoint of pass3, compact-index scatter with
        3-stream ping-pong H2D / compute / D2H via StreamPipe.

        Per-band (theta_i, x_i) int32 pairs are pre-decoded in
        ``_per_band_precompute`` (once at construction) and uploaded
        per band inside the compute callback — cheap H2D on the
        compute stream, doesn't block the pipe.

        Per band: pinned_in holds the compact sino slab `(nz, max_nel)`,
        pinned_out holds one fde slice `(nz, chunk_xy, 2n)`, both
        double-buffered.  Band N's D2H overlaps with N+1's compute and
        N+2's H2D.
        """
        n        = self.n
        nz       = fde.shape[0]
        twon     = 2 * n
        m, mua   = self.m, self.mua
        chunk_xy = self._chunks_rt[2]

        theta_gpu = cp.asarray(self.theta)
        per_band  = self._pb_cache
        n_bands   = len(per_band)
        max_nel   = int(max((ti.size for ti, _ in per_band), default=0))
        if max_nel == 0:
            return
        threads = 1024

        pipe = self._get_pipe('pRT2',
            in_shape=(nz, max_nel),          in_dtype=np.complex64,
            out_shape=(nz, chunk_xy, twon),  out_dtype=np.complex64)

        def load(b, dst_pinned):
            theta_i, x_i = per_band[b]
            nel = theta_i.size
            if nel == 0:
                return
            for z in range(nz):
                dst_pinned[z, :nel] = sino[theta_i, z, x_i]

        def compute(b, in_gpu, out_gpu):
            theta_i, x_i = per_band[b]
            nel = int(theta_i.size)
            if nel == 0:
                return
            y_lo = b * chunk_xy
            y_hi = min(y_lo + chunk_xy, twon)
            theta_i_d = cp.asarray(theta_i)
            x_i_d     = cp.asarray(x_i)
            out_gpu.fill(0)
            grid, block = (int(np.ceil(nel / threads)), 1, nz), (threads, 1, 1)
            scatter_compact_kernel(grid, block,
                (in_gpu, out_gpu, x_i_d, theta_i_d, theta_gpu,
                 m, mua, n, nel, nz,
                 np.int32(y_lo), np.int32(y_hi),
                 np.int32(max_nel), np.int32(chunk_xy)))

        def store(b, src_pinned):
            y_lo = b * chunk_xy
            y_hi = min(y_lo + chunk_xy, twon)
            fde.copy_from(src_pinned[:, :y_hi - y_lo, :],
                          np.s_[:, y_lo:y_hi, :])

        pipe.run(load, compute, store, n_bands)

    def _passRT3_yifft(self, fde, chunk_n):
        """PassRT3 — adjoint of pass2's y-FFT.  For each x-strip of the
        (nz, 2n, 2n) c64 fde: read on host, multiply by c2dfftshift-in-y
        and c2dfftshift-in-x, 1-D IFFT along y, mask again, D2H.

        The full-complex layout uses c2dfftshift (both axes centred);
        pass2 of forward R uses raw fftfreq order (rfft path), but RT's
        full-complex path stays on the centred layout so its result
        matches TomoReal.RT.
        """
        n, nz = self.n, fde.shape[0]
        c2d1d_gpu = cp.asarray(self.c2dfftshift1d)

        shape = (nz, 2 * n, chunk_n)
        # Slot 'pRT3' — separate from forward 'p2' whose ping-pong shape
        # here (2 * n, chunk_n) happens to match; kept distinct so R's
        # pipe cache is not disturbed.
        pipe = self._get_pipe('pRT3', shape, shape,
                              np.complex64, np.complex64)

        def load(k, dst):
            st = k * chunk_n
            fde.copy_to(dst, np.s_[:, :, st:st + chunk_n])

        def compute(k, in_gpu, out_gpu):
            st, end = k * chunk_n, (k + 1) * chunk_n
            c2dx = c2d1d_gpu[st:end]
            cp.multiply(in_gpu, c2d1d_gpu[None, :, None], out=out_gpu)
            out_gpu *= c2dx[None, None, :]
            # cufft.overwrite_x is unreliable for non-innermost axes here
            # — use cp.fft.ifft.
            out_gpu[...] = cp.fft.ifft(out_gpu, axis=1)
            out_gpu *= c2d1d_gpu[None, :, None]
            out_gpu *= c2dx[None, None, :]

        def store(k, src):
            st = k * chunk_n
            fde.copy_from(src, np.s_[:, :, st:st + chunk_n])

        pipe.run(load, compute, store, 2 * n // chunk_n)

    def _passRT4_xifft(self, fde, obj, chunk_n):
        """PassRT4 — adjoint of pass1.  For each z-strip of fde (chunk_n
        rows of the 2n y-axis): read on host, mask twice, 1-D IFFT along
        x, mask twice, crop the center-n columns, multiply by
        phi_scale·phi1d·phi1d (the same separable φ as pass1), D2H the
        result into obj.  Only the center chunk_n rows of the y-axis
        contribute — outer rows come out ~0 after the IFFT, so we skip.
        """
        n, nz     = self.n, obj.shape[0]
        phi_scale = self.phi_scale
        # phi1d is float32 for the forward rfft path; RT still multiplies
        # a complex intermediate by it — cupy will broadcast the f32 phi
        # against the c64 buffer without an intermediate promotion.
        phi1d_gpu = cp.asarray(self.phi1d)                # (n,) f32
        c2d1d_gpu = cp.asarray(self.c2dfftshift1d)        # (2n,) i8

        # We only need the center-n y-rows for the final obj crop, but
        # x-IFFT operates on the full 2n columns → we load full-2n
        # strips like pass1's out_shape.
        in_shape  = (nz, chunk_n, 2 * n)
        out_shape = (nz, chunk_n, n)                       # after center crop
        # Slot 'pRT4' — RT's shape/dtype match forward 'p1' partially but
        # we keep separate slots to isolate RT's pipe state from R's.
        pipe = self._get_pipe('pRT4', in_shape, out_shape,
                              np.complex64, np.complex64)

        # We iterate over the center-n rows of y — the outer half is
        # zero-padding for pass1 and does not carry object information.
        n_iter = n // chunk_n

        def load(k, dst):
            # k-th chunk in the CENTER n rows → y-offset (n//2 + k·chunk_n).
            st = n // 2 + k * chunk_n
            fde.copy_to(dst, np.s_[:, st:st + chunk_n, :])

        def compute(k, in_gpu, out_gpu):
            # k is a chunk index in the CENTER n rows; the pass1 phi row
            # factor at row (n//2 + k*chunk_n + i) equals phi1d[k*chunk_n + i]
            # since pass1 wrote `fde[y = n//2 + st + i] = phi1d[st + i] * obj[i]`.
            st, end = k * chunk_n, (k + 1) * chunk_n
            c2dx = c2d1d_gpu[n // 2 + st : n // 2 + end]   # (chunk_n,)
            in_gpu *= c2dx[None, :, None]
            in_gpu *= c2d1d_gpu[None, None, :]
            in_gpu[...] = cp.fft.ifft(in_gpu, axis=-1)
            in_gpu *= c2dx[None, :, None]
            in_gpu *= c2d1d_gpu[None, None, :]
            cropped = in_gpu[:, :, n // 2 : n // 2 + n]
            phiy = phi1d_gpu[st:end]                        # (chunk_n,) f32
            out_gpu[...] = (phi_scale
                            * phiy[None, :, None]
                            * phi1d_gpu[None, None, :]
                            * cropped)

        def store(k, src):
            st = k * chunk_n
            if obj.dtype == np.float32:
                obj[:, st:st + chunk_n, :] = src.real
            else:
                obj[:, st:st + chunk_n, :] = src

        pipe.run(load, compute, store, n_iter)
