"""TomoLarge — host-chunked USFFT Radon for volumes too big for GPU-only Tomo.

Vendored from radon_large/tomo_large.py.  Stages small pieces of the padded
(2N × 2N) frequency-domain buffer through the GPU while keeping the big
`fde` and `sino` arrays on the HOST.  Peak GPU memory is proportional to
the chunk sizes rather than to (2N)².
"""
from __future__ import annotations

import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft

from processing.kernels import gather_kernel1, gather_kernel_rfft
from processing.pipeline import StreamPipe, ComputeD2HPipe, alloc_pinned, BandedPinned

# Number of pinned bands `_get_fde` splits its (nz, 2n, ...) buffer into
# along the 2n axis.  cudaHostAlloc has a per-request cap (~64 GiB on some
# boxes, more on others); fde grows as N² so at large UPS a single alloc
# hits that ceiling.  4 bands keeps each cudaHostAlloc call well under
# any observed cap while adding only a per-load band-stitch loop.
FDE_N_BANDS = 4


class TomoLarge:
    """Radon transform via USFFT with host-staged chunking.

    R(obj, chunks) is the only entry-point used by model_radon_large.py;
    the adjoint RT is included for round-trip debugging.
    """

    def __init__(self, n, theta, rotation_axis=None):
        """USFFT parameter setup (host-side); no per-call GPU allocations."""
        eps = 1e-3
        mu  = -np.log(eps) / (2 * n * n)
        m   = int(np.ceil(2 * n * 1 / np.pi *
                          np.sqrt(-mu * np.log(eps) + (mu * n) ** 2 / 4)))

        ntheta = len(theta)

        # phi is the Gaussian pre-multiplication kernel; phi[i, j] =
        # exp(mu·n²·(t[i]²+t[j]²))·(1−n%4) is separable → store only the
        # 1-D factor and the ±1 scalar.  Saves n² c64 at UPS≥8.
        t = np.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        phi1d     = np.exp(mu * (n * n) * (t * t)).astype("complex64")
        phi_scale = np.complex64(1 - n % 4)

        # c2dfftshift is the outer product of the (2n,) ±1 checkerboard
        # with itself; store just the 1-D vector.  Saves (2n)² i8.
        c1dfftshift    = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")
        c2dfftshift1d  = (1 - 2 * (np.arange(1, 2 * n + 1) % 2)).astype("int8")

        # Sample coordinates x = cos(theta)·r, y = −sin(theta)·r are now
        # recomputed inside the gather kernel from (cos_theta, sin_theta,
        # full_idx).  Store only the two (ntheta,) trig tables here.
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
        self.phi1d       = phi1d
        self.phi_scale   = phi_scale
        self.c1dfftshift = c1dfftshift
        self.c2dfftshift1d = c2dfftshift1d
        # GPU-side trig tables + 1-D FFT weights uploaded once per R() call.
        self._cos_theta_gpu = None
        self._sin_theta_gpu = None
        # rotation_axis is accepted for API compatibility but unused here
        # (rotation is centred at N/2 via the sample-point formulas).
        self.rotation_axis = rotation_axis

        # Cached pageable host buffers (biggest allocations in R()).
        self._fde  = None
        self._sino = None
        # Cached pipes + per-bin GPU staging (preserved across R() calls;
        # rebuilt only when the driving shape/dtype changes).  Keeps the
        # pinned ping-pong buffers alive across calls so we don't burn
        # tens of GB of pinned pages on every invocation.
        self._pipe1  = None
        self._pipe2  = None
        self._pipe4  = None
        self._pipe_gather = None
        # Contiguous pinned staging for Pass-3 fde fetch — reused across
        # bins; sized to max bin fetch.  1D so bin-width variations reshape
        # to a contiguous head slice without stride issues.
        self._pin_fde_stg = None
        # Cache the (nel, idx) result of _sort_into_chunks — deterministic
        # given chunk_xy, so re-doing it every R() call wastes ~16·ntheta·n
        # bytes (~116 GB @ UPS=32) of pageable-host alloc/free churn.
        self._sort_key   = None
        self._sort_cache = None
        # Shared pinned scratch pool for the sequentially-run StreamPipes
        # (pipe1/pipe2/pipe4).  Each is a pair of pinned byte buffers grown
        # on demand to the max pass need; every pass takes fresh views out
        # of the same bytes.  Saves the per-pipe pinned duplication that
        # used to sit idle whenever a pipe wasn't the currently-running one.
        self._scratch_in      = None
        self._scratch_out     = None
        self._scratch_in_cap  = 0
        self._scratch_out_cap = 0
        # GPU-side analog of the pinned scratch: cupy complex64 buffers
        # viewed per-pass as the required (shape, dtype).  Because
        # pipe1/pipe2/pipe4 run sequentially, only one holds live GPU
        # buffers at a time; without this, all three cached pipes retain
        # their own ping-pong GPU allocations in the cupy pool.
        self._scratch_in_gpu  = None
        self._scratch_out_gpu = None
        self._scratch_in_gpu_cap  = 0
        self._scratch_out_gpu_cap = 0

    def _sort_into_chunks(self, chunk_xy):
        """Precompute per-chunk sample lists for a given XY chunk size.

        qid is built theta-by-theta on the fly so we never materialise the
        full x/y coordinate table on the host (that used to cost 2·4·ntheta·n
        bytes = 54 GB at UPS=32).  Stored as int32 — max qid = (2n/chunk_xy)²
        which fits comfortably (< 2^31 for every UPS we run).  Result is
        cached on the instance and reused across R() calls with the same
        chunk_xy.
        """
        if self._sort_key == chunk_xy:
            return self._sort_cache

        n = self.n
        n_bin = 2 * n // chunk_xy
        r = (np.arange(n, dtype="float32") - n / 2) / n
        qid = np.empty(self.ntheta * n, dtype="int32")
        for k in range(self.ntheta):
            x_k = np.clip( self.cos_theta[k] * r, -0.5, 0.5 - 1e-5)
            y_k = np.clip(-self.sin_theta[k] * r, -0.5, 0.5 - 1e-5)
            fx  = np.floor(2 * n * x_k).astype("int32") + n
            fy  = np.floor(2 * n * y_k).astype("int32") + n
            qid[k * n:(k + 1) * n] = (fy // chunk_xy) * n_bin + fx // chunk_xy

        # idx is int64 because ntheta·n exceeds 2^31 at UPS=32.  No x_s/y_s
        # copies — the kernel recomputes from full_idx = idx[bin_slice].
        idx = np.argsort(qid, kind="stable")
        qid_s = qid[idx]

        nel = np.zeros(n_bin * n_bin, dtype="int64")
        change_points = np.flatnonzero(np.diff(qid_s, prepend=qid_s[0] - 1))
        run_lengths = np.diff(np.append(change_points, len(qid_s)))
        nel[qid_s[change_points]] = run_lengths
        del qid, qid_s   # release the ~2·(ntheta·n·4)-byte scratch

        self._sort_key   = chunk_xy
        self._sort_cache = (nel, idx)
        return self._sort_cache

    def _get_st_end(self, indx, indy, chunk_xy):
        """Halo-extended chunk range in the ABSOLUTE (2n × 2n) grid frame.
        Unclamped: may be negative or exceed 2n; edge chunks are stitched
        together with the opposite-edge wrap in `_fetch_fde_chunk` so that
        the padded (2n × 2n) Fourier grid is treated as periodic — same
        semantics as `Tomo`'s modular `(n + ell + twon) % twon` gather.
        """
        m = self.m
        stx = indx * chunk_xy - m
        endx = (indx + 1) * chunk_xy + m + 1
        sty = indy * chunk_xy - m
        endy = (indy + 1) * chunk_xy + m + 1
        return [stx, endx, sty, endy]

    def _fetch_fde_chunk_to_gpu(self, fde, sty, endy, stx, endx):
        """Return a contiguous cupy c64 array of shape (nz, endy−sty, endx−stx)
        containing fde[:, sty:endy, stx:endx] with WRAP on both trailing axes
        (periodic 2n × 2n).

        fde is banded on axis 1, so a single fancy-index expression like
        `fde[:, ky[:, None], kx[None, :]]` no longer works.  We split into
        two paths and construct the GPU patch directly:
          * fast path (no wrap): per-band basic-slice H2D via copy_to_gpu.
          * wrap path (edge bins only): build a small host temp using
            per-band fancy indexing on the wrapped ky vector, then upload.
        """
        n2 = 2 * self.n
        nz = fde.shape[0]
        dst = cp.empty((nz, endy - sty, endx - stx), dtype=cp.complex64)
        if 0 <= sty and endy <= n2 and 0 <= stx and endx <= n2:
            fde.copy_to_gpu(dst, np.s_[:, sty:endy, stx:endx])
            return dst
        # Wrap path — small edge-bin allocation on host, then one H2D.
        ky = np.arange(sty, endy) % n2
        kx = np.arange(stx, endx) % n2
        patch = np.empty((nz, endy - sty, endx - stx), dtype=np.complex64)
        for bi in range(fde.n_bands):
            b_lo = bi * fde.band_rows
            b_hi = b_lo + fde.band_rows
            mask = (ky >= b_lo) & (ky < b_hi)
            if not mask.any():
                continue
            local_ky = ky[mask] - b_lo
            patch[:, mask, :] = fde.bands[bi][:, local_ky[:, None], kx[None, :]]
        dst.set(patch)
        return dst

    # ---------- explicit teardown -------------------------------------------
    def free(self):
        """Release cached pinned host + GPU buffers back to their pools.
        Call this between iterations of a size sweep (or before switching
        to a very different size) so the previous instance's fde/sino
        (multi-hundred-GB pinned) don't linger through the next
        allocation attempt.  Cheaper than waiting for GC."""
        self._fde  = None
        self._sino = None
        self._pipe1 = None
        self._pipe2 = None
        self._pipe4 = None
        self._pipe_gather = None
        self._scratch_in  = None
        self._scratch_out = None
        self._scratch_in_cap  = 0
        self._scratch_out_cap = 0
        self._scratch_in_gpu  = None
        self._scratch_out_gpu = None
        self._scratch_in_gpu_cap  = 0
        self._scratch_out_gpu_cap = 0

    # ---------- shared pinned scratch pool for pipe1/2/4 --------------------
    def _scratch_views(self, which, shape, dtype):
        """Return a 2-element list of pinned numpy views (shape, dtype) into
        one of the shared byte scratch buffers.  Grows the underlying pinned
        allocation to the requested size on demand.

        `which` = 'in' or 'out' — pipe1/2/4 each need one ping-pong pair on
        each side, but only one pipe runs at a time, so all three passes
        share the same underlying bytes.  Peak pinned pipe RAM drops from
        ``sum(pipe_bytes)`` to ``max(pipe_bytes)``.
        """
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

    # ---------- shared GPU scratch pool for pipe1/2/4 -----------------------
    def _scratch_gpu_views(self, which, shape, dtype):
        """Return 2 cupy views (shape, dtype) into the shared GPU scratch
        buffers.  Analog of _scratch_views but for the pipe in_gpu/out_gpu
        ping-pongs.  Underlying storage is a pair of complex64 buffers big
        enough for the largest pass; views reinterpret the bytes.
        """
        dtp = np.dtype(dtype)
        n_elem = int(np.prod(shape))
        need = n_elem * dtp.itemsize
        n_c64 = (need + 7) // 8         # complex64 = 8 B; upper bound on elems
        if which == 'in':
            if self._scratch_in_gpu_cap < need:
                # Drop caches on the cached pipes so their old (smaller) views
                # release their refcount on the old scratch bytes before we
                # allocate the new (bigger) scratch — keeps peak = max, not
                # old+new.
                for p in (self._pipe1, self._pipe2, self._pipe4):
                    if p is not None:
                        p.in_gpu = []
                self._scratch_in_gpu = None      # release first
                self._scratch_in_gpu = [cp.empty((n_c64,), dtype=cp.complex64)
                                        for _ in range(2)]
                self._scratch_in_gpu_cap = n_c64 * 8
            bufs = self._scratch_in_gpu
        else:
            if self._scratch_out_gpu_cap < need:
                for p in (self._pipe1, self._pipe2, self._pipe4):
                    if p is not None:
                        p.out_gpu = []
                self._scratch_out_gpu = None
                self._scratch_out_gpu = [cp.empty((n_c64,), dtype=cp.complex64)
                                         for _ in range(2)]
                self._scratch_out_gpu_cap = n_c64 * 8
            bufs = self._scratch_out_gpu
        # Reinterpret each c64 buffer as the requested dtype, then slice
        # and reshape to the target shape.
        return [b.view(dtype)[:n_elem].reshape(shape) for b in bufs]

    # ---------- pinned host-buffer cache ------------------------------------
    # Allocated PINNED so the Pass-3 fde fetch (cp.array(fde[..slice..]))
    # is a fast pinned→GPU H2D and so the step2/step3 HDD-read pipelines
    # can read chunks directly into these buffers.  Cached across R()
    # calls when the shape matches so we don't re-pin many GB per call.
    def _get_fde(self, nz):
        """(nz, 2n, 2n) complex64 buffer with the y-padding rows zeroed —
        Pass 1 writes only rows [n/2, 3n/2), so only the outer rows need
        to start at zero (Pass 2's y-FFT depends on that padding).
        Banded on the 2n (band_axis=1) axis; see FDE_N_BANDS."""
        n = self.n
        shape = (nz, 2 * n, 2 * n)
        if self._fde is None or self._fde.shape != shape:
            self._fde = BandedPinned(shape, np.complex64,
                                     n_bands=FDE_N_BANDS, band_axis=1)
            # First alloc: touch every page once so the whole buffer is
            # resident (subsequent runs skip the page-fault storm).
            self._fde.fill(0)
        else:
            # Reused buffer — zero only the y-padding rows that Pass 1
            # doesn't overwrite (avoids a full 275 GB write at UPS=32).
            self._fde[:, :n // 2, :]     = 0
            self._fde[:, n // 2 + n:, :] = 0
        return self._fde

    def _get_sino(self, nz):
        """(ntheta, nz, n) complex64 sino buffer.  Gather overwrites every
        element, so contents may be undefined on entry."""
        shape = (self.ntheta, nz, self.n)
        if self._sino is None or self._sino.shape != shape:
            self._sino = alloc_pinned(shape, np.complex64)
        return self._sino

    # ---------- forward Radon ------------------------------------------------
    def R(self, obj, chunks):
        """(nz, n, n) obj → (ntheta, nz, n) sinogram; obj/sino live on host.

        chunks = [CHUNK_N, CHUNK_THETA, CHUNK_XY] — chunk sizes for the
        1-D FFTs, angle grouping, and gather bin size respectively.
        """
        chunk_n, chunk_theta, chunk_xy = chunks
        nz = obj.shape[0]

        # Pinned host accumulators — allocated once, reused across calls.
        fde  = self._get_fde(nz)                     # (nz, 2n, 2n) c64 zeros
        sino = self._get_sino(nz)                    # (ntheta, nz, n) c64

        # Per-chunk NUFFT sample sort — theta-major (nel, idx) only.
        nel, idx = self._sort_into_chunks(chunk_xy)

        # Upload the trig tables the gather kernel reads.  Tiny (~4·ntheta
        # bytes each) so we just re-upload per R() call.
        self._cos_theta_gpu = cp.asarray(self.cos_theta)
        self._sin_theta_gpu = cp.asarray(self.sin_theta)

        # No per-pass pool clearing — the cached pipes reuse their pinned +
        # GPU buffers across R() calls, and we WANT the pool to keep those
        # blocks live.  Fragmentation from mid-pass cufft plan workspaces
        # is bounded by the (halved) chunk-picker budget.
        self._pass1_xfft   (obj,  fde, chunk_n)
        self._pass2_yfft   (      fde, chunk_n)
        self._pass3_gather (      fde, sino, nel, idx, chunk_xy)
        self._pass4_ifft   (            sino, chunk_theta)

        return sino

    # ---------- individual passes -------------------------------------------
    def _pass1_xfft(self, obj, fde, chunk_n):
        """Pass 1 — pipelined x-axis FFT strips.

        For each chunk k, transfers obj[:, k·chunk_n:(k+1)·chunk_n] to the
        GPU, multiplies by phi (rebuilt from the 1-D factor via broadcast),
        zero-pads x from n to 2n, FFTs along x, multiplies by c2dfftshift
        twice (before and after the FFT — the fftshift-via-multiply trick),
        then writes the padded output row strip into fde[...].
        """
        n, nz = self.n, obj.shape[0]
        phi_scale = self.phi_scale
        phi1d_gpu = cp.asarray(self.phi1d)                # (n,)  c64  ~ 1 MB @ UPS=32
        c2d1d_gpu = cp.asarray(self.c2dfftshift1d)        # (2n,) i8   ~200 KB @ UPS=32

        in_shape  = (nz, chunk_n, n)
        out_shape = (nz, chunk_n, 2 * n)
        # Fresh pinned views come from the shared scratch pool every call
        # (scratch may have grown since last time); the GPU ping-pong
        # buffers are the expensive part, so we keep the StreamPipe object
        # cached whenever the shapes match — otherwise `self._pipe1 =
        # StreamPipe(...)` momentarily holds two sets of GPU ping-pongs
        # while the old is still referenced, doubling cupy's pool peak.
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

        def load(k, dst):
            st = k * chunk_n
            dst[:] = obj[:, st:st + chunk_n, :]

        def compute(k, in_gpu, out_gpu):
            st, end = k * chunk_n, (k + 1) * chunk_n
            phix = phi1d_gpu[st:end]                             # (chunk_n,)  c64
            c2dx = c2d1d_gpu[st:end]                             # (chunk_n,)  i8
            out_gpu.fill(0)
            # phi[i, j] = phi_scale · phi1d[i] · phi1d[j] — broadcast in-kernel.
            out_gpu[:, :, n // 2 : n // 2 + n] = (
                phi_scale * phix[None, :, None] * phi1d_gpu[None, None, :] * in_gpu
            )
            # Chained ±1 mask via 1-D broadcast; two multiplies fuse better
            # than materialising the (chunk_n, 2n) outer product.
            out_gpu *= c2dx[None, :, None]
            out_gpu *= c2d1d_gpu[None, None, :]
            out_gpu[...] = cp.fft.fft(out_gpu, axis=-1)
            out_gpu *= c2dx[None, :, None]
            out_gpu *= c2d1d_gpu[None, None, :]

        def store(k, src):
            st = k * chunk_n
            fde.copy_from(src, np.s_[:, n // 2 + st : n // 2 + st + chunk_n, :])

        pipe.run(load, compute, store, n // chunk_n)

    def _pass2_yfft(self, fde, chunk_n):
        """Pass 2 — pipelined y-axis FFT strips (chunked along x)."""
        n, nz = self.n, fde.shape[0]
        c2d1d_gpu = cp.asarray(self.c2dfftshift1d)        # (2n,) i8

        shape = (nz, 2 * n, chunk_n)
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
            st = k * chunk_n
            fde.copy_to(dst, np.s_[:, :, st:st + chunk_n])

        def compute(k, in_gpu, out_gpu):
            st, end = k * chunk_n, (k + 1) * chunk_n
            c2dx = c2d1d_gpu[st:end]                     # (chunk_n,) i8
            # c2d_full[y, x] = c2d1d[y] · c2d1d[st + x] — apply as two
            # broadcasted multiplies; avoids the (2n, chunk_n) intermediate.
            cp.multiply(in_gpu, c2d1d_gpu[None, :, None], out=out_gpu)
            out_gpu *= c2dx[None, None, :]
            # cufft's overwrite_x is unreliable for non-innermost axes in
            # this cupy version → use cp.fft.fft (returns a new array).
            out_gpu[...] = cp.fft.fft(out_gpu, axis=1)
            out_gpu *= c2d1d_gpu[None, :, None]
            out_gpu *= c2dx[None, None, :]

        def store(k, src):
            st = k * chunk_n
            fde.copy_from(src, np.s_[:, :, st:st + chunk_n])

        pipe.run(load, compute, store, 2 * n // chunk_n)

    def _pass3_gather(self, fde, sino, nel, idx, chunk_xy):
        """Pass 3 — NUFFT gather per (indx, indy) bin, with the inner z
        loop streamed via ComputeD2HPipe.

        Each bin uploads its fde chunk (H2D once) plus its per-sample flat
        index list (int64), then runs gather_kernel1 for every z-slice on
        the compute stream while the previous slice's D2H drains on a
        second stream, and the main thread scatters the pinned result into
        `sino` via a precomputed flat index.  The kernel recomputes
        (x, y) = (cos_theta·r, −sin_theta·r) from full_idx.
        """
        n = self.n
        nz = fde.shape[0]
        m, mua = self.m, self.mua
        n_chunk_xy = 2 * n // chunk_xy
        cos_theta_gpu = self._cos_theta_gpu
        sin_theta_gpu = self._sin_theta_gpu

        # Buffers sized to the largest bin; reused across bins AND across
        # R() calls (cached on the instance).
        max_nel = int(max((int(v) for v in nel), default=0))
        if max_nel == 0:
            return
        if (self._pipe_gather is None
                or self._pipe_gather.out_shape != (max_nel,)):
            self._pipe_gather = ComputeD2HPipe((max_nel,), np.complex64)
        gather_pipe = self._pipe_gather

        offset = 0
        # Reshape sino to a flat 1D view so gather-index scatter is simple.
        sino_flat = sino.reshape(-1)

        for indy in range(n_chunk_xy):
            for indx in range(n_chunk_xy):
                nel_i = int(nel[indy * n_chunk_xy + indx])
                if nel_i == 0:
                    continue
                stx, endx, sty, endy = self._get_st_end(indx, indy, chunk_xy)

                # Upload on the compute stream so the kernel reads after
                # the H2D completes.  fde is banded pinned — the fetch
                # helper builds a contiguous GPU patch either via per-band
                # copy_to_gpu (fast path) or a small host stitch + one
                # H2D (wrap path for edge bins).
                full_idx  = idx[offset : offset + nel_i]      # int64, host
                flat_base = (full_idx // n) * nz * n + (full_idx % n)
                with gather_pipe.s_comp:
                    fde_d = self._fetch_fde_chunk_to_gpu(fde, sty, endy, stx, endx)
                    fidx_d = cp.asarray(full_idx)             # int64

                grid, block = (int(np.ceil(nel_i / 1024)),), (1024,)

                def compute(zc, out_gpu,
                            _nel=nel_i, _fde=fde_d, _fidx=fidx_d,
                            _stx=stx, _endx=endx, _sty=sty, _endy=endy,
                            _grid=grid, _block=block):
                    out_gpu[:_nel].fill(0)
                    gather_kernel1(_grid, _block,
                        (out_gpu[:_nel], _fde[zc],
                         _fidx, cos_theta_gpu, sin_theta_gpu,
                         m, mua, _nel,
                         _stx, _endx, _sty, _endy, n, 0))

                def store(zc, src_pinned, _nel=nel_i, _base=flat_base):
                    sino_flat[_base + zc * n] = src_pinned[:_nel]

                gather_pipe.run(compute, store, nz)
                offset += nel_i

    def _pass4_ifft(self, sino, chunk_theta):
        """Pass 4 — pipelined 1-D IFFT along the r axis + normalisation."""
        n, ntheta = self.n, self.ntheta
        nz = sino.shape[1]
        c1d_gpu = cp.asarray(self.c1dfftshift)
        scale = np.float32(1.0 / (4 * n * np.sqrt(n * ntheta)))

        shape = (chunk_theta, nz, n)
        pin_in  = self._scratch_views('in',  shape, np.complex64)
        pin_out = self._scratch_views('out', shape, np.complex64)
        if (self._pipe4 is None
                or self._pipe4.in_shape  != shape
                or self._pipe4.out_shape != shape
                or self._pipe4.out_dtype != np.dtype(np.complex64)):
            self._pipe4 = StreamPipe(shape, shape, np.complex64, np.complex64,
                                     pinned_in=pin_in, pinned_out=pin_out)
        else:
            self._pipe4.in_pin  = pin_in
            self._pipe4.out_pin = pin_out
        pipe = self._pipe4

        def load(k, dst):
            st = k * chunk_theta
            dst[:] = sino[st:st + chunk_theta]

        def compute(k, in_gpu, out_gpu):
            cp.multiply(in_gpu, c1d_gpu, out=out_gpu)
            cufft.ifft(out_gpu, axis=-1, overwrite_x=True)   # in-place on innermost
            out_gpu *= c1d_gpu
            out_gpu *= scale

        def store(k, src):
            st = k * chunk_theta
            sino[st:st + chunk_theta] = src

        pipe.run(load, compute, store, ntheta // chunk_theta)


# =============================================================================
# TomoLargeReal — float32 obj + rfft-along-x variant of TomoLarge.
# =============================================================================
class TomoLargeReal:
    """Same forward Radon as TomoLarge, but the object is REAL (float32)
    and the x-axis FFT uses rfft — cutting the (nz, 2n, 2n) complex64
    ``fde`` buffer down to (nz, 2n, n+1) (half the host RAM, half the
    x-axis FFT cost).  Along y we still do a full complex FFT of length
    2n.  The gather kernel exploits ``X[fx, fy] = conj(X[-fx, -fy])``
    (real-input symmetry) to reach into the missing negative-fx half
    without ever storing it.

    Semantics: R(obj_float32) is bit-equivalent (up to fp roundoff) to
    ``TomoLarge.R(obj_complex64_with_imag_zero).real`` for the same obj.

    Layout choices (both fftshift tricks dropped, RAW fftfreq order):
      * fde stored in fftfreq order along BOTH x (rfft, so [0, n]) and
        y (full fft, so [0, 2n) with wrap).  No c2dfftshift needed.
      * sino stored in centered (fftshift-along-r) order for backward
        compatibility with downstream step3 / analysis code.  The Pass 4
        1-D IFFT along r therefore still uses c1dfftshift.

    Output sino: **REAL float32**, shape ``(ntheta, nz, n)`` — Pass 4
    takes ``.real`` after the r-axis IFFT (imag ≈ 0 for real obj).
    """

    def __init__(self, n, theta, rotation_axis=None):
        eps = 1e-3
        mu  = -np.log(eps) / (2 * n * n)
        m   = int(np.ceil(2 * n / np.pi *
                          np.sqrt(-mu * np.log(eps) + (mu * n) ** 2 / 4)))
        ntheta = len(theta)

        # phi: real Gaussian pre-multiplication.  Separable — store 1-D
        # factor + ±1 scalar (see TomoLarge.__init__ for full derivation).
        t = np.linspace(-1 / 2, 1 / 2, n, endpoint=False).astype("float32")
        phi1d     = np.exp(mu * (n * n) * (t * t)).astype("float32")
        phi_scale = np.float32(1 - n % 4)

        # c1dfftshift kept — needed for the centered sino layout in Pass 4.
        c1dfftshift = (1 - 2 * ((cp.arange(1, n + 1) % 2))).astype("int8")

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
        self.rotation_axis = rotation_axis

        # Cached pinned host buffers.  sino_real is a VIEW into the first
        # half of sino's bytes (float32 vs complex64), so there's no
        # separate cache — it's derived per R() call in _pass4_ifft.
        self._fde       = None                   # (nz, 2n, n+1) complex64
        self._sino      = None                   # (ntheta, nz, n) complex64
        self._obj       = None                   # (nz, n, n) float32 — pinned
        # Cached pipes + Pass-3 pinned staging (see TomoLarge for rationale).
        self._pipe1  = None
        self._pipe2  = None
        self._pipe4  = None
        self._pipe_gather = None
        self._pin_fde_stg = None       # 1D pinned pool for the fde fetch
        # GPU trig tables uploaded once per R() call.
        self._cos_theta_gpu = None
        self._sin_theta_gpu = None
        # See TomoLarge._sort_into_chunks for the caching rationale.
        self._sort_key   = None
        self._sort_cache = None
        # Shared pinned scratch for pipe1/pipe2/pipe4 — see TomoLarge for
        # the rationale (sequential passes → dedupe the ping-pong pinned).
        self._scratch_in      = None
        self._scratch_out     = None
        self._scratch_in_cap  = 0
        self._scratch_out_cap = 0
        self._scratch_in_gpu  = None
        self._scratch_out_gpu = None
        self._scratch_in_gpu_cap  = 0
        self._scratch_out_gpu_cap = 0

    # ---------- bin sort for the gather -----------------------------------
    def _sort_into_chunks(self, chunk_xy):
        """Bin samples by rk_x only (the POST-reflection column in the
        stored [0, n+1) half-spectrum).  qid built theta-by-theta from the
        trig tables + r — no ntheta·n coordinate table ever materialised
        on the host.  qid stored as int32; max rk_x = n < 2^31.  Result
        cached per chunk_xy across R() calls (same rationale as TomoLarge).
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

    def _get_st_end(self, indx, chunk_xy):
        """Halo-extended x-column range in the stored [0, n+1) half-spectrum."""
        n, m = self.n, self.m
        stx = max(0, indx * chunk_xy - m)
        endx = min(n + 1, (indx + 1) * chunk_xy + m + 1)
        return stx, endx

    # ---------- explicit teardown — see TomoLarge.free -----------------
    def free(self):
        self._fde  = None
        self._sino = None
        self._obj  = None
        self._pipe1 = None
        self._pipe2 = None
        self._pipe4 = None
        self._pipe_gather = None
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
                for p in (self._pipe1, self._pipe2, self._pipe4):
                    if p is not None:
                        p.in_gpu = []
                self._scratch_in_gpu = None
                self._scratch_in_gpu = [cp.empty((n_c64,), dtype=cp.complex64)
                                        for _ in range(2)]
                self._scratch_in_gpu_cap = n_c64 * 8
            bufs = self._scratch_in_gpu
        else:
            if self._scratch_out_gpu_cap < need:
                for p in (self._pipe1, self._pipe2, self._pipe4):
                    if p is not None:
                        p.out_gpu = []
                self._scratch_out_gpu = None
                self._scratch_out_gpu = [cp.empty((n_c64,), dtype=cp.complex64)
                                         for _ in range(2)]
                self._scratch_out_gpu_cap = n_c64 * 8
            bufs = self._scratch_out_gpu
        return [b.view(dtype)[:n_elem].reshape(shape) for b in bufs]

    # ---------- pinned host-buffer cache — see TomoLarge for rationale
    def _get_fde(self, nz):
        """Return the pinned (nz, 2n, n+1) c64 fde buffer, split into
        FDE_N_BANDS chunks along the 2n (band_axis=1) axis so no single
        cudaHostAlloc exceeds the driver's per-call cap at large UPS.
        """
        shape = (nz, 2 * self.n, self.n + 1)
        if self._fde is None or self._fde.shape != shape:
            self._fde = BandedPinned(shape, np.complex64,
                                     n_bands=FDE_N_BANDS, band_axis=1)
        self._fde.fill(0)
        return self._fde

    def _get_sino(self, nz):
        shape = (self.ntheta, nz, self.n)
        if self._sino is None or self._sino.shape != shape:
            self._sino = alloc_pinned(shape, np.complex64)
        return self._sino

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

        chunks = [CHUNK_N, CHUNK_THETA, CHUNK_XY] — same knobs as TomoLarge.
        """
        chunk_n, chunk_theta, chunk_xy = chunks
        nz = obj.shape[0]

        fde  = self._get_fde(nz)
        sino = self._get_sino(nz)

        nel, idx, nx_bins = self._sort_into_chunks(chunk_xy)

        self._cos_theta_gpu = cp.asarray(self.cos_theta)
        self._sin_theta_gpu = cp.asarray(self.sin_theta)

        # Pipes and per-bin GPU buffers are cached across R() calls — see
        # TomoLarge.R for the rationale (avoid re-pinning tens of GB of
        # pipe buffers on every call).
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
        pin_in  = self._scratch_views('in',  in_shape,  np.float32)
        pin_out = self._scratch_views('out', out_shape, np.complex64)
        if (self._pipe1 is None
                or self._pipe1.in_shape  != in_shape
                or self._pipe1.out_shape != out_shape
                or self._pipe1.in_dtype  != np.dtype(np.float32)):
            self._pipe1 = StreamPipe(in_shape, out_shape,
                                     np.float32, np.complex64,
                                     pinned_in=pin_in, pinned_out=pin_out)
        else:
            self._pipe1.in_pin  = pin_in
            self._pipe1.out_pin = pin_out
        pipe = self._pipe1

        phi_scale = self.phi_scale
        phi1d_gpu = cp.asarray(self.phi1d)                    # (n,) f32

        def load(k, dst):
            st = k * chunk_n
            dst[:] = obj[:, st:st + chunk_n, :]

        def compute(k, in_gpu, out_gpu):
            st, end = k * chunk_n, (k + 1) * chunk_n
            phix = phi1d_gpu[st:end]                          # (chunk_n,) f32
            # Center-place phi*obj in the padded 2n buffer — same convention
            # as TomoLarge — so the resulting rfft matches the FFT of a
            # centered signal (X_c[k]).  Left-aligning at [0, n) would give
            # a spectrum that differs by a (-i)^k phase per sample, which
            # breaks parity with the complex64 gather.
            padded = cp.zeros((nz, chunk_n, 2 * n), dtype=cp.float32)
            # phi[i, j] = phi_scale · phi1d[i] · phi1d[j] — separable factor.
            padded[:, :, n // 2 : n // 2 + n] = (
                phi_scale * phix[None, :, None] * phi1d_gpu[None, None, :] * in_gpu
            )
            out_gpu[...] = cufft.rfft(padded, axis=-1)

        def store(k, src):
            st = k * chunk_n
            # Center-place along Y (offset by n//2) so the obj sits at
            # fde[:, n//2:3n//2, :] — same convention as TomoLarge's
            # Pass 1 and TomoReal's padded buffer.  Left-aligning at
            # [0, n) would leave a (-i)^fy phase in the Y spectrum.
            # BandedPinned dispatches this axis-1 write to the intersecting
            # band(s) — chunk_n typically divides band_rows so it lands in one.
            fde.copy_from(src, np.s_[:, n // 2 + st : n // 2 + st + chunk_n, :])

        pipe.run(load, compute, store, n // chunk_n)

    def _pass2_yfft(self, fde, chunk_n):
        """Pass 2 — pipelined complex FFT along y (chunked along x).

        The x axis stores only n+1 columns (rfft half spectrum); we chunk
        it into strips of `chunk_n` columns.  Full complex FFT of length
        2n along y, no c2dfftshift (raw fftfreq order).
        """
        n, nz = self.n, fde.shape[0]
        n_x = n + 1
        n_chunks = (n_x + chunk_n - 1) // chunk_n   # last chunk may be short

        # Simpler: process a fixed chunk size that divides n, then handle
        # the tail (the single column at kx = n) separately.  For clean
        # power-of-2 chunking we round chunk sizes down to a divisor of n.
        assert n % chunk_n == 0, \
            f"CHUNK_N={chunk_n} must divide n={n} (rfft x-axis strips)"
        n_full = n // chunk_n   # strips covering [0, n)

        shape = (nz, 2 * n, chunk_n)
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
            st = k * chunk_n
            # Full-y stripe crosses all fde bands — copy_to iterates them
            # into the contiguous scratch dst with no host-side temp.
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
        """Pass 3 — NUFFT gather via `gather_kernel_rfft`.

        Bin samples by rk_x (their column in the stored [0, n+1) half-
        spectrum).  For each bin, fetch a FULL-y x-strip
        fde[:, :, stx:endx] — a plain contiguous slice, no synthesis —
        and let the kernel do reflection with conj; because the strip
        covers all 2n rows, the reflected access rk_y = (2n - k1) % 2n
        is always in-range.  The kernel recomputes (x, y) from full_idx +
        cos_theta / sin_theta (see gather_kernel_rfft in kernels.py).
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
                fidx_d = cp.asarray(full_idx)                # int64

            grid, block = (int(np.ceil(nel_i / 1024)),), (1024,)

            def compute(zc, out_gpu,
                        _nel=nel_i, _fde=fde_d, _fidx=fidx_d,
                        _stx=stx, _endx=endx,
                        _grid=grid, _block=block):
                out_gpu[:_nel].fill(0)
                gather_kernel_rfft(_grid, _block,
                    (out_gpu[:_nel], _fde[zc],
                     _fidx, cos_theta_gpu, sin_theta_gpu,
                     m, mua, _nel, _stx, _endx, n))

            def store(zc, src_pinned, _nel=nel_i, _base=flat_base):
                sino_flat[_base + zc * n] = src_pinned[:_nel]

            gather_pipe.run(compute, store, nz)
            offset += nel_i

    def _pass4_ifft(self, sino, chunk_theta):
        """Pass 4 — pipelined r-axis IFFT + normalisation, then .real.

        Same c1dfftshift-based centred-spectrum → centred-sample IFFT as
        TomoLarge, but the output is copied out as **REAL float32**
        (imag part ≈ 0 for real obj input, and is discarded).
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
        pin_in  = self._scratch_views('in',  in_shape,  np.complex64)
        pin_out = self._scratch_views('out', out_shape, np.float32)
        if (self._pipe4 is None
                or self._pipe4.in_shape  != in_shape
                or self._pipe4.out_shape != out_shape
                or self._pipe4.out_dtype != np.dtype(np.float32)):
            self._pipe4 = StreamPipe(in_shape, out_shape,
                                     np.complex64, np.float32,
                                     pinned_in=pin_in, pinned_out=pin_out)
        else:
            self._pipe4.in_pin  = pin_in
            self._pipe4.out_pin = pin_out
        pipe = self._pipe4

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
