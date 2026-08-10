"""Small helpers for 3-stream H2D / compute / D2H pipelining of chunked
GPU loops (as in tomo_large.py's FFT passes and propagation_large.py's
Fresnel passes).

The pattern:
  for k in 0..n_iter-1:
     load     : copy host chunk k    → pinned_in[k%2]  (main thread)
     h2d      : pinned_in[k%2]       → gpu_in[k%2]     (stream s_h2d)
     compute  : gpu_in[k%2]          → gpu_out[k%2]    (stream s_comp)
     d2h      : gpu_out[k%2]         → pinned_out[k%2] (stream s_d2h)
     store    : pinned_out[k%2]      → host chunk k    (main thread)

With two ping-pong buffers per side and dedicated streams, the three
GPU stages overlap fully once the pipeline is primed.

`StreamPipe.run(load, compute, store, n_iter)` schedules this loop; the
three callbacks are what the caller has to write (5-10 lines each).

Pinned memory is allocated once per pipe and pooled via cupy's global
pinned allocator, so repeated setups (e.g. one pipe per FFT pass, all
called at the same in/out shape) reuse the same host pages.
"""
from __future__ import annotations

import cupy as cp
import numpy as np


# Cupy uses this global pinned allocator; setting it once at import
# time makes cp.cuda.alloc_pinned_memory pool the allocations.  Keep a
# module-level reference so callers can drain the pool back to the OS
# between phases (via `free_pinned_pool()`) — otherwise pinned buffers
# from earlier work sit around and starve later large allocations.
_PINNED_POOL = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(_PINNED_POOL.malloc)

# Cap the cuFFT plan cache to bound plan-workspace memory.  Each plan
# holds its own scratch buffer proportional to the transform size, and
# the default cache retains everything ever planned (observed 16+ plans
# at UPS=4, each with 0.1–1 GB workspace on non-innermost axes).  4 slots
# fit our 4-pass pipeline (rfft x, fft y, tail fft, ifft r) with no
# eviction; anything more just retains stale entries.
try:
    cp.fft.config.get_plan_cache().set_size(4)
except Exception:
    pass


def free_pinned_pool():
    """Release all cached pinned-host blocks back to the OS."""
    _PINNED_POOL.free_all_blocks()


def alloc_pinned(shape, dtype):
    """Allocate a pinned host numpy array of the given shape+dtype."""
    dtp = np.dtype(dtype)
    n = int(np.prod(shape))
    mem = cp.cuda.alloc_pinned_memory(n * dtp.itemsize)
    return np.frombuffer(mem, dtp, n).reshape(shape)


class BandedPinned:
    """A logical (shape, dtype) array split across `n_bands` separate
    pinned-host allocations along `band_axis` — used when a single
    `cudaHostAlloc` for the full array would exceed the driver's
    per-allocation cap (that's ~64 GiB on some boxes, larger on others).

    Total pinned bytes are the same as one big allocation; the split
    just ducks under the per-call ceiling.  `shape[band_axis]` must be
    divisible by `n_bands`.

    Supports the subset of numpy semantics the tomo/prop passes use:

      * `.fill(v)`, `.nbytes`, `.shape`, `.dtype`, `.ndim`
      * `self[key]`  — slice/int key.  A `slice(None)` (or explicit
        full-span slice) on `band_axis` stitches all bands into a fresh
        ndarray via `np.concatenate`; a slice landing wholly inside one
        band returns a view; a scalar returns a view of that row.
      * `self[key] = val` — dispatches writes to the intersecting bands.
        `val` may be a scalar (broadcast) or an ndarray whose
        `shape[band_axis]` matches the slice length.
      * `.copy_to(dst, key)` — like `dst[:] = self[key]` but iterates
        bands directly into `dst` with no intermediate allocation.
      * `.copy_from(src, key)` — reverse of copy_to.
      * `.copy_to_gpu(dst_gpu, key)` — same as copy_to but destination
        is a cupy array; each band does its own pinned→GPU H2D.

    Fancy indexing, mixed keys, and step != 1 slices are not supported
    (raise).  Not a general ndarray replacement.
    """

    def __init__(self, shape, dtype, n_bands, band_axis=1):
        self.shape     = tuple(shape)
        self.dtype     = np.dtype(dtype)
        self.ndim      = len(self.shape)
        self.band_axis = band_axis
        if self.shape[band_axis] % n_bands != 0:
            raise ValueError(
                f"BandedPinned: shape[{band_axis}]={self.shape[band_axis]} "
                f"not divisible by n_bands={n_bands}")
        self.n_bands   = n_bands
        self.band_rows = self.shape[band_axis] // n_bands
        band_shape     = list(self.shape)
        band_shape[band_axis] = self.band_rows
        self.bands     = [alloc_pinned(tuple(band_shape), dtype)
                          for _ in range(n_bands)]

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __array__(self, dtype=None):
        """Materialize as a contiguous ndarray by stitching bands.  Cheap
        only for small arrays — this allocates the full (unbanded) size, so
        never trigger it on hundreds-of-GB banded buffers in real pipelines.
        Provided so numpy ops like ``np.abs(banded - ref)`` in parity tests
        transparently work; production code should use copy_to / copy_from /
        copy_to_gpu instead."""
        a = np.concatenate(self.bands, axis=self.band_axis)
        return a if dtype is None else a.astype(dtype)

    def fill(self, val):
        for b in self.bands:
            b.fill(val)

    def _normalize_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # Pad with trailing full slices to reach ndim.
        return key + (slice(None),) * (self.ndim - len(key))

    def _band_span(self, key_on_band_axis):
        """Given the key entry on band_axis, return (st, end, single_band_or_None)."""
        n0 = self.shape[self.band_axis]
        if isinstance(key_on_band_axis, slice):
            st, end, step = key_on_band_axis.indices(n0)
            if step != 1:
                raise NotImplementedError("BandedPinned only supports step=1 slices")
            return st, end, False
        if isinstance(key_on_band_axis, (int, np.integer)):
            i = int(key_on_band_axis)
            if i < 0: i += n0
            return i, i + 1, True
        raise NotImplementedError(
            f"BandedPinned key type {type(key_on_band_axis).__name__} not supported")

    def _bands_covering(self, st, end):
        """Yield (band_idx, local_lo, local_hi, dst_lo, dst_hi) for bands
        intersecting [st, end) on band_axis."""
        b0 = st // self.band_rows
        b1 = (end - 1) // self.band_rows + 1
        for bi in range(b0, b1):
            base = bi * self.band_rows
            lo   = max(base, st)
            hi   = min(base + self.band_rows, end)
            yield bi, lo - base, hi - base, lo - st, hi - st

    def __getitem__(self, key):
        key = self._normalize_key(key)
        st, end, scalar_axis = self._band_span(key[self.band_axis])
        parts = []
        for bi, lo, hi, _, _ in self._bands_covering(st, end):
            idx = list(key)
            idx[self.band_axis] = (lo if scalar_axis else slice(lo, hi))
            parts.append(self.bands[bi][tuple(idx)])
        if len(parts) == 1:
            return parts[0]
        # multi-band → allocate + concat.  Only place BandedPinned actually
        # materialises a full-width copy; the pass code prefers copy_to.
        return np.concatenate(parts, axis=self.band_axis)

    def __setitem__(self, key, val):
        key = self._normalize_key(key)
        st, end, scalar_axis = self._band_span(key[self.band_axis])
        scalar_val = np.isscalar(val)
        for bi, lo, hi, s_lo, s_hi in self._bands_covering(st, end):
            dst_idx = list(key)
            dst_idx[self.band_axis] = (lo if scalar_axis else slice(lo, hi))
            if scalar_val:
                self.bands[bi][tuple(dst_idx)] = val
            else:
                src_idx = [slice(None)] * self.ndim
                src_idx[self.band_axis] = slice(s_lo, s_hi)
                self.bands[bi][tuple(dst_idx)] = val[tuple(src_idx)]

    def copy_to(self, dst, key):
        """Zero-alloc read: `dst[:] = self[key]` iterated per band."""
        key = self._normalize_key(key)
        st, end, _ = self._band_span(key[self.band_axis])
        for bi, lo, hi, d_lo, d_hi in self._bands_covering(st, end):
            src_idx = list(key)
            src_idx[self.band_axis] = slice(lo, hi)
            dst_idx = [slice(None)] * self.ndim
            dst_idx[self.band_axis] = slice(d_lo, d_hi)
            dst[tuple(dst_idx)] = self.bands[bi][tuple(src_idx)]

    def copy_from(self, src, key):
        """Zero-alloc write: `self[key] = src` iterated per band."""
        key = self._normalize_key(key)
        st, end, _ = self._band_span(key[self.band_axis])
        for bi, lo, hi, s_lo, s_hi in self._bands_covering(st, end):
            dst_idx = list(key)
            dst_idx[self.band_axis] = slice(lo, hi)
            src_idx = [slice(None)] * self.ndim
            src_idx[self.band_axis] = slice(s_lo, s_hi)
            self.bands[bi][tuple(dst_idx)] = src[tuple(src_idx)]

    def copy_to_gpu(self, dst_gpu, key):
        """H2D read into a cupy array; each band uploads its slice via
        cp.asarray (goes through pinned) then assigns into the strided
        destination slice of dst_gpu.  Two-hop (H2D contiguous + GPU
        strided copy) is necessary because dst_gpu's slice on band_axis
        is non-contiguous and cupy.ndarray.set requires contiguous dst."""
        key = self._normalize_key(key)
        st, end, _ = self._band_span(key[self.band_axis])
        for bi, lo, hi, d_lo, d_hi in self._bands_covering(st, end):
            src_idx = list(key)
            src_idx[self.band_axis] = slice(lo, hi)
            dst_idx = [slice(None)] * self.ndim
            dst_idx[self.band_axis] = slice(d_lo, d_hi)
            src_np = np.ascontiguousarray(self.bands[bi][tuple(src_idx)])
            dst_gpu[tuple(dst_idx)] = cp.asarray(src_np)


class StreamPipe:
    """3-stream H2D / compute / D2H pipeline with double-buffered
    pinned+GPU pairs.

    Parameters
    ----------
    in_shape, out_shape : tuple[int, ...]
        Shape of one CHUNK on the input and output sides (per iter).
    in_dtype, out_dtype : numpy dtype
        Corresponding dtypes.
    """

    def __init__(self, in_shape, out_shape, in_dtype, out_dtype,
                 pinned_in=None, pinned_out=None,
                 gpu_in=None, gpu_out=None):
        # Kept so callers can cache pipes and check-and-reuse across calls
        # (the pinned + GPU ping-pong buffers survive as long as the pipe
        # object does; recreating them is expensive at large chunk sizes).
        self.in_shape  = tuple(in_shape)
        self.out_shape = tuple(out_shape)
        self.in_dtype  = np.dtype(in_dtype)
        self.out_dtype = np.dtype(out_dtype)
        # Two ping-pong buffers per side.  If the caller passes pre-
        # allocated pinned or GPU views (e.g. sliced from a shared scratch
        # pool), use them verbatim; otherwise allocate fresh.
        if pinned_in is None:
            self.in_pin  = [alloc_pinned(in_shape,  in_dtype)  for _ in range(2)]
        else:
            self.in_pin  = list(pinned_in)
        if pinned_out is None:
            self.out_pin = [alloc_pinned(out_shape, out_dtype) for _ in range(2)]
        else:
            self.out_pin = list(pinned_out)
        if gpu_in is None:
            self.in_gpu  = [cp.empty(in_shape,  dtype=in_dtype)  for _ in range(2)]
        else:
            self.in_gpu  = list(gpu_in)
        if gpu_out is None:
            self.out_gpu = [cp.empty(out_shape, dtype=out_dtype) for _ in range(2)]
        else:
            self.out_gpu = list(gpu_out)

        # Three streams — one per stage.
        self.s_h2d  = cp.cuda.Stream(non_blocking=True)
        self.s_comp = cp.cuda.Stream(non_blocking=True)
        self.s_d2h  = cp.cuda.Stream(non_blocking=True)

        # Events (double-buffered) — record on each stream, wait on the
        # downstream one to enforce H2D → compute → D2H order per buffer.
        self.h2d_evt  = [cp.cuda.Event() for _ in range(2)]
        self.comp_evt = [cp.cuda.Event() for _ in range(2)]
        self.d2h_evt  = [cp.cuda.Event() for _ in range(2)]

    def run(self, load, compute, store, n_iter):
        """Run the pipeline for `n_iter` chunks.

        Callbacks:
          load(k, dst_pinned)          — main-thread numpy → pinned copy
          compute(k, in_gpu, out_gpu)  — runs inside `with self.s_comp:`
                                          (may allocate/mutate as needed)
          store(k, src_pinned)         — main-thread pinned → numpy copy
        """
        for k in range(n_iter + 3):
            b_h2d = k % 2

            # 1. numpy → pinned  (chunk k)
            if k < n_iter:
                load(k, self.in_pin[b_h2d])
                # 2. pinned → gpu   (chunk k)
                with self.s_h2d:
                    if k >= 2:
                        self.s_h2d.wait_event(self.comp_evt[b_h2d])
                    self.in_gpu[b_h2d].set(self.in_pin[b_h2d],
                                            stream=self.s_h2d)
                    self.h2d_evt[b_h2d].record(self.s_h2d)

            # 3. compute (chunk k-1)
            if 0 < k <= n_iter:
                j = (k - 1) % 2
                with self.s_comp:
                    self.s_comp.wait_event(self.h2d_evt[j])
                    if k >= 3:
                        # ensure previous D2H released out_gpu[j]
                        self.s_comp.wait_event(self.d2h_evt[j])
                    compute(k - 1, self.in_gpu[j], self.out_gpu[j])
                    self.comp_evt[j].record(self.s_comp)

            # 4. gpu → pinned    (chunk k-2)
            if 1 < k <= n_iter + 1:
                j = (k - 2) % 2
                with self.s_d2h:
                    self.s_d2h.wait_event(self.comp_evt[j])
                    self.out_gpu[j].get(out=self.out_pin[j],
                                        stream=self.s_d2h, blocking=False)
                    self.d2h_evt[j].record(self.s_d2h)

            # 5. pinned → numpy  (chunk k-3)
            if 2 < k <= n_iter + 2:
                j = (k - 3) % 2
                self.d2h_evt[j].synchronize()
                store(k - 3, self.out_pin[j])

        self.s_h2d.synchronize()
        self.s_comp.synchronize()
        self.s_d2h.synchronize()


class ComputeD2HPipe:
    """2-stage (compute ‖ D2H) pipeline for loops whose input is already
    GPU-resident (or trivially produced) — no H2D stage.

    Use case: TomoLarge's gather over z-slices.  For each zc the same
    (fde_d, x0, y0) inputs are reused (uploaded once per bin outside
    the loop), the kernel writes into a small output buffer, and we
    D2H that buffer to pinned host then scatter into the host sino
    array via flat indexing.

    Callbacks:
      compute(k, out_gpu)   — runs inside `with self.s_comp:`
      store(k, src_pinned)  — main-thread scatter/write

    Buffers are sized to `out_shape` once at __init__ and reused across
    every call to run() (which itself works over `n_iter` iterations).
    """

    def __init__(self, out_shape, out_dtype):
        self.out_shape = tuple(out_shape)
        self.out_dtype = np.dtype(out_dtype)
        self.out_gpu = [cp.empty(out_shape, dtype=out_dtype) for _ in range(2)]
        self.out_pin = [alloc_pinned(out_shape, out_dtype) for _ in range(2)]
        self.s_comp  = cp.cuda.Stream(non_blocking=True)
        self.s_d2h   = cp.cuda.Stream(non_blocking=True)
        self.comp_evt = [cp.cuda.Event() for _ in range(2)]
        self.d2h_evt  = [cp.cuda.Event() for _ in range(2)]

    def run(self, compute, store, n_iter):
        for k in range(n_iter + 2):
            b = k % 2
            # Stage 1: compute (k)
            if k < n_iter:
                with self.s_comp:
                    if k >= 2:
                        # ensure previous D2H released this out buffer
                        self.s_comp.wait_event(self.d2h_evt[b])
                    compute(k, self.out_gpu[b])
                    self.comp_evt[b].record(self.s_comp)
            # Stage 2: D2H (k-1)
            if 0 < k <= n_iter:
                prev = (k - 1) % 2
                with self.s_d2h:
                    self.s_d2h.wait_event(self.comp_evt[prev])
                    self.out_gpu[prev].get(
                        out=self.out_pin[prev],
                        stream=self.s_d2h, blocking=False)
                    self.d2h_evt[prev].record(self.s_d2h)
            # Stage 3: store (k-2)
            if k > 1:
                prev = (k - 2) % 2
                self.d2h_evt[prev].synchronize()
                store(k - 2, self.out_pin[prev])
        self.s_comp.synchronize()
        self.s_d2h.synchronize()
