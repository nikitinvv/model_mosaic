"""ScratchMixin — grow-only pinned-host + GPU scratch pools plus a cached
StreamPipe registry, shared by TomoLarge/TomoLargeReal/PropagationLarge/
PaganinLarge.

The four "large" classes each maintain:
  * pinned-host ping-pong buffers for `StreamPipe` H2D/D2H (one pair
    keyed by 'in', one by 'out') — the passes run sequentially so the
    same bytes back every pass's ping-pong.
  * (optionally) GPU ping-pong buffers on the same pattern — used by
    Paganin, opt-in for the others.
  * a small set of cached `StreamPipe` objects keyed by pass slot (so
    the events/streams and, when applicable, the auto-allocated GPU
    buffers survive across calls).

Before this mixin, each class carried its own copy of the pool
bookkeeping and the "check-shape-or-rebuild" pass boilerplate.  Now
they inherit `ScratchMixin` and call:

    pipe = self._get_pipe('p1', in_shape, out_shape,
                          in_dtype, out_dtype)                 # CPU-only
    pipe = self._get_pipe('p1', ..., use_gpu_scratch=True)     # + GPU pool

`self._free_scratch()` empties every pool (pinned bytes, GPU bytes,
cufft plans, pipe cache) and returns the cupy blocks to the pool.
Callers' `free()` methods drop their persistent buffers (fde/sino/out
/psi/obj/...) and end with `self._free_scratch()`.
"""
from __future__ import annotations

import cupy as cp
import numpy as np

from processing.pipeline import StreamPipe, alloc_pinned


class ScratchMixin:
    """Provides grow-only pinned/GPU scratch pools + a StreamPipe cache.

    State (created lazily on first use):
      _scratch_pinned      : dict[name -> [uint8-pinned-buf, uint8-pinned-buf]]
      _scratch_pinned_cap  : dict[name -> int (bytes)]
      _scratch_gpu         : dict[name -> [c64-buf, c64-buf]]
      _scratch_gpu_cap     : dict[name -> int (bytes)]
      _pipes               : dict[name -> StreamPipe]
    """

    # ------------- lazy state init ---------------------------------------
    def _init_scratch(self):
        # Idempotent — subclasses may still touch these in their __init__
        # for readability; this method just fills in any that are missing.
        if not hasattr(self, '_scratch_pinned'):
            self._scratch_pinned = {}
            self._scratch_pinned_cap = {}
            self._scratch_gpu = {}
            self._scratch_gpu_cap = {}
            self._pipes = {}

    # ------------- pinned pool -------------------------------------------
    def _get_pinned(self, name, shape, dtype):
        """Return a 2-element list of pinned numpy views of the requested
        (shape, dtype) into the shared byte pool ``name``.  The pool grows
        (grow-only) on demand — a smaller subsequent request reuses the
        existing bytes.
        """
        self._init_scratch()
        dtp = np.dtype(dtype)
        n_elem = int(np.prod(shape))
        need = n_elem * dtp.itemsize
        if self._scratch_pinned_cap.get(name, 0) < need:
            self._scratch_pinned[name] = [alloc_pinned((need,), np.uint8)
                                          for _ in range(2)]
            self._scratch_pinned_cap[name] = need
        bufs = self._scratch_pinned[name]
        return [np.frombuffer(b, dtp, n_elem).reshape(shape) for b in bufs]

    # ------------- GPU pool ----------------------------------------------
    def _get_gpu(self, name, shape, dtype):
        """Return 2 cupy views (shape, dtype) into the shared GPU scratch
        pool ``name``.  Storage is a pair of complex64 buffers big enough
        for the largest pass; views reinterpret the bytes per-call.

        Growing the pool first drops every cached pipe's reference to the
        old buffer of the same slot ('in' → ``p.in_gpu = []``; 'out' →
        ``p.out_gpu = []``) so cupy can free the old memory before the
        new (bigger) allocation happens — keeps peak = max, not old+new.
        """
        self._init_scratch()
        dtp = np.dtype(dtype)
        n_elem = int(np.prod(shape))
        need = n_elem * dtp.itemsize
        # complex64 = 8 B; round up so any dtype's view fits.
        n_c64 = (need + 7) // 8
        if self._scratch_gpu_cap.get(name, 0) < n_c64 * 8:
            for p in self._pipes.values():
                if p is None:
                    continue
                if name == 'in':
                    p.in_gpu = []
                elif name == 'out':
                    p.out_gpu = []
            self._scratch_gpu[name] = None            # release before alloc
            self._scratch_gpu[name] = [cp.empty((n_c64,), dtype=cp.complex64)
                                       for _ in range(2)]
            self._scratch_gpu_cap[name] = n_c64 * 8
        bufs = self._scratch_gpu[name]
        return [b.view(dtype)[:n_elem].reshape(shape) for b in bufs]

    # ------------- pipe cache --------------------------------------------
    def _get_pipe(self, name, in_shape, out_shape, in_dtype, out_dtype,
                  use_gpu_scratch=False):
        """Return a cached `StreamPipe` for pass slot ``name`` — pinned
        ping-pong buffers come from the shared 'in'/'out' pinned pools;
        the GPU ping-pong buffers come from the shared 'in'/'out' GPU
        pools when ``use_gpu_scratch=True`` (Paganin), otherwise
        StreamPipe allocates its own.

        Rebuilds when the cached pipe's shape or dtype no longer matches
        (which forces new events/streams for the fresh sizes); reuses
        otherwise, only rewiring its buffer attributes to the fresh views
        from the shared pools (the underlying bytes may have grown).
        """
        self._init_scratch()
        in_shape  = tuple(in_shape)
        out_shape = tuple(out_shape)
        in_dtype  = np.dtype(in_dtype)
        out_dtype = np.dtype(out_dtype)

        pin_in  = self._get_pinned('in',  in_shape,  in_dtype)
        pin_out = self._get_pinned('out', out_shape, out_dtype)
        if use_gpu_scratch:
            gpu_in  = self._get_gpu('in',  in_shape,  in_dtype)
            gpu_out = self._get_gpu('out', out_shape, out_dtype)
        else:
            gpu_in = gpu_out = None

        pipe = self._pipes.get(name)
        rebuild = (pipe is None
                   or pipe.in_shape  != in_shape
                   or pipe.out_shape != out_shape
                   or pipe.in_dtype  != in_dtype
                   or pipe.out_dtype != out_dtype)
        if rebuild:
            self._pipes[name] = StreamPipe(
                in_shape, out_shape, in_dtype, out_dtype,
                pinned_in=pin_in, pinned_out=pin_out,
                gpu_in=gpu_in, gpu_out=gpu_out)
        else:
            pipe.in_pin  = pin_in
            pipe.out_pin = pin_out
            if use_gpu_scratch:
                pipe.in_gpu  = gpu_in
                pipe.out_gpu = gpu_out
        return self._pipes[name]

    # ------------- teardown -----------------------------------------------
    def _free_scratch(self):
        """Empty every pool: drop cached pipes (releasing their pinned + GPU
        refs), drop the pinned/GPU pool buffers, and return cupy's freed
        blocks to the pool.

        Call this at the end of the subclass's own `free()` (which is
        still responsible for releasing the persistent fde/sino/out/etc
        buffers).
        """
        self._init_scratch()
        self._pipes = {}
        self._scratch_pinned = {}
        self._scratch_pinned_cap = {}
        self._scratch_gpu = {}
        self._scratch_gpu_cap = {}
        cp.get_default_memory_pool().free_all_blocks()
