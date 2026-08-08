"""Thin conveniences on top of dxchange_hdf5_chunks for the mosaic steps.

The pipeline scripts (step1_upsample, step2_model_*, upsample_extract)
all follow the same pattern from test_h5_buffer_io.py:

  1. Rank 0 calls `tomo_initx` to create a VDS master + nvchunks·nbanks
     empty bank files.  Ctx is broadcast to the other ranks.
  2. Each rank allocates one shared-memory `vchunk` (super-chunk) buffer
     of shape --XXX-vchunks.
  3. Ranks iterate `_iter_vchunks(shape, vchunks)[RANK::SIZE]`,
     filling the buffer with per-compute-chunk outputs, then calling
     `tomo_writex` to fan the buffer across nbanks bank files in parallel.
  4. Reads go through the VDS master via plain h5py (transparent VDS
     resolution) — simpler than tomo_readx and no worker pool needed
     since the compute is usually the bottleneck, not the read.

Helpers here just deduplicate the SHM setup, ivchunk iteration, and
init/broadcast/cleanup boilerplate.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
from multiprocessing import shared_memory

from iohdf5.dxchange_hdf5_chunks import tomo_initx as _initx


def cleanup_h5(path: str) -> None:
    """Remove a VDS master and its sibling bank directory (name/name_*.h5)."""
    if os.path.isfile(path):
        os.remove(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    bank_dir = os.path.join(os.path.dirname(path) or ".", stem)
    if os.path.isdir(bank_dir):
        shutil.rmtree(bank_dir)


def iter_vchunks(shape, vchunks):
    """Yield ivchunk tuples in (proj-major, x-outer, y-inner) order —
    same order as doe_chunks_t4_aps.ipynb / test_h5_buffer_io.py."""
    n0 = (shape[0] + vchunks[0] - 1) // vchunks[0]
    n1 = (shape[1] + vchunks[1] - 1) // vchunks[1]
    n2 = (shape[2] + vchunks[2] - 1) // vchunks[2]
    for i0 in range(n0):
        for i2 in range(n2):
            for i1 in range(n1):
                yield (i0, i1, i2)


def alloc_shm(shape, dtype):
    """Allocate a shared-memory buffer of the given shape+dtype.  Returns
    (shm, ndarray-view).  Caller must call `free_shm(shm)` when done."""
    dtp = np.dtype(dtype)
    shm = shared_memory.SharedMemory(create=True,
                                     size=int(np.prod(shape)) * dtp.itemsize)
    buf = np.ndarray(shape=shape, dtype=dtp, buffer=shm.buf)
    return shm, buf


def free_shm(shm) -> None:
    try:
        shm.close()
    finally:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def initx_and_bcast(path, shape, dtype, vchunks, stype="proj",
                    nbanks=8, meta=None, rank=0, comm=None):
    """Rank 0 clears any prior master + creates VDS + all bank files.
    Other ranks compute the banking plan locally (deterministic in the
    params, so no bcast is needed).  Barrier at the end ensures the
    master + all bank files exist on the FS before any rank returns."""
    if rank == 0:
        cleanup_h5(path)
        ctx = _initx(filename=path, shape=shape, dtype=dtype,
                     vchunks=vchunks, stype=stype, nbanks=nbanks,
                     meta=meta or {})
    else:
        # Deterministic plan — recompute the paths+sizes without
        # touching the filesystem.
        from iohdf5.dxchange_hdf5_chunks import _create_banking_plan
        sitems_idx = 0 if stype.lower().startswith("proj") else 1
        banks_filename_path, banks_size, _ = _create_banking_plan(
            filename=path, shape=shape, vchunks=vchunks,
            nbanks_per_svchunk=nbanks, sitems_idx=sitems_idx,
            meta=meta or {})
        ctx = {'banks_filename_path': banks_filename_path,
               'banks_size': banks_size}
    if comm is not None:
        comm.Barrier()
    return ctx


def vchunk_bytes(vchunks, dtype) -> int:
    return int(np.prod(vchunks)) * np.dtype(dtype).itemsize


def n_vchunks(shape, vchunks) -> int:
    return int(np.prod([-(-s // c) for s, c in zip(shape, vchunks)]))
