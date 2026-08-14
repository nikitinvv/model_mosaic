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
                    nbanks=8, rank=0, comm=None, chunks=None):
    """Rank 0 clears any prior master + creates VDS + all bank files.
    Other ranks compute the banking plan locally (deterministic in the
    params, so no bcast is needed).  Barrier at the end ensures the
    master + all bank files exist on the FS before any rank returns.

    `chunks` overrides the HDF5 chunk shape inside the bank files without
    touching the banking — see tomo_initx.  It does not affect the
    banking plan, so the non-root branch below stays chunk-agnostic."""
    if rank == 0:
        cleanup_h5(path)
        ctx = _initx(filename=path, shape=shape, dtype=dtype,
                     vchunks=vchunks, stype=stype, nbanks=nbanks,
                     chunks=chunks)
    else:
        # Deterministic plan — recompute the paths+sizes without
        # touching the filesystem.
        from iohdf5.dxchange_hdf5_chunks import _create_banking_plan
        sitems_idx = 0 if stype.lower().startswith("proj") else 1
        banks_filename_path, banks_size, _ = _create_banking_plan(
            filename=path, shape=shape, vchunks=vchunks,
            nbanks_per_svchunk=nbanks, sitems_idx=sitems_idx)
        ctx = {'banks_filename_path': banks_filename_path,
               'banks_size': banks_size}
    if comm is not None:
        comm.Barrier()
    return ctx


def vchunk_bytes(vchunks, dtype) -> int:
    return int(np.prod(vchunks)) * np.dtype(dtype).itemsize


def n_vchunks(shape, vchunks) -> int:
    return int(np.prod([-(-s // c) for s, c in zip(shape, vchunks)]))


def _hb(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} PB"


def describe_input(path: str) -> None:
    """Print shape + chunk + VDS-source-count for an existing input file.
    For VDS masters (`d.chunks` is None), peek into the first bank file
    to report the actual on-disk bank shape and HDF5 chunk shape.
    Call from rank 0 only (opens the file)."""
    import h5py
    if not os.path.isfile(path):
        print(f"  IN : {path}  (not present)", flush=True)
        return
    bank_shape = bank_chunks = None
    n_sources = None
    with h5py.File(path, "r") as f:
        if "exchange/data" not in f:
            print(f"  IN : {path}  (no /exchange/data)", flush=True)
            return
        d = f["exchange/data"]
        shape, dtype, chunks = d.shape, d.dtype, d.chunks
        is_virtual = d.is_virtual
        if is_virtual:
            sources = d.virtual_sources()
            n_sources = len(sources)
            src = sources[0]
            bank_path = src.file_name
            if not os.path.isabs(bank_path):
                bank_path = os.path.join(
                    os.path.dirname(os.path.abspath(path)), bank_path)
            try:
                with h5py.File(bank_path, "r") as bf:
                    bd = bf[src.dset_name]
                    bank_shape = bd.shape
                    bank_chunks = bd.chunks
            except (OSError, KeyError):
                pass
    total = int(np.prod(shape)) * dtype.itemsize
    print(f"  IN : {path}", flush=True)
    print(f"       shape={tuple(shape)} {dtype}  total={_hb(total)}", flush=True)
    if is_virtual:
        if bank_shape is not None:
            print(f"       VDS master → {n_sources} bank files  "
                  f"bank shape={bank_shape}  HDF5 chunk={bank_chunks}",
                  flush=True)
        else:
            print(f"       VDS master → {n_sources} bank files  "
                  f"(bank open failed)", flush=True)
    else:
        print(f"       HDF5 chunk={chunks}", flush=True)


def describe_output(path: str, shape, dtype, vchunks, stype: str,
                    nbanks: int, chunks=None) -> None:
    """Print planned shape / vchunks / bank layout / HDF5 chunk for an
    output file about to be created via initx_and_bcast.  Call from
    rank 0 only.  Pure math — does not touch the filesystem."""
    dtp = np.dtype(dtype)
    sitems_idx = 0 if stype.lower().startswith("proj") else 1
    sitems_per_bank = (vchunks[sitems_idx] + nbanks - 1) // nbanks
    nsvchunks = (shape[sitems_idx] + vchunks[sitems_idx] - 1) // vchunks[sitems_idx]
    total_banks = nsvchunks * nbanks

    bank_shape = list(vchunks)
    bank_shape[sitems_idx] = sitems_per_bank
    bank_shape = tuple(bank_shape)

    if chunks is not None:
        h5chunk = tuple(min(c, s) for c, s in zip(chunks, bank_shape))
    elif stype.lower().startswith("proj"):
        h5chunk = (1,) + tuple(vchunks[1:])
    else:
        h5chunk = (vchunks[0], 1, vchunks[2])

    total = int(np.prod(shape)) * dtp.itemsize
    bank_bytes = int(np.prod(bank_shape)) * dtp.itemsize
    chunk_bytes = int(np.prod(h5chunk)) * dtp.itemsize
    print(f"  OUT: {path}", flush=True)
    print(f"       shape={tuple(shape)} {dtp}  total={_hb(total)}", flush=True)
    print(f"       vchunks={tuple(vchunks)}  stype={stype}  nbanks={nbanks}",
          flush=True)
    print(f"       → {nsvchunks} super-chunks × {nbanks} banks = "
          f"{total_banks} bank files", flush=True)
    sino_ordered = h5chunk[0] > 1 and h5chunk[1] < bank_shape[1]
    print(f"       bank shape={bank_shape} ({_hb(bank_bytes)})  "
          f"HDF5 chunk={h5chunk} ({_hb(chunk_bytes)})"
          f"{'  [sinogram-ordered]' if sino_ordered else ''}", flush=True)
