"""MPI-IO slab helpers.

MPI's C API uses a signed 32-bit int for the element count in
MPI_File_read_at_all / MPI_File_write_at_all, so any single collective
hyperslab whose byte-size exceeds ~2^31 fails (ROMIO returns an error or
silently truncates on some stacks).  Same convention as
holotomocupy_mpi_deform/src/holotomocupy/reader.py:391-402 and
writer.py:114: cap every collective transfer at 1<<28 bytes (256 MB,
~8x margin under the 2 GiB limit).

These helpers loop over axis 0 in slabs no larger than MAX_MPIIO_BYTES.
Non-MPI (POSIX) datasets are also fine — the loop just executes once.
"""
from __future__ import annotations

import numpy as np


MAX_MPIIO_BYTES = 1 << 28  # 256 MB
H5_MAX_CHUNK_BYTES = 1 << 32  # h5py hard 4 GiB per-chunk limit


def check_chunk_bytes(chunks: tuple, dtype_bytes: int, label: str = "") -> None:
    """Raise if a chunk shape would exceed h5py's 4 GiB per-chunk limit."""
    b = int(np.prod(chunks)) * dtype_bytes
    if b >= H5_MAX_CHUNK_BYTES:
        raise SystemExit(
            f"{label or 'chunk'} bytes = {b/1e9:.2f} GB exceeds h5py 4 GiB "
            f"per-chunk hard limit (chunks={chunks}, itemsize={dtype_bytes})")


def _slab_step(tail_shape, dtype_bytes: int, max_bytes: int) -> int:
    per = int(np.prod(tail_shape)) * dtype_bytes
    return max(1, max_bytes // max(1, per))


def mpiio_write_axis0(dset, i0: int, i1: int, arr: np.ndarray,
                      max_bytes: int = MAX_MPIIO_BYTES) -> None:
    """Write arr into dset[i0:i1, ...] in slabs <= max_bytes along axis 0.

    Preconditions: arr.shape[0] == i1 - i0 and arr.shape[1:] == dset.shape[1:].
    """
    n = i1 - i0
    if n == 0:
        return
    step = _slab_step(dset.shape[1:], arr.dtype.itemsize, max_bytes)
    for s in range(0, n, step):
        e = min(s + step, n)
        dset[i0 + s:i0 + e] = arr[s:e]


def mpiio_read_axis0(dset, i0: int, i1: int,
                     max_bytes: int = MAX_MPIIO_BYTES) -> np.ndarray:
    """Read dset[i0:i1, ...] into a new array in slabs <= max_bytes."""
    n = i1 - i0
    tail = dset.shape[1:]
    out = np.empty((n,) + tail, dtype=dset.dtype)
    if n == 0:
        return out
    step = _slab_step(tail, dset.dtype.itemsize, max_bytes)
    for s in range(0, n, step):
        e = min(s + step, n)
        out[s:e] = dset[i0 + s:i0 + e]
    return out


def mpiio_write_slab(dset, index: tuple, arr: np.ndarray,
                     max_bytes: int = MAX_MPIIO_BYTES) -> None:
    """Write arr into dset[i_slice, *tail_slices] slab-safely along axis 0.

    index is (slice_over_axis0, *fixed_or_slice_tail).  The axis-0 slice is
    split; tail indexing is passed through verbatim.  Used when the
    destination has non-trivial tail slicing (e.g. z-band writes into
    proj.h5 like proj_dset[tb0:tb1, z0:z1, :]).
    """
    axis0 = index[0]
    tail  = index[1:]
    if not isinstance(axis0, slice):
        # Single-plane write, no batching needed.
        dset[index] = arr
        return
    i0 = 0 if axis0.start is None else int(axis0.start)
    i1 = dset.shape[0] if axis0.stop is None else int(axis0.stop)
    n = i1 - i0
    if n == 0:
        return
    # Estimate per-slab bytes from arr's tail (matches the actual transfer).
    step = _slab_step(arr.shape[1:], arr.dtype.itemsize, max_bytes)
    for s in range(0, n, step):
        e = min(s + step, n)
        dset[(slice(i0 + s, i0 + e),) + tail] = arr[s:e]


def mpiio_read_slab(dset, index: tuple,
                    max_bytes: int = MAX_MPIIO_BYTES) -> np.ndarray:
    """Read dset[i_slice, *tail_slices] slab-safely along axis 0."""
    axis0 = index[0]
    tail  = index[1:]
    if not isinstance(axis0, slice):
        return dset[index]
    i0 = 0 if axis0.start is None else int(axis0.start)
    i1 = dset.shape[0] if axis0.stop is None else int(axis0.stop)
    n = i1 - i0
    # Determine output tail shape by peeking the first row (cheap for h5).
    # Faster: reason from index + dset.shape.
    tail_shape = []
    for ax, ix in enumerate(tail, start=1):
        if isinstance(ix, slice):
            start, stop, stride = ix.indices(dset.shape[ax])
            tail_shape.append(max(0, (stop - start + (stride - (1 if stride > 0 else -1))) // stride))
        elif isinstance(ix, (list, np.ndarray)):
            tail_shape.append(len(ix))
        # scalar index removes the axis
    out = np.empty((n, *tail_shape), dtype=dset.dtype)
    if n == 0:
        return out
    step = _slab_step(tuple(tail_shape), dset.dtype.itemsize, max_bytes)
    for s in range(0, n, step):
        e = min(s + step, n)
        out[s:e] = dset[(slice(i0 + s, i0 + e),) + tail]
    return out
