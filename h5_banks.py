"""Per-rank bank-file HDF5 layout with a top-level Virtual Dataset.

Replaces parallel-HDF5 collective writes with independent per-rank POSIX
writes into rank-owned bank files, stitched together by an h5py VDS master
that reads transparently like a single file.

Disk layout for one dataset:
    /path/name.h5                        # VDS master (rank 0 writes)
    /path/name/name_data_000000.h5       # bank owned by rank 0
    /path/name/name_data_000001.h5       # bank owned by rank 1
    ...

Ownership axis is user-picked (0 or 1).  Rank R owns a contiguous slab
[start_R, end_R) along that axis and its bank file has that shape.

Reads: `open_reader()` opens the master; VDS routes to bank files.
Writes: `open_writer()` opens the rank's own bank file; only that rank
        writes there — no coordination needed.

The `Accumulator` helper batches several small compute outputs along the
banking axis into one bigger write, cutting h5py write() call count when
each compute step's output is much smaller than an h5 chunk.
"""
from __future__ import annotations

import os
import shutil

import h5py
import numpy as np


def _default_ranges(n: int, size: int) -> list[tuple[int, int]]:
    per = (n + size - 1) // size
    return [(min(r * per, n), min((r + 1) * per, n)) for r in range(size)]


def _bank_name(stem: str, rank: int) -> str:
    return f"{stem}_data_{rank:06d}.h5"


class BankedH5:
    """One master VDS + one bank file per rank for a single dataset."""

    def __init__(self, master_path, shape, dtype, axis, chunks,
                 rank: int = 0, size: int = 1, comm=None,
                 bank_ranges=None, rdcc_mb: int = 128):
        self.master  = str(master_path)
        self.shape   = tuple(int(x) for x in shape)
        self.dtype   = np.dtype(dtype)
        self.axis    = int(axis)
        self.chunks  = tuple(int(x) for x in chunks)
        self.rank    = int(rank)
        self.size    = int(size)
        self.comm    = comm
        self.rdcc_bytes = int(rdcc_mb) * (1 << 20)

        assert 0 <= self.axis < len(self.shape), \
            f"axis {self.axis} out of range for shape {self.shape}"
        assert len(self.chunks) == len(self.shape), \
            f"chunks {self.chunks} rank mismatch shape {self.shape}"

        if bank_ranges is None:
            bank_ranges = _default_ranges(self.shape[self.axis], self.size)
        self.bank_ranges = [tuple(r) for r in bank_ranges]
        assert len(self.bank_ranges) == self.size
        self.my_start, self.my_end = self.bank_ranges[self.rank]
        self.my_extent = self.my_end - self.my_start

        stem = os.path.splitext(os.path.basename(self.master))[0]
        self.stem = stem
        self.bank_dir = os.path.join(os.path.dirname(self.master) or ".", stem)
        self.bank_path = os.path.join(self.bank_dir, _bank_name(stem, self.rank))

    # ---------------- bootstrap ----------------------------------------
    def create(self, extra_datasets=None) -> None:
        """Rank 0 clears any prior master+bank_dir, then every rank creates
        its own bank file, then rank 0 stitches the VDS master."""
        if self.rank == 0:
            if os.path.isfile(self.master):
                os.remove(self.master)
            if os.path.isdir(self.bank_dir):
                shutil.rmtree(self.bank_dir)
            os.makedirs(self.bank_dir, exist_ok=True)
        self._barrier()

        if self.my_extent > 0:
            bank_shape = list(self.shape)
            bank_shape[self.axis] = self.my_extent
            bank_shape = tuple(bank_shape)
            bank_chunks = tuple(min(c, s) for c, s in zip(self.chunks, bank_shape))
            with h5py.File(self.bank_path, "w", libver="latest") as f:
                g = f.create_group("exchange")
                g.create_dataset("data", shape=bank_shape,
                                 dtype=self.dtype, chunks=bank_chunks)

        self._barrier()

        if self.rank == 0:
            layout = h5py.VirtualLayout(shape=self.shape, dtype=self.dtype)
            for r, (start, end) in enumerate(self.bank_ranges):
                if end <= start:
                    continue
                bshape = list(self.shape)
                bshape[self.axis] = end - start
                rel = os.path.join(self.stem, _bank_name(self.stem, r))
                vsrc = h5py.VirtualSource(rel, "/exchange/data",
                                          shape=tuple(bshape),
                                          dtype=self.dtype)
                sel = [slice(None)] * len(self.shape)
                sel[self.axis] = slice(start, end)
                layout[tuple(sel)] = vsrc
            with h5py.File(self.master, "w", libver="latest") as f:
                g = f.create_group("exchange")
                g.create_virtual_dataset("data", layout, fillvalue=0)
                if extra_datasets:
                    for name, data in extra_datasets.items():
                        g.create_dataset(name, data=data)

        self._barrier()

    def _barrier(self) -> None:
        if self.comm is not None:
            self.comm.Barrier()

    # ---------------- open helpers -------------------------------------
    def open_writer(self) -> "BankWriter":
        return BankWriter(self)

    def open_reader(self) -> "BankReader":
        return BankReader(self)

    # ---------------- diagnostics --------------------------------------
    def chunk_bytes(self) -> int:
        return int(np.prod(self.chunks)) * self.dtype.itemsize


class BankWriter:
    """Context manager: opens this rank's bank file for repeated writes."""

    def __init__(self, banked: BankedH5):
        self.b = banked
        self._h = None
        self._d = None

    def __enter__(self):
        if self.b.my_extent > 0:
            self._h = h5py.File(self.b.bank_path, "r+", libver="latest",
                                rdcc_nbytes=self.b.rdcc_bytes)
            self._d = self._h["/exchange/data"]
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.close()
            self._h, self._d = None, None
        return False

    def write(self, dest_slice, arr) -> None:
        """Write `arr` to global position `dest_slice`.  The banked-axis
        slice in dest_slice must lie inside this rank's [start, end)."""
        if self.b.my_extent == 0:
            return
        local = list(dest_slice)
        ax = local[self.b.axis]
        assert isinstance(ax, slice), \
            f"banked axis {self.b.axis} must be a slice"
        start = 0 if ax.start is None else int(ax.start)
        stop  = self.b.shape[self.b.axis] if ax.stop is None else int(ax.stop)
        assert start >= self.b.my_start and stop <= self.b.my_end, (
            f"rank {self.b.rank} write on axis {self.b.axis} "
            f"[{start}, {stop}) exceeds bank range "
            f"[{self.b.my_start}, {self.b.my_end})")
        local[self.b.axis] = slice(start - self.b.my_start,
                                    stop  - self.b.my_start)
        self._d[tuple(local)] = arr


class BankReader:
    """Context manager: opens the VDS master for reads."""

    def __init__(self, banked: BankedH5):
        self.b = banked
        self._h = None
        self._d = None

    def __enter__(self):
        self._h = h5py.File(self.b.master, "r", libver="latest",
                            rdcc_nbytes=self.b.rdcc_bytes)
        self._d = self._h["/exchange/data"]
        return self

    def __exit__(self, *exc):
        if self._h is not None:
            self._h.close()
            self._h, self._d = None, None
        return False

    def read(self, src_slice):
        return self._d[tuple(src_slice)]

    @property
    def dset(self):
        return self._d


class Accumulator:
    """Batch contiguous writes along one axis before flushing to a
    BankWriter.  Nothing MPI-aware — one instance per rank.

    Usage:
        with banked.open_writer() as w:
            acc = Accumulator(w, axis=1, capacity=64, dtype=np.float32)
            for chunk in ...:
                acc.append((slice(tb0, tb1), slice(z0, z1), slice(None)),
                            chunk)
            acc.close()

    Or use `with Accumulator(...) as acc:` which closes on exit.
    """

    def __init__(self, writer: BankWriter, axis: int, capacity: int,
                 dtype):
        self.w        = writer
        self.axis     = int(axis)
        self.capacity = int(capacity)
        self.dtype    = np.dtype(dtype)
        self._buf     = None       # allocated on first append
        self._fixed   = None       # tuple of non-banked-axis slices
        self._base    = None       # global start on banked axis
        self._filled  = 0

    def append(self, dest_slice, arr) -> None:
        ax_slice = dest_slice[self.axis]
        assert isinstance(ax_slice, slice), \
            f"banked axis {self.axis} of dest_slice must be a slice"
        arr_ax = arr.shape[self.axis]
        assert (ax_slice.stop - ax_slice.start) == arr_ax, (
            f"dest_slice axis extent {ax_slice.stop-ax_slice.start} != "
            f"arr axis extent {arr_ax}")

        fixed = tuple(s for i, s in enumerate(dest_slice) if i != self.axis)

        # If the incoming chunk can't be merged into the current run, flush
        # what we have and start fresh with this one.
        if self._buf is not None:
            contiguous = (fixed == self._fixed
                          and ax_slice.start == self._base + self._filled)
            fits = self._filled + arr_ax <= self.capacity
            if not (contiguous and fits):
                self.flush()

        if self._buf is None:
            buf_shape = list(arr.shape)
            buf_shape[self.axis] = self.capacity
            self._buf = np.empty(tuple(buf_shape), dtype=self.dtype)
        if self._fixed is None:
            self._fixed = fixed
            self._base = ax_slice.start
            self._filled = 0

        bsel = [slice(None)] * self._buf.ndim
        bsel[self.axis] = slice(self._filled, self._filled + arr_ax)
        self._buf[tuple(bsel)] = arr
        self._filled += arr_ax

    def flush(self) -> None:
        if self._buf is None or self._filled == 0:
            return
        dest = list(self._fixed)
        dest.insert(self.axis, slice(self._base, self._base + self._filled))
        bsel = [slice(None)] * self._buf.ndim
        bsel[self.axis] = slice(0, self._filled)
        self.w.write(tuple(dest), self._buf[tuple(bsel)])
        self._filled = 0
        self._base = None
        self._fixed = None

    def close(self) -> None:
        self.flush()
        self._buf = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def buf_bytes(self) -> int:
        if self._buf is None:
            return 0
        return int(self._buf.nbytes)
