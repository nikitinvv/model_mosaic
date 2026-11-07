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

# MOSAIC_SINGLE_H5=1: every step writes one plain h5 with parallel HDF5
# (h5py mpio, collective create/open/close, independent writes) instead of
# a VDS master + one bank file per writer.  mpirun needs --mca io romio341.
SINGLE_H5 = os.environ.get("MOSAIC_SINGLE_H5") == "1"


def cleanup_h5(path: str) -> None:
    """Remove a VDS master and its sibling bank directory (name/name_*.h5)."""
    if os.path.isfile(path):
        os.remove(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    bank_dir = os.path.join(os.path.dirname(path) or ".", stem)
    if os.path.isdir(bank_dir):
        shutil.rmtree(bank_dir)


def read_src_header(path, rank=0, comm=None, dataset="exchange/data",
                    theta="exchange/theta"):
    """Rank 0 reads a source store's shape (+ theta grid) and broadcasts it.

    Every step needs the true angle count BEFORE it can size its own output:
    iohdf5.layout defaults NTHETA to 3N/4 when nobody passes `ntheta=`, which
    is HALF the schematic's 360-degree per-tile count, so stitched/paganin/
    sino silently plan at half their real angle extent.  Steps 2 and 3 pass it
    from the positions file; steps 7/7b/8 learn it from their input's shape,
    which means the plan has to be resolved after this read rather than at
    import time.

    One rank opens the file, so there is no metadata-read storm at high rank
    counts.  Returns (shape, theta_deg); theta_deg is None when absent.
    """
    import h5py
    if rank == 0:
        with h5py.File(path, "r") as f:
            shape = tuple(f[dataset].shape)
            th = f[theta][:] if theta and theta in f else None
    else:
        shape = th = None
    if comm is not None:
        shape = comm.bcast(shape, root=0)
        th = comm.bcast(th, root=0)
    return shape, th


def rm_src(path, *, enabled=False, rank=0, comm=None, label=None):
    """Delete a finished step's input once every rank is done reading it.

    The pipeline writes 20x the reconstruction volume across its seven
    intermediate stores -- 384 TiB at UPS=8, 1.3 PiB at UPS=12 -- but only
    ever needs one producer/consumer pair resident at a time.  Dropping each
    input as its step completes turns that into a ~98 TiB peak at UPS=8.

    Barriers FIRST so no rank is still reading, then rank 0 removes it.
    Handles both a VDS master + bank dir (cleanup_h5) and a plain directory
    of per-tile files (step4/step5 mosaic_h5*).  Never raises: losing an
    intermediate to a permissions error must not kill a multi-hour job that
    has already produced its output.
    """
    if not enabled:
        return
    if comm is not None:
        comm.Barrier()
    if rank == 0:
        what = label or os.path.basename(path.rstrip("/")) or path
        try:
            if os.path.isdir(path) and not os.path.isfile(path):
                shutil.rmtree(path)
            else:
                cleanup_h5(path)
            print(f"  [rm-src] removed {what} ({path})", flush=True)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [rm-src] WARNING could not remove {path}: {exc}",
                  flush=True)
    if comm is not None:
        comm.Barrier()


def add_rm_src_arg(p, what):
    """The --rm-src flag, worded per step so --help names the actual file."""
    p.add_argument("--rm-src", action="store_true",
                   help=f"delete {what} once this step has finished writing "
                        f"(keeps peak disk to one producer/consumer pair)")
    return p


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


# Bytes this process currently holds in /dev/shm via alloc_shm.  Used only
# by the preflight below; free_shm() decrements it.
_SHM_HELD = {}


def _local_ranks() -> int:
    """Ranks of this job sharing this node, from whichever launcher put it in
    the environment (PALS/mpiexec on Polaris, mpich, OpenMPI, Slurm).  1 if
    none of them did, which makes the preflight check per-process only."""
    for var in ("PALS_LOCAL_SIZE", "PMI_LOCAL_SIZE", "MPI_LOCALNRANKS",
                "OMPI_COMM_WORLD_LOCAL_SIZE", "SLURM_NTASKS_PER_NODE"):
        try:
            n = int(os.environ[var])
        except (KeyError, ValueError):
            continue
        if n > 0:
            return n
    return 1


def _shm_preflight(nbytes: int) -> None:
    """Refuse an allocation the node's /dev/shm cannot hold.

    SharedMemory is a file in the tmpfs at /dev/shm, which Linux caps at half
    of RAM (252 GiB on a 503 GiB Polaris node).  tmpfs pages are allocated
    lazily, so asking for more than the cap succeeds here and then kills the
    rank with SIGBUS on the first touch past it -- signal 7, shell rc=135, no
    OOM message anywhere.  Checking the *capacity* (not the free space, which
    for the same lazy-allocation reason says nothing about what the other
    ranks are about to touch) turns that into a message naming the knob.

    Set MOSAIC_SHM_CHECK=0 to skip.
    """
    if os.environ.get("MOSAIC_SHM_CHECK", "1") == "0":
        return
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return                      # no tmpfs to reason about
    cap = st.f_blocks * st.f_frsize
    if cap <= 0:
        return
    nloc = _local_ranks()
    want = (sum(_SHM_HELD.values()) + nbytes) * nloc
    if want <= cap * 0.95:
        return
    g = 2 ** 30
    raise MemoryError(
        f"/dev/shm too small for this layout: {nloc} rank(s) on this node x "
        f"{(sum(_SHM_HELD.values()) + nbytes) / g:.1f} GiB = {want / g:.1f} "
        f"GiB, but /dev/shm holds {cap / g:.1f} GiB.  The shared-memory "
        f"buffers are sized by --mem-budget (GiB per rank); lower it to at "
        f"most {cap * 0.85 / nloc / g:.1f}.  Note /dev/shm is half of node "
        f"RAM, so budgeting off total RAM overcommits it -- and because tmpfs "
        f"pages come lazily, the overcommit would otherwise not fail here but "
        f"much later, as SIGBUS (signal 7, rc=135) on first touch."
    )


def alloc_shm(shape, dtype):
    """Allocate a shared-memory buffer of the given shape+dtype.  Returns
    (shm, ndarray-view).  Caller must call `free_shm(shm)` when done."""
    dtp = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtp.itemsize
    _shm_preflight(nbytes)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    _SHM_HELD[shm.name] = nbytes
    buf = np.ndarray(shape=shape, dtype=dtp, buffer=shm.buf)
    return shm, buf


def free_shm(shm) -> None:
    _SHM_HELD.pop(getattr(shm, "name", None), None)
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
    if SINGLE_H5:
        from iohdf5.dxchange_hdf5_chunks import tomo_initx_single
        if rank == 0:
            cleanup_h5(path)
        if comm is not None:
            comm.Barrier()
        return tomo_initx_single(path, shape, dtype, vchunks=vchunks,
                                 stype=stype, nbanks=nbanks, chunks=chunks,
                                 comm=comm)
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


def open_out(ctx) -> None:
    """Single-file output: open it for writing, collectively (every rank).
    Call after rank 0's serial edits (theta, attrs) and the barrier after
    them -- a serial handle and the MPI-IO one must not overlap.  No-op for
    VDS + banks."""
    if ctx.get('single'):
        import h5py
        from iohdf5.dxchange_hdf5_chunks import romio_info
        f = h5py.File(ctx['path'], 'r+', driver='mpio', comm=ctx['comm'],
                      info=romio_info())
        ctx['h5'], ctx['dset'] = f, f['exchange/data']


def close_out(ctx) -> None:
    """Single-file output: close the MPI-IO handle (collective, every rank).
    No-op for VDS + banks."""
    f = ctx.pop('h5', None)
    ctx.pop('dset', None)
    if f is not None:
        f.close()


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
                    nbanks: int, chunks=None, read_granule=None,
                    companion_bytes: int = 0) -> None:
    """Print planned shape / vchunks / bank layout / HDF5 chunk for an
    output file about to be created via initx_and_bcast.  Call from
    rank 0 only.  Pure math — does not touch the filesystem.

    `read_granule` is the (t, z, x) slab the *next* step will read back.
    Given it, this also prints how many HDF5 ops that read costs per bank
    file and how big each one is — the number the Polaris sweep is
    actually comparing, so the log carries it next to the timing instead
    of leaving it to be reconstructed from the chunk shape afterwards.

    `companion_bytes` is the step's other large buffer (the prefetched
    input slab).  Reporting only the vchunk buffer is what let step8 look
    like it needed 9.7 GB when it really needed 11.5."""
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
    nchunks_bank = int(np.prod([-(-b // c)
                                for b, c in zip(bank_shape, h5chunk)]))
    print(f"       bank shape={bank_shape} ({_hb(bank_bytes)})  "
          f"HDF5 chunk={h5chunk} ({_hb(chunk_bytes)})"
          f"  × {nchunks_bank}/bank"
          f"{'  [sinogram-ordered]' if sino_ordered else ''}", flush=True)

    buf = int(np.prod(vchunks)) * dtp.itemsize
    print(f"       per-rank buffers: vchunk={_hb(buf)}"
          + (f" + companion={_hb(companion_bytes)}"
             f" = {_hb(buf + companion_bytes)}" if companion_bytes else ""),
          flush=True)

    if read_granule is not None:
        nops, per_op = _read_cost(bank_shape, h5chunk, read_granule,
                                  dtp.itemsize)
        print(f"       next-step read {tuple(read_granule)} → "
              f"{nops} op(s)/bank × {_hb(per_op)}", flush=True)


def _read_cost(bank_shape, chunk, granule, itemsize):
    """(ops per bank file, bytes per op) for reading `granule` out of one
    bank file laid out with `chunk`.

    Inside a chunk the layout is plain C-order, so a sub-box is one
    contiguous run iff every axis inside some axis k is taken in full.
    Let k be the outermost axis whose successors are all full; the read
    then costs prod(extent_i) for i < k separate runs per chunk, times the
    number of chunks the granule spans.

    So (1, 512, N) chunks read as (1, 8, N) cost ONE op — axis 0 has
    extent 1, so the 8 z-rows are a contiguous interior run — while
    (48, 64, N) chunks read as (48, 8, N) cost 48.  That asymmetry is why
    the chunk policy clamps z to the consumer granule only when the θ
    extent is greater than 1.
    """
    gran = [min(int(g), int(b)) for g, b in zip(granule, bank_shape)]
    inner = [min(g, int(c)) for g, c in zip(gran, chunk)]   # per-chunk box

    # k = smallest axis such that every axis AFTER it is taken in full.
    k = 2
    if inner[2] == int(chunk[2]):
        k = 1
        if inner[1] == int(chunk[1]):
            k = 0
    runs_per_chunk = 1
    for ax in range(k):
        runs_per_chunk *= inner[ax]

    nchunks = 1
    for ax in (0, 1, 2):
        nchunks *= -(-gran[ax] // int(chunk[ax]))

    ops = max(1, nchunks * runs_per_chunk)
    total = itemsize
    for g in gran:
        total *= g
    return ops, max(1, total // ops)
