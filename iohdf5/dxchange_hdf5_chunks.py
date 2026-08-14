#!/usr/bin/env python

import atexit
import multiprocessing
import os
from functools import partial
from multiprocessing import shared_memory

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Persistent spawn-context worker pools.
#
# tomo_readx/tomo_writex fan I/O across a Pool.  The default fork start
# method inherits an already-initialised CUDA context (from callers like
# the mosaic step scripts) and breaks — cupy raises on any op in the
# child.  We use spawn instead, and cache pools by size so we don't pay
# the ~500 ms spawn-startup cost on every I/O call.
#
# Pool workers only need h5py + numpy, not cupy, so their module import
# is cheap.  Pools are shut down at process exit via atexit.
# ---------------------------------------------------------------------------
_SPAWN_CTX = multiprocessing.get_context("spawn")
_POOLS = {}   # nbanks -> Pool

# Env-var prefixes stripped from spawn workers.  On Cray/Polaris the parent
# was launched by PALS/mpiexec so PMI_* / PALS_* env vars are set; those
# propagate to spawn workers, and on worker startup Cray MPICH's library
# init sees them and tries `PMI2_Init()` — which asserts because the worker
# was NOT launched by PALS ("_pmi_smp_barrier failed").  Stripping these
# just for the Pool spawn makes libmpich fall back to singleton mode.
# Parent's own MPI is unaffected: mpi4py already called MPI_Init before
# tomo_writex reaches this point, and Cray MPI doesn't re-read env after.
_SCRUB_ENV_PREFIXES = ("PMI_", "PALS_")


def _get_pool(nbanks: int):
    p = _POOLS.get(nbanks)
    if p is None:
        saved = {}
        for k in list(os.environ):
            if k.startswith(_SCRUB_ENV_PREFIXES):
                saved[k] = os.environ.pop(k)
        try:
            p = _SPAWN_CTX.Pool(processes=int(nbanks))
        finally:
            os.environ.update(saved)
        _POOLS[nbanks] = p
    return p


@atexit.register
def _shutdown_pools():
    # Runs at interpreter shutdown, BEFORE mpi4py's MPI.Finalize
    # (imported later → registered earlier → runs later; atexit is LIFO).
    #
    # close() + join() lets each worker finish idling on the empty task
    # queue and exit cleanly — which is what releases the pool's
    # semaphores and internal queues.  If we SIGTERM'd via terminate()
    # instead, workers wouldn't run their own cleanup, resource_tracker
    # would keep the semaphores around, and MPI.Finalize would then
    # hang trying to reconcile with those orphaned SysV/POSIX SHM
    # resources on Cray MPICH.
    #
    # Safe because by atexit time all our compute is done — workers are
    # idle-waiting on the queue and close() just tells them "no more
    # tasks", they return cleanly, join() collects the exit statuses.
    for p in _POOLS.values():
        try:
            p.close()
            p.join()
        except Exception:
            pass
    _POOLS.clear()

# Cache for tomo_info() — master VDS attrs are immutable after tomo_initx,
# so we open the master + first bank once per (process, filename) and
# reuse.  Eliminates the h5py.File(..., 'r') storm that would otherwise
# fire on every tomo_writex/tomo_readx call, which is what forced us to
# set HDF5_USE_FILE_LOCKING=FALSE (and then invited the write-side
# B-tree / addr-overflow corruption bugs on Lustre).
_INFO_CACHE = {}


def _norm_stype(stype):
    """Normalise the stype attr (str / bytes / numpy.str_) to 'proj'|'slice'."""
    if isinstance(stype, bytes):
        stype = stype.decode()
    stype = str(stype).lower()
    if stype.startswith("proj"):
        return "proj"
    if stype.startswith("slice"):
        return "slice"
    raise Exception(f"Storage type {stype!r} is neither projection nor slice.")


def tomo_info(filename):
    cached = _INFO_CACHE.get(filename)
    if cached is not None:
        return cached
    info = {}
    with h5py.File(filename, 'r') as hf:
        dset = hf['/exchange/data']
        dtype = np.dtype(dset.dtype)
        nproj, ny, nx = dset.shape
        info["nproj"] = nproj
        info["ny"] = ny
        info["nx"] = nx
        info["dtype"] = dtype
        info["is_virtual"] = dset.is_virtual
        info["nbanks_per_svchunk"] = dset.attrs['nbanks_per_svchunk']
        vchunks = (int(dset.attrs['vchunks_0']),
                   int(dset.attrs['vchunks_1']),
                   int(dset.attrs['vchunks_2']))
        info["vchunks"] = vchunks
        info["shape"] = (nproj, ny, nx)
        stype = dset.attrs['stype']
        info["stype"] = stype
        # Chunks: tomo_initx records the HDF5 chunk shape it used on the
        # master, because it is no longer derivable from stype alone —
        # see the `chunks=` override there (sinogram-ordered chunks on a
        # θ-banked file).  Older files predate the attrs, so fall back to
        # the historical derivation.  Either way we avoid opening a bank
        # file, which caused BlockingIOError under concurrent tomo_info
        # calls from multiple ranks/workers.
        if 'chunks_0' in dset.attrs:
            info["chunks"] = (int(dset.attrs['chunks_0']),
                              int(dset.attrs['chunks_1']),
                              int(dset.attrs['chunks_2']))
        elif (isinstance(stype, str) and stype.lower().startswith("proj")) \
                or stype == b"proj" or stype == "proj":
            info["chunks"] = (1,) + vchunks[1:]
        else:
            info["chunks"] = (vchunks[0], 1, vchunks[2])
        if not dset.is_virtual:
            # Contiguous / non-VDS case: read actual chunks off the dset.
            info["chunks"] = dset.chunks
    _INFO_CACHE[filename] = info
    return info

def tomo_readx(filename, ntasks=1, shm=None, ivchunk=(0,0,0), vchunks=None):
    info = tomo_info(filename)
    nproj, ny, nx, dtp, chunks = info['nproj'], info['ny'], info['nx'], info['dtype'], info['chunks']
    vchunks = vchunks if vchunks is not None else (nproj, ny, nx)
    ivchunk = ivchunk if ivchunk is not None else (0,0,0)
    # stype describes how the BANKS are split (θ for 'proj', z for 'slice'),
    # which is what decides how work is sharded here.  It is no longer
    # inferable from the chunk shape: with the `chunks=` override a
    # θ-banked file can carry sinogram-ordered (nt, 1, nx) chunks.
    stype = _norm_stype(info['stype'])
    if shm is None:
        # pre-allocated data
        shm = shared_memory.SharedMemory(create=True, size=np.prod(vchunks)*dtp.itemsize)
        shm_selfmanaged = True
    else:
        shm_selfmanaged = False
    data = np.ndarray(shape=vchunks, dtype=dtp, buffer=shm.buf)
    # read data
    if stype == 'proj':
        # projections
        _process = _process_read_projs
    elif stype == 'slice':
        # slices
        _process = _process_read_slices
    else:
        raise Exception("Storage type is neither projection or slice.")
    try:
        pool = _get_pool(ntasks)
        results = pool.map(partial(_process, ntasks=ntasks, filename=filename, direct_chunk=~info['is_virtual'],
                                  vchunks=vchunks, ivchunk=ivchunk, shm=shm), list(np.arange(ntasks)))
        if shm_selfmanaged:
            data_out = data.copy()
    finally:
        if shm_selfmanaged:
            shm.close()
            shm.unlink()

    return data_out if shm_selfmanaged else data

def _create_banking_plan(filename, shape, vchunks=None, nbanks_per_svchunk=1, sitems_idx=0):
    vchunks = vchunks if vchunks is not None else shape

    nchunks = (shape[sitems_idx] + vchunks[sitems_idx] - 1) // vchunks[sitems_idx]

    fstorages = [''] * nbanks_per_svchunk

    # create filename formatter
    basename_no_ext, fext = os.path.splitext(os.path.basename(filename))
    realfdirname = os.path.dirname(filename)
    filename_fmt_vsrc = os.path.join(basename_no_ext, basename_no_ext) + '_data_%06d' + fext
    if fstorages[0] == '':
        filename_fmt_file = os.path.join(realfdirname, basename_no_ext, basename_no_ext) + '_data_%06d' + fext
    else:
        # storage part will be added on-the-fly
        filename_fmt_file = os.path.join(basename_no_ext, basename_no_ext) + '_data_%06d' + fext

    sitems_per_bank = (vchunks[sitems_idx] + nbanks_per_svchunk - 1) // nbanks_per_svchunk
    nchunks = (shape[sitems_idx] + vchunks[sitems_idx] - 1) // vchunks[sitems_idx]

    banks_filename_vsrc = [None] * (nchunks*nbanks_per_svchunk)
    banks_filename_path = [None] * (nchunks*nbanks_per_svchunk)
    banks_size = [0] * (nchunks*nbanks_per_svchunk)
    
    for ivchunk in range(nchunks):
        for ibank in range(nbanks_per_svchunk):
            filename_data_file = filename_fmt_file % (ivchunk*nbanks_per_svchunk+ibank,)
            filename_data_vsrc = filename_fmt_vsrc % (ivchunk*nbanks_per_svchunk+ibank,)
            filename_data_file = filename_data_file if fstorages[ibank] == '' else os.path.join(fstorages[ibank], filename_data_file)
            filename_data_vsrc = filename_data_vsrc if fstorages[ibank] == '' else os.path.join(fstorages[ibank], filename_data_vsrc)
            banks_filename_vsrc[ivchunk*nbanks_per_svchunk+ibank] = filename_data_vsrc
            banks_filename_path[ivchunk*nbanks_per_svchunk+ibank] = filename_data_file
            sitems_offset = ivchunk*vchunks[sitems_idx]
            sitems_start = sitems_offset + ibank * sitems_per_bank
            sitems_end = sitems_offset + (ibank+1) * sitems_per_bank
            sitems_end = sitems_end if (sitems_end - sitems_offset) < vchunks[sitems_idx] else sitems_offset + vchunks[sitems_idx]
            sitems_end = sitems_end if sitems_end < shape[sitems_idx] else shape[sitems_idx]
            banks_size[ivchunk*nbanks_per_svchunk+ibank] = (sitems_end-sitems_start) if sitems_end > sitems_start else 0

    return banks_filename_path, banks_size, banks_filename_vsrc
            
def tomo_initx(filename, shape, dtype, vchunks=None, mode='a', stype='proj',
               nbanks=1, chunks=None):
    """Create a VDS master file + all bank files (single-rank).

    Fast because each bank file's h5py.File 'w' + create_dataset just
    writes an empty superblock + metadata; HDF5's default LATE
    allocation means no per-chunk storage is reserved until writes.
    Call from rank 0; broadcast or recompute ctx on other ranks (the
    banking plan is deterministic in the params).

    `stype` picks which axis the BANK FILES are split along — θ for
    'proj', z for 'slice'.  That is a correctness constraint, not a
    performance one: ranks shard their work along the same axis, so
    banking on that axis is what keeps every bank file owned by exactly
    one writer.  Never change it just to speed up a downstream read.

    `chunks` overrides the HDF5 chunk shape inside those bank files,
    which IS purely a performance knob and is independent of the
    banking.  Default follows stype: (1, ny, nx) projection-ordered for
    'proj', (nproj, 1, nx) sinogram-ordered for 'slice'.  Passing e.g.
    (nt_per_bank, 1, nx) on a θ-banked file gives sinogram-ordered
    chunks with θ-split banks — the layout paganin.h5 wants, because it
    is written a θ-slab at a time (so banking must be on θ) but read a
    z-slab at a time (so chunks must be on z).  Without it, an FBP
    z-slab read touches every (1, ny, nx) chunk to use ny/zslab of it.
    """
    nproj, ny, nx = shape
    stype = 'proj' if stype.lower() in ('proj', 'projs') else 'slice'
    vchunks = vchunks if vchunks is not None else (nproj, ny, nx)

    sitems_idx = 0 if stype == 'proj' else 1

    banks_filename_path, banks_size, banks_filename_vsrc = _create_banking_plan(
        filename=filename, shape=shape, vchunks=vchunks,
        nbanks_per_svchunk=nbanks, sitems_idx=sitems_idx)

    if os.path.isfile(filename):
        with h5py.File(filename, mode, libver='latest') as fid:
            if '/exchange/data' in fid:
                raise Exception("Virtual dataset exists already.")

    # create directory(ies)
    for filepath in banks_filename_path:
        fdirname = os.path.dirname(filepath)
        if fdirname:
            os.makedirs(fdirname, exist_ok=True)

    layout = h5py.VirtualLayout(shape=(nproj, ny, nx), dtype=dtype)
    sitems_per_bank = (vchunks[sitems_idx] + nbanks - 1) // nbanks
    nchunks = (shape[sitems_idx] + vchunks[sitems_idx] - 1) // vchunks[sitems_idx]

    # Nominal (non-ragged) bank shape, used to clamp the chunk shape and to
    # record it on the master.  A short trailing bank gets its own clamp
    # below; tomo_info reports this nominal one.
    if stype == 'proj':
        nominal_bank = (sitems_per_bank, ny, nx)
        default_chunks = (1,) + tuple(vchunks[1:])
    else:
        nominal_bank = (nproj, sitems_per_bank, nx)
        default_chunks = (vchunks[0], 1, vchunks[2])
    want_chunks = tuple(int(c) for c in chunks) if chunks is not None \
        else default_chunks
    if len(want_chunks) != 3 or any(c < 1 for c in want_chunks):
        raise ValueError(f"chunks must be 3 positive ints, got {want_chunks}")
    nominal_chunks = tuple(min(c, s) for c, s in zip(want_chunks, nominal_bank))

    for _ivchunk in range(nchunks):
        for ibank in range(nbanks):
            bank_idx = _ivchunk * nbanks + ibank
            filename_data_file = banks_filename_path[bank_idx]
            filename_data_vsrc = banks_filename_vsrc[bank_idx]
            sitems_offset = _ivchunk * vchunks[sitems_idx]
            sitems_start = sitems_offset + ibank * sitems_per_bank
            sitems_end = sitems_offset + (ibank + 1) * sitems_per_bank
            sitems_end = min(sitems_end, sitems_offset + vchunks[sitems_idx], shape[sitems_idx])
            if sitems_end > sitems_start:
                if stype == 'proj':
                    vsource = h5py.VirtualSource(filename_data_vsrc, "/exchange/data",
                                                 shape=(sitems_end - sitems_start, ny, nx))
                    layout[sitems_start:sitems_end, :, :] = vsource
                else:
                    vsource = h5py.VirtualSource(filename_data_vsrc, "/exchange/data",
                                                 shape=(nproj, sitems_end - sitems_start, nx))
                    layout[:, sitems_start:sitems_end, :] = vsource
                if stype == 'proj':
                    bank_shape = (sitems_end - sitems_start, ny, nx)
                else:
                    bank_shape = (nproj, sitems_end - sitems_start, nx)
                bank_chunks = tuple(min(c, s)
                                    for c, s in zip(want_chunks, bank_shape))
                with h5py.File(filename_data_file, 'w') as hf_out:
                    g = hf_out.create_group('/exchange')
                    g.create_dataset('data', shape=bank_shape,
                                     chunks=bank_chunks,
                                     dtype=dtype, fillvalue=None)
    # create master file
    with h5py.File(filename, 'w', libver='latest') as hf:
        dset = hf.create_virtual_dataset('/exchange/data', layout, fillvalue=-5)
        dset.attrs['nbanks_per_svchunk'] = nbanks
        dset.attrs['stype'] = stype
        dset.attrs['vchunks_0'] = vchunks[0]
        dset.attrs['vchunks_1'] = vchunks[1]
        dset.attrs['vchunks_2'] = vchunks[2]
        dset.attrs['chunks_0'] = nominal_chunks[0]
        dset.attrs['chunks_1'] = nominal_chunks[1]
        dset.attrs['chunks_2'] = nominal_chunks[2]

    return {'banks_filename_path': banks_filename_path, 'banks_size': banks_size}
    
def tomo_writex(filename, data, shm=None, ivchunk=(0,0,0), ctx=None):

    info = tomo_info(filename)

    stype = info['stype']
    shape = info['shape']
    vchunks = info['vchunks']
    dtp = info['dtype']
    nbanks_per_svchunk = info['nbanks_per_svchunk']

    nproj, ny, nx = shape
    assert data.dtype == dtp

    sitems_idx = 0 if stype == 'proj' else 1

    if ctx is None:
        banks_filename_path, banks_size, _ = _create_banking_plan(
            filename=filename, shape=shape, vchunks=vchunks,
            nbanks_per_svchunk=nbanks_per_svchunk, sitems_idx=sitems_idx)
        ctx = {'banks_filename_path': banks_filename_path, 'banks_size': banks_size}
    
    if shm is None:
        # pre-allocated data
        shm = shared_memory.SharedMemory(create=True, size=np.prod(vchunks)*dtp.itemsize)
        shm_selfmanaged = True
        _data = np.ndarray(shape=vchunks, dtype=dtp, buffer=shm.buf)
        np.copyto(_data, data)
    else:
        shm_selfmanaged = False
    # write data (from shm)
    if stype == 'proj':
        # projections
        _process = _process_write_projs
    elif stype == 'slice':
        # slices
        _process = _process_write_slices
    else:
        raise Exception("Storage type is neither projection or slice.")
    try:
        pool = _get_pool(nbanks_per_svchunk)
        results = pool.map(partial(_process, ntasks=nbanks_per_svchunk, filename=filename, shape=shape, dtype=dtp,
                                   vchunks=vchunks, ivchunk=ivchunk, shm=shm, ctx=ctx), list(np.arange(nbanks_per_svchunk)))
    finally:
        if shm_selfmanaged:
            shm.close()
            shm.unlink()

def _process_read_projs(itask, ntasks, filename, shm, vchunks, ivchunk, direct_chunk=False):
    info = tomo_info(filename)
    nproj, ny, nx, dtp = info['nproj'], info['ny'], info['nx'], info['dtype']
    assert _norm_stype(info['stype']) == 'proj'

    projs_per_task = (vchunks[0] + ntasks - 1) // ntasks
    projs_offset = ivchunk[0]*vchunks[0]
    projs_start = projs_offset + itask * projs_per_task
    projs_end = projs_offset + (itask+1) * projs_per_task
    projs_end = projs_end if (projs_end - projs_offset) < vchunks[0] else projs_offset + vchunks[0]
    projs_end = projs_end if projs_end < nproj else nproj

    y_start = ivchunk[1] * vchunks[1]
    y_end = (ivchunk[1]+1) * vchunks[1]
    y_end = y_end if y_end < ny else ny
    
    x_start = ivchunk[2] * vchunks[2]
    x_end = (ivchunk[2]+1) * vchunks[2]
    x_end = x_end if x_end < nx else nx
    
    if projs_end > projs_start:
        out = np.ndarray(shape=vchunks, dtype=dtp, buffer=shm.buf)
        with h5py.File(filename, 'r') as hf_in:
            dset = hf_in['/exchange/data']
            if direct_chunk:
                dset.read_direct(out, source_sel=np.s_[projs_start:projs_end,y_start:y_end,x_start:x_end],
                                 dest_sel=np.s_[projs_start-projs_offset:projs_end-projs_offset,:y_end-y_start,:x_end-x_start])
            else:
                out[projs_start-projs_offset:projs_end-projs_offset,:y_end-y_start,:x_end-x_start] = ...
                dset[projs_start:projs_end,y_start:y_end,x_start:x_end]
    return itask

def _process_read_slices(itask, ntasks, filename, shm, vchunks, ivchunk, direct_chunk=False):
    info = tomo_info(filename)
    nproj, ny, nx, dtp = info['nproj'], info['ny'], info['nx'], info['dtype']
    assert _norm_stype(info['stype']) == 'slice'

    slices_offset = ivchunk[1]*vchunks[1]
    slices_per_task = (vchunks[1] + ntasks - 1) // ntasks
    slices_start = slices_offset + itask * slices_per_task
    slices_end = slices_offset + (itask+1) * slices_per_task
    slices_end = slices_end if (slices_end - slices_offset) < vchunks[1] else slices_offset + vchunks[1]
    slices_end = slices_end if slices_end < ny else ny
    
    projs_start = ivchunk[0]*vchunks[0]
    projs_end = (ivchunk[0]+1)*vchunks[0]
    projs_end = projs_end if projs_end < nproj else nproj

    x_start = ivchunk[2] * vchunks[2]
    x_end = (ivchunk[2]+1) * vchunks[2]
    x_end = x_end if x_end < nx else nx
    
    out = np.ndarray(shape=vchunks, dtype=dtp, buffer=shm.buf)
    if slices_end > slices_start:
        with h5py.File(filename, 'r') as hf_in:
            dset = hf_in['/exchange/data']
            if direct_chunk:
                dset.read_direct(out, source_sel=np.s_[projs_start:projs_end,slices_start:slices_end,x_start:x_end],
                                 dest_sel=np.s_[:projs_end-projs_start,slices_start-slices_offset:slices_end-slices_offset,:x_end-x_start])
            else:
                out[:projs_end-projs_start,slices_start-slices_offset:slices_end-slices_offset,:x_end-x_start] = ...
                dset[projs_start:projs_end,slices_start:slices_end,x_start:x_end]
    return itask

def _process_write_slices(itask, ntasks, filename, shape, dtype, shm, vchunks, ivchunk, ctx):
    
    nproj, ny, nx = shape
    
    slices_per_task = (vchunks[1] + ntasks - 1) // ntasks
    slices_offset = ivchunk[1]*vchunks[1]
    slices_start = slices_offset + itask * slices_per_task
    slices_end = slices_offset + (itask+1) * slices_per_task
    slices_end = slices_end if (slices_end - slices_offset) < vchunks[1] else slices_offset + vchunks[1]
    slices_end = slices_end if slices_end < ny else ny
    
    projs_start = ivchunk[0]*vchunks[0]
    projs_end = (ivchunk[0]+1)*vchunks[0]
    projs_end = projs_end if projs_end < nproj else nproj

    x_start = ivchunk[2] * vchunks[2]
    x_end = (ivchunk[2]+1) * vchunks[2]
    x_end = x_end if x_end < nx else nx

    data = np.ndarray(shape=vchunks, dtype=dtype, buffer=shm.buf)
    
    try:
        if slices_end > slices_start:
            filename_path = ctx['banks_filename_path'][ivchunk[1]*ntasks+itask]
            with h5py.File(filename_path, 'r+') as hf_out:
                dset = hf_out['/exchange/data']
                dset.write_direct(data, source_sel=np.s_[:projs_end-projs_start,slices_start-slices_offset:slices_end-slices_offset,:x_end-x_start],
                                  dest_sel=np.s_[projs_start:projs_end,:slices_end-slices_start,x_start:x_end])
    finally:
        pass
    return itask

def _process_write_projs(itask, ntasks, filename, shape, dtype, shm, vchunks, ivchunk, ctx):
    
    nproj, ny, nx = shape
    
    projs_per_task = (vchunks[0] + ntasks - 1) // ntasks
    projs_offset = ivchunk[0]*vchunks[0]
    projs_start = projs_offset + itask * projs_per_task
    projs_end = projs_offset + (itask+1) * projs_per_task
    projs_end = projs_end if (projs_end - projs_offset) < vchunks[0] else projs_offset + vchunks[0]
    projs_end = projs_end if projs_end < nproj else nproj

    y_start = ivchunk[1] * vchunks[1]
    y_end = (ivchunk[1]+1) * vchunks[1]
    y_end = y_end if y_end < ny else ny
    
    x_start = ivchunk[2] * vchunks[2]
    x_end = (ivchunk[2]+1) * vchunks[2]
    x_end = x_end if x_end < nx else nx

    data = np.ndarray(shape=vchunks, dtype=dtype, buffer=shm.buf)
    try:
        if projs_end > projs_start:
            filename_path = ctx['banks_filename_path'][ivchunk[0]*ntasks+itask]
            with h5py.File(filename_path, 'r+') as hf_out:
                dset = hf_out['/exchange/data']
                dset.write_direct(data, source_sel=np.s_[projs_start-projs_offset:projs_end-projs_offset,:y_end-y_start,:x_end-x_start],
                                  dest_sel=np.s_[0:projs_end-projs_start,y_start:y_end,x_start:x_end])
    finally:
        pass
    return itask


# ---------------------------------------------------------------------------
# vchunkx: "super-vchunk" — buffers larger than the file's on-disk vchunks.
#
# Two use cases:
#   (a) READ  — a big θ- or z-slab, split across parallel workers each doing
#               its own fancy-slice on the VDS master.  Cost is set by the
#               file's HDF5 chunk shape, not by the slab shape: a slab is
#               cheap when it covers whole chunks and expensive when it
#               clips them (reading a z-slab out of (1, ny, nx) projection
#               chunks touches every chunk to keep zslab/ny of it).
#   (b) WRITE — dice a big vchunkx buffer into vchunk-sized pieces and fan
#               each piece through tomo_writex (aligned parallel bank writes).
#
# Ported (with small tweaks) from doe-maxiv/dxchange_hdf5_chunks.py.
# ---------------------------------------------------------------------------


def _process_read_box(task_meta, filename, vchunksx, dtype, shm):
    """Worker: read one sub-box of a vchunkx from the VDS master straight
    into its slot in the shm buffer.

    read_direct (rather than `out[...] = dset[...]`) matters here: the
    latter makes HDF5 materialise the whole shard as a fresh array before
    numpy copies it into shm, which on a full-size shard is a spare
    multi-hundred-MB allocation per worker per call.
    """
    out = np.ndarray(shape=vchunksx, dtype=dtype, buffer=shm.buf)
    (t0, t1), (z0, z1), (x0, x1) = task_meta['src']
    dt, dz, dx = task_meta['dst']
    with h5py.File(filename, 'r') as hf_in:
        hf_in['/exchange/data'].read_direct(
            out,
            source_sel=np.s_[t0:t1, z0:z1, x0:x1],
            dest_sel=np.s_[dt:dt + (t1 - t0),
                           dz:dz + (z1 - z0),
                           dx:dx + (x1 - x0)])
    return task_meta['itask']


def _bank_axis(filename):
    """Axis the file's bank files are split along: 0 (θ) when proj-banked,
    1 (z) when slice-banked.  None if the file carries no tomo_initx attrs.

    Sharding a read's worker pool on this axis is what keeps each worker
    on a disjoint set of bank files.  Shard on the other axis and every
    worker opens every bank file and strides through it, which costs both
    the file handles and the sequentiality.
    """
    try:
        return 0 if _norm_stype(tomo_info(filename)['stype']) == 'proj' else 1
    except Exception:
        return None


def _split(lo, hi, ntasks, align=1):
    """Split [lo, hi) into at most ntasks contiguous non-empty ranges.

    `align` rounds the per-task extent up to a multiple of the HDF5 chunk
    extent along the split axis, so a chunk is never straddled by two
    workers.  Two workers sharing a chunk each issue a strided sub-chunk
    read, and the kernel's readahead then pulls the whole chunk in for
    each of them -- measured as a clean 2x on the sinogram layout, whose
    chunk spans all theta in a bank.  Fewer, whole-chunk tasks beat more,
    partial-chunk ones; if align is coarse enough that fewer than ntasks
    ranges come out, that is the correct answer, not a bug.
    """
    span = hi - lo
    per = (span + ntasks - 1) // ntasks
    if align > 1:
        per = ((per + align - 1) // align) * align
    per = max(per, 1)
    out = []
    for itask in range(ntasks):
        s0 = lo + itask * per
        s1 = min(s0 + per, hi)
        if s1 > s0:
            out.append((itask, s0, s1))
    return out


def _chunk_extent(info, axis):
    """Chunk extent along `axis`, for aligning a read's task split to it."""
    try:
        return max(1, int(info['chunks'][axis]))
    except Exception:
        return 1


def read_projs_vchunkx(filename, shm, ntasks, vchunksx, ivchunkx,
                       shard_axis=None):
    """Read a θ-slab vchunkx (vchunksx[0], NZ, N) from a VDS+banks file
    with ntasks parallel workers, each doing a fancy-slice through the
    VDS master.

    Returns a numpy view of the shm buffer with shape=vchunksx.
    """
    info = tomo_info(filename)
    shape, dtp = info['shape'], info['dtype']
    data = np.ndarray(shape=vchunksx, dtype=dtp, buffer=shm.buf)

    t_off = vchunksx[0] * ivchunkx[0]
    t_hi  = min(t_off + vchunksx[0], shape[0])
    nz    = min(vchunksx[1], shape[1])
    nx    = min(vchunksx[2], shape[2])

    if shard_axis is None:
        shard_axis = _bank_axis(filename)
        shard_axis = 0 if shard_axis is None else shard_axis

    task_meta = []
    if shard_axis == 0:
        for itask, s0, s1 in _split(t_off, t_hi, ntasks,
                                    align=_chunk_extent(info, 0)):
            task_meta.append({'itask': itask,
                              'src': ((s0, s1), (0, nz), (0, nx)),
                              'dst': (s0 - t_off, 0, 0)})
    else:
        for itask, s0, s1 in _split(0, nz, ntasks,
                                    align=_chunk_extent(info, 1)):
            task_meta.append({'itask': itask,
                              'src': ((t_off, t_hi), (s0, s1), (0, nx)),
                              'dst': (0, s0, 0)})

    pool = _get_pool(ntasks)
    pool.map(partial(_process_read_box, filename=filename,
                     vchunksx=vchunksx, dtype=dtp, shm=shm), task_meta)
    return data


def read_slices_vchunkx(filename, shm, ntasks, vchunksx, ivchunkx,
                        shard_axis=None):
    """Read a z-slab vchunkx (NTHETA, vchunksx[1], N) from a VDS+banks file
    with ntasks parallel workers, each doing a fancy-slice through the
    VDS master.

    Workers are sharded along the file's bank axis (see _bank_axis), so on
    the θ-banked, sinogram-chunked paganin.h5 that step7 writes each worker
    streams whole chunks out of its own quarter of the bank files.

    Returns a numpy view of the shm buffer with shape=vchunksx.
    """
    info = tomo_info(filename)
    shape, dtp = info['shape'], info['dtype']
    data = np.ndarray(shape=vchunksx, dtype=dtp, buffer=shm.buf)

    z_off = vchunksx[1] * ivchunkx[1]
    z_hi  = min(z_off + vchunksx[1], shape[1])
    ntheta = min(vchunksx[0], shape[0])
    nx     = min(vchunksx[2], shape[2])

    if shard_axis is None:
        shard_axis = _bank_axis(filename)
        shard_axis = 1 if shard_axis is None else shard_axis

    task_meta = []
    if shard_axis == 0:
        for itask, s0, s1 in _split(0, ntheta, ntasks,
                                    align=_chunk_extent(info, 0)):
            task_meta.append({'itask': itask,
                              'src': ((s0, s1), (z_off, z_hi), (0, nx)),
                              'dst': (s0, 0, 0)})
    else:
        for itask, s0, s1 in _split(z_off, z_hi, ntasks,
                                    align=_chunk_extent(info, 1)):
            task_meta.append({'itask': itask,
                              'src': ((0, ntheta), (s0, s1), (0, nx)),
                              'dst': (0, s0 - z_off, 0)})

    pool = _get_pool(ntasks)
    pool.map(partial(_process_read_box, filename=filename,
                     vchunksx=vchunksx, dtype=dtp, shm=shm), task_meta)
    return data


def write_vchunkx(filename, shm, vchunksx, vchunks, ctx, ivchunkx):
    """Write a big vchunkx buffer (in shm, shape=vchunksx) to a VDS+banks
    file by splitting along the z axis (axis 1) into vchunks-sized pieces
    and calling tomo_writex for each piece.  Ivchunkx is the position of
    the wide vchunkx; each piece gets ivchunk=(ivchunkx[0], iblock,
    ivchunkx[2]).
    """
    dtp = np.dtype(np.float32)
    data = np.ndarray(shape=vchunksx, dtype=dtp, buffer=shm.buf)

    nblocks = (vchunksx[1] + vchunks[1] - 1) // vchunks[1]
    for iblock in range(nblocks):
        b0 = iblock * vchunks[1]
        b1 = min(b0 + vchunks[1], vchunksx[1])
        if b1 > b0:
            tomo_writex(filename, data[:, b0:b1, :], shm=None,
                        ivchunk=(ivchunkx[0], iblock, ivchunkx[2]), ctx=ctx)

