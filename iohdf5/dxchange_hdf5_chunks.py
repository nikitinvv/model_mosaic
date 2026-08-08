#!/usr/bin/env python

import atexit
import h5py
import numpy as np

import multiprocessing
from functools import partial
from multiprocessing import shared_memory

import time

import json
import os


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


def shutdown_pools():
    """SIGKILL every cached spawn Pool worker and drop references.

    Call this explicitly at the end of each step's main() — after the
    last tomo_writex/tomo_readx, before the final MPI barrier + return.
    Relying on the @atexit fallback below is unreliable under MPI on
    some systems (tomo5): atexit hook order is LIFO, and if mpi4py's
    MPI.Finalize atexit hook was registered before ours (which it
    usually is, since mpi4py is imported first), it runs AFTER ours —
    and if for any reason ours doesn't fire (uncaught signal, exit via
    os._exit, worker teardown deadlock), MPI never finalizes and every
    rank hangs at interpreter shutdown.  Explicit call at end of main
    sidesteps all of that.

    We SIGKILL the worker PIDs directly rather than call Pool.terminate()
    or Pool.close()+join(): those wind through multiprocessing's own
    supervisor threads and can themselves deadlock on tomo5.  All our
    tasks have completed by shutdown time so workers are idle on the
    task queue; SIGKILL is safe and instant.  Idempotent.
    """
    import signal
    for p in _POOLS.values():
        try:
            for w in getattr(p, '_pool', []):
                pid = getattr(w, 'pid', None)
                if pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # already gone
                    except Exception:
                        pass
        except Exception:
            pass
    _POOLS.clear()


# Belt-and-suspenders: still register the atexit fallback for scripts
# that forget to call shutdown_pools() explicitly.
atexit.register(shutdown_pools)

# Cache for tomo_info() — master VDS attrs are immutable after tomo_initx,
# so we open the master + first bank once per (process, filename) and
# reuse.  Eliminates the h5py.File(..., 'r') storm that would otherwise
# fire on every tomo_writex/tomo_readx call, which is what forced us to
# set HDF5_USE_FILE_LOCKING=FALSE (and then invited the write-side
# B-tree / addr-overflow corruption bugs on Lustre).
_INFO_CACHE = {}


def _create_chunked_early(group, name: str, shape, dtype, chunks):
    """Create a chunked dataset with ALLOC_TIME_EARLY.

    HDF5's default alloc_time for chunked datasets is LATE — each chunk's
    space is allocated on first write to it.  Under concurrent writers +
    disabled file locking (Lustre workaround), two writers can both
    allocate the "next" chunk region and one loses the race, producing
    'wrong B-tree signature' or 'addr overflow' at write time.

    EARLY allocation reserves all chunk storage inside tomo_initx (rank
    0 only), before any writer touches the file.  Later writes never
    allocate — they only fill.  Eliminates the alloc race entirely.

    h5py's high-level Group.create_dataset does not expose alloc_time,
    so we drop to the low-level DCPL API here.
    """
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_chunk(tuple(chunks))
    plist.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    space = h5py.h5s.create_simple(tuple(shape))
    type_id = h5py.h5t.py_create(np.dtype(dtype), logical=True)
    dset_id = h5py.h5d.create(group.id, name.encode('ascii'),
                              type_id, space, dcpl=plist)
    return h5py.Dataset(dset_id)


def tomo_info(filename):
    cached = _INFO_CACHE.get(filename)
    if cached is not None:
        return cached
    fullpath_vs = None
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
        info["vchunks"] = (dset.attrs['vchunks_0'],dset.attrs['vchunks_1'],dset.attrs['vchunks_2'])
        info["shape"] = (nproj, ny, nx)
        info["meta"] = json.loads(dset.attrs['meta'])
        info["stype"] = dset.attrs['stype']
        
        if dset.is_virtual:
            fname_vs = dset.virtual_sources()[0].file_name
            fullpath_vs = os.path.join(os.path.dirname(filename), fname_vs)
        else:
            # read chunks dim directly
            chunks = dset.chunks
    if fullpath_vs is not None:
        with h5py.File(fullpath_vs, 'r') as hf:
            dset = hf['/exchange/data']
            chunks = dset.chunks
    info["chunks"] = chunks
    _INFO_CACHE[filename] = info
    return info

def tomo_readx(filename, ntasks=1, shm=None, ivchunk=(0,0,0), vchunks=None):
    info = tomo_info(filename)
    nproj, ny, nx, dtp, chunks = info['nproj'], info['ny'], info['nx'], info['dtype'], info['chunks']
    vchunks = vchunks if vchunks is not None else (nproj, ny, nx)
    ivchunk = ivchunk if ivchunk is not None else (0,0,0)
    if chunks[0] == 1:
        stype = 'proj'
    elif chunks[1] == 1:
        stype = 'slice'
    else:
        raise Exception("Storage type is neither projection or slice.")
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

def _create_banking_plan(filename, shape, vchunks=None, nbanks_per_svchunk=1, sitems_idx = 0, meta={}):
    vchunks = vchunks if vchunks is not None else shape

    #sitems_per_bank = (vchunks[sitems_idx] + nbanks - 1) // nbanks
    nchunks = (shape[sitems_idx] + vchunks[sitems_idx] - 1) // vchunks[sitems_idx]

    fstorages = ['']*nbanks_per_svchunk

    if 'infra' in meta: 
        if meta['infra'] == 'aps25':
            for ibank in range(nbanks_per_svchunk):
                if ibank % 3 == 0:
                    fstorages[ibank] = '/data/scratch/'
                elif ibank % 3 == 1:
                    fstorages[ibank] = '/data2/scratch/'
                else:
                    fstorages[ibank] = '/data3/scratch/'

        if meta['infra'] == 'aps25_13':
            for ibank in range(nbanks_per_svchunk):
                if ibank % 2 == 0:
                    fstorages[ibank] = '/data/scratch/'
                else:
                    fstorages[ibank] = '/data3/scratch/'

        elif meta['infra'] == 'maxiv-aps25':
            for ibank in range(nbanks_per_svchunk):
                if ibank % 3 == 0:
                    fstorages[ibank] = '/scratch/tomo-doe/data/scratch/'
                elif ibank % 3 == 1:
                    fstorages[ibank] = '/scratch/tomo-doe/data2/scratch/'
                else:
                    fstorages[ibank] = '/scratch/tomo-doe/data3/scratch/'

        elif meta['infra'] in ['',None,[],()]:
            pass
        else:
            raise Exception("Unsupported infrastructure: %s" % (meta['infra'],))

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
               nbanks=1, meta={}, rank=0, size=1):
    """Create a VDS master file + per-rank subset of bank files.

    Bank files are pre-allocated with ALLOC_TIME_EARLY, which reserves
    on-disk space (posix_fallocate) upfront to avoid alloc races between
    concurrent writers with disabled Lustre file locking.  On a 72 GB
    dataset that's ~1000+ files each reserving tens of MB — expensive
    when done sequentially by rank 0.

    Parallel-init: pass `rank` and `size` (all ranks call in lockstep).
    Bank files are sharded round-robin across ranks (bank_idx % size ==
    rank).  Rank 0 additionally creates the VDS master.  The banking
    plan is deterministic, so all ranks agree without coordination.
    """
    nproj, ny, nx = shape
    stype = 'proj' if stype.lower() == 'proj' or stype.lower() == 'projs' else 'slice'
    dtp = np.dtype(dtype)
    vchunks = vchunks if vchunks is not None else (nproj, ny, nx)

    sitems_idx = 0 if stype == 'proj' else 1

    banks_filename_path, banks_size, banks_filename_vsrc = _create_banking_plan(filename=filename,
                                                                                shape=shape,
                                                                                vchunks=vchunks,
                                                                                nbanks_per_svchunk=nbanks,
                                                                                sitems_idx = sitems_idx,
                                                                                meta=meta)

    # test if VDS present already (rank 0 only — filesystem check).
    if rank == 0:
        _test = True
        if os.path.isfile(filename):
            with h5py.File(filename, mode, libver='latest') as fid:
                try:
                    dset = fid['/exchange/data']
                    _test = False
                except:
                    pass
        if not _test:
            raise Exception("Virtual dataset exists already.")

        # create directory(ies)
        for filepath in banks_filename_path:
            fdirname = os.path.dirname(filepath)
            try:
                if not os.path.isdir(fdirname):
                    os.mkdir(fdirname)
            except:
                pass

    try:
        # Build the layout on all ranks (deterministic; only rank 0
        # actually uses it for the VDS master, but computing it
        # everywhere is cheap and avoids coordination).
        layout = h5py.VirtualLayout(shape=(nproj,ny,nx), dtype=dtype)
        sitems_per_bank = (vchunks[sitems_idx] + nbanks - 1) // nbanks
        nchunks = (shape[sitems_idx] + vchunks[sitems_idx] - 1) // vchunks[sitems_idx]

        total_bytes = int(np.prod(shape)) * dtp.itemsize
        n_bank_files = nchunks * nbanks
        # How many bank files THIS rank actually creates
        my_n_files = sum(1 for k in range(n_bank_files) if k % size == rank)
        my_bytes = int(total_bytes * my_n_files / max(n_bank_files, 1))
        _t_init0 = time.time()
        _t_last = _t_init0
        if rank == 0:
            print(f"[tomo_initx] {filename}: reserving {total_bytes/1e9:.2f} GB "
                  f"across {n_bank_files} bank files (ALLOC_TIME_EARLY, "
                  f"nchunks={nchunks} × nbanks={nbanks}, sharded across "
                  f"{size} ranks → ~{my_n_files} files/rank, "
                  f"~{my_bytes/1e9:.2f} GB/rank)", flush=True)

        for _ivchunk in range(nchunks):
            for ibank in range(nbanks):
                bank_idx = _ivchunk*nbanks + ibank
                filename_data_file = banks_filename_path[bank_idx]
                filename_data_vsrc = banks_filename_vsrc[bank_idx]
                sitems_offset = _ivchunk*vchunks[sitems_idx]
                sitems_start = sitems_offset + ibank * sitems_per_bank
                sitems_end = sitems_offset + (ibank+1) * sitems_per_bank
                sitems_end = sitems_end if (sitems_end - sitems_offset) < vchunks[sitems_idx] else sitems_offset + vchunks[sitems_idx]
                sitems_end = sitems_end if sitems_end < shape[sitems_idx] else shape[sitems_idx]
                if sitems_end > sitems_start:
                    # Build virtual source (deterministic — all ranks agree).
                    if stype == 'proj':
                        vsource = h5py.VirtualSource(filename_data_vsrc, "/exchange/data", shape=(sitems_end-sitems_start,ny,nx))
                        layout[sitems_start:sitems_end,:,:] = vsource
                    else:
                        vsource = h5py.VirtualSource(filename_data_vsrc, "/exchange/data", shape=(nproj,sitems_end-sitems_start,nx))
                        layout[:,sitems_start:sitems_end,:] = vsource
                    # Only THIS rank's assigned bank files get created here
                    # (round-robin across ranks).  Parallelises the
                    # ALLOC_TIME_EARLY pre-allocation cost.
                    if bank_idx % size == rank:
                        with h5py.File(filename_data_file, 'w') as hf_out:
                            g = hf_out.create_group('/exchange')
                            if stype == 'proj':
                                dset = _create_chunked_early(
                                    g, 'data',
                                    shape=(sitems_end-sitems_start, ny, nx),
                                    dtype=dtype,
                                    chunks=(1,) + vchunks[1:])
                            else:
                                dset = _create_chunked_early(
                                    g, 'data',
                                    shape=(nproj, sitems_end-sitems_start, nx),
                                    dtype=dtype,
                                    chunks=(vchunks[0], 1, vchunks[2]))
            # Progress log — rank 0 only, at most every 5 s.
            _now = time.time()
            _pct = (_ivchunk + 1) / nchunks
            if rank == 0 and (_now - _t_last > 5.0 or _pct >= 1.0):
                _elapsed = _now - _t_init0
                _bytes_so_far = int(total_bytes * _pct)
                _rate = _bytes_so_far / max(_elapsed, 1e-9)
                print(f"[tomo_initx]   {_ivchunk+1}/{nchunks} super-chunks  "
                      f"({100*_pct:.0f}%)  {_elapsed:.1f}s elapsed  "
                      f"({_rate/1e9:.2f} GB/s reserved aggregate)", flush=True)
                _t_last = _now
        # Rank 0 creates the VDS master file (other ranks skip).
        if rank == 0:
            with h5py.File(filename, 'w', libver='latest') as hf:
                dset = hf.create_virtual_dataset('/exchange/data', layout, fillvalue=-5)
                dset.attrs['nbanks_per_svchunk'] = nbanks
                dset.attrs['stype'] = stype
                dset.attrs['vchunks_0'] = vchunks[0]
                dset.attrs['vchunks_1'] = vchunks[1]
                dset.attrs['vchunks_2'] = vchunks[2]
                dset.attrs['meta'] = json.dumps(meta)
            _t_end = time.time() - _t_init0
            print(f"[tomo_initx] {filename}: done  "
                  f"({total_bytes/1e9:.2f} GB / {_t_end:.1f}s = "
                  f"{total_bytes/_t_end/1e9:.2f} GB/s aggregate; "
                  f"parallel-init across {size} ranks)", flush=True)
    finally:
        pass

    ctx = {'banks_filename_path': banks_filename_path, 'banks_size': banks_size}
    return ctx
    
def tomo_writex(filename, data, shm=None, ivchunk=(0,0,0), ctx=None):

    info = tomo_info(filename)

    stype = info['stype']
    shape = info['shape']
    vchunks = info['vchunks']
    meta = info['meta']
    dtp = info['dtype']
    nbanks_per_svchunk = info['nbanks_per_svchunk']
    
    nproj, ny, nx = shape
    assert data.dtype == dtp

    sitems_idx = 0 if stype == 'proj' else 1

    if ctx is None:    
        banks_filename_path,
        banks_size,
        banks_filename_vsrc = _create_banking_plan(filename=filename,
                                                   shape=shape,
                                                   vchunks=vchunks,
                                                   nbanks_per_svchunk=nbanks_per_svchunk,
                                                   sitems_idx = sitems_idx,
                                                   meta=meta)
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
    nproj, ny, nx, dtp, chunks = info['nproj'], info['ny'], info['nx'], info['dtype'], info['chunks']
    assert chunks == (1,) + vchunks[1:]
    
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
    nproj, ny, nx, dtp, chunks = info['nproj'], info['ny'], info['nx'], info['dtype'], info['chunks']
    assert chunks == (vchunks[0], 1, vchunks[2])
    
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

#### Specific I/O

def _process_read_slices(task_meta, fname, vchunksx, sitems_offset, direct_chunk, zerocp, dtype, shm):
        
    out = np.ndarray(shape=vchunksx, dtype=np.float32, buffer=shm.buf)
    buffer = np.empty(vchunksx, dtype=dtype) if not zerocp else None  # intermediate buffer
    
    proj_sel = task_meta['proj_sel']
    
    with h5py.File(fname, 'r') as hf_in:
        dset = hf_in['/exchange/data']
        if direct_chunk:
            if zerocp:
                dset.read_direct(out, source_sel=np.s_[proj_sel[0]:proj_sel[1],:,:], dest_sel=np.s_[proj_sel[0]-sitems_offset:proj_sel[1]-sitems_offset,:,:])
            else:
                dset.read_direct(buffer, source_sel=np.s_[proj_sel[0]:proj_sel[1],:,:], dest_sel=np.s_[proj_sel[0]-sitems_offset:proj_sel[1]-sitems_offset,:,:])
                #np.copyto(dst=out, src=buffer)
                out[proj_sel[0]-sitems_offset:proj_sel[1]-sitems_offset,:,:] = buffer[proj_sel[0]-sitems_offset:proj_sel[1]-sitems_offset,:,:]
        else:
            out[proj_sel[0]-sitems_offset:proj_sel[1]-sitems_offset,:,:] = dset[proj_sel[0]:proj_sel[1],:,:]
            
    return np.arange(proj_sel[0],proj_sel[1]).tolist()
     
def read_projs_vchunkx(fname, shm, ntasks, vchunksx, ivchunkx, shm_ret_meta=None):

    _t = time.time()
    
    with h5py.File(fname, 'r') as fid:
        dset = fid['/exchange/data']
        shape = dset.shape
        dset_is_virtual = dset.is_virtual
        dset_is_float32 = np.dtype(dset.dtype) == np.dtype(np.float32)
        dtp = dset.dtype

    data = np.ndarray(shape=vchunksx, dtype=np.float32, buffer=shm.buf)
    
    sitems_per_task = (vchunksx[0] + ntasks - 1) // ntasks
    sitems_offset = vchunksx[0] * ivchunkx[0]

    task_meta = []
    
    for itask in range(ntasks):
        sitems_start = sitems_offset + itask * sitems_per_task
        sitems_end = sitems_offset + (itask+1) * sitems_per_task
        sitems_end = sitems_end if (sitems_end-sitems_offset) < vchunksx[0] else sitems_offset + vchunksx[0]
        sitems_end = sitems_end if sitems_end < shape[0] else shape[0]
        if sitems_end > sitems_start:
            task_meta.append({'itask': itask, 'proj_sel': (sitems_start, sitems_end)})

    with multiprocessing.Pool(processes=ntasks) as pool:
        results = pool.map(partial(_process_read_slices, fname=fname, vchunksx=vchunksx, sitems_offset=sitems_offset, direct_chunk=not dset_is_virtual, zerocp=dset_is_float32, dtype=dtp, shm=shm), task_meta)
    
    _t = time.time() - _t
 
    if shm_ret_meta is not None:
        ret_meta = np.ndarray(shape=(3,), dtype=np.float64, buffer=shm_ret_meta.buf)
        ret_meta[0] = _t
    
    return data

def write_vchunkx(fname_out, shm, vchunksx, vchunks, ctx, ivchunkx, shm_ret_meta=None):
    _t = time.time()

    data = np.ndarray(shape=vchunksx, dtype=np.float32, buffer=shm.buf)
    
    # vchunksx is larger than vchunks. We have to save it block by block manually with data-copy (i.e. not using shm)
    nblocks = (vchunksx[1] + vchunks[1] - 1) // vchunks[1]
    for iblock in range(nblocks):
        block_start = iblock * vchunks[1]
        block_end = (iblock+1) * vchunks[1]
        block_end = block_end if block_end < vchunksx[1] else vchunksx[1]
        if block_end > block_start:
            tomo_writex(fname_out, data[:,block_start:block_end,:], shm=None, ivchunk=(ivchunkx[0],iblock,ivchunkx[2]), ctx=ctx)
            
    _t = time.time() - _t
 
    if shm_ret_meta is not None:
        ret_meta = np.ndarray(shape=(3,), dtype=np.float64, buffer=shm_ret_meta.buf)
        ret_meta[2] = _t

    return 0
