#!/usr/bin/env python

import h5py
import numpy as np

import multiprocessing
from functools import partial
from multiprocessing import shared_memory

import time

import json
import os

def tomo_info(filename):
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
        with multiprocessing.Pool(processes=ntasks) as pool:
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
            
def tomo_initx(filename, shape, dtype, vchunks=None, mode='a', stype='proj', nbanks=1, meta={}):
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

    # test if VDS present already
    _test = True
    if os.path.isfile(filename):
        with h5py.File(filename, mode, libver='latest') as fid:
            try:
                dset = fid['/exchange/data']
                # this is not ok
                _test = False
            except:
                # this is ok
                pass
    if not _test:
        raise Exception("Virtual dataset exists already.")

    # create directory(ies)
    for filepath in banks_filename_path:
        fdirname = os.path.dirname(filepath)
        try:
            if not os.path.isdir(fdirname):
                #print('creating:', fdirname)
                os.mkdir(fdirname)
        except:
            pass

    try:
        # create a VDS file
        layout = h5py.VirtualLayout(shape=(nproj,ny,nx), dtype=dtype)
        sitems_per_bank = (vchunks[sitems_idx] + nbanks - 1) // nbanks
        nchunks = (shape[sitems_idx] + vchunks[sitems_idx] - 1) // vchunks[sitems_idx]
        for _ivchunk in range(nchunks):
            for ibank in range(nbanks):
                filename_data_file = banks_filename_path[_ivchunk*nbanks+ibank]
                filename_data_vsrc = banks_filename_vsrc[_ivchunk*nbanks+ibank]          
                sitems_offset = _ivchunk*vchunks[sitems_idx]
                sitems_start = sitems_offset + ibank * sitems_per_bank
                sitems_end = sitems_offset + (ibank+1) * sitems_per_bank
                sitems_end = sitems_end if (sitems_end - sitems_offset) < vchunks[sitems_idx] else sitems_offset + vchunks[sitems_idx]
                sitems_end = sitems_end if sitems_end < shape[sitems_idx] else shape[sitems_idx]
                if sitems_end > sitems_start:
                    # create virtual source
                    if stype == 'proj':
                        vsource = h5py.VirtualSource(filename_data_vsrc, "/exchange/data", shape=(sitems_end-sitems_start,ny,nx))
                        layout[sitems_start:sitems_end,:,:] = vsource
                    else:
                        vsource = h5py.VirtualSource(filename_data_vsrc, "/exchange/data", shape=(nproj,sitems_end-sitems_start,nx))
                        layout[:,sitems_start:sitems_end,:] = vsource
                    # create target data file
                    with h5py.File(filename_data_file, 'w') as hf_out: # w ... overwrite mode
                        g = hf_out.create_group('/exchange')
                        if stype == 'proj':
                            dset = g.create_dataset('data', shape=(sitems_end-sitems_start, ny, nx),
                                                    chunks=(1,)+vchunks[1:], dtype=dtype, fillvalue=None)
                        else:
                            dset = g.create_dataset('data', shape=(nproj, sitems_end-sitems_start, nx),
                                                    chunks=(vchunks[0], 1, vchunks[2]), dtype=dtype, fillvalue=None)
        # create master file              
        with h5py.File(filename, 'w', libver='latest') as hf:
            dset = hf.create_virtual_dataset('/exchange/data', layout, fillvalue=-5)
            dset.attrs['nbanks_per_svchunk'] = nbanks
            dset.attrs['stype'] = stype
            dset.attrs['vchunks_0'] = vchunks[0]
            dset.attrs['vchunks_1'] = vchunks[1]
            dset.attrs['vchunks_2'] = vchunks[2]
            dset.attrs['meta'] = json.dumps(meta)
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
        with multiprocessing.Pool(processes=nbanks_per_svchunk) as pool:
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
