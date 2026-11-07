#!/usr/bin/env python
"""FBP of the folded 180 deg data with tomocupy's own Fourier-based method.

Input:  {path}/model_big1x/paganin.h5   (NT, NZ, N) float32, -log(Paganin),
        theta 0..180 deg; the fold puts the rotation axis at column N/2.
Output: {path}/model_big1x/rec.h5       (NOUT, N, N) float32, one chunk per
        slice; slice k = stitched row R0 + k.  By default the stitcher's
        padding rows are left out (real_rows from stitched.h5).

The backprojection is exactly tomocupy's (recon_steps, fourierrec):
  fbp_filter_center  edge-padded to ne = 4N, filter x exp(-2 pi i
                     (-center + N/2) t) on the GPU (cfunc_filter)
  FourierRec         slice pairs packed as re + i*im, USFFT with the whole
                     (2N+2m)^2 grid on the GPU, binned scatter (cfunc_fourierrec)
so the values are in tomocupy's units.

Per rank (one GPU): slabs of --nzslab slices, round-robin over ranks.  The
next slab is read in a background thread and the previous one written in
another while the GPU works.  Writes are pwrites into the slice's
pre-allocated chunk (no MPI-IO); MPI is used only for the create barrier.

    mpirun -n R set_affinity_gpu.sh python step8_fbp_fold.py --path OUT \\
        --filter shepp

Run in the tomocupy-mpi env.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import h5py
import cupy as cp
from mpi4py import MPI

from tomocupy.reconstruction.fourierrec import FourierRec
from tomocupy.reconstruction.fbp_filter import FBPFilter

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.rank, COMM.size


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True)
    p.add_argument("--filter", default="shepp")
    p.add_argument("--center", type=float, default=None,
                   help="tomocupy centre in the folded frame (default N/2)")
    p.add_argument("--rows", type=int, nargs=2, default=None,
                   help="rows [R0, R1) of paganin.h5 (default: the real rows)")
    p.add_argument("--nzslab", type=int, default=4,
                   help="slices per FourierRec call (even); GPU grid is "
                        "nzslab/2 x (2N)^2 complex64")
    a = p.parse_args()
    if a.nzslab % 2:
        raise SystemExit("--nzslab must be even")

    d = f"{a.path}/model_big1x"
    src_fn, out_fn = f"{d}/paganin.h5", f"{d}/rec.h5"
    with h5py.File(src_fn, "r") as f:
        NT, NZ, N = f["exchange/data"].shape
        theta_deg = f["exchange/theta"][:].astype(np.float64)
    real = NZ
    if os.path.exists(f"{d}/stitched.h5"):
        with h5py.File(f"{d}/stitched.h5", "r") as f:
            real = int(f["exchange/data"].attrs.get("real_rows", NZ))
    R0, R1 = a.rows if a.rows else (0, real)
    NOUT = R1 - R0
    center = N / 2 if a.center is None else a.center
    slabs = list(range(R0, R1, a.nzslab))
    mine = slabs[RANK::SIZE]
    log(f"src ({NT}, {NZ}, {N}) theta {theta_deg[0]:.3f}..{theta_deg[-1]:.3f}; "
        f"centre {center} (axis column N/2 = {N / 2}); rows [{R0},{R1}) = "
        f"{NOUT} slices, {len(slabs)} slabs of {a.nzslab} over {SIZE} ranks; "
        f"filter {a.filter}; out ({NOUT}, {N}, {N}) = "
        f"{NOUT * N * N * 4 / 1e12:.2f} TB -> {out_fn}")

    # ---- output: rank 0 creates it, every chunk allocated, 4 KiB aligned
    if RANK == 0:
        if os.path.exists(out_fn):
            os.remove(out_fn)
        with h5py.File(out_fn, "w", alignment_threshold=1,
                       alignment_interval=4096) as f:
            g = f.create_group("exchange")
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            dcpl.set_chunk((1, N, N))
            dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
            dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
            dsid = h5py.h5d.create(g.id, b"data", h5py.h5t.IEEE_F32LE,
                                   h5py.h5s.create_simple((NOUT, N, N)), dcpl=dcpl)
            dset = h5py.Dataset(dsid)
            for k, v in dict(source=src_fn, rows=(R0, R1), start_row=R0,
                             center=center, fbp_filter=a.filter,
                             method="tomocupy fourierrec (folded 180 deg)",
                             axes="z:y:x", description="ReconData").items():
                dset.attrs[k] = v
            g.create_dataset("theta", data=theta_deg.astype(np.float32))
    COMM.Barrier()
    with h5py.File(out_fn, "r") as f:
        dsid = f["exchange/data"].id
        off = np.zeros(NOUT, np.int64)
        for i in range(dsid.get_num_chunks()):
            ci = dsid.get_chunk_info(i)
            off[ci.chunk_offset[0]] = ci.byte_offset
    fo = os.open(out_fn, os.O_WRONLY)

    # ---- tomocupy filter + FourierRec on this rank's GPU
    nzs = a.nzslab
    ne = 4 * N
    theta_gpu = cp.asarray((theta_deg / 180 * np.pi).astype(np.float32))
    rec = FourierRec(N, NT, nzs, theta_gpu, "float32")
    flt = FBPFilter(ne, NT, 1, "float32")
    wfilter = flt.calc_filter(a.filter)
    t_ = cp.fft.rfftfreq(ne).astype("float32")
    w = (wfilter * cp.exp(-2 * cp.complex64(cp.pi * 1j) *
                          (-center + N / 2) * t_))[None]
    pad = ne // 2 - N // 2
    tmp = cp.zeros((1, NT, ne), cp.float32)
    stream = cp.cuda.get_current_stream()
    sino_g = cp.zeros((nzs, NT, N), cp.float32)
    obj_g = cp.zeros((nzs, N, N), cp.float32)
    outs = [cp.cuda.alloc_pinned_memory(nzs * N * N * 4) for _ in range(2)]
    outs = [np.frombuffer(m, np.float32, nzs * N * N).reshape(nzs, N, N)
            for m in outs]

    src = h5py.File(src_fn, "r")
    d_in = src["exchange/data"]
    reader = ThreadPoolExecutor(1)
    writer = ThreadPoolExecutor(1)
    wpool = ThreadPoolExecutor(8)

    def read(z0):
        z1 = min(z0 + nzs, R1)
        t = time.time()
        x = np.empty((NT, z1 - z0, N), np.float32)
        d_in.read_direct(x, source_sel=np.s_[:, z0:z1, :])
        return z0, z1, x, time.time() - t

    def write(buf, z0, nsl):
        t = time.time()
        piece = 64 << 20
        jobs = []
        for k in range(nsl):
            mv = memoryview(buf[k]).cast("B")
            o = int(off[z0 - R0 + k])
            for p0 in range(0, len(mv), piece):
                jobs.append((o + p0, mv[p0:p0 + piece]))

        def put(job):
            o, mv = job
            pos = 0
            while pos < len(mv):
                pos += os.pwrite(fo, mv[pos:], o + pos)

        list(wpool.map(put, jobs))
        return time.time() - t

    t0 = time.time()
    tr = tc = tw = 0.0
    pend = [None, None]                           # write in flight per buffer
    nxt = reader.submit(read, mine[0]) if mine else None
    for i in range(len(mine)):
        z0, z1, sino, dt = nxt.result()
        tr += dt
        if i + 1 < len(mine):
            nxt = reader.submit(read, mine[i + 1])
        t = time.time()
        nsl = z1 - z0
        sino_g[:] = 0
        sino_g[:nsl] = cp.asarray(sino).swapaxes(0, 1)
        for k in range(nsl):                      # tomocupy fbp_filter_center
            dk = sino_g[k:k + 1]
            tmp[:, :, pad:pad + N] = dk
            tmp[:, :, :pad] = dk[:, :, :1]
            tmp[:, :, pad + N:] = dk[:, :, -1:]
            flt.filter(tmp, w, stream)
            dk[:] = tmp[:, :, pad:pad + N]
        obj_g[:] = 0
        rec.backprojection(obj_g, sino_g, stream)
        buf = outs[i % 2]
        if pend[i % 2] is not None:               # this buffer's last write
            tw += pend[i % 2].result()
            pend[i % 2] = None
        obj_g.get(out=buf)
        stream.synchronize()
        tc += time.time() - t
        pend[i % 2] = writer.submit(write, buf, z0, nsl)
        print(f"  [rank {RANK}] slab {i + 1}/{len(mine)} rows [{z0},{z1})  "
              f"read={tr:.0f}s fbp={tc:.0f}s write={tw:.0f}s "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    for f_ in pend:
        if f_ is not None:
            tw += f_.result()
    os.fsync(fo)
    os.close(fo)
    for ex in (reader, writer, wpool):
        ex.shutdown()
    src.close()
    print(f"  [rank {RANK}] done: read={tr:.0f}s fbp={tc:.0f}s write={tw:.0f}s "
          f"total={time.time() - t0:.0f}s", flush=True)
    COMM.Barrier()
    log(f"wrote {NOUT} slices to {out_fn} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
