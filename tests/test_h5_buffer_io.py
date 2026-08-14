#!/usr/bin/env python
"""RAM-buffer + multi-bank HDF5 I/O benchmark for the mosaic pipeline.

Adapts the pattern from doe-maxiv/doe_chunks_t4_aps.ipynb — one big
shared-memory buffer per stage (a "super-chunk", or vchunk) that holds
many h5 chunks at once, plus a small pool of worker processes fanning
that buffer into per-bank .h5 files behind a top-level VDS.

Design choices:
  1. No MPI-IO.  Each rank / worker opens its own bank file via POSIX,
     so nothing bounces off a single collective-write coordinator.
  2. Many h5 files per dataset.  init.h5 becomes a VDS that references
     init/init_data_000000.h5, init_data_000001.h5, ...  So writes fan
     across --nbanks files in parallel — that's the throughput lever.
  3. Explicit RAM buffer.  Each stage allocates one np.ndarray backed
     by shared_memory.SharedMemory sized to the super-chunk, and the
     worker pool writes slices of that buffer.

MPI (mpi4py, hard dependency).  Ranks shard the vchunk iteration:
    my_ivchunks = ivchunks[RANK::SIZE]
Rank 0 alone calls tomo_initx (VDS + empty bank files); other ranks wait
on MPI.Barrier and then write into the disjoint subset of bank files
they own — nothing needs cross-rank coordination beyond the initial
init + barrier.  Aggregate throughput = sum(bytes) / max(elapsed).

Recommended launch on Polaris:
    mpiexec -n <NNODES> --ppn 1 python test_h5_buffer_io.py ...
(one Python process per node; multiprocessing inside each rank fans
across that node's CPUs).

Uses helpers from ./dxchange_hdf5_chunks.py (vendored from doe-maxiv)
and ./utils.py (MPI wiring).

Example:
    mpiexec -n 10 --ppn 1 python test_h5_buffer_io.py \\
        --path /eagle/APS_IRI/vnikitin/iotest_buf_ups2 \\
        --ups 2 --nbanks 8 --ntasks 8 \\
        --init-vchunks 32 3072 3072 \\
        --big-vchunks  32 5488 5488 \\
        --proj-vchunks 128 32 5488 \\
        --data-vchunks 128 5120 5488

Add --pgn-slice to switch STAGE 3 into "Option B" mode:
  · paganin.h5 becomes a plain slice-stored HDF5 (chunks per z-row)
  · θ-slab writes go through HDF5 read-modify-write
  · ranks serialize writes to avoid concurrent RMW on the same chunk
  · STAGE 4 FBP then hits ALIGNED z-row reads (1× amp vs 365×)
This mirrors step7_paganin_slice.py.  A/B compare by running with and
without the flag.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import h5py
import numpy as np
from multiprocessing import shared_memory

from iohdf5.dxchange_hdf5_chunks import tomo_initx, tomo_readx, tomo_writex
from mpi_utils import MPI, COMM, RANK, SIZE, barrier, rprint, allreduce


def _report_stage(label: str, bytes_local: float, time_local: float) -> None:
    """Print aggregate + per-rank spread for a stage.  Called by ALL ranks."""
    total_bytes  = allreduce(bytes_local, MPI.SUM)
    max_time     = allreduce(time_local,  MPI.MAX)
    min_time     = allreduce(time_local,  MPI.MIN)
    per_rank_bps = bytes_local / max(time_local, 1e-9)
    max_bps      = allreduce(per_rank_bps, MPI.MAX)
    min_bps      = allreduce(per_rank_bps, MPI.MIN)
    aggregate_bps = total_bytes / max(max_time, 1e-9)
    rprint(f"  {label:22s}  aggregate={_hb(aggregate_bps)}/s   "
           f"per-rank[min..max]=[{_hb(min_bps)}/s..{_hb(max_bps)}/s]   "
           f"wall[min..max]=[{min_time:.1f}s..{max_time:.1f}s]  "
           f"total={_hb(total_bytes)}")


def _hb(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"


def _describe(name, shape, vchunks, nbanks, dtype_bytes):
    total = int(np.prod(shape)) * dtype_bytes
    per_vc = int(np.prod(vchunks)) * dtype_bytes
    n_vc = int(np.prod([-(-s // c) for s, c in zip(shape, vchunks)]))
    n_files = n_vc * nbanks + 1  # bank files + master VDS
    print(f"  {name:14s}  shape={tuple(shape)}   dtype=f32  total={_hb(total)}")
    print(f"  {name:14s}  vchunk={tuple(vchunks)}   buffer={_hb(per_vc)}   "
          f"nvchunks={n_vc}   nbanks={nbanks}   files~{n_files}")


def _cleanup_h5(path: str) -> None:
    """Remove the master .h5 file and its sibling bank directory."""
    if os.path.isfile(path):
        os.remove(path)
    base = os.path.splitext(os.path.basename(path))[0]
    subd = os.path.join(os.path.dirname(path) or ".", base)
    if os.path.isdir(subd):
        shutil.rmtree(subd)


def _iter_vchunks(shape, vchunks):
    """Iterate ivchunk tuples like the doe-maxiv notebook does."""
    n0 = (shape[0] + vchunks[0] - 1) // vchunks[0]
    n1 = (shape[1] + vchunks[1] - 1) // vchunks[1]
    n2 = (shape[2] + vchunks[2] - 1) // vchunks[2]
    for i0 in range(n0):
        for i2 in range(n2):
            for i1 in range(n1):
                yield (i0, i1, i2)


def _alloc_shm(shape, dtype):
    """Allocate a shared-memory buffer of the given shape+dtype and
    return (shm, ndarray-view).  Caller must close+unlink shm when done."""
    dtp = np.dtype(dtype)
    shm = shared_memory.SharedMemory(create=True,
                                      size=int(np.prod(shape)) * dtp.itemsize)
    buf = np.ndarray(shape=shape, dtype=dtp, buffer=shm.buf)
    return shm, buf


def _free_shm(shm):
    try:
        shm.close()
    finally:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True)
    p.add_argument("--ups",     type=int, default=1)
    p.add_argument("--in-nz",   type=int, default=3072)
    p.add_argument("--in-nyx",  type=int, default=3072)
    p.add_argument("--ntheta",  type=int, default=None,
                   help="default = 3·N/4 where N = in-nyx·ups")
    p.add_argument("--nbanks",  type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--ntasks",  type=int, default=8,
                   help="worker processes used by tomo_readx (per stage)")
    p.add_argument("--nzchunk", type=int, default=8,
                   help="inner z-slab (== step8_fbp --nzchunk); drives fbp "
                        "per-iteration sinogram read size")
    p.add_argument("--init-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"),
                   help="super-chunk for init.h5 (default fits ~ nbanks planes)")
    p.add_argument("--big-vchunks",  type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"))
    p.add_argument("--proj-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"))
    p.add_argument("--data-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"))
    p.add_argument("--pgn-vchunks",  type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"),
                   help="super-chunk for paganin.h5 (default: nbanks, OUT_NZ, N)")
    p.add_argument("--rec-vchunks",  type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"),
                   help="super-chunk for rec.h5 (default: nbanks, N, N)")
    p.add_argument("--pgn-slice", action="store_true",
                   help="Option B: write paganin.h5 as a plain slice-stored "
                        "HDF5 file (chunks per z-row) with serialized "
                        "partial-θ writes across ranks.  Mirrors "
                        "step7_paganin_slice.py.  STAGE 4 FBP then reads "
                        "aligned z-row chunks (1× amp instead of 365×).")
    return p.parse_args()


# --------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    UPS    = args.ups
    IN_NZ  = args.in_nz
    IN_NYX = args.in_nyx
    OUT_NZ = IN_NZ  * UPS
    N      = IN_NYX * UPS
    NTHETA = args.ntheta if args.ntheta is not None else 3 * N // 4

    # paganin.h5 has half the angles after 180° stitching (matches step7)
    N_HALF = NTHETA // 2

    init_shape = (IN_NZ,  IN_NYX, IN_NYX)
    big_shape  = (OUT_NZ, N,      N     )
    proj_shape = (NTHETA, OUT_NZ, N     )
    data_shape = (NTHETA, OUT_NZ, N     )
    pgn_shape  = (N_HALF, OUT_NZ, N     )
    rec_shape  = (OUT_NZ, N,      N     )

    init_vc = tuple(args.init_vchunks) if args.init_vchunks else (args.nbanks, IN_NYX, IN_NYX)
    big_vc  = tuple(args.big_vchunks)  if args.big_vchunks  else (args.nbanks, N,      N     )
    proj_vc = tuple(args.proj_vchunks) if args.proj_vchunks else (args.nbanks, OUT_NZ, N     )
    data_vc = tuple(args.data_vchunks) if args.data_vchunks else (args.nbanks, OUT_NZ, N     )
    pgn_vc  = tuple(args.pgn_vchunks)  if args.pgn_vchunks  else (args.nbanks, OUT_NZ, N     )
    rec_vc  = tuple(args.rec_vchunks)  if args.rec_vchunks  else (args.nbanks, N,      N     )

    def _validate(name, shape, vc):
        if any(c > s for c, s in zip(vc, shape)):
            axes = ", ".join(f"axis{i}: vchunk {c} > shape {s}"
                             for i, (c, s) in enumerate(zip(vc, shape)) if c > s)
            raise SystemExit(f"[test_h5_buffer_io] {name} vchunks={vc} exceed "
                             f"shape={shape} ({axes}).")
    _validate("init.h5",       init_shape, init_vc)
    _validate(f"big{UPS}x.h5", big_shape,  big_vc)
    _validate("proj.h5",       proj_shape, proj_vc)
    _validate("data.h5",       data_shape, data_vc)
    _validate("paganin.h5",    pgn_shape,  pgn_vc)
    _validate("rec.h5",        rec_shape,  rec_vc)

    os.makedirs(args.path, exist_ok=True)
    INIT = os.path.join(args.path, "init.h5")
    BIG  = os.path.join(args.path, f"big{UPS}x.h5")
    PROJ = os.path.join(args.path, "proj.h5")
    DATA = os.path.join(args.path, "data.h5")
    PGN  = os.path.join(args.path, "paganin.h5")
    REC  = os.path.join(args.path, "rec.h5")

    rprint(f"[test_h5_buffer_io]  UPS={UPS}   nbanks={args.nbanks}   "
           f"ntasks={args.ntasks}   MPI ranks={SIZE}")
    rprint("")
    rprint("File layout (each dataset = master VDS + nvchunks·nbanks bank files):")
    if RANK == 0:
        _describe("init.h5",       init_shape, init_vc, args.nbanks, 4)
        _describe(f"big{UPS}x.h5", big_shape,  big_vc,  args.nbanks, 4)
        _describe("proj.h5",       proj_shape, proj_vc, args.nbanks, 4)
        _describe("data.h5",       data_shape, data_vc, args.nbanks, 4)
        _describe("paganin.h5",    pgn_shape,  pgn_vc,  args.nbanks, 4)
        _describe("rec.h5",        rec_shape,  rec_vc,  args.nbanks, 4)
    rprint("")

    dtype = np.float32
    dtp   = np.dtype(dtype)
    # per-rank seed keeps random content distinct across ranks (so h5's dedup
    # heuristics can't hide bandwidth) but reproducible per rank.
    rng   = np.random.default_rng(1234 + RANK)

    # ================== STAGE 1: seed init + upsample -> big ==============
    rprint("─" * 70)
    rprint("STAGE 1  init.h5 ── read ─▶ big.h5 ── write   (per super-chunk)")
    rprint("─" * 70)

    # Rank 0 creates master VDS + all empty bank files; others wait.
    if RANK == 0:
        _cleanup_h5(INIT)
        ctx_init = tomo_initx(filename=INIT, shape=init_shape, dtype=dtype,
                              vchunks=init_vc, stype="proj", nbanks=args.nbanks)
    else:
        ctx_init = None
    barrier()
    
    ctx_init = COMM.bcast(ctx_init, root=0)

    shm_i, buf_i = _alloc_shm(init_vc, dtype)
    rprint(f"  buffer for init: {_hb(buf_i.nbytes)}   ({init_vc})   "
           f"(per rank; {SIZE} ranks total)")

    ivchunks = list(_iter_vchunks(init_shape, init_vc))
    my_ivchunks = ivchunks[RANK::SIZE]     # <-- MPI shard
    total = len(ivchunks)
    my_total = len(my_ivchunks)
    step = max(1, my_total // 10) if my_total else 1

    t_seed = 0.0
    n_vc_init = 0
    for k, ivc in enumerate(my_ivchunks, start=1):
        z0 = ivc[0] * init_vc[0]
        z1 = min(z0 + init_vc[0], init_shape[0])
        buf_i[: z1 - z0].fill(0)
        buf_i[: z1 - z0] = rng.random(
            (z1 - z0, init_vc[1], init_vc[2]), dtype=np.float32)
        t = time.perf_counter()
        tomo_writex(INIT, data=buf_i, shm=shm_i, ivchunk=ivc, ctx=ctx_init)
        t_seed += time.perf_counter() - t
        n_vc_init += 1
        if (k % step == 0 or k == my_total) and RANK == 0:
            print(f"    [rank 0] init  vchunk {k}/{my_total} "
                  f"(of {total} global)  (write={t_seed:.1f}s)", flush=True)
    bytes_seed = n_vc_init * int(np.prod(init_vc)) * dtp.itemsize
    barrier()
    _report_stage("init.h5 seed", bytes_seed, t_seed)

    if RANK == 0:
        _cleanup_h5(BIG)
        ctx_big = tomo_initx(filename=BIG, shape=big_shape, dtype=dtype,
                             vchunks=big_vc, stype="proj", nbanks=args.nbanks)
    else:
        ctx_big = None
    barrier()
    
    ctx_big = COMM.bcast(ctx_big, root=0)

    shm_b, buf_b = _alloc_shm(big_vc, dtype)
    rprint(f"  buffer for big:  {_hb(buf_b.nbytes)}   ({big_vc})")

    big_ivchunks = list(_iter_vchunks(big_shape, big_vc))
    my_big = big_ivchunks[RANK::SIZE]
    total = len(big_ivchunks)
    my_total = len(my_big)
    step = max(1, my_total // 10) if my_total else 1

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_big, start=1):
        z0_out = ivc[0] * big_vc[0]
        z1_out = min(z0_out + big_vc[0], big_shape[0])
        z0_in = z0_out // UPS
        z1_in = (z1_out + UPS - 1) // UPS
        in_ivc = (z0_in // init_vc[0], 0, 0)
        t = time.perf_counter()
        src = tomo_readx(INIT, ntasks=args.ntasks, shm=shm_i,
                         ivchunk=in_ivc, vchunks=init_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(init_vc)) * dtp.itemsize

        local_z_lo = z0_in - in_ivc[0] * init_vc[0]
        local_z_hi = z1_in - in_ivc[0] * init_vc[0]
        planes_in = src[local_z_lo:local_z_hi]
        up = np.repeat(np.repeat(planes_in, UPS, axis=1), UPS, axis=2)
        up = np.repeat(up, UPS, axis=0).astype(np.float32, copy=False)
        buf_b[: z1_out - z0_out] = up[: z1_out - z0_out]
        del up

        t = time.perf_counter()
        tomo_writex(BIG, data=buf_b, shm=shm_b, ivchunk=ivc, ctx=ctx_big)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(big_vc)) * dtp.itemsize
        if (k % step == 0 or k == my_total) and RANK == 0:
            print(f"    [rank 0] upsample {k}/{my_total} (of {total} global)  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    barrier()
    _report_stage("stage 1  read",  bytes_read,  t_read)
    _report_stage("stage 1  write", bytes_write, t_write)
    _free_shm(shm_i)
    rprint("")

    # ================== STAGE 2 RADON: big -> proj =========================
    rprint("─" * 70)
    rprint("STAGE 2 RADON   big.h5 ── read ─▶ proj.h5 ── write   (per super-chunk)")
    rprint("─" * 70)

    if RANK == 0:
        _cleanup_h5(PROJ)
        ctx_proj = tomo_initx(filename=PROJ, shape=proj_shape, dtype=dtype,
                              vchunks=proj_vc, stype="proj", nbanks=args.nbanks)
    else:
        ctx_proj = None
    barrier()
    
    ctx_proj = COMM.bcast(ctx_proj, root=0)

    shm_p, buf_p = _alloc_shm(proj_vc, dtype)
    rprint(f"  buffer for proj: {_hb(buf_p.nbytes)}   ({proj_vc})")

    proj_ivchunks = list(_iter_vchunks(proj_shape, proj_vc))
    my_proj = proj_ivchunks[RANK::SIZE]
    total = len(proj_ivchunks)
    my_total = len(my_proj)
    step = max(1, my_total // 10) if my_total else 1
    fake_proj = rng.random(proj_vc, dtype=np.float32)

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_proj, start=1):
        z0 = ivc[1] * proj_vc[1]
        big_i = (z0 // big_vc[0], 0, 0)
        t = time.perf_counter()
        _ = tomo_readx(BIG, ntasks=args.ntasks, shm=shm_b,
                       ivchunk=big_i, vchunks=big_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(big_vc)) * dtp.itemsize

        buf_p[:] = fake_proj

        t = time.perf_counter()
        tomo_writex(PROJ, data=buf_p, shm=shm_p, ivchunk=ivc, ctx=ctx_proj)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(proj_vc)) * dtp.itemsize
        if (k % step == 0 or k == my_total) and RANK == 0:
            print(f"    [rank 0] radon {k}/{my_total} (of {total} global)  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    barrier()
    _report_stage("radon    read",  bytes_read,  t_read)
    _report_stage("radon    write", bytes_write, t_write)
    _free_shm(shm_b)
    rprint("")

    # ================== STAGE 2 PROPAGATION: proj -> data ======================
    rprint("─" * 70)
    rprint("STAGE 2 PROPAGATION  proj.h5 ── read ─▶ data.h5 ── write  (per super-chunk)")
    rprint("─" * 70)

    if RANK == 0:
        _cleanup_h5(DATA)
        ctx_data = tomo_initx(filename=DATA, shape=data_shape, dtype=dtype,
                              vchunks=data_vc, stype="proj", nbanks=args.nbanks)
    else:
        ctx_data = None
    barrier()
    
    ctx_data = COMM.bcast(ctx_data, root=0)

    shm_d, buf_d = _alloc_shm(data_vc, dtype)
    rprint(f"  buffer for data: {_hb(buf_d.nbytes)}   ({data_vc})")

    data_ivchunks = list(_iter_vchunks(data_shape, data_vc))
    my_data = data_ivchunks[RANK::SIZE]
    total = len(data_ivchunks)
    my_total = len(my_data)
    step = max(1, my_total // 10) if my_total else 1

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_data, start=1):
        proj_i = (ivc[0], 0, 0)
        t = time.perf_counter()
        src = tomo_readx(PROJ, ntasks=args.ntasks, shm=shm_p,
                         ivchunk=proj_i, vchunks=proj_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(proj_vc)) * dtp.itemsize

        buf_d[: src.shape[0], : src.shape[1], : src.shape[2]] = src

        t = time.perf_counter()
        tomo_writex(DATA, data=buf_d, shm=shm_d, ivchunk=ivc, ctx=ctx_data)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(data_vc)) * dtp.itemsize
        if (k % step == 0 or k == my_total) and RANK == 0:
            print(f"    [rank 0] propagation {k}/{my_total} (of {total} global)  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    barrier()
    _report_stage("propagation  read",  bytes_read,  t_read)
    _report_stage("propagation  write", bytes_write, t_write)
    _free_shm(shm_p)
    _free_shm(shm_d)
    rprint("")

    # ================== STAGE 3 PAGANIN: data -> paganin ===================
    rprint("─" * 70)
    mode_tag = "SLICE-STORED (Option B)" if args.pgn_slice else "PROJ-STORED (OLD)"
    rprint(f"STAGE 3 PAGANIN     data.h5 ── read ─▶ paganin.h5 ── write  "
           f"[{mode_tag}]")
    rprint("─" * 70)
    # OLD    : mimics step7_paganin.py — proj-stored VDS+banks, aligned writes.
    # Option B: mimics step7_paganin_slice.py — plain HDF5 slice-stored
    #           (chunks per z-row), θ-slab writes go through HDF5 RMW.
    #           Multiple ranks cannot RMW the same chunk concurrently, so
    #           writes are serialized across ranks per iteration.

    if args.pgn_slice:
        # Plain slice-stored HDF5 file (no VDS+banks)
        if RANK == 0:
            _cleanup_h5(PGN)
            with h5py.File(PGN, "w") as fp:
                g = fp.create_group("exchange")
                g.create_dataset(
                    "data",
                    shape=pgn_shape, dtype=dtype,
                    chunks=(pgn_shape[0], 1, pgn_shape[2]),   # (N_HALF, 1, N)
                )
        ctx_pgn = None
    else:
        if RANK == 0:
            _cleanup_h5(PGN)
            ctx_pgn = tomo_initx(filename=PGN, shape=pgn_shape, dtype=dtype,
                                 vchunks=pgn_vc, stype="proj", nbanks=args.nbanks)
        else:
            ctx_pgn = None
    barrier()

    if not args.pgn_slice:
        ctx_pgn = COMM.bcast(ctx_pgn, root=0)

    shm_pg, buf_pg = _alloc_shm(pgn_vc, dtype)
    rprint(f"  buffer for paganin: {_hb(buf_pg.nbytes)}   ({pgn_vc})")

    pgn_ivchunks = list(_iter_vchunks(pgn_shape, pgn_vc))
    my_pgn = pgn_ivchunks[RANK::SIZE]
    total = len(pgn_ivchunks)
    my_total = len(my_pgn)
    step = max(1, my_total // 10) if my_total else 1
    fake_pgn = rng.random(pgn_vc, dtype=np.float32)

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_pgn, start=1):
        t0_vc = ivc[0] * pgn_vc[0]
        t1_vc = min(t0_vc + pgn_vc[0], pgn_shape[0])

        t = time.perf_counter()
        with h5py.File(DATA, "r") as fp:
            _ = fp["exchange/data"][t0_vc:t1_vc, :pgn_vc[1], :pgn_vc[2]]
        t_read += time.perf_counter() - t
        bytes_read += (t1_vc - t0_vc) * pgn_vc[1] * pgn_vc[2] * dtp.itemsize

        buf_pg[:] = fake_pgn

        t = time.perf_counter()
        if args.pgn_slice:
            # Serialized partial-θ write to slice-stored plain HDF5.
            # Each rank writes its own θ-slab; ranks take turns to avoid
            # concurrent RMW on the same z-row chunks.
            for r in range(SIZE):
                if r == RANK:
                    with h5py.File(PGN, "r+") as fp:
                        fp["exchange/data"][t0_vc:t1_vc, :, :] = buf_pg
                barrier()
        else:
            tomo_writex(PGN, data=buf_pg, shm=shm_pg, ivchunk=ivc, ctx=ctx_pgn)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(pgn_vc)) * dtp.itemsize
        if (k % step == 0 or k == my_total) and RANK == 0:
            print(f"    [rank 0] paganin {k}/{my_total} (of {total} global)  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    barrier()
    _report_stage("paganin  read",  bytes_read,  t_read)
    _report_stage("paganin  write", bytes_write, t_write)
    rprint("")

    # ================== STAGE 4 FBP: paganin -> rec ========================
    rprint("─" * 70)
    rprint("STAGE 4 FBP        paganin.h5 ── SLICE read ─▶ rec.h5 ── write  "
           "(per super-chunk)")
    rprint("─" * 70)
    # Mimics step8_fbp.py: for each rec.h5 super-chunk of (VZ, N, N),
    # loops nzchunk-sized z-slabs and reads a sinogram slab
    # [:, zc0:zc1, :] from proj-stored paganin.h5 — cross-bank access
    # that stresses VDS resolution (spans every θ-bank of the source).

    if RANK == 0:
        _cleanup_h5(REC)
        ctx_rec = tomo_initx(filename=REC, shape=rec_shape, dtype=dtype,
                             vchunks=rec_vc, stype="proj", nbanks=args.nbanks)
    else:
        ctx_rec = None
    barrier()

    ctx_rec = COMM.bcast(ctx_rec, root=0)

    shm_r, buf_r = _alloc_shm(rec_vc, dtype)
    rprint(f"  buffer for rec:     {_hb(buf_r.nbytes)}   ({rec_vc})")

    rec_ivchunks = list(_iter_vchunks(rec_shape, rec_vc))
    my_rec = rec_ivchunks[RANK::SIZE]
    total = len(rec_ivchunks)
    my_total = len(my_rec)
    step = max(1, my_total // 10) if my_total else 1
    fake_rec = rng.random(rec_vc, dtype=np.float32)

    NZCHUNK = args.nzchunk

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_rec, start=1):
        z0_vc = ivc[0] * rec_vc[0]
        z1_vc = min(z0_vc + rec_vc[0], rec_shape[0])

        for zc0 in range(z0_vc, z1_vc, NZCHUNK):
            zc1 = min(zc0 + NZCHUNK, z1_vc)
            t = time.perf_counter()
            with h5py.File(PGN, "r") as fp:
                _ = fp["exchange/data"][:, zc0:zc1, :]
            t_read += time.perf_counter() - t
            bytes_read += pgn_shape[0] * (zc1 - zc0) * pgn_shape[2] * dtp.itemsize

        buf_r[:] = fake_rec

        t = time.perf_counter()
        tomo_writex(REC, data=buf_r, shm=shm_r, ivchunk=ivc, ctx=ctx_rec)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(rec_vc)) * dtp.itemsize
        if (k % step == 0 or k == my_total) and RANK == 0:
            print(f"    [rank 0] fbp {k}/{my_total} (of {total} global)  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    barrier()
    _report_stage("fbp      read",  bytes_read,  t_read)
    _report_stage("fbp      write", bytes_write, t_write)
    _free_shm(shm_pg)
    _free_shm(shm_r)


if __name__ == "__main__":
    main()
