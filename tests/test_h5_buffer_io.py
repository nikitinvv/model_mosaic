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

Every super-chunk and HDF5 chunk shape comes from iohdf5/layout.py — the
same byte-budget policy the pipeline steps use — so the two knobs that
actually decide the layout are --mem-budget (GiB/rank) and --chunk-bytes
(MiB).  The six --*-vchunks flags remain as per-dataset overrides.

At UPS >= 8 the full volumes are 54 TB (big) to 3.4 PB (rec at UPS=32),
which no test filesystem holds and which would make rank 0 create ~10^5
empty bank files per dataset before a single byte moves.  --max-vchunks
truncates every dataset along its banked axis to that many super-chunks,
leaving the planned vchunk / chunk / nbanks shapes — the things being
measured — exactly as they are at full size.

Example:
    # UPS=1: whole volume, sweep the chunk size
    mpiexec -n 8 --ppn 4 python -m tests.test_h5_buffer_io \\
        --path /eagle/APS_IRI/vnikitin/iotest_buf_ups1 \\
        --ups 1 --nbanks 4 --ntasks 4 --chunk-bytes 64 --mem-budget 96

    # UPS=16: real chunk shapes, 8 super-chunks per dataset
    mpiexec -n 8 --ppn 4 python -m tests.test_h5_buffer_io \\
        --path /eagle/APS_IRI/vnikitin/iotest_buf_ups16 \\
        --ups 16 --nbanks 4 --ntasks 4 --chunk-bytes 64 --mem-budget 96 \\
        --max-vchunks 8
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import h5py
import numpy as np
from multiprocessing import shared_memory

from iohdf5.layout import add_layout_args, plan_pipeline, describe_plan
from iohdf5.dxchange_hdf5_chunks import (
    tomo_initx, tomo_readx, tomo_writex,
    read_projs_vchunkx, read_slices_vchunkx,
)
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


def _describe(name, shape, vchunks, nbanks, dtype_bytes, chunks=None):
    total = int(np.prod(shape)) * dtype_bytes
    per_vc = int(np.prod(vchunks)) * dtype_bytes
    n_vc = int(np.prod([-(-s // c) for s, c in zip(shape, vchunks)]))
    n_files = n_vc * nbanks + 1  # bank files + master VDS
    if chunks is None:
        chunks = (1,) + tuple(vchunks[1:])
    print(f"  {name:14s}  shape={tuple(shape)}   dtype=f32  total={_hb(total)}")
    print(f"  {name:14s}  vchunk={tuple(vchunks)}   buffer={_hb(per_vc)}   "
          f"nvchunks={n_vc}   nbanks={nbanks}   files~{n_files}")
    print(f"  {name:14s}  h5chunk={tuple(chunks)} "
          f"({_hb(int(np.prod(chunks)) * dtype_bytes)})"
          f"{'   [sinogram-ordered]' if chunks[1] == 1 else ''}")


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


def _fill_random(buf, rng):
    """Fill a super-chunk buffer in place, one axis-0 slab at a time.

    `rng.random(buf.shape)` would materialise a second array the size of
    the buffer before the copy — 40 GiB for a proj super-chunk at UPS=8,
    which is exactly the duplicate that makes 4 ranks/node overrun a
    512 GB node.  Random content matters (it defeats any dedup/compress
    heuristic that would flatter the bandwidth number), the *pattern*
    does not, so generate it a slab at a time straight into the buffer.
    """
    slab = max(1, (64 * 2 ** 20) // max(1, buf[0].nbytes))
    for z0 in range(0, buf.shape[0], slab):
        z1 = min(z0 + slab, buf.shape[0])
        buf[z0:z1] = rng.random((z1 - z0,) + buf.shape[1:], dtype=np.float32)


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
    p.add_argument("--max-vchunks", type=int, default=0,
                   help="write at most this many super-chunks per dataset "
                        "(0 = the whole volume).  Truncates each dataset "
                        "along its BANKED axis only, so the vchunk / chunk / "
                        "nbanks shapes -- what this benchmark measures -- are "
                        "the full-size ones.  Needed from UPS=8 on: rec.h5 is "
                        "54 TB at UPS=8 and 3.4 PB at UPS=32, and rank 0 "
                        "would create ~1e5 empty bank files per dataset "
                        "before any I/O happens.  Make it a multiple of the "
                        "rank count so no rank idles.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned layout, the bytes each stage will "
                        "write and the bank-file count, then exit without "
                        "touching the filesystem.  Use it to size a run "
                        "(and a --max-vchunks) before spending a job on it.")
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
    p.add_argument("--pgn-chunk-order", choices=("sino", "proj"),
                   default="sino",
                   help="HDF5 chunk order inside paganin.h5's bank files "
                        "(matches step7 --chunk-order).  'sino' = "
                        "(θ_per_bank, pgn_chunk_z, N), aligned with the "
                        "stage-4 FBP z-slab read.  'proj' = (1, NZ, N), the "
                        "old layout — use it to A/B the stage-4 read.")
    p.add_argument("--pgn-chunk-z", type=int, default=0,
                   help="z-extent of the sinogram chunk (matches step7 "
                        "--chunk-z).  0 (default) = the stage-4 FBP z-slab, "
                        "rec-vchunks C0, which makes that read one whole-"
                        "chunk sequential op per bank file.  Set 1 for the "
                        "pure-sinogram extreme (one HDF5 op per z row per "
                        "bank — latency-bound on Lustre) to A/B it.")
    add_layout_args(p)
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

    # Defaults come from the shared byte-budget policy, so this harness
    # measures the layout the steps will actually write.  A flat `nbanks`
    # planes per super-chunk is 288 MB at UPS=1 but 1.2 TB at UPS=32, and
    # the (1, ny, nx) chunk tomo_initx defaults to is past HDF5's 4 GiB
    # limit from UPS=16 on -- neither survives the sweep this script exists
    # to run.  Every --*-vchunks flag still overrides its dataset.
    #
    # NOTE the harness banks proj.h5 on θ (stype='proj'), unlike step2
    # which banks it on z, so it takes the `data` plan for both: same
    # shape, same banked axis, same chunk.
    plans = plan_pipeline(UPS, in_nz=IN_NZ, in_nyx=IN_NYX, ntheta=NTHETA,
                          nbanks=args.nbanks,
                          budget=int(args.mem_budget * 2 ** 30),
                          chunk_bytes=int(args.chunk_bytes * 2 ** 20),
                          nzchunk=args.nzchunk, nranks=SIZE)
    if RANK == 0:
        print(f"[layout] budget={args.mem_budget} GiB/rank  "
              f"chunk~{args.chunk_bytes} MiB  ranks={SIZE}", flush=True)
        describe_plan(plans)

    def _pick(flag, plan):
        """CLI override, else the planned (vchunks, chunks, nbanks)."""
        if flag:
            vc = tuple(int(v) for v in flag)
            bank1 = (vc[0] + plan.nbanks - 1) // plan.nbanks
            ch = tuple(min(c, b) for c, b in
                       zip(plan.chunks, (bank1,) + vc[1:]))
            return vc, ch, plan.nbanks
        return plan.vchunks, plan.chunks, plan.nbanks

    init_vc, init_ch, init_nb = _pick(args.init_vchunks, plans["init"])
    big_vc,  big_ch,  big_nb  = _pick(args.big_vchunks,  plans["big"])
    proj_vc, proj_ch, proj_nb = _pick(args.proj_vchunks, plans["data"])
    data_vc, data_ch, data_nb = _pick(args.data_vchunks, plans["data"])
    pgn_vc,  pgn_ch,  pgn_nb  = _pick(args.pgn_vchunks,  plans["paganin"])
    rec_vc,  rec_ch,  rec_nb  = _pick(args.rec_vchunks,  plans["rec"])

    # ---- stage 1's read buffer is a harness artifact, not the plan --------
    # Every other stage reads a super-chunk its producer wrote, so the pair
    # of buffers it holds is the pair the policy planned.  Stage 1 is the
    # exception: it prefetches a WHOLE init super-chunk, while step1 streams
    # input planes (which is why big's companion_unit is one thin slab, not
    # 54 GiB).  Left alone that makes the harness peak at 54 + 72 = 126 GiB
    # a rank at UPS=32 -- over a 512 GB node at 4 ranks.  Shrink the read
    # side instead: more read calls, each still one whole sequential
    # super-chunk of the same chunk shape, so nothing measured changes.
    budget_b = int(args.mem_budget * 2 ** 30)
    room = max(1, (budget_b - int(np.prod(big_vc)) * 4)
               // (init_shape[1] * init_shape[2] * 4))
    if init_vc[0] > room:
        c0 = next((c for c in range(min(init_vc[0], (room // init_nb) * init_nb),
                                    0, -init_nb) if init_shape[0] % c == 0),
                  init_nb if init_shape[0] % init_nb == 0 else 1)
        if RANK == 0:
            print(f"[layout] init read super-chunk {init_vc[0]} -> {c0} planes "
                  f"so it fits beside the big write buffer "
                  f"({_hb(int(np.prod(big_vc)) * 4)}) inside "
                  f"{args.mem_budget} GiB", flush=True)
        init_vc = (c0,) + tuple(init_vc[1:])
        bank0 = max(1, c0 // init_nb)
        init_ch = (init_ch[0], min(init_ch[1], init_shape[1]),
                   min(init_ch[2], init_shape[2]))
        init_ch = (min(init_ch[0], bank0),) + init_ch[1:]

    # paganin.h5 is θ-banked (ranks shard on θ, one writer per bank file) but
    # sinogram-chunked, so stage 4's (NTHETA, zslab, N) read covers whole
    # chunks.  Both chunk extents matter and they fail differently:
    #   θ  — θ_per_bank, so a chunk never straddles two bank files and the
    #        read's θ-sharded workers each get whole chunks.
    #   z  — the FBP z-slab (rec_vc[0]).  At z-extent 1 the read is correct
    #        and unamplified but costs rec_vc[0] HDF5 ops per bank file
    #        instead of 1, and on Lustre that per-op latency is the whole
    #        cost.  Matching the slab turns stage 4 into one sequential
    #        whole-chunk read per bank file.
    # The old (1, NZ, N) projection chunk is the 'proj' branch: same bytes
    # off disk (HDF5 does partial-chunk I/O on unfiltered datasets) but
    # strided NZ·N·4 apart, which is what made stage 4 the slowest read here.
    # The 'sino' branch is what the policy already planned (it clamps cz to
    # the FBP z-slab whenever the θ extent is > 1); 'proj' is the old
    # layout, kept so the sweep can A/B the stage-4 read.
    pgn_theta_per_bank = (pgn_vc[0] + pgn_nb - 1) // pgn_nb
    if args.pgn_chunk_order == "sino":
        pgn_chunks = pgn_ch
        if args.pgn_chunk_z > 0:       # explicit z override
            pgn_chunks = (pgn_chunks[0],
                          max(1, min(args.pgn_chunk_z, pgn_vc[1])),
                          pgn_chunks[2])
    else:
        pgn_chunks = (1, pgn_vc[1], pgn_vc[2])

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

    # ---- --max-vchunks: shrink the VOLUME, keep the LAYOUT ---------------
    # The plan above was computed at full size and stays untouched; only the
    # number of super-chunks each dataset actually holds is capped.  Every
    # dataset here is banked on axis 0 (the harness banks proj.h5 on θ, see
    # the note above), so the cut is always axis 0 and the vchunk still
    # tiles the shape exactly -- no ragged tail, same bank count, same
    # chunk.  What changes is only how many iterations each stage runs.
    MAXVC = max(0, args.max_vchunks)

    def _cap(shape, vc):
        if MAXVC <= 0:
            return shape
        n = min(-(-shape[0] // vc[0]), MAXVC)
        return (n * vc[0],) + tuple(shape[1:])

    init_shape = _cap(init_shape, init_vc)
    big_shape  = _cap(big_shape,  big_vc)
    proj_shape = _cap(proj_shape, proj_vc)
    data_shape = _cap(data_shape, data_vc)
    pgn_shape  = _cap(pgn_shape,  pgn_vc)
    rec_shape  = _cap(rec_shape,  rec_vc)
    if MAXVC > 0 and RANK == 0:
        print(f"[layout] --max-vchunks {MAXVC}: datasets truncated on their "
              f"banked axis to init={init_shape[0]} big={big_shape[0]} "
              f"proj/data={data_shape[0]} paganin={pgn_shape[0]} "
              f"rec={rec_shape[0]}", flush=True)
        if MAXVC % SIZE:
            print(f"[layout] WARNING: {MAXVC} super-chunks over {SIZE} ranks "
                  f"is uneven -- some ranks idle and the aggregate number "
                  f"understates the filesystem.", flush=True)

    def _src_ivc(idx, shape, vc):
        """Clamp a derived source-vchunk index into what the (possibly
        truncated) source dataset actually has.  With --max-vchunks the
        consumer can out-run its producer: e.g. rec keeps MAXVC z-slabs of
        rec_vc[0] while paganin keeps MAXVC θ-slabs, and the two counts
        need not line up.  Re-reading the last super-chunk keeps the op
        size and the op count honest, which is all this measures."""
        return min(int(idx), max(0, -(-shape[0] // vc[0]) - 1))

    # Prefetch super-chunks the two read-heavy stages allocate on top of
    # their write buffer: a θ-slab of data.h5 for paganin (step7) and the
    # full-θ sinogram slab of paganin.h5 for fbp (step8).
    pgn_read_vc  = (pgn_vc[0], pgn_shape[1], pgn_shape[2])
    sino_read_vc = (pgn_shape[0], rec_vc[0], pgn_shape[2])

    if not args.dry_run:
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
        _describe("init.h5",       init_shape, init_vc, init_nb, 4, chunks=init_ch)
        _describe(f"big{UPS}x.h5", big_shape,  big_vc,  big_nb, 4, chunks=big_ch)
        _describe("proj.h5",       proj_shape, proj_vc, proj_nb, 4, chunks=proj_ch)
        _describe("data.h5",       data_shape, data_vc, data_nb, 4, chunks=data_ch)
        _describe("paganin.h5",    pgn_shape,  pgn_vc,  pgn_nb, 4,
                  chunks=pgn_chunks)
        _describe("rec.h5",        rec_shape,  rec_vc,  rec_nb, 4, chunks=rec_ch)

        # What this run will actually cost the filesystem, before it costs
        # it.  Stage 1 writes init AND big; the other four write one dataset
        # each and read their predecessor, so bytes-on-disk is the sum.
        sets = (("init.h5", init_shape, init_vc, init_nb),
                (f"big{UPS}x.h5", big_shape, big_vc, big_nb),
                ("proj.h5", proj_shape, proj_vc, proj_nb),
                ("data.h5", data_shape, data_vc, data_nb),
                ("paganin.h5", pgn_shape, pgn_vc, pgn_nb),
                ("rec.h5", rec_shape, rec_vc, rec_nb))
        tot_b = sum(int(np.prod(s)) * 4 for _, s, _, _ in sets)
        tot_f = sum(-(-s[0] // v[0]) * nb + 1 for _, s, v, nb in sets)
        print(f"\n  TOTAL written this run: {_hb(tot_b)} in ~{tot_f} h5 files "
              f"({'full volume' if MAXVC <= 0 else f'--max-vchunks {MAXVC}'})",
              flush=True)

        # Peak RAM per rank, stage by stage.  Each stage holds the super-
        # chunk it writes plus the one it reads, so the peak is a PAIR --
        # which is what a --mem-budget sized for a single buffer misses.
        # Multiply by the ranks per node before trusting it.
        def _b(vc):
            return int(np.prod(vc)) * 4
        pairs = (("1 upsample",   _b(init_vc) + _b(big_vc)),
                 ("2 radon",      _b(big_vc)  + _b(proj_vc)),
                 ("2 propagation", _b(proj_vc) + _b(data_vc)),
                 ("3 paganin",    _b(pgn_vc)  + _b(pgn_read_vc)),
                 ("4 fbp",        _b(rec_vc)  + _b(sino_read_vc)))
        peak = max(b for _, b in pairs)
        print("  peak RAM per rank (buffers held together):  "
              + "   ".join(f"{n}={_hb(b)}" for n, b in pairs), flush=True)
        if peak > budget_b:
            print(f"  WARNING: peak {_hb(peak)}/rank exceeds --mem-budget "
                  f"{args.mem_budget} GiB.  With R ranks per node you need "
                  f"R x that much; lower --mem-budget (which shrinks the "
                  f"super-chunks) if the node cannot hold it.", flush=True)
    rprint("")
    if args.dry_run:
        rprint("[dry-run] nothing written; drop --dry-run to run it.")
        return

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
                              vchunks=init_vc, stype="proj", nbanks=init_nb,
                              chunks=init_ch)
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
    _fill_random(buf_i, rng)          # once: 27 GiB per re-roll, and the
                                      # write is what is being timed
    for k, ivc in enumerate(my_ivchunks, start=1):
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
                             vchunks=big_vc, stype="proj", nbanks=big_nb,
                              chunks=big_ch)
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

    # A big super-chunk whose C0 is not a multiple of UPS is legal (see the
    # NOTE in step1_upsample.py): the layout policy drops below UPS
    # alignment when the budget cannot hold UPS whole output planes, which
    # from UPS=16 on it cannot -- one plane is 9.7 GB at UPS=16 and 38.7 GB
    # at UPS=32.  The cost is the seam: the input plane a vchunk edge falls
    # inside gets read and upsampled again by the next vchunk.
    if RANK == 0 and big_vc[0] % UPS:
        print(f"  NOTE: big vchunk C0={big_vc[0]} is not a multiple of "
              f"UPS={UPS}; input planes at the vchunk seams are read "
              f"~{max(1.0, UPS / big_vc[0]):.0f}x more often.", flush=True)

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_big, start=1):
        z0_out = ivc[0] * big_vc[0]
        z1_out = min(z0_out + big_vc[0], big_shape[0])
        z0_in = z0_out // UPS
        z1_in = (z1_out + UPS - 1) // UPS
        in_ivc = (_src_ivc(z0_in // init_vc[0], init_shape, init_vc), 0, 0)
        t = time.perf_counter()
        src = tomo_readx(INIT, ntasks=args.ntasks, shm=shm_i,
                         ivchunk=in_ivc, vchunks=init_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(init_vc)) * dtp.itemsize

        # `src` covers input planes [base, base + init_vc[0]).  Take the
        # [z0_in, z1_in) window out of it, clipped to what this one
        # super-chunk holds -- the window spans at most two input planes,
        # so it only clips when a seam lands on the very last plane of an
        # init super-chunk.  Content is irrelevant here (this measures
        # bytes and ops, not physics); step1 does the real read.
        base = in_ivc[0] * init_vc[0]
        lo = min(max(z0_in - base, 0), src.shape[0] - 1)
        hi = min(max(z1_in - base, lo + 1), src.shape[0])
        planes_in = src[lo:hi]
        up = np.repeat(np.repeat(planes_in, UPS, axis=1), UPS, axis=2)
        up = np.repeat(up, UPS, axis=0).astype(np.float32, copy=False)
        # `up` starts at output plane (base + lo) * UPS, which equals z0_out
        # only when big_vc[0] IS a multiple of UPS.  Offset by the remainder
        # instead of silently writing the wrong planes.
        off = min(z0_out - (base + lo) * UPS, up.shape[0] - 1)
        nz = min(z1_out - z0_out, up.shape[0] - off)
        buf_b[:nz] = up[off:off + nz]
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
                              vchunks=proj_vc, stype="proj", nbanks=proj_nb,
                              chunks=proj_ch)
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
    # Fill the write buffer once instead of keeping a same-size `fake_proj`
    # copy alongside it: at UPS=8 a proj super-chunk is 40 GiB, and the
    # duplicate is what pushes 4 ranks/node past a 512 GB node.  The content
    # was already constant across iterations, so nothing changes but RAM.
    _fill_random(buf_p, rng)

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_proj, start=1):
        z0 = ivc[1] * proj_vc[1]
        big_i = (_src_ivc(z0 // big_vc[0], big_shape, big_vc), 0, 0)
        t = time.perf_counter()
        _ = tomo_readx(BIG, ntasks=args.ntasks, shm=shm_b,
                       ivchunk=big_i, vchunks=big_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(big_vc)) * dtp.itemsize

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
                              vchunks=data_vc, stype="proj", nbanks=data_nb,
                              chunks=data_ch)
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
        proj_i = (_src_ivc(ivc[0], proj_shape, proj_vc), 0, 0)
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
    rprint("STAGE 3 PAGANIN     data.h5 ── read ─▶ paganin.h5 ── write  "
           "(per super-chunk)")
    rprint("─" * 70)
    # Mimics step7_paganin.py: reads a θ-slab of pgn_vc[0] angles from a
    # proj-stored source via read_projs_vchunkx (parallel workers, each
    # doing its own θ-shard slice on the VDS master), then fans a same-
    # shape write across nbanks banks.

    if RANK == 0:
        _cleanup_h5(PGN)
        ctx_pgn = tomo_initx(filename=PGN, shape=pgn_shape, dtype=dtype,
                             vchunks=pgn_vc, stype="proj", nbanks=pgn_nb,
                             chunks=pgn_chunks)
    else:
        ctx_pgn = None
    barrier()

    ctx_pgn = COMM.bcast(ctx_pgn, root=0)

    shm_pg, buf_pg = _alloc_shm(pgn_vc, dtype)
    rprint(f"  buffer for paganin: {_hb(buf_pg.nbytes)}   ({pgn_vc})")

    # Prefetch shm for read_projs_vchunkx (θ-slab of pgn_vc[0] angles).
    shm_pg_read, _pg_read_buf = _alloc_shm(pgn_read_vc, dtype)

    pgn_ivchunks = list(_iter_vchunks(pgn_shape, pgn_vc))
    my_pgn = pgn_ivchunks[RANK::SIZE]
    total = len(pgn_ivchunks)
    my_total = len(my_pgn)
    step = max(1, my_total // 10) if my_total else 1
    _fill_random(buf_pg, rng)

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_pgn, start=1):
        t0_vc = ivc[0] * pgn_vc[0]
        t1_vc = min(t0_vc + pgn_vc[0], pgn_shape[0])

        src_i = _src_ivc(ivc[0], data_shape, pgn_read_vc)
        t = time.perf_counter()
        read_projs_vchunkx(DATA, shm_pg_read, ntasks=args.ntasks,
                           vchunksx=pgn_read_vc, ivchunkx=(src_i, 0, 0))
        t_read += time.perf_counter() - t
        bytes_read += (t1_vc - t0_vc) * pgn_vc[1] * pgn_vc[2] * dtp.itemsize

        t = time.perf_counter()
        tomo_writex(PGN, data=buf_pg, shm=shm_pg, ivchunk=ivc, ctx=ctx_pgn)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(pgn_vc)) * dtp.itemsize
        if (k % step == 0 or k == my_total) and RANK == 0:
            print(f"    [rank 0] paganin {k}/{my_total} (of {total} global)  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    barrier()
    _report_stage("paganin  read",  bytes_read,  t_read)
    _report_stage("paganin  write", bytes_write, t_write)
    # Released here, not at the very end: stage 4 never touches them, and at
    # UPS>=8 holding a paganin super-chunk plus its θ-slab prefetch (72 GiB
    # combined at UPS=32) through the FBP stage would OOM the node.
    _free_shm(shm_pg)
    _free_shm(shm_pg_read)
    rprint("")

    # ================== STAGE 4 FBP: paganin -> rec ========================
    rprint("─" * 70)
    rprint("STAGE 4 FBP        paganin.h5 ── SLICE read ─▶ rec.h5 ── write  "
           "(per super-chunk)")
    rprint("─" * 70)
    # Mimics step8_fbp.py: for each rec.h5 super-chunk of (VZ, N, N),
    # PREFETCHES the full (NTHETA, VZ, N) sinogram slab via
    # read_slices_vchunkx (ntasks parallel workers), then the inner
    # nzchunk-sized loop just slices from RAM.
    #
    # This is the stage the paganin.h5 chunk shape decides.  With
    # --pgn-chunk-order sino and --pgn-chunk-z == VZ the slab is exactly
    # one whole chunk per bank file and each worker streams its own
    # quarter of the bank files sequentially;
    # with proj it clips every (1, NZ, N) chunk to VZ/NZ of it, which is
    # the 192× amplification this benchmark was built to expose.

    if RANK == 0:
        _cleanup_h5(REC)
        ctx_rec = tomo_initx(filename=REC, shape=rec_shape, dtype=dtype,
                             vchunks=rec_vc, stype="proj", nbanks=rec_nb,
                              chunks=rec_ch)
    else:
        ctx_rec = None
    barrier()

    ctx_rec = COMM.bcast(ctx_rec, root=0)

    shm_r, buf_r = _alloc_shm(rec_vc, dtype)
    rprint(f"  buffer for rec:     {_hb(buf_r.nbytes)}   ({rec_vc})")

    # Prefetch shm for the sinogram vchunkx (NTHETA, rec_vc[0], N).  One
    # read_slices_vchunkx call per rec vchunk replaces NZCHUNK-many
    # per-inner plain-h5py reads.  Amp drops from NZ/NZCHUNK to NZ/rec_vc[0].
    shm_sino, _sino_buf = _alloc_shm(sino_read_vc, dtype)

    rec_ivchunks = list(_iter_vchunks(rec_shape, rec_vc))
    my_rec = rec_ivchunks[RANK::SIZE]
    total = len(rec_ivchunks)
    my_total = len(my_rec)
    step = max(1, my_total // 10) if my_total else 1
    _fill_random(buf_r, rng)

    NZCHUNK = args.nzchunk

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    for k, ivc in enumerate(my_rec, start=1):
        z0_vc = ivc[0] * rec_vc[0]
        z1_vc = min(z0_vc + rec_vc[0], rec_shape[0])

        # One parallel prefetch of the whole vchunk's sino slab.  paganin.h5
        # is θ-banked, so --max-vchunks cut its θ extent, not its z extent —
        # but rec's z-slab index still has to stay inside paganin's z.
        sino_i = min(ivc[0], max(0, pgn_shape[1] // rec_vc[0] - 1))
        t = time.perf_counter()
        read_slices_vchunkx(PGN, shm_sino, ntasks=args.ntasks,
                            vchunksx=sino_read_vc, ivchunkx=(0, sino_i, 0))
        t_read += time.perf_counter() - t
        bytes_read += pgn_shape[0] * (z1_vc - z0_vc) * pgn_shape[2] * dtp.itemsize

        # Inner nzchunk loop kept for parity with step8_fbp — but reads are
        # now RAM slices from _sino_buf (free), not h5py calls.
        for zc0 in range(z0_vc, z1_vc, NZCHUNK):
            zc1 = min(zc0 + NZCHUNK, z1_vc)
            _ = _sino_buf[:, zc0 - z0_vc : zc1 - z0_vc, :]

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
    _free_shm(shm_r)
    _free_shm(shm_sino)


if __name__ == "__main__":
    main()
