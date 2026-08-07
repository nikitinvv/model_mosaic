#!/usr/bin/env python
"""End-to-end parallel-h5 I/O benchmark for the mosaic pipeline.

Purpose
-------
The real pipeline (step1_upsample.py + step2_model_*.py) mixes h5 I/O with
GPU-heavy Radon and Fresnel compute, which makes it hard to tell whether
a slow run is disk-bound, GPU-bound, or a chunking mismatch.  This script
replays only the I/O patterns of each stage — no CUDA, no Radon, no
Fresnel — using random payloads so any wall-clock time is pure I/O.

What each stage does
--------------------

STAGE 1 — upsample I/O pattern
    Files:  init.h5  (IN_NZ, IN_NYX, IN_NYX)   chunks (1, IN_NYX, IN_NYX)
            big.h5   (IN_NZ·UPS, N, N)          chunks (1, N, N)
    Loop per rank:
        seed = rank owns a contiguous z-range in init.h5, writes random
               planes into it (one-time cost; reported as "init seed").
        read  = one input plane at a time from init.h5
        write = UPS output planes to big.h5 (one chunk each; per-plane
                writes are chunk-aligned so no partial-chunk R-M-W)

STAGE 2 RADON I/O pattern
    Files:  big.h5   (as above)
            proj.h5  (NTHETA, NZ, N)             chunks (NTHETACHUNK, NZCHUNK, N)
    Loop per rank (contiguous z-chunk sharding, NZ/NZCHUNK/SIZE per rank):
        read  = NZCHUNK z-slices from big.h5 (one chunk in z per z-chunk)
        write = the full (NTHETA, NZCHUNK, N) slab into proj.h5 at
                proj[0:NTHETA, z0:z1, :] — hits ceil(NTHETA/NTHETACHUNK)
                chunks in θ, exactly one chunk in z, all owned by this rank

STAGE 2 FRESNEL I/O pattern
    Files:  proj.h5  (as above)
            data.h5  (NTHETA, NZ, N)             chunks (1, NZ, N)   ← one plane/chunk
    Loop per rank (contiguous θ sharding, NTHETA/SIZE per rank):
        read  = NPROPCHUNK θ-planes at a time from proj.h5
        write = the same NPROPCHUNK planes into data.h5 — each plane is
                exactly one h5 chunk, so writes are perfectly aligned

What the output means
---------------------
Each stage prints:
  - the file shapes and chunk shapes used
  - per-rank read + write wall-clock times
  - aggregate throughput in bytes/s (per-rank bytes × SIZE / stage-time)

To spot inefficiencies:
  - If aggregate throughput << Lustre's known peak, striping is probably wrong
    (default stripe_count=1 pins each file to one OST; see README).
  - If Fresnel read time is much larger than Fresnel write time, the θ-chunk
    of proj.h5 is bigger than what NPROPCHUNK asks for → read amplification.
  - If Radon write time dominates, NTHETACHUNK might be too small (excess
    chunk-metadata overhead).

Launch example (Polaris, 32 ranks / 8 nodes / 4 GPUs per node):
    mpiexec -n 32 --ppn 4 --depth=8 --cpu-bind depth \\
        ./set_affinity_gpu_polaris.sh python test_h5_io.py \\
        --path /eagle/APS_IRI/vnikitin/mosaic_brain/iotest \\
        --ups 1 --npropchunk 1 \\
        --init-chunks 1 2744 2744 \\
        --big-chunks  1 2744 2744 \\
        --proj-chunks 8 1 2744 \\
        --data-chunks 1 2560 2744
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np


try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
    RANK  = _COMM.Get_rank()
    SIZE  = _COMM.Get_size()
except ImportError:
    MPI, _COMM = None, None
    RANK, SIZE = 0, 1


def rprint(*a, **k) -> None:
    if RANK == 0:
        k.setdefault("flush", True)
        print(*a, **k)


def _barrier() -> None:
    if _COMM is not None:
        _COMM.Barrier()


_H5_HAS_MPI = h5py.get_config().mpi
_H5_KW = {"driver": "mpio", "comm": _COMM} if _COMM is not None and _H5_HAS_MPI else {}


def _hb(b: float) -> str:
    """Human-readable byte formatting."""
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"


def _describe(name: str, shape, chunks, dtype_bytes: int) -> None:
    total = int(np.prod(shape)) * dtype_bytes
    per_chunk = int(np.prod(chunks)) * dtype_bytes
    n_chunks = int(np.prod([-(-s // c) for s, c in zip(shape, chunks)]))
    rprint(f"  {name:12s}  shape={tuple(shape)}   dtype=f32  total={_hb(total)}")
    rprint(f"  {name:12s}  chunks={tuple(chunks)}   per-chunk={_hb(per_chunk)}   "
           f"chunk count={n_chunks:,}")


def _create_h5(path: str, name: str, shape, chunks, dtype="float32") -> None:
    """Collective create — all ranks call together with same args."""
    if RANK == 0 and os.path.exists(path):
        os.remove(path)
    _barrier()
    with h5py.File(path, "w", **_H5_KW) as f:
        f.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)
    _barrier()


# --------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True,
                   help="dir for the test h5 files (must be on the same Lustre "
                        "as the real data)")
    p.add_argument("--ups",     type=int, default=1)
    p.add_argument("--in-nz",   type=int, default=2560)
    p.add_argument("--in-nyx",  type=int, default=2744)
    p.add_argument("--ntheta",  type=int, default=None,
                   help="default = 3·N/4 where N = in-nyx·ups")
    # h5 chunk shapes — pass three ints per file.  Defaults (if omitted):
    #   init: 1 IN_NYX IN_NYX
    #   big:  1 N N
    #   proj: 1 1 N          (i.e. NTHETACHUNK=1, NZCHUNK=1)
    #   data: 1 NZ N         (one full plane per chunk)
    p.add_argument("--init-chunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="init.h5 chunk shape (default: 1 IN_NYX IN_NYX)")
    p.add_argument("--big-chunks",  type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="big{UPS}x.h5 chunk shape (default: 1 N N)")
    p.add_argument("--proj-chunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="proj.h5 chunk shape (NTHETACHUNK, NZCHUNK, chunk_x). "
                        "Default: 1 1 N.  proj-chunks[1] doubles as the "
                        "z-slices-per-Radon-call for the write loop.")
    p.add_argument("--data-chunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="data.h5 chunk shape (default: 1 NZ N — full plane)")
    p.add_argument("--npropchunk",  type=int, default=1,
                   help="planes per Fresnel call (bounded by GPU memory)")
    return p.parse_args()


# --------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    UPS         = args.ups
    IN_NZ       = args.in_nz
    IN_NYX      = args.in_nyx
    OUT_NZ      = IN_NZ  * UPS
    N           = IN_NYX * UPS
    NTHETA      = args.ntheta if args.ntheta is not None else 3 * N // 4
    NPROPCHUNK  = args.npropchunk

    # Shapes are fixed by the pipeline geometry; chunks come from CLI (or defaults).
    init_shape = (IN_NZ,  IN_NYX, IN_NYX)
    big_shape  = (OUT_NZ, N,      N     )
    proj_shape = (NTHETA, OUT_NZ, N     )
    data_shape = (NTHETA, OUT_NZ, N     )

    init_chunks = tuple(args.init_chunks) if args.init_chunks else (1, IN_NYX, IN_NYX)
    big_chunks  = tuple(args.big_chunks)  if args.big_chunks  else (1, N,      N     )
    proj_chunks = tuple(args.proj_chunks) if args.proj_chunks else (1, 1,      N     )
    data_chunks = tuple(args.data_chunks) if args.data_chunks else (1, OUT_NZ, N     )

    # z-slices per Radon call = proj-chunks[1] so each rank writes to whole
    # z-chunks (no partial-chunk R-M-W).
    NZCHUNK     = proj_chunks[1]
    NTHETACHUNK = proj_chunks[0]

    if RANK == 0:
        os.makedirs(args.path, exist_ok=True)
    _barrier()

    INIT_H5 = os.path.join(args.path, "init.h5")
    BIG_H5  = os.path.join(args.path, f"big{UPS}x.h5")
    PROJ_H5 = os.path.join(args.path, "proj.h5")
    DATA_H5 = os.path.join(args.path, "data.h5")

    # ---- header ---------------------------------------------------------
    rprint(f"[test_h5_io]  ranks={SIZE}   h5 mpi={_H5_HAS_MPI}   "
           f"UPS={UPS}   NZCHUNK={NZCHUNK}   NPROPCHUNK={NPROPCHUNK}")
    rprint("")
    rprint("File layout:")
    _describe("init.h5",       init_shape, init_chunks, 4)
    _describe(f"big{UPS}x.h5", big_shape,  big_chunks,  4)
    _describe("proj.h5",       proj_shape, proj_chunks, 4)
    _describe("data.h5",       data_shape, data_chunks, 4)
    rprint("")

    # =========== stage 1: upsample =======================================
    rprint("─" * 70)
    rprint("STAGE 1  init.h5 ── read ─▶ big.h5 ── write  (per-plane)")
    rprint("─" * 70)

    _create_h5(INIT_H5, "data", init_shape, init_chunks)
    _create_h5(BIG_H5,  "data", big_shape,  big_chunks)

    # Populate init.h5 (rank r owns z-slices [r*per_rank, (r+1)*per_rank)).
    per_rank_in = (IN_NZ + SIZE - 1) // SIZE
    i0_in = min(RANK * per_rank_in, IN_NZ)
    i1_in = min(i0_in + per_rank_in, IN_NZ)

    rng = np.random.default_rng(1234 + RANK)
    _barrier()
    t0 = time.perf_counter()
    with h5py.File(INIT_H5, "r+", **_H5_KW) as f:
        dset = f["data"]
        for zi in range(i0_in, i1_in):
            dset[zi, :, :] = rng.random((IN_NYX, IN_NYX), dtype=np.float32)
    _barrier()
    t_init_write = time.perf_counter() - t0

    # Stage 1 pattern: read one init plane at a time, write UPS output planes.
    _barrier()
    t_read = t_write = 0.0
    with h5py.File(INIT_H5, "r",  **_H5_KW) as fsrc, \
         h5py.File(BIG_H5,  "r+", **_H5_KW) as fdst:
        src_dset = fsrc["data"]
        dst_dset = fdst["data"]
        for zi in range(i0_in, i1_in):
            t = time.perf_counter()
            plane = src_dset[zi, :, :]
            t_read += time.perf_counter() - t

            # Fake xy-upsample by tiling (avoids scipy).
            up_plane = np.repeat(np.repeat(plane, UPS, axis=0), UPS, axis=1
                                 ).astype(np.float32, copy=False)
            for r in range(UPS):
                t = time.perf_counter()
                dst_dset[zi * UPS + r, :, :] = up_plane
                t_write += time.perf_counter() - t
            del up_plane
    _barrier()

    slices_per_rank = i1_in - i0_in
    bytes_read  = slices_per_rank * IN_NYX * IN_NYX * 4
    bytes_write = slices_per_rank * UPS * N * N * 4
    rprint(f"  stage 1: read={t_read:.2f}s  ({_hb(bytes_read*SIZE/max(t_read,1e-9))}/s aggregate)")
    rprint(f"  stage 1: write={t_write:.2f}s ({_hb(bytes_write*SIZE/max(t_write,1e-9))}/s aggregate)")
    rprint(f"           (init.h5 seed write on {SIZE} ranks: {t_init_write:.2f}s)")
    rprint("")

    # =========== stage 2 Radon ===========================================
    rprint("─" * 70)
    rprint("STAGE 2 RADON   big.h5 ── read NZCHUNK z ─▶ proj.h5 ── write "
           "(NTHETA, NZCHUNK, N)")
    rprint("─" * 70)

    _create_h5(PROJ_H5, "data", proj_shape, proj_chunks)

    n_z_chunks = (OUT_NZ + NZCHUNK - 1) // NZCHUNK
    per_rank_z = (n_z_chunks + SIZE - 1) // SIZE
    my_lo = RANK * per_rank_z
    my_hi = min(my_lo + per_rank_z, n_z_chunks)
    my_z_chunks = list(range(my_lo, my_hi))

    _barrier()
    t_read = t_write = 0.0
    fake_proj = rng.random((NTHETA, NZCHUNK, N), dtype=np.float32)  # payload buffer
    with h5py.File(BIG_H5,  "r",  **_H5_KW) as fsrc, \
         h5py.File(PROJ_H5, "r+", **_H5_KW) as fdst:
        src_dset = fsrc["data"]
        proj_dset = fdst["data"]
        for ci, cidx in enumerate(my_z_chunks):
            z0 = cidx * NZCHUNK
            z1 = min(z0 + NZCHUNK, OUT_NZ)
            k  = z1 - z0

            t = time.perf_counter()
            _ = src_dset[z0:z1, :, :]
            t_read += time.perf_counter() - t

            t = time.perf_counter()
            proj_dset[0:NTHETA, z0:z1, :] = fake_proj[:, :k, :]
            t_write += time.perf_counter() - t

            if (ci + 1) % 16 == 0 or ci + 1 == len(my_z_chunks):
                print(f"    [rank {RANK}] radon z-chunk {ci+1}/{len(my_z_chunks)}",
                      flush=True)
    _barrier()
    bytes_read  = len(my_z_chunks) * NZCHUNK * N * N * 4
    bytes_write = len(my_z_chunks) * NTHETA * NZCHUNK * N * 4
    rprint(f"  stage 2 radon: read={t_read:.1f}s   "
           f"({_hb(bytes_read*SIZE/max(t_read,1e-9))}/s aggregate)")
    rprint(f"  stage 2 radon: write={t_write:.1f}s  "
           f"({_hb(bytes_write*SIZE/max(t_write,1e-9))}/s aggregate)")
    rprint("")

    # =========== stage 2 Fresnel =========================================
    rprint("─" * 70)
    rprint("STAGE 2 FRESNEL  proj.h5 ── read NPROPCHUNK planes ─▶ data.h5 "
           "── write same shape")
    rprint("─" * 70)

    _create_h5(DATA_H5, "data", data_shape, data_chunks)

    per_rank_theta = (NTHETA + SIZE - 1) // SIZE
    i_start = min(RANK * per_rank_theta, NTHETA)
    i_end   = min(i_start + per_rank_theta, NTHETA)

    _barrier()
    t_read = t_write = 0.0
    with h5py.File(PROJ_H5, "r",  **_H5_KW) as fp, \
         h5py.File(DATA_H5, "r+", **_H5_KW) as fd:
        proj_dset = fp["data"]
        data_dset = fd["data"]
        for i0 in range(i_start, i_end, NPROPCHUNK):
            i1 = min(i0 + NPROPCHUNK, i_end)

            t = time.perf_counter()
            batch = proj_dset[i0:i1, :, :]
            t_read += time.perf_counter() - t

            t = time.perf_counter()
            data_dset[i0:i1, :, :] = batch
            t_write += time.perf_counter() - t
    _barrier()
    n_planes = i_end - i_start
    bytes_side = n_planes * OUT_NZ * N * 4
    rprint(f"  stage 2 fresnel: read={t_read:.1f}s   "
           f"({_hb(bytes_side*SIZE/max(t_read,1e-9))}/s aggregate)")
    rprint(f"  stage 2 fresnel: write={t_write:.1f}s  "
           f"({_hb(bytes_side*SIZE/max(t_write,1e-9))}/s aggregate)")

    if RANK == 0:
        rprint("")
        rprint("On-disk sizes:")
        for p in (INIT_H5, BIG_H5, PROJ_H5, DATA_H5):
            try:
                rprint(f"  {p}: {_hb(os.path.getsize(p))}")
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
