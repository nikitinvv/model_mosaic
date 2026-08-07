#!/usr/bin/env python
"""Zarr counterpart to test_h5_io.py.

Same 3-stage I/O benchmark, same CLI, same per-stage timing output — but
each dataset is a Zarr array (a directory of chunk files) instead of an
HDF5 file.  No MPI-IO in the storage layer: every rank opens the array
independently and writes its chunks via POSIX.  Because each Zarr chunk
is a separate file, ranks writing to different chunks never touch each
other's data.

Run this alongside test_h5_io.py with matching --path/--ups/chunks to
compare the two backends on the same Lustre.  Key thing to eyeball:

  writes:  zarr does POSIX open/write/close per chunk file — no MPI-IO
           collective sync, so it can be faster on setups where parallel
           HDF5 stalls on metadata, but scales badly if chunk COUNT is
           huge (each chunk = one file = one MDS op).

  reads:   same trade-off inverted — a Fresnel plane read fans out into
           many per-chunk file opens.  On Lustre, keep the chunk count
           to O(few thousand per file) or it will get slow.

If chunk count blows past ~1e5 per dataset, that's a warning sign; use
larger chunks or switch to Zarr v3 sharding (not enabled here — keep the
comparison one-knob-at-a-time first).

Launch example (Polaris, matches polaris_test_h5_io.sh's user knobs):
    mpiexec -n 32 --ppn 4 --depth=8 --cpu-bind depth \\
        ./set_affinity_gpu_polaris.sh python test_zarr_io.py \\
        --path /eagle/APS_IRI/vnikitin/iotest_zarr_ups2 \\
        --ups 2 --npropchunk 1 \\
        --init-chunks 1 2744 2744 \\
        --big-chunks  1 5488 5488 \\
        --proj-chunks 1 32   5488 \\
        --data-chunks 1 5120 5488
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import numpy as np
import zarr


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


def _hb(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"


def _describe(name: str, shape, chunks, dtype_bytes: int) -> None:
    total = int(np.prod(shape)) * dtype_bytes
    per_chunk = int(np.prod(chunks)) * dtype_bytes
    n_chunks = int(np.prod([-(-s // c) for s, c in zip(shape, chunks)]))
    rprint(f"  {name:14s}  shape={tuple(shape)}   dtype=f32  total={_hb(total)}")
    rprint(f"  {name:14s}  chunks={tuple(chunks)}   per-chunk={_hb(per_chunk)}   "
           f"chunk-file count={n_chunks:,}")


def _create_zarr(path: str, shape, chunks, dtype="float32",
                 reuse: bool = False) -> bool:
    """Return True if fresh-created (needs seed), False if reused an
    existing array whose shape/dtype/chunks match."""
    can_reuse = False
    if reuse and RANK == 0 and os.path.exists(path):
        try:
            z = zarr.open_array(path, mode="r")
            if (tuple(z.shape) == tuple(shape)
                    and z.dtype == np.dtype(dtype)
                    and tuple(z.chunks) == tuple(chunks)):
                can_reuse = True
        except Exception:
            can_reuse = False
    if _COMM is not None:
        can_reuse = _COMM.bcast(can_reuse, root=0)
    if can_reuse:
        rprint(f"  reusing existing {path}  (shape/dtype/chunks match)")
        return False

    if RANK == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        # write_empty_chunks=False → don't materialise chunks that only
        # contain the fill value, so create is O(metadata) not O(dataset).
        zarr.create(
            store=path,
            shape=tuple(shape),
            chunks=tuple(chunks),
            dtype=dtype,
            overwrite=True,
            write_empty_chunks=False,
        )
    _barrier()
    return True


# --------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True,
                   help="dir for the test zarr stores (each dataset lives in "
                        "its own subdir there)")
    p.add_argument("--ups",     type=int, default=1)
    p.add_argument("--in-nz",   type=int, default=2560)
    p.add_argument("--in-nyx",  type=int, default=2744)
    p.add_argument("--ntheta",  type=int, default=None,
                   help="default = 3·N/4 where N = in-nyx·ups")
    p.add_argument("--init-chunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="init.zarr chunk shape (default: 1 IN_NYX IN_NYX)")
    p.add_argument("--big-chunks",  type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="big{UPS}x.zarr chunk shape (default: 1 N N)")
    p.add_argument("--proj-chunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="proj.zarr chunk shape (default: 1 1 N)")
    p.add_argument("--data-chunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="data.zarr chunk shape (default: 1 NZ N)")
    p.add_argument("--npropchunk", type=int, default=1,
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

    init_shape = (IN_NZ,  IN_NYX, IN_NYX)
    big_shape  = (OUT_NZ, N,      N     )
    proj_shape = (NTHETA, OUT_NZ, N     )
    data_shape = (NTHETA, OUT_NZ, N     )

    init_chunks = tuple(args.init_chunks) if args.init_chunks else (1, IN_NYX, IN_NYX)
    big_chunks  = tuple(args.big_chunks)  if args.big_chunks  else (1, N,      N     )
    proj_chunks = tuple(args.proj_chunks) if args.proj_chunks else (1, 1,      N     )
    data_chunks = tuple(args.data_chunks) if args.data_chunks else (1, OUT_NZ, N     )

    def _validate(name, shape, chunks):
        if any(c > s for c, s in zip(chunks, shape)):
            axes = ", ".join(f"axis{i}: chunk {c} > shape {s}"
                             for i, (c, s) in enumerate(zip(chunks, shape))
                             if c > s)
            raise SystemExit(
                f"[test_zarr_io] {name} chunks={chunks} exceed shape={shape} "
                f"({axes}).  Every chunk axis must be ≤ the dataset axis.")
    _validate("init.zarr",       init_shape, init_chunks)
    _validate(f"big{UPS}x.zarr", big_shape,  big_chunks)
    _validate("proj.zarr",       proj_shape, proj_chunks)
    _validate("data.zarr",       data_shape, data_chunks)

    NZCHUNK     = proj_chunks[1]
    NTHETACHUNK = proj_chunks[0]

    if RANK == 0:
        os.makedirs(args.path, exist_ok=True)
    _barrier()

    INIT_Z = os.path.join(args.path, "init.zarr")
    BIG_Z  = os.path.join(args.path, f"big{UPS}x.zarr")
    PROJ_Z = os.path.join(args.path, "proj.zarr")
    DATA_Z = os.path.join(args.path, "data.zarr")

    rprint(f"[test_zarr_io]  ranks={SIZE}   zarr v{zarr.__version__}   "
           f"UPS={UPS}   NZCHUNK={NZCHUNK}   NPROPCHUNK={NPROPCHUNK}")
    rprint("")
    rprint("File layout (each dataset = a directory of chunk files):")
    _describe("init.zarr",       init_shape, init_chunks, 4)
    _describe(f"big{UPS}x.zarr", big_shape,  big_chunks,  4)
    _describe("proj.zarr",       proj_shape, proj_chunks, 4)
    _describe("data.zarr",       data_shape, data_chunks, 4)
    rprint("")

    # =========== stage 1: upsample =======================================
    rprint("─" * 70)
    rprint("STAGE 1  init.zarr ── read ─▶ big.zarr ── write  (per-plane)")
    rprint("─" * 70)

    init_created = _create_zarr(INIT_Z, init_shape, init_chunks, reuse=True)
    big_created  = _create_zarr(BIG_Z,  big_shape,  big_chunks)

    per_rank_in = (IN_NZ + SIZE - 1) // SIZE
    i0_in = min(RANK * per_rank_in, IN_NZ)
    i1_in = min(i0_in + per_rank_in, IN_NZ)

    rng = np.random.default_rng(1234 + RANK)
    total = max(1, i1_in - i0_in)
    step  = max(1, total // 10)

    t_init_write = 0.0
    if init_created:
        _barrier()
        t0 = time.perf_counter()
        rprint(f"  seeding init.zarr  ({total} planes/rank)…")
        z_init = zarr.open_array(INIT_Z, mode="r+")
        for i, zi in enumerate(range(i0_in, i1_in), start=1):
            z_init[zi, :, :] = rng.random((IN_NYX, IN_NYX), dtype=np.float32)
            if i % step == 0 or i == total:
                print(f"    [rank {RANK}] init seed {i}/{total}", flush=True)
        _barrier()
        t_init_write = time.perf_counter() - t0
    else:
        rprint("  skipping init.zarr seed (existing store reused)")

    _barrier()
    t_read = t_write = 0.0
    if not big_created:
        rprint("  skipping upsample loop (existing big.zarr reused)")
    else:
        rprint(f"  upsample loop  ({total} planes/rank × {UPS} out planes each)…")
        z_src = zarr.open_array(INIT_Z, mode="r")
        z_dst = zarr.open_array(BIG_Z,  mode="r+")
        for i, zi in enumerate(range(i0_in, i1_in), start=1):
            t = time.perf_counter()
            plane = z_src[zi, :, :]
            t_read += time.perf_counter() - t

            up_plane = np.repeat(np.repeat(plane, UPS, axis=0), UPS, axis=1
                                 ).astype(np.float32, copy=False)
            for r in range(UPS):
                t = time.perf_counter()
                z_dst[zi * UPS + r, :, :] = up_plane
                t_write += time.perf_counter() - t
            del up_plane
            if i % step == 0 or i == total:
                print(f"    [rank {RANK}] upsample {i}/{total}  "
                      f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)
    _barrier()

    if big_created:
        slices_per_rank = i1_in - i0_in
        bytes_read  = slices_per_rank * IN_NYX * IN_NYX * 4
        bytes_write = slices_per_rank * UPS * N * N * 4
        rprint(f"  stage 1: read={t_read:.2f}s  "
               f"({_hb(bytes_read*SIZE/max(t_read,1e-9))}/s aggregate)")
        rprint(f"  stage 1: write={t_write:.2f}s "
               f"({_hb(bytes_write*SIZE/max(t_write,1e-9))}/s aggregate)")
    if init_created:
        rprint(f"           (init.zarr seed write on {SIZE} ranks: {t_init_write:.2f}s)")
    rprint("")

    # =========== stage 2 Radon ===========================================
    rprint("─" * 70)
    rprint("STAGE 2 RADON   big.zarr ── read NZCHUNK z ─▶ proj.zarr ── write "
           "(NTHETA, NZCHUNK, N)")
    rprint("─" * 70)

    proj_created = _create_zarr(PROJ_Z, proj_shape, proj_chunks)

    n_z_chunks = (OUT_NZ + NZCHUNK - 1) // NZCHUNK
    per_rank_z = (n_z_chunks + SIZE - 1) // SIZE
    my_lo = RANK * per_rank_z
    my_hi = min(my_lo + per_rank_z, n_z_chunks)
    my_z_chunks = list(range(my_lo, my_hi))

    _barrier()
    t_read = t_write = 0.0
    if not proj_created:
        rprint("  skipping radon loop (existing proj.zarr reused)")
    else:
        fake_proj = rng.random((NTHETA, NZCHUNK, N), dtype=np.float32)
        total = max(1, len(my_z_chunks))
        step  = max(1, total // 10)
        rprint(f"  radon loop  ({total} z-chunks/rank, {NZCHUNK} slices each)…")
        z_src  = zarr.open_array(BIG_Z,  mode="r")
        z_proj = zarr.open_array(PROJ_Z, mode="r+")
        for ci, cidx in enumerate(my_z_chunks, start=1):
            z0 = cidx * NZCHUNK
            z1 = min(z0 + NZCHUNK, OUT_NZ)
            k  = z1 - z0

            t = time.perf_counter()
            _ = z_src[z0:z1, :, :]
            t_read += time.perf_counter() - t

            t = time.perf_counter()
            z_proj[0:NTHETA, z0:z1, :] = fake_proj[:, :k, :]
            t_write += time.perf_counter() - t

            if ci % step == 0 or ci == total:
                print(f"    [rank {RANK}] radon {ci}/{total}  "
                      f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)
    _barrier()
    if proj_created:
        bytes_read  = len(my_z_chunks) * NZCHUNK * N * N * 4
        bytes_write = len(my_z_chunks) * NTHETA * NZCHUNK * N * 4
        rprint(f"  stage 2 radon: read={t_read:.1f}s   "
               f"({_hb(bytes_read*SIZE/max(t_read,1e-9))}/s aggregate)")
        rprint(f"  stage 2 radon: write={t_write:.1f}s  "
               f"({_hb(bytes_write*SIZE/max(t_write,1e-9))}/s aggregate)")
    rprint("")

    # =========== stage 2 Fresnel =========================================
    rprint("─" * 70)
    rprint("STAGE 2 FRESNEL  proj.zarr ── read NPROPCHUNK planes ─▶ data.zarr "
           "── write same shape")
    rprint("─" * 70)

    data_created = _create_zarr(DATA_Z, data_shape, data_chunks)

    per_rank_theta = (NTHETA + SIZE - 1) // SIZE
    i_start = min(RANK * per_rank_theta, NTHETA)
    i_end   = min(i_start + per_rank_theta, NTHETA)

    _barrier()
    t_read = t_write = 0.0
    if not data_created:
        rprint("  skipping fresnel loop (existing data.zarr reused)")
    else:
        n_iters = max(1, -(-(i_end - i_start) // NPROPCHUNK))
        step    = max(1, n_iters // 10)
        rprint(f"  fresnel loop  ({n_iters} batches/rank of {NPROPCHUNK} planes)…")
        z_proj = zarr.open_array(PROJ_Z, mode="r")
        z_data = zarr.open_array(DATA_Z, mode="r+")
        for it, i0 in enumerate(range(i_start, i_end, NPROPCHUNK), start=1):
            i1 = min(i0 + NPROPCHUNK, i_end)

            t = time.perf_counter()
            batch = z_proj[i0:i1, :, :]
            t_read += time.perf_counter() - t

            t = time.perf_counter()
            z_data[i0:i1, :, :] = batch
            t_write += time.perf_counter() - t

            if it % step == 0 or it == n_iters:
                print(f"    [rank {RANK}] fresnel {it}/{n_iters}  "
                      f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)
    _barrier()
    if data_created:
        n_planes = i_end - i_start
        bytes_side = n_planes * OUT_NZ * N * 4
        rprint(f"  stage 2 fresnel: read={t_read:.1f}s   "
               f"({_hb(bytes_side*SIZE/max(t_read,1e-9))}/s aggregate)")
        rprint(f"  stage 2 fresnel: write={t_write:.1f}s  "
               f"({_hb(bytes_side*SIZE/max(t_write,1e-9))}/s aggregate)")

    if RANK == 0:
        rprint("")
        rprint("On-disk sizes (du -sh):")
        for p in (INIT_Z, BIG_Z, PROJ_Z, DATA_Z):
            if os.path.isdir(p):
                total = 0
                nfiles = 0
                for root, _, files in os.walk(p):
                    for f in files:
                        total += os.path.getsize(os.path.join(root, f))
                        nfiles += 1
                rprint(f"  {p}: {_hb(total)}   ({nfiles:,} files)")


if __name__ == "__main__":
    main()
