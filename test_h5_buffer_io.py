#!/usr/bin/env python
"""RAM-buffer + multi-bank HDF5 I/O benchmark for the mosaic pipeline.

Adapts the pattern from doe-maxiv/doe_chunks_t4_aps.ipynb — one big
shared-memory buffer per stage (a "super-chunk", or vchunk) that holds
many h5 chunks at once, plus a small pool of worker processes fanning
that buffer into per-bank .h5 files behind a top-level VDS.

Differs from test_h5_io.py in three ways:
  1. No MPI-IO.  Each rank / worker opens its own bank file via POSIX,
     so nothing bounces off a single collective-write coordinator.
  2. Many h5 files per dataset.  init.h5 becomes a VDS that references
     init/init_data_000000.h5, init_data_000001.h5, ...  So writes fan
     across --nbanks files in parallel — that's the throughput lever.
  3. Explicit RAM buffer.  Each stage allocates one np.ndarray backed
     by shared_memory.SharedMemory sized to the super-chunk, and the
     worker pool writes slices of that buffer.

Meant to be run single-node (no MPI) so we can compare against
test_h5_io.py under matched shapes and pick a direction before hybridising.

Uses helpers from ./doe-maxiv/dxchange_hdf5_chunks.py — the repo must
sit next to this script.

Example:
    python test_h5_buffer_io.py --path /eagle/APS_IRI/vnikitin/iotest_buf_ups2 \\
        --ups 2 --nbanks 8 --ntasks 8 \\
        --init-vchunks 32 2744 2744 \\
        --big-vchunks  32 5488 5488 \\
        --proj-vchunks 128 32 5488 \\
        --data-vchunks 128 5120 5488
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
from multiprocessing import shared_memory

_DOE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doe-maxiv")
if _DOE_DIR not in sys.path:
    sys.path.insert(0, _DOE_DIR)

from dxchange_hdf5_chunks import tomo_initx, tomo_readx, tomo_writex  # noqa: E402


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
    p.add_argument("--in-nz",   type=int, default=2560)
    p.add_argument("--in-nyx",  type=int, default=2744)
    p.add_argument("--ntheta",  type=int, default=None,
                   help="default = 3·N/4 where N = in-nyx·ups")
    p.add_argument("--nbanks",  type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--ntasks",  type=int, default=8,
                   help="worker processes used by tomo_readx (per stage)")
    p.add_argument("--init-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"),
                   help="super-chunk for init.h5 (default fits ~ nbanks planes)")
    p.add_argument("--big-vchunks",  type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"))
    p.add_argument("--proj-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"))
    p.add_argument("--data-vchunks", type=int, nargs=3, default=None,
                   metavar=("C0","C1","C2"))
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

    init_shape = (IN_NZ,  IN_NYX, IN_NYX)
    big_shape  = (OUT_NZ, N,      N     )
    proj_shape = (NTHETA, OUT_NZ, N     )
    data_shape = (NTHETA, OUT_NZ, N     )

    init_vc = tuple(args.init_vchunks) if args.init_vchunks else (args.nbanks, IN_NYX, IN_NYX)
    big_vc  = tuple(args.big_vchunks)  if args.big_vchunks  else (args.nbanks, N,      N     )
    proj_vc = tuple(args.proj_vchunks) if args.proj_vchunks else (args.nbanks, OUT_NZ, N     )
    data_vc = tuple(args.data_vchunks) if args.data_vchunks else (args.nbanks, OUT_NZ, N     )

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

    os.makedirs(args.path, exist_ok=True)
    INIT = os.path.join(args.path, "init.h5")
    BIG  = os.path.join(args.path, f"big{UPS}x.h5")
    PROJ = os.path.join(args.path, "proj.h5")
    DATA = os.path.join(args.path, "data.h5")

    print(f"[test_h5_buffer_io]  UPS={UPS}   nbanks={args.nbanks}   "
          f"ntasks={args.ntasks}   (no MPI; multiprocessing per stage)")
    print("")
    print("File layout (each dataset = master VDS + nvchunks·nbanks bank files):")
    _describe("init.h5",       init_shape, init_vc, args.nbanks, 4)
    _describe(f"big{UPS}x.h5", big_shape,  big_vc,  args.nbanks, 4)
    _describe("proj.h5",       proj_shape, proj_vc, args.nbanks, 4)
    _describe("data.h5",       data_shape, data_vc, args.nbanks, 4)
    print("")

    dtype = np.float32
    dtp   = np.dtype(dtype)
    rng   = np.random.default_rng(1234)

    # ================== STAGE 1: seed init + upsample -> big ==============
    print("─" * 70)
    print("STAGE 1  init.h5 ── read ─▶ big.h5 ── write   (per super-chunk)")
    print("─" * 70)

    _cleanup_h5(INIT)
    ctx_init = tomo_initx(filename=INIT, shape=init_shape, dtype=dtype,
                          vchunks=init_vc, stype="proj", nbanks=args.nbanks)

    shm_i, buf_i = _alloc_shm(init_vc, dtype)
    print(f"  buffer for init: {_hb(buf_i.nbytes)}   ({init_vc})")

    t_seed = 0.0
    n_vc_init = 0
    ivchunks = list(_iter_vchunks(init_shape, init_vc))
    total = len(ivchunks)
    step = max(1, total // 10)
    for k, ivc in enumerate(ivchunks, start=1):
        # figure out actual valid extent of this vchunk (may clip at edges)
        z0 = ivc[0] * init_vc[0]
        z1 = min(z0 + init_vc[0], init_shape[0])
        # fill only the valid part with random; rest is left as-is
        buf_i[: z1 - z0].fill(0)  # cheap; real seed is a copy below
        buf_i[: z1 - z0] = rng.random(
            (z1 - z0, init_vc[1], init_vc[2]), dtype=np.float32)
        t = time.perf_counter()
        tomo_writex(INIT, data=buf_i, shm=shm_i, ivchunk=ivc, ctx=ctx_init)
        t_seed += time.perf_counter() - t
        n_vc_init += 1
        if k % step == 0 or k == total:
            print(f"    init  vchunk {k}/{total}  (write={t_seed:.1f}s)", flush=True)
    bytes_seed = n_vc_init * int(np.prod(init_vc)) * dtp.itemsize
    print(f"  init.h5 seed: {t_seed:.2f}s   ({_hb(bytes_seed/max(t_seed,1e-9))}/s)")

    _cleanup_h5(BIG)
    ctx_big = tomo_initx(filename=BIG, shape=big_shape, dtype=dtype,
                         vchunks=big_vc, stype="proj", nbanks=args.nbanks)
    shm_b, buf_b = _alloc_shm(big_vc, dtype)
    print(f"  buffer for big:  {_hb(buf_b.nbytes)}   ({big_vc})")

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    big_ivchunks = list(_iter_vchunks(big_shape, big_vc))
    total = len(big_ivchunks)
    step = max(1, total // 10)
    src_vc_z = init_vc[0]  # planes per read
    for k, ivc in enumerate(big_ivchunks, start=1):
        z0_out = ivc[0] * big_vc[0]
        z1_out = min(z0_out + big_vc[0], big_shape[0])
        # for each output plane, find the input plane (z_in = z_out // UPS)
        # simplest: fetch the init vchunk that covers this out-range, upsample
        z0_in = z0_out // UPS
        z1_in = (z1_out + UPS - 1) // UPS
        # read the covering init vchunk(s) — assume init vchunks are UPS-aligned
        in_ivc = (z0_in // init_vc[0], 0, 0)
        t = time.perf_counter()
        src = tomo_readx(INIT, ntasks=args.ntasks, shm=shm_i,
                         ivchunk=in_ivc, vchunks=init_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(init_vc)) * dtp.itemsize

        # nearest-neighbour upsample the covering slab into buf_b
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
        if k % step == 0 or k == total:
            print(f"    upsample {k}/{total}  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    print(f"  stage 1  read : {t_read:.2f}s  ({_hb(bytes_read/max(t_read,1e-9))}/s)")
    print(f"  stage 1  write: {t_write:.2f}s ({_hb(bytes_write/max(t_write,1e-9))}/s)")
    _free_shm(shm_i)
    print("")

    # ================== STAGE 2 RADON: big -> proj =========================
    print("─" * 70)
    print("STAGE 2 RADON   big.h5 ── read ─▶ proj.h5 ── write   (per super-chunk)")
    print("─" * 70)

    _cleanup_h5(PROJ)
    ctx_proj = tomo_initx(filename=PROJ, shape=proj_shape, dtype=dtype,
                          vchunks=proj_vc, stype="proj", nbanks=args.nbanks)
    shm_p, buf_p = _alloc_shm(proj_vc, dtype)
    print(f"  buffer for proj: {_hb(buf_p.nbytes)}   ({proj_vc})")

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    proj_ivchunks = list(_iter_vchunks(proj_shape, proj_vc))
    total = len(proj_ivchunks)
    step = max(1, total // 10)
    fake_proj = rng.random(proj_vc, dtype=np.float32)  # stand-in for radon output

    for k, ivc in enumerate(proj_ivchunks, start=1):
        # Radon consumes ALL z (from big) to produce this (θ, z, x) super-chunk;
        # simplify: read one big vchunk (covering the z-range) as proxy for work.
        z0 = ivc[1] * proj_vc[1]
        big_i = (z0 // big_vc[0], 0, 0)
        t = time.perf_counter()
        _ = tomo_readx(BIG, ntasks=args.ntasks, shm=shm_b,
                       ivchunk=big_i, vchunks=big_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(big_vc)) * dtp.itemsize

        buf_p[:] = fake_proj  # (real code would put actual radon output here)

        t = time.perf_counter()
        tomo_writex(PROJ, data=buf_p, shm=shm_p, ivchunk=ivc, ctx=ctx_proj)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(proj_vc)) * dtp.itemsize
        if k % step == 0 or k == total:
            print(f"    radon {k}/{total}  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    print(f"  radon    read : {t_read:.2f}s  ({_hb(bytes_read/max(t_read,1e-9))}/s)")
    print(f"  radon    write: {t_write:.2f}s ({_hb(bytes_write/max(t_write,1e-9))}/s)")
    _free_shm(shm_b)
    print("")

    # ================== STAGE 2 FRESNEL: proj -> data ======================
    print("─" * 70)
    print("STAGE 2 FRESNEL  proj.h5 ── read ─▶ data.h5 ── write  (per super-chunk)")
    print("─" * 70)

    _cleanup_h5(DATA)
    ctx_data = tomo_initx(filename=DATA, shape=data_shape, dtype=dtype,
                          vchunks=data_vc, stype="proj", nbanks=args.nbanks)
    shm_d, buf_d = _alloc_shm(data_vc, dtype)
    print(f"  buffer for data: {_hb(buf_d.nbytes)}   ({data_vc})")

    t_read = t_write = 0.0
    bytes_read = bytes_write = 0
    data_ivchunks = list(_iter_vchunks(data_shape, data_vc))
    total = len(data_ivchunks)
    step = max(1, total // 10)

    # for fresnel, read pattern is per-theta, so proj_vc[0] should equal data_vc[0]
    # (otherwise we need multiple reads per write).  Simplest: 1:1 super-chunk map.
    for k, ivc in enumerate(data_ivchunks, start=1):
        # read the matching proj vchunk (same theta range; z=0 super-chunk)
        proj_i = (ivc[0], 0, 0)
        t = time.perf_counter()
        src = tomo_readx(PROJ, ntasks=args.ntasks, shm=shm_p,
                         ivchunk=proj_i, vchunks=proj_vc)
        t_read += time.perf_counter() - t
        bytes_read += int(np.prod(proj_vc)) * dtp.itemsize

        # if proj_vc[1] != data_vc[1], take the covering z-slab of src
        buf_d[: src.shape[0], : src.shape[1], : src.shape[2]] = src

        t = time.perf_counter()
        tomo_writex(DATA, data=buf_d, shm=shm_d, ivchunk=ivc, ctx=ctx_data)
        t_write += time.perf_counter() - t
        bytes_write += int(np.prod(data_vc)) * dtp.itemsize
        if k % step == 0 or k == total:
            print(f"    fresnel {k}/{total}  "
                  f"(read={t_read:.1f}s write={t_write:.1f}s)", flush=True)

    print(f"  fresnel  read : {t_read:.2f}s  ({_hb(bytes_read/max(t_read,1e-9))}/s)")
    print(f"  fresnel  write: {t_write:.2f}s ({_hb(bytes_write/max(t_write,1e-9))}/s)")
    _free_shm(shm_p)
    _free_shm(shm_d)


if __name__ == "__main__":
    main()
