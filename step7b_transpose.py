#!/usr/bin/env python
"""Transpose the banking of the Paganin line integrals: θ-banked → z-banked.

Reads {path}/model_big{UPS}x/paganin.h5 (N_HALF, NZ, N) and writes
{path}/model_big{UPS}x/sino.h5 with the *same* shape, dtype and values —
only the on-disk decomposition changes, from stype='proj' (bank files
split along θ) to stype='slice' (split along z).

Why a whole step exists just to copy a volume
---------------------------------------------
step7 shards on θ, so paganin.h5 *has* to be θ-banked: a bank file needs
exactly one writer.  step8 shards on z and reads (all θ, z-slab, N).  The
layout policy papers over the mismatch with a sinogram-ordered chunk
(θ_per_bank, cz, N), but that only pays while a bank holds more than one
angle — from UPS=8 up plan_banking drops θ_per_bank to 1-2, cz collapses
to the FBP z-slab and the chunk lands at ~4.7 MB.  Lustre then sees
thousands of small strided ops; the measured result was

    fbp     read   aggregate=517.77 MB/s      <-- transposed
    paganin read   aggregate=28.00 GB/s       <-- sequential

A naive "read a z-block, write a z-block" transpose does not help: its
read is the same transposed access that is slow in step8.  So this step
is an MPI all-to-all instead.  Per round, with P ranks and output z-block
Zb:

  1. Rank r reads θ ∈ [r·N_HALF/P, (r+1)·N_HALF/P) over a z-block of
     **P·Zb** rows — P times taller than the block it will own, which is
     what lets paganin's chunk z be P times larger and every read op be a
     whole chunk (at UPS=8 / 32 ranks: cz 24 → 256, chunk 4.7 → 50 MB).
  2. One all-to-all: rank r sends peer s the sub-box (its θ range, s's
     z-block) and receives (every θ, its own z-block).
  3. That receive buffer *is* an output super-chunk, so it goes out in a
     single tomo_writex to z-banked bank files this rank alone owns.

Both ends are sequential and the read granule grows with rank count.

The exchange runs as a loop of θ-pieces rather than one big Alltoall
because the buffers are the binding constraint: at UPS=8 the read buffer
and the output vchunk are 40.5 GiB each, so the shuffle staging has to
stay small (see --msg-mb).  The receive buffer of each piece lands
directly in the output vchunk — Alltoall orders receives by source rank,
and source rank s owns exactly θ-block s, which is contiguous there.

Launch (same shape as step8):
    mpirun -n <NGPU> python step7b_transpose.py --ups 8 --path /eagle/...

No GPU is used; it is pure I/O plus one shuffle.
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from iohdf5.dxchange_hdf5_chunks import tomo_writex, read_boxx
from iohdf5.h5_vchunks import (
    initx_and_bcast, alloc_shm, free_shm, vchunk_bytes, n_vchunks,
    describe_input, describe_output, read_src_header, rm_src, add_rm_src_arg,
)
from iohdf5.layout import add_layout_args, add_geometry_args, input_dims, resolve_step
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=1,
                   help="matches step7 --ups (drives paths)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/model_big{UPS}x/paganin.h5, "
                        "writes {path}/model_big{UPS}x/sino.h5")
    p.add_argument("--nbanks", type=int, default=8,
                   help="bank files per super-chunk (parallel POSIX writers)")
    p.add_argument("--ntasks", type=int, default=8,
                   help="parallel workers for the paganin read")
    p.add_argument("--nzchunk", type=int, default=8,
                   help="step8's --nzchunk; only used so the planned Zb "
                        "divides the FBP z-slab it will be read back in")
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for sino.h5 (N_HALF, Zb, N); default "
                        "comes from iohdf5.layout")
    p.add_argument("--msg-mb", type=float, default=64.0,
                   help="target size of one per-peer shuffle message, MiB.  "
                        "Sets how finely the exchange is split along θ: "
                        "staging RAM is nranks × this, and mpi4py cannot "
                        "carry a single message past 2 GiB.")
    add_rm_src_arg(p, "the input paganin.h5")
    add_layout_args(p)
    add_geometry_args(p)
    return p.parse_args()


_A = _parse_args()

UPS      = _A.ups
BASE_DIR = _A.path
DST_DIR  = f"{BASE_DIR}/model_big{UPS}x"
SRC_H5   = f"{DST_DIR}/paganin.h5"
DST_H5   = f"{DST_DIR}/sino.h5"

IN_NZ, IN_N = input_dims(_A, UPS)   # 3072 (init.h5 after step00) unless --nz/--n
NZ    = IN_NZ * UPS
N     = IN_N  * UPS
NTASKS = _A.ntasks


def _theta_pieces(theta_loc: int, bytes_per_theta: int, target: int) -> int:
    """How many equal θ-pieces one per-peer message splits into.

    Must divide θ_loc exactly (Alltoall wants identical counts on every
    peer, and equal pieces keep it that way without Alltoallv's
    displacement bookkeeping).  Returns the smallest such divisor whose
    piece is under `target` bytes; θ_loc itself is always a divisor, so
    the loop terminates.
    """
    for d in range(1, theta_loc + 1):
        if theta_loc % d == 0 and (theta_loc // d) * bytes_per_theta <= target:
            return d
    return theta_loc


def main() -> None:
    from mpi_utils import banner
    banner("7b", "paganin.h5 -> sino.h5  (MPI all-to-all bank transpose)")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    src_shape, theta_deg = read_src_header(SRC_H5, rank=RANK, comm=COMM)
    N_HALF = src_shape[0]
    if src_shape[1:] != (NZ, N):
        raise SystemExit(
            f"paganin.h5 shape {src_shape} incompatible with (NZ={NZ}, N={N})")

    plan = resolve_step("sino", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                        ntheta=2 * N_HALF, nbanks=_A.nbanks,
                        mem_budget_gb=_A.mem_budget, chunk_mb=_A.chunk_bytes,
                        nzchunk=_A.nzchunk, vchunks=_A.vchunks, nranks=SIZE)
    NBANKS   = plan.nbanks
    VCHUNKS  = plan.vchunks               # (N_HALF, Zb, N)
    H5CHUNKS = plan.chunks
    ZB       = VCHUNKS[1]

    if VCHUNKS[0] != N_HALF or VCHUNKS[2] != N:
        raise SystemExit(
            f"--vchunks C0×C2 must equal N_HALF×N ({N_HALF}×{N}).  "
            f"Got {VCHUNKS[0]}×{VCHUNKS[2]}.  The transpose writes whole "
            f"sinograms; only the z extent is a free parameter.")
    if NZ % ZB:
        raise SystemExit(f"Zb={ZB} must divide NZ={NZ}")
    # The read shards θ across ranks, so every rank must get the same
    # number of angles -- otherwise the all-to-all messages differ in size.
    if N_HALF % SIZE:
        raise SystemExit(
            f"n_half={N_HALF} must be divisible by the rank count {SIZE}; "
            f"pick a rank count that divides it (N_HALF = 2304·UPS).")

    THETA_LOC = N_HALF // SIZE
    NVZ       = NZ // ZB                  # output super-chunks
    NROUNDS   = (NVZ + SIZE - 1) // SIZE

    # One piece per peer per Alltoall.  Staging is SIZE × piece on each
    # side, so this is the knob that keeps the shuffle off the budget.
    bytes_per_theta = ZB * N * 4
    NPIECE    = _theta_pieces(THETA_LOC, bytes_per_theta,
                              int(_A.msg_mb * 2 ** 20))
    THETA_PC  = THETA_LOC // NPIECE
    msg_bytes = THETA_PC * bytes_per_theta
    if msg_bytes >= 2 ** 31:
        raise SystemExit(
            f"per-peer message {msg_bytes/2**30:.2f} GiB exceeds mpi4py's "
            f"2 GiB count limit; lower --msg-mb")

    read_shape  = (THETA_LOC, SIZE * ZB, N)
    stage_shape = (SIZE, THETA_PC, ZB, N)

    rprint(f"UPS={UPS}  nz={NZ}  n={N}  n_half={N_HALF}  ranks={SIZE}")
    rprint(f"Zb={ZB}  nvchunks={NVZ}  rounds={NROUNDS}  "
           f"theta/rank={THETA_LOC}")
    rprint(f"read block=(θ {THETA_LOC}, z {SIZE*ZB}, x {N}) = "
           f"{np.prod(read_shape)*4/2**30:.1f} GiB   "
           f"out vchunk={np.prod(VCHUNKS)*4/2**30:.1f} GiB   "
           f"staging=2×{np.prod(stage_shape)*4/2**30:.2f} GiB "
           f"({NPIECE} θ-pieces × {msg_bytes/2**20:.0f} MiB/peer)")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(DST_H5, (N_HALF, NZ, N), np.float32,
                        VCHUNKS, "slice", NBANKS, chunks=H5CHUNKS,
                        read_granule=plan.read_granule,
                        companion_bytes=int(np.prod(read_shape)) * 4)

    ctx = initx_and_bcast(DST_H5, shape=(N_HALF, NZ, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="slice", nbanks=NBANKS, chunks=H5CHUNKS,
                          rank=RANK, comm=COMM)
    if RANK == 0:
        with h5py.File(DST_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()

    shm_read,  read_buf  = alloc_shm(read_shape,  np.float32)
    shm_out,   out_buf   = alloc_shm(VCHUNKS,     np.float32)
    send_buf = np.empty(stage_shape, dtype=np.float32)
    recv_buf = np.empty(stage_shape, dtype=np.float32)

    t_read = t_shuf = t_write = 0.0
    b_read = b_shuf = b_write = 0

    try:
        for k in range(NROUNDS):
            iz0   = k * SIZE                       # first output vchunk
            nblk  = min(SIZE, NVZ - iz0)           # live z-blocks this round
            z_lo  = iz0 * ZB
            z_hi  = z_lo + nblk * ZB
            t_lo  = RANK * THETA_LOC

            # ── read (θ sub-slab × P·Zb-tall z-block) ──────────────────
            t0 = time.perf_counter()
            blk = read_boxx(SRC_H5, shm_read, ntasks=NTASKS,
                            box=((t_lo, t_lo + THETA_LOC), (z_lo, z_hi),
                                 (0, N)))
            t_read += time.perf_counter() - t0
            b_read += blk.size * 4

            # ── shuffle ────────────────────────────────────────────────
            t0 = time.perf_counter()
            # Peers beyond nblk own no z-block in a short final round;
            # their slot stays zero and they discard what they receive.
            send_buf[nblk:] = 0
            for p in range(NPIECE):
                p0 = p * THETA_PC
                for s in range(nblk):
                    send_buf[s] = blk[p0:p0 + THETA_PC,
                                      s * ZB:(s + 1) * ZB, :]
                # Alltoall orders receives by source rank, and source s
                # holds θ-block s -- which is exactly where it belongs in
                # a (N_HALF, Zb, N) vchunk, so this lands in place.
                recv = out_buf[:, :, :].reshape(SIZE, THETA_LOC, ZB, N)[
                    :, p0:p0 + THETA_PC]
                COMM.Alltoall([send_buf, MPI.FLOAT], [recv_buf, MPI.FLOAT])
                recv[...] = recv_buf
                b_shuf += recv_buf.size * 4
            t_shuf += time.perf_counter() - t0

            # ── write one whole z-banked super-chunk ───────────────────
            if RANK < nblk:
                t0 = time.perf_counter()
                tomo_writex(DST_H5, data=out_buf, shm=shm_out,
                            ivchunk=(0, iz0 + RANK, 0), ctx=ctx)
                t_write += time.perf_counter() - t0
                b_write += out_buf.size * 4

            print(f"  [rank {RANK}] round {k+1}/{NROUNDS}  "
                  f"z=[{z_lo},{z_hi})  "
                  f"(read={t_read:.1f}s shuffle={t_shuf:.1f}s "
                  f"write={t_write:.1f}s)", flush=True)
    finally:
        free_shm(shm_read)
        free_shm(shm_out)

    barrier()
    report_stage("step7b read (paganin)", b_read,  t_read)
    report_stage("step7b shuffle",        b_shuf,  t_shuf)
    report_stage("step7b write (sino)",   b_write, t_write)
    rprint(f"wrote {N_HALF}x{NZ}x{N} z-banked (Zb={ZB}) to {DST_H5}")

    rm_src(SRC_H5, enabled=_A.rm_src, rank=RANK, comm=COMM)


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
