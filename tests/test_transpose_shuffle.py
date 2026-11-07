"""The index arithmetic of step7b_transpose's all-to-all, on toy arrays.

The transpose itself is trivially correct on paper -- it is a permutation
of where bytes live on disk, not of their values -- so what can actually
go wrong is the bookkeeping: which sub-box each rank reads, which peer
each piece is destined for, and where an Alltoall's receive blocks land
in the output super-chunk.  Getting any of those subtly wrong produces a
sino.h5 that is the right shape, full of plausible numbers, and wrong.

This replays exactly that loop (same variable names as
step7b_transpose.main) against a volume small enough to hold whole, and
compares to the reference answer.  No HDF5, no GPU.

Run:  mpirun -n 4 python tests/test_transpose_shuffle.py
      python tests/test_transpose_shuffle.py        (serial, SIZE=1)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpi4py import MPI                                          # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

from step7b_transpose import _theta_pieces                      # noqa: E402


def reference(n_half, nz, n):
    """A volume whose every element encodes its own (θ, z, x) index, so a
    misrouted piece cannot coincidentally compare equal."""
    t = np.arange(n_half, dtype=np.float64)[:, None, None]
    z = np.arange(nz,     dtype=np.float64)[None, :, None]
    x = np.arange(n,      dtype=np.float64)[None, None, :]
    return (t * 1e6 + z * 1e3 + x).astype(np.float32)


def run_case(n_half, nz, n, zb, msg_bytes):
    vol = reference(n_half, nz, n)

    theta_loc = n_half // SIZE
    nvz       = nz // zb
    nrounds   = (nvz + SIZE - 1) // SIZE
    npiece    = _theta_pieces(theta_loc, zb * n * 4, msg_bytes)
    theta_pc  = theta_loc // npiece

    send = np.empty((SIZE, theta_pc, zb, n), dtype=np.float32)
    recv = np.empty_like(send)
    out  = np.empty((n_half, zb, n), dtype=np.float32)

    for k in range(nrounds):
        iz0  = k * SIZE
        nblk = min(SIZE, nvz - iz0)
        z_lo = iz0 * zb
        t_lo = RANK * theta_loc

        # The read: a θ sub-slab over an nblk*zb-tall z block.
        blk = vol[t_lo:t_lo + theta_loc, z_lo:z_lo + nblk * zb, :]

        send[nblk:] = 0
        for p in range(npiece):
            p0 = p * theta_pc
            for s in range(nblk):
                send[s] = blk[p0:p0 + theta_pc, s * zb:(s + 1) * zb, :]
            COMM.Alltoall([send, MPI.FLOAT], [recv, MPI.FLOAT])
            out.reshape(SIZE, theta_loc, zb, n)[:, p0:p0 + theta_pc] = recv

        if RANK < nblk:
            z0 = (iz0 + RANK) * zb
            want = vol[:, z0:z0 + zb, :]
            assert np.array_equal(out, want), (
                f"rank {RANK} round {k}: output vchunk for z=[{z0},{z0+zb}) "
                f"differs (first bad at "
                f"{np.argwhere(out != want)[0].tolist()})")
    return npiece


def main():
    cases = [
        # (n_half, nz, n, zb) -- the last two make the final round short,
        # which is the branch where peers beyond nblk get zero-filled.
        (SIZE * 6,  SIZE * 2, 5, 1),
        (SIZE * 4,  SIZE * 4, 3, 2),
        (SIZE * 8,  SIZE * 3, 4, 1),
        (SIZE * 4,  SIZE + 1, 7, 1),
        (SIZE * 12, 3 * SIZE - 1 if SIZE > 1 else 3, 2, 1),
    ]
    for n_half, nz, n, zb in cases:
        if nz % zb:
            continue
        for msg_bytes in (4, 1 << 20):      # force many pieces, then one
            npiece = run_case(n_half, nz, n, zb, msg_bytes)
            COMM.Barrier()
            if RANK == 0:
                print(f"  ok    n_half={n_half} nz={nz} n={n} zb={zb} "
                      f"ranks={SIZE} pieces={npiece}")
    if RANK == 0:
        print(f"\nall cases passed on {SIZE} rank(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
