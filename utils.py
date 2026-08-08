"""Shared helpers used across the mosaic pipeline scripts.

MPI wiring: mpi4py is a hard dependency (every step + test script runs
under mpirun/mpiexec).  No optional fallback — a missing import should
error loudly rather than silently degrading to serial.

Everything the step scripts need for MPI + rank-aware printing lives
here, so a step script's boilerplate stays down to:

    from utils import COMM, RANK, SIZE, barrier, rprint, allreduce
"""
from __future__ import annotations

from mpi4py import MPI


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def barrier() -> None:
    COMM.Barrier()


def rprint(*a, **k) -> None:
    """print() only from rank 0, with flush on by default."""
    if RANK == 0:
        k.setdefault("flush", True)
        print(*a, **k)


def allreduce(val, op):
    """Wrapper for COMM.allreduce (kept as a helper so step scripts don't
    need to import MPI just to name the reduction op — MPI.MIN etc. are
    available as `utils.MPI.MIN` if needed)."""
    return COMM.allreduce(val, op=op)
