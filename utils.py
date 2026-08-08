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


def hb(b: float) -> str:
    """Human-readable bytes."""
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.2f} {u}"
        b /= 1024
    return f"{b:.2f} PB"


def report_stage(label: str, bytes_local: float, time_local: float) -> None:
    """Print an aggregate + per-rank spread throughput line for a stage.

    Called by ALL ranks (collective).  Rank 0 prints one line:
        <label>  aggregate=<X GB/s>   per-rank[min..max]=[..]   wall[..]

    aggregate = sum(bytes) / max(rank elapsed) — how fast the whole run
    moved data through this stage.
    """
    total_bytes = allreduce(float(bytes_local), MPI.SUM)
    max_time    = allreduce(float(time_local),  MPI.MAX)
    min_time    = allreduce(float(time_local),  MPI.MIN)
    per_rank_bps = float(bytes_local) / max(float(time_local), 1e-9)
    max_bps = allreduce(per_rank_bps, MPI.MAX)
    min_bps = allreduce(per_rank_bps, MPI.MIN)
    aggregate_bps = total_bytes / max(max_time, 1e-9)
    rprint(f"  {label:22s}  aggregate={hb(aggregate_bps)}/s   "
           f"per-rank[min..max]=[{hb(min_bps)}/s..{hb(max_bps)}/s]   "
           f"wall[min..max]=[{min_time:.1f}s..{max_time:.1f}s]  "
           f"total={hb(total_bytes)}")


def hard_exit(code: int = 0, finalize_timeout_s: int = 5) -> None:
    """Barrier + flush + best-effort MPI_Finalize with hard timeout + os._exit.

    Call as the last line of `main()` in every step script.

    Why this exists: on tomo5 (Rocky 8 + OpenMPI + UCX) `MPI_Finalize`
    can hang collectively at interpreter shutdown, likely a UCX teardown
    race with lingering shared_memory / multiprocessing spawn workers.
    All ranks have passed the final barrier and every h5 write has
    flushed to disk by this point, so the tail of Python's shutdown is
    all noise from our perspective.

    We first try to call MPI_Finalize explicitly so mpirun sees a clean
    shutdown (avoiding OpenMPI's "process ... exiting improperly"
    warning + non-zero mpirun exit).  If Finalize doesn't return within
    `finalize_timeout_s`, SIGALRM fires and we os._exit anyway — mpirun
    still returns non-zero in that case, so the launcher scripts wrap
    each mpirun in `|| true` as belt-and-suspenders.
    """
    import os
    import signal
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    COMM.Barrier()

    def _timeout_bail(signum, frame):
        os._exit(int(code))

    signal.signal(signal.SIGALRM, _timeout_bail)
    signal.alarm(int(finalize_timeout_s))
    try:
        MPI.Finalize()
    except Exception:
        pass
    signal.alarm(0)
    os._exit(int(code))
