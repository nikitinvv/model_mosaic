"""Shared helpers used across the mosaic pipeline scripts.

MPI wiring: mpi4py is a hard dependency in the MAIN process (every step
runs under mpirun/mpiexec).  BUT — spawn-context Pool workers (used by
iohdf5.dxchange_hdf5_chunks for parallel bank writes) re-execute the
parent script's module-level code via multiprocessing.spawn's
`_fixup_main_from_path` mechanism.  If we unconditionally import mpi4py
here, the worker also imports it, which calls MPI_Init_thread in a
process that was NOT launched by mpirun — OpenMPI's SM BTL then
segfaults in `mca_btl_sm_poll_handle_frag`.

Detection: `multiprocessing.parent_process()` returns None only in the
top-level Python process.  In a spawn worker it returns a Process
object.  We use that to skip mpi4py import in workers.  Workers don't
call barrier/allreduce/rprint anyway — those helpers exist only for
step scripts running in the main process.

Everything the step scripts need lives here, so a step's boilerplate
stays down to:

    from utils import COMM, RANK, SIZE, barrier, rprint, allreduce
"""
from __future__ import annotations

import multiprocessing as _mp
import warnings as _warnings

# multiprocessing.resource_tracker prints a UserWarning at interpreter
# shutdown for every SharedMemory segment whose refcount didn't drop
# cleanly — which is normal in our workflow because spawn Pool workers
# are SIGTERM'd before they can release their shm handles.  The kernel
# unlinks the segments on process exit regardless; the warnings are
# just noise.  Suppress before any shm allocation happens.
_warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore",
)


_IS_WORKER = _mp.parent_process() is not None

if not _IS_WORKER:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
else:
    # Spawn worker: leave MPI un-initialised so importing this module
    # (which happens transitively via _fixup_main_from_path) doesn't
    # call MPI_Init in a non-MPI process.
    MPI = None
    COMM = None
    RANK = 0
    SIZE = 1


def barrier() -> None:
    if COMM is not None:
        COMM.Barrier()


def rprint(*a, **k) -> None:
    """print() only from rank 0, with flush on by default."""
    if RANK == 0:
        k.setdefault("flush", True)
        print(*a, **k)


def allreduce(val, op):
    """Wrapper for COMM.allreduce.  In a worker (COMM is None) returns
    the local value unchanged — workers should never call this, but
    behaving sensibly is cheap."""
    if COMM is None:
        return val
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
    if COMM is None:
        return  # worker context — nothing to report
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


def hard_exit(code: int = 0, watchdog_s: int = 60) -> None:
    """Flush + Barrier, then return to let Python shut down naturally.

    Call as the last line of `main()` in every step script.

    Normal flow: main() returns → atexit handlers run (Pool shutdown,
    then mpi4py's MPI.Finalize) → process exits cleanly with code 0 →
    mpirun sees clean shutdown and returns 0.

    If natural shutdown hangs > `watchdog_s` (a MPI teardown race that
    we've seen on some hosts), a daemon watchdog thread SIGKILLs the
    process so the launcher script can move on.  Normally you'll never
    see the watchdog fire — if you do, its stderr marker tells you.
    """
    import os
    import signal
    import sys
    import threading
    import time

    sys.stdout.flush()
    sys.stderr.flush()

    def _watchdog():
        time.sleep(watchdog_s)
        try:
            sys.stderr.write(
                f"[hard_exit] watchdog fired after {watchdog_s}s — "
                f"MPI shutdown hung; SIGKILL\n")
            sys.stderr.flush()
        except Exception:
            pass
        os.kill(os.getpid(), signal.SIGKILL)
    threading.Thread(target=_watchdog, daemon=True).start()

    # Kill any lingering spawn Pool workers BEFORE the final Barrier.
    # If workers are still alive at MPI.Finalize time, atexit ordering
    # on some systems (tomo5) can wedge — the pool's own supervisor
    # thread holds resources MPI wants.  Lazy import to avoid a
    # circular dep (iohdf5 imports utils indirectly via step scripts).
    try:
        from iohdf5.dxchange_hdf5_chunks import shutdown_pools
        shutdown_pools()
    except Exception:
        pass

    if COMM is not None:
        try:
            COMM.Barrier()
        except Exception:
            pass
    # Return normally — Python's atexit runs MPI.Finalize.
    # (shutdown_pools atexit fallback is a no-op now — pools already killed.)
