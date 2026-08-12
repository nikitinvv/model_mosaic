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

    from mpi_utils import COMM, RANK, SIZE, barrier, rprint, allreduce
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
    # By default, let mpi4py's auto MPI.Finalize atexit run at
    # interpreter shutdown.  For it to complete cleanly under
    # multiprocessing spawn, the pool workers must exit CLEANLY before
    # Finalize (releasing their multiprocessing semaphores + SHM
    # segments); that's what iohdf5.dxchange_hdf5_chunks._shutdown_pools
    # does with close()+join() at atexit — registered after mpi4py, so
    # it runs FIRST (atexit is LIFO), draining workers before Finalize.
    #
    # Opt-out: setting MOSAIC_SKIP_MPI_FINALIZE=1 skips Finalize
    # entirely.  Use this only for launchers where Finalize is known
    # to misbehave (e.g. tomo_pipeline_run.sh, when MPI's SM/UCX teardown races
    # can't be avoided some other way).  Cost: OpenMPI's mpirun then
    # prints an "abnormal termination" warning per run.
    import os as _os
    if _os.environ.get("MOSAIC_SKIP_MPI_FINALIZE", "0") in ("1", "true", "True"):
        import mpi4py as _mpi4py
        _mpi4py.rc.finalize = False
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


def banner(step: str, title: str = "", width: int = 80) -> None:
    """Print a rank-0-only banner marking the start of a pipeline stage.

    Format: `======== Step <n> ========` centred to `width`, followed by
    an optional description line.  Kept concise so long run logs stay
    scannable — earlier versions used a 5-row block-letter STEP header
    that ate too much vertical space."""
    if RANK != 0:
        return
    label = f" Step {step} "
    pad   = max(4, (width - len(label)) // 2)
    line  = "=" * pad + label + "=" * (width - pad - len(label))
    print("", flush=True)
    print(line, flush=True)
    if title:
        print(title, flush=True)
    print("=" * width, flush=True)
    print("", flush=True)


def allreduce(val, op):
    """Wrapper for COMM.allreduce.  In a worker (COMM is None) returns
    the local value unchanged — workers should never call this, but
    behaving sensibly is cheap."""
    if COMM is None:
        return val
    return COMM.allreduce(val, op=op)


def run_main(main_func) -> None:
    """Wrap a step's main() with rank-aware crash reporting.

    Use as the entry point in every step script:
        if __name__ == "__main__":
            from mpi_utils import run_main
            run_main(main)

    On clean exit: main() returns, Python exits normally.  We skip
    MPI.Finalize (see mpi_utils.py MPI init), so mpirun/mpiexec may print
    a generic "abnormal termination" warning — that is expected and
    means nothing is wrong.

    On a real crash: catches the exception, prints a rank-tagged
    traceback so you can tell WHICH rank died and WHY, then calls
    MPI.COMM_WORLD.Abort(1) — which propagates SIGTERM to peers so
    they don't hang forever waiting on collectives from the dead rank,
    and gives mpirun an unambiguous "abort" signal (different message
    from the no-Finalize warning) with a non-zero job exit code.

    Rule of thumb for reading a log after a run:
      • no Python traceback  → clean exit, ignore mpirun's warning
      • Python traceback     → real crash on rank R (see the [rank R]
                                prefix), other ranks got SIGTERM'd
    """
    import sys
    import traceback
    try:
        main_func()
    except SystemExit:
        raise
    except BaseException:
        try:
            sys.stderr.write(f"[rank {RANK}] EXCEPTION — traceback follows:\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        except Exception:
            pass
        if COMM is not None:
            try:
                # Abort tells the launcher this was a real failure and
                # signals other ranks so they don't hang waiting on
                # collectives that will never come.  Exits non-zero.
                COMM.Abort(1)
            except Exception:
                pass
        raise


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


