"""Shared GPU-budget-based chunk pickers for TomoLarge and PropagationLarge.

Both large classes stage per-chunk buffers through the GPU.  Each stage's
peak intermediate size is a linear function of one chunk knob, so we
invert to find the largest chunk that fits under budget/15 — an empirical
factor that accounts for cuFFT plan workspace + padding intermediates +
cupy pool fragmentation (the actual GPU peak-to-live ratio grows with UPS,
from ~7× at UPS=1–4 up past ~15× at UPS=32; the 15 constant leaves a
comfortable margin across the whole sweep).
"""
from __future__ import annotations


def largest_divisor_le(m: int, cap: int) -> int:
    """Largest positive divisor of m that is ≤ cap.  Falls back to 1."""
    d = int(min(cap, m))
    while d > 1 and m % d:
        d -= 1
    return d


# --- TomoLarge ---
# Peak intermediates in TomoLarge.R:
#   x-FFT stage : nz * CHUNK_N * (3·n)   * 8    (obj0 + padded fde0)
#   y-FFT stage : nz * (2n)    * CHUNK_N * 8
#   gather      : nz * (CHUNK_XY + 2m + 1)² * 8    (rarely dominates)
#   IFFT stage  : CHUNK_THETA * nz * n * 8
def pick_tomo_chunks(n: int, ntheta: int, nz: int,
                     gpu_budget_bytes: int) -> tuple[int, int, int]:
    """Return (CHUNK_N, CHUNK_THETA, CHUNK_XY) for TomoLarge.R at (nz, n, n),
    NTHETA angles, sized for a given GPU budget."""
    # Divisor tuned so that even with cupy pool fragmentation across the
    # x-FFT / y-FFT / gather / IFFT stages (multiple different-sized
    # allocations retained side-by-side), no fresh allocation trips the
    # budget.  Empirically 10 gives near-full GPU utilisation at large N
    # while still leaving room for cufft plan workspaces + transients.
    stage_bytes = gpu_budget_bytes // 15
    cap_n     = max(1, stage_bytes // (3 * nz * n * 8))
    cap_theta = max(1, stage_bytes //      (nz * n * 8))
    return (largest_divisor_le(n,      cap_n),
            largest_divisor_le(ntheta, cap_theta),
            largest_divisor_le(2 * n,  cap_n))     # CHUNK_XY: gather < cap_n


# --- PropagationLarge ---
# Peak intermediates in PropagationLarge.D (Pass 2 is the fattest):
#   Pass 1 : ntheta * CHUNK_NZ * (2n)      * 8
#   Pass 2 : ntheta * (2nz)    * CHUNK_2N  * 8    (+ pad-y copy + cuFFT plan)
#   Pass 3 : ntheta * CHUNK_NZ * (2n)      * 8
def pick_prop_chunks(nz: int, n: int, ntheta: int,
                     gpu_budget_bytes: int) -> tuple[int, int]:
    """Return (CHUNK_NZ, CHUNK_2N) for PropagationLarge.D at (ntheta, nz, n),
    sized for a given GPU budget."""
    # Divisor tuned so that even with cupy pool fragmentation across the
    # x-FFT / y-FFT / gather / IFFT stages (multiple different-sized
    # allocations retained side-by-side), no fresh allocation trips the
    # budget.  On A100 the empirical peak-to-live ratio reaches ~8× when
    # several stages have run.
    stage_bytes = gpu_budget_bytes // 15
    cap_nz = max(1, stage_bytes // (ntheta * 2 * n  * 8))
    cap_2n = max(1, stage_bytes // (ntheta * 2 * nz * 8))
    return (largest_divisor_le(nz,     cap_nz),
            largest_divisor_le(2 * n,  cap_2n))
