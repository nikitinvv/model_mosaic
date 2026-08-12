"""Shared GPU-budget-based chunk pickers for the four "large" classes.

Every picker inverts the peak-live formula for its class to find the
largest chunks whose worst-case single buffer fits under
``budget / BUDGET_DIVISOR``.  BUDGET_DIVISOR = 24 leaves room for the
6× peak-to-live ratio typical of these passes: 2 pinned + 2 GPU
ping-pong buffers alive at once, plus 2 in-flight FFT/mul intermediates
that cupy allocates per compute step, plus cuFFT plan workspace that
grows with transform size.  Empirically it clears OOMs up to
UPS=32-128 on 40 GB GPUs without measurably slowing the sub-UPS≤4
cases (they saturate at "full n" anyway since the max divisor pins
them at n).
"""
from __future__ import annotations


BUDGET_DIVISOR = 24


def largest_divisor_le(m: int, cap: int) -> int:
    """Largest positive divisor of m that is ≤ cap.  Falls back to 1."""
    d = int(min(cap, m))
    while d > 1 and m % d:
        d -= 1
    return d


def pick_tomo_chunks(n: int, ntheta: int, nz: int,
                     gpu_budget_bytes: int) -> tuple[int, int, int]:
    """(CHUNK_N, CHUNK_THETA, CHUNK_XY) for TomoLargeReal.R and .RT.

    Both directions take the same triplet; RT is the dominant sizing
    constraint (passRT3/4's (nz, 2n, chunk_n) c64 ping-pong is wider
    than R's rfft (nz, chunk_n, n+1)).  chunk_xy bounds passRT2's
    scatter fde slice.
    """
    target = max(1, gpu_budget_bytes // BUDGET_DIVISOR)
    cap_n  = max(1, target // (nz * 2 * n * 8))
    cap_th = max(1, target // (nz *     n * 8))
    cap_xy = max(1, target // (nz * 2 * n * 8))
    return (largest_divisor_le(n,      cap_n),
            largest_divisor_le(ntheta, cap_th),
            largest_divisor_le(2 * n,  cap_xy))


def pick_prop_chunks(nz: int, n: int, ntheta: int,
                     gpu_budget_bytes: int) -> tuple[int, int]:
    """(CHUNK_NZ, CHUNK_2N) for PropagationLarge.D.  Pass 2's
    (ntheta, 2nz, chunk_2n) and (ntheta, chunk_nz, 2n) c64 buffers
    dominate."""
    target = max(1, gpu_budget_bytes // BUDGET_DIVISOR)
    cap_nz = max(1, target // (ntheta * 2 * n  * 8))
    cap_2n = max(1, target // (ntheta * 2 * nz * 8))
    return (largest_divisor_le(nz,    cap_nz),
            largest_divisor_le(2 * n, cap_2n))


def pick_paganin_chunks(nz: int, n: int, ntheta: int,
                        gpu_budget_bytes: int) -> tuple[int, int]:
    """(CHUNK_NZ, CHUNK_N) for PaganinLarge.retrieve.  Pass 2's
    2-pinned + 2-GPU (ntheta, nz, chunk_n) c64 ping-pong plus
    in-flight y-FFT/y-IFFT outputs dominate."""
    target = max(1, gpu_budget_bytes // BUDGET_DIVISOR)
    cap_nz = max(1, target // (n  * ntheta * 8))
    cap_n  = max(1, target // (nz * ntheta * 8))
    return (largest_divisor_le(nz, cap_nz),
            largest_divisor_le(n,  cap_n))
