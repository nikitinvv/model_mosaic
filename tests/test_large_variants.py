"""Numerical parity tests: TomoReal vs TomoLargeReal, Propagation vs
PropagationLarge, Paganin vs PaganinLarge.

Each *_large class stages its work through the GPU host-side, so it produces
the same math as its non-large counterpart — just with a bounded GPU
memory footprint.  These tests pin that equivalence at small sizes where
both variants fit on GPU, then sweep several chunk configurations to
prove the chunking itself does not perturb the result.

Run with:
    cd mosaic_modeling && python -m tests.test_large_variants
"""
from __future__ import annotations

import numpy as np
import cupy as cp

from processing.tomo             import TomoReal
from processing.tomo_large       import TomoLargeReal
from processing.propagation      import Propagation
from processing.propagation_large import PropagationLarge
from processing.paganin          import Paganin
from processing.paganin_large    import PaganinLarge


# ---------------------------------------------------------------------------
def _rel_max(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _report(name, ref, got, rtol):
    err = _rel_max(got, ref)
    ok  = err < rtol
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}]  {name:38s}  rel_max={err:.3e}  (< {rtol})")
    return ok


# ---------------------------------------------------------------------------
def test_tomo_R_vs_large():
    """TomoReal.R vs TomoLargeReal.R — parity of the forward Radon.

    Both use USFFT + rfft-along-x.  TomoLargeReal stages each chunk's
    fde patch through the GPU one strip at a time; the two should
    match to float32 precision across every chunk configuration.
    Test asserts:
      (a) chunking-invariance INSIDE TomoLargeReal — fine chunking
          matches coarse.
      (b) full parity with TomoReal to FP tolerance.
    """
    print("\n─── TomoReal  vs  TomoLargeReal  (forward Radon R) ───")
    n, nz     = 64, 16
    ntheta    = 24
    theta     = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")

    np.random.seed(0)
    obj_real = np.random.randn(nz, n, n).astype(np.float32)

    tomo   = TomoReal     (n, nz, theta)
    sino_ref = cp.asnumpy(tomo.R(cp.asarray(obj_real)))

    # tomo_l.R returns a view into a reused internal buffer; every call
    # would overwrite `coarse` in place.  Copy each result so this test
    # actually compares independent snapshots.  chunk_xy is fixed at
    # ctor time, so we build a fresh TomoLargeReal per chunk_xy sweep.
    tomo_l = TomoLargeReal(n, theta, n)
    coarse = tomo_l.R(obj_real, [n, ntheta, n]).copy()   # 1 x-FFT iter, 1 gather bin
    passed = True
    for chunks in [(32, 12, 32), (16, 6, 16), (8, 6, 8)]:
        tomo_l = TomoLargeReal(n, theta, chunks[2])
        fine = tomo_l.R(obj_real, list(chunks)).copy()
        passed &= _report(f"chunk-invariance  fine={chunks} vs coarse",
                          coarse, fine, rtol=1e-4)

    # (b) parity with TomoReal
    passed &= _report("TomoReal vs TomoLargeReal — full-sino rel_max",
                      sino_ref, coarse, rtol=1e-4)
    return passed


# ---------------------------------------------------------------------------
def test_tomo_RT_vs_large():
    """TomoReal.RT vs TomoLargeReal.RT — parity of the adjoint (backprojection).

    Both keep the full-complex64 layout internally (pragmatic port from
    the retired Tomo / TomoLarge).  TomoLargeReal.RT streams the four
    reversed passes through the GPU one strip at a time (and chunks the
    passRT2 scatter along ky per the bin-chunked kernel); the result
    should match TomoReal.RT to fp precision across every chunk config.
    """
    print("\n─── TomoReal.RT  vs  TomoLargeReal.RT  (adjoint / backprojection) ───")
    n, nz    = 64, 8
    ntheta   = 24
    theta    = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")

    np.random.seed(0)
    passed = True
    for dtype_label, sino in [
        ("complex64",
         (np.random.randn(ntheta, nz, n) + 1j * np.random.randn(ntheta, nz, n))
         .astype(np.complex64)),
        ("float32   ", np.random.randn(ntheta, nz, n).astype(np.float32)),
    ]:
        tomo   = TomoReal(n, nz, theta)
        ref    = cp.asnumpy(tomo.RT(cp.asarray(sino)))
        # chunk_xy fixed at ctor time — rebuild per chunk_xy sweep.
        tomo_l = TomoLargeReal(n, theta, 2 * n)
        coarse = np.asarray(tomo_l.RT(sino, [n, ntheta, 2 * n])).copy()
        passed &= _report(f"sino {dtype_label}  chunks=(n, ntheta, 2n)",
                          ref, coarse, rtol=1e-4)
        # Sweep chunk_xy (drives the passRT2 ky-band chunking) plus the
        # r-axis and theta chunks that pipe RT1/3/4.
        for chunks in [(32, 12, 32), (16, 6, 16), (8, 6, 8)]:
            tomo_l = TomoLargeReal(n, theta, chunks[2])
            fine = np.asarray(tomo_l.RT(sino, list(chunks))).copy()
            passed &= _report(f"sino {dtype_label}  chunks={chunks}",
                              ref, fine, rtol=1e-4)
    return passed


# ---------------------------------------------------------------------------
def test_tomo_R_RT_adjoint():
    """Adjoint identity  ⟨R(x), y⟩ = ⟨x, RT(y)⟩  — no filter.

    A necessary condition for RT to be the mathematical adjoint of R.
    Both R and RT are exercised on REAL float32 tensors (TomoReal.R is
    real-only; RT accepts real y and returns a real obj), so the inner
    products are ordinary real dot products.  Applies to both the
    GPU-only TomoReal and the host-chunked TomoLargeReal; the equality
    holds to fp precision for any implementation that consistently
    normalises R and RT.
    """
    print("\n─── TomoReal/TomoLargeReal  R–RT adjoint identity  "
          "⟨R(x), y⟩ = ⟨x, RT(y)⟩ ───")
    n, nz    = 64, 8
    ntheta   = 24
    theta    = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")

    np.random.seed(0)
    x_np = np.random.randn(nz, n, n).astype(np.float32)
    y_np = np.random.randn(ntheta, nz, n).astype(np.float32)

    def _inner(a, b):
        return float(np.sum(a * b))

    passed = True

    # (a) TomoReal (GPU-only) adjoint identity
    tomo = TomoReal(n, nz, theta)
    Rx   = cp.asnumpy(tomo.R (cp.asarray(x_np)))         # real f32 sino
    RTy  = cp.asnumpy(tomo.RT(cp.asarray(y_np)))         # real f32 obj
    lhs, rhs = _inner(Rx, y_np), _inner(x_np, RTy)
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    ok  = err < 1e-4
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}]  TomoReal      ⟨R(x),y⟩={lhs:.6g}  ⟨x,RT(y)⟩={rhs:.6g}  "
          f"rel_err={err:.3e}")
    passed &= ok

    # (b) TomoLargeReal (host-chunked) adjoint identity
    tomo_l = TomoLargeReal(n, theta, n)     # eager precompute; runtime chunk_xy sweeps re-fill the cache
    Rx_l   = np.asarray(tomo_l.R (x_np, [n, ntheta, n])).copy()
    RTy_l  = np.asarray(tomo_l.RT(y_np, [n, ntheta, n])).copy()
    lhs_l, rhs_l = _inner(Rx_l, y_np), _inner(x_np, RTy_l)
    err_l = abs(lhs_l - rhs_l) / max(abs(lhs_l), abs(rhs_l), 1e-30)
    ok_l  = err_l < 1e-4
    tag_l = "OK  " if ok_l else "FAIL"
    print(f"  [{tag_l}]  TomoLargeReal ⟨R(x),y⟩={lhs_l:.6g}  ⟨x,RT(y)⟩={rhs_l:.6g}  "
          f"rel_err={err_l:.3e}")
    passed &= ok_l

    return passed


# ---------------------------------------------------------------------------
def test_propagation_vs_large():
    """Propagation.D vs PropagationLarge.D over several (chunk_nz, chunk_2n)
    and several distances (kernel indices)."""
    print("\n─── Propagation  vs  PropagationLarge  (forward Fresnel D) ───")
    n, nz     = 64, 32
    ntheta    = 3
    wavelength = 4.13e-11
    voxelsize  = 1.4e-6
    distances  = [0.0, 0.02, 0.1]      # covers identity + moderate Fresnel

    np.random.seed(0)
    psi_h = (np.random.randn(ntheta, nz, n)
             + 1j * np.random.randn(ntheta, nz, n)).astype(np.complex64)

    prop_ref = Propagation      (n, nz, ntheta, len(distances),
                                 wavelength, voxelsize, distances)
    prop_lrg = PropagationLarge (n, nz, wavelength, voxelsize, distances)

    passed = True
    for j, L in enumerate(distances):
        ref = cp.asnumpy(prop_ref.D(cp.asarray(psi_h), j))
        for chunks in [(32, 128), (16, 32), (8, 16), (4, 8)]:
            got = prop_lrg.D(psi_h, j, list(chunks))
            passed &= _report(f"L={L}m  chunks={chunks}",
                              ref, got, rtol=1e-3)
    return passed


# ---------------------------------------------------------------------------
def test_propagation_validation():
    """Chunk-divisor assertions must fire for non-dividing chunk sizes."""
    print("\n─── PropagationLarge input validation ───")
    n, nz = 32, 16
    prop_lrg = PropagationLarge(n, nz, 4.13e-11, 1.4e-6, [0.0])
    psi = np.zeros((1, nz, n), dtype=np.complex64)
    tests = [
        ((5,  16), "CHUNK_NZ=5 (doesn't divide nz=16)"),
        ((8,  9),  "CHUNK_2N=9 (doesn't divide 2n=64)"),
    ]
    passed = True
    for chunks, label in tests:
        try:
            prop_lrg.D(psi, 0, list(chunks))
            print(f"  [FAIL]  {label}  — expected AssertionError, none raised")
            passed = False
        except AssertionError as e:
            print(f"  [OK  ]  {label}  → {e}")
    return passed


# ---------------------------------------------------------------------------
def test_paganin_vs_large():
    """Paganin.retrieve vs PaganinLarge.retrieve over several
    (chunk_nz, chunk_n) and a couple of distances.

    Paganin's filter H = α/(λ·z·|k|²/(4π) + α) is not separable, so
    PaganinLarge rebuilds it per x-strip inside pass 2.  This test
    pins the chunked variant to the full-2-D reference to float32
    precision across a sweep of chunk sizes and propagation distances.
    """
    print("\n─── Paganin  vs  PaganinLarge  (single-distance Paganin retrieve) ───")
    n, nz     = 64, 32
    ntheta    = 3
    wavelength = 4.13e-11
    voxelsize  = 1.4e-6
    alpha      = 1e-3
    distances  = [0.02, 0.1, 1.0]        # small + moderate + large NA·NA·λ product

    np.random.seed(0)
    # Random transmission-like intensities in a plausible range.
    intensity_h = (0.5 + 0.5 * np.random.rand(ntheta, nz, n)).astype(np.float32)

    passed = True
    for L in distances:
        pgn_ref = Paganin      (n, nz, ntheta, wavelength, voxelsize, L, alpha)
        pgn_lrg = PaganinLarge (n, nz,         wavelength, voxelsize, L, alpha)

        ref = cp.asnumpy(pgn_ref.retrieve(cp.asarray(intensity_h)))
        for chunks in [(nz, n), (16, 32), (8, 16), (4, 8)]:
            got = np.asarray(pgn_lrg.retrieve(intensity_h, list(chunks)))
            passed &= _report(f"L={L}m  chunks={chunks}",
                              ref, got, rtol=1e-4)
    return passed


# ---------------------------------------------------------------------------
def test_paganin_validation():
    """Chunk-divisor assertions must fire for non-dividing chunk sizes."""
    print("\n─── PaganinLarge input validation ───")
    n, nz = 32, 16
    pgn_lrg = PaganinLarge(n, nz, 4.13e-11, 1.4e-6, 0.1, 1e-3)
    intensity = np.zeros((1, nz, n), dtype=np.float32)
    tests = [
        ((5,  16), "CHUNK_NZ=5 (doesn't divide nz=16)"),
        ((8,  9),  "CHUNK_N=9  (doesn't divide n=32)"),
    ]
    passed = True
    for chunks, label in tests:
        try:
            pgn_lrg.retrieve(intensity, list(chunks))
            print(f"  [FAIL]  {label}  — expected AssertionError, none raised")
            passed = False
        except AssertionError as e:
            print(f"  [OK  ]  {label}  → {e}")
    return passed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Numerical parity — large variants vs GPU-only reference")
    print("=" * 70)
    results = [
        test_tomo_R_vs_large(),
        test_tomo_RT_vs_large(),
        test_tomo_R_RT_adjoint(),
        test_propagation_vs_large(),
        test_propagation_validation(),
        test_paganin_vs_large(),
        test_paganin_validation(),
    ]
    print("=" * 70)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)
