"""Numerical parity tests: TomoLarge vs Tomo, PropagationLarge vs Propagation.

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

from processing.tomo             import Tomo, TomoReal
from processing.tomo_large       import TomoLarge, TomoLargeReal
from processing.propagation      import Propagation
from processing.propagation_large import PropagationLarge


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
def test_tomo_vs_large():
    """Tomo.R vs TomoLarge.R.

    Both use USFFT.  After TomoLarge was updated to stitch each chunk's
    fde patch with the opposite-edge wrap-halo (matching Tomo's
    modular `(n + ell + twon) % twon` gather), the two match to
    float32 precision.  Test asserts:
      (a) chunking-invariance INSIDE TomoLarge — fine chunking is
          bit-exact vs coarse.
      (b) full parity with Tomo to FP tolerance.
    """
    print("\n─── Tomo  vs  TomoLarge  (forward Radon R) ───")
    n, nz     = 64, 16
    ntheta    = 24
    theta     = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")

    np.random.seed(0)
    obj_c = np.empty((nz, n, n), dtype=np.complex64)
    obj_c.real = np.random.randn(nz, n, n).astype("float32")
    obj_c.imag = 0

    tomo   = Tomo     (n, nz, theta, mask_r=0.0)
    tomo_l = TomoLarge(n,     theta)

    sino_ref = cp.asnumpy(tomo.R(cp.asarray(obj_c))).real

    # tomo_l.R returns a view into a reused internal buffer; every call
    # would overwrite `coarse` in place.  Copy each result so this test
    # actually compares independent snapshots.
    coarse = tomo_l.R(obj_c, [n, ntheta, n]).real.copy()  # 1 x-FFT iter, 1 gather bin
    passed = True
    for chunks in [(32, 12, 32), (16, 6, 16), (8, 6, 8)]:
        fine = tomo_l.R(obj_c, list(chunks)).real.copy()
        passed &= _report(f"chunk-invariance  fine={chunks} vs coarse",
                          coarse, fine, rtol=1e-5)

    # (b) parity with Tomo
    passed &= _report("Tomo vs TomoLarge — full-sino rel_max",
                      sino_ref, coarse, rtol=1e-4)
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
def test_tomo_real_gpu_only():
    """TomoReal(x_float32) vs Tomo(x_complex64_with_imag_0).

    Same-shape sinograms; the rfft/float32 path should reproduce the
    complex64 path to fp precision when obj has zero imag part."""
    print("\n─── TomoReal  vs  Tomo  (GPU-only, float32 vs complex64+imag=0) ───")
    n, nz  = 64, 16
    ntheta = 24
    theta  = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")

    np.random.seed(0)
    obj_real  = np.random.randn(nz, n, n).astype("float32")
    obj_cmplx = np.empty((nz, n, n), dtype=np.complex64)
    obj_cmplx.real = obj_real
    obj_cmplx.imag = 0

    tomo      = Tomo    (n, nz, theta, mask_r=0.0)
    tomo_real = TomoReal(n, nz, theta, mask_r=0.0)

    ref  = cp.asnumpy(tomo     .R(cp.asarray(obj_cmplx))).real
    got  = cp.asnumpy(tomo_real.R(cp.asarray(obj_real)))

    return _report("TomoReal vs Tomo", ref, got, rtol=1e-4)


# ---------------------------------------------------------------------------
def test_tomo_real_vs_complex():
    """TomoLargeReal(x_float32) vs TomoLarge(x_complex64_with_imag_0).

    The float32/rfft variant should produce a bit-equivalent result
    (up to fp roundoff) to running the complex64 pipeline on the same
    data with imag part zeroed — that's the mathematical guarantee of
    rfft: FFT of a real signal has conjugate symmetry, so the missing
    negative-fx half of the spectrum is exactly what the conjugate
    reflection in `gather_kernel_rfft` synthesises.
    """
    print("\n─── TomoLargeReal  vs  TomoLarge  (float32 vs complex64+imag=0) ───")
    n, nz  = 64, 16
    ntheta = 24
    theta  = np.linspace(0, 2 * np.pi, ntheta, endpoint=False).astype("float32")

    np.random.seed(0)
    obj_real  = np.random.randn(nz, n, n).astype("float32")
    obj_cmplx = np.empty((nz, n, n), dtype=np.complex64)
    obj_cmplx.real = obj_real
    obj_cmplx.imag = 0

    t_lrg  = TomoLarge     (n, theta)
    t_real = TomoLargeReal (n, theta)

    # Both need chunks that divide n and ntheta.  Take the same knobs
    # from the existing test_tomo_vs_large: (n, ntheta, n).  Copies —
    # both return views into reused internal buffers.
    ref  = t_lrg .R(obj_cmplx, [n, ntheta, n]).real.copy()   # (ntheta, nz, n) f32
    real = t_real.R(obj_real,  [n, ntheta, n]).copy()        # (ntheta, nz, n) f32

    passed = True
    passed &= _report("full-sino rel_max        (n=64, one chunk)",
                      ref, real, rtol=1e-4)

    # Sweep several chunkings — the rfft-based gather + fresh bin sort
    # should not care about chunk_xy.
    for chunks in [(32, 12, 32), (16, 6, 16), (8, 6, 8)]:
        ref_c  = t_lrg .R(obj_cmplx, list(chunks)).real.copy()
        real_c = t_real.R(obj_real,  list(chunks)).copy()
        passed &= _report(f"chunks={chunks}", ref_c, real_c, rtol=1e-4)

    return passed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Numerical parity — large variants vs GPU-only reference")
    print("=" * 70)
    results = [
        test_tomo_vs_large(),
        test_tomo_real_gpu_only(),
        test_tomo_real_vs_complex(),
        test_propagation_vs_large(),
        test_propagation_validation(),
    ]
    print("=" * 70)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)
