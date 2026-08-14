"""Pure-arithmetic checks on the chunk / vchunk policy in iohdf5.layout.

No filesystem, no MPI, no GPU — this is the cheap net that catches a
layout regression before a 3.4 PB run finds it on Polaris.  What it
asserts is exactly the set of properties the rest of the pipeline relies
on and that arithmetic alone can decide:

  * every chunk is legal HDF5 (0 < extent, < 4 GiB) at every UPS;
  * chunks[0] == 1 on init / big / data / rec (the stated requirement);
  * vchunks tile ONLY the banked axis — the other two are full, because
    tomo_initx lays the VDS out with whole planes per bank;
  * nbanks divides the super-chunk extent, so no bank gets a short tail;
  * every chunk extent divides its bank extent, so chunks tile the bank
    exactly: whole-chunk writes AND whole-chunk reads;
  * per-rank buffer (vchunk + input prefetch) fits the budget, except
    where it provably cannot (step1 `big`, see test_big_is_the_only_
    over_budget_step);
  * chunk bytes stay near the target instead of scaling with N^2, which
    is the whole point of the policy;
  * paganin's z extent divides the FBP z-slab whenever paganin actually
    ends up sinogram-ordered — the one case a consumer granule binds.

Run:  python -m pytest tests/test_layout_policy.py -q
      python tests/test_layout_policy.py        (no pytest needed)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iohdf5.layout import (HDF5_MAX_CHUNK_BYTES, _sitems_axis,  # noqa: E402
                           nominal_bank_shape, plan_chunks, plan_pipeline,
                           resolve_step)

UPS_LIST = (1, 2, 4, 8, 16, 32)
NBANKS_LIST = (1, 2, 4, 8)
DIM0_ONE = ("init", "big", "data", "rec")

# 96 GiB/rank = a 512 GB Polaris node / 4 ranks with ~25% held back.
BUDGET = 96 * 2 ** 30
CHUNK_BYTES = 64 * 2 ** 20


def _all_plans(budget=BUDGET, chunk_bytes=CHUNK_BYTES, nranks=8):
    """Every (ups, nbanks) combination the pipeline is meant to run at."""
    for ups in UPS_LIST:
        for nbanks in NBANKS_LIST:
            plans = plan_pipeline(ups, nbanks=nbanks, budget=budget,
                                  chunk_bytes=chunk_bytes, nranks=nranks)
            for p in plans.values():
                yield ups, nbanks, p


def test_chunks_are_legal_hdf5():
    """A chunk >= 4 GiB does not fail at write time — it fails at file
    creation, with an error that names none of this."""
    for ups, nbanks, p in _all_plans():
        where = f"ups={ups} nbanks={nbanks} {p.name}"
        assert len(p.chunks) == 3, where
        assert all(c >= 1 for c in p.chunks), f"{where}: {p.chunks}"
        assert p.chunk_bytes <= HDF5_MAX_CHUNK_BYTES, \
            f"{where}: chunk {p.chunks} is {p.chunk_bytes} B"


def test_dim0_one_where_required():
    """init / big / data / rec are read slice-wise, so chunk[0] must be 1.
    This is a stated requirement, not an optimisation."""
    for ups, nbanks, p in _all_plans():
        if p.name in DIM0_ONE:
            assert p.chunks[0] == 1, \
                f"ups={ups} nbanks={nbanks} {p.name}: chunks={p.chunks}"


def test_vchunks_tile_only_the_banked_axis():
    """tomo_initx builds nominal_bank = (sitems_per_bank, ny, nx) and the
    VDS layout takes full ny, nx — a vchunk that split another axis would
    silently produce a wrong file."""
    for ups, nbanks, p in _all_plans():
        ax = _sitems_axis(p.stype)
        for a in range(3):
            if a != ax:
                assert p.vchunks[a] == p.shape[a], \
                    f"ups={ups} {p.name}: vchunks={p.vchunks} shape={p.shape}"
        assert 1 <= p.vchunks[ax] <= p.shape[ax]


def test_super_chunks_and_banks_divide_evenly():
    """No ragged tail super-chunk, and every bank file the same size —
    otherwise one rank/worker straggles at the end of every iteration."""
    for ups, nbanks, p in _all_plans():
        ax = _sitems_axis(p.stype)
        where = f"ups={ups} nbanks<={nbanks} {p.name}"
        assert p.shape[ax] % p.vchunks[ax] == 0, \
            f"{where}: {p.shape[ax]} % {p.vchunks[ax]}"
        assert p.vchunks[ax] % p.nbanks == 0, \
            f"{where}: vchunk {p.vchunks[ax]} % nbanks {p.nbanks}"
        assert 1 <= p.nbanks <= nbanks


def test_chunks_tile_the_bank_exactly():
    """A chunk that straddles the bank edge turns the last read of every
    bank into a partial-chunk read.  Each extent divides its bank extent,
    so reads and writes are both whole-chunk."""
    for ups, nbanks, p in _all_plans():
        bank = p.bank_shape
        for a in range(3):
            assert bank[a] % p.chunks[a] == 0, \
                (f"ups={ups} nbanks<={nbanks} {p.name}: chunk {p.chunks} "
                 f"does not tile bank {bank}")


def test_chunk_bytes_stay_near_the_target():
    """The point of the policy: chunk BYTES are roughly constant while the
    shape follows N.  The old full-plane default was 36 MB at UPS=1 and
    9 GB at UPS=16.  The floor is loose because a small bank extent can
    cap the chunk (e.g. init.h5 at nbanks=1), the ceiling is not.

    A sinogram-ordered chunk is exempt from the floor.  It is capped by
    theta_bank * z_granule * N -- the whole of that bank file's share of
    one FBP read, and so the largest shape that keeps the read
    whole-chunk.  When that lands under the target there is nothing
    better to grow into, so assert instead that the chunk really is that
    clamped maximum."""
    for ups, nbanks, p in _all_plans():
        where = f"ups={ups} nbanks<={nbanks} {p.name}"
        assert p.chunk_bytes <= CHUNK_BYTES, \
            f"{where}: {p.chunk_bytes} B > target {CHUNK_BYTES} B"
        if p.chunk_bytes >= CHUNK_BYTES // 8:
            continue
        assert p.effective_order == "sino", \
            f"{where}: chunk {p.chunks} is only {p.chunk_bytes} B"
        bank = p.bank_shape
        assert p.chunks[0] == bank[0] and p.chunks[2] == bank[2], \
            f"{where}: sino chunk {p.chunks} is short of the bank {bank} " \
            f"on theta or x, so the granule clamp is not what capped it"
        assert p.z_granule and p.z_granule % p.chunks[1] == 0, \
            f"{where}: sino cz {p.chunks[1]} does not divide the FBP " \
            f"z-slab {p.z_granule}"


def test_every_buffer_fits_the_budget():
    """Nothing may exceed --mem-budget at any UPS.  This used to exempt
    step1's `big`, back when its super-chunk was forced to a multiple of
    UPS: one output plane is 9.7 GB at UPS=16, so 16 of them is 144 GB.
    That alignment turned out to be a preference (a misaligned vchunk edge
    only makes step1 re-read the straddled input plane), so the policy now
    drops below it when the budget bites and everything fits."""
    for ups, nbanks, p in _all_plans():
        assert not p.over_budget, \
            (f"ups={ups} nbanks<={nbanks} {p.name}: buffer "
             f"{p.buffer_bytes / 2**30:.1f} GiB > budget "
             f"{BUDGET / 2**30:.1f} GiB")


def test_big_keeps_ups_alignment_whenever_the_budget_allows():
    """The seam re-read is a real cost, so UPS alignment must only be
    given up when it is that or not running.  At 96 GiB/rank that means
    UPS<=8 stays aligned and UPS>=16 does not."""
    aligned = {}
    for ups in UPS_LIST:
        p = plan_pipeline(ups, nbanks=8, budget=BUDGET,
                          chunk_bytes=CHUNK_BYTES, nranks=8)["big"]
        aligned[ups] = (p.vchunks[0] % ups == 0)
        # Where alignment is dropped it must be because it could not fit:
        # one aligned super-chunk would have to be ups whole output planes.
        if not aligned[ups]:
            plane = p.shape[1] * p.shape[2] * p.itemsize
            assert ups * plane > BUDGET, \
                f"ups={ups}: dropped UPS alignment but {ups} planes fit"
    assert all(aligned[u] for u in (1, 2, 4, 8)), aligned


def test_paganin_z_divides_the_fbp_slab_when_sino_ordered():
    """The one place a consumer granule binds: step8 reads (all θ, zslab,
    N) out of paganin.h5.  With chunk[0] > 1 a z sub-range inside a chunk
    is chunk[0] separate runs, so cz must divide the slab exactly.  With
    chunk[0] == 1 — which is where the sino order lands on its own once a
    bank holds a single θ — the z range is the outermost non-trivial axis,
    so it is one contiguous run and a fatter chunk is strictly better."""
    for ups in UPS_LIST:
        for nbanks in NBANKS_LIST:
            plans = plan_pipeline(ups, nbanks=nbanks, budget=BUDGET,
                                  chunk_bytes=CHUNK_BYTES, nranks=8)
            pgn, rec = plans["paganin"], plans["rec"]
            slab = rec.vchunks[0]
            if pgn.chunks[0] > 1:
                assert slab % pgn.chunks[1] == 0, \
                    (f"ups={ups} nbanks<={nbanks}: paganin chunk "
                     f"{pgn.chunks} vs FBP z-slab {slab}")
            else:
                # proj-ordered: the z range is the outermost non-trivial
                # axis, so any contiguous sub-range is one run.
                assert pgn.chunk_bytes >= CHUNK_BYTES // 4, \
                    f"ups={ups}: proj fallback gave only {pgn.chunk_bytes} B"


def test_plan_chunks_respects_a_tiny_byte_target():
    """A 1 MB target must still produce a legal, bank-tiling chunk — the
    sweep in polaris_test_h5.sh moves this knob by two orders of
    magnitude."""
    bank = (32, 3072, 3072)
    for target in (2 ** 20, 4 * 2 ** 20, 256 * 2 ** 20):
        for order in ("proj", "sino"):
            ch = plan_chunks(bank, order, 4, target)
            assert all(c >= 1 for c in ch)
            assert all(b % c == 0 for b, c in zip(bank, ch)), (order, ch)
            assert 4 * ch[0] * ch[1] * ch[2] <= target, (order, target, ch)


def test_resolve_step_override_keeps_the_chunk_consistent():
    """An explicit --vchunks must re-derive the chunk and the compute-loop
    alignment from itself, or the two silently drift apart."""
    p = resolve_step("rec", ups=2, in_nz=3072, in_nyx=3072, nbanks=4,
                     mem_budget_gb=96, chunk_mb=64, nzchunk=32,
                     vchunks=(64, 6144, 6144), nranks=8)
    assert p.vchunks == (64, 6144, 6144)
    assert p.nbanks == 4 and 64 % p.nbanks == 0
    assert p.chunks[0] == 1                       # dim0_one survives override
    bank = nominal_bank_shape(p.shape, p.vchunks, p.stype, p.nbanks)
    assert all(b % c == 0 for b, c in zip(bank, p.chunks))
    assert 64 % p.align == 0


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {fn.__name__}\n        {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
