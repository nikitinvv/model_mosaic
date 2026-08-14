"""Chunk / vchunk sizing policy for the mosaic pipeline.

Pure arithmetic — nothing here touches the filesystem, so the steps, the
throughput test, and the drawing script can all ask the same question and
get the same answer.

Two different things get sized here, and conflating them is the usual
source of confusion:

  vchunks  the super-chunk a rank buffers in RAM and hands to tomo_writex.
           It can only tile the BANKED axis (θ for stype='proj', z for
           'slice') because tomo_initx lays the VDS out with full ny, nx
           per bank -- and because Radon/Fresnel need whole planes anyway.
           So the per-rank buffer floor is nbanks x one full plane, which
           is 36 MB at UPS=1 and 36 GB at UPS=32.  That is why nbanks is
           an output of plan_banking, not an input it obeys.

  chunks   the HDF5 chunk shape inside the bank files.  Purely a
           performance knob, independent of the banking (see the warning
           in dxchange_hdf5_chunks.tomo_initx).

The old code hardcoded both for UPS=1 -- fixed counts like 8*NZCHUNK and
full-plane chunks -- so both grew as N^2 or N^3.  At UPS=16 a full-plane
chunk is 9 GB, past HDF5's 4 GiB/chunk limit, and step8's default vchunk
buffer was already 9.7 GB at UPS=1.  The policy here fixes the *bytes*
and lets the shape follow N:

    UPS=1   (1, 3072, 3072)   36 MB        UPS=8   (1,  512, 24576)  48 MB
    UPS=2   (1, 2048, 6144)   48 MB        UPS=16  (1,  256, 49152)  48 MB
    UPS=4   (1, 1024, 12288)  48 MB        UPS=32  (1,  128, 98304)  48 MB
"""
from __future__ import annotations

from math import gcd

# HDF5 stores a chunk's size in a uint32, so a chunk must be < 4 GiB.
# Hit it and file creation fails outright, before any I/O happens.
HDF5_MAX_CHUNK_BYTES = 2 ** 32 - 1

# Per-rank RAM ceiling covering BOTH buffers every step holds: the output
# vchunk and the prefetched input slab.  Sizing only the first is what let
# step8 ask for 9.7 GB at UPS=1 while also allocating a 1.8 GB sinogram.
#
# 96 GiB = a 512 GB Polaris node / 4 ranks, less ~25% for the worker pool,
# page cache and CUDA host staging.  Scale it with ranks-per-node: the run
# scripts pass NODE_GB * 0.75 / NRANKS.
DEFAULT_MEM_BUDGET = 96 * 2 ** 30

# Target bytes for one HDF5 chunk.  This is the op size Lustre sees; the
# last Polaris sweep only ever measured 12.6 MB, so treat the default as
# a starting point and sweep it (polaris_test_h5.sh --chunk-bytes).
DEFAULT_CHUNK_BYTES = 64 * 2 ** 20


# --------------------------------------------------------------------------
# small integer helpers
# --------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b) if a and b else max(a, b, 1)


def _divisors(n: int) -> list[int]:
    """All divisors of n, ascending.  n here is at most ~1e5, so trial
    division to sqrt(n) is a few hundred iterations."""
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


def _largest_divisor_le(n: int, want: int) -> int:
    """Largest divisor of `n` that is <= want.

    Used to snap every extent onto a value that tiles its axis exactly, so
    a chunk is never straddled and a super-chunk never leaves a ragged
    tail.
    """
    want = max(1, min(int(want), n))
    best = 1
    for d in _divisors(n):
        if d <= want:
            best = d
    return best


def _pick_extent(total: int, want: int, hard: int, soft: int) -> int:
    """Largest divisor of `total` that is <= `want` and a multiple of
    `hard`, preferring one that is also a multiple of `soft`.

    `hard` is a genuine correctness constraint -- an extent that is not a
    multiple of it produces a WRONG file, so it wins over the budget.  No
    dataset currently needs one (step1's C0-multiple-of-UPS used to be
    declared here, but its z-interpolation turned out to be indexed by
    absolute output z, so a misaligned edge only costs a re-read); the
    mechanism stays because the next constraint of that kind should not
    have to be retrofitted.

    `soft` is a preference the budget may overrule: a compute-loop
    granularity (NZCHUNK / NPROPCHUNK / NPGNCHUNK), or step1's UPS, where
    missing it costs some repeated work but nothing else.

    When even the smallest legal extent busts the budget the floor wins:
    the caller reports it rather than returning something unusable.
    """
    hard = max(1, int(hard))
    soft = max(1, int(soft))
    cands = [d for d in _divisors(total) if d % hard == 0] or [min(hard, total)]
    ok = [d for d in cands if d <= want]
    if not ok:
        return min(cands)
    pref = [d for d in ok if d % soft == 0]
    return max(pref) if pref else max(ok)


def _sitems_axis(stype: str) -> int:
    """Axis the BANK FILES split along: θ for 'proj', z for 'slice'."""
    return 0 if str(stype).lower().startswith("proj") else 1


def _plane_bytes(shape, axis: int, itemsize: int) -> int:
    """Bytes of one index along `axis` -- one θ-plane, or one z-sinogram."""
    n = itemsize
    for a, s in enumerate(shape):
        if a != axis:
            n *= int(s)
    return n


# --------------------------------------------------------------------------
# vchunks + nbanks
# --------------------------------------------------------------------------

def plan_banking(shape, stype, nbanks_want, itemsize, budget,
                 companion_unit: int = 0, align: int = 1, align_hard: int = 1,
                 nranks: int = 1):
    """Size the super-chunk (and the bank count) to a per-rank RAM budget.

    `companion_unit` is what the step's *input prefetch* buffer costs per
    unit of the banked axis.  step8, for instance, allocates both a
    (C0, N, N) rec vchunk and a (NTHETA, C0, N) sinogram slab, so its true
    cost per unit of C0 is (N*N + NTHETA*N)*itemsize.  Passing 0 here is
    what made the old formulas understate memory by up to 2x.

    `align_hard` is a correctness constraint on the super-chunk extent --
    an extent violating it makes a wrong file, so it beats the budget.
    `align` is only a preference: the step's compute-loop chunk (NZCHUNK /
    NPROPCHUNK / NPGNCHUNK), or step1's UPS.  Missing it costs a smaller
    loop or a repeated read, never correctness, so it is bent rather than
    allowed to inflate the buffer.

    The extent is chosen from the divisors of the banked axis, so there is
    never a ragged tail super-chunk; nbanks is then a divisor of it, so
    every bank gets identical work.  nbanks therefore comes back smaller
    than requested at high UPS: one full plane is 36 GB at UPS=32, so
    eight banks would need 288 GB.  Write parallelism has to come from
    ranks instead -- unavoidable while a vchunk must hold whole planes.

    Returns (nbanks, vchunks, buffer_bytes, align_eff).
    """
    shape = tuple(int(s) for s in shape)
    ax = _sitems_axis(stype)
    total = shape[ax]

    unit = _plane_bytes(shape, ax, itemsize) + int(companion_unit)
    cap = max(1, int(budget) // unit)

    # A super-chunk is also the unit of rank parallelism (ranks iterate
    # ivchunks[RANK::SIZE]), so filling the budget with one huge vchunk
    # can leave most ranks idle -- at 96 GiB and UPS=1 a maximal rec
    # vchunk gives 4 super-chunks for 8 ranks.  Cap the extent so there is
    # at least one super-chunk per rank; smaller buffers are a bonus.
    cap = min(cap, max(1, total // max(1, int(nranks))))

    per = _pick_extent(total, min(cap, total), align_hard, align)
    nbanks = _largest_divisor_le(per, max(1, int(nbanks_want)))
    align_eff = max(1, gcd(max(1, int(align)), per))

    vchunks = list(shape)
    vchunks[ax] = per
    return nbanks, tuple(vchunks), per * unit, align_eff


# --------------------------------------------------------------------------
# HDF5 chunk shape
# --------------------------------------------------------------------------

# Axis growth order.  x always goes first so a chunk is a whole number of
# full rows; what follows decides whether the chunk reads well by θ or by z.
_PRIORITY = {
    "proj": (2, 1, 0),   # -> (1, cz, N)          projection-ordered
    "sino": (2, 0, 1),   # -> (theta_bank, cz, N) sinogram-ordered
}


def nominal_bank_shape(shape, vchunks, stype, nbanks):
    """The bank-file shape tomo_initx will create -- mirrors its own
    `nominal_bank` computation so chunk sizing sees the same extents."""
    ax = _sitems_axis(stype)
    per_bank = (int(vchunks[ax]) + int(nbanks) - 1) // int(nbanks)
    bank = list(int(s) for s in shape)
    bank[ax] = per_bank
    return tuple(bank)


def plan_chunks(bank_shape, order, itemsize, chunk_bytes,
                dim0_one: bool = False, z_granule: int | None = None):
    """Grow a chunk toward `chunk_bytes`, snapping every extent onto a
    divisor of the bank extent so chunks tile the bank exactly.

    On when the consumer's read granule matters: inside a chunk the layout
    is plain C-order, so a contiguous range along the OUTERMOST axis is
    contiguous storage, and a chunk bigger than the read costs nothing
    there.  A sub-range on axis `a` only costs extra ops when some axis
    before it has extent > 1.  x is always grown to full width first, so
    that reduces to a single case: axis 1 (z) binds only when chunk[0] > 1,
    i.e. only for sinogram-ordered paganin.h5, where cz must divide the FBP
    z-slab.  Hence `z_granule` is the one granule argument here -- clamping
    the others would shrink chunks below the byte target for no gain.

    (This is the mechanism behind the last Polaris sweep: sino:0 read
    whole 12.6 MB chunks, sino:1 and proj:0 both paid 1152 ops.)
    """
    bank_shape = tuple(int(s) for s in bank_shape)
    cap = min(int(chunk_bytes), HDF5_MAX_CHUNK_BYTES)
    order = "sino" if str(order).lower().startswith("sino") else "proj"

    limit = list(bank_shape)
    if dim0_one:
        limit[0] = 1

    chunk = [1, 1, 1]
    for ax in _PRIORITY[order]:
        lim, tile = limit[ax], bank_shape[ax]
        # z is the one axis a consumer granule can bind, and only once θ
        # has actually grown past 1 -- with chunk[0] == 1 the z range is
        # the outermost non-trivial axis, so any contiguous z sub-range is
        # one contiguous run and a fatter chunk is strictly better.
        if ax == 1 and chunk[0] > 1 and z_granule and int(z_granule) > 0:
            lim = min(lim, int(z_granule))
            tile = gcd(tile, int(z_granule))
        rest = itemsize
        for a in range(3):
            if a != ax:
                rest *= chunk[a]
        room = max(1, cap // rest)
        chunk[ax] = _largest_divisor_le(tile, min(lim, room))
    nbytes = itemsize
    for c in chunk:
        nbytes *= c

    # No fallback to the projection order when the clamped sino chunk comes
    # out under the byte target.  It is tempting -- chunk[0] == 1 there, so
    # the consumer's z sub-range is one contiguous run and the chunk itself
    # can be much fatter -- but the run is only z_granule*N*itemsize long
    # and there is one PER θ, where the sino order pays one op per (bank,
    # z-chunk).  Measured at UPS=2 / 32 ranks, FBP read:
    #     (18, 96, 6144)  40.5 MB sino  ->  3.55 GB/s   (64 ops of 40.5 MB)
    #     (1, 6144, 6144) 144  MB proj  ->  0.25 GB/s   (576 ops of 4.7 MB)
    # A clamped sino chunk is already the largest shape that keeps the FBP
    # read whole-chunk (θ full within the bank, z = the granule, x full), so
    # when it lands small there is nothing better to grow into.  Where the
    # bank holds a single θ (UPS=32 at a 64 GiB budget) chunk[0] is 1 anyway
    # and the order degenerates to proj on its own.

    if nbytes > HDF5_MAX_CHUNK_BYTES:      # unreachable; cheap to assert
        raise ValueError(f"planned chunk {tuple(chunk)} is {nbytes} B, "
                         f"over HDF5's {HDF5_MAX_CHUNK_BYTES} B limit")
    return tuple(chunk)


# --------------------------------------------------------------------------
# whole-pipeline plan
# --------------------------------------------------------------------------

class Plan:
    """Sizing for one dataset.  `note` explains why, for the logs."""

    __slots__ = ("name", "shape", "stype", "nbanks", "vchunks", "chunks",
                 "order", "buffer_bytes", "itemsize", "align", "budget",
                 "read_granule", "unit_bytes", "dim0_one", "z_granule",
                 "note")

    def __init__(self, name, shape, stype, nbanks, vchunks, chunks, order,
                 buffer_bytes, itemsize, align=1, budget=0,
                 read_granule=None, unit_bytes=0, dim0_one=False,
                 z_granule=None, note=""):
        self.read_granule = tuple(int(g) for g in read_granule) \
            if read_granule is not None else None
        # bytes of buffer per unit of the banked axis (output + companion),
        # kept so an overridden vchunk can re-price itself
        self.unit_bytes = int(unit_bytes)
        self.dim0_one = bool(dim0_one)      # chunk[0] must stay 1
        self.z_granule = int(z_granule) if z_granule else None
        self.name = name
        self.shape = tuple(int(s) for s in shape)
        self.stype = stype
        self.nbanks = int(nbanks)
        self.vchunks = tuple(int(v) for v in vchunks)
        self.chunks = tuple(int(c) for c in chunks)
        self.order = order
        self.buffer_bytes = int(buffer_bytes)
        self.itemsize = int(itemsize)
        self.align = int(align)          # compute-loop chunk that fits vchunks
        self.budget = int(budget)
        self.note = note

    @property
    def effective_order(self) -> str:
        """What the chunk actually is, which can differ from what was
        asked for: once a bank holds a single θ (nbanks >= vchunks[0], the
        norm from UPS=8 up) a sinogram chunk degenerates into a projection
        chunk, and growing z is then the right thing to do."""
        return "sino" if self.chunks[0] > 1 and self.order == "sino" else "proj"

    @property
    def over_budget(self) -> bool:
        """True when even the smallest legal super-chunk busts the budget --
        i.e. the step cannot run in `budget` RAM without a code change."""
        return bool(self.budget) and self.buffer_bytes > self.budget

    @property
    def bank_shape(self):
        return nominal_bank_shape(self.shape, self.vchunks,
                                  self.stype, self.nbanks)

    @property
    def chunk_bytes(self):
        n = self.itemsize
        for c in self.chunks:
            n *= c
        return n

    @property
    def total_bytes(self):
        n = self.itemsize
        for s in self.shape:
            n *= s
        return n

    @property
    def n_vchunks(self):
        return int(-(-self.shape[_sitems_axis(self.stype)]
                     // self.vchunks[_sitems_axis(self.stype)]))

    def __repr__(self):
        return (f"Plan({self.name}, shape={self.shape}, stype={self.stype}, "
                f"nbanks={self.nbanks}, vchunks={self.vchunks}, "
                f"chunks={self.chunks})")


def plan_pipeline(ups, *, in_nz=3072, in_nyx=3072, ntheta=None, nbanks=8,
                  budget=DEFAULT_MEM_BUDGET, chunk_bytes=DEFAULT_CHUNK_BYTES,
                  nzchunk=32, npropchunk=8, npgnchunk=None, itemsize=4,
                  nranks=1):
    """Size every dataset in the pipeline.

    Planned BACKWARDS -- rec first, init last -- because each producer's
    chunk shape depends on how its consumer reads it.  That is what
    dissolves the step7 <-> step8 coupling: both call this and read
    rec.vchunks[0], so there is no --chunk-z for the run scripts to keep
    in sync with a rec vchunk they never pass.
    """
    ups = int(ups)
    npgnchunk = int(npgnchunk if npgnchunk is not None else npropchunk)
    nzchunk, npropchunk = int(nzchunk), int(npropchunk)

    NZ = int(in_nz) * ups
    N = int(in_nyx) * ups
    NTHETA = int(ntheta) if ntheta else 3 * N // 4
    N_HALF = NTHETA // 2

    plane = NZ * N * itemsize          # one θ-plane of a (θ, NZ, N) volume
    out = {}

    def add(name, shape, stype, order, align, companion_unit,
            dim0_one, z_granule=None, align_hard=1, read_granule=None,
            note=""):
        nb, vc, buf, aeff = plan_banking(
            shape, stype, nbanks, itemsize, budget,
            companion_unit=companion_unit, align=align,
            align_hard=align_hard, nranks=nranks)
        ch = plan_chunks(nominal_bank_shape(shape, vc, stype, nb),
                         order, itemsize, chunk_bytes,
                         dim0_one=dim0_one, z_granule=z_granule)
        ax = _sitems_axis(stype)
        out[name] = Plan(name, shape, stype, nb, vc, ch, order, buf,
                         itemsize, align=aeff, budget=int(budget),
                         read_granule=read_granule,
                         unit_bytes=buf // max(1, vc[ax]),
                         dim0_one=dim0_one, z_granule=z_granule, note=note)
        return out[name]

    # 8. rec.h5 -- terminal, nobody reads it in-pipeline.  step8 holds the
    #    (C0, N, N) output plus a (NTHETA, C0, N) sinogram prefetch.
    rec = add("rec", (NZ, N, N), "proj", "proj",
              align=nzchunk, companion_unit=NTHETA * N * itemsize,
              dim0_one=True,
              note="terminal; chunk[0]=1 for slice-wise viewing")

    # 7. paganin.h5 -- written a θ-slab at a time (so banking is on θ) but
    #    read (all θ, zslab, N) by step8 (so chunks must be on z).  The one
    #    place a consumer granule binds.
    pgn = add("paganin", (N_HALF, NZ, N), "proj", "sino",
              align=npgnchunk, companion_unit=plane, dim0_one=False,
              z_granule=rec.vchunks[0],
              read_granule=(N_HALF, rec.vchunks[0], N),
              note=f"sinogram-ordered; cz divides FBP z-slab {rec.vchunks[0]}")

    # 6. stitched.h5 -- step7 reads a θ-slab of full planes, so plain
    #    projection order is already whole-chunk and sequential.
    add("stitched", (N_HALF, NZ, N), "proj", "proj",
        align=1, companion_unit=0, dim0_one=False,
        read_granule=(pgn.vchunks[0], NZ, N),
        note=f"read as ({pgn.vchunks[0]}, NZ, N) θ-slabs by step7")

    # 3. data.h5 -- step3 holds the (C0, NZ, N) output plus a same-sized
    #    proj prefetch.  step4 reads sub-windows, which stay contiguous
    #    per row at full chunk width.
    data = add("data", (NTHETA, NZ, N), "proj", "proj",
               align=npropchunk, companion_unit=plane, dim0_one=True,
               read_granule=(1, NZ, N),
               note="chunk[0]=1 (required); full-width rows for step4 tiles")

    # 2. proj.h5 -- z-banked.  step3 reads (θ-slab, all z, N), so a bank's
    #    whole z extent belongs in one chunk and θ fills the rest.
    prj = add("proj", (NTHETA, NZ, N), "slice", "proj",
              align=nzchunk, companion_unit=N * N * itemsize, dim0_one=False,
              read_granule=(data.vchunks[0], NZ, N),
              note=f"z-banked; read as ({data.vchunks[0]}, NZ, N) θ-slabs "
                   f"by step3")

    # 1. big{U}x.h5 -- step2 reads (proj.vchunks[1], N, N) z-slabs.  step1
    #    streams input planes, ~1/ups of an output plane each.
    #
    #    C0 a multiple of UPS is PREFERRED, not required.  A vchunk edge
    #    inside an input interval costs the seam: step1 re-reads and
    #    re-upsamples the straddled input plane pair, so the fixed ~2
    #    reads + 2 xy-zooms per vchunk get paid ups/C0 times more often.
    #    Its z-interpolation is already written against absolute output z
    #    (z0_in = z0_out//ups, and the out_z range guard skips the rest),
    #    so the result is bit-identical either way.
    #
    #    Making it hard is what used to put big over budget from UPS=16:
    #    one output plane is 9.7 GB at UPS=16 and 38.7 GB at UPS=32, so the
    #    smallest legal super-chunk was 16 resp. 32 of them (144 GB, 1.1 TB)
    #    when 8 resp. 2 planes would have fitted 96 GiB comfortably.
    big = add("big", (NZ, N, N), "proj", "proj",
              align=ups, align_hard=1,
              companion_unit=(in_nyx * in_nyx * itemsize) // ups,
              dim0_one=True,
              read_granule=(prj.vchunks[1], N, N),
              note=f"chunk[0]=1 (required); C0 prefers a multiple of "
                   f"ups={ups} (seam cost only)")

    # 0. init.h5 -- written by step00 as a plain file; the throughput test
    #    banks it.  Either way the chunk shape is the same.
    add("init", (int(in_nz), int(in_nyx), int(in_nyx)), "proj", "proj",
        align=1, companion_unit=0, dim0_one=True,
        read_granule=(max(1, big.vchunks[0] // ups), int(in_nyx), int(in_nyx)),
        note="chunk[0]=1 (required)")

    # Return in pipeline order rather than planning order.
    return {k: out[k] for k in
            ("init", "big", "proj", "data", "stitched", "paganin", "rec")}


# --------------------------------------------------------------------------
# step-facing API
# --------------------------------------------------------------------------

def add_layout_args(p, budget_gb=DEFAULT_MEM_BUDGET / 2 ** 30,
                    chunk_mb=DEFAULT_CHUNK_BYTES / 2 ** 20):
    """Add the two knobs every step shares.  Kept in one place so a sweep
    can move them together across the whole pipeline.

    The budget default assumes a 512 GB Polaris node shared by 4 ranks
    with ~25% left for the worker pool and page cache.  Run scripts that
    use a different ranks-per-node should pass --mem-budget explicitly."""
    p.add_argument("--mem-budget", type=float, default=budget_gb,
                   metavar="GiB",
                   help="per-rank RAM for the vchunk buffer PLUS the input "
                        "prefetch slab; sizes the super-chunk "
                        f"(default {budget_gb})")
    p.add_argument("--chunk-bytes", type=float, default=chunk_mb,
                   metavar="MiB",
                   help="target size of one HDF5 chunk = the op size Lustre "
                        f"sees (default {chunk_mb})")
    return p


def resolve_step(name, *, ups, in_nz, in_nyx, ntheta=None, nbanks=8,
                 mem_budget_gb=DEFAULT_MEM_BUDGET / 2 ** 30,
                 chunk_mb=DEFAULT_CHUNK_BYTES / 2 ** 20,
                 nzchunk=32, npropchunk=8, npgnchunk=None,
                 vchunks=None, chunks=None, order=None, itemsize=4,
                 nranks=1):
    """The Plan for one dataset, with any explicit CLI overrides applied.

    `vchunks` / `chunks` / `order` are the escape hatches: pass what the
    user typed (or None) and the planned value survives untouched when
    they typed nothing.  An overridden vchunk re-derives the chunk shape
    and the compute-loop alignment from it, so the two never drift.
    """
    plans = plan_pipeline(ups, in_nz=in_nz, in_nyx=in_nyx, ntheta=ntheta,
                          nbanks=nbanks,
                          budget=int(mem_budget_gb * 2 ** 30),
                          chunk_bytes=int(chunk_mb * 2 ** 20),
                          nzchunk=nzchunk, npropchunk=npropchunk,
                          npgnchunk=npgnchunk, itemsize=itemsize,
                          nranks=nranks)
    p = plans[name]
    if order is not None:
        p.order = "sino" if str(order).startswith("sino") else "proj"

    if vchunks is not None:
        ax = _sitems_axis(p.stype)
        p.vchunks = tuple(int(v) for v in vchunks)
        p.nbanks = _largest_divisor_le(p.vchunks[ax], max(1, int(nbanks)))
        p.buffer_bytes = p.vchunks[ax] * p.unit_bytes
        soft = {"proj": nzchunk, "data": npropchunk, "rec": nzchunk,
                "paganin": npgnchunk if npgnchunk is not None else npropchunk,
                }.get(name, 1)
        p.align = max(1, gcd(max(1, int(soft)), p.vchunks[ax]))

    if vchunks is not None or order is not None:
        p.chunks = plan_chunks(p.bank_shape, p.order, itemsize,
                               int(chunk_mb * 2 ** 20),
                               dim0_one=p.dim0_one, z_granule=p.z_granule)
    if chunks is not None:
        p.chunks = tuple(min(int(c), int(s))
                         for c, s in zip(chunks, p.bank_shape))
    return p


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

def hbytes(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(b) < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} EB"


def describe_plan(plans, prefix="  ") -> str:
    """One line per dataset, for the run logs.  A leading '!' marks a step
    whose per-rank buffer cannot fit the budget even at its smallest legal
    super-chunk -- raise --mem-budget or the step will OOM."""
    lines = []
    for p in plans.values():
        flag = "!" if p.over_budget else " "
        lines.append(
            f"{prefix}{flag}{p.name:9s} {str(p.shape):26s}"
            f" {hbytes(p.total_bytes):>9s}"
            f"  stype={p.stype:5s} nbanks={p.nbanks:<3d}"
            f" vchunk={str(p.vchunks):24s} buf={hbytes(p.buffer_bytes):>9s}"
            f"  chunk={str(p.chunks):22s}"
            f" ({hbytes(p.chunk_bytes)}, {p.effective_order})")
    return "\n".join(lines)


if __name__ == "__main__":                     # quick manual inspection
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ups", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--nbanks", type=int, default=8)
    ap.add_argument("--in-nz", type=int, default=3072)
    ap.add_argument("--in-nyx", type=int, default=3072)
    ap.add_argument("--nranks", type=int, default=1,
                    help="total MPI ranks; caps the vchunk so none idle")
    ap.add_argument("--mem-budget", type=float, default=96.0,
                    help="GiB per rank")
    ap.add_argument("--chunk-bytes", type=float, default=64.0, help="MiB")
    a = ap.parse_args()
    for u in a.ups:
        print(f"=== UPS={u}  nbanks<={a.nbanks}  nranks={a.nranks}  "
              f"budget={a.mem_budget} GiB  chunk~{a.chunk_bytes} MiB ===")
        print(describe_plan(plan_pipeline(
            u, in_nz=a.in_nz, in_nyx=a.in_nyx, nbanks=a.nbanks,
            nranks=a.nranks, budget=int(a.mem_budget * 2 ** 30),
            chunk_bytes=int(a.chunk_bytes * 2 ** 20))))
        print()
