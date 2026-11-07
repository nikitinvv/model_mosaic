#!/usr/bin/env python
"""Raw 360 deg scans on a grid -> one folded 180 deg stitched.h5, in one pass.

Scans of the same angles (0..360 deg), H x W each, placed on a rectangular
grid of rows x columns (--tiles scan:row:col, or --scans/--step for one
vertical column), e.g. marmoset 180..186 (7 x 1, step 2051) or big_sandstone
082..103 (11 heights x inner/outer, outer column at 3176).
Everything here works on detector rows across ALL angles (fw needs whole
sinograms), so the output is produced one slab of output rows at a time.
For each slab and each scan covering it:

  1. dezinger   tomocupy's remove_outliers: median over [w, 1, w]
                (angles, -, columns), replace where value - median >
                --dezinger-threshold; applied to data, darks and flats
  2. dark-flat  flat for projection i = (1-v) F0 + v F1, v = i/(NT-1);
                F0 = mean of this scan's 20 flats (taken before it),
                F1 = mean of the next scan's 20 flats (taken right after
                this one ended).  The last scan has no successor: F1 = F0.
                (data - dark) / (flat*(1+1e-5) - dark), tomocupy's order
                The flat's light (dark removed) is divided by the tile's
                --bright-ratio (flat exposure / projection exposure).
                "Next scan" = the tile whose scan number is this one + 1.
  3. fw         tomocupy's remove_stripe_fw (sigma, filter, level)
  4. stitch     tiles blended with the 9th-order smootherstep
                S(t) = t^5 (126 - 420t + 540t^2 - 315t^3 + 70t^4) across each
                vertical and horizontal overlap (S(t) + S(1-t) = 1, four
                derivatives zero at the ends); --margin rows / --hmargin
                columns at each tile's edge get weight 0.  Separable weights
                wz(row) * wx(col), normalised to sum 1
  5. fold       360 -> 180 about the axis (--center, tomocupy convention, in
                columns of the wide frame), tomocupy's orientation and the
                same smootherstep across the theta / theta+180 overlap

Output: {path}/model_big1x/stitched.h5, exchange/data (N_HALF, ZO, N)
float32 transmission (no Paganin, no -log), exchange/theta = the first
N_HALF angles.  ZO = stitched rows padded to a multiple of --nz by
repeating the last real row (edge padding, not air: air rows next to the
sample bias Paganin); attr real_rows = the real count.  It is what step7_paganin_large.py reads (--path, --nz ZO, --n N).
Chunks (CT angles, NZ rows, N): one slab = one row of chunks, each a
contiguous block, written with pwrite at the chunk's byte offset.

Roles (no MPI needed; rank/size come from mpirun's env when launched by it):
  --init     create stitched.h5.part (every chunk allocated, 4 KiB aligned)
             and the marker dir stitched.h5.done/
  --worker   one process per GPU; rank r of s takes slabs r, r+s, ...;
             a marker per finished slab, so a rerun continues
  --finish   check all markers, rename .part -> stitched.h5
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/beams2/VNIKITIN/tomocupy/src/tomocupy/processing")
import remove_stripe                      # noqa: E402  tomocupy's fw


class RawRows:
    """Rows [lo, hi) of every frame of an uncompressed (NT, H, W) dataset,
    read with concurrent preads.  h5py reads the 18001 per-frame pieces one
    after another (~1 ms of NFS latency each); a thread pool keeps many in
    flight.  Works for contiguous data and for (1, H, W) chunks without
    filters, whose byte offsets are looked up once."""

    def __init__(self, path, pool):
        self.path, self.pool = path, pool
        with h5py.File(path, "r") as f:
            d = f["exchange/data"]
            self.NT, self.H, self.W = d.shape
            self.dtype = d.dtype
            row = self.W * d.dtype.itemsize
            fb = self.H * row
            if d.chunks is None:
                o = d.id.get_offset()
                self.off = [o + i * fb for i in range(self.NT)]
            else:
                if tuple(d.chunks) != (1, self.H, self.W):
                    raise SystemExit(f"{path}: chunks {d.chunks}")
                self.off = [None] * self.NT
                for i in range(d.id.get_num_chunks()):
                    ci = d.id.get_chunk_info(i)
                    if ci.filter_mask or ci.size != fb:
                        raise SystemExit(f"{path}: compressed/filtered chunks")
                    self.off[ci.chunk_offset[0]] = ci.byte_offset
                if any(o is None for o in self.off):
                    raise SystemExit(f"{path}: missing chunks")
        self.row = row
        self.fd = os.open(path, os.O_RDONLY)

    def read(self, lo, hi):
        out = np.empty((self.NT, hi - lo, self.W), self.dtype)
        n = (hi - lo) * self.row

        def one(i):
            mv = memoryview(out[i]).cast("B")
            got = 0
            while got < n:
                got += os.preadv(self.fd, [mv[got:]], self.off[i] + lo * self.row + got)

        list(self.pool.map(one, range(self.NT), chunksize=64))
        return out

    def close(self):
        os.close(self.fd)


def log(*a):
    print(*a, flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="/data2/19BM/2026-11-Boyer-1026094")
    p.add_argument("--prefix", default="Marmoset_test_1612mm_vertical_full_")
    p.add_argument("--tiles", nargs="+", default=None,
                   help="scan:row:col per tile (unbinned px offsets on the "
                        "stitched frame; a full rectangular grid)")
    p.add_argument("--scans", type=int, nargs="+", default=None,
                   help="one vertical column: scan numbers, top to bottom "
                        "(with --step)")
    p.add_argument("--step", type=int, default=None,
                   help="rows between consecutive --scans")
    p.add_argument("--bright-ratio", type=float, nargs="+", default=[1.0],
                   help="per tile (or one for all): flat exposure / "
                        "projection exposure")
    p.add_argument("--theta", type=float, nargs=2, default=None,
                   metavar=("T0", "T1"),
                   help="linspace(T0, T1, NT) degrees instead of the files' "
                        "exchange/theta")
    p.add_argument("--center", type=float, required=True,
                   help="rotation axis as tomocupy --rotation-axis takes it, "
                        "in columns of the stitched (wide) frame")
    p.add_argument("--dezinger", type=int, default=5)
    p.add_argument("--dezinger-threshold", type=float, default=5000)
    p.add_argument("--fw-sigma", type=float, default=1.0)
    p.add_argument("--fw-filter", default="sym16")
    p.add_argument("--fw-level", type=int, default=7)
    p.add_argument("--margin", type=int, default=64,
                   help="rows at each tile's edge left out of the blend")
    p.add_argument("--hmargin", type=int, default=64,
                   help="columns at each tile's edge left out of the blend")
    p.add_argument("--rows", type=int, nargs=2, default=None,
                   help="stitched rows [R0, R1) only (tests)")
    p.add_argument("--nz", type=int, default=32, help="output rows per slab")
    p.add_argument("--nz-gpu", type=int, default=4, help="rows per GPU pass")
    p.add_argument("--ct", type=int, default=640,
                   help="angles per h5 chunk (capped at 2 GiB per chunk; "
                        "the last chunk may be partial)")
    p.add_argument("--io-threads", type=int, default=16)
    p.add_argument("--min-free-tb", type=float, default=2.0)
    p.add_argument("--path", required=True)
    role = p.add_mutually_exclusive_group(required=True)
    role.add_argument("--init", action="store_true")
    role.add_argument("--worker", action="store_true")
    role.add_argument("--finish", action="store_true")
    return p.parse_args()


def smootherstep(t):
    t = np.clip(t, 0, 1)
    return t**5 * (126 - 420*t + 540*t**2 - 315*t**3 + 70*t**4)


def blend_weights(starts, H, m):
    """Per tile start (H,) weights along one axis, smootherstep across each
    overlap, normalised to sum 1."""
    Z = starts[-1] + H
    w = {}
    for p in starts:
        x = np.arange(p, p + H, dtype=np.float64)
        wp = np.ones(H)
        for q in starts:
            if q == p:
                continue
            if q < p < q + H:                     # q above: rise
                a, b = p + m, q + H - m
                wp = np.minimum(wp, smootherstep((x - a) / max(b - a, 1)))
            elif p < q < p + H:                   # q below: fall
                a, b = q + m, p + H - m
                wp = np.minimum(wp, smootherstep((b - x) / max(b - a, 1)))
        w[p] = wp
    tot = np.zeros(Z)
    for p in starts:
        tot[p:p + H] += w[p]
    if (tot <= 0).any():
        raise SystemExit("uncovered rows; reduce --margin")
    return {p: (w[p] / tot[p:p + H]).astype(np.float32) for p in starts}, Z


def fold_geometry(c, W):
    half = max(c, W - c)
    return 2 * half, half - c, half + c - (W - 1)


def fold_weights(N, W, x0d, x0m):
    """As step56_correct_fold._fold_weights: smootherstep in the overlap,
    and the nearest covered column for each uncovered one."""
    X = np.arange(N)
    cov_d = (X >= x0d) & (X < x0d + W)
    cov_m = (X >= x0m) & (X < x0m + W)
    wd, wm = cov_d.astype(np.float32), cov_m.astype(np.float32)
    both = cov_d & cov_m
    lo, hi = X[both][0], X[both][-1] + 1
    s = smootherstep((X[both] - lo + 0.5) / (hi - lo)).astype(np.float32)
    wd[both] = s if x0d > x0m else 1 - s
    wm[both] = 1 - wd[both]
    covered = cov_d | cov_m
    cidx = np.flatnonzero(covered)
    j = np.searchsorted(cidx, X)
    r = cidx[np.clip(j, 0, len(cidx) - 1)]
    l = cidx[np.clip(j - 1, 0, len(cidx) - 1)]
    nearest = np.where(np.abs(l - X) < np.abs(r - X), l, r)
    return wd, wm, covered, nearest


def dezinger(x, w, thr):
    """tomocupy remove_outliers on an (n, rows, cols) stack, in place."""
    if w > 0:
        f = ndimage.median_filter(x, (w, 1, w))
        x[:] = cp.where((x > f) & (x - f > thr), f, x)
    return x


def main():
    a = parse()
    if a.tiles:
        tiles = [(int(s), int(r), int(c)) for s, r, c in
                 (t.split(":") for t in a.tiles)]
    elif a.scans and a.step is not None:
        tiles = [(s, k * a.step, 0) for k, s in enumerate(a.scans)]
    else:
        raise SystemExit("give --tiles, or --scans with --step")
    names = [s for s, _, _ in tiles]
    brs = a.bright_ratio * len(tiles) if len(a.bright_ratio) == 1 \
        else a.bright_ratio
    if len(brs) != len(tiles):
        raise SystemExit("--bright-ratio: one value, or one per tile")
    files = [f"{a.raw_dir}/{a.prefix}{s:03d}.h5" for s in names]
    with h5py.File(files[0], "r") as f:
        NT, H, W = f["exchange/data"].shape
        theta = f["exchange/theta"][:].astype(np.float64)
    for fn in files[1:]:
        with h5py.File(fn, "r") as f:
            if f["exchange/data"].shape != (NT, H, W):
                raise SystemExit(f"{fn}: shape differs")
    if a.theta is not None:
        theta = np.linspace(a.theta[0], a.theta[1], NT)
    elif not np.any(theta):
        raise SystemExit(f"{files[0]}: exchange/theta is all zero; pass --theta")
    # the next scan's (number + 1) flats close this scan; else its own
    byname = dict(zip(names, files))
    nxt = [byname.get(s + 1) for s in names]

    rows = sorted({r for _, r, _ in tiles})
    cols = sorted({c for _, _, c in tiles})
    if len(rows) * len(cols) != len(tiles) or cols[0] != 0 or rows[0] != 0:
        raise SystemExit("tiles must form a full grid starting at row 0, col 0")
    wz, Z = blend_weights(rows, H, a.margin)
    wx, Wv = blend_weights(cols, W, a.hmargin)
    R0, R1 = a.rows if a.rows else (0, Z)
    ZO = -(-(R1 - R0) // a.nz) * a.nz
    if a.nz % a.nz_gpu:
        raise SystemExit("--nz-gpu must divide --nz")

    i180 = int(np.argmin(np.abs(theta - (theta[0] + 180))))
    if abs(theta[i180] - theta[0] - 180) > 0.5 * abs(theta[1] - theta[0]):
        raise SystemExit("no 180 deg partner for theta[0]")
    # directions theta_j, j < NH, each with its partner j + i180; with a
    # non-integer 180/step (19-BM: 0.0199951 deg) the last few directions
    # before 180 deg have no partner inside the scan and are left out
    NH = min(i180, NT - i180)
    if NH < i180:
        log(f"note: {i180 - NH} directions before 180 deg have no partner; "
            f"N_HALF = {NH} covers {theta[0]:.4f}..{theta[NH - 1]:.4f} deg")
    c = int(a.center)
    left = c < Wv // 2
    if left:
        c -= 1                     # tomocupy's W - c mirrors column c - 1
    N, x0d, x0m = fold_geometry(c, Wv)
    wd, wm, covered, nearest = fold_weights(N, Wv, x0d, x0m)
    # angles per chunk: at most --ct and at most 2 GiB per chunk.  It need
    # not divide N_HALF: theta is the chunk's outermost axis, so the valid
    # part of the partial last chunk is a contiguous prefix of it
    CT = max(1, min(a.ct, NH, 2**31 // (a.nz * N * 4)))

    dst_dir = f"{a.path}/model_big1x"
    out = f"{dst_dir}/stitched.h5"
    part, done_dir = out + ".part", out + ".done"
    slabs = list(range(0, ZO, a.nz))          # output-row slab starts
    out_bytes = NH * ZO * N * 4

    log(f"{len(tiles)} tiles ({len(rows)} rows x {len(cols)} cols), frame "
        f"{H}x{W}, {NT} angles {theta[0]:.4f}..{theta[-1]:.4f}, 180 deg at "
        f"{i180} ({theta[i180]:.4f})")
    log(f"row starts {rows}, col starts {cols}; stitched {Z} x {Wv} "
        f"(margins {a.margin}/{a.hmargin}, smootherstep); rows [{R0},{R1}) "
        f"-> {ZO} with padding; chunk {CT} angles")
    log(f"axis {a.center} (tomocupy) = physical col {c}, "
        f"{'left' if left else 'right'} half; fold N={N} direct@{x0d} "
        f"mirror@{x0m} overlap {int(((wd > 0) & (wm > 0)).sum())} px "
        f"uncovered {int((~covered).sum())}")
    log(f"dezinger {a.dezinger} thr {a.dezinger_threshold}; fw sigma "
        f"{a.fw_sigma} {a.fw_filter} level {a.fw_level}")
    for (s, r, cc), fn, nf, br in zip(tiles, files, nxt, brs):
        log(f"  {os.path.basename(fn)} at ({r}, {cc}) bright-ratio {br}: "
            f"flats own -> {os.path.basename(nf) if nf else 'own (no next scan)'}")
    log(f"out ({NH}, {ZO}, {N}) = {out_bytes / 1e12:.2f} TB -> {out}")

    # ------------------------------------------------------------- init
    if a.init:
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.exists(part):
            with h5py.File(part, "r") as f:
                if f["exchange/data"].shape != (NH, ZO, N):
                    raise SystemExit(f"{part}: shape differs; delete it")
            log(f"  {part} exists: keeping it and its markers (resume)")
        else:
            if os.path.exists(out):
                raise SystemExit(f"{out} exists; delete it to redo")
            free = shutil.disk_usage(dst_dir).free
            log(f"  free {free / 1e12:.1f} TB, need {out_bytes / 1e12:.2f} TB")
            if free - out_bytes < a.min_free_tb * 1e12:
                raise SystemExit("not enough free space")
            shutil.rmtree(done_dir, ignore_errors=True)
            with h5py.File(part, "w", alignment_threshold=1,
                           alignment_interval=4096) as f:
                g = f.create_group("exchange")
                dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
                dcpl.set_chunk((CT, a.nz, N))
                dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
                dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
                dsid = h5py.h5d.create(g.id, b"data", h5py.h5t.IEEE_F32LE,
                                       h5py.h5s.create_simple((NH, ZO, N)),
                                       dcpl=dcpl)
                bad = [i for i in range(dsid.get_num_chunks())
                       if dsid.get_chunk_info(i).byte_offset % 4096]
                dsid.close()
                if bad:
                    raise SystemExit(f"{len(bad)} chunks not 4 KiB aligned")
                g.create_dataset("theta", data=theta[:NH].astype(np.float32))
                for k, v in dict(
                        source=" ".join(files),
                        tiles=" ".join(f"{s}:{r}:{cc}" for s, r, cc in tiles),
                        bright_ratio=brs, rows=(R0, R1), wide_width=Wv,
                        real_rows=R1 - R0, margin=a.margin, hmargin=a.hmargin,
                        center=a.center, axis_convention="tomocupy",
                        axis_physical=c, theta_mirrored=int(left),
                        x0_direct=x0d, x0_mirror=x0m, axis_column=N // 2,
                        dezinger=a.dezinger,
                        dezinger_threshold=a.dezinger_threshold,
                        fw=f"sigma={a.fw_sigma} {a.fw_filter} level={a.fw_level}",
                        flats="per projection, own flats -> next scan's flats",
                        blend="9th-order smootherstep (tiles and fold)",
                        content="dark-flat + fw transmission, stitched and "
                                "folded; no Paganin, no -log; rows past "
                                "real_rows repeat the last real row"
                        ).items():
                    g["data"].attrs[k] = v
        os.makedirs(done_dir, exist_ok=True)
        log(f"  init done: {part}")
        return

    # ----------------------------------------------------------- finish
    if a.finish:
        left_ = [z for z in slabs if not os.path.exists(f"{done_dir}/{z:06d}")]
        if left_:
            raise SystemExit(f"INCOMPLETE: {len(left_)} of {len(slabs)} slabs "
                             f"missing (first {left_[:5]})")
        os.rename(part, out)
        shutil.rmtree(done_dir, ignore_errors=True)
        log(f"DONE -> {out}  (--nz {ZO} --n {N})")
        return

    # ----------------------------------------------------------- worker
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    with h5py.File(part, "r") as f:
        d = f["exchange/data"]
        if tuple(d.chunks) != (CT, a.nz, N):
            raise SystemExit(f"{part}: chunks {d.chunks}, want {(CT, a.nz, N)}")
        choff = {}
        for i in range(d.id.get_num_chunks()):
            ci = d.id.get_chunk_info(i)
            choff[(ci.chunk_offset[0], ci.chunk_offset[1])] = ci.byte_offset
    mine = slabs[rank::size]
    log(f"  [rank {rank}/{size}] {os.uname().nodename.split('.')[0]} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"{len(mine)} slabs")

    pool = ThreadPoolExecutor(a.io_threads)
    wpool = ThreadPoolExecutor(a.io_threads)
    writer = ThreadPoolExecutor(1)
    raws = [RawRows(fn, pool) for fn in files]
    fds = [h5py.File(fn, "r") for fn in files]
    nfds = [h5py.File(fn, "r") if fn else None for fn in nxt]
    fo = os.open(part, os.O_WRONLY)
    v = cp.linspace(0, 1, NT, dtype=cp.float32)[:, None, None]
    wd_d, wm_d = cp.asarray(wd), cp.asarray(wm)
    dst_d = slice(max(x0d, 0), min(x0d + Wv, N))
    src_d = slice(dst_d.start - x0d, dst_d.stop - x0d)
    dst_m = slice(max(x0m, 0), min(x0m + Wv, N))
    src_m = slice(dst_m.start - x0m, dst_m.stop - x0m)
    uncov = cp.asarray(np.flatnonzero(~covered))
    uncov_src = cp.asarray(nearest[np.flatnonzero(~covered)])
    thr = a.dezinger_threshold

    def write_slab(slab, z0, marker):
        tw = time.time()
        piece = 64 << 20
        jobs = []
        for t0 in range(0, NH, CT):
            off = choff[(t0, z0)]
            mv = memoryview(slab[t0:t0 + CT]).cast("B")
            for p0 in range(0, len(mv), piece):
                jobs.append((off + p0, mv[p0:p0 + piece]))

        def put(job):
            off, mv = job
            pos = 0
            while pos < len(mv):
                pos += os.pwrite(fo, mv[pos:], off + pos)

        list(wpool.map(put, jobs))
        os.fdatasync(fo)
        open(marker, "w").close()
        return time.time() - tw

    t_start = time.time()
    t_read = t_comp = t_write = 0.0
    pending = None
    try:
        for n_done, z0 in enumerate(mine, 1):
            marker = f"{done_dir}/{z0:06d}"
            if os.path.exists(marker):
                continue
            if shutil.disk_usage(dst_dir).free < a.min_free_tb * 1e12:
                log(f"  [rank {rank}] free space below {a.min_free_tb} TB, stopping")
                break
            s0, s1 = R0 + z0, min(R0 + z0 + a.nz, R1)   # stitched rows
            cover = []
            for k, (_, p, _) in enumerate(tiles):
                lo, hi = max(s0, p), min(s1, p + H)
                if lo < hi:
                    cover.append((k, p, lo, hi))
            t = time.time()
            raw, aux = {}, {}
            for k, p, lo, hi in cover:
                raw[k] = raws[k].read(lo - p, hi - p)
                dk = fds[k]["exchange/data_dark"][:, lo - p:hi - p, :]
                f0 = fds[k]["exchange/data_white"][:, lo - p:hi - p, :]
                f1 = (nfds[k]["exchange/data_white"][:20, lo - p:hi - p, :]
                      if nfds[k] is not None else None)
                aux[k] = (dk, f0, f1)
            t_read += time.time() - t

            t = time.time()
            slab = np.empty((NH, a.nz, N), np.float32)
            for attempt in range(10):
                try:
                    for g0 in range(s0, s0 + a.nz, a.nz_gpu):
                        g1 = g0 + a.nz_gpu
                        gr = min(g1, s1)          # real rows end
                        res = cp.ones((NH, a.nz_gpu, N), cp.float32)
                        if g0 < gr:
                            acc = cp.zeros((NT, gr - g0, Wv), cp.float32)
                            for k, p, lo, hi in cover:
                                a0, a1 = max(g0, lo), min(gr, hi)
                                if a0 >= a1:
                                    continue
                                i0, i1 = a0 - lo, a1 - lo
                                dk, f0, f1 = aux[k]
                                dk = dezinger(cp.asarray(dk[:, i0:i1], cp.float32), a.dezinger, thr)
                                fl0 = dezinger(cp.asarray(f0[:, i0:i1], cp.float32), a.dezinger, thr).mean(0)
                                if f1 is not None:
                                    fl1 = dezinger(cp.asarray(f1[:, i0:i1], cp.float32), a.dezinger, thr).mean(0)
                                else:
                                    fl1 = fl0
                                dk0 = dk.mean(0)
                                d = cp.asarray(raw[k][:, i0:i1], cp.float32)
                                d = dezinger(d, a.dezinger, thr)
                                fl = (1 - v) * fl0 + v * fl1
                                fl *= cp.float32(1 + 1e-5)
                                fl -= dk0
                                if brs[k] != 1:
                                    fl /= cp.float32(brs[k])
                                d -= dk0
                                d /= fl
                                del fl
                                d = remove_stripe.remove_stripe_fw(
                                    d, a.fw_sigma, a.fw_filter, a.fw_level)
                                cc = tiles[k][2]
                                acc[:, a0 - g0:a1 - g0, cc:cc + W] += \
                                    d * (cp.asarray(wz[p][a0 - p:a1 - p])[None, :, None]
                                         * cp.asarray(wx[cc])[None, None, :])
                                del d
                            if left:    # tomocupy: theta mirrored, theta+180 as is
                                pd = acc[i180:i180 + NH]
                                pm = acc[:NH, :, ::-1]
                            else:
                                pd = acc[:NH]
                                pm = acc[i180:i180 + NH, :, ::-1]
                            o = cp.zeros((NH, gr - g0, N), cp.float32)
                            o[:, :, dst_d] += pd[:, :, src_d] * wd_d[dst_d]
                            o[:, :, dst_m] += pm[:, :, src_m] * wm_d[dst_m]
                            if len(uncov):
                                o[:, :, uncov] = o[:, :, uncov_src]
                            res[:, :gr - g0] = o
                            del acc, pd, pm, o
                        slab[:, g0 - s0:g1 - s0] = res.get()
                        del res
                        cp.get_default_memory_pool().free_all_blocks()
                    break
                except cp.cuda.memory.OutOfMemoryError as e:
                    cp.get_default_memory_pool().free_all_blocks()
                    log(f"  [rank {rank}] GPU OOM rows [{s0},{s1}) attempt "
                        f"{attempt + 1}/10: {e}; retry in 60 s")
                    time.sleep(60)
            else:
                raise RuntimeError(f"rows [{s0},{s1}): GPU memory unavailable")
            if s1 - s0 < a.nz:                  # padding: repeat the last real row
                slab[:, s1 - s0:] = slab[:, s1 - s0 - 1:s1 - s0]
            t_comp += time.time() - t

            t = time.time()
            if pending is not None:
                t_write += pending.result()
            pending = writer.submit(write_slab, slab, z0, marker)
            log(f"  [rank {rank}] slab {n_done}/{len(mine)} rows [{s0},{s1}) "
                f"scans {[names[k] for k, *_ in cover]} read={t_read:.0f}s "
                f"comp={t_comp:.0f}s write={t_write:.0f}s "
                f"elapsed={time.time() - t_start:.0f}s")
            del raw, aux, slab
        if pending is not None:
            t_write += pending.result()
    finally:
        writer.shutdown(wait=True)
        wpool.shutdown()
        os.close(fo)
        for f in fds + [f for f in nfds if f is not None]:
            f.close()
        for r in raws:
            r.close()
        pool.shutdown()
    log(f"  [rank {rank}] finished after {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
