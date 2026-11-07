#!/usr/bin/env python
"""Slice a single-file `data.h5` (from step2_model_*.py) into per-tile HDF5
files using the mosaic layout from step0_schematic.py.

Reads {path}/model_big{UPS}x/data.h5 which has
    /exchange/data   (NTHETA, NZ, N) float32
    /exchange/theta  (NTHETA,)       float32   angles in DEGREES

For every (z_tile, x_tile) mosaic position, writes one h5:
    {path}/mosaic_h5/{z_idx}_{x_idx}.h5
        /exchange/data   (NTHETA, h, w) float32   variable per-tile crop
        /exchange/theta  (NTHETA,)      float32

Tile placement is read from
    mosaic_modeling/mosaic_positions/mosaic_positions{UPS}.txt
(the file step0_schematic.py writes).  Each row gives the detector centre
pixel in the big projection plus crop_{top,bottom,left,right}; the stored
tile is exactly the part of the detector footprint that falls inside
[0, NZ) x [0, N) — no air padding.

Multi-rank via MPI.  There are only n_z*n_x tiles (6 at UPS=2) but each is
hundreds of GB, so one tile per rank both idles most of the job and asks a
single rank to hold ~700 GB.  Instead COMM is split into a group per tile;
each member owns a contiguous theta span, streams it in --block-gb blocks,
and writes its own plain HDF5 *bank* file.  Rank 0 of the group then stitches
the banks into the tile's VDS master, so step5 still opens one
mosaic_h5/{z}_{x}.h5 and reads it as a single (NTHETA, h, w) dataset.

Banks rather than parallel HDF5 on purpose: h5py has mpio here, but a file
created with MPI-IO gets H5D_ALLOC_TIME_EARLY, and on Lustre that allocation
is not sparse -- measured 532 s to create one empty 348 GB tile before any
real data is written.  Independent per-rank banks allocate lazily and need
no coordination, which is why the rest of the pipeline is built this way too.
"""
from __future__ import annotations

import argparse
import os
import re

import h5py
import numpy as np

import time
from mpi_utils import COMM, RANK, SIZE, barrier as _barrier, rprint, report_stage
from iohdf5.h5_vchunks import describe_input
from iohdf5.h5_vchunks import rm_src, add_rm_src_arg


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_POSITIONS_DIR = os.path.join(_SCRIPT_DIR, "mosaic_positions")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=2,
                   help="matches step2_model_*.py --ups (drives src path + positions file)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic",
                   help="base dir; reads {path}/model_big{UPS}x/data.h5, "
                        "writes {path}/mosaic_h5/*.h5")
    p.add_argument("--block-gb", type=float, default=4.0,
                   help="host buffer budget for one theta-block; each rank "
                        "holds a direct + a mirror block, so peak is ~2x this")
    add_rm_src_arg(p, "the input data.h5")
    return p.parse_args()


def read_placements(txt_path: str):
    """Parse mosaic_positions{UPS}.txt.  Returns (meta, placements) where
    `meta` carries NZ/N/DET_H/DET_W/n_z/n_x/NTHETA/N_HALF/OVERLAP and
    `placements` is a list of per-tile dicts.  Only the DIRECT placement
    is stored in the txt; the mirror is derived here: same z_center,
    x_center_mir = N - x_center, z-crops shared, x-crops swap under the
    h-flip.  Each placement carries both direct and mirror big-proj
    crop rectangles precomputed."""
    with open(txt_path, "r") as f:
        header_lines = []
        rows = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header_lines.append(s.lstrip("#").strip())
                continue
            rows.append([int(x) for x in s.split()])
    kv = {}
    for line in header_lines:
        for m in re.finditer(r"(\w+)\s*=\s*(-?\d+)", line):
            kv[m.group(1)] = int(m.group(2))
    need = ("NZ", "N", "DET_H", "DET_W", "n_z", "n_x",
            "NTHETA", "N_HALF", "OVERLAP")
    missing = [k for k in need if k not in kv]
    if missing:
        raise ValueError(f"positions file {txt_path} missing header keys: {missing}")
    meta = {k: kv[k] for k in need}
    DET_H, DET_W, N = meta["DET_H"], meta["DET_W"], meta["N"]

    placements = []
    for r in rows:
        zi, xi, zc, xc, ct, cb, cl, cr = r
        # Direct footprint.
        z0d = zc - DET_H // 2
        x0d = xc - DET_W // 2
        r_lo_d, r_hi_d = z0d + ct, z0d + DET_H - cb
        c_lo_d, c_hi_d = x0d + cl, x0d + DET_W - cr
        # Mirror footprint — derived.  Same z; x reflected around N/2;
        # h-flip swaps left/right crops.
        xc_mir = N - xc
        x0m    = xc_mir - DET_W // 2
        r_lo_m, r_hi_m = r_lo_d, r_hi_d
        c_lo_m, c_hi_m = x0m + cr, x0m + DET_W - cl
        placements.append(dict(
            zi=zi, xi=xi,
            z_center=zc, x_center=xc,
            crop_top=ct, crop_bottom=cb, crop_left=cl, crop_right=cr,
            r_lo_dir=r_lo_d, r_hi_dir=r_hi_d,
            c_lo_dir=c_lo_d, c_hi_dir=c_hi_d,
            r_lo_mir=r_lo_m, r_hi_mir=r_hi_m,
            c_lo_mir=c_lo_m, c_hi_mir=c_hi_m,
        ))
    return meta, placements


def bank_rel(zi: int, xi: int, r: int) -> str:
    """Bank path as written into the VDS master -- relative to the master, so
    the whole mosaic_h5/ tree stays relocatable."""
    return f"{zi}_{xi}/bank{r:05d}.h5"


def theta_spans(n: int, k: int):
    """k contiguous, near-equal spans covering [0, n).

    Contiguous rather than round-robin: each span becomes one bank file and
    two VDS mappings (its direct half and its mirror half), so a rank's work
    must be one unbroken theta range.
    """
    base, rem = divmod(n, k)
    out, s = [], 0
    for i in range(k):
        ln = base + (1 if i < rem else 0)
        out.append((s, s + ln))
        s += ln
    return out


def main() -> None:
    args = _parse_args()

    from mpi_utils import banner
    banner("4", f"data.h5 -> mosaic_h5/*.h5  (crop tiles, synthesise 360 deg scan)")

    src_h5   = f"{args.path}/model_big{args.ups}x/data.h5"
    dst_dir  = f"{args.path}/mosaic_h5"
    positions_path = os.path.join(_POSITIONS_DIR,
                                  f"mosaic_positions{args.ups}.txt")

    if RANK == 0:
        os.makedirs(dst_dir, exist_ok=True)
    _barrier()

    if not os.path.exists(src_h5):
        raise SystemExit(f"missing input: {src_h5}")
    if not os.path.exists(positions_path):
        raise SystemExit(
            f"missing positions file: {positions_path}\n"
            f"run `python step0_schematic.py --ups {args.ups}` first.")

    meta, placements = read_placements(positions_path)
    DET_H, DET_W = meta["DET_H"], meta["DET_W"]
    n_z, n_x = meta["n_z"], meta["n_x"]
    NTHETA_TILE = meta["NTHETA"]        # 360°-scan angle count in tile files
    N_HALF      = meta["N_HALF"]        # 180°-tomo angle count in data.h5

    # Peek at data.h5 to get shape + 180°-tomo theta.
    with h5py.File(src_h5, "r") as f:
        n_data, NZ, N = f["exchange/data"].shape
        theta_data_deg = f["exchange/theta"][:]

    if n_data != N_HALF:
        raise SystemExit(
            f"data.h5 has {n_data} angles but positions expect N_HALF={N_HALF} "
            f"(180° tomo).  Re-run step2/step3 with the new NTHETA=3·N/8 default.")
    if (NZ, N) != (meta["NZ"], meta["N"]):
        raise SystemExit(
            f"positions file expects big proj (NZ={meta['NZ']}, N={meta['N']}) "
            f"but data.h5 is ({NZ}, {N}) — regenerate step0 for --ups {args.ups}")

    # Tile files store a synthetic 360° scan: first N_HALF angles are the direct
    # data.h5 slice; second N_HALF are the mirror slice with a horizontal flip —
    # what a real detector at the direct position would record after the sample
    # rotated another 180°.  θ_tile spans [0°, 360°) accordingly.
    theta_tile_deg = np.concatenate([theta_data_deg,
                                     theta_data_deg + 180.0]).astype(np.float32)

    # Rank -> tile-group -> theta-span decomposition.
    #
    # A tile is enormous (348 GB at UPS=2, 696 GB at UPS=4) and there are only
    # n_z*n_x of them -- 6 at UPS=2.  Sharding one tile per rank therefore left
    # 90 of 96 ranks idle AND asked each worker to hold direct + mirror + tile
    # (~700 GB) in host RAM at once, which is more than a whole Polaris node.
    #
    # Split COMM into a group per tile instead.  Each member owns a contiguous
    # theta span of the source, streams it in blocks, and writes its own bank;
    # rank 0 of the group publishes the VDS master.  Every rank does I/O, and
    # peak host memory per rank is ~2*--block-gb regardless of UPS.
    n_tiles  = len(placements)
    n_groups = min(SIZE, n_tiles)
    color    = (RANK * n_groups) // SIZE      # contiguous ranks share a group
    sub      = COMM.Split(color=color, key=RANK) if COMM is not None else None
    sub_rank = sub.Get_rank() if sub is not None else 0
    sub_size = sub.Get_size() if sub is not None else 1
    my_tiles = placements[color::n_groups]    # groups round-robin over tiles

    if RANK == 0:
        describe_input(src_h5)
        max_tile = NTHETA_TILE * DET_H * DET_W * 4
        print(f"  OUT: {dst_dir}/{{z}}_{{x}}.h5   (per-tile, plain HDF5)")
        print(f"       max shape=({NTHETA_TILE}, {DET_H}, {DET_W}) float32 "
              f"({max_tile/1e9:.2f} GB/tile), edge tiles cropped smaller")
        print(f"       θ_tile spans 360° in {NTHETA_TILE} angles "
              f"(first {N_HALF} from data.h5 direct, second {N_HALF} from "
              f"mirror crop + h-flip)")
        print(f"       {n_z} z-tiles × {n_x} x-tiles = {n_z*n_x} files   "
              f"MPI ranks={SIZE}")
        print(f"       writers: {n_groups} group(s) of ~{SIZE//n_groups} ranks; "
              f"each writes one bank, group rank 0 publishes the VDS master; "
              f"θ-block budget={args.block_gb} GB/buffer", flush=True)

    t_read = t_write = 0.0
    b_read = b_write = 0

    with h5py.File(src_h5, "r") as fsrc:
        data_dset = fsrc["exchange/data"]

        for p in my_tiles:
            zi, xi = p["zi"], p["xi"]
            r_lo_d, r_hi_d = p["r_lo_dir"], p["r_hi_dir"]
            c_lo_d, c_hi_d = p["c_lo_dir"], p["c_hi_dir"]
            r_lo_m, r_hi_m = p["r_lo_mir"], p["r_hi_mir"]
            c_lo_m, c_hi_m = p["c_lo_mir"], p["c_hi_mir"]
            h_out = r_hi_d - r_lo_d
            w_out = c_hi_d - c_lo_d
            # Direct + mirror crops always share the stored shape (mirror is
            # x-reflected, z is unchanged; and x-crops swap under the flip),
            # so 2·N_HALF frames all fit the same (h_out, w_out).
            assert (r_hi_m - r_lo_m, c_hi_m - c_lo_m) == (h_out, w_out), (
                "mirror crop shape must match direct")

            spans      = theta_spans(N_HALF, sub_size)
            s0, s1     = spans[sub_rank]
            span       = s1 - s0
            out_path   = os.path.join(dst_dir, f"{zi}_{xi}.h5")
            bank_dir   = os.path.join(dst_dir, f"{zi}_{xi}")

            if sub_rank == 0:
                os.makedirs(bank_dir, exist_ok=True)
                print(f"  [grp {color}] {out_path}  "
                      f"dir rows[{r_lo_d}:{r_hi_d}] cols[{c_lo_d}:{c_hi_d}]  "
                      f"mir cols[{c_lo_m}:{c_hi_m}](h-flipped)  "
                      f"shape=({NTHETA_TILE},{h_out},{w_out})  "
                      f"{sum(1 for a, b in spans if b > a)} bank(s)",
                      flush=True)
            if sub is not None:
                sub.Barrier()          # bank dir must exist before anyone writes

            # A bank holds its span's direct half then its mirror half, so the
            # theta-block loop writes both from one read.  Block depth comes
            # from the byte budget; peak host memory is ~2x that.
            if span > 0:
                frame_b = h_out * w_out * 4
                dt = max(1, min(span, int(args.block_gb * 1e9) // frame_b))
                with h5py.File(os.path.join(dst_dir, bank_rel(zi, xi, sub_rank)), "w",
                               libver="latest") as fb:
                    gb = fb.create_group("exchange")
                    db = gb.create_dataset("data",
                                           shape=(2 * span, h_out, w_out),
                                           dtype=np.float32,
                                           chunks=(1, h_out, w_out),
                                           fillvalue=None)
                    for t0 in range(s0, s1, dt):
                        t1 = min(t0 + dt, s1)
                        tt = time.perf_counter()
                        direct = data_dset[t0:t1, r_lo_d:r_hi_d, c_lo_d:c_hi_d]
                        mirror = data_dset[t0:t1,
                                           r_lo_m:r_hi_m,
                                           c_lo_m:c_hi_m][:, :, ::-1]
                        t_read += time.perf_counter() - tt
                        b_read += direct.nbytes + mirror.nbytes

                        o0, o1 = t0 - s0, t1 - s0
                        tt = time.perf_counter()
                        db[o0:o1]                = direct
                        db[span + o0:span + o1]  = mirror   # already h-flipped
                        t_write += time.perf_counter() - tt
                        b_write += direct.nbytes + mirror.nbytes
                        del direct, mirror

            if sub is not None:
                sub.Barrier()          # every bank closed before the master

            if sub_rank == 0:
                # VDS master.  Each bank contributes two mappings: its direct
                # half lands at [a:b), its mirror half at N_HALF + [a:b).
                layout = h5py.VirtualLayout(shape=(NTHETA_TILE, h_out, w_out),
                                            dtype=np.float32)
                for r, (a, b) in enumerate(spans):
                    ln = b - a
                    if ln <= 0:
                        continue
                    vs = h5py.VirtualSource(bank_rel(zi, xi, r), "/exchange/data",
                                            shape=(2 * ln, h_out, w_out))
                    layout[a:b]                       = vs[0:ln]
                    layout[N_HALF + a:N_HALF + b]     = vs[ln:2 * ln]

                with h5py.File(out_path, "w", libver="latest") as fout:
                    g = fout.create_group("exchange")
                    d = g.create_virtual_dataset("data", layout, fillvalue=-5)
                    g.create_dataset("theta", data=theta_tile_deg)
                    # Synthetic flat/dark fields for downstream reconstruction:
                    # data_white = ones (perfect flat), data_dark = zeros.
                    g.create_dataset(
                        "data_white",
                        data=np.ones((1, h_out, w_out), dtype=np.float32))
                    g.create_dataset(
                        "data_dark",
                        data=np.zeros((1, h_out, w_out), dtype=np.float32))
                    d.attrs["z_tile"]        = zi
                    d.attrs["x_tile"]        = xi
                    d.attrs["z_center"]      = p["z_center"]
                    d.attrs["x_center"]      = p["x_center"]
                    d.attrs["crop_top"]      = p["crop_top"]
                    d.attrs["crop_bottom"]   = p["crop_bottom"]
                    d.attrs["crop_left"]     = p["crop_left"]
                    d.attrs["crop_right"]    = p["crop_right"]
                    d.attrs["full_det_h"]    = DET_H
                    d.attrs["full_det_w"]    = DET_W
                    # Back-compat: top-left corner of stored (direct) tile.
                    d.attrs["z_start"]       = r_lo_d
                    d.attrs["x_start"]       = c_lo_d
                    d.attrs["det_h"]         = h_out
                    d.attrs["det_w"]         = w_out

    if sub is not None:
        sub.Free()

    _barrier()
    report_stage("step4 read (data)",   b_read,  t_read)
    report_stage("step4 write (tiles)", b_write, t_write)
    rprint(f"done. wrote {n_z*n_x} h5 files.")

    rm_src(src_h5, enabled=args.rm_src,
           rank=RANK, comm=COMM)


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
