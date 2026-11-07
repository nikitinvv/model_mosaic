"""step4_extract's tile decomposition, against a naive whole-tile reference.

step4 used to read a whole tile -- direct slab, mirror slab and the joined
360 deg volume, ~700 GB at UPS=2 -- into one rank's host memory, with one
tile per rank.  It now splits COMM into a group per tile, gives each member a
contiguous theta span written as its own bank, and stitches the banks into a
VDS master.  That turns a single obvious slicing expression into four
interacting index spaces (group -> tile, rank -> theta span, span -> blocks,
bank -> the direct and mirror halves of the master's theta axis), and a
mistake in any of them yields a tile file of the right shape full of
plausible numbers.

So: build a toy data.h5 whose every voxel encodes its own (theta, z, x), run
the real main(), and compare each tile to the crop-and-join written out
longhand.  Exercises uncropped and cropped tiles, and the mirror h-flip.

Run with more ranks than tiles -- that is what puts several ranks in a
group and so actually exercises the bank/VDS split:

Run:  mpirun -n 16 python tests/test_step4_extract.py
      python tests/test_step4_extract.py             (serial, SIZE=1)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpi4py import MPI                                          # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

import step4_extract                                            # noqa: E402

UPS     = 9999                 # toy layout, never collides with a real one
DET_H   = DET_W = 8
NZ      = N = 16
N_HALF  = 6
NTHETA  = 2 * N_HALF
OVERLAP = 2

# zi xi z_center x_center crop_top crop_bottom crop_left crop_right
# Centres are chosen so both the direct and the mirrored (x -> N - x)
# footprints land inside [0, NZ) x [0, N).  Tile (0,0) carries crops so the
# cropped path and the shape-match assertion are both exercised.
ROWS = [
    (0, 0,  4,  6,  1, 2, 1, 3),
    (0, 1,  4, 10,  0, 0, 0, 0),
    (1, 0, 12,  6,  0, 0, 0, 0),
    (1, 1, 12, 10,  0, 0, 0, 0),
]


def reference_volume():
    """Every element encodes its own (theta, z, x), so a misrouted block
    cannot coincidentally compare equal."""
    t = np.arange(N_HALF, dtype=np.float64)[:, None, None]
    z = np.arange(NZ,     dtype=np.float64)[None, :, None]
    x = np.arange(N,      dtype=np.float64)[None, None, :]
    return (t * 1e4 + z * 1e2 + x).astype(np.float32)


def reference_tile(vol, p, theta_deg):
    """The old whole-tile code path, written out longhand."""
    direct = vol[:, p["r_lo_dir"]:p["r_hi_dir"], p["c_lo_dir"]:p["c_hi_dir"]]
    mirror = vol[:, p["r_lo_mir"]:p["r_hi_mir"],
                 p["c_lo_mir"]:p["c_hi_mir"]][:, :, ::-1]
    tile = np.empty((NTHETA,) + direct.shape[1:], dtype=np.float32)
    tile[:N_HALF] = direct
    tile[N_HALF:] = mirror
    return tile


def build_inputs(base):
    """data.h5 + a positions file matching the toy layout."""
    os.makedirs(f"{base}/model_big{UPS}x", exist_ok=True)
    os.makedirs(f"{base}/positions", exist_ok=True)
    theta = np.linspace(0.0, 180.0, N_HALF, endpoint=False).astype(np.float32)
    with h5py.File(f"{base}/model_big{UPS}x/data.h5", "w") as f:
        g = f.create_group("exchange")
        g.create_dataset("data", data=reference_volume())
        g.create_dataset("theta", data=theta)
    with open(f"{base}/positions/mosaic_positions{UPS}.txt", "w") as f:
        f.write(f"# NZ={NZ} N={N} DET_H={DET_H} DET_W={DET_W} "
                f"n_z=2 n_x=2 NTHETA={NTHETA} N_HALF={N_HALF} "
                f"OVERLAP={OVERLAP}\n")
        for r in ROWS:
            f.write(" ".join(str(v) for v in r) + "\n")
    return theta


def main() -> int:
    base = tempfile.mkdtemp(prefix="step4_test_") if RANK == 0 else None
    base = COMM.bcast(base, root=0)
    if RANK == 0:
        build_inputs(base)
    COMM.Barrier()

    # Point the module at the toy positions file and drive its real main().
    step4_extract._POSITIONS_DIR = f"{base}/positions"
    argv = ["step4_extract.py", "--ups", str(UPS), "--path", base,
            "--block-gb", "1e-6"]          # forces many tiny theta-blocks
    old_argv, sys.argv = sys.argv, argv
    try:
        step4_extract.main()
    finally:
        sys.argv = old_argv
    COMM.Barrier()

    rc = 0
    if RANK == 0:
        vol = reference_volume()
        theta = np.linspace(0.0, 180.0, N_HALF, endpoint=False).astype(np.float32)
        theta_tile = np.concatenate([theta, theta + 180.0]).astype(np.float32)
        _, placements = step4_extract.read_placements(
            f"{base}/positions/mosaic_positions{UPS}.txt")
        # Group sizes are ragged whenever n_groups does not divide SIZE
        # (SIZE=5, 4 tiles -> 2,1,1,1), so count each group rather than
        # assuming SIZE // n_groups.
        n_groups = min(SIZE, len(placements))
        grp_size = [sum(1 for r in range(SIZE)
                        if (r * n_groups) // SIZE == g)
                    for g in range(n_groups)]
        for ip, p in enumerate(placements):
            path = f"{base}/mosaic_h5/{p['zi']}_{p['xi']}.h5"
            want = reference_tile(vol, p, theta)
            with h5py.File(path, "r") as f:
                is_virtual = f["exchange/data"].is_virtual
                n_src = len(f["exchange/data"].virtual_sources()) if is_virtual else 0
                got   = f["exchange/data"][...]
                gth   = f["exchange/theta"][...]
                white = f["exchange/data_white"][...]
                dark  = f["exchange/data_dark"][...]
                attrs = dict(f["exchange/data"].attrs)
            n_banks = min(grp_size[ip % n_groups], N_HALF)
            checks = [
                ("virtual", is_virtual),
                # two mappings per bank: the direct half and the mirror half
                ("vds-srcs", n_src == 2 * n_banks),
                ("shape",  got.shape == want.shape),
                ("values", np.array_equal(got, want)),
                ("theta",  np.array_equal(gth, theta_tile)),
                ("white",  np.array_equal(white,
                                          np.ones((1,) + want.shape[1:], np.float32))),
                ("dark",   np.array_equal(dark,
                                          np.zeros((1,) + want.shape[1:], np.float32))),
                ("attrs",  attrs.get("det_h") == want.shape[1]
                           and attrs.get("det_w") == want.shape[2]
                           and attrs.get("z_start") == p["r_lo_dir"]
                           and attrs.get("x_start") == p["c_lo_dir"]),
            ]
            bad = [name for name, ok in checks if not ok]
            tag = "ok  " if not bad else "FAIL"
            print(f"  [{tag}] tile {p['zi']}_{p['xi']} shape={got.shape}"
                  + (f"   failed: {', '.join(bad)}" if bad else ""))
            if bad:
                rc = 1
        # Report the decomposition actually used: with SIZE <= n_tiles every
        # group is one rank and one bank, so a green run at 4 ranks says
        # nothing about the multi-bank stitch.
        banks = [min(g, N_HALF) for g in grp_size]
        empty = sum(max(0, g - N_HALF) for g in grp_size)
        print(f"{'PASS' if rc == 0 else 'FAIL'}: step4 extract "
              f"({n_groups} group(s) sized {grp_size} -> {banks} bank(s)/tile"
              + (f", {empty} empty span(s)" if empty else "")
              + f", {SIZE} rank(s), {len(placements)} tiles)")
        shutil.rmtree(base, ignore_errors=True)

    return COMM.bcast(rc, root=0)


if __name__ == "__main__":
    sys.exit(main())
