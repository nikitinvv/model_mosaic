#!/usr/bin/env python
"""Visualise each pipeline step's output as its own PNG in --out-dir.

Files written (one per step, skipped if source is absent).  Step
numbers match tomo_pipeline_run.sh (step1 = upsample → init.h5 is its input;
step2..8 are the intermediate outputs):

    step1_init.png       init.h5             xy@z + xz@y  (2 panels)
    step2_proj.png       proj.h5[θ]          1 panel
    step3_data.png       data.h5[θ]          1 panel
    step4_tile.png       mosaic_h5/…[θ]      1 panel
    step5_corrected.png  mosaic_h5_pre/…[θ]  1 panel
    step6_stitched.png   stitched.h5[θ]      1 panel
    step7_paganin.png    paganin.h5[θ]       1 panel
    step8_rec.png        rec.h5              xy@z + xz@y  (2 panels)

xy@z is a horizontal z-slice (arr[z, :, :]).  xz@y is a vertical
y-slice (arr[:, y, :]) — the sample side-view showing all NZ z-levels.

Location: mosaic_modeling/drawings/visualize_pipeline.py.  Default
output is `mosaic_modeling/drawings/pipeline_viz/` (next to this
script); override with --out-dir.

Usage (from any cwd):
    python mosaic_modeling/drawings/visualize_pipeline.py --ups 1 \\
        --path /local/tomodata2/brain_sym_mosaic
    python mosaic_modeling/drawings/visualize_pipeline.py --ups 8 \\
        --theta 100 --tile 2 0 --out-dir /tmp/pipe8
    python mosaic_modeling/drawings/visualize_pipeline.py --ups 1 --step rec
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


ALL_STEPS = ("init", "proj", "data", "tile", "corrected",
             "stitched", "paganin", "rec")

# One-to-one with ALL_STEPS.  Numbers match tomo_pipeline_run.sh step numbering:
#   step1 = upsample (init.h5 is its input)
#   step2 = Radon    (proj.h5)
#   step3 = Fresnel  (data.h5)
#   step4 = extract  (mosaic_h5/*.h5)
#   step5 = correct  (mosaic_h5_pre/*.h5)
#   step6 = stitch   (stitched.h5)
#   step7 = Paganin  (paganin.h5)
#   step8 = FBP      (rec.h5)
STEP_INDEX = {"init": 1, "proj": 2, "data": 3, "tile": 4, "corrected": 5,
              "stitched": 6, "paganin": 7, "rec": 8}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",   type=int, default=1)
    p.add_argument("--path",  default="/data2/brain_sym_mosaic")
    p.add_argument("--theta", type=int, default=0,
                   help="θ index into proj / data / stitched / paganin (default 0)")
    p.add_argument("--tile",  type=int, nargs=2, default=[0, 3],
                   metavar=("ZI", "XI"),
                   help="(zi, xi) of the tile to display (default 0 3)")
    p.add_argument("--z-slice", type=int, default=None,
                   help="init.h5 + rec.h5 horizontal (xy) slice; default mid-z")
    p.add_argument("--y-slice", type=int, default=None,
                   help="init.h5 + rec.h5 vertical (xz) slice; default mid-y")
    p.add_argument("--step",  choices=ALL_STEPS, default=None,
                   help="render just one step (default: all present ones)")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: <this script>/pipeline_viz "
                        "— i.e. mosaic_modeling/drawings/pipeline_viz)")
    p.add_argument("--dpi",   type=int, default=110)
    p.add_argument("--clip",  type=float, nargs=2, default=[1.0, 99.0],
                   help="percentile clip for grayscale panels (default 1 99)")
    return p.parse_args()


def _peek(path: str, key: str = "exchange/data"):
    with h5py.File(path, "r") as f:
        return f[key].shape, f[key].dtype, bool(f[key].is_virtual)


def _slice(path: str, sel, key: str = "exchange/data"):
    with h5py.File(path, "r") as f:
        return f[key][sel]


def _slice_xz(path: str, y: int, key: str = "exchange/data") -> np.ndarray:
    """Read `dset[:, y, :]` — the vertical y-slice.

    Fast path: `dset[:, y, :]` via the VDS master.  If that fails
    (h5py "insufficient elements in destination selection", a
    long-standing VDS bug when the master's virtual_sources shape
    disagrees with the actual bank shapes), fall back to reading the
    bank files directly by globbing `{stem}/{stem}_data_*.h5` and
    concatenating one bank per z-slice in name order.  Under
    `dxchange_hdf5_chunks` the banks are written sequentially over z."""
    with h5py.File(path, "r") as f:
        d = f[key]
        NZ, _, N = d.shape
        dtype = d.dtype
        try:
            out = np.empty((NZ, N), dtype=dtype)
            out[:] = d[:, y, :]
            return out
        except OSError:
            pass                          # broken VDS; fall through

    import glob
    base = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]      # e.g. "rec"
    banks = sorted(glob.glob(f"{base}/{stem}/{stem}_data_*.h5"))
    if not banks:
        raise RuntimeError(
            f"VDS master read failed and no bank files at "
            f"{base}/{stem}/{stem}_data_*.h5")
    out = np.empty((NZ, N), dtype=dtype)
    z = 0
    for bpath in banks:
        with h5py.File(bpath, "r") as fb:
            b = fb[key]
            b_nz = b.shape[0]
            if z + b_nz > NZ:
                raise RuntimeError(
                    f"bank {bpath} would overflow — z={z} + b_nz={b_nz} "
                    f"> NZ={NZ}.  Bank ordering assumed sequential; if "
                    f"this fires the layout must be reconstructed from "
                    f"the VDS mapping instead.")
            out[z:z + b_nz] = b[:, y, :]
            z += b_nz
    if z != NZ:
        raise RuntimeError(
            f"read {z} of {NZ} z-rows from banks — banks under "
            f"{base}/{stem}/ don't cover the full volume")
    return out


def _pct(*arrs: np.ndarray, lo=1.0, hi=99.0) -> tuple[float, float]:
    cat = np.concatenate([a.ravel() for a in arrs if a is not None])
    return tuple(np.percentile(cat, [lo, hi]))


def _figure(title: str, dpi: int, out: str,
            panels: list[tuple[np.ndarray, str, tuple[float, float] | None,
                               str, str, str]]) -> None:
    """Save a figure with `panels = [(img, title, vrange, cmap, xlab, ylab)]`."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 8),
                             gridspec_kw={"wspace": 0.18})
    if n == 1:
        axes = [axes]
    for ax, (img, subtitle, vrange, cmap, xlab, ylab) in zip(axes, panels):
        vmin, vmax = (None, None) if vrange is None else vrange
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _stats(label: str, arr: np.ndarray) -> None:
    print(f"  {label:12s}: shape={arr.shape}  "
          f"min={arr.min():.4g}  max={arr.max():.4g}  mean={arr.mean():.4g}")


def main() -> None:
    args = _parse_args()

    p = args.path.rstrip("/")
    srcs = {
        "init":     f"{p}/init.h5",
        "proj":     f"{p}/model_big{args.ups}x/proj.h5",
        "data":     f"{p}/model_big{args.ups}x/data.h5",
        "stitched": f"{p}/model_big{args.ups}x/stitched.h5",
        "paganin":  f"{p}/model_big{args.ups}x/paganin.h5",
        "rec":      f"{p}/model_big{args.ups}x/rec.h5",
        "tile":     f"{p}/mosaic_h5/{args.tile[0]}_{args.tile[1]}.h5",
        "corrected": f"{p}/mosaic_h5_pre/{args.tile[0]}_{args.tile[1]}.h5",
    }
    present = {k: os.path.exists(v) for k, v in srcs.items()}
    for k, v in srcs.items():
        if not present[k]:
            print(f"note: {v} not found — step{STEP_INDEX[k]}_{k}.png "
                  f"will be skipped")

    def _png(name: str) -> str:
        return os.path.join(args.out_dir, f"step{STEP_INDEX[name]}_{name}.png")

    todo = [args.step] if args.step else list(ALL_STEPS)

    # Resolve z_slice / y_slice from init (or rec if init absent).
    ref_shape = None
    if present["init"]:
        ref_shape, _, _ = _peek(srcs["init"])
    elif present["rec"]:
        ref_shape, _, _ = _peek(srcs["rec"])
    if ref_shape is None and (args.z_slice is None or args.y_slice is None):
        raise SystemExit("no init.h5 or rec.h5 to derive default z/y — "
                         "pass --z-slice and --y-slice explicitly")
    z_slice = args.z_slice if args.z_slice is not None else ref_shape[0] // 2
    y_slice = args.y_slice if args.y_slice is not None else ref_shape[1] // 2

    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "pipeline_viz")
    os.makedirs(args.out_dir, exist_ok=True)
    lo, hi = args.clip
    ups_tag = f"UPS={args.ups}"

    # ---- init ---------------------------------------------------------
    if "init" in todo and present["init"]:
        print(f"[init]  reading z={z_slice}, y={y_slice} slices ...")
        init_xy = _slice(srcs["init"], np.s_[z_slice, :, :])
        init_xz = _slice(srcs["init"], np.s_[:, y_slice, :])
        r = _pct(init_xy, init_xz, lo=lo, hi=hi)
        _figure(f"init.h5  ({ups_tag},  z_slice={z_slice}, y_slice={y_slice})",
                args.dpi, _png("init"),
                panels=[
                    (init_xy,
                     f"init.h5[z={z_slice}, :, :]  xy  shape={init_xy.shape}",
                     r, "gray", "x [pixel]", "y [pixel]"),
                    (init_xz,
                     f"init.h5[:, y={y_slice}, :]  xz  shape={init_xz.shape}",
                     r, "gray", "x [pixel]", "z [pixel]"),
                ])
        _stats("init xy", init_xy)
        _stats("init xz", init_xz)

    # ---- proj / data / stitched / paganin (single θ-slice each) ------
    for name, subtitle_prefix in [
            ("proj",     "Radon of δ"),
            ("data",     "Fresnel intensity"),
            ("stitched", "180° big proj"),
            ("paganin",  "φ (single-distance Paganin)"),
    ]:
        if name not in todo or not present[name]:
            continue
        print(f"[{name}]  reading θ={args.theta} ...")
        img = _slice(srcs[name], np.s_[args.theta, :, :])
        r = tuple(np.percentile(img, [lo, hi]))
        _figure(f"{name}.h5  ({ups_tag},  θ_idx={args.theta})",
                args.dpi, _png(name),
                panels=[(img,
                         f"{name}.h5[{args.theta}]  {subtitle_prefix}  "
                         f"shape={img.shape}",
                         r, "gray", "x [pixel]", "z [pixel]")])
        _stats(name, img)

    # ---- tile / corrected --------------------------------------------
    for name, dir_label in [("tile", "mosaic_h5"),
                            ("corrected", "mosaic_h5_pre")]:
        if name not in todo or not present[name]:
            continue
        print(f"[{name}]  reading {args.tile} θ={args.theta} ...")
        with h5py.File(srcs[name], "r") as f:
            tile_shape = f["exchange/data"].shape
            tile_img   = f["exchange/data"][args.theta, :, :]
            tile_attrs = {k: v.tolist() if hasattr(v, "tolist") else v
                          for k, v in f["exchange/data"].attrs.items()}
        zi, xi = args.tile
        z_start = tile_attrs.get("z_start", "?")
        x_start = tile_attrs.get("x_start", "?")
        r = tuple(np.percentile(tile_img, [lo, hi]))
        _figure(f"{dir_label}/{zi}_{xi}.h5  ({ups_tag},  θ_idx={args.theta})",
                args.dpi, _png(name),
                panels=[(tile_img,
                         f"{dir_label}/{zi}_{xi}.h5[{args.theta}]  "
                         f"z_start={z_start} x_start={x_start}  "
                         f"shape={tile_shape}",
                         r, "gray", "x [pixel]", "z [pixel]")])
        _stats(name, tile_img)

    # ---- rec ----------------------------------------------------------
    if "rec" in todo and present["rec"]:
        print(f"[rec]  reading z={z_slice} then xz z-by-z (VDS workaround) ...")
        rec_xy = _slice(srcs["rec"], np.s_[z_slice, :, :])
        rec_xz = _slice_xz(srcs["rec"], y_slice)
        r = _pct(rec_xy, rec_xz, lo=lo, hi=hi)
        _figure(f"rec.h5  ({ups_tag},  z_slice={z_slice}, y_slice={y_slice})",
                args.dpi, _png("rec"),
                panels=[
                    (rec_xy,
                     f"rec.h5[z={z_slice}, :, :]  FBP xy  shape={rec_xy.shape}",
                     r, "gray", "x [pixel]", "y [pixel]"),
                    (rec_xz,
                     f"rec.h5[:, y={y_slice}, :]  FBP xz  shape={rec_xz.shape}",
                     r, "gray", "x [pixel]", "z [pixel]"),
                ])
        _stats("rec xy", rec_xy)
        _stats("rec xz", rec_xz)


if __name__ == "__main__":
    main()
