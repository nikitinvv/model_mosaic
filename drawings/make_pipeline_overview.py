#!/usr/bin/env python
"""Compose a horizontal single-figure overview of the 8-step pipeline.

One column per step, laid out left-to-right with arrows between them:

    ┌ step 1 ┐  →  ┌ step 2 ┐  →  …  →  ┌ step 8 ┐

Each column stacks vertically:
    · step number + name (title bar)
    · preview slice (from drawings/pipeline_viz/)
    · script filename
    · operation description
    · reads:  input dataset  (name / shape / size / vchunk / chunk)
    · writes: output dataset (name / shape / size / vchunk / chunk)

vchunk = super-chunk = RAM buffer worked on by one MPI rank per iteration.
chunk  = HDF5 on-disk chunk (read/write granularity + compression unit).

Both come from `iohdf5.layout.plan_pipeline` — the same policy the steps
themselves call — so this picture always shows the layout a run will
actually use, for whatever --ups / --mem-budget / --chunk-bytes you give
it.  A dataset whose smallest legal super-chunk busts the RAM budget is
flagged in red (that is step1's `big` at UPS≥16).

Uses the per-step slice renders written by `visualize_pipeline.py` under
`drawings/pipeline_viz/`.

Output: drawings/pipeline_overview.png (override with --out).
Nominal geometry is UPS=1; pass --ups K to relabel shape/size text.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iohdf5.layout import (DEFAULT_CHUNK_BYTES, DEFAULT_MEM_BUDGET,  # noqa: E402
                           plan_pipeline)


_HERE = os.path.dirname(os.path.abspath(__file__))
_VIZ_DIR = os.path.join(_HERE, "pipeline_viz")


def _fmt_gb(nbytes: int) -> str:
    """Match the pipeline logs — base-2 units labelled 'GB' (i.e. GiB).
    Goes up to PB: rec.h5 is 3.4 PB at UPS=32."""
    KB = 1024
    for lim, unit in ((KB ** 5, "PB"), (KB ** 4, "TB"), (KB ** 3, "GB")):
        if nbytes >= lim:
            return f"{nbytes / lim:.1f} {unit}"
    return f"{nbytes / KB**2:.0f} MB"


def _sh(t):
    return "(" + ", ".join(str(x) for x in t) + ")"


def _steps(ups: int, nbanks: int, nranks: int, budget: int, chunk_bytes: int):
    """8 pipeline steps annotated for a given upsample factor.

    Every shape, vchunk and chunk below is read straight off
    `plan_pipeline`, so nothing here can drift from what the steps do."""
    plans    = plan_pipeline(ups, nbanks=nbanks, nranks=nranks,
                             budget=budget, chunk_bytes=chunk_bytes)
    N        = 3072 * ups
    NTHETA   = 3 * N // 4
    TILE_H   = 303 * ups
    TILE_W   = 404 * ups
    NT_TILE  = 2 * NTHETA
    N_TILES  = 44

    tiles  = N_TILES * NT_TILE * TILE_H * TILE_W * 4

    def block(name, shape, size, vc, ch, note="", buf="", warn=False):
        return dict(name=name,
                    shape=_sh(shape) + " f32",
                    size=size,
                    vc=("vchunk " + _sh(vc)) if vc else "(no vchunks)",
                    buf=buf,
                    ch="chunk  " + _sh(ch),
                    note=note, warn=warn)

    def pblock(key, name):
        """Card for a planned dataset — shape/size/vchunk/chunk verbatim
        from the policy, plus the two numbers that decide whether it runs:
        the per-rank buffer and the on-disk chunk size.

        `buf` is the vchunk AND the step's input-prefetch slab together,
        which is the figure that has to fit --mem-budget."""
        p = plans[key]
        return dict(
            name=name,
            shape=_sh(p.shape) + " f32",
            size=_fmt_gb(p.total_bytes),
            vc=f"vchunk {_sh(p.vchunks)}",
            buf=(f"  buf {_fmt_gb(p.buffer_bytes)}  ⚠ > budget"
                 if p.over_budget else
                 f"  buf {_fmt_gb(p.buffer_bytes)} × {p.n_vchunks} sc"),
            ch=f"chunk  {_sh(p.chunks)}  {_fmt_gb(p.chunk_bytes)}",
            note=(f"stype={p.stype}, nbanks={p.nbanks}, "
                  f"{p.effective_order}-ordered"),
            warn=p.over_budget)

    init      = pblock("init", "init.h5")
    big       = pblock("big", f"big{ups}x.h5")
    proj      = pblock("proj", "proj.h5")
    data      = pblock("data", "data.h5")
    stitched  = pblock("stitched", "stitched.h5")
    paganin   = pblock("paganin", "paganin.h5")
    rec       = pblock("rec", "rec.h5")
    tile_out  = block("mosaic_h5/*.h5",
                      (NT_TILE, TILE_H, TILE_W),
                      f"≈{_fmt_gb(tiles)} (44 tiles)",
                      None, (1, TILE_H, TILE_W), "per-tile plain HDF5")
    tile_pre  = block("mosaic_h5_pre/*.h5",
                      (NT_TILE, TILE_H, TILE_W),
                      f"≈{_fmt_gb(tiles)} (44 tiles)",
                      None, (1, TILE_H, TILE_W), "per-tile plain HDF5")

    return [
        dict(num=1, name="upsample", script="step1_upsample.py",
             img="step1_init.png",
             op="trilinear upsample\n(bilinear xy + linear z)",
             inp=init, out=big),
        dict(num=2, name="Radon",
             script="step2_radon.py", img="step2_proj.png",
             op=f"USFFT Radon over\n{NTHETA} angles (180°)",
             inp=big, out=proj),
        dict(num=3, name="Fresnel propagation",
             script="step3_propagation.py", img="step3_data.png",
             op="D(ψ) = |propagate to\ndetector|² (near-field)",
             inp=proj, out=data),
        dict(num=4, name="extract tiles",
             script="step4_extract.py", img="step4_tile.png",
             op="crop 11×4=44 tiles;\nmirror-fold to 360°",
             inp=data, out=tile_out),
        dict(num=5, name="correct tiles",
             script="step5_correct.py", img="step5_corrected.png",
             op="dezinger + dark/flat +\nFourier-wavelet rings",
             inp=tile_out, out=tile_pre),
        dict(num=6, name="stitch tiles",
             script="step6_stitch.py", img="step6_stitched.png",
             op="tent-blend across x;\nfold 360°→180°",
             inp=tile_pre, out=stitched),
        dict(num=7, name="Paganin",
             script="step7_paganin.py", img="step7_paganin.png",
             op="single-distance Paganin\nper angle (2-D FFT · K · IFFT)",
             inp=stitched, out=paganin),
        dict(num=8, name="FBP reconstruction",
             script="step8_fbp.py", img="step8_rec.png",
             op=f"ramp filter along x +\nadjoint Radon RT",
             inp=paganin, out=rec),
    ]


_BADGE_BG   = "#2c3e50"
_TEXT_MAIN  = "#111111"
_TEXT_MUTE  = "#555555"
_ARROW_C    = "#2c3e50"
_INP_TAG    = "#7a5a1a"    # amber
_OUT_TAG    = "#1c5a7a"    # teal
_WARN_C     = "#a01010"    # over-RAM-budget note


def _draw_io_block(ax, y_top, tag_label, tag_col, blk):
    """Render one input/output card top-anchored at y_top (axis fraction).
    Returns the y-coordinate just below the block."""
    lh = 0.125
    ax.text(0.03, y_top, tag_label, fontsize=18, weight="bold",
            color=tag_col, va="top", ha="left")
    ax.text(0.36, y_top, blk["name"], fontsize=18, weight="bold",
            color=_TEXT_MAIN, family="monospace", va="top", ha="left")
    y = y_top - lh
    ax.text(0.03, y, blk["shape"], fontsize=16, family="monospace",
            color=_TEXT_MAIN, va="top", ha="left")
    y -= lh
    ax.text(0.03, y, blk["size"], fontsize=16, family="monospace",
            weight="bold", color=_TEXT_MUTE, va="top", ha="left")
    y -= lh
    ax.text(0.03, y, blk["vc"], fontsize=16, family="monospace",
            color=_TEXT_MAIN, va="top", ha="left")
    if blk.get("buf"):
        y -= lh
        ax.text(0.03, y, blk["buf"], fontsize=15, family="monospace",
                color=_WARN_C if blk.get("warn") else _TEXT_MUTE,
                weight="bold" if blk.get("warn") else "normal",
                va="top", ha="left")
    y -= lh
    ax.text(0.03, y, blk["ch"], fontsize=16, family="monospace",
            color=_TEXT_MAIN, va="top", ha="left")
    if blk["note"]:
        y -= lh
        ax.text(0.03, y, blk["note"], fontsize=14, style="italic",
                color=_WARN_C if blk.get("warn") else _TEXT_MUTE,
                weight="bold" if blk.get("warn") else "normal",
                va="top", ha="left")
    return y - lh * 0.3


def _draw_step(fig, gs_col, step: dict, viz_dir: str):
    """Fill one step column: [header | image | script+op | reads | writes]."""
    sub = gs_col.subgridspec(
        5, 1,
        # Compact heights matched to actual content; image gets the most.
        height_ratios=[0.45, 2.20, 0.65, 1.30, 1.30],
        hspace=0.04)

    # 1. header — big badge number + step name
    ax_h = fig.add_subplot(sub[0])
    ax_h.set_axis_off()
    ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)
    ax_h.text(0.02, 0.5, str(step["num"]),
              fontsize=54, weight="bold", color=_BADGE_BG,
              va="center", ha="left")
    ax_h.text(0.22, 0.55, step["name"],
              fontsize=24, weight="bold", color=_TEXT_MAIN,
              va="center", ha="left")

    # 2. image — natural aspect preserved (wide step1/step8 get 2× columns).
    ax_img = fig.add_subplot(sub[1])
    img_path = os.path.join(viz_dir, step["img"])
    if os.path.isfile(img_path):
        ax_img.imshow(imread(img_path))
    else:
        ax_img.text(0.5, 0.5, f"(missing {step['img']})",
                    fontsize=8, color="#a00000",
                    ha="center", va="center", transform=ax_img.transAxes)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.8)

    # 3. script + operation
    ax_op = fig.add_subplot(sub[2])
    ax_op.set_axis_off()
    ax_op.set_xlim(0, 1); ax_op.set_ylim(0, 1)
    ax_op.text(0.03, 0.95, step["script"],
               fontsize=17, family="monospace", color=_TEXT_MUTE,
               va="top", ha="left")
    ax_op.text(0.03, 0.55, step["op"],
               fontsize=17, color=_TEXT_MAIN, va="top", ha="left")

    # 4. reads
    ax_in = fig.add_subplot(sub[3])
    ax_in.set_axis_off()
    ax_in.set_xlim(0, 1); ax_in.set_ylim(0, 1)
    _draw_io_block(ax_in, 0.97, "reads",  _INP_TAG, step["inp"])

    # 5. writes
    ax_out = fig.add_subplot(sub[4])
    ax_out.set_axis_off()
    ax_out.set_xlim(0, 1); ax_out.set_ylim(0, 1)
    _draw_io_block(ax_out, 0.97, "writes", _OUT_TAG, step["out"])


def build(ups: int, out_path: str, dpi: int, viz_dir: str,
          nbanks: int = 8, nranks: int = 8,
          mem_budget_gb: float = DEFAULT_MEM_BUDGET / 2 ** 30,
          chunk_mb: float = DEFAULT_CHUNK_BYTES / 2 ** 20) -> None:
    steps = _steps(ups, nbanks=nbanks, nranks=nranks,
                   budget=int(mem_budget_gb * 2 ** 30),
                   chunk_bytes=int(chunk_mb * 2 ** 20))
    n = len(steps)

    # Column widths: step 1 & step 8 render 2×-wide two-panel views
    # (xy@z + xz@y), so give them 2× column slots so aspect is preserved
    # without shrinking.  Others get 1× slots.  Arrows sit in narrow slots
    # between steps.
    STEP_W_WIDE = 2.0
    STEP_W_NORM = 1.0
    ARROW_W     = 0.14
    widths = []
    for i in range(n):
        widths.append(STEP_W_WIDE if i in (0, n - 1) else STEP_W_NORM)
        if i < n - 1:
            widths.append(ARROW_W)
    fig_w = sum(widths) * 3.6
    fig_h = 19.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    outer = GridSpec(1, len(widths), width_ratios=widths,
                     left=0.015, right=0.99, top=0.925, bottom=0.02,
                     wspace=0.10)

    fig.suptitle(f"Mosaic-modelling pipeline — 8 steps  (UPS={ups})",
                 fontsize=34, weight="bold", y=0.988, color=_TEXT_MAIN)
    fig.text(0.5, 0.955,
             f"layout from iohdf5.layout:  {mem_budget_gb:g} GiB/rank"
             f" × {nranks} ranks  ·  chunk target ~{chunk_mb:g} MiB"
             f"  ·  nbanks ≤ {nbanks}",
             fontsize=20, color=_TEXT_MUTE, ha="center", va="center")

    for i, step in enumerate(steps):
        col_idx = 2 * i
        _draw_step(fig, outer[col_idx], step, viz_dir)

        if i < n - 1:
            ax_arrow = fig.add_subplot(outer[col_idx + 1])
            ax_arrow.set_axis_off()
            ax_arrow.set_xlim(0, 1); ax_arrow.set_ylim(0, 1)
            # Center the arrow vertically at ~the image band.
            # Vertically centered on the image band (~50% of column).
            ax_arrow.add_patch(FancyArrowPatch(
                (0.05, 0.60), (0.95, 0.60),
                arrowstyle="-|>", mutation_scale=18,
                lw=2.0, color=_ARROW_C))

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}  ({fig_w:.0f} × {fig_h:.1f} in @ {dpi} dpi)")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups", type=int, default=1)
    p.add_argument("--viz-dir", default=_VIZ_DIR)
    p.add_argument("--out", default=None)
    p.add_argument("--dpi", type=int, default=140)
    # Layout knobs — same names and defaults the steps use, so the picture
    # can be regenerated for exactly the configuration a run will have.
    p.add_argument("--nbanks", type=int, default=8,
                   help="max bank files per super-chunk (the policy lowers "
                        "it when one super-chunk cannot hold that many "
                        "whole planes)")
    p.add_argument("--nranks", type=int, default=8,
                   help="total MPI ranks; caps the super-chunk so none idle")
    p.add_argument("--mem-budget", type=float,
                   default=DEFAULT_MEM_BUDGET / 2 ** 30, metavar="GiB",
                   help="per-rank RAM for the vchunk plus the input prefetch")
    p.add_argument("--chunk-bytes", type=float,
                   default=DEFAULT_CHUNK_BYTES / 2 ** 20, metavar="MiB",
                   help="target size of one HDF5 chunk")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    name = ("pipeline_overview.png" if args.ups == 1
            else f"pipeline_overview_ups{args.ups}.png")
    out = args.out or os.path.join(_HERE, name)
    build(ups=args.ups, out_path=out, dpi=args.dpi, viz_dir=args.viz_dir,
          nbanks=args.nbanks, nranks=args.nranks,
          mem_budget_gb=args.mem_budget, chunk_mb=args.chunk_bytes)
