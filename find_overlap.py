#!/usr/bin/env python
"""Measure the detector shift between two 360 deg tiles taken at different
sample positions (same angles): tile B's frame at angle theta equals tile A's
frame shifted by (dy, dx) inside the overlap, B[y, x] = A[y + dy, x + dx].

For a few angles: dark-flat (with each tile's bright ratio), bin, -log, then
  * phase correlation (default) -- good when the overlap is a large part of
    the frame (the side-by-side pairs: ~60% of the width);
  * --search DY DX RY RX: brute-force normalised cross-correlation for
    |dy-DY| <= RY, |dx-DX| <= RX on high-passed images -- for thin overlaps
    at the frame edge (the vertical pairs: ~23% of the height), where the
    phase correlation's window fades the overlap band out.
Sub-pixel by a parabola through the peak on each axis.  Writes a PNG with both
frames, the correlation, the per-angle shifts and the two frames pasted.

    python find_overlap.py A.h5 B.h5 --bright-ratio 0.6 1 --bin 4 --out ov.png
    python find_overlap.py A.h5 C.h5 --bin 4 --search 529 0 80 20 --out ov.png
"""
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

p = argparse.ArgumentParser()
p.add_argument("a")
p.add_argument("b")
p.add_argument("--bright-ratio", type=float, nargs=2, default=(1.0, 1.0))
p.add_argument("--bin", type=int, default=4)
p.add_argument("--frames", type=int, nargs="+", default=[0, 4500, 9000, 13500])
p.add_argument("--search", type=int, nargs=4, default=None,
               metavar=("DY", "DX", "RY", "RX"),
               help="brute-force NCC around (DY, DX) binned px, +-RY, +-RX")
p.add_argument("--out", required=True)
a = p.parse_args()
B = a.bin


def load(fn, br, idx):
    with h5py.File(fn, "r") as f:
        dark = f["exchange/data_dark"][:].astype(np.float32).mean(0)
        flat = f["exchange/data_white"][:].astype(np.float32).mean(0)
        fl = np.maximum((flat * (1 + 1e-5) - dark) / br, 1)
        out = []
        for i in idx:
            x = (f["exchange/data"][i].astype(np.float32) - dark) / fl
            h, w = x.shape[0] // B * B, x.shape[1] // B * B
            out.append(x[:h, :w].reshape(h // B, B, w // B, B).mean((1, 3)))
    return np.array(out)


def parabola(vm, v0, vp):
    den = vm - 2 * v0 + vp
    return 0.5 * (vm - vp) / den if den else 0.0


def phase_corr(A, Bm):
    """(dy, dx, curve) with Bm[y, x] ~ A[y + dy, x + dx]."""
    h, w = A.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    FA = np.fft.fft2((A - A.mean()) * win, s=(2 * h, 2 * w))
    FB = np.fft.fft2((Bm - Bm.mean()) * win, s=(2 * h, 2 * w))
    R = FA * np.conj(FB)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-12)).real
    ky, kx = np.unravel_index(np.argmax(r), r.shape)
    n0, n1 = r.shape
    dy = ky + parabola(r[(ky - 1) % n0, kx], r[ky, kx], r[(ky + 1) % n0, kx])
    dx = kx + parabola(r[ky, (kx - 1) % n1], r[ky, kx], r[ky, (kx + 1) % n1])
    dy = dy - n0 if dy > h else dy
    dx = dx - n1 if dx > w else dx
    rs = np.fft.fftshift(r)
    return dy, dx, (np.arange(n1) - n1 // 2, rs.max(axis=0), "dx")


def highpass(x, s=15):
    return x - uniform_filter(x, s)


def ncc_search(A, Bm, DY, DX, RY, RX):
    """Brute-force NCC over the overlap for every integer (dy, dx) in the
    window; (dy, dx, curve) with Bm[y, x] ~ A[y + dy, x + dx]."""
    h, w = A.shape
    Ah, Bh = highpass(A), highpass(Bm)
    m = 8                                   # skip the high-pass border
    dys = np.arange(DY - RY, DY + RY + 1)
    dxs = np.arange(DX - RX, DX + RX + 1)
    S = np.full((len(dys), len(dxs)), -1.0)
    for i, dy in enumerate(dys):
        for j, dx in enumerate(dxs):
            ya, yb = max(dy, 0) + m, min(h + dy, h) - m   # rows of A
            xa, xb = max(dx, 0) + m, min(w + dx, w) - m
            if yb - ya < 16 or xb - xa < 16:
                continue
            u = Ah[ya:yb, xa:xb].ravel()
            v = Bh[ya - dy:yb - dy, xa - dx:xb - dx].ravel()
            u = u - u.mean(); v = v - v.mean()
            S[i, j] = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    i, j = np.unravel_index(np.argmax(S), S.shape)
    sy = parabola(S[i - 1, j], S[i, j], S[i + 1, j]) if 0 < i < len(dys) - 1 else 0
    sx = parabola(S[i, j - 1], S[i, j], S[i, j + 1]) if 0 < j < len(dxs) - 1 else 0
    if i in (0, len(dys) - 1) or j in (0, len(dxs) - 1):
        print("  WARNING: NCC peak on the edge of the search window")
    return dys[i] + sy, dxs[j] + sx, (dys, S.max(axis=1), "dy"), S[i, j]


PA = load(a.a, a.bright_ratio[0], a.frames)
PB = load(a.b, a.bright_ratio[1], a.frames)
LA, LB = -np.log(np.clip(PA, 1e-3, None)), -np.log(np.clip(PB, 1e-3, None))
H, W = LA.shape[1:]

res = []
for k, i in enumerate(a.frames):
    if a.search:
        dy, dx, curve, peak = ncc_search(LA[k], LB[k], *a.search)
        extra = f"  NCC={peak:.3f}"
    else:
        dy, dx, curve = phase_corr(LA[k], LB[k])
        extra = ""
    iy, ix = int(round(dy)), int(round(dx))
    ya, yb = max(iy, 0), H + min(iy, 0)
    xa, xb = max(ix, 0), W + min(ix, 0)
    oa = PA[k][ya:yb, xa:xb]
    ob = PB[k][ya - iy:yb - iy, xa - ix:xb - ix]
    ratio = np.median(ob / np.maximum(oa, 1e-3))
    res.append((i, dy, dx))
    if k == 0:
        curve0 = curve
    print(f"frame {i:6d}: dy={dy:+.2f}  dx={dx:+.2f} binned px "
          f"({dy * B:+.1f}, {dx * B:+.1f} unbinned)  overlap={yb - ya} rows x "
          f"{xb - xa} cols  I_B/I_A={ratio:.4f}{extra}")
dys = np.array([r[1] for r in res]); dxs = np.array([r[2] for r in res])
print(f"mean dy={dys.mean():+.2f} (std {dys.std():.2f})  "
      f"mean dx={dxs.mean():+.2f} (std {dxs.std():.2f}) binned px")
print(f"=> unbinned dy={dys.mean() * B:+.1f}  dx={dxs.mean() * B:+.1f} px")

# Paste A and B on one canvas at the mean integer shift (frame 0).
iy, ix = int(round(dys.mean())), int(round(dxs.mean()))
oy, ox = max(-iy, 0), max(-ix, 0)           # A's offset on the canvas
Hc, Wc = H + abs(iy), W + abs(ix)
ca = np.full((Hc, Wc), np.nan, np.float32)
cb = np.full((Hc, Wc), np.nan, np.float32)
ca[oy:oy + H, ox:ox + W] = LA[0]
cb[oy + iy:oy + iy + H, ox + ix:ox + ix + W] = LB[0]   # B row y = A row y+iy
both = np.where(np.isnan(ca), cb, np.where(np.isnan(cb), ca, 0.5 * (ca + cb)))
lo, hi = np.nanpercentile(both, [1, 99])

vertical = abs(iy) > abs(ix)
fig = plt.figure(figsize=(18, 16 if vertical else 13))
g = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2.2 if vertical else 1])
ax = fig.add_subplot(g[0, 0]); ax.imshow(LA[0], cmap="gray", vmin=lo, vmax=hi)
ax.set_title(f"A {a.a.split('/')[-1]}  frame {a.frames[0]}  (bin {B}, -log)")
ax = fig.add_subplot(g[0, 1]); ax.imshow(LB[0], cmap="gray", vmin=lo, vmax=hi)
ax.set_title(f"B {a.b.split('/')[-1]}  frame {a.frames[0]}")
ax = fig.add_subplot(g[1, 0])
ax.plot(curve0[0], curve0[1])
ax.set_xlabel(f"{curve0[2]} (binned px)")
ax.set_title("NCC, max over the other axis" if a.search
             else "phase correlation, max over dy")
ax = fig.add_subplot(g[1, 1])
ax.plot([r[0] for r in res], dxs, "o-", label="dx")
ax.plot([r[0] for r in res], dys, "s-", label="dy")
ax.set_xlabel("frame"); ax.legend(); ax.set_title("shift per angle (binned px)")
ax = fig.add_subplot(g[2, :])
ax.imshow(both, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
if vertical:
    o0, o1 = max(oy, oy + iy), min(oy, oy + iy) + H
    ax.axhline(o0, color="r"); ax.axhline(o1, color="r")
    what = f"rows [{o0}, {o1}) = {o1 - o0}"
else:
    o0, o1 = max(ox, ox + ix), min(ox, ox + ix) + W
    ax.axvline(o0, color="r"); ax.axvline(o1, color="r")
    what = f"columns [{o0}, {o1}) = {o1 - o0}"
ax.set_title(f"A and B pasted at dy={iy}, dx={ix}: overlap {what} binned px "
             f"(red); average inside")
fig.tight_layout()
fig.savefig(a.out, dpi=80)
print(a.out)
