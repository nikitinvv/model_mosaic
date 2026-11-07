#!/usr/bin/env python
"""Single-distance Paganin phase retrieval on the stitched big projection,
host-chunked variant — same math as step7_paganin.py but the full 2-D
FFT never lives on the GPU.  For UPS ≥ 8 where the (NPGNCHUNK, NZ, N)
complex64 buffer no longer fits on a 40 GB device.

Reads {path}/model_big{UPS}x/stitched.h5 and writes
{path}/model_big{UPS}x/paganin.h5.  Same shape and theta grid as the
non-large variant.

For each --vchunks super-chunk (θ_super, NZ, N) this rank owns:
  1. Read the θ-slab from stitched.h5.
  2. Loop NPGNCHUNK θ-batches through PaganinLarge.retrieve() —
     3-pass streaming x-FFT / y-FFT+H·mult / x-IFFT+log+scale, one
     CHUNK_NZ or CHUNK_N strip on the GPU at a time.
  3. tomo_writex fans the vchunk buffer to disk across --nbanks writers.

Multi-GPU via MPI + set_affinity_gpu.sh.  Launch:
    mpirun -n <NGPU> set_affinity_gpu.sh python step7_paganin_large.py \\
        --ups 8 --path /data2/brain_sym_mosaic
"""
from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import cupy as cp

from processing.paganin_large import PaganinLarge
from iohdf5.dxchange_hdf5_chunks import (tomo_writex, read_projs_vchunkx,
                                         write_region_single)
from iohdf5.h5_vchunks import SINGLE_H5
from concurrent.futures import ThreadPoolExecutor
from iohdf5.h5_vchunks import (
    initx_and_bcast, open_out, close_out, alloc_shm, free_shm, iter_vchunks,
    vchunk_bytes, n_vchunks, describe_input, describe_output,
    read_src_header, rm_src, add_rm_src_arg,
)
from iohdf5.layout import add_layout_args, add_geometry_args, input_dims, resolve_step
from mpi_utils import COMM, RANK, SIZE, MPI, barrier, rprint, allreduce, report_stage


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ups",  type=int, default=8,
                   help="matches step3/step5 --ups (drives paths)")
    p.add_argument("--path", default="/data2/brain_sym_mosaic")
    p.add_argument("--energy",     type=float, default=30.0,   help="keV")
    p.add_argument("--voxelsize",  type=float, default=0.72222e-6,
                   help="voxel = detector pixel, meters (parallel beam)")
    p.add_argument("--distance",   type=float, default=0.15,
                   help="sample → detector distance, meters (matches step3)")
    p.add_argument("--alpha",      type=float, default=1e-3,
                   help="Tikhonov regularisation added to T² in the filter")
    p.add_argument("--npgnchunk",  type=int, default=1,
                   help="angles per PaganinLarge.retrieve() call.  Default 1 "
                        "at high UPS since the host fde scales with ntheta.")
    p.add_argument("--chunk-nz",   type=int, default=768,
                   help="pass1/3 x-strip depth")
    p.add_argument("--chunk-n",    type=int, default=768,
                   help="pass2 y-strip width")
    p.add_argument("--nbanks",     type=int, default=8)
    p.add_argument("--ntasks",     type=int, default=8,
                   help="parallel workers for read_projs_vchunkx (stitched prefetch)")
    p.add_argument("--vchunks", type=int, nargs=3, default=None,
                   metavar=("C0", "C1", "C2"),
                   help="super-chunk for paganin.h5; default comes from "
                        "iohdf5.layout (--mem-budget / --chunk-bytes)")
    p.add_argument("--chunk-order", choices=("sino", "proj"), default="sino",
                   help="HDF5 chunk order inside paganin.h5's bank files.  "
                        "'sino' (default) = (θ_per_bank, chunk_z, N), what "
                        "step8's z-slab FBP read wants.  'proj' = (1, NZ, N), "
                        "the old layout — makes that read touch every chunk "
                        "to use zslab/NZ of it.  Banking is θ-split either "
                        "way.")
    p.add_argument("--chunk-z", type=int, default=0,
                   help="z-extent of the sinogram chunk.  0 (default) takes "
                        "it from iohdf5.layout, which derives step8's FBP "
                        "z-slab from the same policy, so each bank file "
                        "serves that read with one whole-chunk sequential "
                        "op.  --chunk-z 1 is the pure-sinogram extreme and "
                        "costs one HDF5 op per z row per bank, which is "
                        "latency-bound on Lustre.  Ignored for 'proj'.")
    p.add_argument("--pad", choices=("none", "edge"), default="none",
                   help="'edge': as tomocupy's paganin_filter -- every "
                        "projection edge-replicated to the next power of 2 "
                        "of (size + pi*lambda*z/p^2) in both axes before the "
                        "filter, then cropped back, so the filter does not "
                        "wrap one edge onto the other.  'none' (default): "
                        "the FFT treats the frame as periodic")
    add_layout_args(p)
    add_geometry_args(p)
    add_rm_src_arg(p, "the input stitched.h5")
    return p.parse_args()


_A = _parse_args()

UPS       = _A.ups
BASE_DIR  = _A.path
DST_DIR   = f"{BASE_DIR}/model_big{UPS}x"
SRC_H5    = f"{DST_DIR}/stitched.h5"
DST_H5    = f"{DST_DIR}/paganin.h5"

IN_NZ, IN_N = input_dims(_A, UPS)   # 3072 (init.h5 after step00) unless --nz/--n
NZ    = IN_NZ * UPS
N     = IN_N  * UPS

ENERGY     = _A.energy
VOXELSIZE  = _A.voxelsize
DISTANCE   = _A.distance
ALPHA      = _A.alpha

NTASKS      = _A.ntasks

# Layout from the shared byte-budget policy -- same plan step7_paganin
# uses, so the host-chunked twin writes an identically-laid-out paganin.h5.
# Banking stays θ-split (stype 'proj') because ranks shard on θ and each
# bank file must have exactly one writer; only the chunks are laid out for
# the reader.  step8 reads (NTHETA, zslab, N) sinograms, and the policy
# derives that z-slab from the same function, so there is no --chunk-z for
# the run scripts to keep in sync.  It also falls back to projection order
# once θ_per_bank reaches 1, where a sinogram chunk is all cost.
# Resolved in main(): the policy needs the true 360-degree angle count to
# size stitched/paganin/sino, and only stitched.h5's shape knows it.
_PLAN = NBANKS = VCHUNKS = NPGNCHUNK = H5CHUNKS = None
THETA_PER_BANK = CHUNK_Z = None


def _resolve_plan(n_half):
    global _PLAN, NBANKS, VCHUNKS, NPGNCHUNK, H5CHUNKS
    global THETA_PER_BANK, CHUNK_Z
    _PLAN     = resolve_step("paganin", ups=UPS, in_nz=IN_NZ, in_nyx=IN_N,
                             ntheta=2 * n_half,
                             nbanks=_A.nbanks, mem_budget_gb=_A.mem_budget,
                             chunk_mb=_A.chunk_bytes, npgnchunk=_A.npgnchunk,
                             vchunks=_A.vchunks, order=_A.chunk_order,
                             nranks=SIZE)
    NBANKS    = _PLAN.nbanks
    VCHUNKS   = _PLAN.vchunks
    NPGNCHUNK = _PLAN.align
    H5CHUNKS  = _PLAN.chunks
    if _A.chunk_z:                      # explicit override of the z extent only
        H5CHUNKS = (H5CHUNKS[0], max(1, min(_A.chunk_z, _PLAN.bank_shape[1])),
                    H5CHUNKS[2])
    THETA_PER_BANK = _PLAN.bank_shape[0]
    CHUNK_Z = H5CHUNKS[1]

CHUNK_NZ = _A.chunk_nz
CHUNK_N  = _A.chunk_n


def _validate_chunks() -> None:
    problems = []
    if _A.pad == "none" and NZ % CHUNK_NZ:
        problems.append(f"--chunk-nz={CHUNK_NZ} must divide NZ={NZ}")
    if _A.pad == "none" and N % CHUNK_N:
        problems.append(f"--chunk-n={CHUNK_N} must divide N={N}")
    if VCHUNKS[0] % NPGNCHUNK != 0:
        problems.append(f"--vchunks C0={VCHUNKS[0]} must be a multiple "
                        f"of --npgnchunk={NPGNCHUNK}")
    if VCHUNKS[1] != NZ or VCHUNKS[2] != N:
        problems.append(f"--vchunks C1×C2 must equal NZ×N ({NZ}×{N}) "
                        f"— got {VCHUNKS[1]}×{VCHUNKS[2]}")
    if problems:
        raise SystemExit("chunk-size problems:\n  " + "\n  ".join(problems))


def main() -> None:
    from mpi_utils import banner
    banner("7", f"stitched.h5 -> paganin.h5  "
                f"(single-distance Paganin, large/host-chunked, α={ALPHA})")

    if RANK == 0:
        os.makedirs(DST_DIR, exist_ok=True)
    barrier()

    wavelength = 1.24e-9 / ENERGY

    src_shape, theta_deg = read_src_header(SRC_H5, rank=RANK, comm=COMM)
    N_HALF = src_shape[0]
    if src_shape[1:] != (NZ, N):
        raise SystemExit(
            f"stitched.h5 shape {src_shape} incompatible with (NZ={NZ}, N={N})")
    _resolve_plan(N_HALF)
    _validate_chunks()

    dev_id   = cp.cuda.runtime.getDevice()
    dev_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode()
    rprint(f"[MPI] size={SIZE}  (GPU affinity via set_affinity_gpu.sh)")
    print(f"  rank {RANK}: gpu={dev_id} ({dev_name})  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}",
          flush=True)
    barrier()

    rprint(f"UPS={UPS}  nz={NZ}  n={N}  n_half={N_HALF}  "
           f"npgnchunk={NPGNCHUNK}  chunks=(chunk_nz={CHUNK_NZ}, chunk_n={CHUNK_N})")
    rprint(f"paganin: E={ENERGY} keV  λ={wavelength:.4e} m  "
           f"voxel={VOXELSIZE} m  distance={DISTANCE} m  α={ALPHA}")
    peak_gb = max(NPGNCHUNK * CHUNK_NZ * N * 12,
                  NPGNCHUNK * NZ * CHUNK_N * 16) * 2 / 1e9
    rprint(f"GPU est. — peak stream buffers ≈ {peak_gb:.2f} GB")
    rprint(f"host — fde + out ≈ "
           f"{NPGNCHUNK * NZ * N * (8 + 4) / 1e9:.2f} GB (banded)")

    if RANK == 0:
        describe_input(SRC_H5)
        describe_output(DST_H5, (N_HALF, NZ, N), np.float32,
                        VCHUNKS, "proj", NBANKS, chunks=H5CHUNKS,
                        read_granule=_PLAN.read_granule,
                        companion_bytes=VCHUNKS[0] * NZ * N * 4)

    ctx = initx_and_bcast(DST_H5, shape=(N_HALF, NZ, N),
                          dtype=np.float32, vchunks=VCHUNKS,
                          stype="proj", nbanks=NBANKS,
                          rank=RANK, comm=COMM, chunks=H5CHUNKS)
    if RANK == 0:
        with h5py.File(DST_H5, "r+") as f:
            if "exchange/theta" in f:
                del f["exchange/theta"]
            f["exchange"].create_dataset("theta", data=theta_deg)
    barrier()
    open_out(ctx)                 # single-file output: collective MPI-IO open

    buf_gb = vchunk_bytes(VCHUNKS, np.float32) / 1e9
    rprint(f"per-rank shm buffer={buf_gb:.2f} GB   "
           f"nvchunks={n_vchunks((N_HALF, NZ, N), VCHUNKS)}")

    if _A.pad == "edge":
        # tomocupy retrieve_phase._calc_pad / _calc_pad_width
        pad_pix = int(np.ceil(np.pi * wavelength * DISTANCE / VOXELSIZE**2))
        pz = int((2**int(np.ceil(np.log2(NZ + pad_pix))) - NZ) * 0.5)
        px = int((2**int(np.ceil(np.log2(N + pad_pix))) - N) * 0.5)
    else:
        pz = px = 0
    NZP, NP = NZ + 2 * pz, N + 2 * px

    def _div(n, cap):
        return next(c for c in range(min(cap, n), 0, -1) if n % c == 0)
    cnz, cn = (_div(NZP, CHUNK_NZ), _div(NP, CHUNK_N)) if pz or px \
        else (CHUNK_NZ, CHUNK_N)
    if pz or px:
        rprint(f"edge padding (tomocupy): ({NZ}, {N}) -> ({NZP}, {NP}), "
               f"pad z {pz} x {px}; strips chunk_nz={cnz} chunk_n={cn}")
        pbuf = np.empty((NPGNCHUNK, NZP, NP), np.float32)
        fill_pool = ThreadPoolExecutor(8)

        def pad_into(src):
            b_ = src.shape[0]
            dst = pbuf[:b_]
            rows = np.array_split(np.arange(NZ), 8)

            def cp_rows(r):
                if len(r):
                    dst[:, pz + r[0]:pz + r[-1] + 1, px:px + N] = src[:, r[0]:r[-1] + 1]
            list(fill_pool.map(cp_rows, rows))
            if px:
                dst[:, pz:pz + NZ, :px] = dst[:, pz:pz + NZ, px:px + 1]
                dst[:, pz:pz + NZ, px + N:] = dst[:, pz:pz + NZ, px + N - 1:px + N]
            if pz:
                dst[:, :pz] = dst[:, pz:pz + 1]
                dst[:, pz + NZ:] = dst[:, pz + NZ - 1:pz + NZ]
            return dst

    cl = PaganinLarge(NP, NZP,
                      wavelength=wavelength, voxelsize=VOXELSIZE,
                      distance=DISTANCE, alpha=ALPHA)

    p_min, p_max = np.inf, -np.inf
    p_sum, p_cnt = 0.0, 0
    p_has_nan    = False

    t_read = t_comp = t_write = 0.0
    b_read = b_write = 0

    ivchunks = list(iter_vchunks((N_HALF, NZ, N), VCHUNKS))
    my_ivchunks = ivchunks[RANK::SIZE]
    shm, buf = alloc_shm(VCHUNKS, np.float32)
    # MOSAIC_ASYNC_WRITE=1 (single-file output only): every finished row of
    # HDF5 chunks along theta goes to a writer thread while the next
    # projections are computed, instead of one write after the whole vchunk
    async_w = SINGLE_H5 and os.environ.get("MOSAIC_ASYNC_WRITE") == "1"
    writer = ThreadPoolExecutor(1) if async_w else None
    futs = []
    ch0 = ctx['dset'].chunks[0] if async_w else 0
    if async_w:
        rprint(f"async writes: one theta row of chunks ({ch0} angles) at a time")

    def wait_writes():
        nonlocal t_write
        t = time.perf_counter()
        for f_ in futs:
            f_.result()
        futs.clear()
        t_write += time.perf_counter() - t

    # Prefetch shm for the vchunkx θ-slab (VCHUNKS[0], NZ, N) — aligned
    # read on proj-stored stitched.h5 via NTASKS parallel workers.
    stitched_slab_shape = (VCHUNKS[0], NZ, N)
    shm_slab, stitched_slab_buf = alloc_shm(stitched_slab_shape, np.float32)

    try:
        for k, ivc in enumerate(my_ivchunks, start=1):
            t0_vc = ivc[0] * VCHUNKS[0]
            t1_vc = min(t0_vc + VCHUNKS[0], N_HALF)
            if async_w:
                wait_writes()             # buf is about to be reused
            done_t = 0
            buf.fill(0)

            t0 = time.perf_counter()
            read_projs_vchunkx(SRC_H5, shm_slab, ntasks=NTASKS,
                               vchunksx=stitched_slab_shape,
                               ivchunkx=(ivc[0], 0, 0))
            slab_h = stitched_slab_buf
            t_read += time.perf_counter() - t0
            b_read += (t1_vc - t0_vc) * NZ * N * 4

            for tb0 in range(t0_vc, t1_vc, NPGNCHUNK):
                tb1 = min(tb0 + NPGNCHUNK, t1_vc)
                b   = tb1 - tb0

                t0 = time.perf_counter()
                src_b = slab_h[tb0 - t0_vc : tb1 - t0_vc]
                phase_bh = cl.retrieve(
                    pad_into(src_b) if (pz or px) else src_b,
                    chunks=[cnz, cn])
                cp.get_default_memory_pool().free_all_blocks()
                t_comp += time.perf_counter() - t0

                # phase_bh is a BandedPinned view of cl._out — materialise
                # the (b, NZ, N) slice into buf via BandedPinned.copy_to.
                crop = np.s_[:, pz:pz + NZ, px:px + N]
                if hasattr(phase_bh, 'copy_to'):
                    phase_bh.copy_to(buf[tb0 - t0_vc : tb1 - t0_vc], crop)
                else:
                    buf[tb0 - t0_vc : tb1 - t0_vc] = phase_bh[crop]

                sub = buf[tb0 - t0_vc : tb1 - t0_vc]
                p_min = min(p_min, float(sub.min()))
                p_max = max(p_max, float(sub.max()))
                p_sum += float(sub.sum())
                p_cnt += b * NZ * N
                if np.isnan(sub).any():
                    p_has_nan = True
                if async_w:
                    ready = (t1_vc - t0_vc if tb1 == t1_vc
                             else (tb1 - t0_vc) // ch0 * ch0)
                    if ready > done_t:
                        futs.append(writer.submit(
                            write_region_single, ctx['dset'],
                            buf[done_t:ready], [t0_vc + done_t, 0, 0]))
                        done_t = ready

            del slab_h

            if not async_w:
                t0 = time.perf_counter()
                tomo_writex(DST_H5, data=buf, shm=shm, ivchunk=ivc, ctx=ctx)
                t_write += time.perf_counter() - t0
            b_write += (t1_vc - t0_vc) * NZ * N * 4

            print(f"  [rank {RANK}] vchunk {k}/{len(my_ivchunks)}  "
                  f"θ=[{t0_vc},{t1_vc})  "
                  f"(read={t_read:.1f}s comp={t_comp:.1f}s "
                  f"write={t_write:.1f}s)", flush=True)
        if async_w:
            wait_writes()
    finally:
        if writer is not None:
            writer.shutdown(wait=True)    # no write may outlive buf / the handle
        free_shm(shm)
        free_shm(shm_slab)
    close_out(ctx)

    p_min     = allreduce(p_min,     MPI.MIN)
    p_max     = allreduce(p_max,     MPI.MAX)
    p_sum     = allreduce(p_sum,     MPI.SUM)
    p_cnt     = allreduce(p_cnt,     MPI.SUM)
    p_has_nan = allreduce(p_has_nan, MPI.LOR)
    barrier()

    report_stage("step7 read (stitched)", b_read,  t_read)
    report_stage("step7 write (paganin)", b_write, t_write)

    rprint(f"paganin stats: min={p_min:.4g} max={p_max:.4g} "
           f"mean={p_sum/max(p_cnt,1):.4g} nan={p_has_nan}")
    rprint(f"wrote {N_HALF} angles to {DST_H5}")

    rm_src(SRC_H5, enabled=_A.rm_src, rank=RANK, comm=COMM)


if __name__ == "__main__":
    from mpi_utils import run_main
    run_main(main)
