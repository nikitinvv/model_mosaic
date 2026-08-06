#!/usr/bin/env python
"""Model detector intensities for a plane-wave illumination through the
2560x3000x3000 init volume:

    psi  = exp(i R(delta + i beta))       # transmission wave
    data = | D_distance(psi) |^2          # Fresnel propagation, intensity

delta is the loaded volume, beta = delta / BETA_RATIO.

The volume is streamed from disk in z-chunks for the Radon step. Fresnel
propagation is then applied per angle-batch on the GPU. Two per-angle TIFF
stacks are written: complex64 psi_XXX.tif and float32 data_XXX.tif.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp
import tifffile

sys.path.insert(0, "/home/beams2/VNIKITIN/holotomocupy_mpi/src")
from holotomocupy.tomo import Tomo
from holotomocupy.propagation import Propagation


SRC_DIR = "/data2/brain_sym_mosaic/init"
DST_DIR = "/data2/brain_sym_mosaic/model"

NZ         = 2560
N          = 3000
NTHETA     = 16
BETA_RATIO = 100.0   # beta = delta / BETA_RATIO
MASK_R     = 0.0     # in-plane apodisation mask radius; 0 disables it (a circular
                     # mask is already baked into the init volume by upsample_extract)

# Fresnel propagation (all lengths in meters)
ENERGY     = 30.0        # keV
VOXELSIZE  = 0.7e-6      # m
DISTANCE   = 0.5         # m
NPROP_BATCH = int(os.environ.get("NPROP_BATCH", "4"))  # angles per propagation pass

NCHUNK  = int(os.environ.get("NCHUNK", "8"))    # z-slices per GPU pass
N_LOAD_THREADS = int(os.environ.get("N_LOAD_THREADS", "16"))
GPU_ID  = int(os.environ.get("GPU_ID", "0"))


def load_chunk(z_start: int, z_end: int, pool: ThreadPoolExecutor) -> np.ndarray:
    """Read TIFFs [z_start, z_end) into a stacked [k, N, N] float32 array."""
    k = z_end - z_start
    buf = np.empty((k, N, N), dtype=np.float32)

    def _read(i: int) -> None:
        buf[i] = tifffile.imread(os.path.join(SRC_DIR, f"init_{z_start + i:05d}.tif"))

    list(pool.map(_read, range(k)))
    return buf


def main() -> None:
    os.makedirs(DST_DIR, exist_ok=True)
    cp.cuda.Device(GPU_ID).use()

    theta = np.linspace(0.0, np.pi, NTHETA, endpoint=False).astype("float32")
    print(f"gpu={GPU_ID}  nz={NZ} n={N} ntheta={NTHETA} nchunk={NCHUNK} "
          f"mask_r={MASK_R} beta_ratio={BETA_RATIO}")
    print(f"theta (rad): {theta}")

    cl_tomo = Tomo(N, NCHUNK, theta, mask_r=MASK_R)

    psi = np.empty((NTHETA, NZ, N), dtype=np.complex64)

    with ThreadPoolExecutor(max_workers=N_LOAD_THREADS) as pool:
        for z0 in range(0, NZ, NCHUNK):
            z1 = min(z0 + NCHUNK, NZ)
            k  = z1 - z0
            chunk_h = load_chunk(z0, z1, pool)

            # Tomo buffers are sized for exactly NCHUNK z-slices; pad the tail.
            if k < NCHUNK:
                pad = np.zeros((NCHUNK, N, N), dtype=np.float32)
                pad[:k] = chunk_h
                chunk_h = pad

            # delta + i*beta on the GPU
            delta_d = cp.asarray(chunk_h)
            vol_d   = cp.empty(delta_d.shape, dtype=cp.complex64)
            vol_d.real = delta_d
            vol_d.imag = delta_d * cp.float32(1.0 / BETA_RATIO)
            del delta_d

            proj_d = cl_tomo.R(vol_d)                    # complex64 [ntheta, NCHUNK, N]
            del vol_d

            psi_d  = cp.exp(1j * proj_d).astype(cp.complex64)
            del proj_d

            psi[:, z0:z1] = cp.asnumpy(psi_d[:, :k])
            del psi_d
            cp.get_default_memory_pool().free_all_blocks()
            print(f"  z {z1}/{NZ}", flush=True)

    print(f"psi stats: |psi| min={np.abs(psi).min():.4g} max={np.abs(psi).max():.4g}  "
          f"arg range=[{np.angle(psi).min():.4g}, {np.angle(psi).max():.4g}]  "
          f"nan={np.isnan(psi).any()}")

    # --- Fresnel propagation, intensity at the detector ---------------------
    wavelength = 1.24e-9 / ENERGY
    fresnel_number = (VOXELSIZE ** 2) / (wavelength * DISTANCE)
    print(f"prop: E={ENERGY} keV  lambda={wavelength:.4e} m  "
          f"voxel={VOXELSIZE} m  distance={DISTANCE} m  "
          f"Fresnel number (per pixel)={fresnel_number:.4g}")

    cl_prop = Propagation(N, NZ, NPROP_BATCH, 1, wavelength,
                          VOXELSIZE, [DISTANCE])

    data = np.empty((NTHETA, NZ, N), dtype=np.float32)
    for i0 in range(0, NTHETA, NPROP_BATCH):
        i1 = min(i0 + NPROP_BATCH, NTHETA)
        b  = i1 - i0
        psi_d = cp.asarray(psi[i0:i1])
        if b < NPROP_BATCH:
            pad = cp.zeros((NPROP_BATCH, NZ, N), dtype=cp.complex64)
            pad[:b] = psi_d
            psi_d = pad
        prop_d = cl_prop.D(psi_d, 0)
        intens_d = (prop_d.real * prop_d.real +
                    prop_d.imag * prop_d.imag).astype(cp.float32)
        data[i0:i1] = cp.asnumpy(intens_d[:b])
        del psi_d, prop_d, intens_d
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  prop {i1}/{NTHETA}", flush=True)

    print(f"data stats: min={data.min():.4g} max={data.max():.4g} "
          f"mean={data.mean():.4g} nan={np.isnan(data).any()}")

    # --- Write per-angle TIFFs (complex psi + intensity data) --------------
    def _write(i: int) -> None:
        tifffile.imwrite(
            os.path.join(DST_DIR, f"psi_{i:05d}.tif"), psi[i], compression=None
        )
        tifffile.imwrite(
            os.path.join(DST_DIR, f"data_{i:05d}.tif"), data[i], compression=None
        )

    with ThreadPoolExecutor(max_workers=min(NTHETA, N_LOAD_THREADS)) as pool:
        list(pool.map(_write, range(NTHETA)))
    print(f"wrote {NTHETA} psi + {NTHETA} data tiffs to {DST_DIR}")


if __name__ == "__main__":
    main()
