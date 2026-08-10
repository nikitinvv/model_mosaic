#!/usr/bin/env bash
# End-to-end mosaic-modelling pipeline on a local tomo machine (tomo5).
#
# Edit UPS / PATH_DATA / N_GPUS + the NBANKS/VCHUNKS knobs below, then:
#     bash tomo_run.sh
#
# Assumes init.h5 exists at $PATH_DATA/init.h5.  init.h5 is a 3072^3
# float32 volume (step00 crops the source TIFF to 2560^3, upsamples to
# 3072^3 by factor 1.2, applies a cylindrical mask of diameter ≈ 0.95·N
# with a cosine taper, and leaves ~50 zero voxels at each end of z with
# a cosine ramp).  Voxel at UPS=1 = 11.04 µm; at UPS=8 the pipeline
# voxel becomes 1.38 µm = detector pixel.  Physical dataset = constant
# 33.92 mm cube; sample ≈ ⌀32.2 × 32.8 mm cylinder inside.  Defaults
# (--circle-diam=2432 --z-pad=42) match the schematic's
# SAMPLE_D_PX = 2918·UPS, SAMPLE_H_PX = 2972·UPS.
#
# For UPS ≥ 4 the (nz, 2n, 2n) GPU-only padded buffer in step2_radon
# starts to strain a 40 GB GPU; swap to step2_radon_large.py which uses
# the host-chunked TomoLargeReal (rfft/float32, half the host fde, same
# output).  Same idea for step3_fresnel → step3_fresnel_large.
# On Polaris use polaris_run.sh (PBS wrapper + set_affinity_gpu_polaris.sh).

#set -euo pipefail 

# ================== USER KNOBS ==================
UPS=4 # ${UPS:-1}
PATH_DATA=${PATH_DATA:-/data2/brain_sym_mosaic}
N_GPUS=${N_GPUS:-4}                          # total ranks (= total GPUs)

# Compute chunk sizes
NZCHUNK=${NZCHUNK:-8}                        # z-slices per Radon call
NPROPCHUNK=${NPROPCHUNK:-8}                  # angles per Fresnel batch

# Bank / vchunk knobs (see test_h5_buffer_io.py for details).
# On local NVMe/NFS keep NBANKS modest — total writers/node = N_GPUS × NBANKS,
# and a single local disk backend saturates at ~8-16 concurrent writers.
# On Lustre (Polaris, `-c -1`) you can push NBANKS=8 since each stripe = one OST.
NBANKS=${NBANKS:-8}                          # bank files per super-chunk
# Optional overrides; leave empty for defaults.
# Defaults: BIG  = (8·UPS, OUT_NYX, OUT_NYX)
#           PROJ = (NTHETA, 8·NZCHUNK, N)
#           DATA = (8·NPROPCHUNK, NZ, N)
BIG_VCHUNKS=${BIG_VCHUNKS:-}
PROJ_VCHUNKS=${PROJ_VCHUNKS:-}
DATA_VCHUNKS=${DATA_VCHUNKS:-}
# ================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Raise FD limit — bank layouts can have hundreds of h5 files open.
ulimit -n 65536 2>/dev/null || true

# Disable HDF5 POSIX-lock probes.  On tomo5's local ext4, N_GPUS × NBANKS
# concurrent Python processes opening the same master VDS + bank files
# back-to-back can trip EAGAIN.  Single-node local disk has no
# cross-client coherency issue, so disabling locks is safe here.
# (On Polaris/Lustre this is unsafe; keep it OFF in polaris_run.sh.)
export HDF5_USE_FILE_LOCKING=FALSE

# Skip mpi4py's auto MPI.Finalize.  On tomo5, MPI's SM/UCX teardown
# races with multiprocessing spawn Pool cleanup during Finalize and
# has historically hung the process.  utils.py sees this env var and
# sets mpi4py.rc.finalize=False.  Cost: OpenMPI's mpirun prints an
# "abnormal termination" warning at the end of each stage — cosmetic,
# not an error (compare against a real crash: run_main's [rank R]
# EXCEPTION traceback would appear above the warning).
export MOSAIC_SKIP_MPI_FINALIZE=1

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$N_GPUS  NBANKS=$NBANKS ==="

# helper: turn optional VCHUNKS env into "--<name>-vchunks C0 C1 C2" args
vcarg() { local name="$1"; local val="$2"; [[ -n "$val" ]] && echo "--${name}-vchunks $val"; }

# 0. Plan the mosaic (schematic PNG + tile-origin txt).
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# 1. init.h5 → big{UPS}x.h5  (bilinear xy + linear z upsample; VDS+banks).
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" $(vcarg big "$BIG_VCHUNKS")

# 2. big{UPS}x.h5 → model_big{UPS}x/proj.h5   (Radon; VDS+banks).
#    step2_radon.py uses TomoReal (GPU-only rfft/float32).  For UPS ≥ 4
#    swap for step2_radon_large.py (host-chunked TomoLargeReal — same
#    rfft/float32 math but the fde lives on host, so N is bounded only
#    by host RAM).

# mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
#     python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
#         --nzchunk "$NZCHUNK" --nbanks "$NBANKS" \
#         $(vcarg proj "$PROJ_VCHUNKS")

# UPS ≥ 4 (host-chunked TomoLargeReal; chunk-n/-theta/-xy auto-picked
# from --gpu-budget-gb — override with --chunk-n/-theta/-xy if you need
# to fit tighter host RAM).  Recommended --nzchunk=1 at high UPS since
# the (nz, 2n, n+1) host fde scales with nz.
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step2_radon_large.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk 1 --nbanks "$NBANKS" \
        $(vcarg proj "$PROJ_VCHUNKS")

# 3. proj.h5 → data.h5   (Fresnel propagation to detector intensities).
# mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
#     python step3_fresnel.py --ups "$UPS" --path "$PATH_DATA" \
#         --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" \
#         $(vcarg data "$DATA_VCHUNKS")
# UPS ≥ 4 (host-chunked PropagationLarge; chunk-nz/-2n auto-picked from
# --gpu-budget-gb, override with --chunk-nz/--chunk-2n if needed).
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step3_fresnel_large.py --ups "$UPS" --path "$PATH_DATA" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" \
        $(vcarg data "$DATA_VCHUNKS")

# 4. data.h5 → mosaic_h5/{z}_{x}.h5   (MPI-parallel over tiles; CPU only).
#    Tile layout comes from step0_schematic (30 mm sample cylinder inscribed
#    in the (4096·UPS)^3 dataset).  --z-pad default is 0 since init.h5 IS
#    the sample; pass --z-pad N only if your init.h5 has an air band
#    ABOVE the sample.  --air-fill defaults to 1.0 (transmission of air)
#    for out-of-bounds tile regions.
mpirun --quiet -n "$N_GPUS" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
