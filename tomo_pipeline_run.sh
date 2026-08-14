#!/usr/bin/env bash
# End-to-end mosaic-modelling pipeline on a local tomo machine (tomo5).
#
# Edit UPS / PATH_DATA / N_GPUS + the NBANKS/VCHUNKS knobs below, then:
#     bash tomo_pipeline_run.sh
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
# output).  Same idea for step3_propagation → step3_propagation_large,
# and step7_paganin → step7_paganin_large, step8_fbp → step8_fbp_large.
# On Polaris use polaris_pipeline_run.sh (PBS wrapper + set_affinity_gpu_polaris.sh).

# ================== USER KNOBS ==================
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/data2/brain_sym_mosaic}
N_GPUS=${N_GPUS:-4}                          # total ranks (= total GPUs)

# Physics knobs — kept in sync between step3 (forward Fresnel) and step7
# (Paganin inversion).  DISTANCE is the sample→detector propagation
# distance in metres (near-field regime).
DISTANCE=${DISTANCE:-0.2}

# Compute chunk sizes
NZCHUNK=${NZCHUNK:-8}                        # z-slices per Radon call
NPROPCHUNK=${NPROPCHUNK:-8}                  # angles per Fresnel batch

# Bank / vchunk knobs (see test_h5_buffer_io.py for details).
# On local NVMe/NFS keep NBANKS modest — total writers/node = N_GPUS × NBANKS,
# and a single local disk backend saturates at ~8-16 concurrent writers.
# On Lustre (Polaris, `-c -1`) you can push NBANKS=8 since each stripe = one OST.
NBANKS=${NBANKS:-8}                          # bank files per super-chunk
NTASKS=${NTASKS:-8}                          # parallel workers for read_{projs,slices}_vchunkx
                                             # (steps 3, 7, 8 prefetch reads via VDS+banks pool)
# Optional --vchunks overrides per step ("C0 C1 C2" as a single string);
# leave empty for each step's default.
VCHUNKS_STEP1=${VCHUNKS_STEP1:-}   # step1 big.h5  default (8·UPS, OUT_NYX, OUT_NYX)
VCHUNKS_STEP2=${VCHUNKS_STEP2:-}   # step2 proj.h5 default (NTHETA, 8·NZCHUNK, N)
VCHUNKS_STEP3=${VCHUNKS_STEP3:-}   # step3 data.h5 default (8·NPROPCHUNK, NZ, N)
# ================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Raise FD limit — bank layouts can have hundreds of h5 files open.
ulimit -n 65536 2>/dev/null || true

# Disable HDF5 POSIX-lock probes.  On tomo5's local ext4, N_GPUS × NBANKS
# concurrent Python processes opening the same master VDS + bank files
# back-to-back can trip EAGAIN.  Single-node local disk has no
# cross-client coherency issue, so disabling locks is safe here.
# (On Polaris/Lustre this is unsafe; keep it OFF in polaris_pipeline_run.sh.)
export HDF5_USE_FILE_LOCKING=FALSE

# Skip mpi4py's auto MPI.Finalize.  On tomo5, MPI's SM/UCX teardown
# races with multiprocessing spawn Pool cleanup during Finalize and
# has historically hung the process.  mpi_utils.py sees this env var and
# sets mpi4py.rc.finalize=False.  Cost: OpenMPI's mpirun prints an
# "abnormal termination" warning at the end of each stage — cosmetic,
# not an error (compare against a real crash: run_main's [rank R]
# EXCEPTION traceback would appear above the warning).
export MOSAIC_SKIP_MPI_FINALIZE=1

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$N_GPUS  NBANKS=$NBANKS  NTASKS=$NTASKS  DISTANCE=${DISTANCE}m ==="

# helper: turn an optional "C0 C1 C2" string into --vchunks args, or nothing.
vcarg() { local val="$1"; [[ -n "$val" ]] && echo "--vchunks $val"; }

# 0. Plan the mosaic (schematic PNG + tile-origin txt).
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# 1. init.h5 → big{UPS}x.h5  (bilinear xy + linear z upsample; VDS+banks).
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" $(vcarg "$VCHUNKS_STEP1")

# 2. big{UPS}x.h5 → model_big{UPS}x/proj.h5   (Radon; VDS+banks).
#    step2_radon.py uses TomoReal (GPU-only rfft/float32).  For UPS ≥ 4
#    swap for step2_radon_large.py (host-chunked TomoLargeReal — same
#    rfft/float32 math but the fde lives on host, so N is bounded only
#    by host RAM).

mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        $(vcarg "$VCHUNKS_STEP2")
# For UPS≥4 swap in step2_radon_large.py (see README).

# 3. proj.h5 → data.h5   (Fresnel propagation to detector intensities).
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step3_propagation.py --ups "$UPS" --path "$PATH_DATA" \
        --distance "$DISTANCE" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        $(vcarg "$VCHUNKS_STEP3")
# For UPS≥4 swap in step3_propagation_large.py (see README).

# 4. data.h5 → mosaic_h5/{z}_{x}.h5   (MPI-parallel over tiles; CPU only).
#    Tile layout comes from step0_schematic (30 mm sample cylinder inscribed
#    in the (4096·UPS)^3 dataset).  --z-pad default is 0 since init.h5 IS
#    the sample; pass --z-pad N only if your init.h5 has an air band
#    ABOVE the sample.  --air-fill defaults to 1.0 (transmission of air)
#    for out-of-bounds tile regions.
mpirun --quiet -n "$N_GPUS" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

# 5. mosaic_h5/*.h5 → mosaic_h5_pre/*.h5   (per-tile GPU preprocessing:
#    dezinger + dark-flat field correction + FW ring removal).  Tiles are
#    round-robin sharded across ranks; each rank runs on one GPU
#    (set_affinity_gpu.sh).  On synthetic tiles data_white=1 / data_dark=0
#    the dezinger + darkflat pieces are effectively no-ops; the FW ring
#    removal is always applied.
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step5_correct.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK"

# 6. mosaic_h5_pre/*.h5 → stitched.h5   (tent-weight blend, mirror-fold to 180°).
mpirun --quiet -n "$N_GPUS" \
    python step6_stitch.py --ups "$UPS" --path "$PATH_DATA"

# 7. stitched.h5 → paganin.h5   (single-distance Paganin, GPU per θ batch).
#    step7_paganin.py uses a full 2-D FFT per batch (fits on a 40 GB GPU
#    up to UPS≈4 with --npgnchunk 8).  For UPS ≥ 8 swap for
#    step7_paganin_large.py (host-chunked 3-pass streaming, --chunk-nz
#    / --chunk-n bound peak GPU memory).
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step7_paganin.py --ups "$UPS" --path "$PATH_DATA" \
        --distance "$DISTANCE" \
        --npgnchunk "$NPROPCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS"
# For UPS≥8 swap in step7_paganin_large.py (see README).

# 8. paganin.h5 → rec.h5   (filtered backprojection, default filter=shepp).
#    step8_fbp.py uses GPU-only Tomo.RT (fits up to UPS≈4 with --nzchunk 8).
#    For UPS ≥ 4 swap for step8_fbp_large.py (host-chunked TomoLarge.RT,
#    4 reversed passes: r-FFT → adj scatter → y-IFFT → x-IFFT+phi+crop).
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step8_fbp.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" --filter ramp
# For UPS≥4 swap in step8_fbp_large.py (see README).

echo "=== pipeline done ==="
