#!/usr/bin/env bash
# End-to-end mosaic-modelling pipeline on a local tomo machine (tomo5).
#
# Edit UPS / PATH_DATA / N_GPUS + the NBANKS/VCHUNKS knobs below, then:
#     bash tomo_run.sh
#
# For UPS ≥ 8 swap step2_radon.py → step2_radon_large.py (host-chunked
# TomoLarge — same output format, safe for large N on a 40 GB GPU).
# On Polaris use polaris_run.sh (PBS wrapper + set_affinity_gpu_polaris.sh).

#set -euo pipefail 

# ================== USER KNOBS ==================
UPS=${UPS:-1}
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
# mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
#     python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
#         --nbanks "$NBANKS" $(vcarg big "$BIG_VCHUNKS")

# 2. big{UPS}x.h5 → model_big{UPS}x/proj.h5   (Radon; VDS+banks).
#    For UPS ≥ 8 swap for step2_radon_large.py with --chunk-n/-theta/-xy.
# mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
#     python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
#         --nzchunk "$NZCHUNK" --nbanks "$NBANKS" \
#         $(vcarg proj "$PROJ_VCHUNKS")

# UPS ≥ 8 (host-chunked TomoLarge; tune chunk-n/-theta/-xy to fit host RAM):
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step2_radon_large.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" \
        --chunk-n 686 --chunk-theta 343 --chunk-xy 686 \
        $(vcarg proj "$PROJ_VCHUNKS")

# 3. proj.h5 → data.h5   (Fresnel propagation to detector intensities).
mpirun --quiet -n "$N_GPUS" set_affinity_gpu.sh \
    python step3_fresnel.py --ups "$UPS" --path "$PATH_DATA" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" \
        $(vcarg data "$DATA_VCHUNKS")

# 4. data.h5 → mosaic_h5/{z}_{x}.h5   (MPI-parallel over tiles; CPU only).
mpirun --quiet -n "$N_GPUS" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
