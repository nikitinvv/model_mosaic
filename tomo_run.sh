#!/usr/bin/env bash
# End-to-end mosaic-modelling pipeline on a local tomo machine (handyn etc.).
#
# Edit UPS / PATH_DATA / N_GPUS + the NBANKS/VCHUNKS knobs below, then:
#     bash tomo_run.sh
#
# For UPS ≥ 8 swap step2_radon.py → step2_radon_large.py (host-chunked
# TomoLarge — same output format, safe for large N on a 40 GB GPU).
# On Polaris use polaris_run.sh (PBS wrapper + set_affinity_gpu_polaris.sh).

set -euo pipefail

# ================== USER KNOBS ==================
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/data2/brain_sym_mosaic}
N_GPUS=${N_GPUS:-4}                          # total ranks (= total GPUs)

# Compute chunk sizes
NZCHUNK=${NZCHUNK:-32}                       # z-slices per Radon call
NPROPCHUNK=${NPROPCHUNK:-8}                  # angles per Fresnel batch

# Bank / vchunk knobs (see test_h5_buffer_io.py for details).
# On local NVMe/NFS keep NBANKS modest — total writers/node = N_GPUS × NBANKS,
# and a single local disk backend saturates at ~8-16 concurrent writers.
# On Lustre (Polaris, `-c -1`) you can push NBANKS=8 since each stripe = one OST.
NBANKS=${NBANKS:-4}                          # bank files per super-chunk
# Optional overrides; leave empty for defaults.
# Defaults: BIG  = (8·UPS, OUT_NYX, OUT_NYX)
#           PROJ = (NTHETA, 8·NZCHUNK, N)
#           DATA = (8·NPROPCHUNK, NZ, N)
BIG_VCHUNKS=${BIG_VCHUNKS:-}
PROJ_VCHUNKS=${PROJ_VCHUNKS:-}
DATA_VCHUNKS=${DATA_VCHUNKS:-}
# ================================================

# UCX transport tuning — avoid the benign teardown races that flood
# stderr at MPI_Finalize on Rocky/RHEL 8:
#   - `mm_sysv.c ... shmat failed`     ← SysV shared-memory cleanup race
#   - `mm_posix.c ... open failed`     ← POSIX shm cleanup race (peer's /proc/*/fd/*)
# Disabling BOTH shared-memory transports pushes intra-node MPI onto
# self+cma+tcp.  Reductions are tiny (a few scalars per stage) so the
# tcp fallback cost is negligible on this pipeline.
export UCX_TLS="^sysv,^posix"
export UCX_LOG_LEVEL=error

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$N_GPUS  NBANKS=$NBANKS ==="

# helper: turn optional VCHUNKS env into "--<name>-vchunks C0 C1 C2" args
vcarg() { local name="$1"; local val="$2"; [[ -n "$val" ]] && echo "--${name}-vchunks $val"; }

# 0. Plan the mosaic (schematic PNG + tile-origin txt).
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# 1. init.h5 → big{UPS}x.h5  (bilinear xy + linear z upsample; VDS+banks).
mpirun -n "$N_GPUS" set_affinity_gpu.sh \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" $(vcarg big "$BIG_VCHUNKS")

# 2. big{UPS}x.h5 → model_big{UPS}x/proj.h5   (Radon; VDS+banks).
#    For UPS ≥ 8 swap for step2_radon_large.py with --chunk-n/-theta/-xy.
mpirun -n "$N_GPUS" set_affinity_gpu.sh \
    python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" \
        $(vcarg proj "$PROJ_VCHUNKS")

# 3. proj.h5 → data.h5   (Fresnel propagation to detector intensities).
mpirun -n "$N_GPUS" set_affinity_gpu.sh \
    python step3_fresnel.py --ups "$UPS" --path "$PATH_DATA" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" \
        $(vcarg data "$DATA_VCHUNKS")

# 4. data.h5 → mosaic_h5/{z}_{x}.h5   (MPI-parallel over tiles; CPU only).
mpirun -n "$N_GPUS" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
