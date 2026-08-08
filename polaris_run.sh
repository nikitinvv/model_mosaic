#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:15:00
#PBS -q debug
#PBS -N holotomo
#PBS -j oe
#
# End-to-end mosaic-modelling pipeline on Polaris (ALCF).
# Submit:  qsub polaris_run.sh
#
# Writes VDS+banks h5 stores under $PATH_DATA:
#   init.h5, big{UPS}x.h5, model_big{UPS}x/{proj.h5, data.h5}, mosaic_h5/*
#
# For UPS ≥ 8 swap step2_radon.py → step2_radon_large.py.

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4              # ranks per node (= GPUs per node on Polaris)
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Sample dir:  ${SCRIPT_DIR}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

module use /soft/modulefiles
module load conda
conda activate base

cd "${SCRIPT_DIR}"

# ================== USER KNOBS ==================
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/eagle/APS_IRI/vnikitin/mosaic_brain}

NZCHUNK=${NZCHUNK:-32}                       # z-slices per Radon call
NPROPCHUNK=${NPROPCHUNK:-8}                  # angles per Fresnel batch

NBANKS=${NBANKS:-8}                          # bank files per super-chunk
BIG_VCHUNKS=${BIG_VCHUNKS:-}
PROJ_VCHUNKS=${PROJ_VCHUNKS:-}
DATA_VCHUNKS=${DATA_VCHUNKS:-}
# ================================================

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$NTOTRANKS  NBANKS=$NBANKS ==="

# Lustre striping (all OSTs, 4 MB stripes) on every dir that will hold
# bank files.  New files inherit this.
mkdir -p "${PATH_DATA}" \
         "${PATH_DATA}/init"                    \
         "${PATH_DATA}/big${UPS}x"              \
         "${PATH_DATA}/model_big${UPS}x"        \
         "${PATH_DATA}/model_big${UPS}x/proj"   \
         "${PATH_DATA}/model_big${UPS}x/data"
for d in "${PATH_DATA}" \
         "${PATH_DATA}/init" \
         "${PATH_DATA}/big${UPS}x" \
         "${PATH_DATA}/model_big${UPS}x" \
         "${PATH_DATA}/model_big${UPS}x/proj" \
         "${PATH_DATA}/model_big${UPS}x/data"; do
    lfs setstripe -c -1 -S 4M "$d" 2>/dev/null || true
done

vcarg() { local name="$1"; local val="$2"; [[ -n "$val" ]] && echo "--${name}-vchunks $val"; }

# HDF5 file locking is left at the default (enabled).  With the
# tomo_info() cache + ALLOC_TIME_EARLY preallocation in
# iohdf5/dxchange_hdf5_chunks.py, there is no metadata-read storm and
# no concurrent chunk-allocation race, so per-file POSIX locks are
# uncontended and safe on Lustre.
MPIEXEC=(mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}"
         --depth="${NDEPTH}" --cpu-bind depth
         --env OMP_NUM_THREADS="${NTHREADS}"
         "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh")

# ---------- 0. plan mosaic layout ----------------------------------------
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 1. init.h5 → big{UPS}x.h5 ------------------------------------
"${MPIEXEC[@]}" \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" $(vcarg big "$BIG_VCHUNKS")

# ---------- 2. Radon → proj.h5 -------------------------------------------
"${MPIEXEC[@]}" \
    python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" \
        $(vcarg proj "$PROJ_VCHUNKS")
# UPS ≥ 8 (host-chunked TomoLarge; tune chunk-n/-theta/-xy to fit host RAM):
# "${MPIEXEC[@]}" \
#     python step2_radon_large.py --ups "$UPS" --path "$PATH_DATA" \
#         --nzchunk 1 --nbanks "$NBANKS" \
#         --chunk-n 686 --chunk-theta 343 --chunk-xy 686 \
#         $(vcarg proj "$PROJ_VCHUNKS")

# ---------- 3. Fresnel → data.h5 -----------------------------------------
"${MPIEXEC[@]}" \
    python step3_fresnel.py --ups "$UPS" --path "$PATH_DATA" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" \
        $(vcarg data "$DATA_VCHUNKS")

# ---------- 4. data.h5 → mosaic_h5/{z}_{x}.h5 -----------------------------
"${MPIEXEC[@]}" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
