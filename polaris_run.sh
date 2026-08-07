#!/bin/bash
#PBS -A 14347
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
# The pipeline writes h5 files under $PATH_DATA:
#   init.h5, big{UPS}x.h5, model_big{UPS}x/{proj.h5, data.h5}, mosaic_h5/*
#
# For UPS ≥ 8 swap step2_model_big.py → step2_model_large.py (host-chunked
# TomoLarge; keep --nchunk 1 and set --chunk-n/--chunk-theta/--chunk-xy).

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
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="/home/vvnikitin/venvs/${CONDA_NAME}"
source "${VENV_DIR}/bin/activate"

cd "${SCRIPT_DIR}"

# ---------- job config ---------------------------------------------------
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/eagle/APS_IRI/vnikitin/mosaic_brain}

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$NTOTRANKS ==="

# Set Lustre striping on the model output directory the first time this
# UPS is used — spreads each h5 file across all OSTs, avoids single-OST
# contention when many ranks write in parallel.  Harmless to re-run
# (setstripe on an existing dir affects only newly-created files).
mkdir -p "${PATH_DATA}/model_big${UPS}x"
lfs setstripe -c -1 -S 4M "${PATH_DATA}/model_big${UPS}x" 2>/dev/null || true

MPIEXEC=(mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}"
         --depth="${NDEPTH}" --cpu-bind depth
         --env OMP_NUM_THREADS="${NTHREADS}"
         "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh")

# ---------- 0. plan mosaic layout ----------------------------------------
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 1. init.h5 → big{UPS}x.h5 ------------------------------------
"${MPIEXEC[@]}" \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 2. Radon + Fresnel → proj.h5, data.h5 -------------------------
# For UPS ≥ 8, swap to step2_model_large.py:
#   python step2_model_large.py --ups "$UPS" --path "$PATH_DATA" \
#       --nchunk 1 --nprop-batch 1 \
#       --chunk-n 686 --chunk-theta 343 --chunk-xy 686
"${MPIEXEC[@]}" \
    python step2_model_big.py --ups "$UPS" --path "$PATH_DATA" \
                              --nchunk 32 --nprop-batch 8

# ---------- 3. data.h5 → mosaic_h5/{z}_{x}.h5 -----------------------------
# CPU-only but MPI-shards tiles across ranks.  set_affinity_gpu_polaris.sh
# is harmless here (just sets CUDA_VISIBLE_DEVICES that Python ignores).
"${MPIEXEC[@]}" \
    python step3_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
