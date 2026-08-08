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
# TomoLarge; keep --nzchunk 1 and set --chunk-n/--chunk-theta/--chunk-xy).

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

CONDA_ENV_CANDIDATES=(holotomocupy)
VENV_CANDIDATES=(
    "${HOME}/venvs/vvnikitin/bin/activate"
    "${HOME}/venvs/${CONDA_NAME}/bin/activate"
    "/home/vvnikitin/venvs/vvnikitin/bin/activate"
    "/home/vvnikitin/venvs/${CONDA_NAME}/bin/activate"
)
_env_activated=0
for e in "${CONDA_ENV_CANDIDATES[@]}"; do
    if [[ -d "${HOME}/.conda/envs/${e}" ]]; then
        echo "activating conda env: ${e}"
        conda activate "${e}"
        _env_activated=1
        break
    fi
done
if (( ! _env_activated )); then
    for v in "${VENV_CANDIDATES[@]}"; do
        if [[ -f "$v" ]]; then
            echo "activating venv: $v"
            source "$v"
            _env_activated=1
            break
        fi
    done
fi
if (( ! _env_activated )); then
    echo "WARNING: no project env activated; using base conda at ${CONDA_PREFIX}" >&2
fi

cd "${SCRIPT_DIR}"

# ---------- job config ---------------------------------------------------
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/eagle/APS_IRI/vnikitin/mosaic_brain}

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$NTOTRANKS ==="

# Set Lustre striping on every dir that will hold a big h5 file.  All new
# files in these dirs inherit the setting.  We also `lfs migrate` any
# already-existing big h5 files in case they were created with the default
# stripe_count=1 (which serialises many-rank reads onto one OST).
mkdir -p "${PATH_DATA}" "${PATH_DATA}/model_big${UPS}x"
lfs setstripe -c -1 -S 4M "${PATH_DATA}"                    2>/dev/null || true
lfs setstripe -c -1 -S 4M "${PATH_DATA}/model_big${UPS}x"   2>/dev/null || true
for f in "${PATH_DATA}/init.h5" "${PATH_DATA}/big${UPS}x.h5"; do
    if [[ -f "$f" ]] && [[ "$(lfs getstripe -c "$f" 2>/dev/null | tail -1)" == "1" ]]; then
        echo "  migrating $f to full-stripe layout..."
        lfs migrate -c -1 -S 4M "$f"
    fi
done

# Disable HDF5 POSIX lock probes on Lustre — same rationale as in the
# I/O test scripts: N ranks all opening the same master/bank files for
# metadata reads race on file locks and get BlockingIOError.
export HDF5_USE_FILE_LOCKING=FALSE

MPIEXEC=(mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}"
         --depth="${NDEPTH}" --cpu-bind depth
         --env OMP_NUM_THREADS="${NTHREADS}"
         --env HDF5_USE_FILE_LOCKING=FALSE
         "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh")

# ---------- 0. plan mosaic layout ----------------------------------------
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 1. init.h5 → big{UPS}x.h5 ------------------------------------
"${MPIEXEC[@]}" \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 2. Radon + Fresnel → proj.h5, data.h5 -------------------------
# For UPS ≥ 8, swap to step2_model_large.py:
#   python step2_model_large.py --ups "$UPS" --path "$PATH_DATA" \
#       --nzchunk 1 --npropchunk 1 \
#       --chunk-n 686 --chunk-theta 343 --chunk-xy 686
"${MPIEXEC[@]}" \
    python step2_model_big.py --ups "$UPS" --path "$PATH_DATA" \
                              --nzchunk 32 --npropchunk 8

# ---------- 3. data.h5 → mosaic_h5/{z}_{x}.h5 -----------------------------
# CPU-only but MPI-shards tiles across ranks.  set_affinity_gpu_polaris.sh
# is harmless here (just sets CUDA_VISIBLE_DEVICES that Python ignores).
"${MPIEXEC[@]}" \
    python step3_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
