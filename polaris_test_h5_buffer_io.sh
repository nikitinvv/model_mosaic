#!/bin/bash
#PBS -A 14238
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:20:00
#PBS -q debug
#PBS -N mosaic_buf_io
#PBS -j oe
#
# Single-node RAM-buffer + multi-bank HDF5 I/O benchmark.
# 4 MPI ranks on 1 node (matches 4 GPUs/node in polaris_run.sh).
# Use this as a single-node baseline; scale up via polaris_test_h5_buffer_io_mpi.sh.

# ================== USER KNOBS ==================
UPS=1
PATH_DATA=/eagle/APS_IRI/vnikitin/iotest_buf_ups${UPS}

# NBANKS = bank files per super-chunk (also = multiprocessing pool per rank).
# With NRANKS=4 per node, total writers/node = NRANKS × NBANKS = 16.
NBANKS=4
NTASKS=4

# Super-chunk (RAM buffer) shapes — three ints each: (c0, c1, c2)
INIT_VCHUNKS="32 2744 2744"                              # (nz, ny, nx)   for init.h5
BIG_VCHUNKS="$((32*UPS)) $((2744*UPS)) $((2744*UPS))"    # (nz, ny, nx)   for big{UPS}x.h5
PROJ_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"           # (nθ, nz, nx)   for proj.h5
DATA_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"           # (nθ, nz, nx)   for data.h5
# ================================================


SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS=4                                                  # matches 4 GPUs/node
NTOTRANKS=$(( NNODES * NRANKS ))
echo "Jobid: $PBS_JOBID   Nodes=${NNODES}   Ranks total=${NTOTRANKS}   (--ppn ${NRANKS})"

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

# Disable HDF5 POSIX lock probes on Lustre (see mpi launcher for rationale).
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "${PATH_DATA}"
lfs setstripe -c -1 -S 4M "${PATH_DATA}" 2>/dev/null || true

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NRANKS/node=${NRANKS} ==="
echo "    init-vchunks = ${INIT_VCHUNKS}"
echo "    big-vchunks  = ${BIG_VCHUNKS}"
echo "    proj-vchunks = ${PROJ_VCHUNKS}"
echo "    data-vchunks = ${DATA_VCHUNKS}"

mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}" --cpu-bind none \
    --env HDF5_USE_FILE_LOCKING=FALSE \
    python test_h5_buffer_io.py \
        --path "${PATH_DATA}" --ups "${UPS}" \
        --nbanks "${NBANKS}" --ntasks "${NTASKS}" \
        --init-vchunks ${INIT_VCHUNKS} \
        --big-vchunks  ${BIG_VCHUNKS} \
        --proj-vchunks ${PROJ_VCHUNKS} \
        --data-vchunks ${DATA_VCHUNKS}
