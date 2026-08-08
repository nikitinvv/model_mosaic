#!/bin/bash
#PBS -A 14238
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:30:00
#PBS -q debug-scaling
#PBS -N mosaic_buf_io_mpi
#PBS -j oe
#
# Multi-node MPI+VDS throughput test for test_h5_buffer_io.py.
#
# One MPI rank per node (--ppn 1).  All ranks write to ONE VDS-backed
# dataset per stage; rank 0 creates the master + empty bank files, then
# every rank iterates its own subset of vchunks (ivchunks[R::SIZE]) and
# writes disjoint bank files.  No cross-rank file coordination needed
# beyond the barrier after tomo_initx.
#
# Aggregate throughput = sum(bytes) / max(rank elapsed) — printed by the
# script after each stage.

# ================== USER KNOBS ==================
UPS=1
PATH_DATA=/eagle/APS_IRI/vnikitin/iotest_buf_ups${UPS}_mpi

NBANKS=8            # bank files per super-chunk (per rank's multiprocessing pool)
NTASKS=8            # reader worker processes per rank

INIT_VCHUNKS="32 2744 2744"
BIG_VCHUNKS="$((32*UPS)) $((2744*UPS)) $((2744*UPS))"
PROJ_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"
DATA_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"
# ================================================

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS=1            # one Python process per node; multiprocessing inside fans across cores
NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Jobid: $PBS_JOBID"
echo "Nodes: $NNODES   MPI ranks total: $NTOTRANKS   (--ppn $NRANKS)"
cat "$PBS_NODEFILE"

module use /soft/modulefiles
module load conda
conda activate base
CONDA_NAME=$(echo "${CONDA_PREFIX}" | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')

# Prefer a project conda env (has mpi4py, h5py-mpi, cupy).  Fall back to
# a venv, then to bare base conda.  Add your name to the lists as needed.
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

# Raise FD limit — one rank can open hundreds of bank files with VDS.
ulimit -n 65536 || true

mkdir -p "${PATH_DATA}"
lfs setstripe -c -1 -S 4M "${PATH_DATA}" 2>/dev/null || true

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NODES=${NNODES} ==="
echo "    init-vchunks = ${INIT_VCHUNKS}"
echo "    big-vchunks  = ${BIG_VCHUNKS}"
echo "    proj-vchunks = ${PROJ_VCHUNKS}"
echo "    data-vchunks = ${DATA_VCHUNKS}"

# --cpu-bind none: rank's multiprocessing pool needs access to all cores.
mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}" --cpu-bind none \
    python "${SCRIPT_DIR}/test_h5_buffer_io.py" \
        --path "${PATH_DATA}" --ups "${UPS}" \
        --nbanks "${NBANKS}" --ntasks "${NTASKS}" \
        --init-vchunks ${INIT_VCHUNKS} \
        --big-vchunks  ${BIG_VCHUNKS} \
        --proj-vchunks ${PROJ_VCHUNKS} \
        --data-vchunks ${DATA_VCHUNKS}
