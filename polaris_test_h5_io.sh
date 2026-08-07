#!/bin/bash
#PBS -A 14347
#PBS -l select=8:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -N mosaic_io
#PBS -j oe
#
# Parallel-h5 I/O benchmark for the mosaic pipeline on Polaris.
# Edit the variables in the "USER KNOBS" block below, then:  qsub polaris_test_h5_io.sh

# ================== USER KNOBS ==================
UPS=2
PATH_DATA=/eagle/APS_IRI/vnikitin/iotest_ups${UPS}

# Fresnel batch size (planes per Fresnel call)
NPROPCHUNK=1

# h5 chunk shapes — three ints each:  (c0, c1, c2)
INIT_CHUNKS="1 2744 2744"                # (nz, ny, nx)   for init.h5
BIG_CHUNKS="1 $((2744*UPS)) $((2744*UPS))"   # (nz, ny, nx)   for big{UPS}x.h5
PROJ_CHUNKS="1 32 $((2744*UPS))"          # (nθ, nz, nx)   for proj.h5
DATA_CHUNKS="1 $((2560*UPS)) $((2744*UPS))"  # (nθ, nz, nx)   for data.h5
# ================================================


NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Jobid: $PBS_JOBID   ${NNODES} nodes × ${NRANKS} ranks = ${NTOTRANKS} total"

module use /soft/modulefiles
module load conda
conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
source "/home/vvnikitin/venvs/${CONDA_NAME}/bin/activate"
cd "${SCRIPT_DIR}"

# Lustre striping (all OSTs, 4 MB stripes).
mkdir -p "${PATH_DATA}"
lfs setstripe -c -1 -S 4M "${PATH_DATA}" 2>/dev/null || true

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  NPROPCHUNK=${NPROPCHUNK} ==="
echo "    init-chunks = ${INIT_CHUNKS}"
echo "    big-chunks  = ${BIG_CHUNKS}"
echo "    proj-chunks = ${PROJ_CHUNKS}"
echo "    data-chunks = ${DATA_CHUNKS}"

mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}" \
    --depth="${NDEPTH}" --cpu-bind depth \
    --env OMP_NUM_THREADS="${NTHREADS}" \
    "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh" \
    python test_h5_io.py \
        --path "${PATH_DATA}" --ups "${UPS}" \
        --npropchunk "${NPROPCHUNK}" \
        --init-chunks ${INIT_CHUNKS} \
        --big-chunks  ${BIG_CHUNKS} \
        --proj-chunks ${PROJ_CHUNKS} \
        --data-chunks ${DATA_CHUNKS}
