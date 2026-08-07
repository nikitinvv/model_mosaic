#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:20:00
#PBS -q debug
#PBS -N mosaic_zarr_io
#PBS -j oe
#
# Zarr I/O benchmark for the mosaic pipeline on Polaris — mirror of
# polaris_test_h5_io.sh with the same USER KNOBS.  Run both and compare.

# ================== USER KNOBS ==================
UPS=1
PATH_DATA=/eagle/APS_IRI/vnikitin/iotest_zarr_ups${UPS}

# Fresnel batch size (planes per Fresnel call)
NPROPCHUNK=1

# Zarr chunk shapes — three ints each:  (c0, c1, c2)
INIT_CHUNKS="1 2744 2744"                        # (nz, ny, nx)   for init.zarr
BIG_CHUNKS="1 $((2744*UPS)) $((2744*UPS))"       # (nz, ny, nx)   for big{UPS}x.zarr
PROJ_CHUNKS="128 1 $((2744*UPS))"                # (nθ, nz, nx)   for proj.zarr
DATA_CHUNKS="1 $((2560*UPS)) $((2744*UPS))"      # (nθ, nz, nx)   for data.zarr
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

# Lustre striping (all OSTs, 4 MB stripes).  Each zarr chunk is a separate
# file, so striping applies to every chunk file created under this dir.
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
    python test_zarr_io.py \
        --path "${PATH_DATA}" --ups "${UPS}" \
        --npropchunk "${NPROPCHUNK}" \
        --init-chunks ${INIT_CHUNKS} \
        --big-chunks  ${BIG_CHUNKS} \
        --proj-chunks ${PROJ_CHUNKS} \
        --data-chunks ${DATA_CHUNKS}
