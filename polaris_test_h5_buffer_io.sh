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
# No MPI — parallelism comes from `multiprocessing` inside test_h5_buffer_io.py
# (--nbanks parallel POSIX writers per super-chunk, --ntasks readers).

# ================== USER KNOBS ==================
UPS=1
PATH_DATA=/eagle/APS_IRI/vnikitin/iotest_buf_ups${UPS}

# bank files per super-chunk (parallel writers)
NBANKS=8
# reader worker processes
NTASKS=8

# Super-chunk (RAM buffer) shapes — three ints each: (c0, c1, c2)
INIT_VCHUNKS="32 2744 2744"                          # (nz, ny, nx)   for init.h5
BIG_VCHUNKS="$((32*UPS)) $((2744*UPS)) $((2744*UPS))"    # (nz, ny, nx)   for big{UPS}x.h5
PROJ_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"           # (nθ, nz, nx)   for proj.h5
DATA_VCHUNKS="128 $((2560*UPS)) $((2744*UPS))"           # (nθ, nz, nx)   for data.h5
# ================================================


SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Jobid: $PBS_JOBID   1 node, single Python process, multiprocessing inside"

module use /soft/modulefiles
module load conda
conda activate base
CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
source "/home/vvnikitin/venvs/${CONDA_NAME}/bin/activate"
cd "${SCRIPT_DIR}"

# Lustre striping (all OSTs, 4 MB stripes) — applies to every bank file
# created under this dir since they're written after the setstripe.
mkdir -p "${PATH_DATA}"
lfs setstripe -c -1 -S 4M "${PATH_DATA}" 2>/dev/null || true

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  NBANKS=${NBANKS}  NTASKS=${NTASKS} ==="
echo "    init-vchunks = ${INIT_VCHUNKS}"
echo "    big-vchunks  = ${BIG_VCHUNKS}"
echo "    proj-vchunks = ${PROJ_VCHUNKS}"
echo "    data-vchunks = ${DATA_VCHUNKS}"

python test_h5_buffer_io.py \
    --path "${PATH_DATA}" --ups "${UPS}" \
    --nbanks "${NBANKS}" --ntasks "${NTASKS}" \
    --init-vchunks ${INIT_VCHUNKS} \
    --big-vchunks  ${BIG_VCHUNKS} \
    --proj-vchunks ${PROJ_VCHUNKS} \
    --data-vchunks ${DATA_VCHUNKS}
