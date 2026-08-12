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

# NBANKS = bank files per super-chunk (also = multiprocessing pool size).
# With NRANKS=4 per node, total writers/node = NRANKS × NBANKS = 16.
# Keep it modest to avoid NIC/OST contention within one node.
NBANKS=4
NTASKS=4

INIT_VCHUNKS="32 4096 4096"
BIG_VCHUNKS="$((32*UPS)) $((4096*UPS)) $((4096*UPS))"
PROJ_VCHUNKS="128 $((4096*UPS)) $((4096*UPS))"
DATA_VCHUNKS="128 $((4096*UPS)) $((4096*UPS))"
# ================================================

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS=4            # matches 4 GPUs/node on Polaris (mirrors polaris_pipeline_run.sh)
NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Jobid: $PBS_JOBID"
echo "Nodes: $NNODES   MPI ranks total: $NTOTRANKS   (--ppn $NRANKS)"
cat "$PBS_NODEFILE"

module use /soft/modulefiles
module load conda
conda activate base

cd "${SCRIPT_DIR}"

# Raise FD limit if possible (Polaris compute nodes usually forbid this;
# default 1024 is plenty for ~640 bank files so the failure is harmless).
ulimit -n 65536 2>/dev/null || true

mkdir -p "${PATH_DATA}"
lfs setstripe -c -1 -S 4M "${PATH_DATA}" 2>/dev/null || true

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NODES=${NNODES}  NRANKS/node=${NRANKS} ==="
echo "    init-vchunks = ${INIT_VCHUNKS}"
echo "    big-vchunks  = ${BIG_VCHUNKS}"
echo "    proj-vchunks = ${PROJ_VCHUNKS}"
echo "    data-vchunks = ${DATA_VCHUNKS}"

# HDF5 file locking left at the default (enabled).  The tomo_info()
# cache in iohdf5/dxchange_hdf5_chunks.py eliminates the metadata-read
# storm that otherwise caused BlockingIOError under NRANKS×NBANKS
# concurrent metadata opens.
# --cpu-bind none: each rank's multiprocessing pool needs all cores.
mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}" --cpu-bind none \
    python "${SCRIPT_DIR}/tests/test_h5_buffer_io.py" \
        --path "${PATH_DATA}" --ups "${UPS}" \
        --nbanks "${NBANKS}" --ntasks "${NTASKS}" \
        --init-vchunks ${INIT_VCHUNKS} \
        --big-vchunks  ${BIG_VCHUNKS} \
        --proj-vchunks ${PROJ_VCHUNKS} \
        --data-vchunks ${DATA_VCHUNKS}
