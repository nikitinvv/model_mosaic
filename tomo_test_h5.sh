#!/usr/bin/env bash
# Multi-rank MPI+VDS throughput test for test_h5_buffer_io.py on a local
# tomo machine (handyn etc.).  Analog of polaris_test_h5.sh without PBS.
#
# One MPI rank per GPU (default N_GPUS=4).  All ranks write to ONE
# VDS-backed dataset per stage; rank 0 creates the master + empty bank
# files, then every rank iterates its own subset of vchunks
# (ivchunks[R::SIZE]) and writes disjoint bank files.  No cross-rank
# coordination needed beyond the barrier after tomo_initx.
#
# Aggregate throughput = sum(bytes) / max(rank elapsed) — printed by the
# script after each stage.

set -euo pipefail

# ================== USER KNOBS ==================
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/data2/vnikitin/iotest_buf_ups${UPS}}
N_GPUS=${N_GPUS:-4}                          # total ranks

# NBANKS = bank files per super-chunk (also = multiprocessing pool size).
# Total writers on the machine = N_GPUS × NBANKS.
NBANKS=${NBANKS:-8}
NTASKS=${NTASKS:-4}
NZCHUNK=${NZCHUNK:-8}                        # inner z-slab for fbp read loop

INIT_VCHUNKS=${INIT_VCHUNKS:-"32 3072 3072"}
BIG_VCHUNKS=${BIG_VCHUNKS:-"$((32*UPS)) $((3072*UPS)) $((3072*UPS))"}
PROJ_VCHUNKS=${PROJ_VCHUNKS:-"128 $((3072*UPS)) $((3072*UPS))"}
DATA_VCHUNKS=${DATA_VCHUNKS:-"128 $((3072*UPS)) $((3072*UPS))"}
PGN_VCHUNKS=${PGN_VCHUNKS:-"128 $((3072*UPS)) $((3072*UPS))"}
REC_VCHUNKS=${REC_VCHUNKS:-"$((32*UPS)) $((3072*UPS)) $((3072*UPS))"}
# ================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Disable HDF5 POSIX-lock probes so N_GPUS × NBANKS concurrent processes
# opening the same master VDS + bank files don't race on lock probes.
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "${PATH_DATA}"

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  N_GPUS=${N_GPUS}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NZCHUNK=${NZCHUNK} ==="
echo "    init-vchunks = ${INIT_VCHUNKS}"
echo "    big-vchunks  = ${BIG_VCHUNKS}"
echo "    proj-vchunks = ${PROJ_VCHUNKS}"
echo "    data-vchunks = ${DATA_VCHUNKS}"
echo "    pgn-vchunks  = ${PGN_VCHUNKS}"
echo "    rec-vchunks  = ${REC_VCHUNKS}"

cd "${SCRIPT_DIR}"
mpirun -n "${N_GPUS}" set_affinity_gpu.sh \
    python -m tests.test_h5_buffer_io \
        --path "${PATH_DATA}" --ups "${UPS}" \
        --nbanks "${NBANKS}" --ntasks "${NTASKS}" --nzchunk "${NZCHUNK}" \
        --init-vchunks ${INIT_VCHUNKS} \
        --big-vchunks  ${BIG_VCHUNKS} \
        --proj-vchunks ${PROJ_VCHUNKS} \
        --data-vchunks ${DATA_VCHUNKS} \
        --pgn-vchunks  ${PGN_VCHUNKS} \
        --rec-vchunks  ${REC_VCHUNKS}
