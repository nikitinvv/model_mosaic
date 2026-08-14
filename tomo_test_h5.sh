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
#
# Every super-chunk and HDF5 chunk shape comes from iohdf5/layout.py, the
# same policy the pipeline steps use, driven by two numbers:
#
#   --mem-budget GiB   per-rank RAM for the output vchunk PLUS the input
#                      prefetch slab.  Sizes the super-chunk.
#   --chunk-bytes MiB  target size of one HDF5 chunk = the op size the
#                      filesystem sees.  Sizes the chunk.

set -euo pipefail

# ================== USER KNOBS ==================
# N = NZ = 3072·UPS, so the full volumes run from 526 GB at UPS=1 to
# 13 PB at UPS=32 — see MAX_VCHUNKS below.
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/data2/vnikitin/iotest_buf_ups${UPS}}
N_GPUS=${N_GPUS:-4}                          # total ranks

# NBANKS = bank files per super-chunk (also = multiprocessing pool size).
# Total writers on the machine = N_GPUS × NBANKS.  The layout policy
# lowers it per dataset when one super-chunk cannot hold that many whole
# planes (unavoidable at high UPS: one plane is 36 GB at UPS=32).
NBANKS=${NBANKS:-8}
NTASKS=${NTASKS:-4}
NZCHUNK=${NZCHUNK:-8}                        # inner z-slab for fbp read loop

# Super-chunks written per dataset (0 = the whole volume).  From UPS=4 on
# the full volumes are tens of TB and rank 0 would create ~1e5 empty bank
# files per dataset before any byte moves.  --max-vchunks truncates each
# dataset along its banked axis while keeping the planned vchunk / chunk /
# nbanks shapes — the things being measured — at full size.  Keep it a
# multiple of N_GPUS so no rank idles.  The cost is printed (--dry-run)
# before anything is written.
MAX_VCHUNKS=${MAX_VCHUNKS:-8}
FULL_VOLUME_UPTO=${FULL_VOLUME_UPTO:-1}      # UPS <= this runs the whole volume

# Per-rank RAM budget.  Every stage holds a PAIR of super-chunks — the one
# it writes plus the one it reads — so the peak is roughly twice this; the
# harness prints the actual per-stage peak and warns if it exceeds it.
NODE_GB=${NODE_GB:-512}
MEM_FRACTION=${MEM_FRACTION:-0.5}
MEM_BUDGET_GB=${MEM_BUDGET_GB:-$(python3 -c "print(f'{${NODE_GB} * ${MEM_FRACTION} / ${N_GPUS}:.1f}')")}

# HDF5 chunk-byte targets to sweep, MiB.  Per-op filesystem latency is
# what the stage-4 read is bound by, so this is the axis that matters.
# Cut it to a single value once one wins.
CHUNK_MB_LIST=${CHUNK_MB_LIST:-"16 64 256"}
# ================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Disable HDF5 POSIX-lock probes so N_GPUS × NBANKS concurrent processes
# opening the same master VDS + bank files don't race on lock probes.
export HDF5_USE_FILE_LOCKING=FALSE

if [ "${UPS}" -le "${FULL_VOLUME_UPTO}" ]; then MV=0; else MV=${MAX_VCHUNKS}; fi
FIRST_CHUNK_MB=${CHUNK_MB_LIST%% *}

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  N_GPUS=${N_GPUS}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NZCHUNK=${NZCHUNK} ==="
echo "    chunk-MB     = ${CHUNK_MB_LIST}"
echo "    mem-budget   = ${MEM_BUDGET_GB} GiB/rank  (${NODE_GB} GB × ${MEM_FRACTION} / ${N_GPUS} ranks)"
echo "    max-vchunks  = ${MV}   (0 = whole volume)"
echo ""
echo "    Planned layout for these settings (pure arithmetic, no I/O):"
python -m iohdf5.layout --ups "${UPS}" --nbanks "${NBANKS}" \
    --nranks "${N_GPUS}" --mem-budget "${MEM_BUDGET_GB}" \
    --chunk-bytes "${FIRST_CHUNK_MB}" || true

# What this run will cost, before spending an afternoon on it.
echo ""
python -m tests.test_h5_buffer_io --path "${PATH_DATA}" \
    --ups "${UPS}" --nbanks "${NBANKS}" --ntasks "${NTASKS}" \
    --nzchunk "${NZCHUNK}" --mem-budget "${MEM_BUDGET_GB}" \
    --chunk-bytes "${FIRST_CHUNK_MB}" --max-vchunks "${MV}" --dry-run \
    2>/dev/null | grep -E "TOTAL written|peak RAM|WARNING" | sed "s/^/    /" || true

mkdir -p "${PATH_DATA}"

# Each chunk size is a full 5-stage run: the test recreates every dataset
# from scratch (rank 0 cleans the master + bank dir before tomo_initx), so
# the runs do not contaminate each other.
for CHUNK_MB in ${CHUNK_MB_LIST}; do
    echo ""
    echo "######################################################################"
    echo "### RUN  ups=${UPS}  chunk-bytes=${CHUNK_MB} MiB  max-vchunks=${MV}"
    echo "######################################################################"
    T0=$SECONDS

    mpirun -n "${N_GPUS}" set_affinity_gpu.sh \
        python -m tests.test_h5_buffer_io \
            --path "${PATH_DATA}" --ups "${UPS}" \
            --nbanks "${NBANKS}" --ntasks "${NTASKS}" --nzchunk "${NZCHUNK}" \
            --mem-budget "${MEM_BUDGET_GB}" --chunk-bytes "${CHUNK_MB}" \
            --max-vchunks "${MV}"

    echo "### RUN ups=${UPS} chunk=${CHUNK_MB}MiB done in $(( SECONDS - T0 ))s"
done
