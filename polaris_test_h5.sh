#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -N mosaic_buf_io_mpi
#PBS -j oe
#
# Multi-node MPI+VDS throughput test for test_h5_buffer_io.py.
#
# NRANKS MPI ranks per node (--ppn ${NRANKS}).  All ranks write to ONE
# VDS-backed dataset per stage; rank 0 creates the master + empty bank
# files, then every rank iterates its own subset of vchunks
# (ivchunks[R::SIZE]) and writes disjoint bank files.  No cross-rank file
# coordination needed beyond the barrier after tomo_initx.
#
# Aggregate throughput = sum(bytes) / max(rank elapsed) — printed by the
# script after each stage.
#
# Every super-chunk and HDF5 chunk shape now comes from iohdf5/layout.py,
# the same policy the pipeline steps use, driven by two numbers:
#
#   --mem-budget GiB   per-rank RAM for the output vchunk PLUS the input
#                      prefetch slab.  Sizes the super-chunk.
#   --chunk-bytes MiB  target size of one HDF5 chunk = the op size Lustre
#                      sees.  Sizes the chunk.
#
# So the sweep below is over those two knobs, not over six hand-written
# vchunk strings — which is what let the old settings silently become
# 9 GB chunks and TB-scale buffers at UPS>1.

# ================== USER KNOBS ==================
# N = NZ = 3072·UPS, so UPS=32 is N=98304 and the full rec.h5 would be
# 3.4 PB — see MAX_VCHUNKS below, which is what makes the high-UPS runs
# possible at all.  Run one UPS per job:
#     UPS=16 qsub -v UPS polaris_test_h5.sh
UPS=${UPS:-1}

# Super-chunks written per dataset (0 = the whole volume).  From UPS=4 on
# the full volumes are 40 TB to 3.4 PB and rank 0 would create ~1e5 empty
# bank files per dataset before any byte moves.  --max-vchunks truncates
# each dataset along its banked axis while keeping the planned vchunk /
# chunk / nbanks shapes — the things being measured — at full size.  Keep
# it a multiple of NTOTRANKS so no rank idles.
# At UPS=1 the whole volume is only 8 super-chunks wide anyway, so the cap
# does nothing there; it starts biting at UPS=2 (3.4 TB full -> 1.4 TB).
# The volume this costs is printed (from --dry-run) before anything runs.
MAX_VCHUNKS=${MAX_VCHUNKS:-8}
FULL_VOLUME_UPTO=1               # UPS <= this runs the whole volume

PATH_BASE=/eagle/APS_IRI/vnikitin

# Per-rank RAM budget.  Polaris nodes are 512 GB.  The naive split is
# 512/NRANKS, but every stage here holds a PAIR of super-chunks — the one
# it writes plus the one it reads — so the peak is roughly twice the
# budget.  0.5 keeps 4 ranks × ~70 GB well inside the node; the harness
# prints the actual per-stage peak and warns if it exceeds the budget.
NODE_GB=512
MEM_FRACTION=0.5

# HDF5 chunk-byte targets to sweep, MiB.  The last Polaris sweep only ever
# measured 12.6 MB chunks, and per-op Lustre latency (~3 ms) is what the
# stage-4 read is bound by — so this is the axis most likely to matter.
# Cut it to a single value once one wins.
CHUNK_MB_LIST=(16 64 256)

# NBANKS = bank files per super-chunk (also = multiprocessing pool size).
# With NRANKS=4 per node, total writers/node = NRANKS × NBANKS = 16.
# Keep it modest to avoid NIC/OST contention within one node.  The layout
# policy lowers it per dataset when one super-chunk cannot hold that many
# whole planes (unavoidable at high UPS: one plane is 36 GB at UPS=32).
NBANKS=4
# NTASKS = read-pool workers.  Stage 4 shards them along θ, aligned to the
# chunk's θ extent, so raising this is free parallelism as long as
# NTHETA/NTASKS stays a multiple of θ_per_bank.
NTASKS=4
NZCHUNK=1                        # inner z-slab for fbp read loop

# paganin.h5 is always written sinogram-ordered with the policy's z extent
# — settled by the last sweep at UPS=1 / NBANKS=4: 36 whole-chunk ops per
# stage-4 read @ 12.6 MB, against 1152 ops @ 0.39 MB for both projection
# order and a 1-plane-deep sino chunk.  That is the harness default, so
# there is nothing to pass here.
# ================================================

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS=4            # matches 4 GPUs/node on Polaris (mirrors polaris_pipeline_run.sh)
NTOTRANKS=$(( NNODES * NRANKS ))

# Per-rank budget in GiB, derived from the node so it tracks NRANKS.
MEM_BUDGET_GB=$(python3 -c "print(f'{${NODE_GB} * ${MEM_FRACTION} / ${NRANKS}:.1f}')")

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

# Super-chunks per dataset: 0 (whole volume) below FULL_VOLUME_UPTO.
if [ "${UPS}" -le "${FULL_VOLUME_UPTO}" ]; then MV=0; else MV=${MAX_VCHUNKS}; fi

PATH_DATA=${PATH_BASE}/iotest_buf_ups${UPS}_mpi

echo "=== UPS=${UPS}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NZCHUNK=${NZCHUNK}  NODES=${NNODES}  NRANKS/node=${NRANKS} ==="
echo "    chunk-MB     = ${CHUNK_MB_LIST[*]}"
echo "    mem-budget   = ${MEM_BUDGET_GB} GiB/rank  (${NODE_GB} GB node × ${MEM_FRACTION} / ${NRANKS} ranks)"
echo "    max-vchunks  = ${MV}   (0 = whole volume)"
echo "    path         = ${PATH_DATA}"
echo ""
echo "    Planned layout for these settings (pure arithmetic, no I/O):"
python3 -m iohdf5.layout --ups "${UPS}" --nbanks "${NBANKS}" \
    --nranks "${NTOTRANKS}" --mem-budget "${MEM_BUDGET_GB}" \
    --chunk-bytes "${CHUNK_MB_LIST[0]}" 2>/dev/null || true

# What this run will cost, before spending the job on it.  --dry-run prints
# the planned shapes, the total bytes and the bank-file count, then exits
# without creating anything.
echo ""
python3 -m tests.test_h5_buffer_io --path "${PATH_DATA}" \
    --ups "${UPS}" --nbanks "${NBANKS}" --ntasks "${NTASKS}" \
    --nzchunk "${NZCHUNK}" --mem-budget "${MEM_BUDGET_GB}" \
    --chunk-bytes "${CHUNK_MB_LIST[0]}" --max-vchunks "${MV}" --dry-run \
    2>/dev/null | grep -E "TOTAL written|peak RAM|WARNING" \
    | sed "s/^/    /" || true

mkdir -p "${PATH_DATA}"
lfs setstripe -c -1 -S 4M "${PATH_DATA}" 2>/dev/null || true

# HDF5 file locking left at the default (enabled).  The tomo_info()
# cache in iohdf5/dxchange_hdf5_chunks.py eliminates the metadata-read
# storm that otherwise caused BlockingIOError under NRANKS×NBANKS
# concurrent metadata opens.
# --cpu-bind none: each rank's multiprocessing pool needs all cores.
#
# Each chunk size is a full 5-stage run: the test recreates every dataset
# from scratch (rank 0 cleans the master + bank dir before tomo_initx), so
# the runs do not contaminate each other.
for CHUNK_MB in "${CHUNK_MB_LIST[@]}"; do
    echo ""
    echo "######################################################################"
    echo "### RUN  ups=${UPS}  chunk-bytes=${CHUNK_MB} MiB  max-vchunks=${MV}"
    echo "######################################################################"
    T0=$SECONDS

    mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}" --cpu-bind none \
        python -m tests.test_h5_buffer_io \
            --path "${PATH_DATA}" --ups "${UPS}" \
            --nbanks "${NBANKS}" --ntasks "${NTASKS}" --nzchunk "${NZCHUNK}" \
            --mem-budget "${MEM_BUDGET_GB}" --chunk-bytes "${CHUNK_MB}" \
            --max-vchunks "${MV}"
    RC=$?

    echo "### RUN ups=${UPS} chunk=${CHUNK_MB}MiB finished rc=${RC} in $(( SECONDS - T0 ))s"
    if [ "${RC}" -ne 0 ]; then
        echo "### aborting sweep"
        exit "${RC}"
    fi
done

echo ""
echo "=== sweep done.  Compare across the chunk sizes:"
echo "===   'paganin  write'  (stage 3) — chunk ops per bank file"
echo "===   'fbp      read'   (stage 4) — ops per read x bytes per op"
echo "===   where per-op latency stops dominating is the size to keep"
