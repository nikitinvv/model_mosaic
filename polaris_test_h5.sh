#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:30:00
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
# Sweeps the paganin.h5 chunk shape (see VARIANTS below), which is what
# stage-3 write cost and stage-4 FBP read cost both hinge on.

# ================== USER KNOBS ==================
UPS=1
PATH_DATA=/eagle/APS_IRI/vnikitin/iotest_buf_ups${UPS}_mpi

# NBANKS = bank files per super-chunk (also = multiprocessing pool size).
# With NRANKS=4 per node, total writers/node = NRANKS × NBANKS = 16.
# Keep it modest to avoid NIC/OST contention within one node.s
NBANKS=4
# NTASKS = read-pool workers.  Stage 4 shards them along θ, aligned to the
# chunk's θ extent, so raising this is free parallelism as long as
# NTHETA/NTASKS stays a multiple of θ_per_bank (= PGN_VCHUNKS C0 / NBANKS).
NTASKS=4
NZCHUNK=1                        # inner z-slab for fbp read loop

INIT_VCHUNKS="32 3072 3072"
BIG_VCHUNKS="$((32*UPS)) $((3072*UPS)) $((3072*UPS))"
PROJ_VCHUNKS="128 $((3072*UPS)) $((3072*UPS))"
DATA_VCHUNKS="128 $((3072*UPS)) $((3072*UPS))"
PGN_VCHUNKS="128 $((3072*UPS)) $((3072*UPS))"
REC_VCHUNKS="$((32*UPS)) $((3072*UPS)) $((3072*UPS))"

# ---- paganin.h5 chunk-shape sweep -------------------------------------
# Stage 4 reads (NTHETA, C0, N) sinogram slabs, where C0 = REC_VCHUNKS C0.
# What that read costs is decided entirely by the chunk shape stage 3
# wrote paganin.h5 with.  Each entry is "order:chunk_z"; chunk_z 0 means
# "auto = REC_VCHUNKS C0".  At UPS=1 / NBANKS=4 (θ_per_bank=32, 36 bank
# files, 1152 θ):
#
#   proj:0   (1, NZ, N)      37.8 MB chunk   1152 ops/read @ 0.39 MB
#            original layout — same bytes off disk, but strided 37.8 MB
#            apart, so every op is a seek.
#   sino:1   (32, 1, N)       0.39 MB chunk  1152 ops/read @ 0.39 MB
#            sequential now, but the op count never dropped — this is
#            the ~3 ms/op Lustre latency wall.
#   sino:0   (32, 32, N)     12.6 MB chunk     36 ops/read @ 12.6 MB
#            one whole-chunk sequential read per bank file.
#
# Also watch stage-3 paganin WRITE across the three: sino:1 costs 3072
# chunk ops per bank file, sino:0 costs 96.
#
# Set to a single entry for a plain (non-sweep) run.
VARIANTS=("proj:0" "sino:1" "sino:0")
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

echo "=== UPS=${UPS}  PATH_DATA=${PATH_DATA}  NBANKS=${NBANKS}  NTASKS=${NTASKS}  NZCHUNK=${NZCHUNK}  NODES=${NNODES}  NRANKS/node=${NRANKS} ==="
echo "    init-vchunks = ${INIT_VCHUNKS}"
echo "    big-vchunks  = ${BIG_VCHUNKS}"
echo "    proj-vchunks = ${PROJ_VCHUNKS}"
echo "    data-vchunks = ${DATA_VCHUNKS}"
echo "    pgn-vchunks  = ${PGN_VCHUNKS}"
echo "    rec-vchunks  = ${REC_VCHUNKS}"
echo "    variants     = ${VARIANTS[*]}"

# HDF5 file locking left at the default (enabled).  The tomo_info()
# cache in iohdf5/dxchange_hdf5_chunks.py eliminates the metadata-read
# storm that otherwise caused BlockingIOError under NRANKS×NBANKS
# concurrent metadata opens.
# --cpu-bind none: each rank's multiprocessing pool needs all cores.
#
# Each variant is a full 5-stage run: the test recreates every dataset
# from scratch (rank 0 cleans the master + bank dir before tomo_initx),
# so the variants do not contaminate each other.  Stages 1-2 are
# identical across variants — their spread is a free noise estimate for
# reading stages 3-4.
for V in "${VARIANTS[@]}"; do
    PGN_ORDER="${V%%:*}"
    PGN_CHUNK_Z="${V##*:}"

    echo ""
    echo "######################################################################"
    echo "### VARIANT  pgn-chunk-order=${PGN_ORDER}  pgn-chunk-z=${PGN_CHUNK_Z}"
    echo "###          (chunk-z 0 = auto, takes REC_VCHUNKS C0)"
    echo "######################################################################"
    VAR_T0=$SECONDS

    mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}" --cpu-bind none \
        python -m tests.test_h5_buffer_io \
            --path "${PATH_DATA}" --ups "${UPS}" \
            --nbanks "${NBANKS}" --ntasks "${NTASKS}" --nzchunk "${NZCHUNK}" \
            --pgn-chunk-order "${PGN_ORDER}" --pgn-chunk-z "${PGN_CHUNK_Z}" \
            --init-vchunks ${INIT_VCHUNKS} \
            --big-vchunks  ${BIG_VCHUNKS} \
            --proj-vchunks ${PROJ_VCHUNKS} \
            --data-vchunks ${DATA_VCHUNKS} \
            --pgn-vchunks  ${PGN_VCHUNKS} \
            --rec-vchunks  ${REC_VCHUNKS}
    RC=$?

    echo "### VARIANT ${V} finished rc=${RC} in $(( SECONDS - VAR_T0 ))s"
    if [ "${RC}" -ne 0 ]; then
        echo "### aborting sweep"
        exit "${RC}"
    fi
done

echo ""
echo "=== sweep done.  Compare across variants:"
echo "===   'paganin  write'  (stage 3) — chunk ops per bank file"
echo "===   'fbp      read'   (stage 4) — ops per read x bytes per op"
