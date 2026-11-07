#!/bin/bash
#PBS -A 14238
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:grand:eagle
#PBS -l walltime=0:15:00
#PBS -q debug
#PBS -N holotomo
#PBS -j oe
#
# End-to-end mosaic-modelling pipeline on Polaris (ALCF).
#
# Submit one UPS per job.  The PBS block above is a *debug* reservation and
# only ever reaches UPS=1; for anything bigger override select/walltime/queue
# on the command line, which takes precedence over the #PBS lines:
#
#   UPS=1   qsub -v UPS,PATH_DATA polaris_pipeline_run.sh
#   UPS=2   qsub -l select=4:system=polaris  -l walltime=2:00:00  -q prod \
#                -v UPS,PATH_DATA polaris_pipeline_run.sh
#   UPS=8   qsub -l select=8:system=polaris  -l walltime=12:00:00 -q prod \
#                -v UPS,PATH_DATA,STEPS,RM_SRC polaris_pipeline_run.sh
#   UPS=12  qsub -l select=16:system=polaris -l walltime=24:00:00 -q prod \
#                -v UPS,PATH_DATA,STEPS,RM_SRC polaris_pipeline_run.sh
#
# STEPS is the resume knob -- at UPS=8 a full run is many hours, so a rerun
# from step0 after a failure is not an option:
#   STEPS="7b 8" qsub -v UPS,PATH_DATA,STEPS polaris_pipeline_run.sh
#
# Writes VDS+banks h5 stores under $PATH_DATA:
#   init.h5, big{UPS}x.h5,
#   model_big{UPS}x/{proj.h5, data.h5, stitched.h5, paganin.h5, sino.h5, rec.h5},
#   mosaic_h5/*.h5, mosaic_h5_pre/*.h5
#
# GEOMETRY.  init.h5 is 3072^3 float32 (step00 crops the source TIFF to
# 2560^3, upsamples to 3072^3 by factor 1.2, applies a cylindrical mask of
# diameter ~0.95*N with a cosine taper, leaves ~50 zero voxels at each end of
# z with a cosine ramp).  The detector is 3072x3072 at 6.5 um / 9x optics, so
# the voxel is 0.7222 um at EVERY UPS -- UPS grows the sample, not the
# resolution.  Defaults (--circle-diam=2432 --z-pad=42) match the schematic
# (SAMPLE_D_PX = 2918*UPS, SAMPLE_H_PX = 2972*UPS).  At 70 fps:
#
#   UPS  N=NZ   180deg  sample OD x H   tiles   frames      scan    disk peak*
#     1   3072    2304    2.1 x  2.1mm    1x1      4 608     1 min     0.2 TiB
#     2   6144    4608    4.2 x  4.3mm    2x3     55 296    13 min     1.9 TiB
#     4  12288    9216    8.4 x  8.6mm    3x5    276 480     1.1 h    12.6 TiB
#     8  24576   18432   16.9 x 17.2mm    5x9  1 658 880     6.6 h     ~98 TiB
#    12  36864   27648   25.3 x 25.8mm   7x14  5 419 008    21.5 h    ~330 TiB
#    16  49152   36864   33.7 x 34.3mm   9x18 11 943 936    47.4 h    ~780 TiB
#    32  98304   73728   67.4 x 68.7mm  18x36 95 551 488   379.0 h     ~6.2 PiB
#
#   * peak with RM_SRC=1 (each step deletes its own input once it has
#     finished) = max over steps of input+output.  Without it, UPS=8 leaves
#     384 TiB resident and UPS=12 leaves 1.3 PiB.  Check free space against
#     this column before submitting.
#
# The GPU-only Radon / Fresnel / Paganin / FBP buffers stop fitting a 40 GB
# A100 as N grows; this script swaps in the host-chunked *_large variants
# automatically (see LARGE_FROM_* below).  All *_large variants keep the same
# rfft/float32 math but host-chunk the padded fde.

# The software environment.  NOT `module load conda; conda activate base`:
# cray-mpich ships libmpi_gnu.so.12 and no libmpi.so.12, so a conda-forge or
# PyPI mpi4py can never resolve here.  env.sh loads PrgEnv-gnu + cray-mpich +
# cudatoolkit + cray-hdf5-parallel, unloads darshan, exports
# CRAY_LD_LIBRARY_PATH, and activates the conda env holding the `cc`-built
# mpi4py / h5py.  It also sets HDF5_USE_FILE_LOCKING=FALSE and the ROMIO
# no-data-sieving hints that Lustre needs.  Set HTC_ENV_CHECK=1 to have it
# print which python / libmpi actually resolved.
HTC_ENV=${HTC_ENV:-/eagle/APS_IRI/vvnikitin/sw/env.sh}

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4              # ranks per node (= GPUs per node on Polaris)
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Keep a copy of the whole run beside the script.  A UPS=8 job is hours long
# and PBS only flushes its own -j oe file at exit, so this is what you tail.
exec > >(tee "${SCRIPT_DIR}/run-${PBS_JOBID:-local}.out") 2>&1

echo "Sample dir:  ${SCRIPT_DIR}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

[ -r "${HTC_ENV}" ] || { echo "ERROR: HTC_ENV not readable: ${HTC_ENV}"; exit 1; }
source "${HTC_ENV}"

cd "${SCRIPT_DIR}"

# ================== USER KNOBS ==================
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/eagle/APS_IRI/vnikitin/mosaic_brain}

# Which steps to run, space separated, in this order.  Use it to resume:
#   STEPS="7b 8"        redo the transpose + FBP only
#   STEPS="0"           just regenerate the schematic / positions file
STEPS=${STEPS:-"0 1 2 3 4 5 6 7 7b 8"}

# Delete each step's input once that step has finished writing.  This is
# what turns UPS=8 from 384 TiB resident into a ~98 TiB peak.  It never
# touches init.h5 (step1's input, the one irreplaceable file) -- deletion
# starts at step2.  Set RM_SRC=0 to keep every intermediate, e.g. while
# debugging a single step.
RM_SRC=${RM_SRC:-1}

# Physics knobs — kept in sync between step3 (forward Fresnel) and step7
# (Paganin inversion).  DISTANCE is the sample→detector propagation
# distance in metres (near-field regime).  The voxel size is a step
# default (0.72222e-6 m = 6.5 µm / 9) and must match the same two steps.
DISTANCE=${DISTANCE:-0.15}

# FW (wavelet-FFT) ring removal in step5.  OFF (0) -- the simulated tiles
# have no rings, and on real data the filter wants tuning against the
# actual artefact rather than the vendored sym16/level-7 defaults.  Set
# e.g. FW_SIGMA=2 to switch it on (needs pywt in the env).
FW_SIGMA=${FW_SIGMA:-0}

NZCHUNK=${NZCHUNK:-32}                       # z-slices per Radon / FBP call
NPROPCHUNK=${NPROPCHUNK:-8}                  # angles per Fresnel / Paganin batch

NBANKS=${NBANKS:-8}                          # bank files per super-chunk
NTASKS=${NTASKS:-8}                          # parallel workers for read_{projs,slices}_vchunkx
                                             # (steps 3, 7, 7b, 8 prefetch reads via VDS+banks pool)

# UPS at which each step switches to its host-chunked *_large twin.  The
# flags differ between the twins (step2_large has no --ntasks, step3_large
# takes --chunk-nz instead of --npropchunk), so each step builds its argv
# as an array below rather than substituting a filename.
LARGE_FROM_STEP2=${LARGE_FROM_STEP2:-4}
LARGE_FROM_STEP3=${LARGE_FROM_STEP3:-4}
LARGE_FROM_STEP7=${LARGE_FROM_STEP7:-8}
LARGE_FROM_STEP8=${LARGE_FROM_STEP8:-4}

# HDF5 layout policy (iohdf5/layout.py).  Every super-chunk and chunk shape
# in the pipeline comes from these two numbers, so they scale with UPS
# instead of being fixed slice counts:
#
#   MEM_BUDGET_GB   per-rank memory for the output vchunk PLUS the input
#                   prefetch slab -- sizes the super-chunk.  Derived from
#                   /dev/shm, NOT from node RAM: those buffers are
#                   shared_memory.SharedMemory, i.e. files in the tmpfs at
#                   /dev/shm, and Linux caps that tmpfs at half of RAM (252
#                   GiB of the 503 GiB Polaris node).  Sizing off 512 GB RAM
#                   asked 4 x 96 = 384 GiB/node of a 252 GiB tmpfs; tmpfs
#                   pages are allocated lazily, so the overcommit did not
#                   fail at allocation -- it arrived minutes later as SIGBUS
#                   on first touch ("rank N died from signal 7", rc=135),
#                   with no OOM message anywhere.  ~15% is left free for the
#                   NBANKS worker pool and any other /dev/shm user.
#   CHUNK_MB        target size of one HDF5 chunk = the op size Lustre
#                   sees -- sizes the chunk.  Settle it with
#                   polaris_test_h5.sh, which sweeps exactly this knob.
#
# NZCHUNK / NPROPCHUNK above stay the *requested* compute-loop batch; each
# step bends its own down to a divisor of the planned super-chunk when the
# budget forces a smaller one (looping in smaller pieces is free; growing
# the buffer is not).
SHM_GB=${SHM_GB:-$(df -B1 --output=size /dev/shm | tail -1 |
                   awk '{printf "%.1f", $1/1073741824}')}
MEM_BUDGET_GB=${MEM_BUDGET_GB:-$(python3 -c "print(f'{${SHM_GB} * 0.85 / ${NRANKS}:.1f}')")}
CHUNK_MB=${CHUNK_MB:-64}

# Optional --vchunks overrides per step ("C0 C1 C2" as a single string).
# Leave empty to take the policy's shape.
VCHUNKS_STEP1=${VCHUNKS_STEP1:-}
VCHUNKS_STEP2=${VCHUNKS_STEP2:-}
VCHUNKS_STEP3=${VCHUNKS_STEP3:-}
# ================================================

vcarg() { local val="$1"; [[ -n "$val" ]] && echo "--vchunks $val"; }

# Is step "$1" in STEPS?  (Steps are labels, not numbers: "7b" is one.)
want() { [[ " ${STEPS} " == *" $1 "* ]]; }

# --rm-src for every step but step1, whose input is init.h5.
RMARG=()
[[ "${RM_SRC}" == "1" ]] && RMARG=(--rm-src)

# Passed to every step that writes a banked dataset.
LAYOUT=(--mem-budget "$MEM_BUDGET_GB" --chunk-bytes "$CHUNK_MB")

# HDF5 file locking: env.sh exports HDF5_USE_FILE_LOCKING=FALSE, because a
# 40-96 rank h5py open on Lustre can block indefinitely on the flock().  That
# is safe for this pipeline for the same reason the note below gives -- with
# tomo_info() cache + ALLOC_TIME_EARLY preallocation in
# iohdf5/dxchange_hdf5_chunks.py, there is no metadata-read storm and
# no concurrent chunk-allocation race, so per-file POSIX locks are
# uncontended and safe on Lustre.
MPIEXEC=(mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}"
         --depth="${NDEPTH}" --cpu-bind depth
         --env OMP_NUM_THREADS="${NTHREADS}"
         "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh")

# Run one step and stop the whole pipeline if it fails -- a later step on a
# half-written store would waste hours and, with RM_SRC=1, could delete a
# good input on the strength of a bad output.
run() {
    local label="$1"; shift
    echo ""
    echo "######################################################################"
    echo "### step ${label}   $(date +%H:%M:%S)"
    echo "######################################################################"
    local t0=$SECONDS
    "$@"
    local rc=$?
    echo "### step ${label} rc=${rc} in $(( SECONDS - t0 ))s"
    if [[ $rc -ne 0 ]]; then
        echo "### ABORTING pipeline at step ${label}"
        exit $rc
    fi
    return 0          # so `want X && run X ...` does not set the job's rc
}

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$NTOTRANKS  NBANKS=$NBANKS  NTASKS=$NTASKS  DISTANCE=${DISTANCE}m ==="
echo "=== layout: mem-budget=${MEM_BUDGET_GB} GiB/rank  chunk-bytes=${CHUNK_MB} MiB  (/dev/shm=${SHM_GB} GiB, ${NRANKS} ranks/node) ==="
echo "=== steps:  ${STEPS}   rm-src=${RM_SRC} ==="
python3 -m iohdf5.layout --ups "$UPS" --nbanks "$NBANKS" --nranks "$NTOTRANKS" \
    --mem-budget "$MEM_BUDGET_GB" --chunk-bytes "$CHUNK_MB" || true

# Lustre striping (all OSTs, 4 MB stripes) on every dir that will hold
# bank files or plain-HDF5 tile files.  New files inherit this.
DIRS=("${PATH_DATA}"
      "${PATH_DATA}/init"
      "${PATH_DATA}/big${UPS}x"
      "${PATH_DATA}/model_big${UPS}x"
      "${PATH_DATA}/model_big${UPS}x/proj"
      "${PATH_DATA}/model_big${UPS}x/data"
      "${PATH_DATA}/model_big${UPS}x/stitched"
      "${PATH_DATA}/model_big${UPS}x/paganin"
      "${PATH_DATA}/model_big${UPS}x/sino"
      "${PATH_DATA}/model_big${UPS}x/rec"
      "${PATH_DATA}/mosaic_h5"
      "${PATH_DATA}/mosaic_h5_pre")
mkdir -p "${DIRS[@]}"
for d in "${DIRS[@]}"; do
    lfs setstripe -c -1 -S 4M "$d" 2>/dev/null || true
done

# ---------- 0. plan mosaic layout ----------------------------------------
want 0 && run 0 python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 1. init.h5 → big{UPS}x.h5 ------------------------------------
# From UPS=16 the super-chunk is thinner than one UPS output plane (a plane
# is 9.7 GB), so step1 prints a NOTE and re-reads the input plane straddling
# each seam.  Output is identical either way.  No --rm-src: init.h5 is the
# only file here that cannot be recomputed.
want 1 && run 1 "${MPIEXEC[@]}" \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" "${LAYOUT[@]}" $(vcarg "$VCHUNKS_STEP1")

# ---------- 2. Radon → proj.h5 -------------------------------------------
if (( UPS >= LARGE_FROM_STEP2 )); then
    STEP2=(python step2_radon_large.py --ups "$UPS" --path "$PATH_DATA"
           --nzchunk "$NZCHUNK" --nbanks "$NBANKS")   # no --ntasks in the twin
else
    STEP2=(python step2_radon.py --ups "$UPS" --path "$PATH_DATA"
           --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS")
fi
want 2 && run 2 "${MPIEXEC[@]}" "${STEP2[@]}" \
        "${LAYOUT[@]}" "${RMARG[@]}" $(vcarg "$VCHUNKS_STEP2")

# ---------- 3. Fresnel → data.h5 -----------------------------------------
if (( UPS >= LARGE_FROM_STEP3 )); then
    # The twin batches on z (--chunk-nz), not on angles (--npropchunk).
    STEP3=(python step3_propagation_large.py --ups "$UPS" --path "$PATH_DATA"
           --distance "$DISTANCE" --chunk-nz "$NPROPCHUNK"
           --nbanks "$NBANKS" --ntasks "$NTASKS")
else
    STEP3=(python step3_propagation.py --ups "$UPS" --path "$PATH_DATA"
           --distance "$DISTANCE" --npropchunk "$NPROPCHUNK"
           --nbanks "$NBANKS" --ntasks "$NTASKS")
fi
want 3 && run 3 "${MPIEXEC[@]}" "${STEP3[@]}" \
        "${LAYOUT[@]}" "${RMARG[@]}" $(vcarg "$VCHUNKS_STEP3")

# ---------- 4. data.h5 → mosaic_h5/{z}_{x}.h5 -----------------------------
# --z-pad defaults to (NZ - SAMPLE_H_PX)/2 = 50·UPS (schematic z=0 =
# sample top lands on that data row).  --air-fill=1.0 by default
# (transmission of air) for out-of-bounds tile pixels.
want 4 && run 4 "${MPIEXEC[@]}" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA" "${RMARG[@]}"

# ---------- 5. mosaic_h5 → mosaic_h5_pre (dezinger + darkflat + FW) -------
# Per-tile GPU preprocessing, tiles round-robin sharded across ranks.
want 5 && run 5 "${MPIEXEC[@]}" \
    python step5_correct.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --fw-sigma "$FW_SIGMA" "${RMARG[@]}"

# ---------- 6. mosaic_h5_pre/*.h5 → stitched.h5 (tent blend, 180° fold) ---
want 6 && run 6 "${MPIEXEC[@]}" \
    python step6_stitch.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" "${LAYOUT[@]}" "${RMARG[@]}"

# ---------- 7. stitched.h5 → paganin.h5 (single-distance Paganin) --------
if (( UPS >= LARGE_FROM_STEP7 )); then
    STEP7=(python step7_paganin_large.py)
else
    STEP7=(python step7_paganin.py)
fi
want 7 && run 7 "${MPIEXEC[@]}" "${STEP7[@]}" \
        --ups "$UPS" --path "$PATH_DATA" \
        --distance "$DISTANCE" --npgnchunk "$NPROPCHUNK" \
        --nbanks "$NBANKS" --ntasks "$NTASKS" "${LAYOUT[@]}" "${RMARG[@]}"

# ---------- 7b. paganin.h5 → sino.h5 (theta-banked → z-banked) -----------
# Pure I/O + one MPI all-to-all, no GPU.  Without it step8's read is
# transposed against paganin's banking and runs at ~0.5 GB/s instead of
# ~20 GB/s; see step7b_transpose.py's docstring for the measurement.
want 7b && run 7b "${MPIEXEC[@]}" \
    python step7b_transpose.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        "${LAYOUT[@]}" "${RMARG[@]}"

# ---------- 8. sino.h5 → rec.h5 (filtered backprojection) ----------------
# Reads sino.h5 by default, falling back to paganin.h5 if step7b was
# skipped.  Pass --src paganin to force the old path for A/B timing.
if (( UPS >= LARGE_FROM_STEP8 )); then
    STEP8=(python step8_fbp_large.py)
else
    STEP8=(python step8_fbp.py)
fi
want 8 && run 8 "${MPIEXEC[@]}" "${STEP8[@]}" \
        --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        --filter ramp "${LAYOUT[@]}" "${RMARG[@]}"

echo ""
echo "=== pipeline done ==="
