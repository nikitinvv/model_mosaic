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
# Submit:  qsub polaris_pipeline_run.sh
#
# Writes VDS+banks h5 stores under $PATH_DATA:
#   init.h5, big{UPS}x.h5,
#   model_big{UPS}x/{proj.h5, data.h5, stitched.h5, paganin.h5, rec.h5},
#   mosaic_h5/*.h5, mosaic_h5_pre/*.h5
#
# init.h5 is 3072^3 float32 (step00 crops the source TIFF to 2560^3,
# upsamples to 3072^3 by factor 1.2, applies a cylindrical mask of
# diameter ≈ 0.95·N with a cosine taper, leaves ~50 zero voxels at each
# end of z with a cosine ramp).  Voxel matches detector px at UPS=8 →
# 1.38 µm; scales as 1.38·8/UPS at other UPS.  Physical dataset = 33.92
# mm cube; sample ≈ ⌀32.2 × 32.8 mm cylinder inside.  Defaults
# (--circle-diam=2432 --z-pad=42) match the schematic
# (SAMPLE_D_PX = 2918·UPS, SAMPLE_H_PX = 2972·UPS).
#
# As N grows the GPU-only Radon / Fresnel / Paganin / FBP buffers stop
# fitting a 40 GB A100 — swap step2_radon.py → step2_radon_large.py,
# step3_propagation.py → step3_propagation_large.py, step8_fbp.py →
# step8_fbp_large.py, and (for UPS ≥ 8) step7_paganin.py →
# step7_paganin_large.py.  All *_large variants keep the same rfft/float32
# math but host-chunk the padded fde.  One UPS per job:
#     UPS=16 qsub -v UPS polaris_pipeline_run.sh

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS=4              # ranks per node (= GPUs per node on Polaris)
NTHREADS=4
NDEPTH=8
export NTOTRANKS=$(( NNODES * NRANKS ))

SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Sample dir:  ${SCRIPT_DIR}"
echo "Jobid: $PBS_JOBID"
echo "Running on host: $(hostname)"
echo "Running on nodes: $(cat $PBS_NODEFILE)"
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}"

module use /soft/modulefiles
module load conda
conda activate base

cd "${SCRIPT_DIR}"

# ================== USER KNOBS ==================
UPS=${UPS:-1}
PATH_DATA=${PATH_DATA:-/eagle/APS_IRI/vnikitin/mosaic_brain}

# Physics knobs — kept in sync between step3 (forward Fresnel) and step7
# (Paganin inversion).  DISTANCE is the sample→detector propagation
# distance in metres (near-field regime).
DISTANCE=${DISTANCE:-0.2}

NZCHUNK=${NZCHUNK:-32}                       # z-slices per Radon / FBP call
NPROPCHUNK=${NPROPCHUNK:-8}                  # angles per Fresnel / Paganin batch

NBANKS=${NBANKS:-8}                          # bank files per super-chunk
NTASKS=${NTASKS:-8}                          # parallel workers for read_{projs,slices}_vchunkx
                                             # (steps 3, 7, 8 prefetch reads via VDS+banks pool)

# HDF5 layout policy (iohdf5/layout.py).  Every super-chunk and chunk shape
# in the pipeline comes from these two numbers, so they scale with UPS
# instead of being fixed slice counts:
#
#   MEM_BUDGET_GB   per-rank RAM for the output vchunk PLUS the input
#                   prefetch slab -- sizes the super-chunk.  Derived from
#                   the node so it tracks NRANKS; ~25% is left free for the
#                   NBANKS worker pool, page cache and CUDA host staging.
#   CHUNK_MB        target size of one HDF5 chunk = the op size Lustre
#                   sees -- sizes the chunk.  Settle it with
#                   polaris_test_h5.sh, which sweeps exactly this knob.
#
# NZCHUNK / NPROPCHUNK above stay the *requested* compute-loop batch; each
# step bends its own down to a divisor of the planned super-chunk when the
# budget forces a smaller one (looping in smaller pieces is free; growing
# the buffer is not).
NODE_GB=${NODE_GB:-512}                      # Polaris node RAM
MEM_BUDGET_GB=${MEM_BUDGET_GB:-$(python3 -c "print(f'{${NODE_GB} * 0.75 / ${NRANKS}:.1f}')")}
CHUNK_MB=${CHUNK_MB:-64}

# Optional --vchunks overrides per step ("C0 C1 C2" as a single string).
# Leave empty to take the policy's shape.
VCHUNKS_STEP1=${VCHUNKS_STEP1:-}
VCHUNKS_STEP2=${VCHUNKS_STEP2:-}
VCHUNKS_STEP3=${VCHUNKS_STEP3:-}
# ================================================

vcarg() { local val="$1"; [[ -n "$val" ]] && echo "--vchunks $val"; }

# Passed to every step that writes a banked dataset.
LAYOUT=(--mem-budget "$MEM_BUDGET_GB" --chunk-bytes "$CHUNK_MB")

# HDF5 file locking is left at the default (enabled).  With the
# tomo_info() cache + ALLOC_TIME_EARLY preallocation in
# iohdf5/dxchange_hdf5_chunks.py, there is no metadata-read storm and
# no concurrent chunk-allocation race, so per-file POSIX locks are
# uncontended and safe on Lustre.
MPIEXEC=(mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}"
         --depth="${NDEPTH}" --cpu-bind depth
         --env OMP_NUM_THREADS="${NTHREADS}"
         "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh")

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$NTOTRANKS  NBANKS=$NBANKS  NTASKS=$NTASKS  DISTANCE=${DISTANCE}m ==="
echo "=== layout: mem-budget=${MEM_BUDGET_GB} GiB/rank  chunk-bytes=${CHUNK_MB} MiB ==="
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
      "${PATH_DATA}/model_big${UPS}x/rec"
      "${PATH_DATA}/mosaic_h5"
      "${PATH_DATA}/mosaic_h5_pre")
mkdir -p "${DIRS[@]}"
for d in "${DIRS[@]}"; do
    lfs setstripe -c -1 -S 4M "$d" 2>/dev/null || true
done

# ---------- 0. plan mosaic layout ----------------------------------------
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 1. init.h5 → big{UPS}x.h5 ------------------------------------
# From UPS=16 the super-chunk is thinner than UPS output planes (one plane
# is 9.7 GB), so step1 prints a NOTE and re-reads the input plane
# straddling each seam.  Output is identical either way.
"${MPIEXEC[@]}" \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" "${LAYOUT[@]}" $(vcarg "$VCHUNKS_STEP1")

# ---------- 2. Radon → proj.h5 -------------------------------------------
"${MPIEXEC[@]}" \
    python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        "${LAYOUT[@]}" $(vcarg "$VCHUNKS_STEP2")
# For UPS≥4 swap in step2_radon_large.py (drop --ntasks, it has none).

# ---------- 3. Fresnel → data.h5 -----------------------------------------
"${MPIEXEC[@]}" \
    python step3_propagation.py --ups "$UPS" --path "$PATH_DATA" \
        --distance "$DISTANCE" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        "${LAYOUT[@]}" $(vcarg "$VCHUNKS_STEP3")
# For UPS≥4 swap in step3_propagation_large.py (--chunk-nz, not --npropchunk).

# ---------- 4. data.h5 → mosaic_h5/{z}_{x}.h5 -----------------------------
# --z-pad defaults to (NZ - SAMPLE_H_PX)/2 = 688·UPS (schematic z=0 =
# sample top lands on that data row).  --air-fill=1.0 by default
# (transmission of air) for out-of-bounds tile pixels.
"${MPIEXEC[@]}" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 5. mosaic_h5 → mosaic_h5_pre (dezinger + darkflat + FW) -------
# Per-tile GPU preprocessing, tiles round-robin sharded across ranks.
"${MPIEXEC[@]}" \
    python step5_correct.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK"

# ---------- 6. mosaic_h5_pre/*.h5 → stitched.h5 (tent blend, 180° fold) ---
"${MPIEXEC[@]}" \
    python step6_stitch.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" "${LAYOUT[@]}"

# ---------- 7. stitched.h5 → paganin.h5 (single-distance Paganin) --------
"${MPIEXEC[@]}" \
    python step7_paganin.py --ups "$UPS" --path "$PATH_DATA" \
        --distance "$DISTANCE" --npgnchunk "$NPROPCHUNK" \
        --nbanks "$NBANKS" --ntasks "$NTASKS" "${LAYOUT[@]}"
# For UPS≥8 swap in step7_paganin_large.py (same flags).

# ---------- 8. paganin.h5 → rec.h5 (filtered backprojection) -------------
"${MPIEXEC[@]}" \
    python step8_fbp.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        --filter ramp "${LAYOUT[@]}"
# For UPS≥4 swap in step8_fbp_large.py (same flags).

echo "=== pipeline done ==="
