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
# For UPS ≥ 4 the GPU-only Radon / Fresnel buffers no longer fit on a
# 40 GB A100 — swap step2_radon.py → step2_radon_large.py, step3_propagation.py
# → step3_propagation_large.py, step8_fbp.py → step8_fbp_large.py, and
# (for UPS ≥ 8) step7_paganin.py → step7_paganin_large.py.  All *_large
# variants keep the same rfft/float32 math but host-chunk the padded fde.

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
# Optional --vchunks overrides per step ("C0 C1 C2" as a single string).
VCHUNKS_STEP1=${VCHUNKS_STEP1:-}
VCHUNKS_STEP2=${VCHUNKS_STEP2:-}
VCHUNKS_STEP3=${VCHUNKS_STEP3:-}
# ================================================

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$NTOTRANKS  NBANKS=$NBANKS  NTASKS=$NTASKS  DISTANCE=${DISTANCE}m ==="

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

vcarg() { local val="$1"; [[ -n "$val" ]] && echo "--vchunks $val"; }

# HDF5 file locking is left at the default (enabled).  With the
# tomo_info() cache + ALLOC_TIME_EARLY preallocation in
# iohdf5/dxchange_hdf5_chunks.py, there is no metadata-read storm and
# no concurrent chunk-allocation race, so per-file POSIX locks are
# uncontended and safe on Lustre.
MPIEXEC=(mpiexec -n "${NTOTRANKS}" --ppn "${NRANKS}"
         --depth="${NDEPTH}" --cpu-bind depth
         --env OMP_NUM_THREADS="${NTHREADS}"
         "${SCRIPT_DIR}/set_affinity_gpu_polaris.sh")

# ---------- 0. plan mosaic layout ----------------------------------------
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 1. init.h5 → big{UPS}x.h5 ------------------------------------
"${MPIEXEC[@]}" \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS" $(vcarg "$VCHUNKS_STEP1")

# ---------- 2. Radon → proj.h5 -------------------------------------------
# step2_radon.py: GPU-only TomoReal (rfft/float32); fits UPS ≤ 4 on a 40 GB
# GPU.  For UPS ≥ 4 use step2_radon_large.py below (TomoLargeReal, host-
# chunked; halved host fde vs the old complex64 TomoLarge).
"${MPIEXEC[@]}" \
    python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        $(vcarg "$VCHUNKS_STEP2")
# For UPS≥4 swap in step2_radon_large.py (see README).

# ---------- 3. Fresnel → data.h5 -----------------------------------------
"${MPIEXEC[@]}" \
    python step3_propagation.py --ups "$UPS" --path "$PATH_DATA" \
        --distance "$DISTANCE" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" \
        $(vcarg "$VCHUNKS_STEP3")
# For UPS≥4 swap in step3_propagation_large.py (see README).

# ---------- 4. data.h5 → mosaic_h5/{z}_{x}.h5 -----------------------------
# --z-pad defaults to (NZ - SAMPLE_H_PX)/2 = (4096·UPS - 2720·UPS)/2 = 688·UPS
# (schematic z=0 = sample top lands on that data row).  --air-fill=1.0 by
# default (transmission of air) for out-of-bounds tile pixels.
"${MPIEXEC[@]}" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

# ---------- 5. mosaic_h5 → mosaic_h5_pre  (dezinger + darkflat + FW) ------
# Per-tile GPU preprocessing.  Tiles are round-robin sharded across ranks;
# one GPU per rank via set_affinity_gpu_polaris.sh.
"${MPIEXEC[@]}" \
    python step5_correct.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK"

# ---------- 6. mosaic_h5_pre/*.h5 → stitched.h5 (tent blend, 180° fold) ---
"${MPIEXEC[@]}" \
    python step6_stitch.py --ups "$UPS" --path "$PATH_DATA" \
        --nbanks "$NBANKS"

# ---------- 7. stitched.h5 → paganin.h5 (single-distance Paganin) --------
# step7_paganin.py: GPU-only 2-D FFT per θ batch; fits UPS ≤ 4 on a 40 GB A100.
# For UPS ≥ 8 swap in step7_paganin_large.py (host-chunked PaganinLarge).
"${MPIEXEC[@]}" \
    python step7_paganin.py --ups "$UPS" --path "$PATH_DATA" \
        --distance "$DISTANCE" \
        --npgnchunk "$NPROPCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS"
# For UPS≥8 swap in step7_paganin_large.py (see README).

# ---------- 8. paganin.h5 → rec.h5 (filtered backprojection) -------------
# step8_fbp.py: GPU-only TomoReal.RT; fits UPS ≤ 2 on a 40 GB A100 at
# NZCHUNK=32.  For UPS ≥ 4 swap in step8_fbp_large.py (host-chunked
# TomoLargeReal.RT), or drop NZCHUNK for step8 only.
"${MPIEXEC[@]}" \
    python step8_fbp.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" --ntasks "$NTASKS" --filter ramp
# For UPS≥4 swap in step8_fbp_large.py (see README).

echo "=== pipeline done ==="
