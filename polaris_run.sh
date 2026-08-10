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
# Submit:  qsub polaris_run.sh
#
# Writes VDS+banks h5 stores under $PATH_DATA:
#   init.h5, big{UPS}x.h5, model_big{UPS}x/{proj.h5, data.h5}, mosaic_h5/*
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
# For UPS ≥ 4 swap step2_radon.py → step2_radon_large.py (host-chunked
# TomoLargeReal — rfft/float32) and step3_fresnel.py → step3_fresnel_large.py.

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
UPS=2 #${UPS:-1}
PATH_DATA=${PATH_DATA:-/eagle/APS_IRI/vnikitin/mosaic_brain}

NZCHUNK=${NZCHUNK:-32}                       # z-slices per Radon call
NPROPCHUNK=${NPROPCHUNK:-8}                  # angles per Fresnel batch

NBANKS=${NBANKS:-8}                          # bank files per super-chunk
BIG_VCHUNKS=${BIG_VCHUNKS:-}
PROJ_VCHUNKS=${PROJ_VCHUNKS:-}
DATA_VCHUNKS=${DATA_VCHUNKS:-}
# ================================================

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$NTOTRANKS  NBANKS=$NBANKS ==="

# Lustre striping (all OSTs, 4 MB stripes) on every dir that will hold
# bank files.  New files inherit this.
mkdir -p "${PATH_DATA}" \
         "${PATH_DATA}/init"                    \
         "${PATH_DATA}/big${UPS}x"              \
         "${PATH_DATA}/model_big${UPS}x"        \
         "${PATH_DATA}/model_big${UPS}x/proj"   \
         "${PATH_DATA}/model_big${UPS}x/data"
for d in "${PATH_DATA}" \
         "${PATH_DATA}/init" \
         "${PATH_DATA}/big${UPS}x" \
         "${PATH_DATA}/model_big${UPS}x" \
         "${PATH_DATA}/model_big${UPS}x/proj" \
         "${PATH_DATA}/model_big${UPS}x/data"; do
    lfs setstripe -c -1 -S 4M "$d" 2>/dev/null || true
done

vcarg() { local name="$1"; local val="$2"; [[ -n "$val" ]] && echo "--${name}-vchunks $val"; }

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
        --nbanks "$NBANKS" $(vcarg big "$BIG_VCHUNKS")

# ---------- 2. Radon → proj.h5 -------------------------------------------
# step2_radon.py: GPU-only TomoReal (rfft/float32); fits UPS ≤ 4 on a 40 GB
# GPU.  For UPS ≥ 4 use step2_radon_large.py below (TomoLargeReal, host-
# chunked; halved host fde vs the old complex64 TomoLarge).
"${MPIEXEC[@]}" \
    python step2_radon.py --ups "$UPS" --path "$PATH_DATA" \
        --nzchunk "$NZCHUNK" --nbanks "$NBANKS" \
        $(vcarg proj "$PROJ_VCHUNKS")
# UPS ≥ 4 (host-chunked TomoLargeReal; chunks auto-picked from --gpu-budget-gb).
# "${MPIEXEC[@]}" \
#     python step2_radon_large.py --ups "$UPS" --path "$PATH_DATA" \
#         --nzchunk 1 --nbanks "$NBANKS" \
#         $(vcarg proj "$PROJ_VCHUNKS")

# ---------- 3. Fresnel → data.h5 -----------------------------------------
"${MPIEXEC[@]}" \
    python step3_fresnel.py --ups "$UPS" --path "$PATH_DATA" \
        --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" \
        $(vcarg data "$DATA_VCHUNKS")
# UPS ≥ 4 (host-chunked PropagationLarge; chunks auto-picked from --gpu-budget-gb):
# "${MPIEXEC[@]}" \
#     python step3_fresnel_large.py --ups "$UPS" --path "$PATH_DATA" \
#         --npropchunk "$NPROPCHUNK" --nbanks "$NBANKS" \
#         $(vcarg data "$DATA_VCHUNKS")

# ---------- 4. data.h5 → mosaic_h5/{z}_{x}.h5 -----------------------------
# --z-pad defaults to (NZ - SAMPLE_H_PX)/2 = (4096·UPS - 2720·UPS)/2 = 688·UPS
# (schematic z=0 = sample top lands on that data row).  --air-fill=1.0 by
# default (transmission of air) for out-of-bounds tile pixels.
"${MPIEXEC[@]}" \
    python step4_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
