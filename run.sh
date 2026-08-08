#!/usr/bin/env bash
# End-to-end mosaic-modelling pipeline.
#
# Edit UPS / PATH_DATA / N_GPUS below, then:  bash run.sh
# For the large variant (UPS ≥ 8) swap step2_model_big → step2_model_large
# and drop --nzchunk to 1.  On Polaris use set_affinity_gpu_polaris.sh
# and mpiexec --ppn 4 --depth=8 --cpu-bind depth instead of mpirun.

set -euo pipefail

UPS=1
PATH_DATA=${PATH_DATA:-/data2/brain_sym_mosaic}
N_GPUS=${N_GPUS:-4}                          # total ranks (= total GPUs)

echo "=== UPS=$UPS  PATH_DATA=$PATH_DATA  N_GPUS=$N_GPUS ==="

# 0. Plan the mosaic (schematic PNG + tile-origin txt).
python step0_schematic.py --ups "$UPS" --path "$PATH_DATA"

# 1. init.h5 → big{UPS}x.h5  (bilinear xy + linear z upsample).
mpirun -n "$N_GPUS" set_affinity_gpu.sh \
    python step1_upsample.py --ups "$UPS" --path "$PATH_DATA"

# 2. big{UPS}x.h5 → model_big{UPS}x/{proj.h5, data.h5}
#    (GPU-only Tomo variant; swap for step2_model_large.py if UPS ≥ 8).
mpirun -n "$N_GPUS" set_affinity_gpu.sh \
    python step2_model_big.py --ups "$UPS" --path "$PATH_DATA" \
                              --nzchunk 32 --npropchunk 8

# 3. data.h5 → mosaic_h5/{z}_{x}.h5   (MPI-parallel over tiles; CPU only).
mpirun -n "$N_GPUS" \
    python step3_extract.py --ups "$UPS" --path "$PATH_DATA"

echo "=== pipeline done ==="
