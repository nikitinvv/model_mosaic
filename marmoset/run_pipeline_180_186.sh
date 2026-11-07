#!/bin/bash
# Marmoset brain, scans 180-186: raw data -> 3D reconstruction.
#
#   step 1  stitch_fold.py          dark/flat, dezinger, ring removal (fw),
#                                   stitch the 7 scans, fold 360 -> 180 deg
#   step 2  step7_paganin_large.py  Paganin phase retrieval
#   step 3  step8_fbp_fold.py       filtered back-projection (tomocupy)
#
# Run on tomo2:
#   bash run_pipeline_180_186.sh
#
# Any setting below can be given on the command line, for example a test
# on 64 slices with one machine:
#   ROWS="6464 6528" OUT=/data2/19BM/2026-11-Boyer-1026094_rec/test HOSTS=tomo5:4 NGPUS=4 bash run_pipeline_180_186.sh
# or only steps 2 and 3 with another Paganin alpha:
#   STEPS="2 3" ALPHA=1e-4 bash run_pipeline_180_186.sh

# ---------------------------- settings ----------------------------
RAW_DIR=${RAW_DIR:-/data2/19BM/2026-11-Boyer-1026094}
PREFIX=${PREFIX:-Marmoset_test_1612mm_vertical_full_}
SCANS=${SCANS:-"180 181 182 183 184 185 186"}   # top to bottom
STEP=${STEP:-2051}            # rows between neighbouring scans
CENTER=${CENTER:-876}         # rotation axis (as tomocupy --rotation-axis)
ENERGY=${ENERGY:-70}          # keV
PIXEL=${PIXEL:-1.9e-6}        # m
DISTANCE=${DISTANCE:-1.62}    # m
ALPHA=${ALPHA:-8e-5}          # Paganin alpha
FILTER=${FILTER:-ramp}        # FBP filter
OUT=${OUT:-/data2/19BM/2026-11-Boyer-1026094_rec/Marmoset_test_1612mm_vertical_full_180_186_stitched}
ROWS=${ROWS:-}                # empty = all rows; "R0 R1" = only those rows
HOSTS=${HOSTS:-tomo5:4,tomo2:3,tomo3:2,tomo4:2}  # machine:number of GPUs
NGPUS=${NGPUS:-11}            # total number of GPUs in HOSTS
STEPS=${STEPS:-"1 2 3"}       # which steps to run
# ------------------------------------------------------------------

source "$(dirname "$0")/../pipeline_common.sh"

if [[ $STEPS == *1* ]]; then
    stitch --scans $SCANS --step $STEP
fi
if [[ $STEPS == *2* ]]; then
    paganin
fi
if [[ $STEPS == *3* ]]; then
    fbp
fi
echo "finished: $OUT/model_big1x/rec.h5  $(date)"
