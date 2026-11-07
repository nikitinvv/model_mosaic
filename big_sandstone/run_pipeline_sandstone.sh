#!/bin/bash
# Big sandstone, scans 082-103 (11 heights x 2 columns): raw -> reconstruction.
#   step 1: stitch + fold   step 2: Paganin   step 3: FBP
# Run on tomo2:  bash run_pipeline_sandstone.sh   (needs ~78 TB on /data2)
# Test:  ROWS="2000 2064" OUT=/data2/.../test HOSTS=tomo5:4 NGPUS=4 bash run_pipeline_sandstone.sh

# ---------------------------- settings ----------------------------
RAW_DIR=${RAW_DIR:-/data2/19BM/2026-11-Boyer-1026094}
PREFIX=${PREFIX:-big_sandstone_full_}
CENTER=${CENTER:-944}         # rotation axis (as tomocupy --rotation-axis)
ENERGY=${ENERGY:-70}          # keV
PIXEL=${PIXEL:-1.889e-6}      # m
DISTANCE=${DISTANCE:-0.6}     # m
ALPHA=${ALPHA:-1e-4}          # Paganin alpha
FILTER=${FILTER:-ramp}        # FBP filter
NZSLAB=2                      # slices per FBP call (the data is wide)
OUT=${OUT:-/data2/19BM/2026-11-Boyer-1026094_rec/big_sandstone_full_082_103_stitched}
ROWS=${ROWS:-}                # empty = all rows; "R0 R1" = only those rows
HOSTS=${HOSTS:-tomo5:4,tomo2:3,tomo3:2,tomo4:2}  # machine:number of GPUs
NGPUS=${NGPUS:-11}            # total number of GPUs in HOSTS
STEPS=${STEPS:-"1 2 3"}       # which steps to run

# scan:row:column of each scan in the stitched image (pixels)
TILES="082:0:0      083:0:3176
       084:2118:0   085:2118:3176
       086:4235:0   087:4235:3176
       088:6352:0   089:6352:3176
       090:8469:0   091:8469:3176
       092:10587:0  093:10587:3176
       094:12704:0  095:12704:3176
       096:14821:0  097:14821:3176
       098:16939:0  099:16939:3176
       100:19056:0  101:19056:3176
       102:21173:0  103:21173:3176"
# bright ratio per scan: 0.617 inner, 1 outer
BRIGHT="0.617 1  0.617 1  0.617 1  0.617 1  0.617 1  0.617 1
        0.617 1  0.617 1  0.617 1  0.617 1  0.617 1"
THETA="0 359.9118"            # angles (the files store zeros)
# ------------------------------------------------------------------

source "$(dirname "$0")/../pipeline_common.sh"

if [[ $STEPS == *1* ]]; then
    stitch --tiles $TILES --bright-ratio $BRIGHT --theta $THETA
fi
if [[ $STEPS == *2* ]]; then
    paganin
fi
if [[ $STEPS == *3* ]]; then
    fbp
fi
echo "finished: $OUT/model_big1x/rec.h5  $(date)"
