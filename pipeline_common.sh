#!/bin/bash
# The three pipeline steps, shared by marmoset/ and big_sandstone/.
# Each dataset's run script sets its settings and then calls:
#   stitch <tile arguments>   step 1: raw scans -> model_big1x/stitched.h5
#   paganin                   step 2: stitched.h5 -> model_big1x/paganin.h5
#   fbp                       step 3: paganin.h5 -> model_big1x/rec.h5
# Every step writes its log to $OUT/step1.log, step2.log, step3.log and the
# script stops if a log shows an error.

MM=/home/beams2/VNIKITIN/mosaic_modeling
PY_ENV=/home/beams2/VNIKITIN/miniforge3/envs/holotomocupy/bin   # steps 1, 2
TC_ENV=/home/beams2/VNIKITIN/miniforge3/envs/tomocupy-mpi/bin   # step 3

export PATH=$PY_ENV:$PATH
export HDF5_USE_FILE_LOCKING=FALSE
export MOSAIC_SKIP_MPI_FINALIZE=1    # (mosaic MPI setting)
export MOSAIC_SINGLE_H5=1            # each step writes one plain h5 file
export MOSAIC_ASYNC_WRITE=1          # step 2 writes while it computes
cd $MM
mkdir -p $OUT

ROWS_OPTION=""
if [ -n "$ROWS" ]; then
    ROWS_OPTION="--rows $ROWS"
fi

# stop with a message if a log contains an error
check_log() {
    if grep -q -a -E "Traceback|EXCEPTION|INCOMPLETE|non-zero status" "$1"; then
        echo "FAILED, see $1"
        exit 1
    fi
}

# run a command on all GPUs (one process per GPU)
on_all_gpus() {
    mpirun -H $HOSTS -n $NGPUS --mca io romio341 \
        -x PATH -x HDF5_USE_FILE_LOCKING -x MOSAIC_SKIP_MPI_FINALIZE \
        -x MOSAIC_SINGLE_H5 -x MOSAIC_ASYNC_WRITE \
        ./set_affinity_gpu.sh "$@"
}

stitch() {
    echo "step 1 (stitch + fold) started  $(date)"
    LOG=$OUT/step1.log
    ARGS="--raw-dir $RAW_DIR --prefix $PREFIX --center $CENTER --path $OUT $ROWS_OPTION $*"
    python stitch_fold.py $ARGS --init > $LOG 2>&1 || { echo "FAILED, see $LOG"; exit 1; }
    on_all_gpus python stitch_fold.py $ARGS --worker >> $LOG 2>&1
    python stitch_fold.py $ARGS --finish >> $LOG 2>&1 || { echo "FAILED, see $LOG"; exit 1; }
    check_log $LOG
    echo "step 1 done  $(date)"
}

paganin() {
    echo "step 2 (Paganin) started  $(date)"
    LOG=$OUT/step2.log
    # size of the stitched data: rows and width
    SIZE=$(python -c "import h5py; s = h5py.File('$OUT/model_big1x/stitched.h5', 'r')['exchange/data'].shape; print(s[1], s[2])")
    NZ=${SIZE% *}
    N=${SIZE#* }
    on_all_gpus python step7_paganin_large.py --ups 1 --path $OUT --nz $NZ --n $N \
        --energy $ENERGY --voxelsize $PIXEL --distance $DISTANCE --alpha $ALPHA \
        --pad edge --chunk-nz 1024 --chunk-n 1024 > $LOG 2>&1
    check_log $LOG
    echo "step 2 done  $(date)"
}

fbp() {
    echo "step 3 (FBP) started  $(date)"
    LOG=$OUT/step3.log
    $TC_ENV/mpirun -H $HOSTS -n $NGPUS -x PATH=$TC_ENV:$PATH -x HDF5_USE_FILE_LOCKING \
        ./set_affinity_gpu.sh $TC_ENV/python step8_fbp_fold.py \
        --path $OUT --filter $FILTER --nzslab ${NZSLAB:-4} > $LOG 2>&1
    check_log $LOG
    echo "step 3 done  $(date)"
}
