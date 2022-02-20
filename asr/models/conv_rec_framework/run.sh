#!/bin/bash

# Retrieve command line args
all_args=("$@")
PYSCRIPT=$1
PYNAME=`basename $PYSCRIPT .py`
OTHER_ARGS=("${all_args[@]:1}")

usage() {
    echo "Usage:" >&2
    echo "bash run.sh <path/to/script.py> [args...]" >&2
}

# Validate command line args
if [ ! -f "$PYSCRIPT" ] ; then
    usage
    exit 1
fi

# Define job params
JOB_NAME="pyanode-$PYNAME"
TIMEOUT="00:30:00"
ACCOUNT="pi-graziul"
GPUS=1
MEM="24G"
PARTITION="gpu"
OUTPUT_DIR="/scratch/midway3/`whoami`/slurm"

# Setup environment
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

# Link to libsndfile, which isnt available on rcc compute nodes
if [[ ! "$LD_LIBRARY_PATH" == *"pyannote"* ]]; then
    LN_PATH_1=/home/`whoami`/.conda/envs/pyannote/lib/libsndfile.a
    LN_PATH_2=/home/`whoami`/.conda/envs/pyannote/lib/libsndfile.so
    LN_PATH_3=/home/`whoami`/.conda/envs/pyannote/lib/libsndfile.so.1
    LN_PATH_4=/home/`whoami`/.conda/envs/pyannote/lib/libsndfile.so.1.0.31
    export LD_LIBRARY_PATH=$LN_PATH_1:$LN_PATH_2:$LN_PATH_3:$LN_PATH_4:$LD_LIBRARY_PATH
fi

# Run actual task
# ntasks-per-node is 1 because we want 1 'main' process.
srun --job-name "$JOB_NAME" \
    --time "$TIMEOUT" \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --mem-per-cpu "$MEM" \
    --gres "gpu:$GPUS" \
    --nodes "1" \
    --ntasks-per-node "1" \
    --output "$OUTPUT_DIR/%j.%N.stdout" \
    --error "$OUTPUT_DIR/%j.%N.stderr" \
    python "$PYSCRIPT" "${OTHER_ARGS[@]}"
 
