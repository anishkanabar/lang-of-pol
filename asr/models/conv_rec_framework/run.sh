#!/bin/bash

# Retrieve command line args
all_args=("$@")
PYSCRIPT=$1
PYNAME=`basename $PYSCRIPT .py`
OTHER_ARGS=("${all_args[@]:1}")

usage() {
    echo "Usage:" >&2
    echo "bash run.sh <path/to/script.py>" >&2
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
PARTITION="gpu"
OUTPUT_DIR="/scratch/midway3/`whoami`/slurm"

# Setup environment
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

# Link to libsndfile, which isnt available on rcc compute nodes
if [[ ! "$LD_LIBRARY_PATH" == *"soundfile"* ]]; then
    LN_PATH=/home/`whoami`/.conda/envs/soundfile/lib
    export LD_LIBRARY_PATH=$LN_PATH:$LD_LIBRARY_PATH
fi

# Run actual task
srun --job-name "$JOB_NAME" \
    --time "$TIMEOUT" \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --output "$OUTPUT_DIR/%j.%N.stdout" \
    --error "$OUTPUT_DIR/%j.%N.stderr" \
    python "$PYSCRIPT" "${OTHER_ARGS[@]}"
 
