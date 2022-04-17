#!/bin/bash

# Validate command line args: cluster
CLUSTER="rcc"

# Define cluster-specific params
OUTPUT_DIR="/project/graziul/ra/`whoami`/slurm_output"
TIMEOUT="01:00:00"
PARTITION="caslake"
ACCOUNT="pi-graziul"
# Trying nodes in order in case some have weird cuda BS

# Define regular params
JOB_NAME="test-data"
MAIL_USER="%u@uchicago.edu"
MAIL_TYPE="all"
OUTPUT="$OUTPUT_DIR/%j.%N.stdout"
ERROR="$OUTPUT_DIR/%j.%N.stderr"
NODES="1"
NTASKS="1"
MEM_PER_CPU="24G" 

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ "$CLUSTER" = "rcc" ]; then
    # Link to libsndfile, which isnt available on rcc compute nodes
    if [[ ! "$LD_LIBRARY_PATH" == *"soundfile"* ]]; then
        LN_PATH=/home/`whoami`/.conda/envs/soundfile/lib
        export LD_LIBRARY_PATH=$LN_PATH:$LD_LIBRARY_PATH
    fi
    # Link to ffmpeg, which isnt available on rcc compute nodes
    if [[ ! "$PATH" == *"ffmpeg"* ]]; then
        BIN_PATH=/home/`whoami`/.conda/envs/ffmpeg/bin
        export PATH=$BIN_PATH:$PATH
    fi
fi

# this node is friendly --nodelist "midway3-0277" \
srun --job-name "$JOB_NAME" \
        --mail-user $MAIL_USER \
        --mail-type $MAIL_TYPE \
        --output "$OUTPUT" \
        --error "$ERROR" \
        --partition "$PARTITION" \
        --nodes "$NODES" \
        --ntasks $NTASKS \
        --mem-per-cpu "$MEM_PER_CPU" \
        --time "$TIMEOUT" \
        --account "$ACCOUNT" \
        python test.py

