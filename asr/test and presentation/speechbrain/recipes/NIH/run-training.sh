#!/bin/bash

# Retrieve command line args
CLUSTER=$1

# Validate command line args
usage() {
    echo "Usage:" >&2
    echo "sh run-training.sh <rcc|ai>" >&2
}
if [ "$CLUSTER" != "rcc" ] && [ "$CLUSTER" != "ai" ]; then
    usage
    exit 1
fi

# Define cluster-specific params
if [ "$CLUSTER" = "rcc" ]; then
    OUTPUT_DIR="/project/graziul/ra/`whoami`/slurm_output"
    TIMEOUT="18:00:00"
    PARTITION="gpu"
    ACCOUNT="pi-graziul"
elif [ "$CLUSTER" = "ai" ]; then
    OUTPUT_DIR="/home/`whoami`/slurm_output"
    TIMEOUT="03:59:00"   # 4 hours is the maximum on AI cluster
    PARTITION="general"
fi

# Define regular params
JOB_NAME="train-sb"
MAIL_USER="%u@uchicago.edu"
MAIL_TYPE="all"
OUTPUT="$OUTPUT_DIR/%j.%N.stdout"
ERROR="$OUTPUT_DIR/%j.%N.stderr"
NODES="1"
GPUS="1"
NTASKS="1"
GPU_TASKS="1"
MEM_PER_CPU="24G"

# Edit this to train different model component!
CMD="python Tokenizer/train.py Tokenizer/hparams/tokenizer_bpe5000.yaml"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

if [ "$CLUSTER" = "rcc" ]; then
    srun --job-name "$JOB_NAME" \
            --mail-user $MAIL_USER \
            --mail-type $MAIL_TYPE \
            --output "$OUTPUT" \
            --error "$ERROR" \
            --partition "$PARTITION" \
            --nodes $NODES \
            --gpus $GPUS \
            --ntasks $NTASKS \
            --ntasks-per-gpu $GPU_TASKS \
            --mem-per-cpu "$MEM_PER_CPU" \
            --time "$TIMEOUT" \
            --account "$ACCOUNT" \
            "$CMD"
elif [ "$CLUSTER" = "ai" ]; then
    srun --job-name "$JOB_NAME" \
            --mail-user $MAIL_USER \
            --mail-type $MAIL_TYPE \
            --output "$OUTPUT" \
            --error "$ERROR" \
            --partition "$PARTITION" \
            --nodes $NODES \
            --gpus $GPUS \
            --ntasks $NTASKS \
            --gpus-per-task $GPU_TASKS \
            --mem-per-cpu "$MEM_PER_CPU" \
            --time "$TIMEOUT" \
            "$CMD"
fi
     
