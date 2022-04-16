#!/bin/bash

# Retrieve command line args
all_args=("$@")
TRAINPY=$1
HPARAMS=$2

# Validate command line args
usage() {
    echo "Usage:" >&2
    echo "sh run-training.sh <path/to/train.py> <path/to/params.yaml> " >&2
    echo "Overrides are named args matching a hparam key" >&2
}

# Validate command line args: cluster
if [[ `hostname` == *"midway3"* ]]; then
    CLUSTER="rcc"
elif [[ `hostname` == *"fe"* ]]; then
    CLUSTER="ai"
elif [[ `hostname` == *"ttic"* ]]; then
    CLUSTER="ttic"
else
    usage
    exit 1
fi

# Validate command line args: script and param file
if [ ! -f "$TRAINPY" ] || [ ! -f "$HPARAMS" ]; then
    usage
    exit 1
fi

# Define cluster-specific params
if [ "$CLUSTER" = "rcc" ]; then
    OUTPUT_DIR="/project/graziul/ra/`whoami`/slurm_output"
    TIMEOUT="06:00:00"
    PARTITION="gpu"
    ACCOUNT="pi-graziul"
    # Trying nodes in order in case some have weird cuda BS
elif [ "$CLUSTER" = "ai" ]; then
    OUTPUT_DIR="/home/`whoami`/slurm_output"
    TIMEOUT="03:59:00"   # 4 hours is the maximum on AI cluster
    PARTITION="general"
elif [ "$CLUSTER" = "ttic" ]; then
    OUTPUT_DIR="/scratch/`whoami`/slurm"
    TIMEOUT="01:00:00"   # 4 hours is the maximum on AI cluster
    PARTITION="gpu"
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

if [ "$CLUSTER" = "rcc" ]; then
    # this node is friendly --nodelist "midway3-0277" \
    # run with --nonfinite_patience=0 as last arg to shortcut
    srun --job-name "$JOB_NAME" \
            --mail-user $MAIL_USER \
            --mail-type $MAIL_TYPE \
            --output "$OUTPUT" \
            --error "$ERROR" \
            --partition "$PARTITION" \
            --nodes "$NODES" \
            --nodelist "midway3-0277" \
            --gpus $GPUS \
            --ntasks $NTASKS \
            --ntasks-per-gpu $GPU_TASKS \
            --mem-per-cpu "$MEM_PER_CPU" \
            --time "$TIMEOUT" \
            --account "$ACCOUNT" \
            python "$TRAINPY" "$HPARAMS" 
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
            python "$TRAINPY" "$HPARAMS"
elif [ "$CLUSTER" = "ttic" ]; then
    srun -c "$GPUS" \
            --job-name "$JOB_NAME" \
            --nodes $NODES \
            --nodelist "gpu-g3" \
            --ntasks $NTASKS \
            --time "$TIMEOUT" \
            --partition "$PARTITION" \
            python "$TRAINPY" "$HPARAMS"
fi

