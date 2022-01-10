#!/bin/bash

OUTPUT_DIR="/home/`whoami`/slurm_output"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
fi

srun --job-name train-asr \
     --mail-user "`whoami`@uchicago.edu" \
     --mail-type all \
     --output "$OUTPUT_DIR/job_1.stdout" \
     --error "$OUTPUT_DIR/job_1.stderr" \
     --partition general \
     --nodes 1 \
     --gpus 1 \
     --ntasks 1 \
     --gpus-per-task 1 \
     --mem-per-cpu 24G \
     --time 03:59:00 \
     sh run-training-ai.job

     
