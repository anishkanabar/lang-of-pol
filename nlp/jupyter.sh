#!/bin/bash
unset XDG_RUNTIME_DIR
#    --gpus 1 \
srun --time 02:30:00 \
     --pty \
     --nodes 1 \
     --ntasks 1 \
     --account pi-graziul \
     --mem-per-cpu 10000 \
     jupyter-notebook --no-browser
