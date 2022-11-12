#!/bin/bash
unset XDG_RUNTIME_DIR
srun -p general \
     --time 02:30:00 \
     --pty \
     --nodes 1 \
     --ntasks 1 \
     --mem-per-cpu 10000 \
     jupyter-notebook --no-browser