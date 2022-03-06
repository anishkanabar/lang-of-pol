#!/bin/bash

TRAIN=$1
PARAMS=$2
for i in {0..1000}
do
  echo "INFO: Running trial $i"
  results_dir=results/train_wav2vec2_char/seed_1987/trial_20
  if [ -d "$results_dir" ]; then
    rm -r "$results_dir"
  fi
  python "$TRAIN" "$PARAMS" --nonfinite_patience=0
done
