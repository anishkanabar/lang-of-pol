#!/bin/bash

ARGS=("$@")
for i in {0..700}
do
  echo "INFO: Running trial $i"
  rm -r results/train_wav2vec2_char/seed_1986/trial_18
  python "${ARGS[@]}"
done
