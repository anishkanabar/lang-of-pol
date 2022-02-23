#!/bin/bash

ARGS=("$@")
for i in {0..100}
do
  echo "INFO: Running trial $i"
  python "${ARGS[@]}"
done
