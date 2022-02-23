#!/bin/bash

ARGS=("$@")
for i in {0..200}
do
  echo "INFO: Running trial $i"
  python "${ARGS[@]}"
done
