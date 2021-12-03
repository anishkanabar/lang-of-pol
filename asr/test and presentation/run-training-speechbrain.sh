#!/bin/bash
# BRIEF:
# Sets up a shared environment and dispatches a neural net training task to the cluster.
# USAGE:
# sh run-training-speechbrain.sh <path_to_pip_env>
# Run this from the current repo directory:

env_dir="$1"
env_parent=$(dirname "$env_dir")

if [ -z "$env_parent" ]; then
    echo "Missing argument. Usage:"
    echo "sh run-training-speechbrain.sh <path_to_pip_env>"
    exit 1
elif [ ! -e "$env_parent" ]; then
    echo "Environment directory not found. Create?"
    echo "\t$env_dir"
    echo "Type 1 for Yes. 2 for No"
    select yn in "Yes" "No"; do
        case $yn in
            Yes) mkdir -p "$env_parent"; break;;
            No) exit 0;;
        esac
    done
fi

sh env/create_env_speechbrain.sh "$env_dir" "$(pwd)/speechbrain"

if [ $? -eq 0 ]; then
    sbatch run-training-speechbrain.job "$env_dir"
else
    echo "Failed to create pip env. Exiting."
    exit 1
fi
