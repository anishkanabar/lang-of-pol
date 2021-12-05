#!/bin/bash
# BRIEF:
# Sets up a shared environment and dispatches a neural net training task to the cluster.
# USAGE:
# sh run-training.sh <path_to_conda_env>
# Run this from the current repo directory:

env_dir="$1"
env_parent=$(dirname "$env_dir")
local_flag=false

print_usage() {
  echo "Usage: sh run-training.sh [-l] <path_to_conda_env>"
}

while getopts 'l' flag; do
  case "${flag}" in
    l) local_flag=true;;
    *) print_usage
       exit 1 ;;
  esac
done

if [ -z "$env_parent" ]; then
    echo "Missing argument."
    print_usage
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

sh env/create_env.sh "$env_dir"

if [ $? -eq 0 ]; then
    if [ $local_flag ]; then
        sh run-training.job "$env_dir" 
    else
        sbatch run-training.job "$env_dir"
    fi
else
    echo "Failed to create conda env. Exiting."
    exit 1
fi
