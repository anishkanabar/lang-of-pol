#!/bin/bash
# BRIEF:
# Sets up a shared environment and dispatches a neural net training task to the cluster.
# USAGE:
# sh run-training.sh <path_to_conda_env>
# Run this from the current repo directory:

scale_flag="$1"
dataset="$2"
env_dir="$3"
env_parent=$(dirname "$env_dir")

print_usage() {
  echo "Usage: sh run-training.sh <local|cluster> <path_to_conda_env>"
}

if [ -z "$scale_flag" ] || [ -z "$env_dir" ] || [ -z "$dataset" ]; then
    print_usage
    exit 1
fi

sh env/create_env.sh "$env_dir" "$scale_flag"

if [ $? -eq 0 ]; then
    if [ "$scale_flag" == "local" ]; then
        logs_dir='/Users/eric/Documents/Work/PoliceBroadcasts/output_logs'
        dataset_dir='/Users/eric/Documents/Work/PoliceBroadcasts/FakeData'
        sh run-training.job "local" "$dataset" "$env_dir" "$dataset_dir" "$logs_dir"
    else
        logs_dir='/project/graziul/ra/echandler'
        dataset_dir='/project/graziul/transcripts'
        sbatch run-training.job "cluster" "$datset" "$env_dir" "$dataset_dir" "$logs_dir"
    fi
else
    echo "Failed to create conda env. Exiting."
    exit 1
fi
