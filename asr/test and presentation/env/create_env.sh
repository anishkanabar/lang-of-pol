#!/bin/bash
# BRIEF:
# Creates a new conda environment from the given requirements list.
# USAGE:
# sh create_env.sh <path_to_new_env>

cd $(dirname $0)
env_dir="$1"

if [ -z "$env_dir" ]; then
    echo "Missing argument. Usage:"
    echo "sh create_env.sh <path_to_new_env>"
    exit 1

elif [ -e "$env_dir" ]; then
    echo "Using existing conda env."

else
    echo "Creating conda env"
    conda create -y -c conda-forge -p "$env_dir"
    env_parent=$(dirname "$env_dir")
    cp --no-clobber 'ENV_README.txt' "$env_parent/README.txt"
    if [ -f "requirements.txt" ]; then
        cp --no-clobber 'requirements.txt' "$env_parent/requirements.txt"
        conda install -y -c conda-forge -p "$env_dir" --file requirements.txt
    else
        conda install -y -c conda-forge -p "$env_dir" tensorflow keras librosa pandas numpy
    fi
fi

