#!/bin/bash
# BRIEF:
# Creates a new conda environment from the given requirements list.
# USAGE:
# sh create_env.sh <path_to_new_env>

env_dir="$1"

if [ -z "$env_dir" ]; then
    echo "Missing argument. Usage:"
    echo "sh create_env.sh <path_to_new_env>"
    exit 1
elif [ -e "$env_dir" ]; then
    echo "Conda env exists. Skipping."
else
    echo "Copying env config into shared location"
    env_parent=$(dirname "$env_dir")
    cp --no-clobber 'requirements.txt' "$env_parent/requirements.txt"
    cp --no-clobber 'ENV_README.txt' "$env_parent/README.txt"
    echo "Creating conda env"
    conda create -y -c conda-forge -p "$env_dir" --file requirements.txt
fi
