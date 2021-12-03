#!/bin/bash
# BRIEF:
# Creates a new pip environment from the given requirements list.
# USAGE:
# sh create_env_speechbrain.sh <path_to_new_env> <path_to_sb>

cd $(dirname $0)
env_dir="$1"
sb_dir="$2"

if [ -z "$env_dir" ]; then
    echo "Missing argument. Usage:"
    echo "sh create_env_speechbrain.sh <path_to_new_env> <path_to_sb>"
    exit 1

elif [ -z "$sb_dir" ]; then
    echo "Missing argument. Usage:"
    echo "sh create_env_speechbrain.sh <path_to_new_env> <path_to_sb>"
    exit 1

elif [ -e "$env_dir" ]; then
    echo "Using existing pip env."

else
    echo "Creating pip env"
    python3 -m venv "$env_dir"
    env_parent=$(dirname "$env_dir")
    cp --no-clobber 'ENV_README.txt' "$env_parent/README.txt"
    cp --no-clobber 'requirements.txt' "$env_parent/requirements.txt"
    source "$env_dir/bin/activate"
    pip install -r requirements.txt
    pip install --editable "$sb_dir"
fi

