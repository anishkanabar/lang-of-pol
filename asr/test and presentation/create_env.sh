#!/bin/bash
# BRIEF:
# Creates a new conda environment from the given requirements list.
# USAGE:
# sh create_env.sh
# Run this IN THE DIRECTORY where you want to create the env.
# Generally, we copy this script into a shared directory and run it there.
conda create -c conda-forge -p ./tensorflow_env --file requirements.txt
