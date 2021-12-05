#!/bin/bash
# BRIEF:
# Creates a new conda environment from the given requirements list.
# USAGE:
# sh create_env.sh <path_to_new_env>

cd $(dirname $0)
env_dir="$1"
scale_flag="$2"

print_usage() {
    echo "Usage sh create_env.sh <path_to_new_env> <local|cluster>"
}

if [ -z "$env_dir" ]; then
    print_usage
    exit 1

elif [ -z "$scale_flag" ]; then
    print_usage
    exit 1

elif [ -e "$env_dir" ]; then
    echo "Using existing conda env."

else
    env_parent=$(dirname "$env_dir")
    if [ ! -e "$env_parent" ]; then
        echo "Environment directory not found. Create?"
        echo "\t$env_dir"
        echo "Type 1 for Yes. 2 for No"
        select yn in "Yes" "No"; do
            case $yn in
                Yes) mkdir -p "$env_parent"; break;;
                No) exit 1;;
            esac
        done
    fi
    echo "Creating conda env"
    conda create -y -c conda-forge -p "$env_dir"
    cp -n 'ENV_README.txt' "$env_parent/README.txt"
    if [ -f "requirements.txt" ]; then
        if [ "$scale_flag" == "cluster" ]; then
            cp -n 'requirements.txt' "$env_parent/requirements.txt"
        fi
        conda install -y -c conda-forge -p "$env_dir" --file requirements.txt
    elif [ "$scale_flag" == "cluster" ]; then
        # numpy version is specified for compatability with deepspeech2 + tensorflow-gpu
        conda install -y -p "$env_dir" "tensorflow-gpu==2.4.1" keras-gpu pandas "numpy==1.19.2"
        conda install -y -c conda-forge -p "$env_dir" librosa 
    else
        # numpy version is specified for compatability with deepspeech2 + tensorflow-gpu
        conda install -y -p "$env_dir" "tensorflow==2.4.1" keras pandas "numpy==1.19.2"
        conda install -y -c conda-forge -p "$env_dir" librosa 
    fi
fi

