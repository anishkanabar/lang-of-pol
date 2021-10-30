#!/bin/bash
# BRIEF:
# Sets up a shared environment and dispatches a neural net training task to the cluster.
# USAGE:
# sh run-training.sh
# Run this from the current repo directory:

this_dir=$(pwd)
env_dir='/project/graziul/ra/team_asr/environments'
bashrc='conda_bashrc'
reqs='requirements.txt'
create='create_env.sh'

cp --no-clobber $bashrc "$env_dir/$bashrc"
cp --no-clobber $reqs "$env_dir/$reqs" 
cp --no-clobber $create "$env_dir/$create"

cd $env_dir

sh $create

cd $this_dir

sbatch run-training.job
