#!/bin/bash
#
#SBATCH --mail-user=ajays@uchicago.edu
#SBATCH --account=pi-graziul
#SBATCH --mail-type=ALL
#SBATCH --output=/project/graziul/ra/ajays/sbatch_codes/slurm/%j.%N.stdout
#SBATCH --error=/project/graziul/ra/ajays/sbatch_codes/slurm/%j.%N.stderr
#SBATCH --chdir=/project/graziul/ra/ajays/sbatch_codes
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00

module load python
source activate new_torch_env
srun python job.py BPC Attention_LSTM 1
