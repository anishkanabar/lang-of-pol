#!/bin/bash
#
#SBATCH --mail-user=ajays@uchicago.edu
#SBATCH --account=pi-graziul
#SBATCH --mail-type=ALL
#SBATCH --output=/project/graziul/ra/ajays/lang-of-pol/vad/slurm/%j.%N.stdout
#SBATCH --error=/project/graziul/ra/ajays/lang-of-pol/vad/slurm/%j.%N.stderr
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10000

module load python
source activate new_torch_env
srun python job.py BPC Vanilla_LSTM 1
