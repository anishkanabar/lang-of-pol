# Test and Presentation!
### This dictionary is for debug and presentation use. In this document, we will present the detail of our coding logic, splitting each of our achievements to units. In this case, this document is a good stating point for everyone to approach our works💡

## Dependencies:
Requires an active conda installation.

## How to Run Deepspeech 2:
In a terminal, activate your conda environment, install the requirements file.
### On a HPC cluster:
`sbatch run-training.job`
### Locally
`sh run-training.job`

## Files:
### Cluster:
- run-training.job: configures the slurm job to train the model on a gpu
### Model:
- train_deepspeech.py: loads the training dataset and trains the model
- deepasr: implementation of neural net models
### Environment Management:
- env/requirements_deepspeech2_gpu.txt: lists the python packages needed to run this model
