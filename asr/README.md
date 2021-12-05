# Test and Presentation!
### This dictionary is for debug and presentation use. In this document, we will present the detail of our coding logic, splitting each of our achievements to units. In this case, this document is a good stating point for everyone to approach our works💡

### How to Run:
Start with 'run-training.sh'. This script creates the necessary environments and dispatches the model training to slurm.

### Files:
### Cluster Mangement:
- run-training.sh: main entrypoint to the project. see 'How to Run' above.
- run-training.job: configures the slurm job to train the model on a gpu
### Model:
- train_deepspeech.py: loads the training dataset and trains the model
- deepasr: implementation of neural net models
### Environment Management:
- env/requirements.txt: lists the python packages needed to run this model
- env/create_env.sh: creates a conda environment in a shared location
- env/conda_bashrc: sources the shared environment and other necessary modules
