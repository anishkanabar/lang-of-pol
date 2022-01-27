# Test and Presentation!
### This dictionary is for debug and presentation use. In this document, we will present the detail of our coding logic, splitting each of our achievements to units. In this case, this document is a good stating point for everyone to approach our works💡

## Installation:
1. Activate conda env

#### Building on Midway
1. `conda install -c conda-forge --file requirements_deepspeech2_gpu.txt`

#### Building on another system
1. `conda install -y "tensorflow-gpu==2.4.1" keras-gpu pandas "numpy==1.19.2"`
2. `conda install -y -c conda-forge librosa`

(If installing the CPU version, omit the [-gpu] suffix.)

#### Building Datset Library
1. `cd ./asr_dataset/`
2. `python setup.py build`
3. `pip install -e .`

## Training:
### On Midway Cluster:
`sbatch run-training-rcc.job`
### On AI Cluster:
`sh run-training-ai.sh`

## Predicting:
1. Locate a saved model file. e.g. in /project/graziul/ra/echandler/job_XXXXX/final_checkpoint
2. Identify number of samples used to train model. e.g. grep "Row count" /path/to/job/general.log
3. Activate conda env for deepspeech2 as directed above
4. `python predict_deepspeech.py --checkpoint /path/to/checkpoint-folder --ntrain XX --npred YY`

## Files:
### Dependencies:
- asr_dataset: library to load datasets into asr-usable format (found in PROJECT/asr/data)
- requirements.txt: conda requirements
### Cluster:
- run-training.job: configures the slurm job to train the model on a gpu
- tf_devices.py: sanity checks to print the available GPU devices
### Model:
- train.py: loads the training dataset and trains the model
- predict.py: loads new examples from training datset and transcribes them with model

