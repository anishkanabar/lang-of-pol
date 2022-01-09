# Test and Presentation!
### This dictionary is for debug and presentation use. In this document, we will present the detail of our coding logic, splitting each of our achievements to units. In this case, this document is a good stating point for everyone to approach our works💡

## Installation:
### Building Deepspeech
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

### Building SpeechBrain
1. Activate virtual environmenrt or conda env
2. `cd asr_dataset`
3. `python setup.py build`
4. `pip install -e .`
5. `cd speechbrain`
6. `pip install -r requirements.txt`
7. `pip install -e .`

## How to Run Deepspeech 2:
### On Midway Cluster:
`sbatch run-training-rcc.job`
### On AI Cluster:
`sh run-training-ai.sh`

## Files:
### Dependencies:
- asr_dataset: library to load datasets into asr-usable format
- requirements.txt: conda requirements
### Cluster:
- run-training.job: configures the slurm job to train the model on a gpu
- tf_devices.py: sanity checks to print the available GPU devices
### Model:
- train_deepspeech.py: loads the training dataset and trains the model
- deepasr: implementation of neural net models
- speechbrain: implementation of neural net models
- espnet: implementation of neural net models

