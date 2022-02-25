# Recipes for training on police language dataset

## Installation
### Midway3
The midway3 compute nodes do not have libsndfile or ffmpeg. As a workaround, we download this in a conda env and link to it at runtime. 
1. a. `conda create -c conda-forge -y --name soundfile librosa`
1. b. `conda create -c conda-forge -y --name ffmpeg ffmpeg`
2. Dont activate this conda env. Just stay on your base env.

### All clusters
4. Create a new virtual environment for speechbrain, activate it, and install dependencies. This sould be a venv, not conda. Conda can't find the right pytorch + cuda combination.
5. `python3 -m venv path/to/new/env` 
6. `source path/to/new/env/bin/activate`
7. `cd <PROJECT_ROOT>/asr/data/asr_dataset`
8.  `python setup.py build`
9.  `pip install -e .`
10.  `cd <PROJECT_ROOT>/asr/models/speechbrain`
11.  `pip install -r requirements.txt`
12.  `pip install -e .`

## Training
1. Edit hparams/params.yaml to change the dataset, cluster, number of transcripts.
2. Check the pretrained model id hard-coded into fetch_model.py . It should match wav2vec_hub in the params.yaml
3. Fetch the pre-trained model with `srun python fetch_model.py`
4. Choose a model component and parameters file. e.g. ctc/train_with_wav2vec.py, ctc/hparams/params_trial_15.yaml
5. Run via `bash run-training.sh <path to train.py> <path to hparams.yaml> [--cluster C] [--dataset_name D] [--num_train N]`

## Development
### param.yaml files
See Tokenizer/hparams/tokenizer\_bpe.yaml for an example of how to set the parameters for the prepare function. 

### prepare.py files
Typically, models can re-use the basic prepare.py file without any changes. In this case, the convention is to create a soft link to the original file instead of copying it. To do this:
1. cd into the component folder
2. create soft link. the syntax is `ln -s ORIGINALPATH LINKPATH` e.g. ln -s ../../bpc\_prepare.py bpc\_prepare.py`

If the model needs to customize the parent data prep, make a local prepare.py, and optionally import and run the parent bpc\_prepare.py and modify the csvs.
