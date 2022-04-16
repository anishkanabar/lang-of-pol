# Recipes for training on police language dataset

## Installation
1. Create a new virtual environment. Do not use conda - it doesnt resolve the dependencies.
`python3 -m venv path/to/venv`
2. Install speechbrain and librosa
`source path/to/venv/bin/activate`
`pip install speechbrain`
3. Install huggingface transformers
`pip install transformers`
4. Install asr data loaders
`cd <PROJECT_ROOT>/asr/data/asr_dataset`
`pip install -e .`

### Special Instructions for Midway
Midway compute nodes don't have libsndfile or ffmpeg. Pip won't install these, so we need to install them to a conda environment. Our slurm wrapper script expects the following environments.
`conda create -c conda-forge -y --name soundfile pysoundfile` and
`conda create -c conda-forge -y --name ffmpeg ffmpeg`

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
