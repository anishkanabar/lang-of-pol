# Recipes for training on police language dataset

## Installation
### Midway3
The midway3 compute nodes do not have libsndfile. As a workaround, we download this in a conda env and link to it at runtime. 
1. `conda create -c conda-forge -y --name soundfile librosa`
2. Dont activate this conda env. Just stay on your base env.
3. `python3 -m venv path/to/new/env` 
4. `source path/to/new/env/bin/activate`
5. `cd asr/data/asr_dataset`
6.  `python setup.py build`
7.  `pip install -e .`
8.  `cd asr/models/speechbrain`
9.  `pip install -r requirements.txt`
10.  `pip install -e .`

## Training
1. Edit hparams/params.yaml to change the dataset, cluster, number of transcripts.
    (Or pass as command line options in step 3)
2. Choose a model component and parameters file. e.g. seq2seq/train.py, seq2seq/hparams/train.yaml
3. Run via `bash run-training.sh <rcc|ai> <path to train.py> <path to hparams.yaml> [--cluster C] [--dataset_name D] [--num_train N]`

## Development
### param.yaml files
See Tokenizer/hparams/tokenizer\_bpe.yaml for an example of how to set the parameters for the prepare function. 

### prepare.py files
Typically, models can re-use the basic prepare.py file without any changes. In this case, the convention is to create a soft link to the original file instead of copying it. To do this:
1. cd into the component folder
2. create soft link. the syntax is `ln -s ORIGINALPATH LINKPATH` e.g. ln -s ../../bpc\_prepare.py bpc\_prepare.py`
