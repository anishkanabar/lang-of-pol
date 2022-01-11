# Recipes for training on police language dataset

## Training
1. Edit run-training.sh, specifically the CMD variable to run a different recipe component.
2. Edit hparams/params.yaml to change the dataset, cluster, number of transcripts
3. Run via `sh run-training.sh <rcc|ai>`

## Development
### param.yaml files
See Tokenizer/hparams/tokenizer\_bpe5000.yaml for an example of how to set the parameters for the prepare function. 
### prepare.py files
By convention, recipes have one "parent" prepare.py file. If a recipe component does not need to modify the file, then the component links to the file instead of copying it. To do this:
1. cd into the component folder
2. create soft link. the syntax is `ln -s ORIGINALPATH LINKPATH` e.g. ln -s ../../nih\_prepare.py nih\_prepare.py`



 
