import os
import logging
import torch
import speechbrain as sb
import corpus_pipeline as pipeline
from asr_dataset.base import AsrETL
import pandas as pd

logger = logging.getLogger('asr.prepare.lm')

def dataio_prepare(hparams):
    """ Dataset transformation pipeline """

    tokenizer = hparams["tokenizer"]

    # Define text processing pipeline. We start from the raw text and then
    # encode it using the tokenizer. The tokens with bos are used for feeding
    # the neural network, the tokens with eos for computing the cost function.
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides("text", "tokens_bos", "tokens_eos")
    def text_pipeline(text):
        yield text
        tokens_list = tokenizer.encode_as_ids(text)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    train_data, val_data, test_data = pipeline.dataio_prepare(hparams, text_pipeline, False)
    return train_data, val_data, test_data, tokenizer
    

def create_manifests(splits: dict, 
                output_folder: str, 
                **kwargs):
    """ See docstring in corpus_pipeline for other params"""
    # LM training can include more than the 'train' split 
    # as long as it doesn't intersect 'val+test'.
    # As a workaround for our custom dataloader design,
    # we first load ALL the data, then subtract the val+test
    # splits defined in the hyperparameters.
    
    # 1. Load ALL data
    logger.info('Loading all corpus data for lm')
    alldata = pipeline.create_manifests(splits=None, 
                        output_folder=output_folder, 
                        text_col="text",
                        **kwargs)
    alldata = alldata['all']
    os.remove(f"{output_folder}/all.csv")

    # 2. Load sampled splitted data
    logger.info('Loading sampled corpus data for asr')
    splitdata = pipeline.create_manifests(splits=splits, 
                        output_folder=output_folder, 
                        text_col="text",
                        **kwargs)

    # 3. Include all non-test data in training
    logger.info('Adding asr non-validation set to lm training set')
    splitdata['train'] = alldata[(~alldata['ID'].isin(splitdata['dev']['ID'])) & 
                                 (~alldata['ID'].isin(splitdata['test']['ID']))]
    for k, v in splitdata.items():
        AsrETL._describe(v, k)
    pipeline.write_splits(splitdata, output_folder)


