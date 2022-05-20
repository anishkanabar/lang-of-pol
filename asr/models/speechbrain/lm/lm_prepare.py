import re
import os
import logging
import torch
import speechbrain as sb
import bpc_prepare as prepare
from asr_dataset.base import AsrETL
from asr_dataset.atczero import ATCZeroETL
import pandas as pd

logger = logging.getLogger('asr.prepare.ctc')


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

    train_data, val_data, test_data = prepare.dataio_prepare(hparams, text_pipeline)
    return train_data, val_data, test_data, tokenizer
    

def prepare_bpc(splits: dict, 
                output_folder: str, 
                **kwargs):
    """ See docstring in bpc_prepare for other params"""
    # LM training can include more than the 'train' split 
    # as long as it doesn't intersect 'val+test'.
    # As a workaround for our custom dataloader design,
    # we first load ALL the data, then subtract the val+test
    # splits defined in the hyperparameters.
    
    # 1. Load ALL data
    prepare.prepare_bpc(splits=None, 
                        output_folder=output_folder, 
                        **kwargs)
    alldata = get_splits({'all':1}, output_folder)['all']
    alldata = lm_prep(alldata)
    alldata = filter_nonalphanum(alldata)
    alldata = filter_ratio(alldata)
    os.remove(f"{output_folder}/all.csv")

    # 2. Load sampled splitted data
    prepare.prepare_bpc(splits=splits, 
                        output_folder=output_folder, 
                        **kwargs)
    splitdata = get_splits(splits, output_folder)
    splitdata = {k: lm_prep(v) for k, v in splitdata.items()}
    splitdata = {k: filter_nonalphanum(v) for k,v in splitdata.items()}
    splitdata = {k: filter_ratio(v) for k, v in splitdata.items()}

    # 3. Include all non-test data in training
    splitdata['train'] = alldata[(~alldata['ID'].isin(splitdata['dev']['ID'])) & 
                                 (~alldata['ID'].isin(splitdata['test']['ID']))]

    for k, v in splitdata.items():
        AsrETL._describe(v, k)

    write_splits(splitdata, output_folder)


def get_splits(splits, output_folder) -> {str, pd.DataFrame}:
    splitdata = {}
    for split in splits.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        splitdata[split] = pd.read_csv(manifest_path)
    return splitdata


def write_splits(splitdata: {str, pd.DataFrame}, output_folder: str):
    for split in splitdata.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        splitdata[split].to_csv(manifest_path, index=False)
    

def lm_prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'transcript':'text'})


def filter_nonalphanum(df: pd.DataFrame) -> pd.DataFrame:
    # regex gotchas: must escape [], -, / even if inside brackets
    special = re.compile("[^A-Za-z0-9 ']")
    # special = re.compile("[()\[\]\-\/`;:.,?!<>\*\{\}…\"]")
    non_special = df['text'].str.upper().str.replace(special, '', regex=True)
    logger.info(f"Filtered out {df['text'].str.len().sum()-non_special.str.len().sum()} special characters")
    return df.assign(text = non_special)


def filter_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Filters out examples where expected MFCC length is close to text length
    Params:
        df - expects columns {duration, text} 
    """
    FRAME_RATE = 49  # (Hz)
    MIN_RATIO = 1.0
    mfcc_lengths = df['duration'] * FRAME_RATE
    num_chars = df['text'].str.len()
    mfcc_ratios = mfcc_lengths / num_chars
    pred = mfcc_ratios > MIN_RATIO
    logger.info(f"Discarding {len(pred) - pred.sum()} bad MFCC ratios of {len(pred)} examples.")
    return df.loc[pred]

