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

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    train_data, val_data, test_data = prepare.dataio_prepare(hparams, text_pipeline)
    return train_data, val_data, test_data, tokenizer
    

def prepare_bpc(splits: dict, 
                output_folder: str, 
                **kwargs):
    """ See docstring in bpc_prepare for other params"""

    prepare.prepare_bpc(splits=splits, 
                        output_folder=output_folder, 
                        **kwargs)

    splitdata = get_splits(splits, output_folder)
    splitdata = {k: ctc_prep(v) for k, v in splitdata.items()}
    # splitdata = {k: filter_duration(v) for k, v in splitdata.items()}
    splitdata = {k: filter_nonalphanum(v) for k,v in splitdata.items()}
    splitdata = {k: filter_ratio(v) for k, v in splitdata.items()}

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
    

def ctc_prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'transcript':'wrd'})


def filter_duration(df: pd.DataFrame) -> pd.DataFrame:
    min_duration = 1.5  # minimum librispeech duration
    predicate = df['duration'] >= min_duration
    logger.info("Filtering out {} audio < {} sec".format(len(df) - predicate.sum(), min_duration))
    df = df.loc[predicate]
    return df


def filter_nonalphanum(df: pd.DataFrame) -> pd.DataFrame:
    # regex gotchas: must escape [], -, / even if inside brackets
    special = re.compile("[^A-Za-z0-9 ']")
    # special = re.compile("[()\[\]\-\/`;:.,?!<>\*\{\}…\"]")
    non_special = df['wrd'].str.upper().str.replace(special, '', regex=True)
    logger.info(f"Filtered out {df['wrd'].str.len().sum()-non_special.str.len().sum()} special characters")
    return df.assign(wrd = non_special)


def filter_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Filters out examples where expected MFCC length is close to text length
    Params:
        df - expects columns {duration, wrd} 
    """
    FRAME_RATE = 49  # (Hz)
    #MIN_RATIO = 2.84375 # (2.0 is bad --- 2.75 was bad -- 2.796875 bad -- 2.84375 ok -- 2.9375 ok -- 3.125 ok -- 3.5 is ok -- 5.0 is ok)
    MIN_RATIO = 1.0
    logger.info(f'Testing with MFCC ratio {MIN_RATIO}')
    mfcc_lengths = df['duration'] * FRAME_RATE
    # logger.debug(f'Min/Avg/Max Num Frames: {mfcc_lengths.min():.2f} : {mfcc_lengths.mean():.2f} : {mfcc_lengths.max():.2f}')
    num_chars = df['wrd'].str.len()
    mfcc_ratios = mfcc_lengths / num_chars
    pred = mfcc_ratios > MIN_RATIO
    logger.info(f"Discarding {len(pred) - pred.sum()} bad MFCC ratios of {len(pred)} examples.")
    return df.loc[pred]

