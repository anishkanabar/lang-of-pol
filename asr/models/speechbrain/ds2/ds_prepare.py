import os
import re
import logging
from typing import TypedDict
import bpc_prepare as prepare
import pandas as pd

DataSplits = TypedDict('DataSplit', {'split': str, 'data': pd.DataFrame})

logger = logging.getLogger('asr.prepare.ds')


def dataio_prepare(hparams):
    """ Dataset transformation pipeline """
    return prepare.dataio_prepare(hparams)


def prepare_bpc(split_ratios: dict, 
                output_folder: str, 
                **kwargs):
    """ See docstring in bpc_prepare for other params"""

    prepare.prepare_bpc(split_ratios=split_ratios, 
                        output_folder=output_folder, 
                        **kwargs)

    splits = get_splits(split_ratios, output_folder)
    splits = {k: ds_prep(v) for k, v in splits.items()}
    splits = {k: ratio_filter(v) for k, v in splits.items()}
    splits = {k: filter_nonalphanum(v) for k, v in splits.items()}
    write_splits(splits, output_folder)

def filter_nonalphanum(df: pd.DataFrame) -> pd.DataFrame:
    # regex gotchas: must escape [], -, / even if inside brackets
    special = re.compile("[()\[\]\-\/`;:.,?!\"]")
    non_special = df['wrd'].str.upper().str.replace(special, '', regex=True)
    logger.debug(f"Filtered out {df['wrd'].str.len().sum()-non_special.str.len().sum()} special characters")
    return df.assign(wrd = non_special)

def ratio_filter(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Filters out examples where expected MFCC length is close to text length
    Params:
        df - expects columns {duration, wrd} 
    """
    HOP_DURATION = 10  # (ms)
    MIN_RATIO = 2.0
    hop_sec = HOP_DURATION / 1000
    mfcc_lengths = df['duration'] / hop_sec
    # logger.debug(f'Min/Avg/Max Num Frames: {mfcc_lengths.min():.2f} : {mfcc_lengths.mean():.2f} : {mfcc_lengths.max():.2f}')
    num_chars = df['wrd'].str.len()
    mfcc_ratios = mfcc_lengths / num_chars
    pred = mfcc_ratios > MIN_RATIO
    logger.info(f"Discarding {len(pred) - pred.sum()} bad MFCC ratios of {len(pred)} examples.")
    # for i in range(len(pred)):
    #     if pred[i]:
    #         logger.debug(f"GOOD, ID, {df.loc[i,'ID']}, NFRAMES, {mfcc_lengths[i]:.2f}, NCHARS, {num_chars[i]}, TXT, {df.loc[i, 'wrd']}")
    #     else:
    #         logger.debug(f"BAD, ID, {df.loc[i,'ID']}, NFRAMES, {mfcc_lengths[i]:.2f}, NCHARS, {num_chars[i]}, TXT, {df.loc[i, 'wrd']}")
    return df.loc[pred]

def get_splits(split_ratios, output_folder) -> DataSplits:
    splits = {}
    for split in split_ratios.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        splits[split] = pd.read_csv(manifest_path)
    return splits


def write_splits(splits: DataSplits, output_folder: str):
    for split in splits.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        splits[split].to_csv(manifest_path, index=False)
    

def ds_prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'transcript':'wrd'})
