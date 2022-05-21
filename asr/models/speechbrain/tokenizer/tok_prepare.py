import re
import os
import logging
import bpc_prepare as prepare
from asr_dataset.base import AsrETL
from asr_dataset.atczero import ATCZeroETL
import pandas as pd

logger = logging.getLogger('asr.prepare.ctc')


def dataio_prepare(hparams):
    """ Dataset transformation pipeline """
    return prepare.dataio_prepare(hparams)


def prepare_bpc(splits: dict, 
                output_folder: str, 
                **kwargs):
    """ See docstring in bpc_prepare for other params"""

    prepare.prepare_bpc(splits=splits,
                        output_folder=output_folder, 
                        **kwargs)

    splitdata = get_splits(splits, output_folder)
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
    

def filter_nonalphanum(df: pd.DataFrame) -> pd.DataFrame:
    # regex gotchas: must escape [], -, / even if inside brackets
    special = re.compile("[^A-Za-z0-9 ']")
    # special = re.compile("[()\[\]\-\/`;:.,?!<>\*\{\}…\"]")
    non_special = df['transcript'].str.upper().str.replace(special, '', regex=True)
    logger.info(f"Filtered out {df['transcript'].str.len().sum()-non_special.str.len().sum()} special characters")
    return df.assign(transcript = non_special)


def filter_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Filters out examples where expected MFCC length is close to text length
    Params:
        df - expects columns {duration, transcript} 
    """
    FRAME_RATE = 49  # (Hz)
    MIN_RATIO = 1.0
    mfcc_lengths = df['duration'] * FRAME_RATE
    num_chars = df['transcript'].str.len()
    mfcc_ratios = mfcc_lengths / num_chars
    pred = mfcc_ratios > MIN_RATIO
    logger.info(f"Discarding {len(pred) - pred.sum()} bad MFCC ratios of {len(pred)} examples.")
    return df.loc[pred]

