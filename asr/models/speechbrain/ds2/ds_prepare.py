import os
import logging
from typing import TypedDict
import bpc_prepare as prepare
import pandas as pd

DataSplits = TypedDict('DataSplit', {'split': str, 'data': pd.DataFrame})

logger = logging.getLogger('asr.prepare.ds')

def prepare_bpc(split_ratios: dict, 
                output_folder: str, 
                **kwargs):
    """ See docstring in bpc_prepare for other params"""

    prepare.prepare_bpc(split_ratios=split_ratios, 
                        output_folder=output_folder, 
                        **kwargs)

    splits = get_splits(split_ratios, output_folder)
    splits = {k: ds_prep(v) for k, v in splits.items()}
    write_splits(splits, output_folder)


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
