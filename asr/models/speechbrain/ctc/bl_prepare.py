import os
import logging
import bpc_prepare as prepare
import pandas as pd

logger = logging.getLogger('asr.prepare.ctc.bl')

def prepare_bpc(split_ratios: dict, 
                save_folder: str, 
                blacklist_file: str, 
                seed: str,
                **kwargs):
    """ See docstring in bpc_prepare for other params"""

    prepare.prepare_bpc(split_ratios=split_ratios, 
                        save_folder=save_folder, 
                        **kwargs)

    splits = get_splits(split_ratios, save_folder)
    splits = {k: ctc_prep(v) for k, v in splits.items()}
    splits = {k: filter_duration(v) for k, v in splits.items()}

    blacklist = get_blacklist(blacklist_file)
    
    splits['train'] = fail_pass(splits['train'], blacklist)

    write_splits(splits, save_folder)


def get_blacklist(blacklist_file: str):
    if not os.path.exists(blacklist_file):
        with open(blacklist_file, "w") as f:
            f.write("seed,pass,batch,ID,is_finite,loss,wrd\n")

    return pd.read_csv('ctc/hparams/blacklist_2.csv')


def get_splits(split_ratios, save_folder) -> {str, pd.DataFrame}:
    splits = {}
    for split in split_ratios.keys():
        manifest_path = os.path.join(save_folder, split) + '.csv'
        splits[split] = pd.read_csv(manifest_path)
    return splits


def write_splits(splits: {str, pd.DataFrame}, save_folder: str):
    for split in splits.keys():
        manifest_path = os.path.join(save_folder, split) + '.csv'
        splits[split].to_csv(manifest_path, index=False)
    

def ctc_prep(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'transcript':'wrd'})


def filter_duration(df: pd.DataFrame) -> pd.DataFrame:
    min_duration = 1.5  # minimum librispeech duration
    predicate = df['duration'] >= min_duration
    logger.info("Filtering out {} audio < {} sec".format(len(df) - predicate.sum(), min_duration))
    df = df.loc[predicate]
    return df


def fail_pass(df: pd.DataFrame, blacklist: pd.DataFrame) -> pd.DataFrame:
    """BL pass: Only keep stuff that FAILED."""
    infinite_ids = blacklist.loc[~(blacklist['is_finite'].astype(bool)), 'ID']
    predicate = df['ID'].isin(infinite_ids)
    logger.info("Blacklist pass: keeping {} audio".format(predicate.sum()))
    df = df.loc[predicate]
    return df

