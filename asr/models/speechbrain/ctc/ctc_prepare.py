import os
import logging
import bpc_prepare as prepare
import pandas as pd

logger = logging.getLogger('asr.prepare.ctc')

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
    splits = {k: first_pass(v, blacklist) for k, v in splits.items()}
    splits = {k: second_pass(v, blacklist) for k, v in splits.items()}

    write_splits(splits, save_folder)


def get_blacklist(blacklist_file: str):
    if not os.path.exists(blacklist_file):
        with open(blacklist_file, "w") as f:
            # sequence column is computed in memory so we can just append writes
            f.write("seed,id,is_finite,wrd\n")

    df = pd.read_csv(blacklist_file)
    counter = 0
    ids = set([])
    sequence = []
    for row in df.itertuples():
        if not row.is_finite:
            sequence.append(counter)
            counter = 0
        elif row.id not in ids:
            counter = 0
            sequence.append(counter)
        else:
            counter = counter + 1
            sequence.append(counter)
        ids.add(row.id)
    return df.assign(sequence=sequence)


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


def first_pass(df: pd.DataFrame, blacklist: pd.DataFrame) -> pd.DataFrame:
    """First pass: Train on a single utterance to identify problems independent of sequencing."""
    singles = blacklist.loc[blacklist['sequence'] == 0, 'id']
    predicate = ~ df['ID'].isin(singles)
    if predicate.any():
        df = df.loc[predicate].head(1)
        logger.info("Subsetting first pass to {}:{}".format(df['id'], df['wrd']))
    return df
        
        
def second_pass(df: pd.DataFrame, blacklist: pd.DataFrame) -> pd.DataFrame:
    """Second pass: Train on all non-blacklisted utterances. Grow blacklist one at a time."""
    infinite_ids = blacklist.loc[~blacklist['is_finite'], 'id']
    predicate = df['ID'].isin(infinite_ids)
    logger.info("Filtering out {} blacklisted second pass audio".format(predicate.sum()))
    df = df.loc[~predicate]
    return df

