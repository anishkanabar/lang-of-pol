import os
import logging
import bpc_prepare as prepare
import pandas as pd

logger = logging.getLogger('asr.prepare.ctc')

def prepare_bpc(split_ratios: dict, 
                output_folder: str, 
                seed: str,
                **kwargs):
    """ See docstring in bpc_prepare for other params"""

    prepare.prepare_bpc(split_ratios=split_ratios, 
                        output_folder=output_folder, 
                        **kwargs)

    splits = get_splits(split_ratios, output_folder)
    splits = {k: ctc_prep(v) for k, v in splits.items()}
    splits = {k: filter_duration(v) for k, v in splits.items()}

    # blacklist = get_blacklist(blacklist_file)

    # Commenting out first pass because it was written assuming it
    # could test all data points independently. This is infeasible
    # due to high computational cost per data point (each incurs the
    # startup costs of re-loading the model from scratch).
    # Instead, we'll run the first pass enough times to be confident
    # we didn't miss any individually problematic data points.
    # ...
    # Ran first pass and got 474 negative results. Previous blacklist
    # had ~ 450 positive results out of 30k training points
    # = probability of positive result = .015
    # => prob of randomly sampling 474 negative results
    # = (1 - prob of positive data point)^m 
    # = (1-.015)^474 = .00077 => very low probability of Type II error
    # ...
    # splits = {k: first_pass(v, blacklist) for k, v in splits.items()}
    # splits = {k: second_pass(v, blacklist) for k, v in splits.items()}

    write_splits(splits, output_folder)


def get_blacklist(blacklist_file: str):
    if not os.path.exists(blacklist_file):
        with open(blacklist_file, "w") as f:
            f.write("seed,pass,batch,ID,is_finite,loss,wrd\n")

    return pd.read_csv(blacklist_file)


def get_splits(split_ratios, output_folder) -> {str, pd.DataFrame}:
    splits = {}
    for split in split_ratios.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        splits[split] = pd.read_csv(manifest_path)
    return splits


def write_splits(splits: {str, pd.DataFrame}, output_folder: str):
    for split in splits.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
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
    singles = blacklist.loc[(blacklist['pass'] == 'first') & (blacklist['batch'] == 0), 'ID']
    predicate = ~ df['ID'].isin(singles)
    if predicate.any():
        df = df.loc[predicate].head(1)
        logger.info("Blacklist first pass data point {}: {}".format(df['ID'].values[0], df['wrd'].values[0]))
    else:
        raise RuntimeError("First pass is done!")
    return df
        
        
def second_pass(df: pd.DataFrame, blacklist: pd.DataFrame) -> pd.DataFrame:
    """Second pass: Train on all non-blacklisted utterances. Grow blacklist one at a time."""
    infinite_ids = blacklist.loc[~(blacklist['is_finite'].astype(bool)), 'ID']
    predicate = df['ID'].isin(infinite_ids)
    logger.info("Blacklist second pass: filtering out {} audio".format(predicate.sum()))
    df = df.loc[~predicate]
    return df

