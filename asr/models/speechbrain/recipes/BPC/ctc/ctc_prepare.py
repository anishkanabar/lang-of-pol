import os
import logging
import bpc_prepare as prepare
import pandas as pd

logger = logging.getLogger('asr.prepare.ctc')

def prepare_bpc(split_ratios: dict, save_folder: str, blacklist_file: str, **kwargs):
    """ See docstring in bpc_prepare for actual params"""
    if not os.path.exists(blacklist_file):
        with open(blacklist_file, "w") as f:
            f.write("id,words\n")

    prepare.prepare_bpc(split_ratios=split_ratios, 
                        save_folder=save_folder, 
                        **kwargs)
    for split in split_ratios.keys():
        manifest_path = os.path.join(save_folder, split) + '.csv'
        df = pd.read_csv(manifest_path).rename(columns={'transcript':'wrd'})
        min_duration = 1.5  # minimum librispeech duration
        olen = len(df)
        df = df.loc[df['duration'] >= min_duration]
        logger.info("Filtering out {} audio < {} sec".format(olen - len(df), min_duration))
        olen = len(df)
        blacklist = pd.read_csv(blacklist_file)
        df = df.loc[~df['ID'].isin(blacklist['id'])]
        logger.info("Filtering out {} blacklisted audio".format(olen - len(df), min_duration))
        df.to_csv(manifest_path, index=False)
