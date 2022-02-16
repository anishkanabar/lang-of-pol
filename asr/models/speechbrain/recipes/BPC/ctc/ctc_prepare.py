import os
import bpc_prepare as prepare
import pandas as pd

def prepare_bpc(split_ratios: dict, save_folder: str, **kwargs):
    """ See docstring in bpc_prepare for actual params"""
    prepare.prepare_bpc(split_ratios=split_ratios, 
                        save_folder=save_folder, 
                        **kwargs)
    for split in split_ratios.keys():
        manifest_path = os.path.join(save_folder, split) + '.csv'
        df = pd.read_csv(manifest_path).rename(columns={'transcript':'wrd'})
        df.to_csv(manifest_path, index=False)
