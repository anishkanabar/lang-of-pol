import os
import logging
from asr_dataset.police import BpcETL
from asr_dataset.librispeech import LibriSpeechETL
from asr_dataset.atczero import ATCZeroETL
from asr_dataset.constants import DataSizeUnit, Cluster
import pandas as pd

logger = logging.getLogger(__name__)

def prepare_bpc(cluster: str, 
                dataset_name:str, 
                num_train: int, 
                num_sec: float,
                split_ratios: dict,
                save_folder: str, 
                skip_prep=False):
    """
    @:param
        cluster: cluster name. either ['rcc', 'ai', 'ttic']
        dataset_name: either ['police', 'librispeech', 'atczero']
        num_train: number of samples to prepare from dataset
        num_sec: total seconds of audio to retrieve from dataset
        split_ratios: dictionary of named train/val/test splits and fraction of dataset
        save_folder: path to manifest (output) folder
        skip_prep: If True, skip data preparation.
    """
    if skip_prep:
        return

    cluster = Cluster[cluster.upper()]
    if dataset_name == 'police':
        etl = BpcETL(cluster)
    elif dataset_name == 'librispeech':
        etl = LibriSpeechETL(cluster)
    elif dataset_name == 'atczero':
        etl = ATCZeroETL(cluster)
    else:
        raise NotImplementedError('dataset ' + dataset_name)

    if num_sec is not None:
        qty = num_sec
        units = DataSizeUnit.SECONDS
    else:
        qty = num_train
        units = DataSizeUnit.ROW_COUNT
    data = etl.etl(qty=qty, units=units)

    # Make IDs global to splits
    data = data.reset_index()
    data = data.assign(ID = data.index.to_series())
    
    # Split into hparams-defined splits
    splits = {}
    other_splits = data
    other_frac = 1
    for split, frac in split_ratios.items():
        current_frac = max(0., min(1., frac / other_frac))
        current_split = other_splits.sample(frac=current_frac, random_state=1234)
        other_splits = other_splits.loc[other_splits.index.difference(current_split.index)]
        other_frac = max(0., min(1., other_frac - frac))
        splits[split] = current_split
        
    #test_data = data.sample(frac=.2, random_state=1234)
    #trainval_data = data.iloc[data.index.difference(test_data.index)]
    #val_data = trainval_data.sample(frac=.25, random_state=1234)
    #train_data = trainval_data.iloc[trainval_data.index.difference(val_data.index)]
    #splits = {'train': train_data, 'val': val_data, 'test': test_data}
    
    for split, splitdata in splits.items():
        manifest_path = os.path.join(save_folder, split) + '.csv'
        logger.info("Preparing %s..." % manifest_path)
        
        new_df = pd.DataFrame(
            {
                "ID": splitdata['ID'],
                "duration": splitdata['duration'],
                "wav": splitdata['audio'],
                "transcript": splitdata['text']
            }
        )
        new_df.to_csv(manifest_path, index=False)
        
        
        
