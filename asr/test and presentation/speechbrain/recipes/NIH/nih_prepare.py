import os
import logging
from asr_dataset.datasets.radio import RadioDataset
from asr_dataset.datasets.librispeech import LibriSpeechDataset
import pandas as pd

logger = logging.getLogger(__name__)

def prepare_nih(data_folder, save_folder, skip_prep=False):
    """
    @:param
        data_folder : path to dataset.
        save_folder: path to manifest (output) folder
        skip_prep: If True, skip data preparation.
    """
    if skip_prep:
        return

    # TODO: pass in constructor arguments somehow
    data = RadioDataset('rcc', nrow=1024).data
    
    # Split into train/val/test
    test_data = data.sample(frac=.2, random_state=1234)
    trainval_data = data.iloc[data.index.difference(test_data.index)]
    val_data = trainval_data.sample(frac=.25, random_state=1234)
    train_data = trainval_data.iloc[trainval_data.index.difference(val_data.index)]
    splits = {'train': train_data, 'val': val_data, 'test': test_data}
    
    for split, splitdata in splits.items():
        manifest_path = os.path.join(save_folder, split) + '.csv'
        if os.path.exists(manifest_path):
            continue
        logger.info("Preparing %s..." % manifest_path)
        
        new_df = pd.DataFrame(
            {
                "ID": splitdata.index.to_series(),
                "duration": splitdata['nsamples'],
                "wav": splitdata['path'],
                "transcript": splitdata['transcript']
            }
        )
        new_df.to_csv(manifest_path, index=False)
        
        
        
