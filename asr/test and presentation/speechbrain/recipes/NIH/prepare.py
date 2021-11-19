import os
import shutil
import logging
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataset_radio_nih import RadioDataset
import glob

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)


def prepare_nih(data_folder, save_folder, skip_prep=False):
    """
    This function prepares the NIH dataset.
    If the folder does not exist, the zip file will be extracted.
    @:param
        data_folder : path to NIH dataset.
        save_folder: path where to store the manifest csv files.
        skip_prep: If True, skip data preparation.
    """

    if skip_prep:
        return

    # This section used to download and extract a tar file of the dataset.
    # Since the NIH dataset isn't a public benchmark dataset, 
    # and since it already exists on our servers, we can skip to the next step.
        
    split_nih(save_folder, data_folder)


def split_nih(save_folder, data_folder):
    """"
    This function generate train, dev, test data set.
    Following the structure of other recipes
    @:param
        save_folder : path where to store the manifest csv files.
        data_folder : path to NIH dataset.
    """
    # TODO: Refactor path column dependency out of write_clips ad move this block to prepare_nih()
    data_loader = RadioDataset()
    metadata = data_loader.load_transcripts(data_folder)
    metadata = metadata.sample(n=1024, random_state=1234) # XXX: For Testing! 
    data_loader.describe(metadata, 'XXX Small Dev Set')
    data_loader.write_clips(metadata)
    
    splits = ["train", "val", "test"]
    test_metadata = metadata.sample(frac=.2, random_state=1234)
    trainval_metadata = metadata.iloc[metadata.index.difference(test_metadata.index)]
    val_metadata = trainval_metadata.sample(frac=.25, random_state=1234)
    train_metadata = trainval_metadata.iloc[trainval_metadata.index.difference(val_metadata.index)]
    split_metadata = {'train': train_metadata, 'val': val_metadata, 'test': test_metadata}
    
    ID_start = 0  # needed to have a unique ID for each audio
    for split in splits:
        manifest_path = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(manifest_path):
            continue
        logger.info("Preparing %s..." % manifest_path)

        data = split_metadata[split]

        # spk_id = []
        ID = list(range(ID_start, len(data)))
        ID_start += len(data)
        wav = data['path']
        transcript = data['transcript']
        sec2frame = lambda x,y: data_loader.audio_slicer(x,y,22050)
        sec2dur = lambda x,y: sec2frame(x,y).end - sec2frame(x,y).start
        duration_func = lambda x: sec2dur(x.offset, x.duration)
        duration = data.transform(duration_func)

        new_df = pd.DataFrame(
            {
                "ID": ID,
                "duration": duration,
                "wav": wav,
                "transcript": transcript,
            }
        )
        new_df.to_csv(manifest_path, index=False)
