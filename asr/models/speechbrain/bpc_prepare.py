import os
import torch
import logging
import librosa
import pandas as pd
from pathlib import Path
import speechbrain as sb
from asr_dataset.police import BpcETL, AmbiguityStrategy
from asr_dataset.librispeech import LibriSpeechETL
from asr_dataset.atczero import ATCZeroETL
from asr_dataset.constants import Cluster

logger = logging.getLogger(__name__)

def prepare_bpc(cluster: str, 
                dataset_name:str, 
                splits: dict,
                output_folder: str, 
                ambiguity_strategy: str='ALL',
                skip_prep=False):
    """
    @:param
        cluster: cluster name. either ['rcc', 'ai', 'ttic']
        dataset_name: either ['police', 'librispeech', 'atczero']
        splits: dictionary of named train/val/test splits and duration
        output_folder: path to manifest (output) folder
        skip_prep: If True, skip data preparation.
    """
    if skip_prep:
        return

    cluster = Cluster[cluster.upper()]
    if dataset_name == 'police':
        etl = BpcETL(cluster, filter_numeric=False, ambiguity=ambiguity_strategy)
    elif dataset_name == 'librispeech':
        etl = LibriSpeechETL(cluster)
    elif dataset_name == 'atczero':
        etl = ATCZeroETL(cluster)
    else:
        raise NotImplementedError('dataset ' + dataset_name)

    data = etl.etl(splits=splits)
    logger.debug(f'Loaded data has {data["split"].nunique()} splits')

    # Write separate df csv per split
    for split in data['split'].unique():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        logger.info("Preparing %s ..." % manifest_path)
        
        splitdata = data[data['split'] == split]
        new_df = pd.DataFrame(
            {
                "ID": splitdata.index.to_series(),
                "duration": splitdata['duration'],
                "wav": splitdata['audio'],
                "transcript": splitdata['text']
            }
        )
        new_df.to_csv(manifest_path, index=False)
        
def dataio_prepare(hparams, text_pipeline):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"]
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"]
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig = librosa.resample(sig.numpy(), 
                               orig_sr=hparams['data_sample_rate'], 
                               target_sr=hparams['model_sample_rate'])
        sig = torch.tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
 
    # 3. Define text pipeline:
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, 
        ["id", "sig"] + [x for x in text_pipeline.provides]
    )
    return train_data, valid_data, test_datasets

 
