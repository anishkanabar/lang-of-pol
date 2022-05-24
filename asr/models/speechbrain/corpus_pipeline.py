"""
Pipeline from model-agnostic ASR dataloaders into speechbrain format
"""
import re
import os
import torch
import logging
import librosa
import pandas as pd
from pathlib import Path
import speechbrain as sb
from asr_dataset.police import BpcETL
from asr_dataset.base import AsrETL
from asr_dataset.librispeech import LibriSpeechETL
from asr_dataset.atczero import ATCZeroETL
from asr_dataset.constants import Cluster

logger = logging.getLogger(__name__)

def create_manifests(cluster: str, 
                dataset_name:str, 
                splits: dict,
                output_folder: str, 
                text_col: str,
                ambiguity_strategy: str='ALL',
                seed: int=1234,
                skip_prep=False,
                stratify: str=None):
    """
    Typical ETL steps into SB model-specific format.
    @:param
        cluster: cluster name. either ['rcc', 'ai', 'ttic']
        dataset_name: either ['police', 'librispeech', 'atczero']
        splits: dictionary of named train/val/test splits and duration
        output_folder: path to manifest (output) folder
        text_col: column name for transcript string
        seed: Random seed for sampling data subsets
        skip_prep: If True, skip data preparation.
    """
    if skip_prep:
        return None

    create_basic_manifests(cluster, 
        dataset_name, 
        splits, 
        output_folder, 
        ambiguity_strategy, 
        seed,
        skip_prep,
        stratify)
    
    splitdata = get_splits(splits, output_folder)
    splitdata = {k: rename_text_col(v, text_col) for k, v in splitdata.items()}
    splitdata = {k: uppercase(v, text_col) for k,v in splitdata.items()}
    splitdata = {k: filter_nonalphanum(v, text_col) for k,v in splitdata.items()}
    splitdata = {k: filter_nonblank(v, text_col) for k,v in splitdata.items()}
    splitdata = {k: filter_ratio(v, text_col) for k, v in splitdata.items()}

    for k, v in splitdata.items():
        AsrETL._describe(v, k)

    write_splits(splitdata, output_folder) 
    return splitdata

def create_basic_manifests(cluster: str, 
                dataset_name:str, 
                splits: dict,
                output_folder: str, 
                ambiguity_strategy: str='ALL',
                seed: int=1234,
                skip_prep=False,
                stratify: str=None,
                **kwargs):
    """
    Preliminary ETL steps into SB model-agnostic format.
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
        etl = BpcETL(
                cluster, 
                filter_numeric=False, 
                ambiguity=ambiguity_strategy,
                **kwargs)
    elif dataset_name == 'librispeech':
        etl = LibriSpeechETL(cluster)
    elif dataset_name == 'atczero':
        etl = ATCZeroETL(cluster)
    else:
        raise NotImplementedError('dataset ' + dataset_name)

    data = etl.etl(splits=splits, seed=seed, stratify=stratify)
    logger.debug(f'Loaded data has {data["split"].nunique()} splits')

    # Write separate df csv per split
    for split in data['split'].unique():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        logger.info("Preparing %s ..." % os.path.basename(manifest_path))
        
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
        
def dataio_prepare(hparams, text_pipeline, add_audio_pipeline=True):
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

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"])

    datasets = [train_data, valid_data, test_data]


    # 2. Define audio pipeline:
    if add_audio_pipeline:
        audio_pipeline = gen_audio_pipeline(hparams)
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
 
    # 3. Define text pipeline:
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    output_keys = ["id"]
    output_keys += [x for x in text_pipeline.provides]
    if add_audio_pipeline:
        output_keys += [x for x in audio_pipeline.provides]
    sb.dataio.dataset.set_output_keys(datasets, output_keys)
    return train_data, valid_data, test_data

def gen_audio_pipeline(hparams):
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig = librosa.resample(sig.numpy(), 
                               orig_sr=hparams['data_sample_rate'], 
                               target_sr=hparams['model_sample_rate'])
        sig = torch.tensor(sig)
        return sig
    return audio_pipeline

def get_splits(splits, output_folder) -> {str, pd.DataFrame}:
    if splits is None:
        splits = {'all': 1}  # hack for LM 
    splitdata = {}
    for split in splits.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        splitdata[split] = pd.read_csv(manifest_path)
    return splitdata


def write_splits(splitdata: {str, pd.DataFrame}, output_folder: str):
    for split in splitdata.keys():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        splitdata[split].to_csv(manifest_path, index=False) 


def uppercase(df: pd.DataFrame, text_col:str) -> pd.DataFrame:
    return df.assign(**{text_col: df[text_col].str.upper()})


def filter_nonalphanum(df: pd.DataFrame, text_col:str) -> pd.DataFrame:
    # regex gotchas: must escape [], -, / even if inside brackets
    special = re.compile("[^A-Za-z0-9 ']")
    # special = re.compile("[()\[\]\-\/`;:.,?!<>\*\{\}…\"]")
    non_special = df[text_col].str.replace(special, '', regex=True)
    logger.info(f"Filtered out {df[text_col].str.len().sum()-non_special.str.len().sum()} special characters")
    return df.assign(**{text_col: non_special})


def filter_nonblank(df: pd.DataFrame, text_col:str) -> pd.DataFrame:
    nonblank = df[text_col].str.contains("[A-Za-z0-9]", regex=True)
    logger.info(f"Discarding {len(nonblank) - nonblank.sum()} blank transcripts")
    return df.loc[nonblank]


def filter_ratio(df: pd.DataFrame, text_col:str) -> pd.DataFrame:
    """
    Filters out examples where expected MFCC length is close to text length
    Params:
        df - expects columns {duration, <text_col>}
    """
    FRAME_RATE = 49  # (Hz)
    MIN_RATIO = 1.0
    mfcc_lengths = df['duration'] * FRAME_RATE
    num_chars = df[text_col].str.len()
    mfcc_ratios = mfcc_lengths / num_chars
    pred = mfcc_ratios > MIN_RATIO
    logger.info(f"Discarding {len(pred) - pred.sum()} bad MFCC ratios of {len(pred)} examples.")
    return df.loc[pred]


def rename_text_col(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    return df.rename(columns={'transcript':text_col})


